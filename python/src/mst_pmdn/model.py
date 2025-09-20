"""Model definition for the PyTorch MST-PMDN head."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .modules import WeightNormLinear
from .utils import build_orthogonal_matrix, parse_constraint


class MSTPMDN(nn.Module):
    """Multivariate skew-t Parsimonious Mixture Density Network head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | Sequence[int],
        n_mixtures: int,
        *,
        constraint: str = "VVVNN",
        constant_attr: str = "",
        activation: type[nn.Module] | Sequence[type[nn.Module]] = nn.ReLU,
        drop_hidden: float = 0.0,
        image_module: Optional[nn.Module] = None,
        tabular_module: Optional[nn.Module] = None,
        fixed_nu: Optional[Sequence[Optional[float]]] = None,
        range_nu: tuple[float, float] = (3.0, 50.0),
        max_alpha: float = 2.5,
        min_vol_shape: float = 1e-2,
        min_mix_weight: float = 1e-4,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.n_mixtures = n_mixtures
        self.constant_attr = constant_attr
        self.max_alpha = max_alpha
        self.min_vol_shape = min_vol_shape
        self.min_mix_weight = min_mix_weight
        self.jitter = jitter
        self.min_nu, self.max_nu = range_nu
        self.cfg = parse_constraint(constraint)
        self.image_module = image_module
        self.tabular_module = tabular_module

        self._build_feature_network(
            input_dim,
            hidden_dim,
            activation=activation,
            drop_hidden=drop_hidden,
        )
        self._build_parameter_heads(fixed_nu)

    # ------------------------------------------------------------------
    # Network construction helpers
    # ------------------------------------------------------------------
    def _infer_output_dim(self, module: nn.Module, fallback: Optional[int], name: str) -> int:
        for attr in ("output_dim", "out_features", "out_channels", "out_dim"):
            value = getattr(module, attr, None)
            if value is not None:
                return int(value)
        if isinstance(module, nn.Sequential) and len(module) > 0:
            last = module[-1]
            for attr in ("out_features", "out_channels"):
                value = getattr(last, attr, None)
                if value is not None:
                    return int(value)
        if fallback is None:
            raise ValueError(
                f"Unable to infer output dimension of {name}. "
                "Provide a module exposing an output dimension attribute."
            )
        return fallback

    def _build_feature_network(
        self,
        input_dim: int,
        hidden_dim: int | Sequence[int],
        *,
        activation: type[nn.Module] | Sequence[type[nn.Module]],
        drop_hidden: float,
    ) -> None:
        if self.tabular_module is None:
            tabular_dim = input_dim
        else:
            tabular_dim = self._infer_output_dim(self.tabular_module, input_dim, "tabular_module")
        total_input_dim = tabular_dim
        if self.image_module is not None:
            image_dim = self._infer_output_dim(self.image_module, None, "image_module")
            total_input_dim += image_dim

        if isinstance(hidden_dim, int):
            hidden_dims: Sequence[int] = (hidden_dim,)
        else:
            hidden_dims = tuple(hidden_dim)
        if len(hidden_dims) == 0 or hidden_dims == (0,):
            self.hidden = nn.Identity()
            self.final_hidden_dim = total_input_dim
            return

        if isinstance(activation, Sequence):
            activations = list(activation)
            if len(activations) != len(hidden_dims):
                raise ValueError("Number of activation functions must match hidden layers")
        else:
            activations = [activation] * len(hidden_dims)

        layers: list[nn.Module] = []
        current = total_input_dim
        for idx, (next_dim, act_cls) in enumerate(zip(hidden_dims, activations)):
            layers.append(nn.Linear(current, next_dim))
            if idx < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(next_dim))
                layers.append(act_cls())
                if drop_hidden > 0:
                    layers.append(nn.Dropout(drop_hidden))
            current = next_dim
        self.hidden = nn.Sequential(*layers)
        self.final_hidden_dim = current

    def _build_parameter_heads(self, fixed_nu: Optional[Sequence[Optional[float]]]) -> None:
        d = self.output_dim
        cfg = self.cfg
        const = self.constant_attr

        if "x" in const:
            self.pi = nn.Parameter(torch.ones(self.n_mixtures) / self.n_mixtures)
        else:
            self.fc_pi = WeightNormLinear(self.final_hidden_dim, self.n_mixtures)

        if "m" in const:
            self.mu = nn.Parameter(torch.randn(self.n_mixtures, d))
        else:
            self.fc_mu = WeightNormLinear(self.final_hidden_dim, self.n_mixtures * d)

        volume_size = 1 if cfg.volume_shared else self.n_mixtures
        if "L" in const:
            self.L_param = nn.Parameter(torch.zeros(volume_size))
        else:
            self.fc_L = WeightNormLinear(self.final_hidden_dim, volume_size)

        if not cfg.shape_identity:
            shape_size = d if cfg.shape_shared else self.n_mixtures * d
            if "A" in const:
                self.A_param = nn.Parameter(0.1 * torch.randn(shape_size))
            else:
                self.fc_A = WeightNormLinear(self.final_hidden_dim, shape_size)

        self.r = d * (d - 1) // 2
        if not cfg.orientation_identity:
            orient_size = self.r if cfg.orientation_shared else self.n_mixtures * self.r
            if "D" in const:
                self.D_param = nn.Parameter(0.1 * torch.randn(orient_size))
            else:
                self.fc_D = WeightNormLinear(self.final_hidden_dim, orient_size)

        if cfg.nu_mode == "F":
            if fixed_nu is None:
                raise ValueError("fixed_nu must be provided when using 'F' in the constraint")
            if len(fixed_nu) != self.n_mixtures:
                raise ValueError("fixed_nu must match the number of mixtures")
            values = torch.tensor(
                [float("nan") if v is None else float(v) for v in fixed_nu],
                dtype=torch.float,
            )
            mask = ~torch.isnan(values)
            self.register_buffer("fixed_nu_mask", mask)
            values = torch.nan_to_num(values, nan=0.0)
            self.register_buffer("fixed_nu_values", values)
            indices = (~mask).nonzero(as_tuple=False).view(-1)
            self.register_buffer("nu_opt_indices", indices.long())
            if indices.numel() > 0:
                if "n" in const:
                    self.nu_param_partial = nn.Parameter(torch.zeros(indices.numel()))
                else:
                    self.fc_nu_partial = WeightNormLinear(self.final_hidden_dim, indices.numel())
                    with torch.no_grad():
                        self.fc_nu_partial.V.normal_(0, 0.02)
                        self.fc_nu_partial.g.zero_()
                        self.fc_nu_partial.bias.zero_()
        elif cfg.nu_mode != "N":
            nu_size = 1 if cfg.nu_mode == "E" else self.n_mixtures
            if "n" in const:
                self.nu_param = nn.Parameter(torch.zeros(nu_size))
            else:
                self.fc_nu = WeightNormLinear(self.final_hidden_dim, nu_size)
                with torch.no_grad():
                    self.fc_nu.V.normal_(0, 0.02)
                    self.fc_nu.g.zero_()
                    self.fc_nu.bias.zero_()

        if cfg.skew_mode != "N":
            alpha_size = d if cfg.skew_mode == "E" else self.n_mixtures * d
            if "s" in const:
                self.alpha_param = nn.Parameter(0.05 * torch.randn(alpha_size))
            else:
                self.fc_alpha = WeightNormLinear(self.final_hidden_dim, alpha_size)
                with torch.no_grad():
                    self.fc_alpha.V.normal_(0, 0.02)
                    self.fc_alpha.g.zero_()
                    self.fc_alpha.bias.normal_(0, 0.02)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: Tensor, image_input: Optional[Tensor] = None) -> dict[str, Tensor]:
        if self.tabular_module is not None:
            tabular_features = self.tabular_module(x)
        else:
            tabular_features = x
        if self.image_module is not None and image_input is not None:
            image_features = self.image_module(image_input)
            features = torch.cat([tabular_features, image_features], dim=1)
        else:
            features = tabular_features

        h = self.hidden(features)
        B = x.size(0)
        d = self.output_dim

        if "x" in self.constant_attr:
            pi_logits = self.pi.unsqueeze(0).expand(B, -1)
        else:
            pi_logits = self.fc_pi(h)
        pi_raw = torch.softmax(pi_logits, dim=1)
        max_weight = 1.0 - (self.n_mixtures - 1) * self.min_mix_weight
        pi_clamped = pi_raw.clamp(min=self.min_mix_weight, max=max_weight)
        pi = pi_clamped / pi_clamped.sum(dim=1, keepdim=True)

        if "m" in self.constant_attr:
            mu = self.mu.unsqueeze(0).expand(B, -1, -1)
        else:
            raw_mu = self.fc_mu(h)
            mu = raw_mu.view(B, self.n_mixtures, d)

        if "L" in self.constant_attr:
            raw_L = self.L_param.clamp(-20, 20)
            L_val = F.softplus(raw_L) + 1e-6
            L_val = L_val.unsqueeze(0).expand(B, -1)
        else:
            raw_L = self.fc_L(h).clamp(-20, 20)
            L_val = F.softplus(raw_L) + 1e-6
        if self.cfg.volume_shared:
            L_val = L_val.expand(-1, self.n_mixtures)
        L_val = L_val.unsqueeze(-1).unsqueeze(-1)

        if self.cfg.shape_identity:
            A_diag = torch.ones(B, self.n_mixtures, d, device=x.device, dtype=mu.dtype)
        elif hasattr(self, "A_param"):
            rawA = self.A_param.clamp(-20, 20)
            rawA = F.softplus(rawA) + 1e-6
            if self.cfg.shape_shared:
                tmp = rawA.view(1, 1, -1)
                A_diag = tmp.expand(B, self.n_mixtures, -1)
            else:
                tmp = rawA.view(1, self.n_mixtures, -1)
                A_diag = tmp.expand(B, -1, -1)
        else:
            rawA = self.fc_A(h).clamp(-20, 20)
            rawA = F.softplus(rawA) + 1e-6
            if self.cfg.shape_shared:
                A_diag = rawA.unsqueeze(1).expand(-1, self.n_mixtures, -1)
            else:
                A_diag = rawA.view(B, self.n_mixtures, -1)
        prodA = A_diag.prod(dim=-1, keepdim=True)
        A_diag = A_diag / (prodA.pow(1 / d))
        A_diag = A_diag.clamp(min=self.min_vol_shape, max=1e3)

        if self.cfg.orientation_identity:
            eye = torch.eye(d, device=x.device, dtype=mu.dtype)
            D_tensor = eye.view(1, 1, d, d).expand(B, self.n_mixtures, -1, -1)
        elif hasattr(self, "fc_D"):
            rawD = self.fc_D(h)
            if self.cfg.orientation_shared:
                rawD = rawD.unsqueeze(1).expand(-1, self.n_mixtures, -1)
            else:
                rawD = rawD.view(B, self.n_mixtures, self.r)
            mats = [build_orthogonal_matrix(rawD[:, j, :], d) for j in range(self.n_mixtures)]
            D_tensor = torch.stack(mats, dim=1)
        else:
            if self.cfg.orientation_shared:
                raw = self.D_param.view(1, -1).expand(self.n_mixtures, -1)
            else:
                raw = self.D_param.view(self.n_mixtures, -1)
            mats = []
            for j in range(self.n_mixtures):
                base = build_orthogonal_matrix(raw[j].unsqueeze(0), d)
                mats.append(base.expand(B, -1, -1))
            D_tensor = torch.stack(mats, dim=1)

        if self.cfg.nu_mode == "N":
            nu = torch.full((B, self.n_mixtures), self.max_nu, device=x.device, dtype=mu.dtype)
        elif self.cfg.nu_mode == "F":
            fixed_values = self.fixed_nu_values.to(device=x.device, dtype=mu.dtype)
            mask = self.fixed_nu_mask.to(device=x.device)
            nu = torch.zeros(B, self.n_mixtures, device=x.device, dtype=mu.dtype)
            if mask.any():
                nu[:, mask] = fixed_values[mask]
            indices = self.nu_opt_indices
            if indices.numel() > 0:
                if hasattr(self, "nu_param_partial"):
                    raw = self.nu_param_partial
                    nu_opt = self.min_nu + (self.max_nu - self.min_nu) * torch.sigmoid(raw)
                    nu[:, indices] = nu_opt.unsqueeze(0)
                else:
                    raw = self.fc_nu_partial(h)
                    nu_opt = self.min_nu + (self.max_nu - self.min_nu) * torch.sigmoid(raw)
                    nu[:, indices] = nu_opt
            if not mask.any():
                nu = nu + 0  # ensure proper tensor
        elif hasattr(self, "nu_param"):
            raw = self.nu_param
            tmp = self.min_nu + (self.max_nu - self.min_nu) * torch.sigmoid(raw)
            if self.cfg.nu_mode == "E":
                nu = tmp.view(1, 1).expand(B, self.n_mixtures)
            else:
                nu = tmp.view(1, -1).expand(B, -1)
        elif hasattr(self, "fc_nu"):
            raw = self.fc_nu(h)
            tmp = self.min_nu + (self.max_nu - self.min_nu) * torch.sigmoid(raw)
            if self.cfg.nu_mode == "E":
                nu = tmp.expand(-1, self.n_mixtures)
            else:
                nu = tmp
        else:
            raise RuntimeError("nu parameters not configured")

        if self.cfg.skew_mode == "N":
            alpha = torch.zeros(B, self.n_mixtures, d, device=x.device, dtype=mu.dtype)
        elif hasattr(self, "alpha_param"):
            raw = self.alpha_param
            if self.cfg.skew_mode == "E":
                alpha = raw.view(1, 1, -1).expand(B, self.n_mixtures, -1)
            else:
                alpha = raw.view(1, self.n_mixtures, -1).expand(B, -1, -1)
        else:
            raw = self.fc_alpha(h)
            if self.cfg.skew_mode == "E":
                alpha = raw.unsqueeze(1).expand(-1, self.n_mixtures, -1)
            else:
                alpha = raw.view(B, self.n_mixtures, -1)
        alpha = self.max_alpha * torch.tanh(alpha)

        lambda_half = torch.sqrt(L_val)
        sqrtA = torch.sqrt(A_diag)
        sqrtA_mats = torch.diag_embed(sqrtA)
        L_direct = torch.matmul(D_tensor, sqrtA_mats)
        L_direct = lambda_half * L_direct
        Sigma = torch.matmul(L_direct, L_direct.transpose(-2, -1))
        eye = torch.eye(d, device=x.device, dtype=mu.dtype).view(1, 1, d, d)
        scale_chol = torch.linalg.cholesky(Sigma + self.jitter * eye)

        return {
            "pi": pi,
            "mu": mu,
            "scale_chol": scale_chol,
            "nu": nu,
            "alpha": alpha,
            "L": L_val,
            "A": A_diag,
            "D": D_tensor,
        }
