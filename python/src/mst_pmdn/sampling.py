"""Sampling utilities for MST-PMDN."""

from __future__ import annotations

import torch
from torch import Tensor

from .utils import sample_gamma


def sample_mst_pmdn(output: dict[str, Tensor], num_samples: int = 1, device: str | torch.device | None = None) -> dict[str, Tensor]:
    """Draw posterior samples from the skew-t mixture defined by ``output``."""

    dev = device or output["pi"].device
    dtype = output["pi"].dtype
    pi = output["pi"].to(device=dev)
    mu = output["mu"].to(device=dev)
    L_all = output["scale_chol"].to(device=dev)
    nu_all = output["nu"].to(device=dev)
    alpha_all = output["alpha"].to(device=dev)

    B, M, d = mu.shape
    cat = torch.distributions.Categorical(pi)
    idx = cat.sample((num_samples,)).transpose(0, 1)  # [B, S]
    batch_indices = torch.arange(B, device=dev).unsqueeze(1).expand(-1, num_samples)

    mu_s = mu[batch_indices, idx]
    L_s = L_all[batch_indices, idx]
    nu_s = nu_all[batch_indices, idx]
    alpha_s = alpha_all[batch_indices, idx]

    chi2 = sample_gamma(nu_s / 2, scale=2.0, device=dev, dtype=dtype)
    W = torch.sqrt(nu_s / chi2.clamp(min=1e-12)).unsqueeze(-1)

    alpha_norm_sq = alpha_s.pow(2).sum(dim=-1, keepdim=True)
    delta = alpha_s / torch.sqrt(1 + alpha_norm_sq + 1e-10)
    delta_norm_sq = delta.pow(2).sum(dim=-1, keepdim=True)

    z0 = torch.randn(B, num_samples, 1, device=dev, dtype=dtype)
    z1 = torch.randn(B, num_samples, d, device=dev, dtype=dtype)
    X = delta * torch.abs(z0) + torch.sqrt((1 - delta_norm_sq).clamp(min=1e-12)) * z1

    Y = mu_s + W * (torch.matmul(L_s, X.unsqueeze(-1)).squeeze(-1))
    samples = Y.permute(1, 0, 2).contiguous()
    components = idx.permute(1, 0).contiguous()

    return {"samples": samples, "components": components}
