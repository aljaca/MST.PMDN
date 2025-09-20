"""Model components for MST-PMDN."""

from __future__ import annotations

import torch
from torch import nn


class WeightNormLinear(nn.Module):
    """Weight normalised linear layer mirroring the R implementation."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.V = nn.Parameter(torch.randn(out_features, in_features) / in_features**0.5)
        self.g = nn.Parameter(torch.ones(out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        V_norm = self.V / (self.V.norm(dim=1, keepdim=True) + 1e-12)
        W = self.g.unsqueeze(1) * V_norm
        output = input @ W.T
        if self.bias is not None:
            output = output + self.bias
        return output


def init_weight_norm(module: nn.Module) -> None:
    """Module initialiser used across the head."""

    if isinstance(module, WeightNormLinear):
        nn.init.kaiming_normal_(module.V, mode="fan_out")
        with torch.no_grad():
            norm = module.V.norm(dim=1)
            module.g.copy_(norm)
