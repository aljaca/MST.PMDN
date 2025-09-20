"""Distributional helpers used by the MST-PMDN loss."""

from __future__ import annotations

import torch
from torch import Tensor


def student_t_cdf(z: Tensor, nu: Tensor | float) -> Tensor:
    df = torch.as_tensor(nu, device=z.device, dtype=z.dtype)
    if torch.any(df <= 0):
        raise ValueError("Degrees of freedom must be positive")
    dist = torch.distributions.StudentT(df=df)
    return dist.cdf(z)


def log_student_t_cdf(z: Tensor, nu: Tensor | float, clamp: float = 1e-12) -> Tensor:
    cdf = student_t_cdf(z, nu)
    cdf = torch.clamp(cdf, min=clamp)
    return torch.log(cdf)
