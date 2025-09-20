"""Loss functions for MST-PMDN."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor

from .distributions import log_student_t_cdf


def mst_pmdn_nll(
    output: Mapping[str, Tensor],
    target: Tensor,
    *,
    lambda_alpha: float = 0.0,
    lambda_nu_inv: float = 0.0,
) -> Tensor:
    """Negative log-likelihood of the skew-t mixture density network."""

    pi = output["pi"]
    mu = output["mu"]
    scale_chol = output["scale_chol"]
    nu = torch.clamp(output["nu"], min=1.0)
    alpha = output["alpha"]

    diff = target.unsqueeze(1) - mu
    diff_unsq = diff.unsqueeze(-1)
    v = torch.linalg.solve_triangular(scale_chol, diff_unsq, upper=False).squeeze(-1)
    maha = v.pow(2).sum(dim=-1).clamp(max=1e6)

    diag_L = scale_chol.diagonal(dim1=-2, dim2=-1)
    log_det = 2 * torch.log(torch.clamp(diag_L, min=1e-12)).sum(dim=-1)
    d = target.size(-1)
    const = torch.tensor(math.pi, device=pi.device, dtype=pi.dtype)
    half_nu = nu / 2
    half_nu_plus_d = (nu + d) / 2
    logC_t = (
        torch.lgamma(half_nu_plus_d)
        - torch.lgamma(half_nu)
        - (d / 2) * torch.log(nu * const)
        - 0.5 * log_det
    )
    logTail = -half_nu_plus_d * torch.log1p(torch.clamp(maha / nu, min=-1 + 1e-7, max=1e7))
    log_pdf_t = logC_t + logTail

    cterm = torch.sqrt((nu + d) / (nu + maha)).unsqueeze(-1).clamp(max=1e6)
    w = cterm * v
    alpha_dot_w = (alpha * w).sum(dim=-1)
    log_skew_factor = torch.log(torch.tensor(2.0, device=pi.device, dtype=pi.dtype))
    log_skew_factor = log_skew_factor + log_student_t_cdf(alpha_dot_w, nu + d)
    log_skewt = log_pdf_t + log_skew_factor

    weighted = torch.log(torch.clamp(pi, min=1e-12)) + log_skewt
    loss = -torch.logsumexp(weighted, dim=1).mean()
    loss = loss + lambda_alpha * alpha.pow(2).mean()
    loss = loss + lambda_nu_inv * nu.pow(-2).mean()
    return loss
