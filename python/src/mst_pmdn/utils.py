"""Utility helpers for the PyTorch MST-PMDN implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConstraintConfig:
    """Parsed constraint string controlling covariance structure."""

    volume_shared: bool
    shape_shared: bool
    shape_identity: bool
    orientation_shared: bool
    orientation_identity: bool
    nu_mode: str
    skew_mode: str


def parse_constraint(constraint: str) -> ConstraintConfig:
    """Parse the five letter constraint string used by the R implementation."""

    if len(constraint) != 5:
        raise ValueError("constraint must be a five character string")
    vol, shape, orient, nu, skew = constraint.upper()
    if vol not in {"E", "V"}:
        raise ValueError("volume constraint must be 'E' or 'V'")
    if shape not in {"E", "V", "I"}:
        raise ValueError("shape constraint must be 'E', 'V' or 'I'")
    if orient not in {"E", "V", "I"}:
        raise ValueError("orientation constraint must be 'E', 'V' or 'I'")
    if nu not in {"E", "V", "N", "F"}:
        raise ValueError("nu constraint must be 'E', 'V', 'N' or 'F'")
    if skew not in {"E", "V", "N"}:
        raise ValueError("skew constraint must be 'E', 'V' or 'N'")
    return ConstraintConfig(
        volume_shared=(vol == "E"),
        shape_shared=(shape == "E"),
        shape_identity=(shape == "I"),
        orientation_shared=(orient == "E"),
        orientation_identity=(orient == "I"),
        nu_mode=nu,
        skew_mode=skew,
    )


def sample_gamma(shape: Tensor | float, scale: Tensor | float = 1.0, device: str | torch.device | None = None,
                 dtype: torch.dtype | None = None) -> Tensor:
    """Gamma samples drawn on-device.

    Parameters mirror the R helper: ``shape`` is the concentration parameter, ``scale``
    the standard deviation scaling (``rate = 1 / scale`` in PyTorch).
    """

    shape_tensor = torch.as_tensor(shape, device=device, dtype=dtype or torch.get_default_dtype())
    scale_tensor = torch.as_tensor(scale, device=shape_tensor.device, dtype=shape_tensor.dtype)
    if torch.any(shape_tensor <= 0):
        raise ValueError("shape parameters must be positive")
    if torch.any(scale_tensor <= 0):
        raise ValueError("scale parameters must be positive")
    rate = 1.0 / scale_tensor
    gamma = torch.distributions.Gamma(concentration=shape_tensor, rate=rate)
    return gamma.rsample()


def build_orthogonal_matrix(params: Tensor, dim: int) -> Tensor:
    """Construct an orthogonal matrix from skew-symmetric parameters.

    The routine mirrors the R helper by first populating a skew-symmetric matrix using
    the upper triangular portion of ``params`` (of size ``dim * (dim - 1) / 2``),
    computing its matrix exponential and finally retrieving the orthogonal factor of
    a QR decomposition.
    """

    if params.dim() == 1:
        params = params.unsqueeze(0)
    batch, param_dim = params.shape
    expected = dim * (dim - 1) // 2
    if param_dim != expected:
        raise ValueError(f"Expected {expected} params for dim={dim}, received {param_dim}")
    device = params.device
    dtype = params.dtype
    indices = torch.triu_indices(row=dim, col=dim, offset=1, device=device)
    rows, cols = indices
    X = torch.zeros(batch, dim, dim, device=device, dtype=dtype)
    X[:, rows, cols] = params
    X = X - X.transpose(1, 2)
    Q = torch.matrix_exp(X)
    # QR decomposition to guarantee orthogonality even with numerical drift
    q, _ = torch.linalg.qr(Q)
    return q


def _kmeans_assign(data: Tensor, centroids: Tensor) -> Tensor:
    """Assign each point to the closest centroid."""

    distances = torch.cdist(data, centroids, p=2)
    return distances.argmin(dim=1)


def _update_centroids(data: Tensor, assignments: Tensor, k: int) -> Tensor:
    new_centroids = []
    for idx in range(k):
        mask = assignments == idx
        if not torch.any(mask):
            # Re-sample a random point when a cluster becomes empty
            rand_idx = torch.randint(0, data.shape[0], (1,), device=data.device)
            new_centroids.append(data[rand_idx])
        else:
            new_centroids.append(data[mask].mean(dim=0, keepdim=True))
    return torch.cat(new_centroids, dim=0)


def simple_kmeans(data: Tensor, k: int, iters: int = 15) -> Tensor:
    """Small utility K-means implementation used for mu initialisation."""

    if data.dim() != 2:
        raise ValueError("k-means expects a 2D tensor")
    if data.shape[0] < k:
        raise ValueError("number of samples must be >= k")
    # Choose random distinct starting points
    perm = torch.randperm(data.shape[0], device=data.device)
    centroids = data[perm[:k]].clone()
    for _ in range(iters):
        assignments = _kmeans_assign(data, centroids)
        centroids = _update_centroids(data, assignments, k)
    return centroids


def init_mu_kmeans(model: torch.nn.Module, outputs_train: Tensor, n_mixtures: int,
                   constant_attr: str, device: torch.device | str | None = None) -> None:
    """Initialise component means in place using K-means centroids."""

    data = outputs_train.to(device or outputs_train.device)
    centroids = simple_kmeans(data, n_mixtures)
    if "m" in constant_attr:
        with torch.no_grad():
            model.mu.copy_(centroids)
    else:
        with torch.no_grad():
            bias = model.fc_mu.bias.view(n_mixtures, -1)
            bias.copy_(centroids)
            model.fc_mu.g.zero_()

