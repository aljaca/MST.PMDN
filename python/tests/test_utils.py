import pytest

torch = pytest.importorskip("torch")

from mst_pmdn.utils import build_orthogonal_matrix, sample_gamma, simple_kmeans


def test_sample_gamma_matches_mean():
    shape = torch.tensor([2.0, 3.0])
    scale = torch.tensor([1.5, 0.7])
    samples = sample_gamma(shape.unsqueeze(0).expand(10000, -1), scale)
    empirical_mean = samples.mean(dim=0)
    expected = shape * scale
    assert torch.allclose(empirical_mean, expected, atol=0.05, rtol=0.05)


def test_build_orthogonal_matrix_returns_orthonormal_rows():
    params = torch.randn(4, 3)
    mat = build_orthogonal_matrix(params, dim=3)
    should_be_identity = mat @ mat.transpose(1, 2)
    eye = torch.eye(3)
    assert torch.allclose(should_be_identity, eye, atol=1e-5, rtol=1e-5)


def test_simple_kmeans_finds_centroids():
    torch.manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
    points = torch.cat([centers[0].unsqueeze(0) + 0.1 * torch.randn(100, 2),
                        centers[1].unsqueeze(0) + 0.1 * torch.randn(100, 2)])
    centroids = simple_kmeans(points, k=2, iters=20)
    dists = torch.cdist(centroids, centers)
    assignment = dists.argmin(dim=1)
    assert torch.all(dists[torch.arange(2), assignment] < 0.3)
