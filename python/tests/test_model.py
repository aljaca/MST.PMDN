import pytest

torch = pytest.importorskip("torch")

from mst_pmdn.model import MSTPMDN


def make_model(**kwargs):
    return MSTPMDN(
        input_dim=4,
        output_dim=2,
        hidden_dim=(8, 8),
        n_mixtures=3,
        **kwargs,
    )


def test_forward_shapes_and_constraints():
    model = make_model()
    x = torch.randn(5, 4)
    out = model(x)
    assert out["pi"].shape == (5, 3)
    assert torch.allclose(out["pi"].sum(dim=1), torch.ones(5))
    assert out["mu"].shape == (5, 3, 2)
    assert out["scale_chol"].shape == (5, 3, 2, 2)
    assert out["alpha"].shape == (5, 3, 2)
    diag = out["scale_chol"].diagonal(dim1=-2, dim2=-1)
    assert torch.all(diag > 0)


def test_degrees_of_freedom_clamped_to_range():
    model = make_model(constraint="VVVVE")
    x = torch.randn(2, 4)
    out = model(x)
    nu = out["nu"]
    assert torch.all((nu >= model.min_nu - 1e-5) & (nu <= model.max_nu + 1e-5))


def test_no_skew_returns_zero_alpha():
    model = make_model(constraint="VVVVN")
    x = torch.randn(3, 4)
    out = model(x)
    assert torch.allclose(out["alpha"], torch.zeros_like(out["alpha"]))


def test_shared_orientation_is_orthogonal():
    model = make_model(constraint="EEEVV")
    x = torch.randn(2, 4)
    out = model(x)
    D = out["D"]  # [B, M, d, d]
    mat = D[0, 0]
    should_be_identity = mat @ mat.t()
    assert torch.allclose(should_be_identity, torch.eye(2), atol=1e-5)


def test_constant_mixture_weights_shared_across_batch():
    model = make_model(constraint="VVVNN", constant_attr="x")
    x = torch.randn(6, 4)
    out = model(x)
    assert torch.allclose(out["pi"], out["pi"][0].unsqueeze(0).expand_as(out["pi"]))
