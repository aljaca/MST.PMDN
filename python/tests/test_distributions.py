import pytest

torch = pytest.importorskip("torch")

from mst_pmdn.distributions import log_student_t_cdf, student_t_cdf


def test_student_t_cdf_matches_torch_distribution():
    z = torch.linspace(-3, 3, 7)
    nu = torch.tensor([3.0])
    expected = torch.distributions.StudentT(df=nu).cdf(z)
    actual = student_t_cdf(z, nu)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_log_student_t_cdf_has_grad():
    z = torch.tensor([0.1, 0.5, -0.2], requires_grad=True)
    nu = torch.tensor([4.0])
    loss = log_student_t_cdf(z, nu).sum()
    loss.backward()
    assert z.grad is not None
    assert torch.all(torch.isfinite(z.grad))
