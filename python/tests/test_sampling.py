import pytest

torch = pytest.importorskip("torch")

from mst_pmdn.model import MSTPMDN
from mst_pmdn.sampling import sample_mst_pmdn


def test_sampling_shapes_and_component_range():
    torch.manual_seed(1)
    model = MSTPMDN(input_dim=3, output_dim=2, hidden_dim=6, n_mixtures=4)
    x = torch.randn(5, 3)
    out = model(x)
    samples = sample_mst_pmdn(out, num_samples=10)
    assert samples["samples"].shape == (10, 5, 2)
    assert samples["components"].shape == (10, 5)
    assert samples["components"].max() < 4
    assert samples["components"].min() >= 0
