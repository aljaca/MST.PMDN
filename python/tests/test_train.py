import pytest

torch = pytest.importorskip("torch")

from mst_pmdn.losses import mst_pmdn_nll
from mst_pmdn.model import MSTPMDN
from mst_pmdn.train import train_mst_pmdn


def test_training_loop_reduces_loss():
    torch.manual_seed(42)
    inputs = torch.randn(200, 3)
    weights = torch.tensor([[1.0, -0.5, 0.25], [-0.75, 0.6, 0.4]])
    noise = 0.1 * torch.randn(200, 2)
    outputs = inputs @ weights.t() + noise

    model = MSTPMDN(input_dim=3, output_dim=2, hidden_dim=(16,), n_mixtures=2, constraint="VVVVE")
    with torch.no_grad():
        initial_loss = mst_pmdn_nll(model(inputs), outputs).item()

    history = train_mst_pmdn(model, inputs, outputs, epochs=20, batch_size=32, lr=5e-3, val_split=0.2, patience=5)

    with torch.no_grad():
        final_loss = mst_pmdn_nll(model(inputs), outputs).item()

    assert history.train_loss
    assert final_loss < initial_loss
