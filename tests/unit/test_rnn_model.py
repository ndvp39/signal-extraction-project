"""
Unit tests for RNNModel — written BEFORE implementation (TDD RED phase).

Requirements source: ASSIGNMENT.txt section 14, PRD_ml_models.md ML-T02/T04/T08/T10/T12/T13.
"""

from __future__ import annotations

import torch
import pytest

from signal_extraction.constants import INPUT_SIZE, OUTPUT_SIZE
from signal_extraction.models.rnn_model import RNNModel

HIDDEN_SIZE = 64
N_LAYERS = 2
BATCH_SIZE = 8


@pytest.fixture()
def model() -> RNNModel:
    return RNNModel(hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS)


# ---------------------------------------------------------------------------
# ML-T02 — Output shape
# ---------------------------------------------------------------------------


def test_output_shape(model: RNNModel) -> None:
    x = torch.randn(BATCH_SIZE, INPUT_SIZE)
    y = model(x)
    assert y.shape == (BATCH_SIZE, OUTPUT_SIZE)


# ---------------------------------------------------------------------------
# ML-T08 — Single-sample batch
# ---------------------------------------------------------------------------


def test_single_sample_batch(model: RNNModel) -> None:
    x = torch.randn(1, INPUT_SIZE)
    y = model(x)
    assert y.shape == (1, OUTPUT_SIZE)


# ---------------------------------------------------------------------------
# ML-T04 — Forward pass does not raise
# ---------------------------------------------------------------------------


def test_forward_does_not_raise(model: RNNModel) -> None:
    x = torch.randn(BATCH_SIZE, INPUT_SIZE)
    try:
        model(x)
    except Exception as exc:
        pytest.fail(f"forward() raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# ML-T10 — Gradient flow including recurrent weights
# ---------------------------------------------------------------------------


def test_gradients_are_not_none_after_backward(model: RNNModel) -> None:
    x = torch.randn(BATCH_SIZE, INPUT_SIZE)
    loss = model(x).sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# ML-T12 — Serialisation round-trip
# ---------------------------------------------------------------------------


def test_save_load_produces_identical_output(model: RNNModel, tmp_path) -> None:
    x = torch.randn(BATCH_SIZE, INPUT_SIZE)
    model.eval()
    with torch.no_grad():
        out_before = model(x)
    path = tmp_path / "rnn_model.pt"
    torch.save(model.state_dict(), path)
    loaded = RNNModel(hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS)
    loaded.load_state_dict(torch.load(path, weights_only=True))
    loaded.eval()
    with torch.no_grad():
        out_after = loaded(x)
    torch.testing.assert_close(out_before, out_after)


# ---------------------------------------------------------------------------
# ML-T13 — Configurable hidden size and layers
# ---------------------------------------------------------------------------


def test_larger_hidden_size_has_more_parameters() -> None:
    small = RNNModel(hidden_size=32, n_layers=1)
    large = RNNModel(hidden_size=128, n_layers=1)
    assert sum(p.numel() for p in large.parameters()) > sum(p.numel() for p in small.parameters())


def test_more_layers_has_more_parameters() -> None:
    shallow = RNNModel(hidden_size=HIDDEN_SIZE, n_layers=1)
    deep = RNNModel(hidden_size=HIDDEN_SIZE, n_layers=3)
    assert sum(p.numel() for p in deep.parameters()) > sum(p.numel() for p in shallow.parameters())
