"""
Unit tests for FCModel — written BEFORE implementation (TDD RED phase).

Requirements source: ASSIGNMENT.txt section 14, PRD_ml_models.md ML-T01/T04/T05/T08/T09/T12.
"""

from __future__ import annotations

import torch
import pytest

from signal_extraction.constants import INPUT_SIZE, OUTPUT_SIZE
from signal_extraction.models.fc_model import FCModel

HIDDEN_SIZE = 64
BATCH_SIZE = 8


@pytest.fixture()
def model() -> FCModel:
    return FCModel(hidden_size=HIDDEN_SIZE)


# ---------------------------------------------------------------------------
# ML-T01 — Output shape
# ---------------------------------------------------------------------------


def test_output_shape(model: FCModel) -> None:
    x = torch.randn(BATCH_SIZE, INPUT_SIZE)
    y = model(x)
    assert y.shape == (BATCH_SIZE, OUTPUT_SIZE)


# ---------------------------------------------------------------------------
# ML-T08 — Single-sample batch
# ---------------------------------------------------------------------------


def test_single_sample_batch(model: FCModel) -> None:
    x = torch.randn(1, INPUT_SIZE)
    y = model(x)
    assert y.shape == (1, OUTPUT_SIZE)


# ---------------------------------------------------------------------------
# ML-T04 — All models accept same input format
# ---------------------------------------------------------------------------


def test_forward_does_not_raise(model: FCModel) -> None:
    x = torch.randn(BATCH_SIZE, INPUT_SIZE)
    try:
        model(x)
    except Exception as exc:
        pytest.fail(f"forward() raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# ML-T09 — Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_are_not_none_after_backward(model: FCModel) -> None:
    x = torch.randn(BATCH_SIZE, INPUT_SIZE)
    loss = model(x).sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# ML-T12 — Model serialisation round-trip
# ---------------------------------------------------------------------------


def test_save_load_produces_identical_output(model: FCModel, tmp_path) -> None:
    x = torch.randn(BATCH_SIZE, INPUT_SIZE)
    model.eval()
    with torch.no_grad():
        out_before = model(x)
    path = tmp_path / "fc_model.pt"
    torch.save(model.state_dict(), path)
    loaded = FCModel(hidden_size=HIDDEN_SIZE)
    loaded.load_state_dict(torch.load(path, weights_only=True))
    loaded.eval()
    with torch.no_grad():
        out_after = loaded(x)
    torch.testing.assert_close(out_before, out_after)


# ---------------------------------------------------------------------------
# ML-T13 — Config-driven hidden size changes parameter count
# ---------------------------------------------------------------------------


def test_larger_hidden_size_has_more_parameters() -> None:
    small = FCModel(hidden_size=32)
    large = FCModel(hidden_size=128)
    small_params = sum(p.numel() for p in small.parameters())
    large_params = sum(p.numel() for p in large.parameters())
    assert large_params > small_params
