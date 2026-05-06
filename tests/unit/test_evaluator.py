"""
Unit tests for EvaluatorService — written BEFORE implementation (TDD RED phase).

Requirements source: ASSIGNMENT.txt section 15, docs/TODO.md T-062.
"""

from __future__ import annotations

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from signal_extraction.constants import INPUT_SIZE, N_SIGNALS, OUTPUT_SIZE, SELECTOR_SIZE
from signal_extraction.models.fc_model import FCModel
from signal_extraction.services.evaluator import EvaluatorService
from signal_extraction.shared.schemas import EvalResult

N = 80
HIDDEN = 16


def _make_loader_with_selectors(n: int = N) -> DataLoader:
    """
    Build a DataLoader where x has valid one-hot selectors in the first 4 dims,
    so per-frequency MSE can be computed correctly.
    """
    x = torch.zeros(n, INPUT_SIZE)
    # Assign one-hot selectors cycling through all 4 frequencies
    for i in range(n):
        x[i, i % N_SIGNALS] = 1.0
    x[:, SELECTOR_SIZE:] = torch.randn(n, OUTPUT_SIZE)   # noisy window
    y = torch.randn(n, OUTPUT_SIZE)
    return DataLoader(TensorDataset(x, y), batch_size=16)


@pytest.fixture()
def model() -> FCModel:
    return FCModel(hidden_size=HIDDEN)


@pytest.fixture()
def loader() -> DataLoader:
    return _make_loader_with_selectors()


# ---------------------------------------------------------------------------
# EvalResult structure
# ---------------------------------------------------------------------------


def test_eval_result_is_correct_type(model, loader) -> None:
    result = EvaluatorService().evaluate(model, loader)
    assert isinstance(result, EvalResult)


def test_mse_overall_is_positive_finite(model, loader) -> None:
    result = EvaluatorService().evaluate(model, loader)
    assert result.mse_overall > 0
    assert result.mse_overall < float("inf")


def test_mse_per_freq_has_all_indices(model, loader) -> None:
    result = EvaluatorService().evaluate(model, loader)
    assert set(result.mse_per_freq.keys()) == set(range(N_SIGNALS))


def test_mse_per_freq_values_are_positive(model, loader) -> None:
    result = EvaluatorService().evaluate(model, loader)
    for v in result.mse_per_freq.values():
        assert v > 0


# ---------------------------------------------------------------------------
# Predictions and targets shape
# ---------------------------------------------------------------------------


def test_predictions_shape(model, loader) -> None:
    result = EvaluatorService().evaluate(model, loader)
    assert result.predictions.shape == (N, OUTPUT_SIZE)


def test_targets_shape(model, loader) -> None:
    result = EvaluatorService().evaluate(model, loader)
    assert result.targets.shape == (N, OUTPUT_SIZE)


# ---------------------------------------------------------------------------
# Perfect predictor has MSE = 0
# ---------------------------------------------------------------------------


def test_perfect_predictor_has_zero_mse() -> None:
    """A model that always outputs the target should have MSE ≈ 0."""
    x = torch.zeros(32, INPUT_SIZE)
    for i in range(32):
        x[i, i % N_SIGNALS] = 1.0
    y = torch.randn(32, OUTPUT_SIZE)
    loader = DataLoader(TensorDataset(x, y), batch_size=32)

    # Override model to return y directly
    class PerfectModel(FCModel):
        def forward(self, inp):  # noqa: N802
            return y  # always returns the exact targets

    result = EvaluatorService().evaluate(PerfectModel(hidden_size=HIDDEN), loader)
    assert result.mse_overall == pytest.approx(0.0, abs=1e-6)
