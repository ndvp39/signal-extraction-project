"""
Unit tests for TrainerService — written BEFORE implementation (TDD RED phase).

Uses tiny models and datasets to keep tests fast (< 10 seconds).
Requirements source: ASSIGNMENT.txt section 13, docs/TODO.md T-060.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from signal_extraction.constants import INPUT_SIZE, OUTPUT_SIZE
from signal_extraction.models.fc_model import FCModel
from signal_extraction.services.trainer import TrainerService
from signal_extraction.shared.schemas import TrainResult

N = 64
HIDDEN = 16
EPOCHS = 3
LR = 0.01
PATIENCE = 5


def _make_loader(n: int = N, batch: int = 16) -> DataLoader:
    """Create a tiny random DataLoader for testing."""
    x = torch.randn(n, INPUT_SIZE)
    y = torch.randn(n, OUTPUT_SIZE)
    return DataLoader(TensorDataset(x, y), batch_size=batch)


@pytest.fixture()
def model() -> FCModel:
    return FCModel(hidden_size=HIDDEN)


@pytest.fixture()
def train_loader() -> DataLoader:
    return _make_loader()


@pytest.fixture()
def val_loader() -> DataLoader:
    return _make_loader(n=16)


# ---------------------------------------------------------------------------
# TrainResult structure
# ---------------------------------------------------------------------------


def test_train_result_has_correct_fields(model, train_loader, val_loader, tmp_path) -> None:
    result = TrainerService().train(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LR, patience=PATIENCE,
        save_path=str(tmp_path / "best.pt"),
    )
    assert isinstance(result, TrainResult)
    assert isinstance(result.train_losses, list)
    assert isinstance(result.val_losses, list)
    assert isinstance(result.best_epoch, int)
    assert isinstance(result.model_path, str)


def test_loss_lists_length_equals_epochs(model, train_loader, val_loader, tmp_path) -> None:
    result = TrainerService().train(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LR, patience=PATIENCE,
        save_path=str(tmp_path / "best.pt"),
    )
    assert len(result.train_losses) == EPOCHS
    assert len(result.val_losses) == EPOCHS


def test_all_losses_are_finite(model, train_loader, val_loader, tmp_path) -> None:
    result = TrainerService().train(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LR, patience=PATIENCE,
        save_path=str(tmp_path / "best.pt"),
    )
    for loss in result.train_losses + result.val_losses:
        assert loss > 0 and loss < float("inf")


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------


def test_checkpoint_is_saved(model, train_loader, val_loader, tmp_path) -> None:
    save_path = str(tmp_path / "best.pt")
    TrainerService().train(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LR, patience=PATIENCE,
        save_path=save_path,
    )
    assert Path(save_path).exists()


def test_model_path_in_result_matches_save_path(model, train_loader, val_loader, tmp_path) -> None:
    save_path = str(tmp_path / "best.pt")
    result = TrainerService().train(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LR, patience=PATIENCE,
        save_path=save_path,
    )
    assert result.model_path == save_path


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


def test_early_stopping_halts_before_max_epochs(model, train_loader, val_loader, tmp_path) -> None:
    # With patience=1 and many epochs, training should stop early.
    result = TrainerService().train(
        model, train_loader, val_loader,
        epochs=50, lr=LR, patience=1,
        save_path=str(tmp_path / "best.pt"),
    )
    assert len(result.train_losses) < 50


# ---------------------------------------------------------------------------
# Loss decreases (sanity — not guaranteed but expected on a learnable task)
# ---------------------------------------------------------------------------


def test_loss_decreases_over_training() -> None:
    # Use a learnable task: y = x[:, :OUTPUT_SIZE] so the model can fit perfectly.
    x = torch.randn(128, INPUT_SIZE)
    y = x[:, :OUTPUT_SIZE]
    loader = DataLoader(TensorDataset(x, y), batch_size=32)
    m = FCModel(hidden_size=64)
    result = TrainerService().train(
        m, loader, loader,
        epochs=20, lr=0.01, patience=20,
        save_path="results/test_best.pt",
    )
    assert result.train_losses[-1] < result.train_losses[0]
