"""
TrainerService — trains a model using MSE loss and Adam optimizer.

Implements the training loop specified in ASSIGNMENT.txt section 13:
loss = mean squared error between predicted and clean target window.
Supports early stopping and best-checkpoint saving.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from signal_extraction.models.base_model import BaseModel
from signal_extraction.shared.schemas import TrainResult


class TrainerService:
    """
    Trains any BaseModel subclass with MSE loss and Adam optimizer.

    Single responsibility: the training loop only. Does not build datasets,
    generate signals, or evaluate models.
    """

    def train(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        patience: int,
        save_path: str,
    ) -> TrainResult:
        """
        Run the full training loop with early stopping.

        Args:
            model:        Model to train (FC, RNN, or LSTM).
            train_loader: DataLoader for training samples.
            val_loader:   DataLoader for validation samples.
            epochs:       Maximum number of training epochs.
            lr:           Adam learning rate.
            patience:     Stop if val loss does not improve for this many epochs.
            save_path:    File path to save the best model checkpoint.

        Returns:
            TrainResult with per-epoch losses, best epoch, and checkpoint path.
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val = float("inf")
        best_epoch = 0
        no_improve = 0

        for epoch in range(epochs):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss = self._validate(model, val_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                no_improve = 0
                self._save_checkpoint(model, save_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return TrainResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            model_path=save_path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _train_epoch(
        model: BaseModel,
        loader: DataLoader,
        criterion: nn.MSELoss,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one training epoch; return mean loss over all batches."""
        model.train()
        total, count = 0.0, 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
            count += 1
        return total / count

    @staticmethod
    def _validate(
        model: BaseModel,
        loader: DataLoader,
        criterion: nn.MSELoss,
    ) -> float:
        """Evaluate model on validation set; return mean loss."""
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                total += criterion(model(x_batch), y_batch).item()
                count += 1
        return total / count

    @staticmethod
    def _save_checkpoint(model: BaseModel, path: str) -> None:
        """Persist the model's state dict to disk."""
        torch.save(model.state_dict(), path)
