"""
EvaluatorService — evaluates a trained model on a test DataLoader.

Computes overall MSE and per-frequency MSE by reading the one-hot selector C
from the first N_SIGNALS dimensions of each input vector x.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from signal_extraction.constants import N_SIGNALS, SELECTOR_SIZE
from signal_extraction.models.base_model import BaseModel
from signal_extraction.shared.schemas import EvalResult


class EvaluatorService:
    """
    Evaluates any BaseModel subclass on a test DataLoader.

    Single responsibility: inference and metric computation only.
    Does not train models or build datasets.
    """

    def evaluate(self, model: BaseModel, loader: DataLoader) -> EvalResult:
        """
        Run inference on all batches and compute MSE metrics.

        Args:
            model:  Trained model (FC, RNN, or LSTM).
            loader: DataLoader for the test set.

        Returns:
            EvalResult with overall MSE, per-frequency MSE, and raw arrays.
        """
        criterion = nn.MSELoss(reduction="none")
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_selectors: list[torch.Tensor] = []

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in loader:
                preds = model(x_batch)
                all_preds.append(preds)
                all_targets.append(y_batch)
                # Selector is the first SELECTOR_SIZE columns of x
                all_selectors.append(x_batch[:, :SELECTOR_SIZE])

        preds_np = torch.cat(all_preds).numpy()
        targets_np = torch.cat(all_targets).numpy()
        selectors = torch.cat(all_selectors)

        # Compute per-sample MSE then average — avoids zip(strict=) which
        # requires Python 3.10; preds_cat and targets_cat are already aligned.
        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)
        mse_overall = float(criterion(preds_cat, targets_cat).mean(dim=1).mean())

        mse_per_freq = self._mse_per_frequency(
            preds_np, targets_np, selectors.numpy()
        )

        return EvalResult(
            mse_overall=mse_overall,
            mse_per_freq=mse_per_freq,
            predictions=preds_np,
            targets=targets_np,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mse_per_frequency(
        preds: np.ndarray,
        targets: np.ndarray,
        selectors: np.ndarray,
    ) -> dict[int, float]:
        """
        Compute MSE separately for each selected sinusoid.

        Groups samples by their one-hot selector index (argmax of the
        selector vector) and computes MSE within each group.

        Args:
            preds:     Predicted windows, shape (N, OUTPUT_SIZE).
            targets:   Clean target windows, shape (N, OUTPUT_SIZE).
            selectors: One-hot selector vectors, shape (N, N_SIGNALS).

        Returns:
            Dict mapping frequency index (0–3) to its mean MSE.
        """
        freq_indices = np.argmax(selectors, axis=1)
        result: dict[int, float] = {}
        for freq_idx in range(N_SIGNALS):
            mask = freq_indices == freq_idx
            if mask.sum() == 0:
                result[freq_idx] = 0.0
                continue
            diff = preds[mask] - targets[mask]
            result[freq_idx] = float(np.mean(diff ** 2))
        return result
