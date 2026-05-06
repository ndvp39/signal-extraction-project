"""
Fully Connected (FC) model for sinusoidal component extraction.

Baseline model with no temporal structure. Treats all 14 input dimensions
(selector C + noisy window) as an unordered feature vector.
Architecture: Linear(14→H) → ReLU → Linear(H→H) → ReLU → Linear(H→10)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from signal_extraction.constants import INPUT_SIZE, OUTPUT_SIZE
from signal_extraction.models.base_model import BaseModel


class FCModel(BaseModel):
    """
    Fully Connected extraction model.

    Strengths:  simple, fast, no vanishing gradient across time.
    Weaknesses: no temporal awareness; treats the 10-sample window as an
                unordered set. Expected to perform best at high frequencies
                where the window contains multiple full cycles.

    Args:
        hidden_size: Number of units in each hidden layer.
        dropout:     Dropout probability applied after each ReLU (default 0.0).
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        # Why two hidden layers: one layer learns feature combinations from
        # the raw input; the second learns higher-order interactions between
        # those features before projecting to the output window.
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, OUTPUT_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass.

        Args:
            x: Float32 tensor, shape (batch_size, 14).
               Columns 0-3: one-hot selector C.
               Columns 4-13: noisy sum window.

        Returns:
            Float32 tensor, shape (batch_size, 10) — predicted clean window.
        """
        return self.net(x)
