"""
Recurrent Neural Network (RNN) model for sinusoidal component extraction.

The 14-dim input is treated as a single time-step sequence (seq_len=1) fed
to a vanilla RNN, then the final hidden state is projected to the 10-dim
output window. See PLAN.md ADR-03 for the rationale behind single-step input.

Architecture: Linear(14→H) → reshape(B,1,H) → RNN(H,H,L) → Linear(H→10)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from signal_extraction.constants import INPUT_SIZE, OUTPUT_SIZE
from signal_extraction.models.base_model import BaseModel


class RNNModel(BaseModel):
    """
    Vanilla RNN extraction model.

    Strengths:  captures short-term temporal patterns; fewer parameters than LSTM.
    Weaknesses: vanishing gradients over long sequences; limited memory capacity.
                With a single time-step input, performance is similar to FC but
                with a recurrent inductive bias.

    Args:
        hidden_size: Number of RNN hidden units per layer.
        n_layers:    Number of stacked RNN layers.
        dropout:     Dropout between RNN layers (only active when n_layers > 1).
    """

    def __init__(self, hidden_size: int, n_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        # Project raw input to hidden dimension before the RNN so that the
        # recurrent weight matrix is always square (H×H), regardless of INPUT_SIZE.
        self.input_proj = nn.Linear(INPUT_SIZE, hidden_size)
        self.rnn = nn.RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass.

        Args:
            x: Float32 tensor, shape (batch_size, 14).

        Returns:
            Float32 tensor, shape (batch_size, 10).
        """
        # Project input and add sequence dimension: (B, 14) → (B, 1, H)
        h = self.input_proj(x).unsqueeze(1)
        # RNN over single time-step: out shape (B, 1, H)
        out, _ = self.rnn(h)
        # Take the output at the only time-step: (B, H)
        last = out[:, -1, :]
        return self.output_proj(last)
