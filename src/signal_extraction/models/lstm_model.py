"""
Long Short-Term Memory (LSTM) model for sinusoidal component extraction.

LSTM uses forget/input/output gates and a cell state to avoid the vanishing
gradient problem of vanilla RNN. Expected to outperform FC and RNN on lower
frequencies where temporal structure spans many samples.

Architecture: Linear(14→H) → reshape(B,1,H) → LSTM(H,H,L) → Linear(H→10)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from signal_extraction.constants import INPUT_SIZE, OUTPUT_SIZE
from signal_extraction.models.base_model import BaseModel


class LSTMModel(BaseModel):
    """
    LSTM extraction model.

    Strengths:  long-term memory via cell state; stable gradients through the
                forget gate; ~4× more expressive than vanilla RNN per layer.
    Weaknesses: more parameters → slower training; may overfit on small datasets.

    Gate equations (per PLAN.md PRD_ml_models section 1.5):
        f_t = σ(W_f·[h_{t-1}, x_t] + b_f)   — forget gate
        i_t = σ(W_i·[h_{t-1}, x_t] + b_i)   — input gate
        g_t = tanh(W_g·[h_{t-1}, x_t] + b_g) — cell candidate
        o_t = σ(W_o·[h_{t-1}, x_t] + b_o)   — output gate
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        h_t = o_t ⊙ tanh(c_t)

    Args:
        hidden_size: Number of LSTM hidden units per layer.
        n_layers:    Number of stacked LSTM layers.
        dropout:     Dropout between LSTM layers (only active when n_layers > 1).
    """

    def __init__(self, hidden_size: int, n_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_proj = nn.Linear(INPUT_SIZE, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
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
        # LSTM over single time-step: out shape (B, 1, H); ignore (h_n, c_n)
        out, _ = self.lstm(h)
        # Take output at the only time-step: (B, H)
        last = out[:, -1, :]
        return self.output_proj(last)
