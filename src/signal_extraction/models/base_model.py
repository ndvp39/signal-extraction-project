"""
Abstract base class for all signal extraction models.

Defines the single contract all models must satisfy: accept a (B, 14)
input tensor and return a (B, 10) output tensor. This allows the SDK,
trainer, and evaluator to work with any model type identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base for FC, RNN, and LSTM extraction models.

    Input:  float32 tensor of shape (batch_size, INPUT_SIZE=14)
            Columns 0-3:  one-hot selector C
            Columns 4-13: noisy sum window (10 samples)

    Output: float32 tensor of shape (batch_size, OUTPUT_SIZE=10)
            Predicted clean sinusoid window for the selected component.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the model.

        Args:
            x: Float32 tensor, shape (batch_size, 14).

        Returns:
            Float32 tensor, shape (batch_size, 10).
        """
