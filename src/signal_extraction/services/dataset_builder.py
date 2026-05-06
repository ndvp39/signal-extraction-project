"""
DatasetBuilderService — builds training samples from a SignalBundle.

Each sample is constructed per ASSIGNMENT.txt sections 8-12:
  Step 1: choose a random selector C (one-hot, size 4)
  Step 2: choose a random window position in the noisy sum signal
  Step 3: take window of length 10 from the noisy sum signal
  Step 4: take the same window from the clean sinusoid selected by C

Also provides SignalDataset (torch Dataset) and a train/val/test splitter.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from signal_extraction.constants import N_SIGNALS, SELECTOR_SIZE, WINDOW_SIZE
from signal_extraction.shared.schemas import Sample, SignalBundle


class SignalDataset(Dataset):
    """
    Torch Dataset wrapping a list of Sample objects.

    Returns (x, y) float32 tensors for use with DataLoader.
    x shape: (INPUT_SIZE,) = (14,)
    y shape: (OUTPUT_SIZE,) = (10,)
    """

    def __init__(self, samples: list[Sample]) -> None:
        """
        Args:
            samples: List of Sample dataclass instances produced by
                     DatasetBuilderService.build().
        """
        self._x = torch.tensor(
            np.stack([s.x for s in samples]), dtype=torch.float32
        )
        self._y = torch.tensor(
            np.stack([s.y for s in samples]), dtype=torch.float32
        )

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self._x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (x, y) tensor pair at index idx."""
        return self._x[idx], self._y[idx]


class DatasetBuilderService:
    """
    Builds dataset samples from a pre-generated SignalBundle.

    Single responsibility: sampling windows and selectors. Does not generate
    signals (SignalGeneratorService) or train models (TrainerService).
    """

    def __init__(self, seed: int = 0) -> None:
        """
        Args:
            seed: Random seed for reproducible window/selector sampling.
                  Separate from the signal generation seed in SignalParams.
        """
        self._seed = seed

    def build(self, bundle: SignalBundle, n_samples: int) -> list[Sample]:
        """
        Generate n_samples training samples from the given SignalBundle.

        Args:
            bundle:    Pre-generated SignalBundle (clean + noisy signals).
            n_samples: Number of samples to generate.

        Returns:
            List of Sample objects, each with x=(14,) and y=(10,).
        """
        rng = np.random.default_rng(self._seed)
        n_positions = len(bundle.t) - WINDOW_SIZE
        samples: list[Sample] = []
        for _ in range(n_samples):
            selector_idx = int(rng.integers(0, N_SIGNALS))
            pos = int(rng.integers(0, n_positions + 1))
            sample = self._make_sample(bundle, pos, selector_idx)
            samples.append(sample)
        return samples

    def split(
        self,
        samples: list[Sample],
        ratios: list[float],
    ) -> tuple[list[Sample], list[Sample], list[Sample]]:
        """
        Split samples into train / validation / test sets.

        Args:
            samples: Full list of Sample objects from build().
            ratios:  Three floats [train, val, test] summing to 1.0.

        Returns:
            Tuple of (train_samples, val_samples, test_samples).
        """
        n = len(samples)
        n_train = round(ratios[0] * n)
        n_val = round(ratios[1] * n)
        train = samples[:n_train]
        val = samples[n_train : n_train + n_val]
        test = samples[n_train + n_val :]
        return train, val, test

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_one_hot(idx: int) -> np.ndarray:
        """Build a one-hot vector of length N_SIGNALS with a 1 at idx."""
        vec = np.zeros(SELECTOR_SIZE, dtype=np.float64)
        vec[idx] = 1.0
        return vec

    @staticmethod
    def _make_sample(bundle: SignalBundle, pos: int, selector_idx: int) -> Sample:
        """
        Construct one Sample from a window position and a selector index.

        x = [one-hot C | noisy_sum[pos:pos+WINDOW_SIZE]]
        y = clean_si[pos:pos+WINDOW_SIZE]  where i = selector_idx + 1
        """
        selector = DatasetBuilderService._make_one_hot(selector_idx)
        noisy_window = bundle.noisy["sum"][pos : pos + WINDOW_SIZE]
        clean_key = f"s{selector_idx + 1}"
        clean_window = bundle.clean[clean_key][pos : pos + WINDOW_SIZE]
        x = np.concatenate([selector, noisy_window])
        return Sample(x=x, y=clean_window)
