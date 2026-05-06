"""
Unit tests for DatasetBuilderService and SignalDataset — written BEFORE
implementation (TDD RED phase).

Requirements source: ASSIGNMENT.txt sections 8–12.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from signal_extraction.constants import (
    INPUT_SIZE,
    N_SIGNALS,
    OUTPUT_SIZE,
    SELECTOR_SIZE,
    WINDOW_SIZE,
)
from signal_extraction.services.dataset_builder import DatasetBuilderService, SignalDataset
from signal_extraction.services.signal_generator import SignalGeneratorService
from signal_extraction.shared.schemas import SignalBundle, SignalParams

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundle() -> SignalBundle:
    """Generate a real SignalBundle once for the whole module."""
    params = SignalParams(
        frequencies=[10.0, 50.0, 120.0, 300.0],
        amplitudes=[1.0, 0.8, 1.2, 0.6],
        phases=[0.0, 0.5, 1.0, 1.5],
        alpha=0.1,
        beta=0.1,
        noise_dist="gaussian",
        sample_rate=1000,
        duration=10.0,
        seed=42,
    )
    return SignalGeneratorService().generate(params)


@pytest.fixture(scope="module")
def builder() -> DatasetBuilderService:
    return DatasetBuilderService(seed=0)


@pytest.fixture(scope="module")
def samples(bundle: SignalBundle, builder: DatasetBuilderService):
    return builder.build(bundle, n_samples=200)


# ---------------------------------------------------------------------------
# T-040: Sample shape and structure
# ---------------------------------------------------------------------------


def test_sample_x_shape(samples) -> None:
    """x must be [C(4) | noisy_sum_window(10)] = shape (14,)."""
    for s in samples[:10]:
        assert s.x.shape == (INPUT_SIZE,), f"Expected ({INPUT_SIZE},), got {s.x.shape}"


def test_sample_y_shape(samples) -> None:
    """y must be the clean target window, shape (10,)."""
    for s in samples[:10]:
        assert s.y.shape == (OUTPUT_SIZE,), f"Expected ({OUTPUT_SIZE},), got {s.y.shape}"


def test_selector_is_one_hot(samples) -> None:
    """First N_SIGNALS elements of x must be a valid one-hot vector."""
    for s in samples[:50]:
        selector = s.x[:SELECTOR_SIZE]
        assert selector.sum() == pytest.approx(1.0), "Selector must sum to 1"
        assert set(selector).issubset({0.0, 1.0}), "Selector must contain only 0s and 1s"


def test_selector_covers_all_signals(samples) -> None:
    """Over many samples, every sinusoid index must be selected at least once."""
    seen = set()
    for s in samples:
        idx = int(np.argmax(s.x[:SELECTOR_SIZE]))
        seen.add(idx)
    assert seen == set(range(N_SIGNALS)), f"Not all signals were selected: {seen}"


# ---------------------------------------------------------------------------
# T-040: Window bounds
# ---------------------------------------------------------------------------


def test_noisy_window_matches_bundle(bundle: SignalBundle, builder: DatasetBuilderService) -> None:
    """
    The noisy portion of x must exactly equal the noisy sum signal
    at the chosen window position.
    """
    samples = builder.build(bundle, n_samples=50)
    noisy_sum = bundle.noisy["sum"]
    for s in samples:
        window = s.x[SELECTOR_SIZE:]
        # Find where this window appears in the noisy sum
        found = False
        for pos in range(len(noisy_sum) - WINDOW_SIZE + 1):
            if np.allclose(noisy_sum[pos : pos + WINDOW_SIZE], window):
                found = True
                break
        assert found, "Noisy window not found in noisy sum signal"


def test_clean_window_matches_selected_signal(bundle: SignalBundle, builder: DatasetBuilderService) -> None:
    """
    y must exactly equal the clean sinusoid selected by C at the same position
    as the noisy window.
    """
    samples = builder.build(bundle, n_samples=50)
    noisy_sum = bundle.noisy["sum"]
    for s in samples:
        sel_idx = int(np.argmax(s.x[:SELECTOR_SIZE]))
        window = s.x[SELECTOR_SIZE:]
        clean_key = f"s{sel_idx + 1}"
        # find window position in noisy sum
        for pos in range(len(noisy_sum) - WINDOW_SIZE + 1):
            if np.allclose(noisy_sum[pos : pos + WINDOW_SIZE], window):
                expected_y = bundle.clean[clean_key][pos : pos + WINDOW_SIZE]
                np.testing.assert_allclose(s.y, expected_y, atol=1e-10)
                break


# ---------------------------------------------------------------------------
# T-043: DataLoader batching
# ---------------------------------------------------------------------------


def test_dataset_len(samples) -> None:
    ds = SignalDataset(samples)
    assert len(ds) == len(samples)


def test_dataset_getitem_types(samples) -> None:
    ds = SignalDataset(samples)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_dataset_getitem_shapes(samples) -> None:
    ds = SignalDataset(samples)
    x, y = ds[0]
    assert x.shape == (INPUT_SIZE,)
    assert y.shape == (OUTPUT_SIZE,)


def test_dataset_dtype_is_float32(samples) -> None:
    ds = SignalDataset(samples)
    x, y = ds[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_dataloader_batch_shapes(samples) -> None:
    from torch.utils.data import DataLoader
    ds = SignalDataset(samples)
    loader = DataLoader(ds, batch_size=16)
    x_batch, y_batch = next(iter(loader))
    assert x_batch.shape == (16, INPUT_SIZE)
    assert y_batch.shape == (16, OUTPUT_SIZE)


# ---------------------------------------------------------------------------
# T-040: Train/val/test split ratios
# ---------------------------------------------------------------------------


def test_split_sizes_sum_to_total(bundle: SignalBundle, builder: DatasetBuilderService) -> None:
    n = 100
    samples = builder.build(bundle, n_samples=n)
    train, val, test = builder.split(samples, ratios=[0.7, 0.15, 0.15])
    assert len(train) + len(val) + len(test) == n


def test_split_ratios_approximately_correct(bundle: SignalBundle, builder: DatasetBuilderService) -> None:
    n = 1000
    samples = builder.build(bundle, n_samples=n)
    train, val, test = builder.split(samples, ratios=[0.7, 0.15, 0.15])
    assert abs(len(train) - 700) <= 5
    assert abs(len(val) - 150) <= 5
    assert abs(len(test) - 150) <= 5


# ---------------------------------------------------------------------------
# T-044: Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_samples(bundle: SignalBundle) -> None:
    b1 = DatasetBuilderService(seed=7).build(bundle, n_samples=50)
    b2 = DatasetBuilderService(seed=7).build(bundle, n_samples=50)
    for s1, s2 in zip(b1, b2):
        np.testing.assert_array_equal(s1.x, s2.x)
        np.testing.assert_array_equal(s1.y, s2.y)


def test_different_seeds_produce_different_samples(bundle: SignalBundle) -> None:
    b1 = DatasetBuilderService(seed=1).build(bundle, n_samples=50)
    b2 = DatasetBuilderService(seed=2).build(bundle, n_samples=50)
    assert not np.array_equal(b1[0].x, b2[0].x)
