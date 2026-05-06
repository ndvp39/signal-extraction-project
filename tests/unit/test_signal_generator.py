"""
Unit tests for SignalGeneratorService — written BEFORE the implementation (TDD).

Tests are derived strictly from ASSIGNMENT.txt and PRD_signal_generation.md.
No feature is tested here that is not explicitly required by those documents.
"""

import numpy as np
import pytest

from signal_extraction.constants import (
    DURATION,
    N_SAMPLES,
    N_SIGNALS,
    NOISE_DIST_GAUSSIAN,
    NOISE_DIST_UNIFORM,
    SAMPLE_RATE,
)
from signal_extraction.services.signal_generator import SignalGeneratorService
from signal_extraction.shared.schemas import SignalBundle, SignalParams

SIGNAL_KEYS = ["s1", "s2", "s3", "s4", "sum"]


@pytest.fixture()
def default_params() -> SignalParams:
    """Minimal valid SignalParams matching config/setup.json defaults."""
    return SignalParams(
        frequencies=[10.0, 50.0, 120.0, 300.0],
        amplitudes=[1.0, 0.8, 1.2, 0.6],
        phases=[0.0, 0.5, 1.0, 1.5],
        alpha=0.1,
        beta=0.1,
        noise_dist=NOISE_DIST_GAUSSIAN,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        seed=42,
    )


@pytest.fixture()
def bundle(default_params: SignalParams) -> SignalBundle:
    """Pre-generated bundle used by multiple tests."""
    return SignalGeneratorService().generate(default_params)


# ---------------------------------------------------------------------------
# SG-T03 — Shape checks
# ---------------------------------------------------------------------------


def test_all_clean_signals_have_correct_shape(bundle: SignalBundle) -> None:
    for key in SIGNAL_KEYS:
        assert bundle.clean[key].shape == (N_SAMPLES,), f"clean[{key!r}] wrong shape"


def test_all_noisy_signals_have_correct_shape(bundle: SignalBundle) -> None:
    for key in SIGNAL_KEYS:
        assert bundle.noisy[key].shape == (N_SAMPLES,), f"noisy[{key!r}] wrong shape"


# ---------------------------------------------------------------------------
# SG-T04 — Time axis
# ---------------------------------------------------------------------------


def test_time_axis_shape(bundle: SignalBundle) -> None:
    assert bundle.t.shape == (N_SAMPLES,)


def test_time_axis_start(bundle: SignalBundle) -> None:
    assert bundle.t[0] == pytest.approx(0.0)


def test_time_axis_end(bundle: SignalBundle) -> None:
    expected_last = (N_SAMPLES - 1) / SAMPLE_RATE
    assert bundle.t[-1] == pytest.approx(expected_last)


# ---------------------------------------------------------------------------
# SG-T02 — Clean sum correctness
# ---------------------------------------------------------------------------


def test_clean_sum_equals_sum_of_components(bundle: SignalBundle) -> None:
    expected = sum(bundle.clean[f"s{i}"] for i in range(1, N_SIGNALS + 1))
    np.testing.assert_allclose(bundle.clean["sum"], expected, atol=1e-10)


def test_noisy_sum_equals_sum_of_noisy_components(bundle: SignalBundle) -> None:
    expected = sum(bundle.noisy[f"s{i}"] for i in range(1, N_SIGNALS + 1))
    np.testing.assert_allclose(bundle.noisy["sum"], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# SG-T08 — Formula spot-check at t=0
# ---------------------------------------------------------------------------


def test_clean_signal_formula_at_t0(default_params: SignalParams) -> None:
    bundle = SignalGeneratorService().generate(default_params)
    for i, (amp, freq, phase) in enumerate(
        zip(
            default_params.amplitudes,
            default_params.frequencies,
            default_params.phases,
        )
    ):
        expected = amp * np.sin(phase)  # sin(2π·f·0 + φ) = sin(φ)
        key = f"s{i + 1}"
        assert bundle.clean[key][0] == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# SG-T01 — Zero noise → noisy == clean
# ---------------------------------------------------------------------------


def test_zero_noise_noisy_equals_clean(default_params: SignalParams) -> None:
    params = SignalParams(**{**default_params.__dict__, "alpha": 0.0, "beta": 0.0})
    bundle = SignalGeneratorService().generate(params)
    for i in range(1, N_SIGNALS + 1):
        key = f"s{i}"
        np.testing.assert_array_equal(bundle.noisy[key], bundle.clean[key])


# ---------------------------------------------------------------------------
# SG-T05 — Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_bundles(default_params: SignalParams) -> None:
    b1 = SignalGeneratorService().generate(default_params)
    b2 = SignalGeneratorService().generate(default_params)
    for key in SIGNAL_KEYS:
        np.testing.assert_array_equal(b1.noisy[key], b2.noisy[key])


def test_different_seeds_produce_different_noisy_signals(
    default_params: SignalParams,
) -> None:
    p2 = SignalParams(**{**default_params.__dict__, "seed": 99})
    b1 = SignalGeneratorService().generate(default_params)
    b2 = SignalGeneratorService().generate(p2)
    assert not np.array_equal(b1.noisy["s1"], b2.noisy["s1"])


# ---------------------------------------------------------------------------
# SG-T09 — Noise actually changes the signal
# ---------------------------------------------------------------------------


def test_nonzero_alpha_changes_amplitude(default_params: SignalParams) -> None:
    bundle = SignalGeneratorService().generate(default_params)
    assert not np.array_equal(bundle.clean["s1"], bundle.noisy["s1"])


# ---------------------------------------------------------------------------
# SG-T07 — Nyquist violation raises ValueError
# ---------------------------------------------------------------------------


def test_nyquist_violation_raises(default_params: SignalParams) -> None:
    params = SignalParams(**{**default_params.__dict__, "frequencies": [600.0, 50.0, 120.0, 300.0]})
    with pytest.raises(ValueError, match="Nyquist"):
        SignalGeneratorService().generate(params)


# ---------------------------------------------------------------------------
# SG-T10 — Invalid noise distribution raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_noise_dist_raises(default_params: SignalParams) -> None:
    params = SignalParams(**{**default_params.__dict__, "noise_dist": "laplace"})
    with pytest.raises(ValueError, match="noise_dist"):
        SignalGeneratorService().generate(params)


# ---------------------------------------------------------------------------
# Uniform noise distribution is also supported
# ---------------------------------------------------------------------------


def test_uniform_noise_dist_runs_without_error(default_params: SignalParams) -> None:
    params = SignalParams(**{**default_params.__dict__, "noise_dist": NOISE_DIST_UNIFORM})
    bundle = SignalGeneratorService().generate(params)
    assert bundle.noisy["s1"].shape == (N_SAMPLES,)


# ---------------------------------------------------------------------------
# Bundle carries back the params used
# ---------------------------------------------------------------------------


def test_bundle_stores_params(bundle: SignalBundle, default_params: SignalParams) -> None:
    assert bundle.params == default_params
