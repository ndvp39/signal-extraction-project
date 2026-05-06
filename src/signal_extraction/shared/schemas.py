"""
Dataclass schemas shared across all services and the SDK.

These are pure data containers with no business logic. They define the
contracts between services (signal generator → dataset builder → trainer →
evaluator) and are the single source of truth for data shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SignalParams:
    """
    All parameters required to generate and reproduce a signal bundle.

    Physical constants (SAMPLE_RATE, DURATION) are stored here so that any
    saved bundle is fully self-describing without referring back to config.
    """

    frequencies: list[float]   # Hz — one per sinusoid, length == N_SIGNALS
    amplitudes: list[float]    # dimensionless — one per sinusoid
    phases: list[float]        # radians — one per sinusoid
    alpha: float               # amplitude noise strength coefficient (≥ 0)
    beta: float                # phase noise strength coefficient (≥ 0)
    noise_dist: str            # "gaussian" | "uniform"
    sample_rate: int           # Hz (fixed: 1000)
    duration: float            # seconds (fixed: 10.0)
    seed: int                  # random seed for reproducibility


@dataclass
class SignalBundle:
    """
    All 10 signal vectors produced by SignalGeneratorService.

    clean["sum"] == sum of clean["s1".."s4"] (exact).
    noisy["sum"] == sum of noisy["s1".."s4"] (exact).
    Keys for individual sinusoids: "s1", "s2", "s3", "s4".
    """

    clean: dict[str, np.ndarray]   # shape (N_SAMPLES,) per key
    noisy: dict[str, np.ndarray]   # shape (N_SAMPLES,) per key
    t: np.ndarray                  # time axis, shape (N_SAMPLES,)
    params: SignalParams


@dataclass
class Sample:
    """
    One training/validation/test sample.

    x = [selector C (one-hot, 4) | noisy_sum_window (10)] → shape (14,)
    y = clean target window for the selected sinusoid → shape (10,)
    """

    x: np.ndarray   # float64, shape (INPUT_SIZE,)  = (14,)
    y: np.ndarray   # float64, shape (OUTPUT_SIZE,) = (10,)


@dataclass
class TrainResult:
    """
    Output of TrainerService.train() — history and checkpoint location.
    """

    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    model_path: str


@dataclass
class EvalResult:
    """
    Output of EvaluatorService.evaluate() — MSE metrics and raw predictions.

    mse_per_freq keys are frequency indices 0–3 corresponding to the order
    in SignalParams.frequencies.
    """

    mse_overall: float
    mse_per_freq: dict[int, float] = field(default_factory=dict)
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    targets: np.ndarray = field(default_factory=lambda: np.array([]))
