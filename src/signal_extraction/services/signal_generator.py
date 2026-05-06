"""
SignalGeneratorService — generates all 10 signal vectors from SignalParams.

Implements the formula A·sin(2πft + φ) with independent Gaussian or uniform
noise applied to amplitude and phase per sinusoid, as specified in
ASSIGNMENT.txt sections 4 and 5.
"""

from __future__ import annotations

import numpy as np

from signal_extraction.constants import (
    N_SIGNALS,
    NOISE_DIST_GAUSSIAN,
    NOISE_DIST_UNIFORM,
    SUPPORTED_NOISE_DISTS,
)
from signal_extraction.shared.schemas import SignalBundle, SignalParams


class SignalGeneratorService:
    """
    Generates clean and noisy sinusoidal signals from a SignalParams config.

    Single responsibility: signal mathematics only. Does not build datasets,
    save files, or interact with the training pipeline.
    """

    def generate(self, params: SignalParams) -> SignalBundle:
        """
        Generate all 10 signal vectors (5 clean + 5 noisy).

        Args:
            params: Full signal configuration including frequencies, amplitudes,
                    phases, noise coefficients, and random seed.

        Returns:
            SignalBundle with clean and noisy dicts, time axis, and params.

        Raises:
            ValueError: If any frequency violates the Nyquist limit, if list
                        lengths mismatch N_SIGNALS, if amplitudes are non-positive,
                        or if noise_dist is not supported.
        """
        self._validate(params)
        rng = np.random.default_rng(params.seed)
        t = self._make_time_axis(params)
        clean = self._make_clean_signals(params, t)
        noisy = self._make_noisy_signals(params, t, rng)
        clean["sum"] = sum(clean[f"s{i}"] for i in range(1, N_SIGNALS + 1))
        noisy["sum"] = sum(noisy[f"s{i}"] for i in range(1, N_SIGNALS + 1))
        return SignalBundle(clean=clean, noisy=noisy, t=t, params=params)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(params: SignalParams) -> None:
        """Validate all parameters before any computation."""
        nyquist = params.sample_rate / 2.0
        for f in params.frequencies:
            if f >= nyquist:
                raise ValueError(
                    f"Frequency {f} Hz violates Nyquist limit "
                    f"({nyquist} Hz for sample_rate={params.sample_rate} Hz)."
                )
        for lst, name in [
            (params.frequencies, "frequencies"),
            (params.amplitudes, "amplitudes"),
            (params.phases, "phases"),
        ]:
            if len(lst) != N_SIGNALS:
                raise ValueError(
                    f"'{name}' must have length {N_SIGNALS}, got {len(lst)}."
                )
        for amp in params.amplitudes:
            if amp <= 0:
                raise ValueError(f"All amplitudes must be positive, got {amp}.")
        if params.noise_dist not in SUPPORTED_NOISE_DISTS:
            raise ValueError(
                f"noise_dist={params.noise_dist!r} is not supported. "
                f"Choose from {SUPPORTED_NOISE_DISTS}."
            )

    @staticmethod
    def _make_time_axis(params: SignalParams) -> np.ndarray:
        """Build discrete time axis t[n] = n / sample_rate."""
        n_samples = int(params.sample_rate * params.duration)
        return np.arange(n_samples) / params.sample_rate

    @staticmethod
    def _make_sinusoid(amp: float, freq: float, phase: float, t: np.ndarray) -> np.ndarray:
        """Compute A·sin(2πft + φ) over the full time axis."""
        return amp * np.sin(2 * np.pi * freq * t + phase)

    def _draw_noise(self, rng: np.random.Generator, dist: str) -> float:
        """Draw a single scalar noise sample from the specified distribution."""
        if dist == NOISE_DIST_GAUSSIAN:
            return float(rng.standard_normal())
        if dist == NOISE_DIST_UNIFORM:
            return float(rng.uniform(-1.0, 1.0))
        # Unreachable after _validate, but kept for explicitness
        raise ValueError(f"Unsupported noise_dist: {dist!r}")

    def _make_clean_signals(
        self, params: SignalParams, t: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Generate the 4 clean sinusoids (no noise)."""
        return {
            f"s{i + 1}": self._make_sinusoid(
                params.amplitudes[i], params.frequencies[i], params.phases[i], t
            )
            for i in range(N_SIGNALS)
        }

    def _make_noisy_signals(
        self, params: SignalParams, t: np.ndarray, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Generate 4 noisy sinusoids.

        Noise is applied to amplitude and phase independently per sinusoid:
            A_noisy = A + alpha * eps_A
            phi_noisy = phi + beta * eps_phi
        where eps_A and eps_phi are drawn once per sinusoid (scalar), not per sample.
        This models slowly-varying channel disturbances (fading + phase jitter).
        """
        noisy: dict[str, np.ndarray] = {}
        for i in range(N_SIGNALS):
            noisy_amp = params.amplitudes[i] + params.alpha * self._draw_noise(rng, params.noise_dist)
            noisy_phase = params.phases[i] + params.beta * self._draw_noise(rng, params.noise_dist)
            noisy[f"s{i + 1}"] = self._make_sinusoid(noisy_amp, params.frequencies[i], noisy_phase, t)
        return noisy
