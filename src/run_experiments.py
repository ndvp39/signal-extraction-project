"""
CLI entry point for running experiments defined in docs/TODO.md T-074 – T-077.

Delegates entirely to ExperimentRunner — no training logic here.
Baseline training (T-070–T-073) is done via src/main.py --model all.

Usage:
    uv run python src/run_experiments.py --exp noise_sweep
    uv run python src/run_experiments.py --exp hidden_size
    uv run python src/run_experiments.py --exp n_layers
    uv run python src/run_experiments.py --exp lr_sweep
    uv run python src/run_experiments.py --exp all
"""

from __future__ import annotations

import argparse
import sys

from signal_extraction.experiments.runner import ExperimentRunner

BASE_CONFIG = "config/setup.json"
RESULTS = "results"

# T-074: 3 noise values — low / default / high, varied independently
_NOISE_ALPHA = [0.0, 0.1, 0.5]
_NOISE_BETA  = [0.0, 0.1, 0.5]
# T-075: 4 hidden sizes
_HIDDEN_SIZES = [64, 128, 256, 512]
# T-076: 4 layer counts
_N_LAYERS = [1, 2, 3, 4]
# T-077: 4 learning rates
_LR_VALUES = [0.0001, 0.001, 0.003, 0.01]


def run_noise_sweep(runner: ExperimentRunner) -> None:
    """T-074 — vary noise level (alpha/beta × 3 values), seed=42."""
    print("=== T-074: Noise Level Sweep ===")
    for alpha in _NOISE_ALPHA:
        label = f"alpha_{alpha:.1f}".replace(".", "_")
        print(f"  alpha={alpha}")
        runner.run_condition(
            {"signals": {"alpha": alpha, "beta": 0.1}},
            f"{RESULTS}/noise_sweep/alpha/{label}",
            seeds=[42],
        )
    for beta in _NOISE_BETA:
        label = f"beta_{beta:.1f}".replace(".", "_")
        print(f"  beta={beta}")
        runner.run_condition(
            {"signals": {"alpha": 0.1, "beta": beta}},
            f"{RESULTS}/noise_sweep/beta/{label}",
            seeds=[42],
        )


def run_hidden_size(runner: ExperimentRunner) -> None:
    """T-075 — vary hidden_size, seed=42."""
    print("=== T-075: Hidden-Size Sensitivity ===")
    for h in _HIDDEN_SIZES:
        print(f"  hidden_size={h}")
        runner.run_condition(
            {"models": {"hidden_size": h}},
            f"{RESULTS}/hidden_size/h{h}",
            seeds=[42],
        )


def run_n_layers(runner: ExperimentRunner) -> None:
    """T-076 — vary n_layers, seed=42."""
    print("=== T-076: Depth Sensitivity ===")
    for n in _N_LAYERS:
        print(f"  n_layers={n}")
        runner.run_condition(
            {"models": {"n_layers": n}},
            f"{RESULTS}/n_layers/L{n}",
            seeds=[42],
        )


def run_lr_sweep(runner: ExperimentRunner) -> None:
    """T-077 — vary learning rate, seed=42."""
    print("=== T-077: Learning-Rate Sweep ===")
    for lr in _LR_VALUES:
        label = f"lr_{lr:.4f}".replace(".", "_")
        print(f"  lr={lr}")
        runner.run_condition(
            {"training": {"learning_rate": lr}},
            f"{RESULTS}/lr_sweep/{label}",
            seeds=[42],
        )


_EXPERIMENTS: dict = {
    "noise_sweep": run_noise_sweep,
    "hidden_size": run_hidden_size,
    "n_layers": run_n_layers,
    "lr_sweep": run_lr_sweep,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run signal extraction experiments (see docs/TODO.md T-074–T-077)."
    )
    parser.add_argument(
        "--exp",
        choices=[*_EXPERIMENTS.keys(), "all"],
        default="noise_sweep",
        help="Experiment to run. Use 'all' to run every experiment.",
    )
    parser.add_argument("--config", default=BASE_CONFIG, help="Path to setup.json.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runner = ExperimentRunner(args.config)
    to_run = list(_EXPERIMENTS.keys()) if args.exp == "all" else [args.exp]
    for name in to_run:
        _EXPERIMENTS[name](runner)
    print("\nAll experiments complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
