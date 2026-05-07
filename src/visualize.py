"""
CLI entry point for generating all result visualizations.

Reads experiment results from results/ and writes PNGs to assets/.
Run after all experiments in run_experiments.py have completed.

Usage:
    uv run python src/visualize.py
    uv run python src/visualize.py --results results/ --assets assets/
"""

from __future__ import annotations

import argparse
import sys

from signal_extraction.sdk.sdk import SignalExtractionSDK
from signal_extraction.visualization.plotter import (
    plot_mse_comparison,
    plot_noise_heatmap,
    plot_sensitivity,
    plot_training_curves,
)
from signal_extraction.visualization.signal_plots import (
    plot_signal_examples,
    plot_signal_overview,
)

_BASE_CONFIG = "config/setup.json"

_HIDDEN_SIZES = [64, 128, 256, 512]
_N_LAYERS = [1, 2, 3, 4]
_LR_VALUES = [0.0001, 0.001, 0.003, 0.01]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate result visualizations.")
    parser.add_argument("--results", default="results", help="Results root directory.")
    parser.add_argument("--assets", default="assets", help="Output assets directory.")
    parser.add_argument("--config", default=_BASE_CONFIG, help="Path to setup.json.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    r = args.results
    a = args.assets

    print("Generating training curves...")
    plot_training_curves(
        results_dir=f"{r}/baseline",
        out_path=f"{a}/training_curves.png",
    )

    print("Generating MSE comparison bar chart...")
    plot_mse_comparison(
        summary_path=f"{r}/baseline/summary.json",
        out_path=f"{a}/mse_comparison.png",
    )

    print("Generating noise heatmap...")
    plot_noise_heatmap(
        noise_sweep_dir=f"{r}/noise_sweep/alpha",
        out_path=f"{a}/noise_heatmap.png",
    )

    print("Generating hidden-size sensitivity curve...")
    plot_sensitivity(
        sweep_dir=f"{r}/hidden_size",
        param_name="hidden_size",
        param_values=_HIDDEN_SIZES,
        out_path=f"{a}/sensitivity_hidden_size.png",
    )

    print("Generating n_layers sensitivity curve...")
    plot_sensitivity(
        sweep_dir=f"{r}/n_layers",
        param_name="n_layers",
        param_values=_N_LAYERS,
        out_path=f"{a}/sensitivity_n_layers.png",
    )

    print("Generating learning-rate sensitivity curve...")
    plot_sensitivity(
        sweep_dir=f"{r}/lr_sweep",
        param_name="learning_rate",
        param_values=_LR_VALUES,
        out_path=f"{a}/sensitivity_lr.png",
    )

    print("Generating signal overview...")
    sdk = SignalExtractionSDK(args.config)
    bundle = sdk.generate_signals()
    plot_signal_overview(bundle, out_path=f"{a}/signal_overview.png")

    print("Generating signal examples (predicted vs. clean)...")
    from signal_extraction.services.dataset_builder import DatasetBuilderService
    builder = DatasetBuilderService(seed=42)
    samples = builder.build(bundle, n_samples=1)
    sample = samples[0]
    cfg_hidden = sdk._cfg.models["hidden_size"]
    cfg_n_layers = sdk._cfg.models["n_layers"]
    plot_signal_examples(
        checkpoint_dir=f"{r}/baseline/seed_42",
        sample_x=sample.x.tolist(),
        sample_y=sample.y.tolist(),
        hidden_size=cfg_hidden,
        out_path=f"{a}/signal_examples.png",
        n_layers=cfg_n_layers,
    )

    print(f"All plots saved to {a}/")


if __name__ == "__main__":
    main(sys.argv[1:])
