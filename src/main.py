"""
CLI entry point for the signal extraction project.

Argument parsing only — all business logic is delegated to SignalExtractionSDK.
No model code, signal generation, or training logic belongs here.

Usage:
    uv run python src/main.py --model fc
    uv run python src/main.py --model all
    uv run python src/main.py --model rnn --config config/setup.json
"""

from __future__ import annotations

import argparse
import sys

from signal_extraction.sdk.sdk import SignalExtractionSDK

MODEL_CHOICES = ["fc", "rnn", "lstm", "all"]
DEFAULT_CONFIG = "config/setup.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Signal extraction: train and compare FC, RNN, and LSTM models."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="all",
        help="Model type to train. Use 'all' to train and compare all three. (default: all)",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to setup.json configuration file. (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the full training and evaluation pipeline via the SDK."""
    args = parse_args(argv)
    sdk = SignalExtractionSDK(args.config)

    model_types = ["fc", "rnn", "lstm"] if args.model == "all" else [args.model]

    bundle = sdk.generate_signals()
    train_loader, val_loader, test_loader = sdk.build_dataset(bundle)

    results = {}
    for model_type in model_types:
        print(f"\n--- Training {model_type.upper()} ---")
        model, train_result = sdk.train_model(model_type, train_loader, val_loader)
        print(f"Best epoch: {train_result.best_epoch} | "
              f"Val loss: {train_result.val_losses[train_result.best_epoch]:.6f}")
        eval_result = sdk.evaluate_model(model, test_loader)
        print(f"Test MSE (overall): {eval_result.mse_overall:.6f}")
        for idx, mse in eval_result.mse_per_freq.items():
            print(f"  Freq index {idx}: MSE = {mse:.6f}")
        results[model_type] = eval_result

    sdk.save_results(results)
    print("\nResults saved.")


if __name__ == "__main__":
    main(sys.argv[1:])
