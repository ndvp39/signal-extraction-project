"""
Tests for visualization/plotter.py.

Creates minimal mock result files in tmp_path so plots can be generated
without requiring full experiment runs.
Requirements source: docs/TODO.md T-080 – T-085.
"""

from __future__ import annotations

import json
import os

from signal_extraction.visualization.plotter import (
    plot_mse_comparison,
    plot_noise_heatmap,
    plot_sensitivity,
    plot_training_curves,
)

_MODELS = ["fc", "rnn", "lstm"]


def _make_per_freq(v: float) -> dict:
    return {str(k): {"mean": v, "std": v * 0.1} for k in range(4)}


def _write_summary(path: str, mse: float = 1e-3) -> None:
    data = {
        "models": {
            m: {"mse_overall": {"mean": mse, "std": mse * 0.1},
                "mse_per_freq": _make_per_freq(mse)}
            for m in _MODELS
        },
        "seeds": [42, 123, 777],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _write_metrics(path: str, n_epochs: int = 5, best_epoch: int = 3) -> None:
    losses = [0.1 * (1 - i * 0.15) for i in range(n_epochs)]
    data = {
        "mse_overall": 1e-3,
        "mse_per_freq": {str(k): 1e-3 for k in range(4)},
        "best_epoch": best_epoch,
        "n_epochs_trained": n_epochs,
        "train_losses": losses,
        "val_losses": [v * 1.05 for v in losses],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# plot_training_curves
# ---------------------------------------------------------------------------


def test_plot_training_curves_creates_png(tmp_path):
    baseline_dir = str(tmp_path / "baseline")
    for model in _MODELS:
        _write_metrics(
            os.path.join(baseline_dir, "seed_42", model, "metrics.json")
        )
    out = str(tmp_path / "training_curves.png")
    plot_training_curves(results_dir=baseline_dir, out_path=out)
    assert os.path.exists(out)


# ---------------------------------------------------------------------------
# plot_mse_comparison
# ---------------------------------------------------------------------------


def test_plot_mse_comparison_creates_png(tmp_path):
    summary_path = str(tmp_path / "summary.json")
    _write_summary(summary_path)
    out = str(tmp_path / "mse_comparison.png")
    plot_mse_comparison(summary_path=summary_path, out_path=out)
    assert os.path.exists(out)


# ---------------------------------------------------------------------------
# plot_noise_heatmap
# ---------------------------------------------------------------------------


def _write_noise_sweep(base_dir: str) -> None:
    for label in ["noise_a0_0_b0_0", "noise_a0_1_b0_1", "noise_a0_5_b0_5"]:
        cond_dir = os.path.join(base_dir, label)
        _write_summary(os.path.join(cond_dir, "summary.json"))


def test_plot_noise_heatmap_creates_png(tmp_path):
    sweep_dir = str(tmp_path / "noise_sweep")
    _write_noise_sweep(sweep_dir)
    out = str(tmp_path / "noise_heatmap.png")
    plot_noise_heatmap(noise_sweep_dir=sweep_dir, out_path=out)
    assert os.path.exists(out)


# ---------------------------------------------------------------------------
# plot_sensitivity
# ---------------------------------------------------------------------------


def _write_sensitivity_sweep(base_dir: str, labels: list[str]) -> None:
    for label in labels:
        cond_dir = os.path.join(base_dir, label)
        _write_summary(os.path.join(cond_dir, "summary.json"))


def test_plot_sensitivity_creates_png(tmp_path):
    sweep_dir = str(tmp_path / "hidden_size")
    _write_sensitivity_sweep(sweep_dir, ["h16", "h32", "h64"])
    out = str(tmp_path / "sensitivity.png")
    plot_sensitivity(
        sweep_dir=sweep_dir,
        param_name="hidden_size",
        param_values=[16, 32, 64],
        out_path=out,
    )
    assert os.path.exists(out)


def test_plot_sensitivity_mismatched_param_values_does_not_raise(tmp_path):
    """If param_values length doesn't match conditions, uses indices."""
    sweep_dir = str(tmp_path / "sweep")
    _write_sensitivity_sweep(sweep_dir, ["a", "b"])
    out = str(tmp_path / "sensitivity.png")
    plot_sensitivity(
        sweep_dir=sweep_dir,
        param_name="x",
        param_values=[1, 2, 3, 4],  # wrong count — should fall back to indices
        out_path=out,
    )
    assert os.path.exists(out)
