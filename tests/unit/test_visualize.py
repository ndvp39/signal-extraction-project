"""
Tests for src/visualize.py CLI.

Uses monkeypatching so no actual plots are generated.
Requirements source: docs/TODO.md T-080 – T-085.
"""

from __future__ import annotations

import visualize as viz_mod
from visualize import parse_args


def test_parse_args_defaults():
    args = parse_args([])
    assert args.results == "results"
    assert args.assets == "assets"
    assert args.config == "config/setup.json"


def test_parse_args_custom():
    args = parse_args(["--results", "r/", "--assets", "a/", "--config", "c.json"])
    assert args.results == "r/"
    assert args.assets == "a/"
    assert args.config == "c.json"


def test_main_calls_all_plotters(monkeypatch, tmp_path):
    """Verify main() calls every plot function when all mocked."""
    called = {}

    def fake_plot_training_curves(results_dir, out_path):
        called["training_curves"] = True

    def fake_plot_mse_comparison(summary_path, out_path):
        called["mse_comparison"] = True

    def fake_plot_noise_heatmap(noise_sweep_dir, out_path):
        called["noise_heatmap"] = True

    def fake_plot_sensitivity(sweep_dir, param_name, param_values, out_path):
        called.setdefault("sensitivity", []).append(param_name)

    def fake_plot_signal_overview(bundle, out_path, **kwargs):
        called["signal_overview"] = True

    def fake_plot_signal_examples(**kwargs):
        called["signal_examples"] = True

    monkeypatch.setattr("visualize.plot_training_curves", fake_plot_training_curves)
    monkeypatch.setattr("visualize.plot_mse_comparison", fake_plot_mse_comparison)
    monkeypatch.setattr("visualize.plot_noise_heatmap", fake_plot_noise_heatmap)
    monkeypatch.setattr("visualize.plot_sensitivity", fake_plot_sensitivity)
    monkeypatch.setattr("visualize.plot_signal_overview", fake_plot_signal_overview)
    monkeypatch.setattr("visualize.plot_signal_examples", fake_plot_signal_examples)

    viz_mod.main([
        "--results", str(tmp_path / "results"),
        "--assets", str(tmp_path / "assets"),
        "--config", "config/test_setup.json",
    ])

    assert "training_curves" in called
    assert "mse_comparison" in called
    assert called.get("noise_heatmap") is True
    assert "hidden_size" in called.get("sensitivity", [])
    assert "n_layers" in called.get("sensitivity", [])
    assert "learning_rate" in called.get("sensitivity", [])
    assert "signal_overview" in called
    assert "signal_examples" in called
