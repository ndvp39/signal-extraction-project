"""
Tests for ExperimentRunner and helpers in experiments/runner.py.

Uses config/test_setup.json (n_samples=400, epochs=3) for fast runs.
Requirements source: docs/EXPERIMENT_PLAN.md; docs/TODO.md T-070 – T-077.
"""

from __future__ import annotations

import os

import numpy as np

from signal_extraction.experiments.runner import (
    ExperimentRunner,
    _deep_merge,
    _summarize,
)

TEST_CONFIG = "config/test_setup.json"


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_flat_key():
    base = {"a": 1, "b": 2}
    _deep_merge(base, {"b": 99})
    assert base == {"a": 1, "b": 99}


def test_deep_merge_nested_preserves_untouched():
    base = {"signals": {"alpha": 0.1, "beta": 0.1}, "training": {"epochs": 3}}
    _deep_merge(base, {"signals": {"alpha": 0.5}})
    assert base["signals"]["alpha"] == 0.5
    assert base["signals"]["beta"] == 0.1


def test_deep_merge_adds_new_key():
    base = {"a": 1}
    _deep_merge(base, {"b": 2})
    assert base["b"] == 2


# ---------------------------------------------------------------------------
# _summarize
# ---------------------------------------------------------------------------


def _make_results(mse_vals: list[float]) -> list[dict]:
    return [
        {"mse_overall": v, "mse_per_freq": {"0": v * 0.5, "1": v * 1.5}}
        for v in mse_vals
    ]


def test_summarize_mean_and_std():
    results = _make_results([0.1, 0.3])
    summary = _summarize(results)
    assert abs(summary["mse_overall"]["mean"] - 0.2) < 1e-6
    assert abs(summary["mse_overall"]["std"] - float(np.std([0.1, 0.3]))) < 1e-6


def test_summarize_per_freq_keys_present():
    summary = _summarize(_make_results([0.2, 0.4]))
    assert "0" in summary["mse_per_freq"]
    assert "1" in summary["mse_per_freq"]


# ---------------------------------------------------------------------------
# ExperimentRunner.__init__
# ---------------------------------------------------------------------------


def test_runner_init_loads_base_config():
    runner = ExperimentRunner(TEST_CONFIG)
    assert isinstance(runner._base_config, dict)
    assert "signals" in runner._base_config


# ---------------------------------------------------------------------------
# ExperimentRunner.run_single
# ---------------------------------------------------------------------------


def test_run_single_writes_metrics_and_checkpoint(tmp_path):
    runner = ExperimentRunner(TEST_CONFIG)
    seed_dir = str(tmp_path / "seed_42")
    result = runner.run_single({}, "fc", 42, seed_dir)

    metrics_path = os.path.join(seed_dir, "fc", "metrics.json")
    checkpoint = tmp_path / "seed_42" / "fc" / "best_model.pt"

    assert os.path.exists(metrics_path), "metrics.json missing"
    assert checkpoint.exists(), "best_model.pt missing"
    assert result["mse_overall"] > 0
    assert result["mse_overall"] < float("inf")
    assert "mse_per_freq" in result
    assert "best_epoch" in result


def test_run_single_config_override_applied(tmp_path):
    runner = ExperimentRunner(TEST_CONFIG)
    seed_dir = str(tmp_path / "seed_42")
    result = runner.run_single({"models": {"hidden_size": 16}}, "fc", 42, seed_dir)
    assert result["mse_overall"] > 0


# ---------------------------------------------------------------------------
# ExperimentRunner.run_condition
# ---------------------------------------------------------------------------


def test_run_condition_creates_summary(tmp_path):
    runner = ExperimentRunner(TEST_CONFIG)
    out_dir = str(tmp_path / "cond")
    runner.run_condition({}, out_dir, models=["fc"], seeds=[42])
    summary_path = os.path.join(out_dir, "summary.json")
    assert os.path.exists(summary_path)


def test_run_condition_summary_structure(tmp_path):
    runner = ExperimentRunner(TEST_CONFIG)
    out_dir = str(tmp_path / "cond")
    summary = runner.run_condition({}, out_dir, models=["fc"], seeds=[42])
    fc = summary["models"]["fc"]
    assert "mean" in fc["mse_overall"]
    assert "std" in fc["mse_overall"]
    assert "mse_per_freq" in fc


def test_run_condition_two_seeds(tmp_path):
    runner = ExperimentRunner(TEST_CONFIG)
    out_dir = str(tmp_path / "cond")
    summary = runner.run_condition({}, out_dir, models=["fc"], seeds=[42, 123])
    assert summary["seeds"] == [42, 123]
    assert summary["models"]["fc"]["mse_overall"]["std"] >= 0
