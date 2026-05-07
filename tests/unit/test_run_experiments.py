"""
Tests for src/run_experiments.py CLI.

Monkeypatches ExperimentRunner.run_condition so no actual training occurs.
Requirements source: docs/TODO.md T-074 – T-077.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import run_experiments as re_mod
from run_experiments import (
    _EXPERIMENTS,
    parse_args,
    run_hidden_size,
    run_lr_sweep,
    run_n_layers,
    run_noise_sweep,
)

TEST_CONFIG = "config/test_setup.json"


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_default():
    args = parse_args([])
    assert args.exp == "noise_sweep"
    assert args.config == "config/setup.json"


def test_parse_args_exp_noise_sweep():
    args = parse_args(["--exp", "noise_sweep"])
    assert args.exp == "noise_sweep"


def test_parse_args_exp_all():
    args = parse_args(["--exp", "all"])
    assert args.exp == "all"


def test_parse_args_custom_config():
    args = parse_args(["--config", TEST_CONFIG])
    assert args.config == TEST_CONFIG


def test_parse_args_invalid_exp():
    with pytest.raises(SystemExit):
        parse_args(["--exp", "nonexistent"])


# ---------------------------------------------------------------------------
# Experiment functions (mocked runner)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_runner():
    runner = MagicMock()
    runner.run_condition.return_value = {}
    return runner


def test_run_noise_sweep_calls_run_condition_6_times(mock_runner):
    run_noise_sweep(mock_runner)
    assert mock_runner.run_condition.call_count == 6  # 3 alpha + 3 beta


def test_run_hidden_size_calls_run_condition_4_times(mock_runner):
    run_hidden_size(mock_runner)
    assert mock_runner.run_condition.call_count == 4


def test_run_n_layers_calls_run_condition_4_times(mock_runner):
    run_n_layers(mock_runner)
    assert mock_runner.run_condition.call_count == 4


def test_run_lr_sweep_uses_single_seed(mock_runner):
    run_lr_sweep(mock_runner)
    for call in mock_runner.run_condition.call_args_list:
        seeds = call[1].get("seeds") or call.kwargs.get("seeds")
        if seeds is not None:
            assert seeds == [42]


def test_run_noise_sweep_uses_single_seed(mock_runner):
    run_noise_sweep(mock_runner)
    for call in mock_runner.run_condition.call_args_list:
        seeds = call[1].get("seeds") or call.kwargs.get("seeds")
        if seeds is not None:
            assert seeds == [42]


# ---------------------------------------------------------------------------
# _EXPERIMENTS registry
# ---------------------------------------------------------------------------


def test_experiments_registry_contains_all_keys():
    expected = {"noise_sweep", "hidden_size", "n_layers", "lr_sweep"}
    assert set(_EXPERIMENTS.keys()) == expected


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_noise_sweep_with_mock(monkeypatch):
    calls = []

    def fake_run_condition(self, override, output_dir, **kwargs):
        calls.append(output_dir)
        return {}

    monkeypatch.setattr(
        "signal_extraction.experiments.runner.ExperimentRunner.run_condition",
        fake_run_condition,
    )
    re_mod.main(["--exp", "noise_sweep", "--config", TEST_CONFIG])
    assert any("noise_sweep" in c for c in calls)


def test_main_all_runs_all_experiments(monkeypatch):
    call_count = []

    def fake_run_condition(self, override, output_dir, **kwargs):
        call_count.append(1)
        return {}

    monkeypatch.setattr(
        "signal_extraction.experiments.runner.ExperimentRunner.run_condition",
        fake_run_condition,
    )
    re_mod.main(["--exp", "all", "--config", TEST_CONFIG])
    # noise(6) + hidden(4) + layers(4) + lr(4) = 18
    assert sum(call_count) == 18
