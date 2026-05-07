"""
ExperimentRunner — multi-seed, multi-config experiment orchestration.

Writes per-run metrics.json and per-condition summary.json (mean ± std).
Checkpoints are saved by TrainerService via the SDK's results_dir config key.

Requirements source: docs/EXPERIMENT_PLAN.md EXP-01 through EXP-07.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np

from signal_extraction.sdk.sdk import SignalExtractionSDK


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in-place."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def _summarize(results: list[dict]) -> dict:
    """Compute mean ± std for mse_overall and mse_per_freq over a list of runs."""
    freq_keys = list(results[0]["mse_per_freq"].keys())
    mse_vals = [r["mse_overall"] for r in results]
    summary: dict = {
        "mse_overall": {
            "mean": float(np.mean(mse_vals)),
            "std": float(np.std(mse_vals)),
        },
        "mse_per_freq": {},
    }
    for k in freq_keys:
        vals = [r["mse_per_freq"][k] for r in results]
        summary["mse_per_freq"][str(k)] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    return summary


class ExperimentRunner:
    """
    Orchestrates training runs across models, seeds, and config overrides.

    Usage:
        runner = ExperimentRunner("config/setup.json")
        runner.run_condition({}, "results/baseline")
    """

    SEEDS: list[int] = [42, 123, 777]
    MODELS: list[str] = ["fc", "rnn", "lstm"]

    def __init__(self, base_config_path: str) -> None:
        self._base_config_path = base_config_path
        with open(base_config_path) as f:
            self._base_config: dict = json.load(f)

    def run_single(
        self,
        config_override: dict,
        model_type: str,
        seed: int,
        seed_dir: str,
    ) -> dict:
        """
        Train and evaluate one (model_type, seed) pair.

        The SDK saves the checkpoint to {seed_dir}/{model_type}/best_model.pt.
        This method writes {seed_dir}/{model_type}/metrics.json and returns it.
        """
        config = copy.deepcopy(self._base_config)
        _deep_merge(config, config_override)
        config["signals"]["seed"] = seed
        config["paths"]["results_dir"] = seed_dir.rstrip("/") + "/"

        os.makedirs(seed_dir, exist_ok=True)
        tmp_path = os.path.join(seed_dir, f"_config_{model_type}.json")
        with open(tmp_path, "w") as f:
            json.dump(config, f)

        sdk = SignalExtractionSDK(tmp_path)
        bundle = sdk.generate_signals()
        train_loader, val_loader, test_loader = sdk.build_dataset(bundle)
        model, train_result = sdk.train_model(model_type, train_loader, val_loader)
        eval_result = sdk.evaluate_model(model, test_loader)

        model_dir = os.path.join(seed_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        metrics = {
            "mse_overall": float(eval_result.mse_overall),
            "mse_per_freq": {
                str(k): float(v) for k, v in eval_result.mse_per_freq.items()
            },
            "best_epoch": train_result.best_epoch,
            "n_epochs_trained": len(train_result.train_losses),
            "train_losses": [float(x) for x in train_result.train_losses],
            "val_losses": [float(x) for x in train_result.val_losses],
        }
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    def run_condition(
        self,
        config_override: dict,
        output_dir: str,
        models: list[str] | None = None,
        seeds: list[int] | None = None,
    ) -> dict:
        """
        Run all (model, seed) combinations for one experimental condition.

        Writes summary.json with mean ± std to output_dir.
        Returns the summary dict.
        """
        models = models or self.MODELS
        seeds = seeds or self.SEEDS
        condition_results: dict = {}
        for model_type in models:
            runs: list[dict] = []
            for seed in seeds:
                seed_dir = os.path.join(output_dir, f"seed_{seed}")
                print(f"    [{model_type.upper()}] seed={seed} -> {seed_dir}")
                runs.append(self.run_single(config_override, model_type, seed, seed_dir))
            condition_results[model_type] = _summarize(runs)
        summary = {"models": condition_results, "seeds": seeds}
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary
