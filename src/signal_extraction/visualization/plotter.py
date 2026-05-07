"""
Plotter — generates all result figures defined in docs/EXPERIMENT_PLAN.md.

Each public method corresponds to one asset file in assets/.
All plots are saved as PNG; caller passes the output path explicitly.
Requirements source: docs/TODO.md T-080 – T-085.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_MODELS = ["fc", "rnn", "lstm"]
_FREQ_LABELS = ["10 Hz", "50 Hz", "120 Hz", "300 Hz"]


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_training_curves(results_dir: str, out_path: str) -> None:
    """
    Loss vs. epoch for FC / RNN / LSTM (seed=42, one curve each).

    Saves to out_path (PNG).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, model in zip(axes, _MODELS):  # noqa: B905
        m_path = os.path.join(results_dir, "seed_42", model, "metrics.json")
        data = _load_json(m_path)
        epochs = list(range(len(data["train_losses"])))
        ax.plot(epochs, data["train_losses"], label="Train")
        ax.plot(epochs, data["val_losses"], label="Val")
        ax.axvline(data["best_epoch"], color="red", linestyle="--", alpha=0.7,
                   label=f"Best epoch {data['best_epoch']}")
        ax.set_title(model.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.set_yscale("log")
    fig.suptitle("Training Curves (seed=42)", fontsize=13)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_mse_comparison(summary_path: str, out_path: str) -> None:
    """
    Grouped bar chart: MSE per frequency × model (from a summary.json).

    Saves to out_path (PNG).
    """
    data = _load_json(summary_path)["models"]
    x = np.arange(len(_FREQ_LABELS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(_MODELS):
        freq_data = data[model]["mse_per_freq"]
        means = [freq_data[str(k)]["mean"] for k in range(4)]
        stds = [freq_data[str(k)]["std"] for k in range(4)]
        ax.bar(x + i * width, means, width, yerr=stds, label=model.upper(),
               capsize=4, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(_FREQ_LABELS)
    ax.set_ylabel("MSE (mean ± std, 3 seeds)")
    ax.set_title("Per-Frequency MSE by Architecture")
    ax.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_noise_heatmap(noise_sweep_dir: str, out_path: str) -> None:
    """
    Heatmap of MSE_overall (model × noise level).

    noise_sweep_dir contains one sub-folder per condition (e.g. noise_a0_0_b0_0/).
    Saves to out_path (PNG).
    """
    conditions = sorted(os.listdir(noise_sweep_dir))
    matrix = np.zeros((len(_MODELS), len(conditions)))
    for j, cond in enumerate(conditions):
        summary = _load_json(os.path.join(noise_sweep_dir, cond, "summary.json"))
        for i, model in enumerate(_MODELS):
            matrix[i, j] = summary["models"][model]["mse_overall"]["mean"]
    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.5), 4))
    sns.heatmap(matrix, ax=ax, xticklabels=conditions,
                yticklabels=[m.upper() for m in _MODELS],
                annot=True, fmt=".2e", cmap="YlOrRd")
    ax.set_title("MSE Heatmap — noise level sweep")
    ax.set_xlabel("noise level (alpha=beta)")
    ax.set_ylabel("Model")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_sensitivity(
    sweep_dir: str,
    param_name: str,
    param_values: list[Any],
    out_path: str,
) -> None:
    """
    MSE vs. hyperparameter line chart (one line per model, error bars from 3 seeds).

    sweep_dir: directory containing one sub-folder per condition.
    Saves to out_path (PNG).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    conditions = sorted(os.listdir(sweep_dir))
    if len(conditions) != len(param_values):
        param_values = list(range(len(conditions)))
    for model in _MODELS:
        means, stds = [], []
        for cond in conditions:
            summary = _load_json(os.path.join(sweep_dir, cond, "summary.json"))
            means.append(summary["models"][model]["mse_overall"]["mean"])
            stds.append(summary["models"][model]["mse_overall"]["std"])
        ax.errorbar(param_values, means, yerr=stds, marker="o", label=model.upper(),
                    capsize=4)
    ax.set_xlabel(param_name)
    ax.set_ylabel("MSE (mean ± std, 3 seeds)")
    ax.set_title(f"Sensitivity: {param_name}")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
