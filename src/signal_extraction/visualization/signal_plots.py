"""
Signal-specific plots for T-082 (signal examples) and T-083 (signal overview).

signal_overview — noisy sum vs. 4 clean sinusoids from a SignalBundle.
signal_examples — predicted vs. clean window for one sample per model.

Requirements source: docs/TODO.md T-082, T-083.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from signal_extraction.models.fc_model import FCModel
from signal_extraction.models.lstm_model import LSTMModel
from signal_extraction.models.rnn_model import RNNModel
from signal_extraction.shared.schemas import SignalBundle

_MODEL_CLASSES = {"fc": FCModel, "rnn": RNNModel, "lstm": LSTMModel}
_FREQ_LABELS = ["10 Hz", "50 Hz", "120 Hz", "300 Hz"]
_WINDOW_SIZE = 10


def plot_signal_overview(bundle: SignalBundle, out_path: str, n_points: int = 500) -> None:
    """
    Plot the noisy composite sum vs. the 4 clean sinusoids (first n_points samples).

    Saves to out_path (PNG).
    """
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    x = np.arange(n_points)

    axes[0].plot(x, bundle.noisy["sum"][:n_points], color="gray", alpha=0.7)
    axes[0].set_title("Noisy composite sum (S1+S2+S3+S4 + noise)")
    axes[0].set_ylabel("Amplitude")

    colors = ["C0", "C1", "C2", "C3"]
    sig_keys = ["s1", "s2", "s3", "s4"]
    for ax, label, color, key in (
        zip(axes[1:], _FREQ_LABELS, colors, sig_keys)  # noqa: B905
    ):
        ax.plot(x, bundle.clean[key][:n_points], color=color)
        ax.set_title(f"Clean component — {label}")
        ax.set_ylabel("Amplitude")

    axes[-1].set_xlabel("Sample index")
    fig.suptitle("Signal Overview: Noisy Sum vs Clean Components", fontsize=13)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_signal_examples(
    checkpoint_dir: str,
    sample_x: list[float],
    sample_y: list[float],
    hidden_size: int,
    out_path: str,
    n_layers: int = 2,
) -> None:
    """
    Predicted vs. clean target window for FC, RNN, LSTM.

    checkpoint_dir: directory containing {fc,rnn,lstm}/best_model.pt.
    sample_x: one flattened input vector of length 14.
    sample_y: clean target window of length 10.
    Saves to out_path (PNG).
    """
    x_tensor = torch.tensor(sample_x, dtype=torch.float32).unsqueeze(0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    time_axis = np.arange(_WINDOW_SIZE)

    for ax, model_type in zip(axes, ["fc", "rnn", "lstm"]):  # noqa: B905
        cls = _MODEL_CLASSES[model_type]
        n_layers_kw = {} if model_type == "fc" else {"n_layers": n_layers}
        model = cls(hidden_size=hidden_size, **n_layers_kw)
        ckpt = Path(checkpoint_dir) / model_type / "best_model.pt"
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()
        with torch.no_grad():
            pred = model(x_tensor).squeeze(0).numpy()
        ax.plot(time_axis, sample_y, label="Clean target", marker="o")
        ax.plot(time_axis, pred, label="Predicted", marker="x", linestyle="--")
        ax.set_title(model_type.upper())
        ax.set_xlabel("Window position")
        ax.legend()

    axes[0].set_ylabel("Amplitude")
    fig.suptitle("Predicted vs. Clean Window (seed=42)", fontsize=13)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
