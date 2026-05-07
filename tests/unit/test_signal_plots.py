"""
Tests for visualization/signal_plots.py.

Uses test config for a fast SignalBundle and creates mock checkpoints.
Requirements source: docs/TODO.md T-082, T-083.
"""

from __future__ import annotations

import os

import torch

from signal_extraction.models.fc_model import FCModel
from signal_extraction.models.lstm_model import LSTMModel
from signal_extraction.models.rnn_model import RNNModel
from signal_extraction.sdk.sdk import SignalExtractionSDK
from signal_extraction.visualization.signal_plots import (
    plot_signal_examples,
    plot_signal_overview,
)

TEST_CONFIG = "config/test_setup.json"
_HIDDEN_SIZE = 32
_MODELS = {"fc": FCModel, "rnn": RNNModel, "lstm": LSTMModel}


def _make_checkpoints(base_dir: str) -> None:
    """Create minimal valid checkpoint files for all three models."""
    for model_type, cls in _MODELS.items():
        n_layers_kw = {} if model_type == "fc" else {"n_layers": 1}
        model = cls(hidden_size=_HIDDEN_SIZE, **n_layers_kw)
        ckpt_dir = os.path.join(base_dir, model_type)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pt"))


# ---------------------------------------------------------------------------
# plot_signal_overview
# ---------------------------------------------------------------------------


def test_plot_signal_overview_creates_png(tmp_path):
    sdk = SignalExtractionSDK(TEST_CONFIG)
    bundle = sdk.generate_signals()
    out = str(tmp_path / "signal_overview.png")
    plot_signal_overview(bundle, out_path=out)
    assert os.path.exists(out)


def test_plot_signal_overview_respects_n_points(tmp_path):
    sdk = SignalExtractionSDK(TEST_CONFIG)
    bundle = sdk.generate_signals()
    out = str(tmp_path / "signal_overview_200.png")
    plot_signal_overview(bundle, out_path=out, n_points=200)
    assert os.path.exists(out)


# ---------------------------------------------------------------------------
# plot_signal_examples
# ---------------------------------------------------------------------------


def test_plot_signal_examples_creates_png(tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    _make_checkpoints(ckpt_dir)

    # Build a minimal sample using the SDK
    sdk = SignalExtractionSDK(TEST_CONFIG)
    bundle = sdk.generate_signals()
    from signal_extraction.services.dataset_builder import DatasetBuilderService
    builder = DatasetBuilderService(seed=42)
    samples = builder.build(bundle, n_samples=1)
    sample = samples[0]

    out = str(tmp_path / "signal_examples.png")
    plot_signal_examples(
        checkpoint_dir=ckpt_dir,
        sample_x=sample.x.tolist(),
        sample_y=sample.y.tolist(),
        hidden_size=_HIDDEN_SIZE,
        out_path=out,
        n_layers=1,
    )
    assert os.path.exists(out)
