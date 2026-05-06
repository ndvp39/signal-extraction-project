"""
Integration test — full pipeline: generate → dataset → train → evaluate.

Uses test_setup.json (n_samples=400, epochs=3) to keep runtime under 30s.
Verifies that all components work together end-to-end without error.
Requirements source: docs/TODO.md T-067.
"""

from __future__ import annotations

import pytest

from signal_extraction.sdk.sdk import SignalExtractionSDK
from signal_extraction.shared.schemas import EvalResult, TrainResult

CONFIG_PATH = "config/test_setup.json"


@pytest.fixture(scope="module")
def sdk() -> SignalExtractionSDK:
    return SignalExtractionSDK(CONFIG_PATH)


@pytest.mark.parametrize("model_type", ["fc", "rnn", "lstm"])
def test_full_pipeline_for_model(sdk: SignalExtractionSDK, model_type: str) -> None:
    """
    End-to-end pipeline test for a single model type.

    Steps: generate signals → build dataset → train → evaluate.
    Asserts that each step returns the correct type and that
    the final MSE is a positive finite number.
    """
    bundle = sdk.generate_signals()
    train_loader, val_loader, test_loader = sdk.build_dataset(bundle)
    model, train_result = sdk.train_model(model_type, train_loader, val_loader)
    eval_result = sdk.evaluate_model(model, test_loader)

    assert isinstance(train_result, TrainResult)
    assert len(train_result.train_losses) > 0
    assert all(loss > 0 for loss in train_result.train_losses)

    assert isinstance(eval_result, EvalResult)
    assert eval_result.mse_overall > 0
    assert eval_result.mse_overall < float("inf")


def test_save_results_creates_metrics_file(sdk: SignalExtractionSDK, tmp_path) -> None:
    """save_results() must write a metrics.json to the results directory."""
    import json
    from signal_extraction.shared.schemas import EvalResult

    dummy = {"fc": EvalResult(mse_overall=0.1, mse_per_freq={0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1})}

    # Temporarily override results path via monkeypatching config
    sdk._cfg._data["paths"]["results_dir"] = str(tmp_path) + "/"
    sdk.save_results(dummy)

    metrics_file = tmp_path / "metrics.json"
    assert metrics_file.exists()
    data = json.loads(metrics_file.read_text())
    assert "fc" in data
    assert "mse_overall" in data["fc"]
