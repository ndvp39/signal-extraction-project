"""
Unit tests for SignalExtractionSDK — written BEFORE implementation (TDD RED phase).

Tests the public interface of the SDK using the real config/setup.json.
Requirements source: PLAN.md SDK interface, docs/TODO.md T-064.
"""

from __future__ import annotations

import pytest
from torch.utils.data import DataLoader

from signal_extraction.models.base_model import BaseModel
from signal_extraction.sdk.sdk import SignalExtractionSDK
from signal_extraction.shared.schemas import EvalResult, SignalBundle, TrainResult

# Use a minimal config so SDK tests run in seconds, not minutes.
# test_setup.json has n_samples=400, epochs=3, hidden_size=32.
CONFIG_PATH = "config/test_setup.json"


@pytest.fixture(scope="module")
def sdk() -> SignalExtractionSDK:
    return SignalExtractionSDK(CONFIG_PATH)


@pytest.fixture(scope="module")
def bundle(sdk: SignalExtractionSDK) -> SignalBundle:
    return sdk.generate_signals()


@pytest.fixture(scope="module")
def loaders(sdk: SignalExtractionSDK, bundle: SignalBundle):
    return sdk.build_dataset(bundle)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_sdk_loads_config_without_error() -> None:
    SignalExtractionSDK(CONFIG_PATH)


def test_sdk_raises_on_missing_config() -> None:
    with pytest.raises(Exception):
        SignalExtractionSDK("config/nonexistent.json")


# ---------------------------------------------------------------------------
# generate_signals()
# ---------------------------------------------------------------------------


def test_generate_signals_returns_bundle(bundle: SignalBundle) -> None:
    assert isinstance(bundle, SignalBundle)


def test_bundle_has_all_clean_keys(bundle: SignalBundle) -> None:
    assert set(bundle.clean.keys()) == {"s1", "s2", "s3", "s4", "sum"}


def test_bundle_has_all_noisy_keys(bundle: SignalBundle) -> None:
    assert set(bundle.noisy.keys()) == {"s1", "s2", "s3", "s4", "sum"}


# ---------------------------------------------------------------------------
# build_dataset()
# ---------------------------------------------------------------------------


def test_build_dataset_returns_three_loaders(loaders) -> None:
    assert len(loaders) == 3


def test_all_loaders_are_dataloader_instances(loaders) -> None:
    train, val, test = loaders
    assert isinstance(train, DataLoader)
    assert isinstance(val, DataLoader)
    assert isinstance(test, DataLoader)


def test_loaders_yield_correct_batch_shapes(loaders) -> None:
    from signal_extraction.constants import INPUT_SIZE, OUTPUT_SIZE
    train, _, _ = loaders
    x, y = next(iter(train))
    assert x.shape[1] == INPUT_SIZE
    assert y.shape[1] == OUTPUT_SIZE


# ---------------------------------------------------------------------------
# train_model()
# ---------------------------------------------------------------------------


def test_train_model_returns_model_and_result(sdk: SignalExtractionSDK, loaders) -> None:
    train, val, _ = loaders
    model, result = sdk.train_model("fc", train, val)
    assert isinstance(model, BaseModel)
    assert isinstance(result, TrainResult)


def test_train_model_rnn(sdk: SignalExtractionSDK, loaders) -> None:
    train, val, _ = loaders
    model, result = sdk.train_model("rnn", train, val)
    assert isinstance(model, BaseModel)
    assert len(result.train_losses) > 0


def test_train_model_lstm(sdk: SignalExtractionSDK, loaders) -> None:
    train, val, _ = loaders
    model, result = sdk.train_model("lstm", train, val)
    assert isinstance(model, BaseModel)
    assert len(result.train_losses) > 0


def test_invalid_model_type_raises(sdk: SignalExtractionSDK, loaders) -> None:
    train, val, _ = loaders
    with pytest.raises(ValueError, match="model_type"):
        sdk.train_model("transformer", train, val)


# ---------------------------------------------------------------------------
# evaluate_model()
# ---------------------------------------------------------------------------


def test_evaluate_model_returns_eval_result(sdk: SignalExtractionSDK, loaders) -> None:
    train, val, test = loaders
    model, _ = sdk.train_model("fc", train, val)
    result = sdk.evaluate_model(model, test)
    assert isinstance(result, EvalResult)
    assert result.mse_overall > 0
