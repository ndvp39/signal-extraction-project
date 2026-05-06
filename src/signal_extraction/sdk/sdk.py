"""
SignalExtractionSDK — single entry point for all business logic.

All external consumers (CLI, notebooks, tests) must use this class.
No consumer should import services or models directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import DataLoader

from signal_extraction.models.base_model import BaseModel
from signal_extraction.models.fc_model import FCModel
from signal_extraction.models.lstm_model import LSTMModel
from signal_extraction.models.rnn_model import RNNModel
from signal_extraction.services.dataset_builder import DatasetBuilderService, SignalDataset
from signal_extraction.services.evaluator import EvaluatorService
from signal_extraction.services.signal_generator import SignalGeneratorService
from signal_extraction.services.trainer import TrainerService
from signal_extraction.shared.config import ConfigManager
from signal_extraction.shared.schemas import EvalResult, SignalBundle, SignalParams, TrainResult

_MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "fc": FCModel,
    "rnn": RNNModel,
    "lstm": LSTMModel,
}


class SignalExtractionSDK:
    """
    Orchestrates signal generation, dataset building, training, and evaluation.

    All hyperparameters are read from config/setup.json via ConfigManager.
    No business logic belongs in the CLI or notebooks — delegate here instead.
    """

    def __init__(self, config_path: str) -> None:
        self._cfg = ConfigManager(config_path)
        self._generator = SignalGeneratorService()
        self._builder = DatasetBuilderService(seed=self._cfg.get("signals", "seed", default=0))
        self._trainer = TrainerService()
        self._evaluator = EvaluatorService()

    def generate_signals(self) -> SignalBundle:
        """Generate all 10 signal vectors from config parameters."""
        sig = self._cfg.signals
        params = SignalParams(
            frequencies=sig["frequencies"],
            amplitudes=sig["amplitudes"],
            phases=sig["phases"],
            alpha=sig["alpha"],
            beta=sig["beta"],
            noise_dist=sig["noise_dist"],
            sample_rate=sig["sample_rate"],
            duration=sig["duration"],
            seed=sig["seed"],
        )
        return self._generator.generate(params)

    def build_dataset(
        self, bundle: SignalBundle
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Build train/val/test DataLoaders from a SignalBundle."""
        ds_cfg = self._cfg.dataset
        samples = self._builder.build(bundle, n_samples=ds_cfg["n_samples"])
        train_s, val_s, test_s = self._builder.split(samples, ratios=ds_cfg["split"])
        batch = self._cfg.get("training", "batch_size", default=256)
        return (
            DataLoader(SignalDataset(train_s), batch_size=batch, shuffle=True),
            DataLoader(SignalDataset(val_s), batch_size=batch),
            DataLoader(SignalDataset(test_s), batch_size=batch),
        )

    def train_model(
        self, model_type: str, train_loader: DataLoader, val_loader: DataLoader
    ) -> tuple[BaseModel, TrainResult]:
        """Instantiate and train a model of the given type."""
        if model_type not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type={model_type!r}. Choose from {list(_MODEL_REGISTRY)}."
            )
        m_cfg = self._cfg.models
        t_cfg = self._cfg.training
        p_cfg = self._cfg.paths
        model = _MODEL_REGISTRY[model_type](
            hidden_size=m_cfg["hidden_size"],
            **({} if model_type == "fc" else {"n_layers": m_cfg["n_layers"]}),
        )
        save_path = str(Path(p_cfg["results_dir"]) / model_type / "best_model.pt")
        result = self._trainer.train(
            model, train_loader, val_loader,
            epochs=t_cfg["epochs"],
            lr=t_cfg["learning_rate"],
            patience=t_cfg["patience"],
            save_path=save_path,
        )
        return model, result

    def evaluate_model(self, model: BaseModel, test_loader: DataLoader) -> EvalResult:
        """Evaluate a trained model on the test set."""
        return self._evaluator.evaluate(model, test_loader)

    def save_results(self, results: dict[str, EvalResult]) -> None:
        """Persist evaluation metrics as JSON to the results directory."""
        results_dir = Path(self._cfg.get("paths", "results_dir", default="results/"))
        results_dir.mkdir(parents=True, exist_ok=True)
        output = {
            name: {
                "mse_overall": r.mse_overall,
                "mse_per_freq": {str(k): v for k, v in r.mse_per_freq.items()},
            }
            for name, r in results.items()
        }
        with open(results_dir / "metrics.json", "w") as fh:
            json.dump(output, fh, indent=2)
