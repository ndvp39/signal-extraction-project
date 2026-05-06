# PLAN — Architecture & Design Document

**Version:** 1.00
**Author:** MSC Student
**Date:** 2026-05-06
**Status:** Approved

---

## 1. C4 Model

### 1.1 Level 1 — System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                        System Context                           │
│                                                                 │
│   ┌──────────┐       runs experiments        ┌───────────────┐ │
│   │  Student │ ─────────────────────────────▶│  Signal       │ │
│   │  (User)  │                               │  Extraction   │ │
│   └──────────┘       reads results           │  System       │ │
│                ◀─────────────────────────────│               │ │
│                                              └───────┬───────┘ │
│                                                      │         │
│                                              reads config /    │
│                                              writes results    │
│                                                      │         │
│                                              ┌───────▼───────┐ │
│                                              │  File System  │ │
│                                              │ (config/,     │ │
│                                              │  results/,    │ │
│                                              │  assets/)     │ │
│                                              └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Note:** No external APIs are used. The system is entirely self-contained.

---

### 1.2 Level 2 — Container Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Signal Extraction System                     │
│                                                                      │
│  ┌─────────────┐     delegates      ┌──────────────────────────────┐ │
│  │   CLI /     │ ──────────────────▶│          SDK Layer           │ │
│  │  main.py    │                   │   SignalExtractionSDK        │ │
│  └─────────────┘                   │   (single entry point)       │ │
│                                    └──────────────┬───────────────┘ │
│  ┌─────────────┐     uses           │                               │ │
│  │  Analysis   │ ──────────────────▶│                               │ │
│  │  Notebook   │                   │                               │ │
│  └─────────────┘                   │                               │ │
│                                    ▼                               │ │
│                    ┌───────────────────────────────┐               │ │
│                    │         Domain Services        │               │ │
│                    │  ┌──────────────────────────┐ │               │ │
│                    │  │   SignalGeneratorService  │ │               │ │
│                    │  ├──────────────────────────┤ │               │ │
│                    │  │   DatasetBuilderService   │ │               │ │
│                    │  ├──────────────────────────┤ │               │ │
│                    │  │      TrainerService       │ │               │ │
│                    │  ├──────────────────────────┤ │               │ │
│                    │  │     EvaluatorService      │ │               │ │
│                    │  └──────────────────────────┘ │               │ │
│                    └───────────────┬───────────────┘               │ │
│                                    │                               │ │
│                    ┌───────────────▼───────────────┐               │ │
│                    │         Infrastructure         │               │ │
│                    │  ┌──────────┐  ┌───────────┐  │               │ │
│                    │  │ Config   │  │  FileIO   │  │               │ │
│                    │  │ Manager  │  │  (save /  │  │               │ │
│                    │  │          │  │   load)   │  │               │ │
│                    │  └──────────┘  └───────────┘  │               │ │
│                    └───────────────────────────────┘               │ │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 1.3 Level 3 — Component Diagram (SDK & Services)

```
┌──────────────────────────────────────────────────────────────────────┐
│                           SDK Layer                                  │
│                                                                      │
│   SignalExtractionSDK                                                │
│   ├── generate_signals(params) ──────────────▶ SignalGeneratorService│
│   ├── build_dataset(signals, cfg) ───────────▶ DatasetBuilderService │
│   ├── train_model(model_type, data, cfg) ────▶ TrainerService        │
│   ├── evaluate_model(model, data) ───────────▶ EvaluatorService      │
│   └── run_experiment(cfg) ──────────────────▶ (all services)        │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                        Domain Services                               │
│                                                                      │
│  SignalGeneratorService                                              │
│  ├── _make_sinusoid(A, f, phi, t) → np.ndarray                      │
│  ├── _add_noise(signal, alpha, beta, dist) → np.ndarray             │
│  └── generate(params) → SignalBundle                                 │
│                                                                      │
│  DatasetBuilderService                                               │
│  ├── _make_sample(bundle, pos, selector) → Sample                   │
│  ├── build(bundle, n_samples) → SampleList                          │
│  └── split(samples, ratios) → (train, val, test)                    │
│                                                                      │
│  TrainerService                                                      │
│  ├── _train_epoch(model, loader, opt) → float (loss)                │
│  ├── _validate(model, loader) → float (loss)                        │
│  └── train(model, train_loader, val_loader, cfg) → TrainResult      │
│                                                                      │
│  EvaluatorService                                                    │
│  ├── _mse_per_frequency(model, loader) → Dict[int, float]           │
│  └── evaluate(model, test_loader) → EvalResult                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                            Models                                    │
│                                                                      │
│  BaseModel (ABC)                                                     │
│  ├── FCModel(BaseModel)   — no temporal structure                    │
│  ├── RNNModel(BaseModel)  — short-term memory, vanishing gradient    │
│  └── LSTMModel(BaseModel) — long-term memory via gating             │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 1.4 Level 4 — Code Diagram (Key Classes)

```
BaseModel (torch.nn.Module, ABC)
│  forward(x: Tensor[B, 14]) → Tensor[B, 10]  [abstract]
│
├── FCModel
│   Layers: Linear(14→H) → ReLU → Linear(H→H) → ReLU → Linear(H→10)
│   H = hidden_size (from config)
│
├── RNNModel
│   Layers: Linear(14→H) → RNN(H, H, n_layers) → Linear(H→10)
│   Input reshaped to (B, 1, 14) — single time-step
│
└── LSTMModel
    Layers: Linear(14→H) → LSTM(H, H, n_layers) → Linear(H→10)
    Input reshaped to (B, 1, 14) — single time-step

SignalBundle (dataclass)
│  clean: Dict[str, np.ndarray]   # keys: "s1","s2","s3","s4","sum"
│  noisy: Dict[str, np.ndarray]   # keys: "s1","s2","s3","s4","sum"
│  t:     np.ndarray              # time axis, shape (10000,)
│  params: SignalParams

SignalParams (dataclass)
│  frequencies:  List[float]      # length 4
│  amplitudes:   List[float]      # length 4
│  phases:       List[float]      # length 4
│  alpha:        float            # amplitude noise strength
│  beta:         float            # phase noise strength
│  noise_dist:   str              # "gaussian" | "uniform"
│  sample_rate:  int              # 1000
│  duration:     float            # 10.0
│  seed:         int

Sample (dataclass)
│  x: np.ndarray   # shape (14,)  = [C(4) | noisy_sum_window(10)]
│  y: np.ndarray   # shape (10,)  = clean target window

TrainResult (dataclass)
│  train_losses: List[float]
│  val_losses:   List[float]
│  best_epoch:   int
│  model_path:   str

EvalResult (dataclass)
│  mse_overall:       float
│  mse_per_freq:      Dict[int, float]   # key = frequency index 0–3
│  predictions:       np.ndarray         # shape (N, 10)
│  targets:           np.ndarray         # shape (N, 10)
```

---

## 2. UML — Training Sequence Diagram

```
Student          CLI/main.py        SDK              TrainerService     FileIO
   │                  │              │                     │               │
   │── uv run ───────▶│              │                     │               │
   │                  │─load config─▶│                     │               │
   │                  │             │─generate_signals()──▶│               │
   │                  │             │─build_dataset() ────▶│               │
   │                  │             │─train_model() ──────▶│               │
   │                  │             │                      │─train_epoch()─│
   │                  │             │                      │◀──loss────────│
   │                  │             │                      │   (× epochs)  │
   │                  │             │◀──TrainResult────────│               │
   │                  │             │──────────────────────────save ──────▶│
   │                  │◀─results───│                     │               │
   │◀─── output ──────│              │                     │               │
```

---

## 3. Project File Structure

```
signal-extraction-project/
├── src/
│   └── signal_extraction/
│       ├── __init__.py
│       ├── constants.py              # Immutable physical constants
│       ├── sdk/
│       │   ├── __init__.py
│       │   └── sdk.py                # SignalExtractionSDK
│       ├── services/
│       │   ├── __init__.py
│       │   ├── signal_generator.py   # SignalGeneratorService
│       │   ├── dataset_builder.py    # DatasetBuilderService
│       │   ├── trainer.py            # TrainerService
│       │   └── evaluator.py          # EvaluatorService
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base_model.py         # BaseModel ABC
│       │   ├── fc_model.py           # FCModel
│       │   ├── rnn_model.py          # RNNModel
│       │   └── lstm_model.py         # LSTMModel
│       └── shared/
│           ├── __init__.py
│           ├── config.py             # ConfigManager
│           ├── version.py            # __version__ = "1.00"
│           └── schemas.py            # Dataclasses: SignalParams, Sample, etc.
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_signal_generator.py
│   │   ├── test_dataset_builder.py
│   │   ├── test_trainer.py
│   │   ├── test_evaluator.py
│   │   ├── test_fc_model.py
│   │   ├── test_rnn_model.py
│   │   ├── test_lstm_model.py
│   │   └── test_sdk.py
│   └── integration/
│       └── test_full_pipeline.py
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   ├── TODO.md
│   ├── PRD_signal_generation.md
│   ├── PRD_ml_models.md
│   └── adr/                          # Individual ADR files
│       ├── ADR-001-noise-distribution.md
│       ├── ADR-002-framework-pytorch.md
│       ├── ADR-003-model-input-concatenation.md
│       ├── ADR-004-loss-function-mse.md
│       ├── ADR-005-sdk-architecture.md
│       ├── ADR-006-one-hot-selector.md
│       └── ADR-007-fixed-window-size.md
├── config/
│   └── setup.json
├── data/                             # Generated signals saved here
├── results/                          # Trained model checkpoints, metrics
├── assets/                           # Plots, graphs, screenshots
├── notebooks/
│   └── analysis.ipynb
├── README.md
├── pyproject.toml
├── uv.lock
├── .env-example
└── .gitignore
```

---

## 4. Data Schemas & Interfaces

### 4.1 `config/setup.json` Schema

```json
{
  "version": "1.00",
  "signals": {
    "frequencies": [10, 50, 120, 300],
    "amplitudes":  [1.0, 0.8, 1.2, 0.6],
    "phases":      [0.0, 0.5, 1.0, 1.5],
    "alpha":       0.1,
    "beta":        0.1,
    "noise_dist":  "gaussian",
    "sample_rate": 1000,
    "duration":    10.0,
    "seed":        42
  },
  "dataset": {
    "n_samples":   50000,
    "window_size": 10,
    "n_signals":   4,
    "split":       [0.7, 0.15, 0.15]
  },
  "training": {
    "epochs":      100,
    "batch_size":  256,
    "learning_rate": 0.001,
    "optimizer":   "adam",
    "patience":    10
  },
  "models": {
    "hidden_size": 128,
    "n_layers":    2,
    "dropout":     0.0
  },
  "paths": {
    "data_dir":    "data/",
    "results_dir": "results/",
    "assets_dir":  "assets/"
  }
}
```

### 4.2 SDK Public Interface

```python
class SignalExtractionSDK:
    """Single entry point for all signal extraction operations."""

    def __init__(self, config_path: str) -> None: ...

    def generate_signals(self) -> SignalBundle:
        """Generate all 10 signal vectors from config parameters."""

    def build_dataset(
        self, bundle: SignalBundle
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Build train/val/test DataLoaders from signal bundle."""

    def train_model(
        self, model_type: str, train_loader: DataLoader, val_loader: DataLoader
    ) -> tuple[BaseModel, TrainResult]:
        """Train a model of given type ('fc'|'rnn'|'lstm')."""

    def evaluate_model(
        self, model: BaseModel, test_loader: DataLoader
    ) -> EvalResult:
        """Evaluate a trained model on the test set."""

    def run_experiment(self, model_types: list[str]) -> dict[str, EvalResult]:
        """Run full pipeline for each model type; return all results."""

    def save_results(self, results: dict[str, EvalResult]) -> None:
        """Persist results and generate plots to assets/."""
```

### 4.3 Model Interface

```python
class BaseModel(nn.Module, ABC):
    """Abstract base for all extraction models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, 14)
               First 4 dims = one-hot selector C
               Last 10 dims = noisy sum window
        Returns:
            Tensor of shape (batch_size, 10) — predicted clean window
        """
```

---

## 5. Architectural Decision Records (ADRs)

Each ADR is maintained as an individual file in `docs/adr/` for traceability and easier review.

| ADR | Title | Status |
|---|---|---|
| [ADR-001](adr/ADR-001-noise-distribution.md) | Noise Distribution: Gaussian | Accepted |
| [ADR-002](adr/ADR-002-framework-pytorch.md) | Deep Learning Framework: PyTorch | Accepted |
| [ADR-003](adr/ADR-003-model-input-concatenation.md) | Model Input: Concatenation of C and Noisy Window | Accepted |
| [ADR-004](adr/ADR-004-loss-function-mse.md) | Loss Function: MSE | Accepted |
| [ADR-005](adr/ADR-005-sdk-architecture.md) | SDK Architecture as Single Entry Point | Accepted |
| [ADR-006](adr/ADR-006-one-hot-selector.md) | Selector C as One-Hot Vector (No Embedding) | Accepted |
| [ADR-007](adr/ADR-007-fixed-window-size.md) | Fixed Context Window Size = 10 | Accepted |

---

## 6. Deployment Diagram

```
Developer Machine
┌──────────────────────────────────────────┐
│                                          │
│   uv sync          (install deps)        │
│   uv run python src/main.py --model fc   │
│   uv run pytest tests/                   │
│   uv run jupyter notebook notebooks/     │
│                                          │
│  ┌─────────────────────────────────────┐ │
│  │  Python 3.10+ virtual env (uv)      │ │
│  │  torch, numpy, matplotlib, jupyter  │ │
│  └─────────────────────────────────────┘ │
│                                          │
│  Outputs:                                │
│    results/   ← model checkpoints        │
│    assets/    ← plots & graphs           │
│    data/      ← generated signals        │
└──────────────────────────────────────────┘
```

No cloud or containerization required — fully local execution.

---

## 7. Key Design Principles Applied

| Principle | How Applied |
|---|---|
| Single Responsibility | Each service class has one job (generate / build / train / evaluate) |
| DRY | `BaseModel` holds shared forward-pass contract; no repeated layer code |
| Open/Closed | New model types can be added by subclassing `BaseModel` without changing SDK |
| Dependency Inversion | Services depend on dataclass schemas, not concrete implementations |
| Config over Code | All tunable values in `config/setup.json`; only physical constants in code |
