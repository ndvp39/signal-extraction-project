# TODO — Task List

**Version:** 1.00
**Author:** MSC Student
**Date:** 2026-05-06

**Status Legend:**
- `[ ]` Not started
- `[~]` In progress
- `[x]` Done

**Priority Legend:** `P1` = Critical path | `P2` = High | `P3` = Medium | `P4` = Nice to have

---

## Phase 0 — Documentation (M1)

> Must be fully approved before any code is written.

| ID | Priority | Status | Task | Owner | Definition of Done |
|---|---|---|---|---|---|
| T-001 | P1 | `[x]` | Write `docs/PRD.md` | Student | File exists, all 14 sections complete, accepted criteria listed |
| T-002 | P1 | `[x]` | Write `docs/PLAN.md` | Student | C4 diagrams, ADRs, schemas, file structure all present |
| T-003 | P1 | `[~]` | Write `docs/TODO.md` | Student | This file — all phases and tasks listed with DoD |
| T-004 | P1 | `[ ]` | Write `docs/PRD_signal_generation.md` | Student | Dedicated PRD covers signal math, noise model, I/O, test scenarios |
| T-005 | P1 | `[ ]` | Write `docs/PRD_ml_models.md` | Student | Dedicated PRD covers FC/RNN/LSTM theory, I/O, hyperparams, test scenarios |

---

## Phase 1 — Project Scaffold (M2-prep)

> Set up the package, tooling, and configuration before writing domain code.

| ID | Priority | Status | Task | Owner | Definition of Done |
|---|---|---|---|---|---|
| T-010 | P1 | `[ ]` | Initialize project with `uv init` | Student | `pyproject.toml` and `uv.lock` exist; `uv sync` runs clean |
| T-011 | P1 | `[ ]` | Add all dependencies via `uv add` | Student | torch, numpy, matplotlib, seaborn, jupyter, pytest, pytest-cov, ruff in `pyproject.toml` |
| T-012 | P1 | `[ ]` | Create full directory structure | Student | All dirs (src/, tests/, docs/, config/, data/, results/, assets/, notebooks/) exist |
| T-013 | P1 | `[ ]` | Create `src/signal_extraction/__init__.py` and all sub-package `__init__.py` files | Student | Package importable via `from signal_extraction import ...` |
| T-014 | P1 | `[ ]` | Write `src/signal_extraction/shared/version.py` | Student | `__version__ = "1.00"` defined and importable |
| T-015 | P1 | `[ ]` | Write `src/signal_extraction/constants.py` | Student | `WINDOW_SIZE=10`, `N_SIGNALS=4`, `SAMPLE_RATE=1000`, `DURATION=10.0` defined |
| T-016 | P1 | `[ ]` | Write `config/setup.json` | Student | Valid JSON matching schema in PLAN.md; version field = "1.00" |
| T-017 | P1 | `[ ]` | Write `src/signal_extraction/shared/config.py` | Student | `ConfigManager` loads `setup.json`, validates version, raises on missing keys |
| T-018 | P2 | `[ ]` | Configure `ruff` in `pyproject.toml` | Student | `uv run ruff check src/` exits with 0 violations |
| T-019 | P2 | `[ ]` | Configure `pytest` and `pytest-cov` in `pyproject.toml` | Student | `uv run pytest tests/` runs; coverage report generated; fail_under = 85 |
| T-020 | P2 | `[ ]` | Create `.env-example` and `.gitignore` | Student | `.env-example` has placeholder values; `.gitignore` covers `.env`, `__pycache__`, `*.pem` |

---

## Phase 2 — Signal Generation (M2)

> Implement and test the signal generator before dataset or model work.

| ID | Priority | Status | Task | Owner | Definition of Done |
|---|---|---|---|---|---|
| T-030 | P1 | `[ ]` | Write `shared/schemas.py` — `SignalParams`, `SignalBundle`, `Sample`, `TrainResult`, `EvalResult` dataclasses | Student | All dataclasses importable; fields match PLAN.md schema |
| T-031 | P1 | `[ ]` | Write `tests/unit/test_signal_generator.py` (TDD — write first) | Student | Tests cover: correct shape (10000,), formula correctness, noise application, bundle structure |
| T-032 | P1 | `[ ]` | Write `services/signal_generator.py` — `SignalGeneratorService` | Student | All T-031 tests pass; file ≤ 150 lines; `ruff` clean |
| T-033 | P2 | `[ ]` | Verify 10 output vectors (S1–S4, Sum, noisy x5) | Student | Manual spot-check + assertion in test: shapes = (10000,), sum = S1+S2+S3+S4 |
| T-034 | P2 | `[ ]` | Test noise parametrization (alpha=0 → no amplitude noise, beta=0 → no phase noise) | Student | Edge-case tests pass |

---

## Phase 3 — Dataset Builder (M3)

| ID | Priority | Status | Task | Owner | Definition of Done |
|---|---|---|---|---|---|
| T-040 | P1 | `[ ]` | Write `tests/unit/test_dataset_builder.py` (TDD — write first) | Student | Tests cover: sample shape (x=(14,), y=(10,)), one-hot validity, window bounds, split ratios |
| T-041 | P1 | `[ ]` | Write `services/dataset_builder.py` — `DatasetBuilderService` | Student | All T-040 tests pass; file ≤ 150 lines; `ruff` clean |
| T-042 | P1 | `[ ]` | Write `SignalDataset` (torch `Dataset` subclass) | Student | `__len__` and `__getitem__` work; tensors of correct dtype (float32) |
| T-043 | P2 | `[ ]` | Verify DataLoader batching | Student | Batch shape = (B, 14) for x, (B, 10) for y |
| T-044 | P2 | `[ ]` | Test reproducibility with fixed seed | Student | Two runs with same seed produce identical datasets |

---

## Phase 4 — Model Implementation (M4)

| ID | Priority | Status | Task | Owner | Definition of Done |
|---|---|---|---|---|---|
| T-050 | P1 | `[ ]` | Write `models/base_model.py` — `BaseModel` ABC | Student | Abstract `forward()` method defined; subclasses must implement it |
| T-051 | P1 | `[ ]` | Write `tests/unit/test_fc_model.py` (TDD — write first) | Student | Tests cover: output shape (B,10), forward pass no-error, parameter count sanity |
| T-052 | P1 | `[ ]` | Write `models/fc_model.py` — `FCModel` | Student | All T-051 tests pass; file ≤ 150 lines; `ruff` clean |
| T-053 | P1 | `[ ]` | Write `tests/unit/test_rnn_model.py` (TDD — write first) | Student | Tests cover: output shape, hidden state handling, configurable layers |
| T-054 | P1 | `[ ]` | Write `models/rnn_model.py` — `RNNModel` | Student | All T-053 tests pass; file ≤ 150 lines; `ruff` clean |
| T-055 | P1 | `[ ]` | Write `tests/unit/test_lstm_model.py` (TDD — write first) | Student | Tests cover: output shape, cell/hidden state handling, configurable layers |
| T-056 | P1 | `[ ]` | Write `models/lstm_model.py` — `LSTMModel` | Student | All T-055 tests pass; file ≤ 150 lines; `ruff` clean |

---

## Phase 5 — Training & Evaluation Pipeline (M5)

| ID | Priority | Status | Task | Owner | Definition of Done |
|---|---|---|---|---|---|
| T-060 | P1 | `[ ]` | Write `tests/unit/test_trainer.py` (TDD — write first) | Student | Tests cover: loss decreases over 2+ epochs, checkpoint saved, early stopping triggers |
| T-061 | P1 | `[ ]` | Write `services/trainer.py` — `TrainerService` | Student | All T-060 tests pass; file ≤ 150 lines; `ruff` clean |
| T-062 | P1 | `[ ]` | Write `tests/unit/test_evaluator.py` (TDD — write first) | Student | Tests cover: MSE computed correctly, per-frequency breakdown, EvalResult populated |
| T-063 | P1 | `[ ]` | Write `services/evaluator.py` — `EvaluatorService` | Student | All T-062 tests pass; file ≤ 150 lines; `ruff` clean |
| T-064 | P1 | `[ ]` | Write `tests/unit/test_sdk.py` (TDD — write first) | Student | Tests cover: SDK public interface, end-to-end minimal run, config loading |
| T-065 | P1 | `[ ]` | Write `sdk/sdk.py` — `SignalExtractionSDK` | Student | All T-064 tests pass; file ≤ 150 lines; `ruff` clean |
| T-066 | P1 | `[ ]` | Write `src/main.py` — CLI entry point | Student | Argument parsing only; delegates entirely to SDK; no business logic |
| T-067 | P2 | `[ ]` | Write `tests/integration/test_full_pipeline.py` | Student | Full pipeline (generate→dataset→train→evaluate) runs end-to-end without error |

---

## Phase 6 — Experiments & Analysis (M6)

| ID | Priority | Status | Task | Owner | Definition of Done |
|---|---|---|---|---|---|
| T-070 | P1 | `[ ]` | Train FC model — baseline experiment | Student | Checkpoint saved to `results/fc/`; train/val loss logged |
| T-071 | P1 | `[ ]` | Train RNN model | Student | Checkpoint saved to `results/rnn/`; train/val loss logged |
| T-072 | P1 | `[ ]` | Train LSTM model | Student | Checkpoint saved to `results/lstm/`; train/val loss logged |
| T-073 | P1 | `[ ]` | Evaluate all three models on test set | Student | MSE overall + MSE per frequency for each model recorded |
| T-074 | P1 | `[ ]` | Experiment: vary noise level (alpha/beta × 3 values) | Student | Results table with MSE vs. noise level for each model |
| T-075 | P2 | `[ ]` | Experiment: vary hidden size | Student | Sensitivity curve: hidden_size vs. test MSE |
| T-076 | P2 | `[ ]` | Experiment: vary number of layers | Student | Sensitivity curve: n_layers vs. test MSE |
| T-077 | P3 | `[ ]` | Experiment: vary learning rate | Student | LR sweep results documented |

---

## Phase 7 — Visualization & Results (M6 cont.)

| ID | Priority | Status | Task | Owner | Definition of Done |
|---|---|---|---|---|---|
| T-080 | P1 | `[ ]` | Plot training curves (loss vs. epoch) for all 3 models | Student | PNG saved to `assets/training_curves.png`; axes labeled |
| T-081 | P1 | `[ ]` | Plot MSE comparison bar chart (model × frequency) | Student | PNG saved to `assets/mse_comparison.png` |
| T-082 | P1 | `[ ]` | Plot predicted vs. clean window examples (one per model) | Student | PNG saved to `assets/signal_examples.png` |
| T-083 | P2 | `[ ]` | Plot noisy sum signal vs. clean sinusoids (signal overview) | Student | PNG saved to `assets/signal_overview.png` |
| T-084 | P2 | `[ ]` | Plot noise sensitivity heatmap (model × noise level) | Student | Heatmap saved to `assets/noise_heatmap.png` |
| T-085 | P3 | `[ ]` | Plot hyperparameter sensitivity curves | Student | PNGs saved to `assets/sensitivity_*.png` |

---

## Phase 8 — Analysis Notebook & README (M7)

| ID | Priority | Status | Task | Owner | Definition of Done |
|---|---|---|---|---|---|
| T-090 | P1 | `[ ]` | Write `notebooks/analysis.ipynb` | Student | Loads results, reproduces all plots, comparative discussion with LaTeX equations |
| T-091 | P1 | `[ ]` | Write `README.md` | Student | Installation, usage, examples, all graphs embedded, conclusions present |
| T-092 | P2 | `[ ]` | Write Prompts Book section in README | Student | Key prompts used documented with context, output, and lessons learned |

---

## Phase 9 — Quality Gate (Final)

| ID | Priority | Status | Task | Owner | Definition of Done |
|---|---|---|---|---|---|
| T-100 | P1 | `[ ]` | `uv run ruff check src/` — zero violations | Student | Exit code 0 |
| T-101 | P1 | `[ ]` | `uv run pytest tests/ --cov=src --cov-fail-under=85` | Student | All tests pass; coverage ≥ 85% |
| T-102 | P1 | `[ ]` | Verify all Python files ≤ 150 lines of code | Student | Automated check passes (blank/comment lines excluded) |
| T-103 | P1 | `[ ]` | Verify no hard-coded values in source (config-driven) | Student | Manual review + grep for magic numbers passes |
| T-104 | P1 | `[ ]` | Verify all mandatory docs exist | Student | PRD.md, PLAN.md, TODO.md, PRD_signal_generation.md, PRD_ml_models.md all present |
| T-105 | P2 | `[ ]` | Verify `uv.lock` is up to date and committed | Student | `uv lock --check` passes |
| T-106 | P2 | `[ ]` | Final README review — screenshots, graphs, conclusions | Student | Peer review checklist passed |

---

## Milestone Summary

| Milestone | Tasks | Target |
|---|---|---|
| M1 — All docs approved | T-001 → T-005 | Before any code |
| M2 — Signal generation | T-010 → T-034 | Scaffold + generator complete |
| M3 — Dataset | T-040 → T-044 | Dataset builder complete |
| M4 — Models | T-050 → T-056 | FC, RNN, LSTM implemented |
| M5 — Training pipeline | T-060 → T-067 | SDK + CLI complete |
| M6 — Experiments | T-070 → T-085 | All experiments run; plots saved |
| M7 — Analysis & README | T-090 → T-092 | Notebook + README complete |
| Final — Quality gate | T-100 → T-106 | Submission ready |
