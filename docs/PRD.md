# PRD — Signal Extraction with Neural Networks

**Version:** 1.00
**Author:** MSC Student
**Date:** 2026-05-06
**Status:** Approved

---

## 1. Project Overview & Context

This project is a graduate-level (MSC) machine-learning engineering assignment in the domain of **signal processing and temporal neural networks**. The goal is to build a system that uses neural networks to separate and extract a specific sinusoidal signal component from a noisy composite signal.

The system demonstrates the fundamental signal-processing problem of **source separation**: given a mixture of signals plus noise, recover a single clean source using a learned model — guided by an explicit selector.

The project serves as a comparative study of three model families:
- **Fully Connected (FC)** networks — baseline, no temporal structure
- **Recurrent Neural Networks (RNN)** — short-term temporal memory
- **Long Short-Term Memory (LSTM)** — long-term temporal dependencies

---

## 2. Problem Statement

A receiver observes a noisy composite signal — a sum of four sinusoids each corrupted by amplitude and phase noise. The receiver also knows **which** sinusoid it wants to recover (encoded as a one-hot vector). The task is to output the clean version of the selected sinusoid for the observed time window.

This mirrors real-world scenarios such as:
- Radio frequency demultiplexing
- EEG/ECG signal separation
- Audio source separation (cocktail party problem, simplified)

---

## 3. Target Audience

| Role | Description |
|---|---|
| MSC Students | Primary users; run experiments, tune hyperparameters, analyze results |
| Course Lecturer (Dr. Yoram Segal) | Evaluator; assesses architecture understanding and comparative analysis |
| Researchers | Secondary; may reuse signal generation and model modules |

---

## 4. Measurable Goals & KPIs

| Goal | KPI | Acceptance Criterion |
|---|---|---|
| Correct signal extraction | MSE on test set | MSE < 0.05 for at least one model |
| Model comparison | MSE per model per frequency | All three models trained and evaluated |
| Noise robustness | MSE vs. noise level | Results documented for ≥ 2 noise levels |
| Code quality | `ruff check` violations | 0 violations |
| Test coverage | `pytest --cov` | ≥ 85% |
| File size compliance | Lines of code per file | Every file ≤ 150 lines |
| Documentation completeness | Presence of required docs | All mandatory files exist and are complete |

---

## 5. Functional Requirements

### 5.1 Signal Generation

| ID | Requirement |
|---|---|
| FR-01 | Generate 4 sinusoidal signals of the form A·sin(2πft + φ) |
| FR-02 | Sampling rate = 1000 Hz; duration = 10 seconds → 10,000 samples per signal |
| FR-03 | Generate a clean sum signal: S_sum = S1 + S2 + S3 + S4 |
| FR-04 | Generate a noisy version of each sinusoid: (A ± α·ε_A)·sin(2πft + φ ± β·ε_φ) where ε is drawn from uniform or Gaussian distribution |
| FR-05 | Generate a noisy sum signal: σ_noisy = noisy_S1 + noisy_S2 + noisy_S3 + noisy_S4 |
| FR-06 | Expose parameters: frequencies, amplitudes, phases, alpha (amplitude noise strength), beta (phase noise strength), noise distribution type |
| FR-07 | Output 10 vectors total: 5 clean (S1–S4, Sum) + 5 noisy (noisy S1–S4, noisy Sum) |

### 5.2 Dataset Construction

| ID | Requirement |
|---|---|
| FR-08 | Each dataset sample consists of: selector C (one-hot, size 4) + noisy sum window (size 10) as input; clean target window (size 10) as output |
| FR-09 | Window position is chosen uniformly at random from valid positions in the signal |
| FR-10 | Selector C is chosen uniformly at random from {e1, e2, e3, e4} |
| FR-11 | The target window is taken from the clean sinusoid selected by C at the same position as the noisy sum window |
| FR-12 | Dataset must be configurable in size (number of samples) |
| FR-13 | Dataset must support train/validation/test split |

### 5.3 Model Implementation

| ID | Requirement |
|---|---|
| FR-14 | Implement a Fully Connected (FC) model: input size = 14 (4 + 10), output size = 10 |
| FR-15 | Implement an RNN model with configurable hidden size and number of layers |
| FR-16 | Implement an LSTM model with configurable hidden size and number of layers |
| FR-17 | All models must accept the same input format: concatenation of C and the noisy window |
| FR-18 | All models must output a vector of size 10 (predicted clean window) |

### 5.4 Training

| ID | Requirement |
|---|---|
| FR-19 | Loss function: Mean Squared Error (MSE) between predicted and clean target window |
| FR-20 | Training must be configurable: epochs, batch size, learning rate, optimizer |
| FR-21 | Training loss and validation loss must be logged per epoch |
| FR-22 | Best model checkpoint must be saved per model type |

### 5.5 Evaluation & Experiments

| ID | Requirement |
|---|---|
| FR-23 | Evaluate all three models on held-out test set |
| FR-24 | Report MSE per model, per selected frequency |
| FR-25 | Run experiments varying noise level (alpha, beta) |
| FR-26 | Produce visualizations: predicted vs. clean signal windows, training curves, MSE comparison bar charts |
| FR-27 | Analysis notebook must include comparative discussion with conclusions |

---

## 6. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-01 | Every Python file must not exceed 150 lines of code (blanks and comment-only lines excluded) |
| NFR-02 | Package manager: `uv` exclusively — no pip, no venv |
| NFR-03 | Linter: `ruff check` must pass with zero violations |
| NFR-04 | Test coverage: ≥ 85% (enforced via `pytest --cov`) |
| NFR-05 | TDD workflow: tests written before or alongside implementation |
| NFR-06 | No hard-coded values in source code — all configurable values in `config/setup.json` |
| NFR-07 | No secrets in source code; use `.env` / environment variables |
| NFR-08 | SDK architecture: all business logic accessible through the SDK layer |
| NFR-09 | Code style: descriptive names, docstrings on every function/class/module, comments explain WHY |
| NFR-10 | Reproducibility: random seeds must be configurable and set at experiment start |

---

## 7. User Stories

| ID | As a... | I want to... | So that... |
|---|---|---|---|
| US-01 | Student | Generate 4 sinusoids with configurable parameters | I can control the experiment setup |
| US-02 | Student | Build a dataset of windowed samples with one-hot selectors | I can train and evaluate all three models |
| US-03 | Student | Train an FC, RNN, and LSTM model on the same dataset | I can perform a fair comparison |
| US-04 | Student | Visualize predicted vs. clean signal windows | I can verify the model is extracting the correct component |
| US-05 | Student | Run experiments with different noise levels | I can analyze robustness of each architecture |
| US-06 | Lecturer | Read a detailed README with graphs and conclusions | I can assess the depth of analysis and understanding |
| US-07 | Researcher | Import the SDK and run signal generation or model inference | I can reuse components without touching internals |

---

## 8. Usage Scenarios

### Scenario A — Training a Model
```
uv run python src/main.py --model lstm --epochs 50 --lr 0.001
```
1. Signals are generated with parameters from `config/setup.json`.
2. Dataset is built (random windows + selectors).
3. LSTM model is trained; loss logged per epoch.
4. Best checkpoint saved to `results/`.

### Scenario B — Running Comparative Experiment
```
uv run python src/main.py --experiment compare_all
```
1. All three models trained sequentially.
2. Test MSE computed per model and per frequency.
3. Bar chart saved to `assets/`.

### Scenario C — Analysis Notebook
```
uv run jupyter notebook notebooks/analysis.ipynb
```
1. Load saved results from `results/`.
2. Reproduce all plots; run sensitivity analysis.

---

## 9. Assumptions

- The four sinusoidal frequencies are fixed for the entire experiment (train and test use the same four frequencies; only window position and noise vary between samples).
- Noise is stationary: alpha and beta are constant per experiment run.
- The context window size (10 samples) is fixed as specified by the lecturer.
- Either uniform or Gaussian noise distribution is acceptable; Gaussian is chosen for this project (rationale documented in `docs/PRD_signal_generation.md`).

---

## 10. Constraints

- Context window = 10 samples (fixed by lecturer).
- Sampling rate = 1000 Hz; signal duration = 10 seconds (fixed by lecturer).
- Number of sinusoids = 4 (fixed by lecturer).
- Every Python source file ≤ 150 lines of code.
- Package manager = `uv` only.

---

## 11. Out of Scope

- Real-time signal processing or streaming input.
- Signals other than sinusoids (e.g., square waves, chirps).
- More than 4 sinusoidal components.
- Transformer or attention-based architectures (not mentioned in the assignment).
- GUI application.
- Deployment to production infrastructure.

---

## 12. Dependencies

| Dependency | Purpose |
|---|---|
| Python ≥ 3.10 | Runtime |
| PyTorch | Neural network implementation (FC, RNN, LSTM) |
| NumPy | Signal generation and array operations |
| Matplotlib / Seaborn | Visualization |
| Jupyter | Analysis notebook |
| pytest + pytest-cov | Testing and coverage |
| ruff | Linting |
| uv | Package management |

---

## 13. Timeline & Milestones

| Milestone | Deliverable | Status |
|---|---|---|
| M1 — Documentation | PRD.md, PLAN.md, TODO.md, PRD_*.md all complete | In Progress |
| M2 — Signal Generation | `signal_generator` module passing all unit tests | Pending |
| M3 — Dataset | `dataset` module passing all unit tests | Pending |
| M4 — Models | FC, RNN, LSTM implemented and unit-tested | Pending |
| M5 — Training Pipeline | Training loop, logging, checkpointing complete | Pending |
| M6 — Experiments | All comparative experiments run; results saved | Pending |
| M7 — Analysis & README | Notebook complete; README with graphs and conclusions | Pending |

---

## 14. Acceptance Criteria (Definition of Done)

- [ ] All 10 signal vectors generated correctly and verified against formula.
- [ ] Dataset builder produces samples matching the `[C | σ_noisy]` → `[Sc]` format.
- [ ] FC, RNN, LSTM models train without errors and converge (loss decreasing).
- [ ] MSE results for all models documented and compared in the analysis notebook.
- [ ] At least 2 noise-level experiments documented with conclusions.
- [ ] `ruff check` passes with 0 violations.
- [ ] `pytest --cov` reports ≥ 85% coverage.
- [ ] All Python files ≤ 150 lines of code.
- [ ] README.md contains screenshots, graphs, training curves, and comparative conclusions.
- [ ] All mandatory documentation files exist: `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md`, `docs/PRD_signal_generation.md`, `docs/PRD_ml_models.md`.
