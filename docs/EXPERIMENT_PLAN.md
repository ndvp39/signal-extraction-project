# EXPERIMENT_PLAN.md — MSc-Level Experimental Design

**Version:** 1.00
**Author:** MSC Student
**Date:** 2026-05-06
**Status:** Approved — ready for execution

---

## 0. Overview and Research Questions

This plan governs all experiments for the signal extraction project. Five research
questions drive the experimental design:

| RQ | Question |
|---|---|
| RQ-1 | Which architecture (FC, RNN, LSTM) achieves lowest MSE on the clean extraction task? |
| RQ-2 | How does increasing amplitude/phase noise degrade each model's performance, and at what noise level does extraction "break"? |
| RQ-3 | Is the performance gap between recurrent and fully-connected models attributable to architecture or to parameter count? |
| RQ-4 | How sensitive is each architecture to hidden size and depth? |
| RQ-5 | Does RNN suffer from vanishing gradients (higher MSE on high-frequency signals: 120 Hz, 300 Hz) relative to LSTM? |

---

## 1. Baseline Comparison — EXP-01

**Research question:** RQ-1

### 1.1 Objective
Train FC, RNN, and LSTM once under identical, default hyperparameters and compare
overall MSE and per-frequency MSE on the held-out test set.

### 1.2 Configuration (default `config/setup.json`)

| Parameter | Value |
|---|---|
| `n_samples` | 50 000 |
| `epochs` | 100 |
| `batch_size` | 256 |
| `learning_rate` | 0.001 |
| `hidden_size` | 128 |
| `n_layers` | 2 |
| `dropout` | 0.0 |
| `alpha` | 0.1 |
| `beta` | 0.1 |
| Seeds | 42, 123, 777 |

### 1.3 Protocol

1. For each seed in {42, 123, 777}:
   a. Generate a fresh `SignalBundle` with that seed.
   b. Build dataset (train 70 % / val 15 % / test 15 %).
   c. Train FC, RNN, LSTM using identical loaders.
   d. Record `TrainResult` (all epoch losses) and `EvalResult` (overall + per-freq MSE).
2. Aggregate results: compute mean ± std across the 3 seeds for every metric.

### 1.4 Metrics

| Metric | Symbol | Description |
|---|---|---|
| Overall test MSE | `MSE_overall` | Mean over all 4 output frequencies |
| Per-frequency MSE | `MSE_f` | Separate value for 10 Hz, 50 Hz, 120 Hz, 300 Hz |
| Best epoch | `epoch*` | Epoch at which early stopping triggered |
| Final val loss | `L_val` | Validation loss at `epoch*` |

### 1.5 Expected Outcomes

- LSTM ≥ RNN > FC on `MSE_overall` (literature prior: recurrent models handle temporal
  structure in the window better than flat FC).
- FC may struggle most with 300 Hz (highest frequency, shortest period relative to
  window size).
- RNN may show higher `MSE_f` for 300 Hz than LSTM due to vanishing gradients.

### 1.6 Output Files

```
results/baseline/
    seed_42/   {fc,rnn,lstm}/  checkpoint.pt, metrics.json
    seed_123/  ...
    seed_777/  ...
    summary.json          # mean ± std per model × metric
assets/
    training_curves.png
    mse_comparison.png
```

---

## 2. Noise Robustness Sweep — EXP-02

**Research question:** RQ-2

### 2.1 Objective
Quantify how MSE scales with amplitude noise `alpha` and phase noise `beta` for each
architecture. Find the "breaking point" beyond which MSE exceeds an acceptable
threshold (defined as 10× baseline `MSE_overall`).

### 2.2 Noise Grid

Two independent 1-D sweeps (all other parameters fixed at default):

**Amplitude noise sweep** (beta fixed = 0.1):

| `alpha` | Interpretation |
|---|---|
| 0.00 | No amplitude noise (control) |
| 0.05 | Low noise |
| 0.10 | Baseline (default) |
| 0.20 | Moderate noise |
| 0.40 | High noise |
| 0.80 | Severe noise (amplitude ± 80 %) |

**Phase noise sweep** (alpha fixed = 0.1):

| `beta` | Interpretation |
|---|---|
| 0.00 | No phase noise (control) |
| 0.05 | Low noise |
| 0.10 | Baseline (default) |
| 0.20 | Moderate noise |
| 0.40 | High noise |
| 0.80 | Severe noise |

### 2.3 Protocol

1. For each `(sweep, noise_value)` pair and each `seed` in {42, 123, 777}:
   - Generate a dedicated config override (only `alpha` or `beta` changes).
   - Train all three models with identical loaders.
   - Record `MSE_overall` and `MSE_f` per model.
2. Aggregate: mean ± std over seeds.

### 2.4 Metrics

Same as EXP-01 plus:

| Metric | Description |
|---|---|
| `MSE_delta` | Relative MSE increase vs baseline: `(MSE - MSE_baseline) / MSE_baseline` |
| Breaking-point `alpha*` | Smallest alpha where `MSE_delta > 10` for any model |
| Breaking-point `beta*` | Same for beta |

### 2.5 Output Files

```
results/noise_sweep/
    alpha/  alpha_0.00/ … alpha_0.80/  {fc,rnn,lstm}/  metrics.json
    beta/   beta_0.00/  … beta_0.80/   {fc,rnn,lstm}/  metrics.json
    summary_alpha.json
    summary_beta.json
assets/
    noise_heatmap.png          # model × noise_level heatmap of MSE_overall
    noise_curves_alpha.png     # MSE vs alpha line chart (3 models)
    noise_curves_beta.png      # MSE vs beta line chart (3 models)
```

---

## 3. Parameter-Matched Fair Comparison — EXP-03

**Research question:** RQ-3

### 3.1 Objective
FC, RNN, and LSTM have very different parameter counts at the same `hidden_size`.
Matching parameter counts removes the confound: any remaining performance gap is
attributable to architecture, not capacity.

### 3.2 Parameter Count Analysis

With `hidden_size = H` and `n_layers = 1`:

| Architecture | Approx. total parameters |
|---|---|
| FC | `14·H + H·H + H·10 + biases` ≈ `H² + 24H` |
| RNN | `14·H + H·H + H·10 + biases` ≈ `H² + 24H` |
| LSTM | `4·(14·H + H·H) + H·10 + biases` ≈ `4H² + 66H` |

At `H=128, L=1`: FC ≈ 18 060 params, RNN ≈ 18 060, LSTM ≈ 68 874.

**Target:** match LSTM(H=64) ≈ FC(H=128) ≈ RNN(H=128).

### 3.3 Hidden-Size Configurations for Matching

| Model | `hidden_size` | Approx. params (L=1) |
|---|---|---|
| FC | 128 | ~18 060 |
| RNN | 128 | ~18 060 |
| LSTM | 64 | ~17 802 |

These three configurations have ≈ equal total parameter count within ±2 %.

### 3.4 Protocol

Run EXP-01 protocol with the parameter-matched configs above. Compare
`MSE_overall` between architectures at matched capacity vs. unmatched (EXP-01
result). If LSTM still outperforms FC at matched capacity, architecture drives the
gain.

### 3.5 Output Files

```
results/param_matched/
    seed_42/ … seed_777/  {fc_h128, rnn_h128, lstm_h64}/  metrics.json
    summary.json
assets/
    param_matched_comparison.png   # bar chart: matched vs unmatched MSE per architecture
```

---

## 4. Hidden-Size Sensitivity — EXP-04

**Research question:** RQ-4

### 4.1 Objective
Quantify how `MSE_overall` changes as `hidden_size` increases from under-fitted to
over-fitted regime. Identify the point of diminishing returns.

### 4.2 Hidden-Size Grid

Values: {16, 32, 64, 128, 256, 512} — spans from clearly under-parameterised to
clearly over-parameterised.

Fixed: `n_layers = 2`, `alpha = 0.1`, `beta = 0.1`, seeds = {42, 123, 777}.

### 4.3 Protocol

Train all three models at each `hidden_size`. Record `MSE_overall`, total parameter
count, and training time per epoch.

### 4.4 Output Files

```
results/hidden_size/
    h16/ … h512/  {fc,rnn,lstm}/  metrics.json
    summary.json
assets/
    sensitivity_hidden_size.png    # MSE vs hidden_size (3 models, error bars)
```

---

## 5. Depth Sensitivity — EXP-05

**Research question:** RQ-4

### 5.1 Objective
Quantify how `MSE_overall` changes as `n_layers` increases from 1 to 4.

### 5.2 Layer Grid

Values: {1, 2, 3, 4} — fixed `hidden_size = 128`.

### 5.3 Output Files

```
results/n_layers/
    L1/ … L4/  {fc,rnn,lstm}/  metrics.json
assets/
    sensitivity_n_layers.png
```

---

## 6. Learning-Rate Sweep — EXP-06

**Research question:** RQ-4 (optimizer sensitivity)

### 6.1 LR Grid

Values: {0.0001, 0.0003, 0.001, 0.003, 0.01} — one decade either side of the
default (0.001).

Fixed: `hidden_size = 128`, `n_layers = 2`, seed = 42 only (LR sweep is a
diagnostic, not a primary result; single seed suffices).

### 6.2 Output Files

```
results/lr_sweep/
    lr_1e-4/ … lr_1e-2/  {fc,rnn,lstm}/  metrics.json
assets/
    sensitivity_lr.png
```

---

## 7. Frequency-Specific MSE Analysis — EXP-07

**Research question:** RQ-5 (vanishing gradient in RNN vs LSTM)

### 7.1 Objective
Compare `MSE_f` at each of the four frequencies {10, 50, 120, 300} Hz across
architectures. The hypothesis is that:
- RNN degrades more sharply at 300 Hz (period = 3.3 samples, within the 10-sample
  window this is 3 full cycles — tightest temporal pattern, most sensitive to
  gradient degradation).
- LSTM gates should preserve this high-frequency information better.

### 7.2 Metric of Interest

`MSE_f_ratio = MSE_f(300 Hz) / MSE_f(10 Hz)` per model.

A ratio >> 1 for RNN but ≈ 1 for LSTM would confirm the vanishing-gradient
hypothesis in this task.

### 7.3 Protocol

Use EXP-01 (baseline) results — no additional training needed. Extract `mse_per_freq`
from each `EvalResult` and compute the ratio.

### 7.4 Output Files

```
assets/
    per_freq_mse.png    # grouped bar chart: frequency × model × MSE
```

---

## 8. Statistical Rigor Protocol

All experiments that produce a primary result (EXP-01, EXP-02, EXP-03, EXP-04,
EXP-05) **must** use 3 seeds: {42, 123, 777}.

### 8.1 Seeds

| Seed | Purpose |
|---|---|
| 42 | Primary (used alone in diagnostic sweeps EXP-06) |
| 123 | Secondary |
| 777 | Tertiary |

### 8.2 Reporting Format

For every metric `M` over seeds `{s1, s2, s3}`:

```
mean(M) ± std(M)
```

95 % confidence interval (normal approximation, n=3):
```
CI = mean ± 1.96 · std / sqrt(3)
```

### 8.3 Summary JSON Schema

Every experiment must produce a `summary.json` with the following structure:

```json
{
  "experiment": "EXP-01",
  "models": {
    "fc":   { "mse_overall": { "mean": 0.0, "std": 0.0 },
               "mse_per_freq": { "0": {"mean":0.0,"std":0.0}, ... } },
    "rnn":  { ... },
    "lstm": { ... }
  },
  "seeds": [42, 123, 777]
}
```

---

## 9. Execution Order

| Step | Experiment | Prerequisite | Est. runs |
|---|---|---|---|
| 1 | EXP-01 Baseline | None | 9 (3 models × 3 seeds) |
| 2 | EXP-07 Freq analysis | EXP-01 results | 0 (post-hoc) |
| 3 | EXP-03 Param-matched | None | 9 |
| 4 | EXP-02 Noise sweep | None | 108 (6+6 noise × 3 models × 3 seeds) |
| 5 | EXP-04 Hidden size | None | 54 (6 sizes × 3 models × 3 seeds) |
| 6 | EXP-05 Depth | None | 36 (4 depths × 3 models × 3 seeds) |
| 7 | EXP-06 LR sweep | None | 15 (5 LRs × 3 models × 1 seed) |
| **Total** | | | **231 training runs** |

---

## 10. Visualisation Checklist

| Asset file | Source experiment | TODO ID |
|---|---|---|
| `assets/training_curves.png` | EXP-01 | T-080 |
| `assets/mse_comparison.png` | EXP-01 + EXP-07 | T-081 |
| `assets/signal_examples.png` | EXP-01 | T-082 |
| `assets/signal_overview.png` | Signal generator | T-083 |
| `assets/noise_heatmap.png` | EXP-02 | T-084 |
| `assets/sensitivity_hidden_size.png` | EXP-04 | T-085 |
| `assets/sensitivity_n_layers.png` | EXP-05 | T-085 |
| `assets/sensitivity_lr.png` | EXP-06 | T-085 |
| `assets/per_freq_mse.png` | EXP-07 | T-081 |
| `assets/param_matched_comparison.png` | EXP-03 | T-081 |
| `assets/noise_curves_alpha.png` | EXP-02 | T-084 |
| `assets/noise_curves_beta.png` | EXP-02 | T-084 |

---

## 11. Scope Constraints

- All experiments use the existing `SignalExtractionSDK` API — no new public methods.
- Config overrides are passed via temporary JSON files written to `config/` before
  each run; the originals are never modified.
- No experiment modifies `constants.py` (WINDOW_SIZE, N_SIGNALS remain fixed).
- Results are deterministic given seed — re-running any experiment with the same seed
  must produce the same checkpoint.
