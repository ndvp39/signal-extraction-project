# PRD — Machine Learning Models

**Version:** 1.00
**Author:** MSC Student
**Date:** 2026-05-06
**Component:** `models/` (fc_model, rnn_model, lstm_model, base_model)
**Status:** Approved

---

## 1. Overview & Theoretical Background

### 1.1 Problem Framing

This is a **supervised regression** problem. Given:
- A one-hot selector `C ∈ {0,1}^4`
- A noisy composite signal window `σ[i:i+10] ∈ ℝ^10`

Predict:
- The clean sinusoidal component window `Sc[i:i+10] ∈ ℝ^10`

The input dimension is `4 + 10 = 14`. The output dimension is `10`.

### 1.2 Loss Function

Mean Squared Error (MSE):

```
L = (1/10) · Σ_{k=0}^{9} (ŷ_k - y_k)²
```

Where `ŷ` is the predicted window and `y` is the clean target window. MSE penalises large deviations quadratically, which is appropriate for signal reconstruction where large errors are disproportionately harmful.

---

### 1.3 Fully Connected (FC) Network

A feedforward network with no temporal structure. Each output neuron is a function of all input neurons simultaneously.

```
Architecture:
  x ∈ ℝ^14
  → Linear(14 → H) → ReLU
  → Linear(H → H)  → ReLU
  → Linear(H → 10)
  → ŷ ∈ ℝ^10
```

**Parameter count:**
```
P_FC = (14·H + H) + (H·H + H) + (H·10 + 10)
     = 14H + H² + H + H·10 + H + 10 + H
     ≈ H² + 27H + 10       (for large H)
```

**Strengths:**
- Simple, fast to train.
- No issues with vanishing gradients across time.
- Good baseline to beat.

**Weaknesses:**
- No temporal awareness — treats the 10 input samples as an unordered set.
- Cannot exploit the periodic structure of sinusoids across the window.
- Performance expected to degrade at lower frequencies where the window captures less than one full cycle.

**Expected behaviour:** Best at high frequencies where 10 samples capture multiple cycles, giving the network sufficient local pattern. Worst at very low frequencies (10 Hz → one cycle = 100 samples; 10-sample window is a tiny fraction of the period).

---

### 1.4 Recurrent Neural Network (RNN)

A network with a hidden state that is updated at each time step, creating a directed cycle that allows information to persist:

```
h_t = tanh(W_ih · x_t + b_ih + W_hh · h_{t-1} + b_hh)
```

**Architecture used:**
```
  x ∈ ℝ^14
  → reshape to (1, 14)          # single time-step sequence
  → RNN(input=14, hidden=H, layers=L, nonlinearity='tanh')
  → take last hidden state h ∈ ℝ^H
  → Linear(H → 10)
  → ŷ ∈ ℝ^10
```

**Note on sequence treatment:** The model receives the entire 10-sample window as the input vector (not as 10 sequential steps). This is because the assignment defines the context window as a single sample in the dataset. To leverage the RNN's temporal processing, a future extension could feed the 10 samples one-by-one (sequence length 10, input dim 14/10). This design decision is documented in ADR-03.

**Vanishing Gradient Problem:**
```
∂L/∂h_0 = ∂L/∂h_T · Π_{t=1}^{T} ∂h_t/∂h_{t-1}
         = ∂L/∂h_T · Π_{t=1}^{T} W_hh · diag(1 - h_t²)
```

When `||W_hh|| < 1`, the product shrinks exponentially. For long sequences this causes gradients to vanish. With a context window of 10 this is manageable, but the RNN still struggles with long-range dependencies relative to LSTM.

**Strengths:** Captures short-term temporal patterns; fewer parameters than LSTM.

**Weaknesses:** Vanishing gradients; cannot remember information across long sequences.

---

### 1.5 Long Short-Term Memory (LSTM)

LSTM introduces a **cell state** `c_t` and three **gates** that control information flow, solving the vanishing gradient problem:

```
Forget gate:   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input gate:    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Cell update:   g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)
Output gate:   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

Cell state:    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
Hidden state:  h_t = o_t ⊙ tanh(c_t)
```

Where `σ` = sigmoid, `⊙` = element-wise product.

**Architecture used:**
```
  x ∈ ℝ^14
  → reshape to (1, 14)
  → LSTM(input=14, hidden=H, layers=L)
  → take last hidden state h ∈ ℝ^H
  → Linear(H → 10)
  → ŷ ∈ ℝ^10
```

**Why LSTM solves vanishing gradients:**
The cell state `c_t` has a gradient path through the forget gate only:
```
∂c_t/∂c_{t-1} = f_t
```
Since `f_t ∈ (0,1)` and is learned, the network can set `f_t ≈ 1` to preserve gradients, unlike the RNN whose gradient path is multiplicative through `W_hh`.

**Parameter count (per layer):**
```
P_LSTM ≈ 4 · (H·H + H·input_size + H)
```
Approximately 4× more parameters than a comparable RNN.

**Strengths:** Long-term memory; stable gradients; suitable for lower-frequency sinusoids where temporal structure spans more samples.

**Weaknesses:** More parameters → slower training; may overfit on small datasets; heavier compute.

---

## 2. Model Interface

### 2.1 `BaseModel` (Abstract Base Class)

```python
class BaseModel(nn.Module, ABC):
    """
    Abstract base for all signal extraction models.

    All subclasses must implement forward() accepting a concatenated
    input tensor [C | noisy_window] and returning a predicted clean window.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor, shape (batch_size, 14)
               Columns 0–3:  one-hot selector C
               Columns 4–13: noisy sum window
        Returns:
            Float tensor, shape (batch_size, 10)
            Predicted clean sinusoid window
        """
```

### 2.2 `FCModel`

```python
class FCModel(BaseModel):
    """
    Fully Connected baseline model.
    No temporal structure; all input dimensions treated equally.
    """
    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

### 2.3 `RNNModel`

```python
class RNNModel(BaseModel):
    """
    Vanilla RNN model. Input treated as single-step sequence.
    Suitable for short-term temporal patterns.
    """
    def __init__(self, hidden_size: int, n_layers: int, dropout: float = 0.0) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

### 2.4 `LSTMModel`

```python
class LSTMModel(BaseModel):
    """
    LSTM model with cell state for long-term memory.
    Suitable for lower-frequency sinusoids and complex temporal dependencies.
    """
    def __init__(self, hidden_size: int, n_layers: int, dropout: float = 0.0) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

---

## 3. Hyperparameter Choices & Rationale

| Hyperparameter | Chosen Value | Rationale |
|---|---|---|
| `hidden_size` | 128 | Large enough to learn frequency-selective filtering; small enough to train quickly on CPU |
| `n_layers` | 2 | Two layers add representational depth without excessive parameters or gradient issues |
| `dropout` | 0.0 (baseline) | No dropout in baseline; may be added in sensitivity experiments |
| `optimizer` | Adam | Adaptive learning rate handles varying gradient magnitudes across layers; standard choice |
| `learning_rate` | 0.001 | Adam default; well-tested starting point for regression tasks |
| `batch_size` | 256 | Balances gradient variance and training speed; fits comfortably in CPU memory |
| `epochs` | 100 | Sufficient for convergence with early stopping (patience=10) |
| `early_stopping patience` | 10 | Stops training if validation loss does not improve for 10 consecutive epochs |

**Why these were not hard-coded:** All values come from `config/setup.json`. The code reads them via `ConfigManager`. This allows sensitivity experiments without code changes.

---

## 4. Expected Performance by Architecture & Frequency

| Frequency | FC | RNN | LSTM | Reasoning |
|---|---|---|---|---|
| 10 Hz (slow) | Poor | Medium | Best | 10-sample window ≪ one period (100 samples); LSTM memory helps |
| 50 Hz (medium) | Medium | Good | Good | Window ~= half period; temporal models leverage structure |
| 120 Hz (medium-high) | Good | Good | Good | Window ≈ 1.2 periods; all models see meaningful oscillation |
| 300 Hz (fast) | Best | Good | Good | Window = 3 full cycles; rich local pattern; FC handles well |

**Key comparison points:**
- FC is expected to match or beat RNN/LSTM at high frequencies.
- LSTM expected advantage at lower frequencies.
- RNN expected to underperform LSTM consistently due to gradient issues with deeper stacks.

---

## 5. Constraints

| Constraint | Value | Source |
|---|---|---|
| Input dimension | 14 (fixed) | Assignment: `[C(4) | window(10)]` |
| Output dimension | 10 (fixed) | Assignment: `window_size = 10` |
| Loss function | MSE | Assignment specification |
| Model types | FC, RNN, LSTM | Assignment specification |
| File size | ≤ 150 lines | Dr. Segal guidelines |
| No hard-coded hyperparameters | All from config | Dr. Segal guidelines |

---

## 6. Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| Transformer / attention | Not in assignment scope; excessive complexity for 10-sample window |
| 1D CNN | Not mentioned in assignment; could be an extension |
| Bidirectional RNN/LSTM | Would require seeing future samples — not causal; out of scope |
| Embedding for selector C | Assignment specifies one-hot vector explicitly (see ADR-06) |
| Separate models per frequency | Assignment requires a single model handling all 4 via the selector |

---

## 7. Success Criteria & Test Scenarios

| Test ID | Scenario | Expected Result |
|---|---|---|
| ML-T01 | FC forward pass | Output shape `(B, 10)` for input shape `(B, 14)` |
| ML-T02 | RNN forward pass | Output shape `(B, 10)`; no error on batch of any size B ≥ 1 |
| ML-T03 | LSTM forward pass | Output shape `(B, 10)`; no error |
| ML-T04 | All models accept same input | `FCModel`, `RNNModel`, `LSTMModel` produce output from identical input tensor |
| ML-T05 | Parameter count sanity — FC | `sum(p.numel() for p in model.parameters())` matches formula |
| ML-T06 | Loss decreases over 2 epochs | Train 2 epochs on small batch; `loss_epoch2 < loss_epoch1` |
| ML-T07 | Zero-selector edge case | Input `C = [0,0,0,0]` does not crash; output is a valid tensor |
| ML-T08 | Single-sample batch | `B=1` input produces `(1, 10)` output for all model types |
| ML-T09 | Gradient flow — FC | `loss.backward()` runs; all parameter gradients are non-None |
| ML-T10 | Gradient flow — RNN | `loss.backward()` runs; gradients non-None including recurrent weights |
| ML-T11 | Gradient flow — LSTM | `loss.backward()` runs; cell state gradients non-None |
| ML-T12 | Model serialisation | `torch.save` / `torch.load` round-trip produces identical outputs |
| ML-T13 | Config-driven hidden size | Changing `hidden_size` in config changes model parameter count accordingly |
