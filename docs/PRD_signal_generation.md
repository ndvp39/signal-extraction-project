# PRD — Signal Generation Mechanism

**Version:** 1.00
**Author:** MSC Student
**Date:** 2026-05-06
**Component:** `services/signal_generator.py` + `shared/schemas.py`
**Status:** Approved

---

## 1. Overview & Theoretical Background

### 1.1 Sinusoidal Signal Model

A continuous sinusoidal signal is defined as:

```
s(t) = A · sin(2πft + φ)
```

Where:
- `A` — amplitude (peak value of the signal)
- `f` — frequency in Hz (cycles per second)
- `φ` — phase offset in radians
- `t` — continuous time variable

In a digital system, the continuous signal is **sampled** at discrete time points. Given:
- Sampling rate `fs = 1000 Hz`
- Duration `T = 10 seconds`
- Total samples `N = fs × T = 10,000`

The discrete time axis is:

```
t[n] = n / fs,   n = 0, 1, 2, ..., N-1
```

The discrete sinusoid becomes:

```
s[n] = A · sin(2π · f · n/fs + φ)
```

### 1.2 Nyquist-Shannon Sampling Theorem

To faithfully represent a sinusoid of frequency `f`, the sampling rate must satisfy:

```
fs > 2 · f_max
```

With `fs = 1000 Hz`, the maximum representable frequency is `f_max = 500 Hz`. All chosen frequencies must remain well below this limit to avoid aliasing.

**Chosen frequencies:** 10 Hz, 50 Hz, 120 Hz, 300 Hz — all satisfy `f < 500 Hz`.

### 1.3 Composite Signal

The combined signal is the superposition of all four sinusoids:

```
S_sum[n] = S1[n] + S2[n] + S3[n] + S4[n]
```

This represents a realistic multi-tone signal, analogous to a received radio signal containing multiple frequency channels.

### 1.4 Noise Model

Noise is applied independently to each sinusoid's amplitude and phase before summation:

```
S_noisy[n] = (A + α · εA) · sin(2π · f · n/fs + φ + β · εφ)
```

Where:
- `α` — amplitude noise strength coefficient
- `β` — phase noise strength coefficient
- `εA ~ N(0, 1)` — amplitude noise sample (Gaussian)
- `εφ ~ N(0, 1)` — phase noise sample (Gaussian)

**Why Gaussian?** By the Central Limit Theorem, the sum of many independent random disturbances converges to a Gaussian distribution. This is the standard model for thermal noise in electronics and measurement error in sensors. It is more challenging than uniform noise due to unbounded tails (controlled by α, β).

**Why noise on amplitude and phase separately?** In real systems:
- Amplitude noise models signal attenuation variations (fading channels).
- Phase noise models oscillator jitter and Doppler shift.

Each noisy sinusoid gets **independently drawn noise** per sample generation (not per time-step). That is, `εA` and `εφ` are scalar values drawn once per sinusoid per dataset sample, representing a slowly-varying disturbance.

The noisy sum is:

```
σ_noisy[n] = noisy_S1[n] + noisy_S2[n] + noisy_S3[n] + noisy_S4[n]
```

---

## 2. Component Responsibilities

### 2.1 `SignalGeneratorService`

**Single Responsibility:** Generate all 10 signal vectors (clean and noisy) from a `SignalParams` configuration object.

**Does NOT:**
- Build the dataset (that is `DatasetBuilderService`'s responsibility)
- Save signals to disk (that is the SDK's responsibility via FileIO)
- Choose window positions (that is `DatasetBuilderService`)

### 2.2 `SignalParams` Dataclass

Encapsulates all parameters needed to reproduce a signal bundle deterministically.

```python
@dataclass
class SignalParams:
    frequencies: list[float]   # Hz — length must equal N_SIGNALS (4)
    amplitudes:  list[float]   # dimensionless — length must equal N_SIGNALS
    phases:      list[float]   # radians — length must equal N_SIGNALS
    alpha:       float         # amplitude noise strength ≥ 0
    beta:        float         # phase noise strength ≥ 0
    noise_dist:  str           # "gaussian" | "uniform"
    sample_rate: int           # Hz (fixed: 1000)
    duration:    float         # seconds (fixed: 10.0)
    seed:        int           # random seed for reproducibility
```

### 2.3 `SignalBundle` Dataclass

Output of `SignalGeneratorService.generate()`.

```python
@dataclass
class SignalBundle:
    clean: dict[str, np.ndarray]   # keys: "s1","s2","s3","s4","sum" — shape (10000,)
    noisy: dict[str, np.ndarray]   # keys: "s1","s2","s3","s4","sum" — shape (10000,)
    t:     np.ndarray              # time axis — shape (10000,)
    params: SignalParams           # parameters that produced this bundle
```

---

## 3. Functional Requirements

| ID | Requirement |
|---|---|
| SGR-01 | Accept a `SignalParams` object as sole configuration input |
| SGR-02 | Generate time axis `t` of shape `(N,)` where `N = sample_rate × duration` |
| SGR-03 | Generate 4 clean sinusoids `S1–S4` each of shape `(N,)` using `A·sin(2πft + φ)` |
| SGR-04 | Generate clean sum `S_sum = S1 + S2 + S3 + S4` of shape `(N,)` |
| SGR-05 | Generate 4 noisy sinusoids, each with independently drawn amplitude and phase noise |
| SGR-06 | Generate noisy sum `σ_noisy = sum of noisy sinusoids` |
| SGR-07 | Return a `SignalBundle` containing all 10 vectors plus time axis and params |
| SGR-08 | Set `numpy` random seed from `params.seed` before any noise sampling |
| SGR-09 | Support both `"gaussian"` and `"uniform"` noise distributions via `params.noise_dist` |
| SGR-10 | When `alpha=0`, noisy amplitude must equal clean amplitude exactly |
| SGR-11 | When `beta=0`, noisy phase must equal clean phase exactly |
| SGR-12 | Validate that `len(frequencies) == len(amplitudes) == len(phases) == N_SIGNALS` |
| SGR-13 | Validate that all frequencies satisfy `f < sample_rate / 2` (Nyquist condition) |
| SGR-14 | Validate that all amplitudes are positive |

---

## 4. Expected Input / Output

### Input

```python
params = SignalParams(
    frequencies = [10.0, 50.0, 120.0, 300.0],   # Hz
    amplitudes  = [1.0, 0.8, 1.2, 0.6],
    phases      = [0.0, 0.5, 1.0, 1.5],          # radians
    alpha       = 0.1,
    beta        = 0.1,
    noise_dist  = "gaussian",
    sample_rate = 1000,
    duration    = 10.0,
    seed        = 42,
)
```

### Output

```python
bundle = SignalBundle(
    clean = {
        "s1":  np.ndarray, shape=(10000,),  # A1·sin(2π·10·t + 0.0)
        "s2":  np.ndarray, shape=(10000,),
        "s3":  np.ndarray, shape=(10000,),
        "s4":  np.ndarray, shape=(10000,),
        "sum": np.ndarray, shape=(10000,),  # s1+s2+s3+s4
    },
    noisy = {
        "s1":  np.ndarray, shape=(10000,),
        "s2":  np.ndarray, shape=(10000,),
        "s3":  np.ndarray, shape=(10000,),
        "s4":  np.ndarray, shape=(10000,),
        "sum": np.ndarray, shape=(10000,),
    },
    t      = np.ndarray, shape=(10000,),    # [0.0, 0.001, ..., 9.999]
    params = params,
)
```

### Performance Constraints

| Metric | Requirement |
|---|---|
| Generation time | < 1 second for 10 signals × 10,000 samples |
| Memory usage | < 10 MB for full signal bundle |
| Numerical precision | `float64` (NumPy default) |

---

## 5. Constraints & Edge Cases

| Case | Expected Behaviour |
|---|---|
| `alpha = 0, beta = 0` | Noisy signals are identical to clean signals |
| `alpha = 0, beta > 0` | Only phase differs; amplitudes identical |
| Very high `alpha` or `beta` | Signal is generated but may be heavily distorted; no error raised |
| Frequency at or above Nyquist (`f ≥ fs/2`) | `ValueError` raised with message citing Nyquist limit |
| `len(frequencies) != 4` | `ValueError` raised |
| Negative amplitude | `ValueError` raised |
| `noise_dist` not in `{"gaussian", "uniform"}` | `ValueError` raised |

---

## 6. Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| Additive noise on signal value: `s[n] + noise` | Assignment specifies noise on amplitude and phase specifically |
| Noise varying per time-step | Would model non-stationary noise; adds complexity without assignment justification |
| Uniform noise as default | Gaussian chosen as more physically realistic (see ADR-01) |
| Storing signals as 2D array `(4, 10000)` | Dict with string keys is more readable and self-documenting |

---

## 7. Success Criteria & Test Scenarios

| Test ID | Scenario | Expected Result |
|---|---|---|
| SG-T01 | Generate with `alpha=0, beta=0` | `bundle.noisy["s1"] == bundle.clean["s1"]` (element-wise) |
| SG-T02 | Check clean sum | `bundle.clean["sum"] ≈ sum(bundle.clean["s1"…"s4"])` within float64 tolerance |
| SG-T03 | Verify signal shape | All 10 arrays have shape `(10000,)` |
| SG-T04 | Verify time axis | `bundle.t[0] == 0.0`, `bundle.t[-1] ≈ 9.999`, `len(bundle.t) == 10000` |
| SG-T05 | Reproducibility | Two calls with same `seed` produce identical bundles |
| SG-T06 | Different seeds differ | Two calls with different seeds produce different noisy arrays |
| SG-T07 | Nyquist violation | `frequencies=[600]` raises `ValueError` |
| SG-T08 | Formula spot-check | `bundle.clean["s1"][0] ≈ A1 · sin(φ1)` (at `t=0`) |
| SG-T09 | Noise scale | `std(bundle.noisy["s1"] - bundle.clean["s1"]) > 0` when `alpha > 0` or `beta > 0` |
| SG-T10 | Invalid noise_dist | `noise_dist="laplace"` raises `ValueError` |
