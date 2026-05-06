"""
Immutable physical and problem constants for the signal extraction project.

These values are fixed by the assignment specification and must never be
overridden via configuration. Tunable hyperparameters belong in config/setup.json.
"""

# Sampling parameters — fixed by lecturer specification
SAMPLE_RATE: int = 1000       # Hz — samples per second
DURATION: float = 10.0        # seconds — total signal length
N_SAMPLES: int = int(SAMPLE_RATE * DURATION)  # 10,000 discrete samples

# Dataset structure — fixed by lecturer specification
WINDOW_SIZE: int = 10         # context window length in samples
N_SIGNALS: int = 4            # number of sinusoidal components

# Input/output dimensions derived from the above constants
SELECTOR_SIZE: int = N_SIGNALS                    # one-hot vector length
INPUT_SIZE: int = SELECTOR_SIZE + WINDOW_SIZE     # 4 + 10 = 14
OUTPUT_SIZE: int = WINDOW_SIZE                    # 10

# Supported noise distributions
NOISE_DIST_GAUSSIAN: str = "gaussian"
NOISE_DIST_UNIFORM: str = "uniform"
SUPPORTED_NOISE_DISTS: tuple[str, ...] = (NOISE_DIST_GAUSSIAN, NOISE_DIST_UNIFORM)

# Config version expected at runtime
EXPECTED_CONFIG_VERSION: str = "1.00"
