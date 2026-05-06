# ADR-007 — Fixed Context Window Size = 10

**Status:** Accepted
**Date:** 2026-05-06

## Context

ASSIGNMENT.txt section 9 explicitly states: "Context Window = 10 samples." This must be implemented as a fixed constraint, but a design decision is needed on whether to make it configurable via `config/setup.json` or treat it as a physical constant.

## Decision

Define `WINDOW_SIZE = 10` as an immutable constant in `src/signal_extraction/constants.py`. It is **not** placed in `config/setup.json`.

## Consequences

**Positive:**
- A constant in `constants.py` communicates to readers that this value is a problem-definition constraint, not a tunable hyperparameter — it cannot be accidentally changed via config.
- Avoids the risk of a config edit silently invalidating all model input/output shapes (INPUT_SIZE=14, OUTPUT_SIZE=10).
- Consistent with SOFTWARE_PROJECT_GUIDELINES.md section 6.2: "physical/mathematical constants" are allowed in code.

**Negative:**
- Experimentation with different window sizes requires a code change rather than a config edit. Acceptable given the assignment fixes this value.

**Alternatives rejected:** Placing `window_size` in `config/setup.json` alongside other dataset parameters — misleadingly implies it is a tunable hyperparameter, risks shape mismatches if changed without updating model architectures.
