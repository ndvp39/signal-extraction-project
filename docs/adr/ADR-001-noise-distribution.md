# ADR-001 — Noise Distribution: Gaussian

**Status:** Accepted
**Date:** 2026-05-06

## Context

The assignment (ASSIGNMENT.txt section 5) specifies that noise must be added to amplitude and phase of each sinusoid, and states that either uniform or Gaussian distribution is acceptable.

## Decision

Use Gaussian (Normal) distribution: ε ~ N(0, 1), scaled by the coefficients alpha (amplitude) and beta (phase).

## Consequences

**Positive:**
- Gaussian noise is the standard model for thermal and electronic noise, justified by the Central Limit Theorem (sum of many independent random disturbances converges to Gaussian).
- Unbounded tails stress-test the model more than uniform noise, producing a harder and more realistic benchmark.
- Standard choice in signal processing literature — makes results comparable to published work.

**Negative:**
- Occasional large outliers (controlled by alpha and beta) may destabilise training if coefficients are set too high.
- Uniform noise would have been simpler to reason about (bounded range).

**Alternatives rejected:** Uniform distribution — simpler but less physically realistic.
