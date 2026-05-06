# ADR-004 — Loss Function: Mean Squared Error (MSE)

**Status:** Accepted
**Date:** 2026-05-06

## Context

ASSIGNMENT.txt section 13 explicitly states: "minimizing the squared error between predicted output window and clean target window." A loss function must be selected for training all three models.

## Decision

Use `torch.nn.MSELoss()` as the sole loss function for all models.

## Consequences

**Positive:**
- Directly implements the assignment specification — no interpretation required.
- MSE penalises large deviations quadratically, which is appropriate for signal reconstruction where large errors are disproportionately harmful to signal fidelity.
- Differentiable everywhere; well-behaved gradient for Adam optimiser.
- MSE is equivalent to Maximum Likelihood Estimation under a Gaussian noise assumption, which is consistent with ADR-001.

**Negative:**
- MSE is sensitive to outliers; a noisy sample with a large spike can dominate the gradient. Mitigated by the moderate alpha/beta values in config.
- MAE (L1) would be more robust to outliers but was not specified by the lecturer.

**Alternatives rejected:**
- MAE (`nn.L1Loss`) — more robust but not specified.
- Huber loss — good compromise but adds a hyperparameter and was not specified.
