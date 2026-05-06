# ADR-006 — Selector C as One-Hot Vector (No Embedding)

**Status:** Accepted
**Date:** 2026-05-06

## Context

ASSIGNMENT.txt section 8 defines the selector as a one-hot vector C of size 4. An alternative is to learn an embedding vector for each of the 4 sinusoid indices, which could capture richer representations.

## Decision

Use the one-hot vector C directly as defined by the assignment. No embedding layer is added.

## Consequences

**Positive:**
- Exactly implements the assignment specification without interpretation.
- One-hot directly encodes which sinusoid to extract — interpretable by both the model and the reader.
- With only 4 classes, an embedding layer would add negligible representational benefit.
- Keeps input dimension fixed at 14 (4 + 10), consistent across all models and tests.

**Negative:**
- An embedding could theoretically learn a richer representation of the frequency relationship between sinusoids (e.g., proximity in frequency space). This is unexplored.

**Alternatives rejected:** Learnable embedding of size D for the 4 classes — adds parameters and complexity without assignment justification; changes INPUT_SIZE from 14, requiring config and test updates.
