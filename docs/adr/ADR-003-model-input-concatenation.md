# ADR-003 — Model Input: Concatenation of C and Noisy Window

**Status:** Accepted
**Date:** 2026-05-06

## Context

ASSIGNMENT.txt section 11 defines the input format as `[C][sigma_noisy]` — the one-hot selector vector concatenated with the noisy sum window. A design decision is required on how to feed this to RNN and LSTM models, which expect sequential input.

## Decision

Concatenate the 4-element one-hot selector C with the 10-element noisy window into a single 14-element vector. Feed this as a **single time-step** (seq_len=1, input_size=14) to the RNN and LSTM models.

## Consequences

**Positive:**
- All three models (FC, RNN, LSTM) receive identical input format, enabling a fair architectural comparison with no input-preprocessing advantage for any model.
- Directly implements the `[C][sigma_noisy]` format from the assignment specification.
- Keeps the dataset and DataLoader simple — one tensor shape fits all models.

**Negative:**
- With seq_len=1, the RNN/LSTM's temporal processing advantage is not fully exploited. The recurrent layer acts more like a parameterised non-linear transform than a sequence model.
- A richer alternative (feeding 10 samples one-by-one as a length-10 sequence) would make temporal models stronger, but would require a different input representation not specified in the assignment.

**Alternatives rejected:** Feeding the 10 window samples sequentially (seq_len=10, input_size=5 per step with C prepended) — not explicitly defined in ASSIGNMENT.txt, deviates from the `[C][sigma_noisy]` spec.
