# ADR-002 — Deep Learning Framework: PyTorch

**Status:** Accepted
**Date:** 2026-05-06

## Context

A deep learning framework is needed to implement the FC, RNN, and LSTM models required by ASSIGNMENT.txt section 14. The framework must provide native support for all three architectures and integrate well with NumPy for signal processing.

## Decision

Use PyTorch (`torch`, `torch.nn`) as the sole deep learning framework.

## Consequences

**Positive:**
- PyTorch provides native `nn.RNN` and `nn.LSTM` modules with well-documented APIs, reducing implementation risk.
- Tight NumPy interoperability (`tensor.numpy()`, `torch.from_numpy()`) fits the signal-processing workflow.
- Standard framework in MSC-level research settings — grader familiarity expected.
- Dynamic computation graph simplifies debugging of gradient flow issues.

**Negative:**
- Larger installation footprint (~230 MB) compared to lighter alternatives.
- CPU-only training on this machine is slower than GPU-accelerated alternatives.

**Alternatives rejected:**
- TensorFlow/Keras — heavier ecosystem, less transparent recurrent layer internals.
- JAX — too low-level for this scope; no built-in RNN/LSTM modules.
