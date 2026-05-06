# ADR-005 — SDK Architecture as Single Entry Point

**Status:** Accepted
**Date:** 2026-05-06

## Context

SOFTWARE_PROJECT_GUIDELINES.md section 3.1 mandates: "Every function containing business logic MUST be reachable through the SDK layer. The SDK is the single entry point for all consumers." The project has multiple consumers: CLI (`main.py`), analysis notebook, and unit tests.

## Decision

Implement `SignalExtractionSDK` as the single public interface. All services (`SignalGeneratorService`, `DatasetBuilderService`, `TrainerService`, `EvaluatorService`) are internal. `main.py` and the notebook only call SDK methods.

## Consequences

**Positive:**
- A notebook, CLI, or future REST API can all use the same code path with zero duplication.
- Business logic is unit-testable independently of the consumer layer.
- Enforces separation of concerns: CLI parses arguments, SDK executes logic.
- Changing a service implementation (e.g., switching optimiser) requires no changes to consumers.

**Negative:**
- Adds one layer of indirection; small projects might find it over-engineered.
- SDK methods must be kept in sync with service interfaces as they evolve.

**Alternatives rejected:** Direct service calls from `main.py` — violates the guidelines' single-entry-point requirement and duplicates orchestration logic if a second consumer is added.
