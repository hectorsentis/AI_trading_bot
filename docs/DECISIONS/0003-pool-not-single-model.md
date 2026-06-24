# ADR 0003 — Maintain a model pool; never collapse to a single retrained model

Status: Accepted

## Context

A recurring question: should the system retrain one model continuously, or keep many?

## Decision

Maintain a **pool** of diverse models keyed by `(symbol, timeframe, horizon, feature_family,
label_family, objective, config)`. Do both continuously: **retrain each slot** on rolling
walk-forward windows (regime drift) **and add new diverse candidates**. **Never delete** models;
rejected/degraded models stay in `model_registry` with reasons. The allocator and lifecycle
manager promote good models and degrade/quarantine bad ones.

## Consequences

- A single model cannot express the diversity the shared-pool allocator exists to exploit.
- Audit is preserved; no silent overwrite of model history.
- Retraining is a rolling-window discipline, not a reason to throw away history.
- See [../MODELING.md](../MODELING.md) for the full rationale.
