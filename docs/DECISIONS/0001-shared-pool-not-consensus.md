# ADR 0001 — Shared capital pool, not consensus signal averaging

Status: Accepted (pre-existing architecture; documented here)

## Context

Multiple models analyze the same markets. A naive design averages their LONG/FLAT/SHORT
outputs into one position per symbol.

## Decision

Models do **not** vote into a consensus signal. Each model emits independent **trade proposals**
that compete for a **shared capital pool**. A central allocator accepts/rejects/resizes/shadows
each proposal; accepted trades remain model-owned with full PnL attribution.

## Consequences

- Edge and blame are attributable per model/trade (`capital_allocator.py`, `ledger.py`).
- Diversity is an asset, not noise to be averaged away.
- Forbidden: averaging signals into a global position; fixed per-model capital budgets as the
  core design; letting models place/cancel orders or close another model's trade.
