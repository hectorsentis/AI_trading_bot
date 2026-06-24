# ADR 0002 — Acceptance is gated on the full trade lifecycle, not one-bar return

Status: Accepted (recent "architecture correction" commit)

## Context

Models were previously accepted/rejected on one-bar `fwd_return_1` signal metrics, which do not
reflect fees, slippage, sizing, risk rejection, or how a trade actually opens and closes.

## Decision

Validation and backtest route every prediction through the **same** proposal -> allocation ->
risk -> trade-builder -> fills -> exit -> ledger lifecycle used by paper/live, via
`historical_trade_simulator.py`. Acceptance (`strategy_evaluator.py`) reads
`trade_lifecycle.metrics` (total return, Sharpe, profit factor, drawdown, trade count). One-bar
`fwd_return_1` is retained for diagnostics only.

## Consequences

- A model is judged on complete, cost-aware trades — the same logic it will run in production.
- The only difference across research/paper/live is the execution venue.
- `missing_trade_lifecycle_evaluation` is a hard rejection reason.
