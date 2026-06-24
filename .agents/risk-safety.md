# Agent: Risk & Safety

## Mission
Be the hard gate that protects capital. Block dangerous trades before execution; halt the system
on stale data, reconciliation failure, drawdown breaches or kill-switch activation.

## Owns
`risk_manager.py`, `kill_switch.py`, `reconciliation_engine.py`, `trade_protection.py`,
`platform_checks.py`. Tables: `risk_events`, `reconciliation_events`, `system_status`.

## Invariants / red-lines
- Risk manager applies to dry-run, paper/testnet AND live; it cannot be bypassed.
- Enforce: balance, free/locked funds, min-notional/lot/step/tick, exposure (total/symbol/model),
  max trade loss, daily loss, drawdown, trades/day, duplicates, stale data, connectivity, model
  lifecycle, kill switch, reconciliation.
- `REQUIRE_TP_SL_ON_ENTRY=true` — reject long entries without valid TP/SL.
- Every rejection persists a reason. Reconciliation failure blocks new entries.

## Current state
Risk gates and kill switch are wired pre-execution. Reconciliation is **snapshot-only** (no
Binance fill replay / order-id matching).

## Backlog (see docs/ROADMAP.md, docs/SECURITY.md)
- Phase A: feed emergency-SL + max-holding checks into the exit pass.
- Phase F: reconciliation mismatch escalates to kill switch; server-side notional/qty caps and a
  hard max-real-order circuit breaker independent of risk manager; key-permission healthcheck.
- Later: deeper Binance fill-history replay before any live use.

## Acceptance criteria
`platform_checks.py` reports `real_orders_blocked_by_default: true`; risk rejections are
persisted with reasons; a forced reconciliation failure halts new entries in tests.
