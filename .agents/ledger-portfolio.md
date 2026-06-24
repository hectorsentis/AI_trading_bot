# Agent: Ledger, Portfolio & Exits

## Mission
Keep the books honest and the positions managed. Attribute every PnL component to the right
model/trade, persist equity across runs, and ensure no position stays open forever.

## Owns
`ledger.py`, `portfolio_manager.py`, `exit_manager.py`, `stop_manager.py`. Tables: `trades`,
`positions`, `portfolio_snapshots`, `account_snapshots`, `balance_snapshots`, `model_performance`.

## Invariants / red-lines
- Attribute realized/unrealized PnL, fees and slippage per model/trade via the full ID chain.
- A trade must close on TP / SL / emergency-SL / expected-value deterioration / confidence drop /
  signal expiry / max-holding / risk-forced reduction. A model cannot hold a losing position
  forever by not emitting a sell.
- Never lose attribution when multiple trades share a symbol.

## Current state (critical gaps)
- `exit_manager.evaluate_virtual_exits()` only marks `CLOSING`; it is **never called** by
  `trading_bot.py`, and nothing flattens `CLOSING` trades.
- Equity/PnL is in-memory; snapshots are **not persisted**, so equity resets each run.
- Emergency stops calculated but never monitored.

## Backlog (see docs/ROADMAP.md — Phase A)
- A1: wire exits into the loop; flatten `CLOSING` trades; add emergency-SL + max-holding checks.
- A2: persist `portfolio_snapshots`/`account_snapshots`/`balance_snapshots`; load starting equity.
- A3: recompute realized/unrealized PnL from fills in `refresh_model_performance()`.

## Acceptance criteria
A trade transitions `OPEN -> CLOSING -> CLOSED` with non-null `realized_pnl_usdt`; snapshots
persist and starting equity carries across runs; `model_performance` reflects booked PnL.
