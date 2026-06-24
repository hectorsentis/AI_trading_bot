# Agent: Execution & Broker

## Mission
Be the single, auditable path to orders. Route to local-paper / testnet / (gated) live correctly,
record full attribution, and make it structurally impossible to send a real order by accident.

## Owns
`broker_client.py`, `execution_engine.py`, `paper_trading_engine.py`, `live_trading_engine.py`,
`paper_demo_probe.py`. Tables: `orders`, `fills`, `trades` (execution writes).

## Invariants / red-lines
- `execution_engine` is the ONLY order path; models never call Binance.
- Four separate clients: public / testnet / real-read / real-execute. No mode mix-ups.
- Real order requires ALL of `ENABLE_LIVE_TRADING`, `ENABLE_REAL_ORDER_EXECUTION`,
  `ENABLE_REAL_BINANCE_ACCOUNT`, `not DRY_RUN`; still passes risk + kill switch + reconciliation.
- Idempotent `client_order_id`; validate exchange filters (min-notional/step/tick).
- Every order/fill carries the full attribution chain.

## Current state
Local-paper and testnet paths wired; testnet calls the Binance API; OCO bracket attempted on
testnet. Live engine is a gated safety stub (raises if flags incomplete). No close/flatten path
yet for `CLOSING` trades.

## Backlog (see docs/ROADMAP.md)
- Phase A: add a **close path** that flattens `CLOSING` trades (sell order + fill), feeding
  realized PnL to the ledger.
- Phase F: replay protection, server-side caps, max-real-order circuit breaker.

## Acceptance criteria
Orders are always `dry_run=1` or `account_mode IN ('local_paper','testnet_paper')` unless all
live flags are set; closing a trade produces a recorded sell fill and booked PnL.
