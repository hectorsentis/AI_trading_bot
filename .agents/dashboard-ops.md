# Agent: Dashboard & Orchestration

## Mission
Run the whole system reliably and make it observable. Coordinate ingestion, prediction, trading,
exits and reconciliation in a restart-safe loop, and surface state to the operator.

## Owns
`trading_bot.py`, `autonomous_runner.py`, `runtime_status.py`, `dashboard.py`,
`dashboard_data.py`, `dashboard_controls.py`. Tables: `bot_status`, `bot_events`,
`bot_control_actions`, `model_control`, `runtime_config*`.

## Invariants / red-lines
- Graceful shutdown; restart-safe state; idempotent steps.
- Dashboard reads operational data (read-only `mode=ro`); it never sends manual buy/sells and
  cannot bypass env safety flags for live trading.
- Heartbeats drive RUNNING vs OFF/STALE status.

## Current state
`autonomous_runner` supervises ingestor + bot + evaluator + maintenance + dashboard. The main
loop runs prediction -> proposal -> allocation -> risk -> build -> execution but does **not** call
exits or persist snapshots (see ledger-portfolio backlog). Dashboard is professional but
read-mostly.

## Backlog (see docs/ROADMAP.md)
- Phase A: call exit evaluation + snapshot persistence in the loop (coordinate with
  ledger-portfolio).
- Phase G: full control-panel rebuild with audited, re-auth-gated controls (see ui-engineer).
- Phase E: server deployment (systemd/Docker, healthchecks, `pathlib`).

## Acceptance criteria
A run-once executes the full lifecycle including exits and snapshot writes; status reflects real
heartbeats; controls are audited.
