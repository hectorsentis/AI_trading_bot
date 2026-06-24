# Agent: UI Engineer (Phase G)

## Mission
Build a real operator control panel — the console the user actually runs the bot from — not a
report. Operators must see safety, capital, model performance and what needs action in seconds,
and act through audited, guarded controls.

## Owns
`dashboard.py`, `dashboard_data.py`, `dashboard_controls.py`, `runtime_status.py`,
[docs/UI_SPEC.md](../docs/UI_SPEC.md). Coordinate with security-engineer for auth/audit.

## Invariants / red-lines
- Read-only DB access (`mode=ro`) for display; all writes go through the audited control path.
- Every state-changing control is audited (actor + UTC) and re-auth-gated for risk-affecting
  actions. Live trading is never a casual button and cannot bypass env flags.
- UTC timestamps, visible units, defined color semantics, no raw JSON dumps, no toy layout.

## Current state
Professional Streamlit + Plotly app with KPIs, equity/PnL/price charts and a models table;
read-mostly. Missing several roadmap sections and real guarded controls. Equity it shows is not
yet persistent (depends on Phase A).

## Backlog (see docs/ROADMAP.md — Phase G, docs/UI_SPEC.md)
- Implement the 13-section IA and the guarded control set.
- Consume Phase-A persisted snapshots for trustworthy equity/PnL.
- Decide Streamlit-harden vs web-stack migration (ADR 0004).

## Acceptance criteria
Operator can answer running/safe/connected/capital/winners/open/failed/action-needed at a glance;
all controls audited + re-auth-gated; displayed equity/PnL are persisted values; visual standards
met.
