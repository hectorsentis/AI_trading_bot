# ADR 0004: UI stack and transition path

Status: Proposed

Date: 2026-06-25

## Context

Phase G requires a professional operator control panel for safety, capital, models, proposals,
trades, reconciliation, and audited controls. The repository already has a substantial
Streamlit dashboard launched with:

```bash
streamlit run src/dashboard.py
```

The current dashboard has a defensive read-only data layer in `src/dashboard_data.py`, but the
control layer in `src/dashboard_controls.py` can directly mutate control/configuration tables and
can launch or stop processes. The final control model requires a stronger API and audit boundary.

The intended professional stack is:

```text
Frontend:
  Next.js + TypeScript
  shadcn/ui + Tailwind CSS
  TanStack Table
  Lightweight Charts

Backend:
  FastAPI
  SQLite initially

Analytics:
  QuantStats

Observability:
  Prometheus + Grafana
```

The custom bot control panel remains the primary operator UI. Grafana is reserved for process,
service, and infrastructure observability.

## Decision drivers

- fastest safe path to a useful Phase G cockpit
- preservation of existing startup and deployment workflows
- strong information hierarchy and dense table support
- clean read/control separation
- audited, authenticated, re-authenticated commands
- safe live-update behavior
- server deployment and role separation
- ability to retain model/trade attribution
- migration cost and operational complexity

## Option A: harden Streamlit first

### Description

Restructure the current Streamlit app around the new cockpit design and preserve its read-only
SQLite adapter. Add missing panels and improve information hierarchy before introducing a new
frontend platform.

### Benefits

- fastest implementation path
- lowest short-term engineering cost
- existing data loaders and startup command remain valid
- good fit for read-heavy paper/testnet supervision
- simple local and server deployment
- allows Phase G data-contract gaps to be discovered before API expansion

### Costs and risks

- lower UX ceiling for dense terminal-style layout
- weaker control over responsive shell and persistent rails
- complex TanStack-style table behavior is difficult
- robust auth, roles, CSRF, signed sessions, and re-auth flows are awkward
- direct process and SQLite control patterns are easier to implement unsafely
- live updates and command status interaction are less natural
- harder to achieve a consistently polished professional terminal

### Appropriate use

Near-term read-only cockpit, paper/testnet operation, and contract validation.

## Option B: FastAPI plus Next.js professional frontend

### Description

Create an explicit API boundary and implement the cockpit with Next.js, TypeScript, shadcn/ui,
Tailwind, TanStack Table, and Lightweight Charts.

### Benefits

- highest final UX and information-density ceiling
- typed component and data-contract model
- clear frontend/backend role separation
- stronger authenticated command and audit patterns
- better table filtering, pinned columns, drawers, and responsive behavior
- better support for polling, SSE, and future live state
- clean deployment boundary for operator UI and bot processes
- shadcn/ui provides accessible primitives without locking the product to a fixed theme

### Costs and risks

- highest engineering and deployment cost
- introduces frontend build, API service, reverse proxy, and versioning
- requires a carefully designed API before controls can move
- creates two server processes in addition to trading services
- migration may distract from Phase F security and Phase E deployment if started too early
- unsafe direct DB writes could be reproduced behind an API unless command ownership is designed
  correctly

### Appropriate use

The final professional operator experience, especially when audited controls, roles, live
updates, and multi-user operation are required.

## Option C: hybrid transition path

### Description

Preserve Streamlit for compatibility and near-term delivery, but design the product and data
contracts for the final FastAPI and Next.js stack from the start.

Sequence:

1. Preserve `streamlit run src/dashboard.py`.
2. Use the shared cockpit, component, and data contracts in `docs/ui/`.
3. Harden the Streamlit cockpit as read-only where it accelerates Phase G.
4. Introduce a FastAPI read API and audited command service.
5. Build the Next.js frontend against the same contracts.
6. Migrate controls only after security and audit parity.
7. Retire Streamlit only through a documented deployment migration.

### Benefits

- combines short-term delivery with a credible final architecture
- avoids redesigning the product during migration
- validates contract and data-quality assumptions early
- preserves existing operator workflow
- allows security/API work to mature before controls move
- lowers big-bang migration risk

### Costs and risks

- temporary dual-renderer maintenance
- requires discipline to avoid Streamlit-specific business logic
- shared fixtures and contract tests are necessary
- migration can stall if explicit exit criteria are not enforced

## Responsibility allocation for the final stack

| Stack piece | Responsibility |
| --- | --- |
| Next.js | Application shell, navigation, routing, cockpit composition, server-rendered read views |
| TypeScript | Typed view models, component contracts, command payloads |
| shadcn/ui | Cards, badges, dialogs, sheets, tabs, command panels, forms, accessible primitives |
| Tailwind CSS | Spacing, density, responsive layout, semantic tokens, visual system |
| TanStack Table | Models, trades, proposals, orders, fills, events, and audit tables |
| Lightweight Charts | OHLCV, entry/exit markers, TP/SL overlays, compact selected-trade views |
| FastAPI | Read-only dashboard API and future authenticated audited command endpoints |
| SQLite | Initial operational and audit persistence |
| QuantStats | Validated performance summaries, returns, and drawdown analytics |
| Prometheus | Service/process metrics and alert inputs |
| Grafana | Infrastructure observability, historical health, alerts, and drilldowns |

## Decision

Recommend **Option C, the hybrid transition path**.

The product design targets the final FastAPI plus Next.js stack. Streamlit remains the near-term
compatibility and delivery path only when it is the fastest safe way to implement read-only
cockpit sections. New UI work must follow the shared contracts so it can move without product
redesign.

State-changing controls must not be expanded through direct Streamlit-to-SQL writes. Before
critical controls are implemented, introduce an audited server-side command boundary. Manual
close, kill-switch clearing, and any live-affecting action should remain unavailable until Phase
F security and command tests pass.

## Consequences

### Positive

- existing launch workflow is preserved
- Phase G can progress without a big-bang rewrite
- final UX direction is explicit
- data contracts become stable migration boundaries
- Grafana remains correctly scoped to observability
- control safety can mature independently from renderer choice

### Negative

- temporary duplication is accepted
- contract tests and fixtures become mandatory
- the team must prevent new business logic from accumulating in Streamlit views
- FastAPI and Next.js deployment work remains pending

## Migration triggers

Move from hardened Streamlit to the full web stack when any two of these are true:

- audited controls require roles, re-auth, CSRF, or multi-user sessions
- table interactions exceed practical Streamlit capabilities
- sub-minute polling creates performance or reliability problems
- responsive cockpit requirements cannot be met cleanly
- command status and event streaming are required
- deployment requires independent UI/API scaling

Security requirements alone may trigger migration even if only one condition is met.

## Compatibility

No startup command changes are made by this ADR. Current command:

```bash
streamlit run src/dashboard.py
```

Future commands for FastAPI/Next.js must be added alongside this command until the migration is
accepted, documented, and verified.

## Non-decision

This ADR does not implement Streamlit changes, FastAPI, Next.js, authentication, controls,
Prometheus, Grafana, or QuantStats integration.
