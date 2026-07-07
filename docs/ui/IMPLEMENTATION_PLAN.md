# UI Implementation Readiness Plan

This document sequences future work. It does not authorize implementation in the current task.

## 1. Navigation model

Primary pages:

1. **Cockpit** - default operator screen.
2. **Models** - registry, lifecycle, degradation, validation, model-owned performance.
3. **Trades** - open and historical trades, protection, lineage, exits.
4. **Proposals** - predictions, proposals, allocator and risk decisions.
5. **Risk & Safety** - limits, events, kill switch, stale-data blocks, action required.
6. **Reconciliation** - account snapshots, differences, order/fill matching, recovery state.
7. **Performance** - portfolio and model analytics.
8. **Shadow Analytics** - rejected/accepted comparison and allocator opportunity cost.
9. **Data Quality** - coverage, gaps, freshness, ingestion.
10. **Audit Log** - control actions, lifecycle events, safety actions.
11. **Server Health** - process heartbeats and observability links.
12. **Settings/Commands** - safe runtime settings and recommended CLI commands.

Cockpit remains the home route. Detail pages deepen investigation; they do not hide basic safety
or capital state.

## 2. Chart strategy

### Main cockpit

| Chart | Question answered | Source | Type/tool | Display rule |
| --- | --- | --- | --- | --- |
| Compact equity curve | Is equity improving or deteriorating in the selected account mode? | `portfolio_snapshots` | Small line chart; Lightweight Charts or simple web chart | Show only with sufficient recent data. |
| Drawdown/risk usage | How close are we to hard limits? | snapshots + config | RiskLimitBar and compact area | Always show limits; no series if history is absent. |
| Exposure by symbol/model | Where is capital concentrated? | positions/trades | Ranked horizontal bars or table summary | Show top contributors, not a pie by default. |
| TP/SL distance | Which open trades are closest to exit or danger? | trades + prices | Inline distance indicator | One per open trade row. |
| Compact price view | What market move explains this selected trade? | prices + trade/order markers | Lightweight Charts | Only after selecting a trade or symbol. |

### Detail views

| Chart | Question | Source | Tool |
| --- | --- | --- | --- |
| OHLCV with entry/exit/TP/SL | What happened around a trade? | prices, trades, orders, fills | Lightweight Charts |
| Equity by model | Which models add or lose equity? | snapshots/performance | QuantStats-derived series or chart layer |
| Drawdown by model | Which model has degraded risk? | model snapshots/performance | Chart layer |
| Realized PnL by model/symbol | Where do outcomes originate? | trades/model performance | Ranked bars and table |
| Expected vs realized return | Are proposal expectations calibrated economically? | predictions/proposals/trades | Scatter/decile summary |
| Probability calibration | Are predicted probabilities reliable? | validation/paper outcomes | Reliability diagram |
| Accepted vs rejected | Is allocator selection adding value? | allocations, trades, shadow trades | Distribution/summary |
| Paper vs shadow | Is model quality independent of allocation? | trades/shadow trades | Comparison series/table |
| Ingestion latency | Is market data arriving on time? | ingestion/status/Prometheus | Time series, Grafana link |
| Reconciliation history | Are mismatches recurring? | reconciliation events | Event timeline/count series |

Do not show a chart when:

- fewer than the minimum useful observations exist
- timestamps or units are ambiguous
- a table answers the operational question more directly
- the source is stale without a visible stale label
- the chart would mix paper and real account modes

## 3. Control model

All controls follow:

```text
operator intent
  -> eligibility check
  -> confirmation
  -> re-auth where required
  -> audited command request
  -> server-side validation
  -> orchestrator/execution owner
  -> command result
  -> immutable audit event
```

The UI never writes trading state directly to SQLite.

| Control | Intent | Confirmation / re-auth | Danger | Disabled when | Transition policy | Future endpoint |
| --- | --- | --- | --- | --- | --- | --- |
| Pause model | Stop new paper/live entries from one model | Confirm; re-auth in live-capable environment | Medium | Unknown model, pending command | Allowed if audited queue exists | `POST /commands pause_model` |
| Resume model | Restore eligible proposal flow | Confirm; re-auth if risk-affecting | Medium | Degraded/quarantined without lifecycle approval, unsafe system | Conservative; may be final-stack only | `resume_model` |
| Quarantine model | Block model and preserve evidence | Confirm with reason; re-auth | High | Already archived, command pending | Allowed only with lifecycle service | `quarantine_model` |
| Archive model | Remove from active pool, preserve history | Typed confirmation; re-auth | High | Open owned trades or unresolved policy | Final stack preferred | `archive_model` |
| Disable new entries | Globally stop new entries but retain monitoring/exits | Confirm; re-auth | High | Command service unavailable | Allowed if server-side guard exists | `disable_new_entries` |
| Shadow-only mode | Route eligible proposals to shadow evaluation | Confirm; re-auth | Medium | Reconciliation or mode transition pending | Allowed if orchestrator supports it | `enable_shadow_only` |
| Request retrain | Queue model maintenance | Confirm scope; no re-auth unless resource policy requires | Low | Training already running, invalid scope | Transition allowed | `request_retrain` |
| Manual close trade | Request central execution to close one trade | Typed trade ID, impact preview, re-auth | Critical | Stale price, reconciliation failure policy, not owner-safe, already closing | Final stack or strongly guarded service only | `close_trade` |
| Activate kill switch | Stop new activity and execute defined emergency policy | Strong confirmation; re-auth may be bypassed only by approved emergency policy | Critical | Never hidden; only disabled if already active | Transition allowed only with tested server path | `activate_kill_switch` |
| Clear kill switch | Return from emergency state after investigation | Typed phrase, reason, re-auth, all safety checks | Critical | Reconciliation/data/connection/risk not healthy | Final stack preferred | `clear_kill_switch` |
| Show CLI commands | Provide safe recovery commands | None | None | Never | Allowed | Read-only endpoint/static mapping |

### Feedback contract

- Pending: show command ID and timestamp.
- Success: describe resulting state, not merely "Success".
- Failure: preserve reason and remediation.
- Timeout: show unknown completion state and instruct the operator to investigate; never retry a
  destructive command automatically.

## 4. Current implementation gap

`src/dashboard_data.py` already uses read-only SQLite for display, which should be preserved.
`src/dashboard_controls.py` currently creates/updates control tables and may launch or terminate
processes directly. Before full Phase G controls, this must be replaced or wrapped by a
server-side audited command worker with:

- default-deny authorization
- CSRF protection and signed sessions
- re-authentication
- idempotency keys
- expected-state/version checks
- append-only audit records
- command status and timeout handling
- no direct Binance access from the UI process

This is a design requirement, not a code change in this task.

## 5. Implementation phases

### Phase 1: design docs and static layout skeleton

- Approve information architecture and visual tokens.
- Build static cockpit structure with representative fixtures.
- Validate desktop density and responsive order.
- Preserve `streamlit run src/dashboard.py`.

Exit criterion: stakeholder can review the complete cockpit without live data.

### Phase 2: read-only data adapters

- Define Python view models matching `DATA_CONTRACTS.md`.
- Add read-only SQLite adapter with explicit account-mode filtering.
- Add contract tests for missing tables, stale data, and null values.
- If starting the final stack, expose equivalent FastAPI read endpoints.

Exit criterion: all cockpit sections receive typed, source-aware data.

### Phase 3: main cockpit read-only panels

- Global status and safety banner.
- Capital/risk strip.
- Action Required.
- Open Trades and protection.
- Reconciliation summary.
- Active models and proposal decisions.

Exit criterion: operator can assess safety in under five seconds.

### Phase 4: detail views and tables

- Models, trades, proposals, orders, fills, reconciliation, data quality, audit, and server pages.
- Filtering, sorting, pinned columns, row drawers, attribution lineage.
- Empty/loading/error states.

Exit criterion: no normal investigation requires raw SQLite inspection.

### Phase 5: chart integration

- Lightweight Charts for OHLCV and trade overlays.
- Compact equity/drawdown/exposure views.
- QuantStats-backed summaries where validated.
- Text/table alternatives for accessibility.

Exit criterion: every chart answers a named operator question.

### Phase 6: audited controls

- Introduce command service/worker boundary.
- Add eligibility checks, confirmations, idempotency, and immutable audit.
- Start with pause model, disable entries, request retrain, and kill switch.
- Keep manual close and kill-switch clearing disabled until security tests pass.

Exit criterion: UI process performs no direct trading-state SQL mutation.

### Phase 7: authentication and re-auth hardening

- Rate limiting, lockout, signed expiring sessions, CSRF, roles, optional TOTP.
- Short-lived confirmation tokens.
- Security event logging and secret redaction.

Exit criterion: Phase F dashboard security acceptance tests pass.

### Phase 8: live updates

- Begin with polling and ETags.
- Add SSE for status/events if polling becomes inefficient.
- Use WebSocket only where bidirectional real-time behavior is justified.
- Preserve visible source timestamps and connection state.

Exit criterion: updates are timely without hiding disconnections.

### Phase 9: UX polish and acceptance tests

- Keyboard and screen-reader review.
- 100/125/150 percent zoom testing.
- Reduced-motion testing.
- Failure injection for stale data, reconciliation failure, and API outage.
- Visual regression tests for critical states.

Exit criterion: all acceptance criteria below pass.

### Phase 10: optional Streamlit-to-web migration

If the hybrid ADR is accepted:

- Freeze view contracts.
- Implement FastAPI read endpoints.
- Build Next.js shell and components.
- Run Streamlit and Next.js against the same fixture/contract tests.
- Move controls only after security and audit parity.
- Retire Streamlit only with documented command and deployment migration.

Exit criterion: product behavior and safety semantics remain unchanged across renderers.

## 6. Acceptance criteria

- Operator determines bot safety in under five seconds.
- Kill switch, reconciliation failure, and live-risk states are impossible to miss.
- Cockpit shows capital, risk, models, open trades, proposals, allocator decisions, and safety
  events without page navigation.
- Every number has units.
- Every timestamp is UTC and freshness age is visible where operationally relevant.
- Paper, shadow, testnet, and real account modes never merge silently.
- No raw JSON is primary UI.
- No chart exists without a documented operational question.
- Models, predictions, proposals, allocations, trades, orders, and fills preserve attribution.
- Controls are visually and technically separated from monitoring.
- Dangerous controls require confirmation, re-auth, server-side eligibility, and audit.
- Live trading cannot be enabled casually.
- UI state-changing actions never write directly to SQLite.
- Streamlit startup compatibility is preserved until a documented migration is approved.
- The design supports server monitoring and deep-links to Grafana without making Grafana the
  primary operator UI.
- The final stack explicitly supports Next.js, TypeScript, shadcn/ui, Tailwind, FastAPI, SQLite,
  Lightweight Charts, TanStack Table, QuantStats, Prometheus, and Grafana.
- Responsive monitoring works below desktop width; critical operations may be desktop-only.
- WCAG 2.2 AA contrast, focus, keyboard, and reduced-motion requirements pass.

## 7. Validation scenarios

Future acceptance tests must render at least:

1. healthy local-paper operation
2. no database / first-run state
3. stale market data
4. testnet disconnected
5. reconciliation failed with open trades
6. active kill switch
7. real-order flags possible
8. daily loss near limit
9. open trade missing protection
10. model degraded and quarantined
11. allocator rejected all recent proposals
12. command pending, failed, timed out, and succeeded
13. partial API outage with last known data

## 8. Explicit non-goals for implementation

- Do not change trading logic to simplify the UI.
- Do not infer exchange truth from local rows when reconciliation is missing.
- Do not place orders from frontend code or dashboard processes.
- Do not use Grafana as the control surface.
- Do not introduce live-enabling controls as part of initial cockpit work.
- Do not remove the Streamlit launch path without an accepted migration plan.
