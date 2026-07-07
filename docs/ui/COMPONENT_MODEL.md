# Component Model

## Component contract rules

- Every component declares loading, empty, stale, warning, error, and disabled behavior.
- Every value declares units and source timestamp.
- Color supplements text, icon, and label; it never carries meaning alone.
- Monitoring components are read-only.
- Control-capable components call an audited command adapter, never SQLite.
- Identifiers remain copyable and are not silently truncated. A compact display may show a short
  form while the full value remains available.
- The Next.js implementation uses typed props. The Streamlit transition uses equivalent data
  dictionaries or DataFrames and the same state vocabulary.

## StatusBadge

- **Purpose:** Express health or lifecycle state in a compact, consistent form.
- **Props:** `label`, `tone`, `icon`, `detail`, `timestampUtc`, `stale`.
- **States:** healthy, info, warning, paused, critical, inactive, unknown, stale.
- **Color:** semantic token plus icon and text.
- **Interaction:** Optional click opens evidence; never acts as a toggle.
- **Capability:** Read-only.
- **Final stack:** shadcn `Badge` with Tailwind variants and accessible label.
- **Streamlit:** Styled markdown badge or status column.

## RiskBadge

- **Purpose:** Mark risk severity and operational consequence.
- **Props:** `severity`, `reasonCode`, `label`, `blockedActions`, `eventId`.
- **States:** advisory, warning, breached, blocked, emergency.
- **Color:** blue, amber, red, dark red respectively.
- **Interaction:** Opens risk event drawer.
- **Capability:** Read-only.
- **Final stack:** Badge plus Popover/Sheet.
- **Streamlit:** Styled label and expander.

## ModeBadge

- **Purpose:** Prevent confusion between research, dry-run, paper, shadow-real, and live.
- **Props:** `mode`, `realOrdersPossible`, `accountMode`.
- **States:** research, dry-run, local-paper, testnet-paper, shadow-real, live.
- **Color:** neutral/blue for non-live; red for live or real-order capability.
- **Interaction:** Opens safety-gate evidence.
- **Capability:** Read-only.
- **Final stack:** Badge with fixed mode mapping.
- **Streamlit:** Header pill.

## KillSwitchBanner

- **Purpose:** Make kill-switch state impossible to miss.
- **Props:** `enabled`, `active`, `activatedAtUtc`, `actor`, `reason`, `affectedServices`.
- **States:** available/inactive, active, unknown, action pending.
- **Color:** Critical red when active; neutral when available.
- **Interaction:** Active banner links to Safety Events. Activation and clearing occur in a
  separate guarded control flow.
- **Capability:** Read-only banner; related control is separate.
- **Final stack:** shadcn `Alert` plus action link.
- **Streamlit:** Full-width error/warning container.

## ReconciliationStatus

- **Purpose:** Summarize DB/exchange agreement and execution eligibility.
- **Props:** `status`, `lastRunUtc`, `balanceDiffUsdt`, `orderDiffCount`, `fillDiffCount`,
  `executionBlocked`, `eventId`.
- **States:** OK, not-required, running, stale, warning, failed, unknown.
- **Color:** Green, blue, amber, red, gray.
- **Interaction:** Opens reconciliation evidence.
- **Capability:** Read-only.
- **Final stack:** Compact panel using Badge, Separator, and Sheet.
- **Streamlit:** Metrics plus expander.

## KPIStatCard

- **Purpose:** Show one capital or operating metric with provenance.
- **Props:** `label`, `value`, `unit`, `delta`, `deltaPeriod`, `tone`, `sourceUtc`, `sourceName`.
- **States:** normal, positive, negative, warning, critical, stale, unavailable.
- **Color:** Neutral by default; semantic only when direction has meaning.
- **Interaction:** Opens trend/evidence drawer if available.
- **Capability:** Read-only.
- **Final stack:** lightweight div or shadcn Card; cards should not be nested.
- **Streamlit:** `st.metric` only with custom compact styling.

## RiskLimitBar

- **Purpose:** Compare current risk usage to a configured limit.
- **Props:** `label`, `used`, `limit`, `unit`, `warningThreshold`, `criticalThreshold`,
  `sourceUtc`.
- **States:** safe, warning, near-limit, breached, unavailable.
- **Color:** Neutral track; green/amber/red fill with numeric label.
- **Interaction:** Opens contributing exposures/events.
- **Capability:** Read-only.
- **Final stack:** custom accessible progress primitive.
- **Streamlit:** `st.progress` plus explicit values and threshold labels.

## ExposureBar

- **Purpose:** Show concentration by symbol or model.
- **Props:** `entityId`, `notionalUsdt`, `portfolioPct`, `limitUsdt`, `limitPct`.
- **States:** normal, concentrated, breached, stale.
- **Color:** One neutral information palette until warning thresholds.
- **Interaction:** Filters open trades and positions to the entity.
- **Capability:** Read-only.
- **Final stack:** compact bar list; avoid pie/donut by default.
- **Streamlit:** horizontal bar chart or styled progress rows.

## ModelStatusTable

- **Purpose:** Compare model lifecycle, activity, quality, and PnL.
- **Fields:** `model_id`, lifecycle status, symbols, timeframe, family, active flags, predictions,
  proposals, accepted/rejected, open trades, PnL, return, drawdown, win rate, profit factor,
  degradation, last prediction UTC.
- **States:** loading skeleton, empty registry, filtered empty, stale, partial-data warning.
- **Color:** Status and PnL cells only; row background reserved for critical states.
- **Interaction:** sort, filter, column visibility, pin identity columns, select row.
- **Capability:** Read-only table; controls live in detail drawer.
- **Final stack:** TanStack Table with shadcn table shell.
- **Streamlit:** styled dataframe with row selection.

## ProposalDecisionTable

- **Purpose:** Connect model intent to allocator outcome.
- **Fields:** attribution IDs, symbol, direction, confidence, expected return/adverse move,
  requested/approved notional, scores, decision, rejection reason, shadow state, UTC/expiry.
- **States:** no predictions, no proposals, no allocations, lineage error, stale.
- **Color:** Accepted green label, resized blue, rejected amber/red by reason, expired gray.
- **Interaction:** filters, row selection, lineage drawer.
- **Capability:** Read-only.
- **Final stack:** TanStack Table.
- **Streamlit:** merged DataFrame with selectable rows.

## OpenTradesTable

- **Purpose:** Provide the main operational view of active exposure and protection.
- **Fields:** trade/model IDs, account mode, symbol, status, notional, quantity, entry/current,
  TP/SL/emergency SL, distances, PnL, age, horizon, linked orders.
- **States:** no open trades, stale price, unprotected, closing, reconciliation error.
- **Color:** PnL and protection risk only.
- **Interaction:** risk-first sorting, row selection, trade drawer.
- **Capability:** Read-only table.
- **Final stack:** TanStack Table with pinned columns and custom distance cells.
- **Streamlit:** styled dataframe.

## TP_SL_DistanceIndicator

- **Purpose:** Explain current price position relative to entry, TP, SL, and emergency SL.
- **Props:** `side`, `entry`, `current`, `tp`, `sl`, `emergencySl`, `currency`, `stale`.
- **States:** inside range, near TP, near SL, past virtual level, emergency level breached,
  missing protection, stale.
- **Color:** Neutral line with labeled green TP and red SL markers; never an unlabeled gradient.
- **Interaction:** Hover/focus reveals exact prices and percentage distances.
- **Capability:** Read-only.
- **Final stack:** custom SVG or CSS component; full chart not required.
- **Streamlit:** compact Plotly bullet-style chart or textual distance columns.

## PnLCell

- **Purpose:** Display monetary and percentage PnL consistently.
- **Props:** `amountUsdt`, `percent`, `realized`, `sourceUtc`.
- **States:** gain, loss, flat, unavailable, stale.
- **Color:** Green/red plus sign and text.
- **Interaction:** Optional detail tooltip.
- **Capability:** Read-only.
- **Final stack:** table cell renderer.
- **Streamlit:** formatted/styled cell.

## EquityCurveCard

- **Purpose:** Answer whether equity is rising or falling and whether data is current.
- **Props:** time series, account mode, selected model, period, source.
- **States:** valid, insufficient history, stale, gap, unavailable.
- **Color:** Neutral line; profit color is not required for the entire curve.
- **Interaction:** period selector and model/account filter.
- **Capability:** Read-only.
- **Final stack:** compact chart layer; QuantStats metrics may accompany it.
- **Streamlit:** Plotly line chart.

## DrawdownCard

- **Purpose:** Show current and maximum drawdown against the hard limit.
- **Props:** drawdown series, current, maximum, limit, period.
- **States:** safe, warning, near-limit, breached, unavailable.
- **Color:** Amber to red only as risk increases.
- **Interaction:** period selection and event links.
- **Capability:** Read-only.
- **Final stack:** compact area/line chart plus RiskLimitBar.
- **Streamlit:** Plotly chart plus progress.

## SafetyEventFeed

- **Purpose:** Show recent risk, reconciliation, exchange, stale-data, and forced-exit events.
- **Props:** event ID, severity, component, message, entity IDs, created UTC, acknowledged state.
- **States:** empty, streaming/loading, partial source failure.
- **Color:** Severity marker plus explicit label.
- **Interaction:** filter and open event drawer.
- **Capability:** Read-only; acknowledgement is a separate audited action.
- **Final stack:** virtualized list or TanStack Table.
- **Streamlit:** compact dataframe/list.

## ActionRequiredPanel

- **Purpose:** Prioritize unresolved operator decisions.
- **Props:** issue ID, severity, title, consequence, entity, first/last seen UTC, recommendation.
- **States:** no action, warning queue, critical queue, stale evaluation.
- **Color:** Red only for critical items; warnings amber.
- **Interaction:** investigate, acknowledge where allowed, open owning page.
- **Capability:** Mostly read-only; acknowledgement uses audited endpoint.
- **Final stack:** list with Sheet links.
- **Streamlit:** right column or expander list.

## ControlButtonGroup

- **Purpose:** Group actions by entity and danger level without mixing them into monitoring.
- **Props:** allowed actions, disabled reasons, entity, mode, permissions.
- **States:** available, disabled, locked, pending, succeeded, failed.
- **Color:** Neutral default; amber for disruptive; red for destructive/emergency.
- **Interaction:** opens confirmation flow. No immediate action on first click.
- **Capability:** Control-capable.
- **Final stack:** shadcn Button, DropdownMenu, AlertDialog.
- **Streamlit:** buttons/forms only for transition-approved actions.

## ConfirmationModal

- **Purpose:** Create deliberate friction for state-changing actions.
- **Props:** action, impact, entity, required phrase, re-auth requirement, current safety context.
- **States:** ready, validation error, re-auth failed, submitting, succeeded, failed.
- **Color:** Severity-dependent; no decorative styling.
- **Interaction:** explicit confirm/cancel; destructive confirmation phrase where required.
- **Capability:** Control-capable.
- **Final stack:** shadcn AlertDialog plus re-auth form.
- **Streamlit:** form/expander approximation; critical controls may remain unavailable.

## AuditLogTable

- **Purpose:** Show who requested what, when, against which entity, and the result.
- **Fields:** audit/action ID, actor, action, target type/ID, request UTC, processed UTC, status,
  reason, before/after summary, correlation ID.
- **States:** empty, partial history, integrity warning.
- **Color:** Status only.
- **Interaction:** filter, export, open immutable detail.
- **Capability:** Read-only.
- **Final stack:** TanStack Table.
- **Streamlit:** dataframe.

## DataFreshnessIndicator

- **Purpose:** Show timestamp and age against a configured stale threshold.
- **Props:** source, latest UTC, age seconds, warning/critical thresholds.
- **States:** fresh, aging, stale, missing.
- **Color:** Green, amber, red, gray plus age text.
- **Interaction:** opens Data Quality.
- **Capability:** Read-only.
- **Final stack:** Badge plus tooltip.
- **Streamlit:** metric/pill.

## ProcessHeartbeatCard

- **Purpose:** Show process state for ingestor, bot, maintenance, evaluator, API, and dashboard.
- **Props:** component, status, PID, host, started UTC, last heartbeat UTC, message.
- **States:** running, stopping, stopped, stale, failed, unknown.
- **Color:** Semantic status.
- **Interaction:** links to Grafana or process logs; no direct terminate in monitoring card.
- **Capability:** Read-only.
- **Final stack:** compact list/card.
- **Streamlit:** status rows.

## DebugDrawer

- **Purpose:** Expose raw payloads and source evidence without making them primary UI.
- **Props:** entity, source table/endpoint, payload, query timestamp, schema version.
- **States:** unavailable, redacted, parse error, loaded.
- **Color:** Neutral.
- **Interaction:** copy redacted payload; secrets must never be displayed.
- **Capability:** Read-only.
- **Final stack:** shadcn Sheet with code block.
- **Streamlit:** collapsed expander.

## Motion contract

- Keyboard navigation and frequent table interactions: no animation.
- Tooltips/popovers: 125 to 180 ms ease-out.
- Investigation drawer: 180 to 240 ms ease-out.
- Confirmation dialogs: 180 to 220 ms; no bounce.
- Button press feedback: 100 to 140 ms.
- Live values update without movement; use a brief background tint or text transition only.
- Respect `prefers-reduced-motion`.
