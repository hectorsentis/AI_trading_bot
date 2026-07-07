# Control Panel Design Model

Status: design foundation for Phase G. No runtime implementation is included.

## 1. Product definition

The control panel is the operator cockpit for the autonomous Binance Spot bot. It supervises:

- bot and process state
- capital, exposure, PnL, and drawdown
- risk limits and safety gates
- Binance public, testnet, and optional real-account connectivity
- reconciliation health
- model lifecycle and degradation
- predictions, proposals, allocator decisions, trades, orders, and fills
- virtual TP, virtual SL, emergency SL, and exit state
- data freshness and ingestion quality
- safety events, audit history, and required operator action

Its purpose is safe supervision and accountable operation. It must never imply guaranteed
profitability. Paper, shadow, and live-ready states must be clearly distinguished.

## 2. Design read

This is a dense trading-operations product for a technically capable operator. The visual
language is dark-first, restrained, precise, and terminal-grade without imitating a command
line. Safety state is the strongest layer. Motion is limited to feedback and state transition.

Design dials:

| Dial | Value | Rationale |
| --- | ---: | --- |
| Design variance | 3/10 | Familiar operational patterns reduce interpretation cost. |
| Motion intensity | 2/10 | Frequent actions and alerts must feel immediate. |
| Visual density | 9/10 | The user needs many related facts in one cockpit. |

## 3. Product layers

### Bot control panel

The custom cockpit owns trading state, model ownership, capital, risk, proposals, trades,
orders, reconciliation, and guarded controls. It is the primary operator experience.

### Analytics layer

QuantStats and project-native calculations provide performance summaries, return series,
drawdown, model comparison, and paper-versus-shadow analysis. Analytics are evidence, not
control state.

### Observability layer

Prometheus and Grafana own process health, service uptime, ingestion latency, resource use,
error rates, and alerts. Grafana may deep-link from Server Health, but it does not replace the
bot control panel.

## 4. System architecture represented in the UI

The UI must preserve the system path instead of flattening it into a single signal:

```text
prediction
  -> proposal
  -> allocation
  -> risk decision
  -> trade
  -> execution
  -> order
  -> fill
  -> ledger
  -> model performance
  -> lifecycle
```

Every applicable detail view exposes the linked identifiers:

`model_id`, `prediction_id`, `proposal_id`, `allocation_id`, `trade_id`, `order_id`, `fill_id`.

## 5. Screen architecture

### Persistent application shell

- **Top command/status bar:** mode, bot state, kill switch, connection, reconciliation, freshness,
  UTC clock, warnings count, operator identity.
- **Left section rail:** Cockpit, Models, Trades, Proposals, Risk & Safety, Reconciliation,
  Performance, Shadow Analytics, Data Quality, Audit Log, Server Health, Settings/Commands.
- **Main viewport:** the cockpit grid or selected detail page.
- **Right safety/action rail:** action-required queue, current safety gates, and guarded controls.
  It collapses into a drawer below 1280 px.
- **Investigation drawer:** opens from selected table rows and shows lineage, events, related
  entities, and debug payloads.

### Main cockpit priority

1. Global safety and mode
2. Action required
3. Capital and hard risk limits
4. Open trades and protection
5. Reconciliation and exchange health
6. Model lifecycle and performance
7. Proposal and allocation activity
8. Supporting analytics

## 6. Information hierarchy

### Level 1: visible immediately

- current operating mode
- bot status
- kill switch status
- Binance connection status
- reconciliation status
- latest market data UTC and age
- latest account sync UTC and age
- total equity, free USDT, locked USDT
- realized, unrealized, and daily PnL
- drawdown and daily-loss usage
- total exposure and open risk
- open trades and orders
- active, degraded, paused, and quarantined model counts
- urgent warnings and required actions

### Level 2: visible in the cockpit

- open trades with TP, SL, emergency SL, and horizon distance
- active model summary
- latest proposals and allocator decisions
- exposure by symbol and model
- reconciliation mismatches
- stale data warnings
- recent risk and safety events

### Level 3: drilldowns

- complete orders and fills
- model calibration and validation evidence
- shadow and allocator opportunity analysis
- data quality history
- process metrics and logs
- audit history
- raw payloads in a debug drawer only

## 7. Primary operator questions and owning surfaces

| Operator question | Primary surface |
| --- | --- |
| Is the bot running? | Global status bar and process heartbeat summary |
| Is it safe? | Kill-switch banner, safety gate strip, Action Required |
| Is Binance connected? | Connection badge and Reconciliation panel |
| Is reconciliation healthy? | Reconciliation status block and mismatch count |
| How much capital is available? | Capital summary |
| How much is at risk? | Exposure and risk-limit bars |
| Which models need attention? | Model status table and degradation queue |
| Which models are making or losing money? | Model PnL columns and detail analytics |
| Which trades are open? | Open Trades table |
| Where are TP and SL? | TP/SL distance indicator and trade drawer |
| What did models want? | Proposal feed |
| What did the allocator allow? | Proposal decision table |
| What failed? | Safety event feed and system status |
| What needs action now? | Right-side Action Required panel |
| Which controls are locked? | Guarded Controls panel with explicit lock reasons |

## 8. State precedence

The cockpit computes visual priority from persisted state but does not invent trading decisions.
When multiple states exist, the display order is:

1. live execution possible
2. kill switch active
3. reconciliation failed or required
4. exchange disconnected
5. stale market/account data
6. hard risk limit breached
7. bot or process error
8. model degraded or quarantined
9. normal warning
10. healthy

The UI may aggregate these states into presentation summaries, but the source rows and timestamps
must remain inspectable.

## 9. Monitoring and control separation

- The main cockpit is read-only.
- Controls live in the right rail or dedicated Settings/Commands page.
- No table row contains an unlabeled destructive icon.
- Selecting a model or trade opens details before offering an action.
- State-changing actions call an audited control service. They never issue direct UI SQL.
- Live enabling is not part of routine cockpit controls.

## 10. Technology ownership

| Technology | Responsibility |
| --- | --- |
| Next.js | Application shell, navigation, routing, server-rendered read views, cockpit layout |
| TypeScript | Typed contracts, state models, safe component APIs |
| shadcn/ui | Accessible primitives for badges, dialogs, tabs, forms, cards, sheets, and table shell |
| Tailwind CSS | Tokens, spacing, density, responsive layout, state styling |
| TanStack Table | Models, trades, proposals, orders, fills, events, and audit tables |
| Lightweight Charts | OHLCV, trade markers, entry/exit, and TP/SL overlays |
| FastAPI | Read-only dashboard API and future authenticated audited command endpoints |
| SQLite | Initial persisted source for operational and audit records |
| QuantStats | Portfolio and model performance summaries where its output is appropriate |
| Prometheus/Grafana | Infrastructure metrics, service health, alerting, and long-range ops history |

## 11. Streamlit-to-web compatibility

The product model is independent of rendering technology:

- Section data is defined as typed view contracts, not DataFrames embedded in layout code.
- Components have explicit state and units.
- Monitoring reads use a read-only adapter.
- Commands use a separate audited adapter.
- Streamlit can approximate the cockpit with containers, metrics, tabs, and styled dataframes.
- Next.js later maps the same contracts to reusable components and richer table interactions.

The transition should replace adapters and renderers, not redesign the product.

## 12. Explicit non-goals

- No dashboard implementation in this task.
- No trading, risk, reconciliation, or model-logic changes.
- No live-trading enablement.
- No credential changes.
- No direct order path.
- No profitability claim.
- No decorative chart wall.
- No generic analytics-report redesign.
- No direct SQLite mutation from future UI controls.
- No replacement of the control panel by Grafana.
