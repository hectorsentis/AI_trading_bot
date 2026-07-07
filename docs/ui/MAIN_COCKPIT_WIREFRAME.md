# Main Cockpit Wireframe

## Desktop target

Primary design width: 1440 to 1920 px. Minimum full-cockpit width: 1280 px.

```text
+--------------------------------------------------------------------------------------+
| MODE | BOT | KILL SWITCH | BINANCE | RECONCILIATION | DATA AGE | SYNC AGE | UTC     |
+------------+-----------------------------------------------------------+-------------+
| NAV        | EQUITY | FREE | LOCKED | DAILY PNL | DRAWDOWN | EXPOSURE | ACTION      |
|            +-----------------------------------------------------------+ REQUIRED    |
| Cockpit    | OPEN TRADES AND PROTECTION                                |             |
| Models     | Symbol | Model | PnL | Entry | TP dist | SL dist | Horizon | 3 critical  |
| Trades     |-----------------------------------------------------------| 2 warnings  |
| Proposals  | ...                                                       |             |
| Risk       +--------------------------------------+--------------------+-------------+
| Recon      | ACTIVE MODELS                        | RECONCILIATION      | CONTROLS    |
| Perf       | status, PnL, DD, activity            | balances, orders,  | read-only   |
| Shadow     |                                      | fills, mismatch    | until opened|
| Data       +--------------------------------------+--------------------+-------------+
| Audit      | LATEST PROPOSALS AND ALLOCATOR DECISIONS                  | SAFETY      |
| Health     | model, symbol, EV, requested, approved, decision, reason   | GATES       |
| Settings   +-----------------------------------------------------------+-------------+
|            | RISK USAGE | EXPOSURE | DATA FRESHNESS | RECENT EVENTS                  |
+------------+-----------------------------------------------------------+-------------+
| DETAIL TABS: Orders & Fills | Performance | Shadow | Data | Audit | Server Health    |
+--------------------------------------------------------------------------------------+
```

## Row 0: global safety and status command bar

**Purpose:** Answer whether operation may continue safely.

**Visible fields:**

- mode: research, dry-run, paper, shadow-real, live
- bot state: running, paused, stopped, error
- kill switch: armed/available and active/inactive
- Binance public/testnet/real-read/real-execution status as applicable
- reconciliation: OK, warning, failed, required
- latest market data timestamp and age
- latest account sync timestamp and age
- UTC clock, host, uptime, active warning count

**Interaction:** Clicking a status opens the relevant detail drawer. The warning count focuses
Action Required. There is no live-enable control in this bar.

**Visual priority:** Highest. Sticky at the top. Any critical state changes the complete bar
border and displays a full-width message below it.

**Empty state:** Unknown states use gray and say why data is unavailable.

**Warning/error state:** Reconciliation failure, active kill switch, or real-order capability
creates a persistent red banner with an explicit consequence such as "New entries blocked".

## Row 1: capital and risk strip

**Purpose:** Show capital availability and current risk without opening analytics.

**Visible fields:**

- total equity, free USDT, locked USDT, invested value
- realized PnL, unrealized PnL, daily PnL
- drawdown and limit
- total exposure and limit
- open trades and open orders

**Interaction:** Each metric opens a compact evidence drawer with source timestamp, source table,
and recent trend. Risk cards link to Risk & Safety.

**Visual priority:** High, but below safety state. Seven to nine compact cells, not oversized
marketing cards.

**Empty state:** Value becomes `No snapshot`; source age and recovery command remain visible.

**Warning/error state:** Limit usage at 70 percent becomes warning; at 90 percent becomes severe;
at or above 100 percent becomes critical. Thresholds are presentation defaults and must not
replace configured risk logic.

## Row 2: open trades and Action Required

### Open Trades and Protection

**Purpose:** Show every active exposure and whether it is protected.

**Visible fields:**

- `trade_id`, `model_id`, symbol, account mode, status
- approved notional, quantity, entry, current price
- unrealized PnL and PnL percent
- TP, SL, emergency SL
- distance to TP and SL
- holding time and horizon remaining
- linked order count and emergency-stop status

**Interaction:** Sort by risk, loss, age, or symbol. Row selection opens the trade lineage drawer.
Manual close is available only from that drawer after eligibility checks.

**Visual priority:** Largest panel in the cockpit.

**Empty state:** "No open trades" plus current mode and whether new entries are enabled.

**Warning/error state:** Missing SL, stale current price, closing timeout, reconciliation error,
or emergency-stop mismatch pins the row to the top.

### Action Required

**Purpose:** Provide one prioritized queue of operator decisions.

**Visible fields:**

- severity, title, affected entity, first observed UTC, age
- operational consequence
- recommended next step
- acknowledge/investigate affordance

**Interaction:** Filter critical/warning/information. Selecting an item opens its owning detail
surface. Acknowledgement never resolves the underlying system state.

**Visual priority:** High and persistent in the right rail.

**Empty state:** "No operator action required" with last evaluation UTC.

**Warning/error state:** Critical items cannot be visually collapsed.

## Row 3: active models and reconciliation

### Active Models

**Purpose:** Show lifecycle, activity, quality, and model-owned PnL.

**Visible fields:**

- `model_id`, lifecycle status, symbols, timeframe, family
- active/paper/shadow eligibility
- realized and unrealized PnL
- total return, max drawdown, win rate, profit factor
- open trades, accepted/rejected proposals
- degradation state and last prediction age

**Interaction:** Select a row for model detail. Quick filters: active, degraded, paused,
quarantined. Controls remain in the model drawer.

**Empty state:** Explain whether no models are registered or none match current filters.

**Warning/error state:** Degraded, stale, paused, and quarantined rows are grouped above healthy
inactive models.

### Reconciliation and Binance Health

**Purpose:** Show whether exchange and database state agree.

**Visible fields:**

- public data, testnet account, real read, real execution connection states
- last account snapshot
- free/locked balance difference
- open-order difference
- fill difference
- latest event severity and message

**Interaction:** "Investigate" opens the Reconciliation page. No "force pass" action exists.

**Empty state:** "No reconciliation event recorded"; execution eligibility is shown as blocked
when reconciliation is required.

**Warning/error state:** Failure uses a full panel red treatment and states which trading actions
are blocked.

## Row 4: proposal and allocator feed

**Purpose:** Show what models wanted and what shared-capital allocation allowed.

**Visible fields:**

- `proposal_id`, `model_id`, symbol, direction
- confidence, expected return, adverse move, MFE/MAE
- requested and approved notional
- proposal score and allocator score
- decision, rejection reason, shadow status
- proposal UTC and validity expiry

**Interaction:** Filter accepted, resized, rejected, deferred, and shadow. Select a row to view
prediction-to-trade lineage.

**Visual priority:** Medium-high. This is a table, not a chart.

**Empty state:** Distinguish "no recent predictions" from "predictions produced no proposals".

**Warning/error state:** Expired accepted proposals, missing allocation rows, or broken lineage
are highlighted as integrity problems.

## Row 5: compact operational evidence

Four compact panels:

1. **Risk usage:** daily loss, drawdown, total exposure, order notional, trade count.
2. **Exposure:** top symbols and models by notional.
3. **Freshness:** candles, features, account snapshots, predictions.
4. **Recent safety events:** risk rejections, stale-data blocks, exchange errors, forced exits.

Each panel has one purpose, one concise visualization or table, and a "View details" link.

## Bottom detail tabs

- Orders & Fills
- Performance
- Shadow Analytics
- Data Quality
- Audit Log
- Server Health

Tabs preserve current filters and selected entity. They are summaries; full pages remain
available from the left rail.

## Investigation drawer

The drawer opens from models, proposals, trades, orders, fills, events, and warnings.

It contains:

1. entity summary and current status
2. attribution lineage
3. timeline of state transitions
4. related orders/fills/events
5. eligible controls
6. raw payload debug section, collapsed by default

## Responsive behavior

### 1024 to 1279 px

- Left rail becomes icon-plus-tooltip.
- Right rail becomes a persistent "Action Required" drawer trigger.
- KPI strip wraps to two rows.
- Open Trades remains first.

### 768 to 1023 px

- Navigation becomes a top drawer.
- Tables use pinned identity/status columns and horizontal scrolling.
- Main cockpit becomes a single column ordered by safety, action, capital, trades, reconciliation,
  models, proposals.

### Below 768 px

The product is monitoring-capable but not control-optimized. Destructive and high-risk controls
are disabled with the message "Use a desktop viewport for guarded operations." Critical safety
state, capital, open trades, and action-required items remain available.
