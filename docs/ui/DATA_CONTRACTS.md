# UI Data Contracts

## 1. Contract principles

- SQLite is the initial persisted operational and audit source.
- Transition-mode display access may use SQLite read-only mode.
- The final frontend consumes FastAPI view endpoints, never SQLite directly.
- State-changing actions use authenticated audited command endpoints.
- Contracts expose source timestamps and staleness, not only business values.
- Timestamps are ISO 8601 UTC with `Z` or explicit `+00:00`.
- Monetary values include an explicit currency field or use a field suffix such as `_usdt`.
- Returns and percentages are decimal fractions in transport, for example `0.0125` means
  `1.25%`.
- Prices use quote-asset units; quantities use base-asset units.
- Missing values are `null`, never magic zeroes.
- Raw JSON columns are excluded from normal view contracts and available only through a redacted
  debug endpoint.

## 2. Common envelope

```json
{
  "as_of_utc": "2026-06-25T12:00:00Z",
  "source": "sqlite",
  "source_version": "v1",
  "stale": false,
  "stale_after_seconds": 120,
  "warnings": [],
  "data": {}
}
```

List endpoints also include `page`, `page_size`, `total`, `sort`, and applied filters.

## 3. Global status contract

**Sources:** `bot_status`, `system_status`, `bot_events`, runtime configuration, market-data
coverage, latest account/reconciliation records.

**Required fields:**

- `mode`
- `bot_status`
- `kill_switch_enabled`
- `kill_switch_active`
- `real_orders_possible`
- `public_data_connected`
- `testnet_account_connected`
- `real_account_read_connected`
- `real_execution_enabled`
- `reconciliation_status`
- `latest_market_data_utc`
- `market_data_age_seconds`
- `latest_account_sync_utc`
- `account_sync_age_seconds`
- `db_path_display`
- `hostname`
- `uptime_seconds`
- `warning_count`
- `critical_count`

**Stale behavior:** Each source has independent age. Do not mark the complete status fresh because
one heartbeat is current.

**Fallback:** Unknown fields remain visible with the reason `No persisted status`.

**Error:** Return the last known status with `stale=true` and an error warning.

**Transition access:** `dashboard_data.load_system_status()` plus read-only joins.

**Final API:** `GET /api/v1/cockpit/status`.

## 4. Capital and risk summary

**Sources:** `portfolio_snapshots`, `account_snapshots`, `balance_snapshots`, `positions`,
`trades`, `orders`, `risk_events`, configured risk limits.

**Required fields:**

- `account_mode`
- `total_equity_usdt`
- `free_usdt`
- `locked_usdt`
- `invested_value_usdt`
- `realized_pnl_usdt`
- `unrealized_pnl_usdt`
- `daily_pnl_usdt`
- `total_return`
- `current_drawdown`
- `max_drawdown`
- `daily_loss_used_usdt`
- `daily_loss_limit_usdt`
- `total_exposure_usdt`
- `total_exposure_limit_usdt`
- `open_risk_usdt`
- `open_trades_count`
- `open_orders_count`
- `snapshot_utc`

**Units:** USDT and decimal returns.

**Stale behavior:** Account snapshot age and mark-price age are shown separately.

**Fallback:** Portfolio snapshots may support local-paper display when account snapshots are
absent. The UI labels the source.

**Error:** Never combine real and paper account modes silently.

**Transition access:** Existing portfolio summary plus dedicated read-only view queries.

**Final API:** `GET /api/v1/cockpit/capital?account_mode=...`.

## 5. Risk-limit usage

**Sources:** configuration, `risk_events`, `positions`, `trades`, `orders`,
`portfolio_snapshots`.

**Required per limit:**

- `limit_key`
- `label`
- `current_value`
- `limit_value`
- `unit`
- `usage_fraction`
- `status`
- `configured_source`
- `updated_at_utc`
- `contributing_entity_count`

**Stale behavior:** Exposure limits inherit price freshness. Daily loss inherits latest equity
snapshot freshness.

**Fallback:** Unknown configured limit disables the bar and displays `Not configured`.

**Final API:** `GET /api/v1/risk/limits`.

## 6. Binance and reconciliation health

**Sources:** `account_snapshots`, `balance_snapshots`, `reconciliation_events`,
`system_status`, `orders`, `fills`.

**Required fields:**

- connection state by client type
- latest account snapshot ID and UTC
- free/locked balance totals
- balance difference USDT
- DB/exchange open-order counts and difference
- DB/exchange fill counts and difference
- reconciliation status/severity/message
- execution blocked boolean
- latest event ID

**Stale behavior:** If reconciliation is required and the latest successful event exceeds the
configured threshold, display `RECONCILIATION REQUIRED` and block control eligibility.

**Fallback:** `No reconciliation recorded` is not equivalent to `OK`.

**Error:** Preserve the last successful snapshot and display the current failure separately.

**Transition access:** Read-only table queries already used by the dashboard.

**Final API:** `GET /api/v1/reconciliation/summary` and
`GET /api/v1/reconciliation/events`.

## 7. Active models

**Sources:** `model_registry`, `model_lifecycle_events`, `model_performance`,
`paper_model_metrics`, `model_predictions`, `trade_proposals`, `trades`, `model_control`.

**Required fields per model:**

- `model_id`
- `status`
- `lifecycle_stage`
- `symbols`
- `timeframe`
- `model_family`
- `training_scope`
- `trained_at_utc`
- `validation_window`
- `oos_status`
- `paper_started_at_utc`
- prediction/proposal/accept/reject/shadow/open/closed counts
- realized/unrealized PnL USDT
- total return, max drawdown, win rate, profit factor
- calibration summary
- degradation status
- last prediction/proposal/trade UTC
- signal/paper/live eligibility

**Stale behavior:** Model activity age is evaluated against timeframe-aware thresholds.

**Fallback:** Preserve registry rows even when no performance row exists.

**Error:** Broken attribution or missing registry relationship produces an integrity warning,
not a dropped row.

**Transition access:** `dashboard_model_activity` view and existing loaders.

**Final API:** `GET /api/v1/models` and `GET /api/v1/models/{model_id}`.

## 8. Open trades and protection

**Sources:** `trades`, `orders`, `fills`, `positions`, `prices`, `risk_events`.

**Required fields per trade:**

- complete attribution IDs
- `account_mode`, `symbol`, `timeframe`, `side`, `status`
- requested/approved notional USDT
- quantity, entry, average entry, current price
- TP, SL, emergency SL
- TP/SL/emergency distance fractions
- realized/unrealized PnL USDT
- fees and slippage USDT
- opened/updated UTC
- holding seconds, horizon bars, horizon remaining
- signal validity UTC
- exit reason
- linked order/fill counts
- emergency stop order ID/status
- price timestamp and stale boolean

**Stale behavior:** Stale current price prevents distance values from appearing current. Keep
the last value but label it stale.

**Fallback:** Trade rows remain visible if position aggregation is missing.

**Error:** Missing SL or broken lineage creates a critical protection/integrity state.

**Transition access:** `dashboard_trade_lineage` plus latest-price query.

**Final API:** `GET /api/v1/trades?status=open` and `GET /api/v1/trades/{trade_id}`.

## 9. Proposals and allocator decisions

**Sources:** `model_predictions`, `trade_proposals`, `allocations`, `shadow_trades`,
`risk_events`, `trades`.

**Required fields:**

- prediction/proposal/allocation/trade/shadow IDs
- model, symbol, timeframe, direction/side
- confidence, probabilities
- expected return/move/adverse move
- return quantiles, expected MFE/MAE
- horizon and validity UTC
- requested/approved notional USDT
- proposal and allocator scores
- proposal status, allocation decision, rejection reason
- shadow status and outcome when resolved
- created/updated UTC

**Stale behavior:** Expired proposal is a lifecycle state, not a data-source error.

**Fallback:** A proposal without allocation is displayed as `Awaiting allocation` only when it
is still valid; otherwise it is an integrity warning.

**Final API:** `GET /api/v1/proposals`.

## 10. Orders and fills

**Sources:** `orders`, `fills`.

**Required order fields:**

- order, trade, model, proposal, allocation IDs where available
- client/exchange order IDs
- account mode, symbol, side, type, status
- quantity, requested/fill price, notional
- protection linkage
- created/updated/filled UTC
- reason/error

**Required fill fields:**

- fill/order/trade/model IDs
- exchange trade ID
- quantity, price
- commission and asset
- fee USDT if normalized
- timestamp UTC

**Stale behavior:** Open order status age is shown; old nonterminal states become warnings.

**Fallback:** An order without fills is valid for unfilled states.

**Error:** A fill without an order is an integrity error.

**Final API:** `GET /api/v1/orders`, `GET /api/v1/fills`.

## 11. Performance analytics

**Sources:** `portfolio_snapshots`, `model_performance`, `paper_model_metrics`, `trades`,
`shadow_trades`; QuantStats-derived summaries where useful.

**Required fields:**

- portfolio/model equity series
- returns series
- drawdown series
- realized/unrealized PnL
- win rate, profit factor, average trade return
- expected versus realized return
- accepted versus rejected outcomes
- paper versus shadow outcomes
- sample count and analysis period

**Stale behavior:** Analytics have an `analysis_end_utc`; they are not labeled live unless they
include current snapshots.

**Fallback:** Insufficient sample produces an explicit message, not a misleading zero metric.

**Final API:** `GET /api/v1/performance/portfolio`,
`GET /api/v1/performance/models/{model_id}`.

## 12. Shadow analytics

**Sources:** `shadow_trades`, `shadow_trade_events`, `trade_proposals`, `allocations`.

**Required fields:**

- total/open/closed shadow counts
- would-have-won count/rate
- missed profit USDT
- avoided loss USDT
- average outcome PnL USDT
- outcome and reason breakdown
- proposal/model/allocator attribution
- evaluation period and matured sample count

**Stale behavior:** Open shadow trades are not included in resolved outcome rates.

**Fallback:** No matured trades yields `Insufficient resolved sample`.

**Final API:** `GET /api/v1/shadow/summary`, `GET /api/v1/shadow/trades`.

## 13. Data quality

**Sources:** `prices`, `data_coverage`, `data_gaps`, `features`, `labels`, `ingestion_log`.

**Required fields:**

- symbol/timeframe
- latest candle UTC and age
- coverage start/end and row count
- missing/duplicate bar counts
- unresolved gap severity
- ingestion latency
- feature version and latest feature UTC
- label version and latest label UTC

**Stale behavior:** Thresholds are timeframe-aware.

**Fallback:** Missing feature/label tables are expected in some modes and must be labeled by
workflow context.

**Final API:** `GET /api/v1/data-quality`.

## 14. Safety events and action required

**Sources:** `risk_events`, `reconciliation_events`, `bot_events`, `system_status`,
`model_lifecycle_events`, control action status.

**Required fields:**

- event/issue ID
- severity and reason code
- component
- affected attribution IDs
- message and operational consequence
- created/last-seen UTC
- resolved/acknowledged state where supported
- recommended investigation route

**Stale behavior:** The evaluation process itself has a heartbeat. If it is stale, Action
Required displays `Safety evaluation stale`.

**Fallback:** Empty means no persisted events in the period, not proof of health.

**Final API:** `GET /api/v1/safety/events`, `GET /api/v1/actions-required`.

## 15. Server health

**Sources:** `bot_status`, `bot_events`, Prometheus; Grafana deep links.

**Required fields:**

- component/service
- host, PID/container
- status
- started UTC, heartbeat UTC, uptime
- restart count
- ingestion latency or queue lag
- error count/rate
- CPU/memory only from observability layer
- dashboard/Grafana/log links

**Stale behavior:** Missing heartbeat transitions running to stale.

**Fallback:** SQLite-only process state remains available if Prometheus is unavailable.

**Final API:** `GET /api/v1/system/health`.

## 16. Audit and controls contract

### Command request

```json
{
  "action": "pause_model",
  "target_type": "model",
  "target_id": "model-123",
  "reason": "Operator investigation",
  "expected_state_version": "2026-06-25T12:00:00Z",
  "confirmation_token": "short-lived-token"
}
```

### Command response

```json
{
  "command_id": "cmd-123",
  "status": "accepted",
  "requested_by": "operator",
  "requested_at_utc": "2026-06-25T12:01:00Z",
  "correlation_id": "corr-123"
}
```

The UI does not assume acceptance means completion. It polls or subscribes to command status.

**Required audit fields:**

- command/audit ID
- actor and role
- action
- target type/ID and attribution IDs
- requested, processed, completed UTC
- reason
- before/after state summary
- confirmation and re-auth method, never secret material
- source IP/session ID where appropriate
- status/result/error
- correlation ID

**Final API:**

- `POST /api/v1/commands`
- `GET /api/v1/commands/{command_id}`
- `GET /api/v1/audit`

## 17. Debug contract

`GET /api/v1/debug/{entity_type}/{entity_id}` is admin-only, redacted, rate-limited, and
disabled in normal operator mode if not required. It must remove credentials, secrets, account
identifiers that are not operationally needed, and private exchange payload fields.

## 18. Recommended read models

Future implementation should create adapter-level read models rather than forcing the frontend
to reconstruct business joins:

- `cockpit_status_view`
- `capital_risk_summary_view`
- `active_models_view`
- `open_trades_protection_view`
- `proposal_decisions_view`
- `reconciliation_summary_view`
- `action_required_view`
- `audit_actions_view`

They may be SQL views, Python query services, or FastAPI response assemblers. They must remain
read-only and preserve the underlying attribution IDs.
