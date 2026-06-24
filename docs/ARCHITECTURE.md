# Architecture

## What this system is

A local-first, server-ready **autonomous algorithmic trading platform for Binance Spot**.
It is not a signal script or a single classifier. Multiple independent models analyze the
market, emit structured predictions, and turn them into **trade proposals** that compete for
a **shared capital pool**. A central allocator decides; a central risk manager gates; a single
execution path places orders; a ledger attributes every fill's PnL back to the model that
produced it.

Core principle:

```
No model trades directly.
All models propose.
The system decides.
The ledger proves.
```

## End-to-end pipeline

```
Market data (Binance REST klines)
  -> Data quality / gap / coverage checks
  -> Feature store (versioned features)        features.py, technical_patterns.py, feature_store.py
  -> Labels (triple-barrier TP/SL)             labels.py
  -> Train / validate / backtest               train.py, validate_model.py, backtest.py
       (acceptance gated on full trade lifecycle via historical_trade_simulator.py)
  -> Model registry / pool maintenance         model_registry.py, model_maintenance.py, model_pool_manager.py
  -> Prediction (structured contract)          prediction_engine.py
  -> Trade proposal                            trade_proposal_engine.py
  -> Capital allocator (shared pool)           capital_allocator.py
  -> Risk manager (hard gates)                 risk_manager.py
  -> Trade builder (TP/SL/emergency SL)        trade_builder.py
  -> Execution engine (ONLY order path)        execution_engine.py -> broker_client.py
  -> Exit / stop management                    exit_manager.py, stop_manager.py
  -> Reconciliation                            reconciliation_engine.py
  -> Ledger / portfolio / PnL attribution      ledger.py, portfolio_manager.py
  -> Dashboard / control panel                 dashboard.py
Orchestration: trading_bot.py, autonomous_runner.py
```

## Attribution chain

Every operational object preserves identity so the system can always answer "which model
made or lost money, on which proposal, through which fill":

```
model_id -> prediction_id -> proposal_id -> allocation_id -> trade_id -> order_id -> fill_id
```

These IDs are persisted across `model_predictions`, `trade_proposals`, `allocations`,
`trades`, `orders`, `fills`, and aggregated in `model_performance`.

## Module map (by responsibility)

| Layer | Modules |
|---|---|
| Data | `download_data.py`, `realtime_ingestor.py`, `data_loader.py`, `data_quality_service.py`, `data_check.py`, `data_gap_fill.py`, `coverage_report.py` |
| Features / labels | `features.py`, `technical_patterns.py`, `labels.py`, `feature_store.py` |
| Models | `train.py`, `validate_model.py`, `backtest.py`, `historical_trade_simulator.py`, `strategy_evaluator.py`, `model_registry.py`, `model_pool_manager.py`, `model_maintenance.py`, `prediction_engine.py`, `modeling_utils.py`, `paper_model_evaluator.py`, `validation_funnel_report.py` |
| Proposal / allocation | `trade_proposal_engine.py`, `capital_allocator.py`, `trade_builder.py`, `signal_engine.py` |
| Risk / safety | `risk_manager.py`, `kill_switch.py`, `reconciliation_engine.py`, `trade_protection.py`, `platform_checks.py` |
| Execution / broker | `broker_client.py`, `execution_engine.py`, `paper_trading_engine.py`, `live_trading_engine.py`, `paper_demo_probe.py` |
| Portfolio / ledger / exits | `portfolio_manager.py`, `ledger.py`, `exit_manager.py`, `stop_manager.py`, `capital_allocator.py` |
| Orchestration | `trading_bot.py`, `autonomous_runner.py`, `runtime_status.py` |
| Dashboard | `dashboard.py`, `dashboard_data.py`, `dashboard_controls.py`, `dashboard_auth.py` |
| Infra | `config.py`, `db_utils.py`, `install_setup.py`, `temporal_utils.py` |

## Persistence (SQLite is the audit source of truth)

Binance account state is the **operational** source of truth for balances/orders/fills;
SQLite is the **historical/audit** source of truth for decisions, predictions, proposals,
allocations, trades, ledger and risk events. Key tables (created by `db_utils.py`):

- Data: `prices`, `data_coverage`, `data_gaps`, `ingestion_log`, `features`
- Models: `model_registry`, `model_lifecycle_events`, `model_predictions`, `model_performance`
- Trading: `trade_proposals`, `allocations`, `trades`, `orders`, `fills`, `signals`, `positions`
- Accounting: `portfolio_snapshots`, `account_snapshots`, `balance_snapshots`
- Shadow: `shadow_trades`, `shadow_trade_events`
- Safety/ops: `risk_events`, `reconciliation_events`, `bot_events`, `bot_status`, `system_status`
- Dashboard control: `bot_control_actions`, `model_control`, `runtime_config`, `runtime_config_audit`

History is never silently overwritten; rejected models, rejected proposals and risk events
are preserved.

## Operating modes

```
research                 download/features/labels/train/validate/backtest — no orders
local_dry_run            same architecture, locally simulated fills
binance_spot_testnet     testnet/demo orders, no real risk
shadow_real              real market data, no real orders
live                     disabled by default; requires all explicit flags
```

The only difference between modes is the execution venue; the prediction -> proposal ->
allocation -> risk -> build -> ledger logic is identical across all of them.

## Safety model

Default state cannot send a real order. Real execution requires **all** of:

```
ENABLE_LIVE_TRADING=true
ENABLE_REAL_ORDER_EXECUTION=true
ENABLE_REAL_BINANCE_ACCOUNT=true
DRY_RUN=false
```

Even then, every real path still passes `KillSwitch`, `RiskManager`, reconciliation checks
and stale-data gates. `broker_client.py` separates public / testnet / real-read / real-execute
clients so a mode mix-up cannot send a real order in testnet mode or vice versa. See
[SECURITY.md](SECURITY.md) for the full hardening model (Phase F).
