# Agent role packs

Each file here defines an **agent role** scoped to one layer of the platform. A role pack lists:
its mission, the modules it owns, the invariants/red-lines it must never violate, the current
state, the backlog, and acceptance criteria.

These are working briefs for whoever (human or AI agent) is operating on that layer. They are
deliberately aligned with — and subordinate to — the single source of truth:
[`docs/ROADMAP.md`](../docs/ROADMAP.md). When the roadmap changes, update the roadmap; the packs
reference phases rather than restating them.

## Roles

| Pack | Layer | Owns (src/) |
|---|---|---|
| [data-engineer](data-engineer.md) | Market data | download_data, realtime_ingestor, data_loader, data_quality_service, data_check, data_gap_fill, coverage_report |
| [feature-label-engineer](feature-label-engineer.md) | Features/labels | features, technical_patterns, labels, feature_store |
| [model-researcher](model-researcher.md) | Models | train, validate_model, backtest, historical_trade_simulator, strategy_evaluator, model_registry, model_pool_manager, model_maintenance, prediction_engine, modeling_utils, paper_model_evaluator |
| [proposal-allocator](proposal-allocator.md) | Proposal/allocation | trade_proposal_engine, capital_allocator, trade_builder, signal_engine |
| [risk-safety](risk-safety.md) | Risk/safety | risk_manager, kill_switch, reconciliation_engine, trade_protection, platform_checks |
| [execution-broker](execution-broker.md) | Execution/broker | broker_client, execution_engine, paper_trading_engine, live_trading_engine |
| [ledger-portfolio](ledger-portfolio.md) | Accounting/exits | ledger, portfolio_manager, exit_manager, stop_manager |
| [dashboard-ops](dashboard-ops.md) | UI/orchestration | dashboard*, runtime_status, autonomous_runner, trading_bot |
| [security-engineer](security-engineer.md) | Security (Phase F) | dashboard_auth, broker_client, execution_engine, risk_manager, kill_switch + CI/tests |
| [ui-engineer](ui-engineer.md) | UI rebuild (Phase G) | dashboard, dashboard_data, dashboard_controls, runtime_status |

## Global red-lines (apply to every role)

```
no model sends/cancels orders directly; execution_engine is the only order path
no real order unless ENABLE_LIVE_TRADING & ENABLE_REAL_ORDER_EXECUTION & ENABLE_REAL_BINANCE_ACCOUNT & not DRY_RUN
never bypass RiskManager, KillSwitch, or reconciliation
temporal splits only; no leakage; never tune on final OOS
never delete models/proposals/risk events; do not overwrite history silently
no secrets, DB, model artifacts, logs or reports committed to git
never claim guaranteed profitability
```
