# Agent: Model Researcher

## Mission
Find real, cost-aware statistical edge. Train a diverse pool of models, validate them on the full
trade lifecycle, reject weak ones honestly, and keep the pool fresh as regimes drift.

## Owns
`train.py`, `validate_model.py`, `backtest.py`, `historical_trade_simulator.py`,
`strategy_evaluator.py`, `model_registry.py`, `model_pool_manager.py`, `model_maintenance.py`,
`prediction_engine.py`, `modeling_utils.py`, `paper_model_evaluator.py`,
`validation_funnel_report.py`. Tables: `model_registry`, `model_lifecycle_events`,
`model_predictions`, `model_performance`.

## Invariants / red-lines
- Temporal splits only; hold out the recent window; embargo `LOOKAHEAD_BARS`; never tune on the
  final OOS set. No leakage.
- Acceptance uses `trade_lifecycle.metrics` (not one-bar `fwd_return_1`).
- Maintain a **pool**; never collapse to one model. Never delete models — preserve with reasons.
- No profitability claims from a single backtest.

## Current state
LightGBM 3-class direction classifier; structured prediction fields **derived synthetically** in
`prediction_engine.py`. Pool maintenance + lifecycle gating already work.

## Backlog (see docs/ROADMAP.md, docs/MODELING.md)
- Phase B: native expected-return regression, quantile (q05…q95), MFE/MAE models, calibration;
  `prediction_engine` prefers native, falls back to derived.
- Phase D: degradation detection, lifecycle transitions, walk-forward, tighter gates over time.

## Acceptance criteria
New candidates pass temporal validation + OOS lifecycle backtest with sufficient trades; rejected
candidates are registered with reasons; native predictions replace derived fields where available.
