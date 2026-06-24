# Skill: validate-backtest

## When to use
After training, to judge a model on the **full trade lifecycle** (not one-bar returns) before it
enters paper.

## Preconditions
A trained model id exists in `model_registry`.

## Commands
```bash
python src/validate_model.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --model-id MODEL_ID
python src/backtest.py --mode oos --timeframe 1h --model-id MODEL_ID
# Optional funnel view
python src/validation_funnel_report.py
```

## Verification
Acceptance reads `trade_lifecycle.metrics` (total return, Sharpe, profit factor, drawdown, trade
count). Reports land in `reports/` (`validation_summary*.json`, `backtest_oos_summary*.json`). A
missing lifecycle evaluation is a hard rejection (`missing_trade_lifecycle_evaluation`).

## Red-lines
Acceptance must use lifecycle metrics, never one-bar `fwd_return_1`. No tuning on the final OOS
set. See [docs/DECISIONS/0002-trade-lifecycle-gating.md](../../docs/DECISIONS/0002-trade-lifecycle-gating.md).
