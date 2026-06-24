# Skill: train-models

## When to use
Create candidate models, or top up the pool when active models fall below target.

## Preconditions
- Data present and gap-checked (see download-data).
- Feature store built: `python src/feature_store.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h`

## Commands
```bash
# Per-symbol (default for strict isolation)
python src/train.py --training-scope per-symbol --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h

# Multi-symbol (one model, symbol_code feature)
python src/train.py --training-scope multi-symbol --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h

# Maintain the pool automatically (train -> validate -> backtest, retain accepted)
python src/model_maintenance.py --training-scope per_symbol --symbols BTCUSDT ETHUSDT SOLUSDT \
    --timeframes 15m 1h 4h --target-accepted-models 5 --max-attempts 50
```

## Verification
`SELECT status, COUNT(*) FROM model_registry GROUP BY status;` shows accepted/rejected candidates
with reasons; accepted models have lifecycle metrics.

## Red-lines
Temporal splits only; embargo `LOOKAHEAD_BARS`; never tune on final OOS; keep a **pool** (do not
collapse to one model); never delete rejected models. See [docs/MODELING.md](../../docs/MODELING.md).
