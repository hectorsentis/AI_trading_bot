# Modeling strategy

## Is the current model adequate?

**Not yet — for the architecture's ambitions.** Today there is a single LightGBM 3-class
direction classifier (SHORT/FLAT/LONG) trained on triple-barrier TP/SL labels, with ~46–56
OHLCV-only features. The structured prediction contract (`expected_return_pct`, quantiles,
MFE/MAE, horizon) is **derived synthetically** from the classifier's probabilities plus
volatility (`prediction_engine.build_prediction_from_probabilities`). That is a deliberate,
self-documented placeholder.

Why it is not adequate:

- The allocator, trade builder and risk manager are designed to consume a **return
  distribution** (expected return, adverse move, quantiles, MFE/MAE). Feeding them values
  synthesized from a direction classifier means TP/SL sizing and allocation quality are only as
  good as a heuristic, not a trained estimate.
- A direction classifier optimizes class separation, not net-of-cost expectancy. Edge in this
  domain is small and cost-sensitive; the objective should reflect that.
- A single model on one objective/timeframe/feature-family cannot express the diversity the
  shared-pool allocator exists to exploit.

## Target model design: a pool of small specialized models

Each "mini-bot" is a combination keyed by:

```
(symbol, timeframe, horizon, feature_family, label_family, objective, model_config)
```

Supported objectives (build incrementally — Phase B onward):

```
direction_model            calibrated 3-class direction (current)
expected_return_model      regression of net-of-cost return over the horizon
quantile_model             q05/q25/q50/q75/q95 of horizon return (LightGBM quantile)
mfe_model / mae_model      expected favorable / adverse excursion
regime_model               trend / volatility / liquidity regime
calibration_model          probability calibration layer
trade_quality_model        net trade EV given proposal features (allocator input)
```

`prediction_engine.py` should prefer **native** model fields and fall back to the derived path
only when a native model is absent. This keeps the architecture intact while removing the
synthetic crutch.

## Train new models or keep retraining one?

**Maintain a pool. Never collapse to a single retrained model.** The shared-pool, competitive-
allocation design *requires* diversity — averaging or a single model defeats the entire point.

Do both, continuously:

1. **Retrain each model slot** on rolling **walk-forward** windows so models track regime drift.
   The most recent `VALIDATION_WINDOW_HOURS` is always held out, with a `LOOKAHEAD_BARS` embargo.
2. **Add new diverse candidates** (new feature families, horizons, objectives, symbols) so the
   pool keeps exploring for edge.
3. **Never delete** models — rejected/degraded models stay in the registry with reasons for
   audit. The allocator and lifecycle manager promote good models and degrade/quarantine bad
   ones; they do not get erased.

This is what `model_maintenance.py` + `model_registry.py` + `capital_allocator.py` are built for.

## Acceptance gating (already correct; keep it)

A candidate is accepted only after passing the **full trade lifecycle** (not one-bar
`fwd_return_1`) via `historical_trade_simulator.py`, evaluated by `strategy_evaluator.py`.
Current gates (`config.py`):

```
MIN_ACCEPTABLE_SHARPE = 0.20
MIN_ACCEPTABLE_PROFIT_FACTOR = 1.05
MAX_ACCEPTABLE_DRAWDOWN = 0.20
MIN_ACCEPTABLE_TRADES = 10
REQUIRE_OUTPERFORM_BASELINE = True
REQUIRE_OOS_FOR_ACCEPTANCE = True
```

These thresholds are intentionally low to admit candidates into paper. **They are not a
profitability claim.** Tighten them, and lean on continuous paper validation
(`paper_model_evaluator.py`: `MIN_PAPER_VALIDATION_DAYS/TRADES`, `PAPER_MIN_PROFIT_FACTOR`,
etc.) before any model becomes `real_ready`.

## Anti-overfitting & leakage discipline (non-negotiable)

```
temporal splits only — never random
hold out the most recent window for validation/OOS
embargo of LOOKAHEAD_BARS before validation so triple-barrier labels do not peek
features at t use only information available up to t
no normalization using future data
never tune on the final OOS set
preserve rejected models; never claim profit from one good backtest
```

## Per-symbol vs multi-symbol

Both are supported (`TRAINING_SCOPE = per_symbol | multi_symbol | both`). `per_symbol` is the
default for strict isolation; `multi_symbol` uses `symbol_code` as a feature for cross-symbol
transfer. The pool can hold both kinds and let the allocator compare their live behavior.
