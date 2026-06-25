# Current state: implemented vs aspirational

This is a code-grounded snapshot. It distinguishes what runs today from what the
[`initial_roadmap`](initial_roadmap) aspires to. Read this before changing code.

## Summary matrix

| Component | Implemented? | Runs in loop? | Notes |
|---|---|---|---|
| prediction -> proposal -> allocation -> trade -> order -> fill | Yes | Yes | Full attribution chain persisted |
| Capital allocator (competitive, shared pool) | Yes | Yes | Real exposure/score gates, resize, shadow on reject |
| Risk manager hard gates | Yes | Yes | Pre-execution validation |
| Execution engine (local_paper + testnet) | Yes | Yes | Single order path; calls Binance testnet API |
| Trade-lifecycle acceptance gating | Yes | n/a | `historical_trade_simulator.py` mirrors paper lifecycle |
| Shadow trades on rejected proposals | Yes | Yes | Recorded, not executed (by design) |
| Virtual TP/SL/expiry exits | Yes | **Yes** (Phase A) | exit pass runs in `trading_bot.py`; local_paper executes closes |
| Flatten of `CLOSING` trades + realized PnL | Yes | Yes (Phase A) | `ExecutionEngine.close_trade` + `ledger.mark_trade_closed` book PnL |
| Emergency stop monitoring | Yes | Yes (Phase A) | emergency-SL folded into `evaluate_virtual_exits` |
| Portfolio/equity persistence across runs | Yes | Yes | `PortfolioManager.snapshot`; reloaded on init; final snapshot after exits |
| Reconciliation | Snapshot-only | Once/run | No Binance fill replay / order-id matching |
| Live trading | Gated stub | No | Safety module; never executes (expected) |
| Native return/quantile/MFE-MAE models | Yes (Phase B) | Yes (paper + acceptance gate) | Trained per artifact and per validation fold; used by live prediction and the OOS-backtest acceptance gate. Calibration included |
| Tests / CI | **Minimal** | n/a | A few `tests/` (dashboard, training-safety, + Phase A smoke); no CI workflows yet |
| Dashboard | Partial | Yes | Professional Streamlit, but read-mostly; not full control panel |

## Verified facts (with evidence)

- **Model:** `LGBMClassifier`, `objective=multiclass`, 3 classes SHORT/FLAT/LONG. Labels are
  `triple_barrier_tp_sl_v2` (TP = 1.5x ATR, SL = 1.0x ATR, lookahead 6 bars).
  See `config.py:262-278` (MODEL_PARAMS) and `labels.py`.
- **Features (Phase C complete, v5):** **82 features** — volatility/regime/momentum,
  microstructure from taker data, leakage-safe cross-asset BTC context, and **multi-timeframe**
  (two higher TFs via close-time as-of). A leakage-safe external-data layer (`external_data.py`:
  funding/OI/fear-greed) exists but is opt-in (`ENABLE_EXTERNAL_DATA=false`) and not in the default
  training contract by design. See [DATA_AND_FEATURES.md](DATA_AND_FEATURES.md).
- **Phase D complete:** model degradation/quarantine lifecycle, shadow-trade outcome analytics,
  walk-forward stability mode, and CI. A `preflight.py` readiness check + `docs/LIVE_PAPER_RUN.md`
  prepare the 3-day local Binance-testnet paper run (15m/1h/4h).
- **Prediction distribution (Phase B native models):** `train._train_native_models` now trains
  LightGBM expected-return regression, quantile (q05…q95) and MFE/MAE regressors, stored in the
  artifact under `native_models`. `prediction_engine.build_structured_prediction` prefers these
  real outputs (cost-adjusted, monotonic quantiles) and falls back to the synthetic
  `build_prediction_from_probabilities` when a model has no native sub-models. Probability
  calibration also landed: `train._fit_probability_calibrator` stores a `calibrator` in the
  artifact and `modeling_utils.predict_class_probabilities` applies it in the live loop.
  **Phase B is complete:** `validate_model` trains native + calibrated models **per fold** and
  persists the calibrated probabilities + cost-adjusted, monotonic native distribution to
  `validation_predictions`; `backtest --mode oos` and the historical simulator consume those
  fields via `build_prediction_from_row_fields`, so the acceptance gate evaluates the same
  native predictions as paper (no mixing of final-artifact models with per-fold probabilities).
- **Orphaned exit loop (FIXED in Phase A):** as found, `exit_manager.evaluate_virtual_exits()`
  only set a trade to `CLOSING` and `trading_bot.py` never called it. Phase A wires an exit pass
  into the loop and adds `ExecutionEngine.close_trade` to flatten `CLOSING` trades and book
  realized PnL (local_paper). Testnet exits remain exchange-side/reconciled (Phase F).
- **Equity persistence (CLARIFIED/CONFIRMED):** `PortfolioManager.snapshot` persists
  `portfolio_snapshots` and the constructor reloads the latest, so equity does carry across runs;
  Phase A adds a final snapshot after the exit pass and mark-to-market of open trades. The earlier
  concern was really that round-trip realized PnL was never booked (no exits) — now fixed.
- **Reconciliation** (`reconciliation_engine.py`) snapshots balances and counts open trades; it
  does not fetch/replay Binance orders/fills or match exchange order IDs.
- **Safety is real:** `broker_client.real_execution_client()` requires
  `ENABLE_LIVE_TRADING and ENABLE_REAL_ORDER_EXECUTION and ENABLE_REAL_BINANCE_ACCOUNT and not DRY_RUN`.
  Defaults (`config.py`, `.env.example`) keep all of these off.
- **Tests are minimal:** a few `tests/` exist (dashboard services, training safety, and the new
  Phase A paper-loop smoke test). There is no CI yet (Phase D).

## Why this ordering matters

Because realized PnL is not booked and equity does not persist, **you currently cannot judge
whether any model has edge** — the accounting is broken. That is why [ROADMAP.md](ROADMAP.md)
Phase A ("trustworthy paper loop") runs before richer models, features, or external data.

## What is genuinely solid

The hard architectural decisions are already correct and wired: shared-pool competitive
allocation (not consensus averaging), full per-trade attribution, single execution path,
trade-lifecycle-based acceptance gating, shadow trades, preserved rejected models/proposals,
and layered live-trading safety. The remaining work is completing accounting/exits, replacing
synthetic predictions with native models, broadening data/features, and hardening security and
the UI — not re-architecting.
