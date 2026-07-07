# Roadmap (single source of truth)

This is the sequenced, executable roadmap. `.agents/`, `.skills/` and `.codex/` reference this
file rather than restating it, to avoid drift. The long-form target vision is
[`initial_roadmap`](initial_roadmap); this document organizes it into phases.

Guiding rule: **trustworthy paper loop first.** You cannot evaluate edge on broken accounting,
and you cannot risk money without security and a real control panel. Phases run roughly in
order, but security (F) and UI (G) are **mandatory prerequisites before any real-trading
consideration**.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done. Keep
[`.codex/PROGRESS.md`](../.codex/PROGRESS.md) in sync.

---

## Phase A — Trustworthy paper loop
Goal: paper PnL/equity is real, persistent, and exits actually execute.

- [x] **A1 — Wire & execute exits.** `trading_bot.py` runs an exit pass after the per-model
  loop; `ExecutionEngine.close_trade` flattens `CLOSING` trades (attributed SELL order + fill),
  books realized PnL via `ledger.mark_trade_closed`, and updates `portfolio_manager`.
  `exit_manager.evaluate_virtual_exits` now also handles emergency-SL and horizon expiry.
  (local_paper executes the close; testnet exits remain exchange-side/reconciled — Phase F.)
- [x] **A2 — Persist equity.** `PortfolioManager.snapshot` persists `portfolio_snapshots` and
  the constructor reloads the latest, so equity carries across runs; a final snapshot is taken
  after the exit pass. (`account/balance_snapshots` come from `reconciliation_engine` at start.)
- [x] **A3 — Realized/unrealized PnL.** `ledger.update_open_trade_unrealized` marks open trades
  to market; `refresh_model_performance` aggregates booked realized PnL.
- [x] **A4 — Regression guard.** `tests/test_paper_loop_smoke.py` asserts
  `OPEN -> CLOSING -> CLOSED` with booked PnL, attributed close order/fill, and persisted,
  carried-over equity. Establishes the `tests/` harness (`pytest.ini`, `requirements-dev.txt`).

## Phase B — Native prediction models
Goal: replace synthetic distribution fields with trained outputs.

- [x] `labels.compute_native_regression_targets` — leakage-safe forward return / MFE / MAE
  targets from the per-symbol close path (embargo-protected).
- [x] `train._train_native_models` trains LightGBM **expected-return regression**, **quantile**
  models (q05…q95) and **MFE/MAE** regressors, stored in the artifact under `native_models`
  (gated by `ENABLE_NATIVE_PREDICTION_MODELS`, lighter `NATIVE_MODEL_PARAMS`).
- [x] `prediction_engine.build_prediction_from_native` + `build_structured_prediction`
  dispatcher: native fields when present (cost-adjusted, monotonic quantiles), falls back to
  `build_prediction_from_probabilities` otherwise. `trading_bot.py` passes
  `native_models` + `feature_frame`. Tests: `tests/test_native_prediction.py`.
- [x] **Probability calibration** for the direction classifier. `train._fit_probability_calibrator`
  (cross-validated `CalibratedClassifierCV`, flagged by `ENABLE_PROBABILITY_CALIBRATION`) stores a
  `calibrator` in the artifact; `modeling_utils.predict_class_probabilities` uses it in the live
  loop (canonical class order, raw-model fallback). Tests: `tests/test_calibration.py`.
- [x] **Native models + calibration wired into validation/backtest acceptance** (ADR 0002
  consistency). `validate_model` trains native + calibrated models **per fold** (no model
  mixing), computes calibrated probabilities + cost-adjusted, monotonic native distribution per
  test row, and **persists** them to `validation_predictions` (columns added idempotently via
  `_ensure_column`). `backtest --mode oos`, `load_oos_predictions` and the
  `historical_trade_simulator` reload and consume those fields via
  `prediction_engine.build_prediction_from_row_fields`, falling back to the derived path only
  when a row lacks native fields. Shared math: `prediction_engine.assemble_native_fields`.
  Tests: `tests/test_phaseb_closure.py`.

**Phase B is complete.** Native return-distribution models + probability calibration power
training, the live paper loop, and the validation/backtest acceptance gate end-to-end (per-fold,
no model mixing). Cost knobs: `ENABLE_NATIVE_PREDICTION_MODELS`, `ENABLE_PROBABILITY_CALIBRATION`.

## Phase C — Features & external data
Goal: broaden the feature store with leakage discipline.

- [x] **Single-symbol expansion** (`features.py`, +21 features → `FEATURE_VERSION=v4`): volatility
  (`volatility_50`, `volatility_ratio_20_50`, `downside_volatility_20`, `volatility_regime_score`),
  regime/trend (`dist_ma_50/200`, `price_above_sma_50`, `trend_strength_50`, `rolling_drawdown_50`,
  `dist_from_high_50`), momentum (`ret_24`, `roc_10`, `rsi_7`).
- [x] **Microstructure from taker data** (`taker_buy_ratio`, `taker_imbalance`,
  `taker_imbalance_zscore_20`, `avg_trade_size_zscore_20`); neutral fallback when taker columns
  are absent. `feature_store`/`features` now load taker/quote/trades columns.
- [x] **Cross-asset BTC context** (`btc_ret_24`, `rel_strength_vs_btc_24`, `corr_btc_50`,
  `beta_btc_50`) via leakage-safe backward as-of merge of the reference close
  (`CROSS_ASSET_REFERENCE_SYMBOL`); neutral when context absent. `feature_store.load_reference_close`.
- [x] **Leakage-safe external ingestion** — new `src/external_data.py`: `external_data` table,
  `attach_external_features_asof` (backward as-of join primitive), and flag-gated ingestion for
  fear/greed, funding rate and open interest (`ENABLE_EXTERNAL_DATA`, off by default; network).
- [x] Versioned (`v4_regime_microstructure_crossasset`); strict leakage discipline (tested:
  `tests/test_features_phasec.py`, `tests/test_external_data.py`). New features are additive, so
  existing `v3` models keep working; run `feature_store --full-rebuild` to populate v4 columns.
- [x] **Multi-timeframe features** (`FEATURE_VERSION=v5`): two strictly-higher timeframes
  (`htf1`/`htf2` per `HIGHER_TIMEFRAME_MAP`) attached via leakage-safe backward as-of on the
  higher-TF **close** time (`feature_store.build_higher_timeframe_context` +
  `features._attach_higher_timeframe_features`); neutral when that TF is absent. Tested.
- [~] Folding external metrics into the training `FEATURE_COLUMNS` contract is intentionally
  **opt-in, not default**: forcing external columns into training would break it whenever
  `ENABLE_EXTERNAL_DATA=false` (the default) since the columns would be NaN. The external layer +
  `attach_external_features_asof` are ready; auto-inclusion is a config-gated future step.

**Phase C complete** (82 features: single-symbol, microstructure, cross-asset, multi-timeframe;
plus the external-data layer). External-into-training stays opt-in by design.

## Phase D — Pool / lifecycle / allocator maturity + walk-forward + CI
Goal: continuous, self-maintaining model pool.

- [x] **Degradation detection + lifecycle transitions**: `paper_model_evaluator.assess_lifecycle`
  maps paper metrics to pass / fail / **degrade** (soft pause, recoverable) / **quarantine**
  (severe); registry gains `mark_model_paper_degraded`, `mark_model_quarantined`,
  `reactivate_paper_model`; degraded models are re-evaluated each cycle and recover or worsen.
  Thresholds in `config.py` (`PAPER_DEGRADE_*`, `PAPER_QUARANTINE_*`). Tested.
- [x] **Shadow analytics**: `shadow_evaluator` resolves matured `SHADOW_OPEN` trades against OHLC
  (TP/SL/expire) and books `outcome_pnl_usdt`; `dashboard_data.load_shadow_analytics` +
  `dashboard.render_shadow_analytics` surface would-have-won rate, missed profit, avoided loss,
  by-reason. Run each evaluator cycle. Tested.
- [x] **Walk-forward** in `backtest.py --walk-forward` (`--mode walk_forward`): runs the trade
  lifecycle per OOS fold and reports stability (mean/std return, profitable-fold fraction, worst
  fold). Reuses persisted per-fold native predictions (no model mixing).
- [x] **CI** (`.github/workflows/ci.yml`): pytest + bandit + pip-audit. **Tests expanded** to 36
  (features/external/lifecycle/shadow/native/calibration/paper-loop/etc.).

**Phase D complete.**

## Phase R — Live paper-run readiness (Binance testnet, local)
- [x] `src/preflight.py` — one-command readiness check: safety flags, DB schema, risk params,
  **real Binance connection** (public) + **bidirectional testnet account read**, exchange filters,
  data coverage/staleness, feature-store version, model pool, pipeline imports. Verified `READY`.
- [x] `docs/LIVE_PAPER_RUN.md` — the 3-day local run runbook for 15m/1h/4h with the exact
  verification checklist (bidirectional Binance, live data storage, retraining, profit-seeking).
- `autonomous_runner` already supervises ingestor + bot + maintenance + evaluator + dashboard,
  restart-safe, across multiple timeframes.

## Phase E — Deployment hardening
Goal: clean Linux server deployment.

- [ ] `pathlib` / Linux-safe path audit.
- [ ] systemd unit examples + optional Dockerfile/compose; healthcheck script.
- [ ] Graceful shutdown / restart-safety review.

## Phase F — Security & cybersecurity (mandatory before real money)
Goal: defense-in-depth, full test coverage, hardened attack surface. See [SECURITY.md](SECURITY.md).

- [ ] **Secrets:** never log/echo keys; load only from env/keyring; redact in logs/UI; startup
  scan refuses to run on a committed/world-readable `.env`.
- [ ] **Auth:** harden `dashboard_auth.py` — strong hashing, rate-limit + lockout, signed
  expiring sessions, CSRF on every action, optional TOTP, audited admin role, default-deny.
- [ ] **API keys:** enforce real keys are trade-enabled, **withdrawal-disabled, IP-allowlisted**;
  healthcheck verifies no-withdraw before live is permitted.
- [ ] **Exec safety:** idempotent `client_order_id`, replay protection, server-side notional/qty
  caps, hard max-real-order circuit breaker independent of `RiskManager`.
- [ ] **Supply chain:** parameterize all SQL, pin deps with hashes, add `bandit` + `pip-audit` +
  secrets scanner to CI/pre-commit.
- [ ] **Network:** dashboard behind HTTPS/reverse-proxy, localhost-bound by default; firewall +
  fail2ban guidance.
- [ ] **Tamper-evidence:** append-only audit log of control actions, kill-switch and live-flag
  changes; reconciliation mismatch escalates to kill switch.
- [ ] **Tests:** adversarial suite — safety-gate, auth (brute-force/forgery/CSRF), SQL-injection,
  kill-switch, reconciliation-halts-trading. High coverage on `risk_manager`, `kill_switch`,
  `broker_client`, `execution_engine`, `dashboard_auth`.

## Phase G — Full UI / control-panel rebuild (mandatory before real money)
Goal: a real operator control panel, not a report. See [UI_SPEC.md](UI_SPEC.md).

Design status (2026-06-25): the Phase G product model, cockpit wireframe, component model,
visual system, data contracts, implementation sequence, and acceptance criteria are complete
under `docs/ui/`. Runtime implementation remains pending. ADR 0004 now recommends the hybrid
Streamlit-to-FastAPI/Next.js transition.

- [~] Operator-first IA covering the 13 sections; design model, cockpit wireframe, component
  model, visual system, data contracts, and acceptance criteria are complete under `docs/ui/`.
  Runtime implementation remains pending.
- [ ] Real, audited, re-auth-gated controls (pause/resume/quarantine/archive, disable entries,
  shadow-only, manual close with confirmation, kill switch). Live trading never a casual button.
- [ ] Live data from Phase-A persisted snapshots; UTC + units + color semantics; no raw JSON.
- [x] ADR on Streamlit hardening vs FastAPI + Next.js and hybrid migration
  ([DECISIONS/0004-ui-stack-choice.md](DECISIONS/0004-ui-stack-choice.md)).

---

## Strategic answers (rationale lives in the linked docs)

- **Is the model adequate?** Not yet — a single 3-class classifier with derived distributions is
  a placeholder. Target: a pool of small specialized models. See [MODELING.md](MODELING.md).
- **Train new vs retrain one?** Maintain a pool; never one monolithic model. Retrain each slot on
  rolling walk-forward windows AND add new diverse candidates; never delete. See
  [MODELING.md](MODELING.md).
- **Profitability:** never guaranteed; earned only through cost-aware lifecycle backtest +
  continuous paper validation on real accounting + diversity + strict risk control.
