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

- [ ] Expand feature families toward the roadmap: multi-timeframe (only closed higher-TF
  candles), cross-asset BTC/ETH context, regime, volatility, microstructure proxies from taker
  data.
- [ ] Add optional **leakage-safe external ingestion** (funding rate, open interest,
  fear/greed) as separate timestamped tables joined as-of. New `src/external_data.py`.
- [ ] Version every feature set; never normalize using future data.

## Phase D — Pool / lifecycle / allocator maturity + walk-forward + CI
Goal: continuous, self-maintaining model pool.

- [ ] Model degradation detection and lifecycle transitions (`paper_degraded`, `quarantined`).
- [ ] Shadow-trade analytics surfaced in the dashboard (rejected-but-would-have-won, etc.).
- [ ] Walk-forward in `backtest.py --walk-forward`.
- [ ] Expand `tests/`; add GitHub Actions CI.

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

- [ ] Operator-first IA covering the 13 sections; answers running/safe/connected/capital/
  winners/open/failed/action-needed at a glance.
- [ ] Real, audited, re-auth-gated controls (pause/resume/quarantine/archive, disable entries,
  shadow-only, manual close with confirmation, kill switch). Live trading never a casual button.
- [ ] Live data from Phase-A persisted snapshots; UTC + units + color semantics; no raw JSON.
- [ ] ADR on Streamlit-harden vs web-stack migration ([DECISIONS/0004-ui-stack-choice.md](DECISIONS/0004-ui-stack-choice.md)).

---

## Strategic answers (rationale lives in the linked docs)

- **Is the model adequate?** Not yet — a single 3-class classifier with derived distributions is
  a placeholder. Target: a pool of small specialized models. See [MODELING.md](MODELING.md).
- **Train new vs retrain one?** Maintain a pool; never one monolithic model. Retrain each slot on
  rolling walk-forward windows AND add new diverse candidates; never delete. See
  [MODELING.md](MODELING.md).
- **Profitability:** never guaranteed; earned only through cost-aware lifecycle backtest +
  continuous paper validation on real accounting + diversity + strict risk control.
