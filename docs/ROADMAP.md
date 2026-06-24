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

- [ ] Add LightGBM **expected-return regression**, **quantile** models (q05…q95), and
  **MFE/MAE** regressors as registry artifacts.
- [ ] `prediction_engine.py` uses native fields when present, falls back to
  `build_prediction_from_probabilities` otherwise.
- [ ] Add **probability calibration** for the direction classifier.
- [ ] Extend `labels.py` with return / MFE / MAE / quantile targets.

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
