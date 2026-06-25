# 3-day local paper-trading run (Binance Spot testnet/demo)

This runbook launches the autonomous bot **locally** (not a VPS) for a multi-day paper run across
the **15m, 1h and 4h** timeframes, and tells you exactly what to verify. It sends **no real
orders** — only Binance testnet/demo paper orders. The pre-flight check confirms readiness.

## 0. Prerequisites (`.env`)

```env
DRY_RUN=true
ENABLE_LIVE_TRADING=false
ENABLE_REAL_ORDER_EXECUTION=false
ENABLE_REAL_BINANCE_ACCOUNT=false
ENABLE_TESTNET_PAPER_TRADING=true
KILL_SWITCH_ENABLED=true
BINANCE_TESTNET_API_KEY=...        # demo/testnet key (trade-enabled, withdrawal-disabled)
BINANCE_TESTNET_API_SECRET=...
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
TIMEFRAMES=15m,1h,4h
```

## 1. One-time preparation

```powershell
# DB schema
python src/db_utils.py --init --check-schema

# Backfill history for all three timeframes (training needs history)
python src/download_data.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 15m --mode full
python src/download_data.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h  --mode full
python src/download_data.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 4h  --mode full

# Build the feature store at the CURRENT feature version (full rebuild — the feature set was
# expanded to v5; older rows must be repopulated so models can train on the full contract)
python src/feature_store.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframes 15m 1h 4h --full-rebuild
```

## 2. Pre-flight readiness check

```powershell
python src/preflight.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframes 15m 1h 4h
```

Expect `READY` with `FAIL=0`. It verifies: safety flags (no real order possible), DB schema, risk
parameters, **real Binance connection (public market data)**, **bidirectional testnet account read**,
exchange filters, data coverage (warns where the bot will backfill), feature-store version, model
pool, and that the ingestor/maintenance/trading-bot/runner import cleanly.

Optional — verify bidirectional order **write** on testnet (places ONE controlled order, refuses if
any real flag is on):

```powershell
python src/paper_demo_probe.py --symbol BTCUSDT --timeframe 1h
```

## 3. Launch the autonomous bot (keep running ~3 days)

```powershell
python src/autonomous_runner.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframes 15m 1h 4h
```

This supervises, restart-safe, all of:
- `realtime_ingestor` — live candle ingestion + storage for 15m/1h/4h
- `trading_bot` (`--mode paper --paper-mode per-model --loop`) — sync, refresh features, predict →
  propose → allocate → risk → execute (testnet) → exits → ledger
- `model_maintenance` — continuous training/validation/backtest to keep the accepted-model pool full
- `paper_model_evaluator` — paper validation + degradation/quarantine lifecycle + shadow analytics
- `dashboard` — Streamlit control panel at http://localhost:8501

Leave the terminal open. Press `Ctrl+C` once after ~3 days for a graceful shutdown.

## 4. What to verify while it runs

| You asked to check | Where to see it |
|---|---|
| Real Binance connection, **bidirectional** | preflight `binance_testnet_read=OK`; dashboard Binance-sync panel; `account_snapshots` / `balance_snapshots` tables grow |
| **New data stored live** | `prices` latest `datetime_utc` advances each timeframe; `ingestion_log`; `data_coverage`; ingestor log |
| **Model retraining operative** | `model_registry` gains candidates/accepted over time; `model_lifecycle_events`; `model_maintenance.log` |
| **Seeks profit** | `trades` (OPEN→CLOSED with realized PnL), `model_performance`, `paper_model_metrics`, dashboard equity curve |
| All parameters working | preflight `risk_params`/`universe`/`retraining_config=OK`; risk rejections in `risk_events` |
| Safety | preflight `safety_flags=OK`; `platform_checks.py` → `real_orders_blocked_by_default: true` |

Quick status / logs:

```powershell
.tools\status.cmd
Get-Content logs\services\trading_bot.log -Tail 40
sqlite3 data\db\market_data.sqlite "SELECT status, COUNT(*) FROM model_registry GROUP BY status;"
sqlite3 data\db\market_data.sqlite "SELECT model_id, account_mode, status, realized_pnl_usdt FROM trades ORDER BY updated_at_utc DESC LIMIT 20;"
```

## Notes & honest expectations

- The first cycles will be busy **backfilling data and training candidates**; trades appear once
  accepted models exist and signals pass the proposal/allocator/risk gates.
- `TARGET_ACCEPTED_MODELS` and `*_INTERVAL_SECONDS` (see `config.py`) control pool size and cadence.
- **No profitability is implied.** A 3-day window is a smoke/operability test, not statistical
  proof of edge — judge operability (data flowing, models training, trades booking PnL, lifecycle
  transitions firing), not the PnL sign.
- Local only: keep the machine awake; `data/`, `logs/`, `models/`, `reports/` persist between
  restarts (the runner is restart-safe).
