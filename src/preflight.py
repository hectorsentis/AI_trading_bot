"""Pre-flight readiness check for a live paper-trading run.

Verifies everything needed before launching the autonomous bot for a multi-day Binance Spot
**testnet/paper** run across multiple timeframes:

  * all key parameters present and sane
  * real connection to Binance (public market data) + bidirectional testnet account read
  * DB schema + live-data storage pipeline
  * data coverage / staleness per symbol x timeframe (the bot will backfill what's missing)
  * model pool + retraining wiring
  * safety: no real order can be sent by default

It NEVER sends an order. To additionally verify bidirectional order *write* on testnet, run
`python src/paper_demo_probe.py --symbol BTCUSDT --timeframe 1h` separately (it places one
controlled testnet order and refuses if any real flag is on).

Usage:
    python src/preflight.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframes 15m 1h 4h
Exit code is non-zero if any check FAILs.
"""
from __future__ import annotations

import argparse
import json
import sqlite3

import pandas as pd

import config
from db_utils import assert_required_schema, init_research_tables

OK, WARN, FAIL, INFO, SKIP = "OK", "WARN", "FAIL", "INFO", "SKIP"


class Report:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, name: str, status: str, detail: str) -> None:
        self.rows.append({"check": name, "status": status, "detail": detail})

    def has_fail(self) -> bool:
        return any(r["status"] == FAIL for r in self.rows)

    def render(self) -> str:
        width = max((len(r["check"]) for r in self.rows), default=10)
        lines = ["", "=" * 72, "PRE-FLIGHT READINESS", "=" * 72]
        for r in self.rows:
            lines.append(f"[{r['status']:4}] {r['check']:<{width}}  {r['detail']}")
        counts = {s: sum(1 for r in self.rows if r["status"] == s) for s in (OK, WARN, FAIL, INFO, SKIP)}
        lines.append("-" * 72)
        lines.append(f"OK={counts[OK]} WARN={counts[WARN]} FAIL={counts[FAIL]} INFO={counts[INFO]} SKIP={counts[SKIP]}")
        lines.append("READY" if not self.has_fail() else "NOT READY — resolve FAILs above")
        return "\n".join(lines)


def _check_safety(rep: Report) -> None:
    flags = {
        "DRY_RUN": config.DRY_RUN,
        "ENABLE_LIVE_TRADING": config.ENABLE_LIVE_TRADING,
        "ENABLE_REAL_ORDER_EXECUTION": config.ENABLE_REAL_ORDER_EXECUTION,
        "ENABLE_REAL_BINANCE_ACCOUNT": config.ENABLE_REAL_BINANCE_ACCOUNT,
        "KILL_SWITCH_ENABLED": config.KILL_SWITCH_ENABLED,
    }
    real_blocked = config.DRY_RUN and not (
        config.ENABLE_LIVE_TRADING and config.ENABLE_REAL_ORDER_EXECUTION and config.ENABLE_REAL_BINANCE_ACCOUNT
    )
    rep.add("safety_flags", OK if real_blocked else FAIL,
            ("real orders blocked by default; " if real_blocked else "REAL ORDERS POSSIBLE — unexpected for a paper run; ") + json.dumps(flags))


def _check_db(rep: Report) -> None:
    try:
        init_research_tables()
        assert_required_schema()
        rep.add("db_schema", OK, f"schema present at {config.DB_FILE}")
    except Exception as exc:
        rep.add("db_schema", FAIL, f"schema check failed: {exc}")


def _check_params(rep: Report, symbols: list[str], timeframes: list[str]) -> None:
    issues = []
    for name in ["MAX_TOTAL_EXPOSURE_USDT", "MAX_DAILY_LOSS_USDT", "MAX_TRADE_LOSS_USDT", "MIN_ORDER_NOTIONAL_USDT", "MIN_CASH_RESERVE_USDT"]:
        val = getattr(config, name, None)
        if val is None or float(val) <= 0:
            issues.append(f"{name}={val}")
    rep.add("risk_params", OK if not issues else FAIL,
            "risk limits set" if not issues else f"invalid/zero: {issues}")
    rep.add("universe", OK if symbols and timeframes else FAIL,
            f"symbols={symbols} timeframes={timeframes}")
    rep.add("feature_label_version", INFO, f"feature_version={config.FEATURE_VERSION} label_version={config.LABEL_VERSION}")
    rep.add("retraining_config", OK if (config.ENABLE_MODEL_POOL_MAINTENANCE and config.MODEL_POOL_TRAINING_ENABLED_IN_BOT) else WARN,
            f"pool_maintenance={config.ENABLE_MODEL_POOL_MAINTENANCE} train_in_bot={config.MODEL_POOL_TRAINING_ENABLED_IN_BOT} target={config.TARGET_ACCEPTED_MODELS}")


def _check_binance_public(rep: Report) -> None:
    try:
        from broker_client import BinanceSpotClient
        client = BinanceSpotClient.public_market_data_client()
        hc = client.healthcheck()
        kl = client.klines("BTCUSDT", "1h", limit=2)
        rep.add("binance_public", OK if hc.get("ok") and kl else FAIL,
                f"reachable base={hc.get('base_url')} server_time={hc.get('server_time')} klines={len(kl)}")
    except Exception as exc:
        rep.add("binance_public", FAIL, f"public market data unreachable: {exc}")


def _check_binance_testnet(rep: Report) -> None:
    has_creds = bool(getattr(config, "BINANCE_TESTNET_API_KEY", "") and getattr(config, "BINANCE_TESTNET_API_SECRET", ""))
    if not (config.ENABLE_TESTNET_PAPER_TRADING and has_creds):
        rep.add("binance_testnet_read", WARN,
                "no testnet creds / disabled -> will run local_paper (simulated fills, no Binance account sync)")
        return
    try:
        from broker_client import BinanceSpotClient
        client = BinanceSpotClient.testnet_execution_client()
        acct = client.account_info()
        balances = [b for b in acct.get("balances", []) if float(b.get("free", 0)) or float(b.get("locked", 0))]
        rep.add("binance_testnet_read", OK,
                f"bidirectional account read OK; canTrade={acct.get('canTrade')} nonzero_assets={len(balances)}")
    except Exception as exc:
        rep.add("binance_testnet_read", FAIL, f"testnet account read failed: {exc}")


def _check_exchange_filters(rep: Report, symbols: list[str]) -> None:
    try:
        from broker_client import BinanceSpotClient
        client = BinanceSpotClient.public_market_data_client()
        found = []
        for s in symbols:
            try:
                info = client.exchange_info(symbol=s)  # per-symbol (demo API rejects batched symbols=)
                if info.get("symbols"):
                    found.append(s)
            except Exception:
                pass
        missing = [s for s in symbols if s not in found]
        rep.add("exchange_filters", OK if not missing else WARN,
                f"filters available for {found}" + (f"; missing {missing}" if missing else ""))
    except Exception as exc:
        rep.add("exchange_filters", WARN, f"could not fetch exchange filters: {exc}")


def _latest_dt(table: str, symbol: str, timeframe: str) -> pd.Timestamp | None:
    with sqlite3.connect(config.DB_FILE) as conn:
        try:
            row = conn.execute(
                f"SELECT MAX(datetime_utc) FROM {table} WHERE symbol=? AND timeframe=?",
                (symbol, timeframe),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    if not row or not row[0]:
        return None
    return pd.to_datetime(row[0], utc=True, errors="coerce")


def _tf_seconds(tf: str) -> int:
    unit, val = tf[-1], int(tf[:-1])
    return val * {"m": 60, "h": 3600, "d": 86400}.get(unit, 3600)


def _check_data_and_features(rep: Report, symbols: list[str], timeframes: list[str]) -> None:
    now = pd.Timestamp.now(tz="UTC")
    missing, stale, fresh = [], [], []
    for s in symbols:
        for tf in timeframes:
            dt = _latest_dt(config.PRICES_TABLE, s, tf)
            if dt is None:
                missing.append(f"{s}:{tf}")
            elif (now - dt).total_seconds() > 5 * _tf_seconds(tf):
                stale.append(f"{s}:{tf}({dt.date()})")
            else:
                fresh.append(f"{s}:{tf}")
    if not missing and not stale:
        rep.add("data_coverage", OK, f"fresh: {fresh}")
    else:
        rep.add("data_coverage", WARN,
                f"the bot will backfill on start. missing={missing} stale={stale} fresh={fresh}")

    # Feature store presence at current version.
    with sqlite3.connect(config.DB_FILE) as conn:
        try:
            ver = conn.execute(f"SELECT DISTINCT feature_version FROM {config.FEATURES_TABLE} LIMIT 5").fetchall()
            n = conn.execute(f"SELECT COUNT(*) FROM {config.FEATURES_TABLE}").fetchone()[0]
        except sqlite3.OperationalError:
            ver, n = [], 0
    versions = [v[0] for v in ver]
    if n and config.FEATURE_VERSION in versions:
        rep.add("feature_store", OK, f"{n} rows at {config.FEATURE_VERSION}")
    else:
        rep.add("feature_store", WARN,
                f"feature store empty or pre-{config.FEATURE_VERSION} (versions={versions}); run feature_store --full-rebuild or let the bot refresh")


def _check_models(rep: Report, timeframes: list[str]) -> None:
    try:
        with sqlite3.connect(config.DB_FILE) as conn:
            rows = conn.execute(
                f"SELECT status, COUNT(*) FROM {config.MODEL_REGISTRY_TABLE} GROUP BY status"
            ).fetchall()
        by_status = {r[0]: int(r[1]) for r in rows}
        active = by_status.get("paper_active", 0) + by_status.get("backtest_accepted", 0)
        rep.add("models", OK if active else INFO,
                f"registry status counts={by_status}" + ("" if active else " - none yet; model_maintenance will train candidates"))
    except Exception as exc:
        rep.add("models", WARN, f"could not read registry: {exc}")


def _check_pipeline_imports(rep: Report) -> None:
    try:
        import realtime_ingestor  # noqa: F401  (live data ingestion + storage)
        import model_maintenance   # noqa: F401  (retraining)
        import trading_bot         # noqa: F401  (paper loop)
        import autonomous_runner   # noqa: F401  (orchestration)
        rep.add("pipeline_imports", OK, "ingestor/maintenance/trading_bot/autonomous_runner import cleanly")
    except Exception as exc:
        rep.add("pipeline_imports", FAIL, f"import error: {exc}")


def run_preflight(symbols: list[str], timeframes: list[str]) -> Report:
    rep = Report()
    _check_safety(rep)
    _check_db(rep)
    _check_params(rep, symbols, timeframes)
    _check_pipeline_imports(rep)
    _check_binance_public(rep)
    _check_binance_testnet(rep)
    _check_exchange_filters(rep, symbols)
    _check_data_and_features(rep, symbols, timeframes)
    _check_models(rep, timeframes)
    return rep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-flight readiness check for live paper trading.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframes", nargs="*", default=None)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON as well.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.upper().strip() for s in args.symbols] if args.symbols else [s.upper() for s in config.SYMBOLS]
    timeframes = [tf.strip() for tf in (args.timeframes or [])] or list(getattr(config, "TIMEFRAMES", [config.TIMEFRAME]))
    rep = run_preflight(symbols, timeframes)
    print(rep.render())
    if args.json:
        print(json.dumps(rep.rows, ensure_ascii=True, indent=2))
    raise SystemExit(1 if rep.has_fail() else 0)


if __name__ == "__main__":
    main()
