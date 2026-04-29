"""Read-only data access layer for the operational Streamlit dashboard.

All functions are defensive: missing DBs, tables, columns, reports or logs
return empty DataFrames / N/A dictionaries instead of crashing the UI.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from config import (  # type: ignore
        BASE_DIR,
        DB_FILE,
        REPORTS_DIR,
        LOGS_DIR,
        SYMBOLS,
        TIMEFRAME,
        DRY_RUN,
        ENABLE_LIVE_TRADING,
        ENABLE_REAL_ORDER_EXECUTION,
        ENABLE_REAL_BINANCE_ACCOUNT,
        ENABLE_TESTNET_PAPER_TRADING,
        ENABLE_LOCAL_SIMULATED_PAPER,
        BINANCE_ENV,
        DASHBOARD_REFRESH_SECONDS,
        MAX_DAILY_LOSS_USDT,
        MAX_EXPOSURE_PER_MODEL_USDT,
        MAX_EXPOSURE_TOTAL_USDT,
        MAX_ORDER_NOTIONAL_USDT,
        MIN_ORDER_NOTIONAL_USDT,
        MAX_TRADES_PER_DAY_PER_MODEL,
        KILL_SWITCH_ENABLED,
    )
except Exception:  # pragma: no cover
    BASE_DIR = Path(__file__).resolve().parent.parent
    DB_FILE = BASE_DIR / "data" / "db" / "market_data.sqlite"
    REPORTS_DIR = BASE_DIR / "reports"
    LOGS_DIR = BASE_DIR / "logs"
    SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",") if s.strip()]
    TIMEFRAME = os.getenv("TIMEFRAME", "1h")
    DRY_RUN = os.getenv("DRY_RUN", "true").lower() != "false"
    ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
    ENABLE_REAL_ORDER_EXECUTION = os.getenv("ENABLE_REAL_ORDER_EXECUTION", "false").lower() == "true"
    ENABLE_REAL_BINANCE_ACCOUNT = os.getenv("ENABLE_REAL_BINANCE_ACCOUNT", "false").lower() == "true"
    ENABLE_TESTNET_PAPER_TRADING = os.getenv("ENABLE_TESTNET_PAPER_TRADING", "true").lower() == "true"
    ENABLE_LOCAL_SIMULATED_PAPER = os.getenv("ENABLE_LOCAL_SIMULATED_PAPER", "true").lower() == "true"
    BINANCE_ENV = os.getenv("BINANCE_ENV", "unknown")
    DASHBOARD_REFRESH_SECONDS = int(os.getenv("DASHBOARD_REFRESH_SECONDS", "30"))
    MAX_DAILY_LOSS_USDT = float(os.getenv("MAX_DAILY_LOSS_USDT", "50"))
    MAX_EXPOSURE_PER_MODEL_USDT = float(os.getenv("MAX_EXPOSURE_PER_MODEL_USDT", "100"))
    MAX_EXPOSURE_TOTAL_USDT = float(os.getenv("MAX_EXPOSURE_TOTAL_USDT", "500"))
    MAX_ORDER_NOTIONAL_USDT = float(os.getenv("MAX_ORDER_NOTIONAL_USDT", "50"))
    MIN_ORDER_NOTIONAL_USDT = float(os.getenv("MIN_ORDER_NOTIONAL_USDT", "10"))
    MAX_TRADES_PER_DAY_PER_MODEL = int(os.getenv("MAX_TRADES_PER_DAY_PER_MODEL", "10"))
    KILL_SWITCH_ENABLED = os.getenv("KILL_SWITCH_ENABLED", "true").lower() == "true"


ALLOWED_TABLES = {
    "prices",
    "data_coverage",
    "data_gaps",
    "features",
    "model_registry",
    "signals",
    "orders",
    "fills",
    "positions",
    "portfolio_snapshots",
    "paper_model_metrics",
    "model_lifecycle_events",
    "bot_events",
    "bot_status",
    "risk_events",
    "ingestion_log",
    "validation_predictions",
}

DB_PATH = Path(DB_FILE).expanduser()
if not DB_PATH.is_absolute():
    DB_PATH = (Path(BASE_DIR) / DB_PATH).resolve()
REPORTS_PATH = Path(REPORTS_DIR)
LOGS_PATH = Path(LOGS_DIR)


def utc_now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def get_db_path() -> Path:
    return DB_PATH


@contextmanager
def get_db_connection() -> Iterable[sqlite3.Connection]:
    """Open SQLite in read-only mode so the dashboard cannot mutate state."""
    if not DB_PATH.exists():
        raise FileNotFoundError(str(DB_PATH))
    conn = sqlite3.connect(f"{DB_PATH.resolve().as_uri()}?mode=ro", uri=True, timeout=10)
    try:
        yield conn
    finally:
        conn.close()


def _empty(message: str | None = None, missing_table: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame()
    if message:
        df.attrs["message"] = message
    if missing_table:
        df.attrs["missing_table"] = missing_table
    return df


def _safe_table(table: str) -> str:
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Unsupported dashboard table: {table}")
    return table


def _quote_identifier(name: str) -> str:
    """Quote an identifier returned by sqlite_master for metadata-only queries."""
    return '"' + str(name).replace('"', '""') + '"'


def db_exists() -> bool:
    return DB_PATH.exists()


def list_tables() -> list[str]:
    if not db_exists():
        return []
    try:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        return [str(r[0]) for r in rows]
    except Exception:
        return []


def table_exists(table: str) -> bool:
    return _safe_table(table) in set(list_tables())


def table_columns(table: str) -> list[str]:
    table = _safe_table(table)
    if not table_exists(table):
        return []
    try:
        with get_db_connection() as conn:
            return [str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    except Exception:
        return []


def table_row_count(table: str) -> int | None:
    table = _safe_table(table)
    if not table_exists(table):
        return None
    try:
        with get_db_connection() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except Exception:
        return None


def _table_row_count_existing(table: str) -> int | None:
    """Count rows for a table discovered from sqlite_master, including legacy tables."""
    if table not in set(list_tables()):
        return None
    try:
        with get_db_connection() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(table)}").fetchone()[0])
    except Exception:
        return None


def load_table_counts() -> pd.DataFrame:
    if not db_exists():
        return _empty(f"SQLite DB not found at {DB_PATH}")
    rows = [{"table": t, "rows": _table_row_count_existing(t)} for t in list_tables()]
    return pd.DataFrame(rows).sort_values("table") if rows else pd.DataFrame(columns=["table", "rows"])


def read_table(table: str, limit: int = 1000, order_by: str | None = None, descending: bool = True) -> pd.DataFrame:
    table = _safe_table(table)
    if not db_exists():
        return _empty(f"SQLite DB not found at {DB_PATH}")
    if not table_exists(table):
        return _empty(f"Table {table} not found. Run ingestion/training/backtest first.", table)
    cols = table_columns(table)
    order_sql = ""
    if order_by and order_by in cols:
        order_sql = f" ORDER BY {order_by} {'DESC' if descending else 'ASC'}"
    try:
        with get_db_connection() as conn:
            return pd.read_sql_query(f"SELECT * FROM {table}{order_sql} LIMIT ?", conn, params=(int(limit),))
    except Exception as exc:
        return _empty(f"Could not read {table}: {exc}")


def _read_sql(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    try:
        with get_db_connection() as conn:
            return pd.read_sql_query(sql, conn, params=params)
    except Exception as exc:
        return _empty(str(exc))


def _num(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _json(value: Any, default: Any = None) -> Any:
    if default is None:
        default = {}
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _latest_per_group(df: pd.DataFrame, group_cols: list[str], ts_col: str) -> pd.DataFrame:
    if df.empty or ts_col not in df.columns:
        return df
    out = df.copy()
    out[ts_col] = _dt(out[ts_col])
    return out.sort_values(ts_col).groupby(group_cols, dropna=False).tail(1).reset_index(drop=True)


def _latest_report(pattern: str) -> Path | None:
    files = sorted(REPORTS_PATH.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _read_latest_csv(pattern: str) -> pd.DataFrame:
    path = _latest_report(pattern)
    if not path:
        return _empty(f"Report not found for {pattern}. Run backtest/validation first.")
    try:
        df = pd.read_csv(path)
        df.attrs["source_file"] = str(path)
        return df
    except Exception as exc:
        return _empty(f"Could not read {path.name}: {exc}")


def _read_latest_json(pattern: str) -> dict[str, Any]:
    path = _latest_report(pattern)
    if not path:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_source_file"] = str(path)
        return data
    except Exception:
        return {"_source_file": str(path), "_error": "Could not parse JSON report"}


def real_order_possible() -> bool:
    return bool(ENABLE_LIVE_TRADING and ENABLE_REAL_ORDER_EXECUTION and ENABLE_REAL_BINANCE_ACCOUNT and not DRY_RUN)


def get_mode() -> str:
    if real_order_possible():
        return "LIVE TRADING"
    if ENABLE_TESTNET_PAPER_TRADING or ENABLE_LOCAL_SIMULATED_PAPER:
        return "PAPER TRADING"
    if DRY_RUN:
        return "DRY RUN"
    return "RESEARCH"


def load_data_coverage() -> pd.DataFrame:
    return _num(read_table("data_coverage", 5000, "updated_at_utc"), ["row_count"])


def load_data_gaps(open_only: bool = False) -> pd.DataFrame:
    df = _num(read_table("data_gaps", 5000, "detected_at_utc"), ["missing_bars", "resolved"])
    if open_only and not df.empty and "resolved" in df.columns:
        df = df[df["resolved"].fillna(0).astype(int) == 0]
    return df


def load_model_registry(limit: int = 1000) -> pd.DataFrame:
    df = read_table("model_registry", limit, "training_ts_utc")
    if df.empty:
        return df
    out = df.copy()
    for col in ["metrics_json", "params_json", "rejection_reasons_json", "symbols_json"]:
        if col in out.columns:
            out[f"{col}_parsed"] = out[col].apply(lambda v: _json(v, [] if "reasons" in col or "symbols" in col else {}))

    def metric(row: pd.Series, path: list[str]) -> Any:
        data = row.get("metrics_json_parsed", {})
        for key in path:
            if not isinstance(data, dict):
                return None
            data = data.get(key)
        return data

    paths = {
        "accuracy": ["classification", "accuracy"],
        "f1_macro": ["classification", "f1_macro"],
        "strategy_return": ["economic", "strategy_return"],
        "buy_hold_return": ["economic", "buy_hold_return"],
        "sharpe": ["economic", "sharpe"],
        "max_drawdown": ["economic", "max_drawdown"],
        "profit_factor": ["economic", "profit_factor"],
        "trade_count": ["economic", "trade_count"],
    }
    for name, path in paths.items():
        out[name] = out.apply(lambda r, p=path: metric(r, p), axis=1)
    out = _num(out, list(paths) + ["is_active"])
    if "training_ts_utc" in out.columns:
        out["training_ts_utc"] = _dt(out["training_ts_utc"])
        out = out.sort_values("training_ts_utc", ascending=False)
    return out


def load_system_status() -> dict[str, Any]:
    bot_status = read_table("bot_status", 100, "component", descending=False)
    now = pd.Timestamp.now(tz="UTC")
    state = "Unknown"
    if not bot_status.empty and "last_heartbeat_utc" in bot_status.columns:
        tmp = bot_status.copy()
        tmp["last_heartbeat_utc"] = _dt(tmp["last_heartbeat_utc"])
        tmp["age"] = (now - tmp["last_heartbeat_utc"]).dt.total_seconds()
        running = tmp["status"].astype(str).str.lower().eq("running") & tmp["age"].lt(180)
        error = tmp["status"].astype(str).str.lower().str.contains("error|failed", regex=True, na=False)
        state = "Error" if bool(error.any()) else ("Running" if bool(running.any()) else "Stopped")
    elif db_exists():
        state = "Unknown"

    coverage = load_data_coverage()
    latest_data_ts = None
    symbols = sorted(set(SYMBOLS))
    if not coverage.empty:
        if "max_datetime_utc" in coverage.columns and coverage["max_datetime_utc"].dropna().any():
            latest_data_ts = str(coverage["max_datetime_utc"].dropna().max())
        if "symbol" in coverage.columns:
            symbols = sorted(set(symbols) | set(coverage["symbol"].dropna().astype(str).tolist()))

    registry = load_model_registry()
    active_model, active_status = "N/A", "unknown"
    if not registry.empty:
        status = registry["status"].astype(str) if "status" in registry.columns else pd.Series(["unknown"] * len(registry), index=registry.index)
        preferred = registry[status.isin(["real_active", "paper_active", "real_ready", "paper_validated", "backtest_accepted"])]
        row = (preferred if not preferred.empty else registry).iloc[0]
        active_model = str(row.get("model_id", "N/A"))
        active_status = str(row.get("status", row.get("acceptance_status", "unknown")))

    return {
        "system_name": "AI Trading Bot",
        "mode": get_mode(),
        "state": state,
        "last_refresh_utc": utc_now_iso(),
        "exchange": "Binance Spot",
        "timeframe": TIMEFRAME,
        "symbols": symbols,
        "active_model_id": active_model,
        "active_model_status": active_status,
        "latest_data_ts": latest_data_ts,
        "db_path": str(DB_PATH),
        "db_exists": db_exists(),
        "binance_env": BINANCE_ENV,
        "refresh_seconds": DASHBOARD_REFRESH_SECONDS,
        "real_order_possible": real_order_possible(),
        "safety_flags": {
            "DRY_RUN": DRY_RUN,
            "ENABLE_TESTNET_PAPER_TRADING": ENABLE_TESTNET_PAPER_TRADING,
            "ENABLE_LOCAL_SIMULATED_PAPER": ENABLE_LOCAL_SIMULATED_PAPER,
            "ENABLE_LIVE_TRADING": ENABLE_LIVE_TRADING,
            "ENABLE_REAL_ORDER_EXECUTION": ENABLE_REAL_ORDER_EXECUTION,
            "ENABLE_REAL_BINANCE_ACCOUNT": ENABLE_REAL_BINANCE_ACCOUNT,
            "KILL_SWITCH_ENABLED": KILL_SWITCH_ENABLED,
        },
    }


def load_recent_signals(limit: int = 200) -> pd.DataFrame:
    df = _num(read_table("signals", limit, "created_at_utc"), ["signal_position", "prob_short", "prob_flat", "prob_long"])
    if df.empty:
        return df
    if {"prob_short", "prob_flat", "prob_long"}.issubset(df.columns):
        df["confidence"] = df[["prob_short", "prob_flat", "prob_long"]].max(axis=1)
    if "signal_label" in df.columns:
        df["signal"] = df["signal_label"]
    if "created_at_utc" in df.columns:
        df["persisted_at"] = df["created_at_utc"]
    return df


def load_recent_orders(limit: int = 200) -> pd.DataFrame:
    df = _num(read_table("orders", limit, "created_at_utc"), ["quantity", "requested_price", "fill_price", "notional", "dry_run", "price_requested", "price_filled"])
    if df.empty:
        return df
    if "order_type" in df.columns and "type" not in df.columns:
        df["type"] = df["order_type"]
    if "requested_price" not in df.columns and "price_requested" in df.columns:
        df["requested_price"] = df["price_requested"]
    if "fill_price" not in df.columns and "price_filled" in df.columns:
        df["fill_price"] = df["price_filled"]
    return df


def load_recent_fills(limit: int = 200) -> pd.DataFrame:
    return _num(read_table("fills", limit, "timestamp_utc"), ["quantity", "price", "commission"])


def load_price_series(symbol: str, timeframe: str | None = None, limit: int = 500) -> pd.DataFrame:
    timeframe = timeframe or TIMEFRAME
    if not table_exists("prices"):
        return _empty("Table prices not found. Run ingestion first.", "prices")
    sql = """
        SELECT symbol, timeframe, datetime_utc, open, high, low, close, volume
        FROM prices
        WHERE symbol = ? AND timeframe = ?
        ORDER BY datetime_utc DESC
        LIMIT ?
    """
    df = _num(_read_sql(sql, (symbol, timeframe, int(limit))), ["open", "high", "low", "close", "volume"])
    if not df.empty and "datetime_utc" in df.columns:
        df["datetime_utc"] = _dt(df["datetime_utc"])
        df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df


def load_latest_prices(symbols: Iterable[str] | None = None, timeframe: str | None = None) -> pd.DataFrame:
    frames = [load_price_series(s, timeframe or TIMEFRAME, 1) for s in list(symbols or SYMBOLS)]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_portfolio_snapshots(limit: int = 10000) -> pd.DataFrame:
    df = _num(read_table("portfolio_snapshots", limit, "datetime_utc"), ["cash", "equity", "realized_pnl", "unrealized_pnl", "dry_run"])
    if not df.empty and "datetime_utc" in df.columns:
        df["datetime_utc"] = _dt(df["datetime_utc"])
        df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df


def load_paper_model_metrics(limit: int = 1000) -> pd.DataFrame:
    cols = ["trades", "filled_trades", "win_rate", "realized_pnl", "unrealized_pnl", "total_pnl", "equity", "total_return", "max_drawdown", "profit_factor", "average_trade_return", "days_active", "current_exposure"]
    return _num(read_table("paper_model_metrics", limit, "evaluated_at_utc"), cols)


def load_portfolio_summary() -> dict[str, Any]:
    summary = {
        "total_equity": None, "cash_usdt": None, "unrealized_pnl": None, "realized_pnl": None,
        "daily_pnl": None, "total_return": None, "max_drawdown": None, "number_of_trades": None,
        "win_rate": None, "profit_factor": None, "sharpe": None, "sortino": None, "exposure_pct": None,
        "source": "N/A",
    }
    snaps = load_portfolio_snapshots()
    if not snaps.empty:
        latest = _latest_per_group(snaps, ["model_id", "account_mode"], "datetime_utc")
        for key, col in [("total_equity", "equity"), ("cash_usdt", "cash"), ("realized_pnl", "realized_pnl"), ("unrealized_pnl", "unrealized_pnl")]:
            if col in latest.columns:
                summary[key] = latest[col].sum()
        grouped = snaps.groupby("datetime_utc")["equity"].sum().sort_index() if {"datetime_utc", "equity"}.issubset(snaps.columns) else pd.Series(dtype=float)
        if len(grouped) > 1:
            summary["total_return"] = grouped.iloc[-1] / grouped.iloc[0] - 1 if grouped.iloc[0] else None
            summary["max_drawdown"] = (grouped / grouped.cummax() - 1).min()
            recent = grouped[grouped.index >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)]
            if len(recent):
                summary["daily_pnl"] = grouped.iloc[-1] - recent.iloc[0]
        summary["source"] = "portfolio_snapshots"

    metrics = load_paper_model_metrics()
    if not metrics.empty:
        latest = _latest_per_group(metrics, ["model_id", "account_mode"], "evaluated_at_utc") if "evaluated_at_utc" in metrics.columns else metrics
        if "filled_trades" in latest.columns:
            summary["number_of_trades"] = int(latest["filled_trades"].fillna(0).sum())
        for key, col, how in [("win_rate", "win_rate", "mean"), ("profit_factor", "profit_factor", "mean"), ("max_drawdown", "max_drawdown", "mean"), ("total_return", "total_return", "mean")]:
            if summary[key] is None and col in latest.columns and latest[col].dropna().size:
                summary[key] = getattr(latest[col].dropna(), how)()
        if summary["source"] == "N/A":
            summary["source"] = "paper_model_metrics"

    orders = load_recent_orders(10000)
    if not orders.empty and "status" in orders.columns:
        summary["number_of_trades"] = int(orders["status"].astype(str).str.upper().isin(["FILLED", "PARTIALLY_FILLED"]).sum())

    positions = read_table("positions", 10000, "updated_at_utc")
    if not positions.empty and summary.get("total_equity"):
        pos = _num(positions, ["quantity", "avg_price"])
        prices = load_latest_prices(pos["symbol"].dropna().unique().tolist() if "symbol" in pos.columns else [])
        if not prices.empty:
            pv = pos.merge(prices[["symbol", "close"]], on="symbol", how="left")
            exposure = (pv["quantity"].abs() * pv["close"]).sum()
            summary["exposure_pct"] = exposure / float(summary["total_equity"]) if float(summary["total_equity"]) else None

    econ = _read_latest_json("backtest_oos_summary*.json").get("economic", {})
    if isinstance(econ, dict):
        summary["sharpe"] = econ.get("sharpe")
        summary["profit_factor"] = summary["profit_factor"] if summary["profit_factor"] is not None else econ.get("profit_factor")
    return summary


def load_open_positions() -> pd.DataFrame:
    df = _num(read_table("positions", 1000, "updated_at_utc"), ["quantity", "avg_price", "realized_pnl", "unrealized_pnl", "dry_run"])
    if df.empty:
        return df
    if "avg_price" in df.columns:
        df["avg_entry_price"] = df["avg_price"]
    active = df[df["quantity"].fillna(0).abs() > 1e-12].copy() if "quantity" in df.columns else df.copy()
    if active.empty:
        active = df.copy()
    prices = load_latest_prices(active["symbol"].dropna().unique().tolist() if "symbol" in active.columns else [])
    if not prices.empty:
        active = active.merge(prices[["symbol", "close", "datetime_utc"]].rename(columns={"close": "current_price", "datetime_utc": "price_timestamp_utc"}), on="symbol", how="left")
    else:
        active["current_price"] = pd.NA
    active["market_value"] = pd.to_numeric(active.get("quantity"), errors="coerce") * pd.to_numeric(active.get("current_price"), errors="coerce")
    if "unrealized_pnl" not in active.columns or active["unrealized_pnl"].isna().all():
        active["unrealized_pnl"] = (pd.to_numeric(active.get("current_price"), errors="coerce") - pd.to_numeric(active.get("avg_entry_price"), errors="coerce")) * pd.to_numeric(active.get("quantity"), errors="coerce")
    equity = load_portfolio_summary().get("total_equity")
    active["exposure_pct"] = active["market_value"].abs() / float(equity) if equity else pd.NA
    return active


def load_equity_curve() -> pd.DataFrame:
    snaps = load_portfolio_snapshots()
    if not snaps.empty and {"datetime_utc", "equity"}.issubset(snaps.columns):
        agg_map = {"equity": ("equity", "sum")}
        if "cash" in snaps.columns:
            agg_map["cash"] = ("cash", "sum")
        if "realized_pnl" in snaps.columns:
            agg_map["realized_pnl"] = ("realized_pnl", "sum")
        if "unrealized_pnl" in snaps.columns:
            agg_map["unrealized_pnl"] = ("unrealized_pnl", "sum")
        grouped = snaps.groupby("datetime_utc", dropna=False).agg(**agg_map).reset_index().sort_values("datetime_utc")
        grouped["drawdown"] = grouped["equity"] / grouped["equity"].cummax() - 1
        grouped["source"] = "portfolio_snapshots"
        return grouped
    report = _read_latest_csv("backtest_oos_equity*.csv")
    if report.empty:
        report = _read_latest_csv("validation_equity*.csv")
    if report.empty:
        return report
    if "datetime_utc" in report.columns:
        report["datetime_utc"] = _dt(report["datetime_utc"])
    if "strategy_equity" in report.columns:
        report["equity"] = pd.to_numeric(report["strategy_equity"], errors="coerce")
    if "market_equity" in report.columns:
        report["benchmark_equity"] = pd.to_numeric(report["market_equity"], errors="coerce")
    if "drawdown" in report.columns:
        report["drawdown"] = pd.to_numeric(report["drawdown"], errors="coerce")
    report["source"] = Path(str(report.attrs.get("source_file", "report"))).name
    return report


def load_trade_pnl(limit: int = 500) -> pd.DataFrame:
    orders = load_recent_orders(limit)
    if not orders.empty and "realized_pnl" in orders.columns:
        out = _num(orders, ["realized_pnl"]).dropna(subset=["realized_pnl"]).copy()
        if not out.empty:
            out["pnl"] = out["realized_pnl"]
            out["source"] = "orders.realized_pnl"
            return out
    signals = _read_latest_csv("backtest_oos_signals*.csv")
    if signals.empty:
        signals = _read_latest_csv("validation_predictions*.csv")
    if signals.empty:
        return signals
    out = signals.copy()
    if "datetime_utc" in out.columns:
        out["datetime_utc"] = _dt(out["datetime_utc"])
    if "strategy_return" in out.columns:
        out["pnl"] = pd.to_numeric(out["strategy_return"], errors="coerce")
    elif {"fwd_return_1", "signal_position"}.issubset(out.columns):
        out["pnl"] = pd.to_numeric(out["fwd_return_1"], errors="coerce") * pd.to_numeric(out["signal_position"], errors="coerce")
    out = out.dropna(subset=["pnl"]).sort_values("datetime_utc").tail(limit)
    out["cumulative_pnl"] = out["pnl"].cumsum()
    out["source"] = Path(str(signals.attrs.get("source_file", "backtest report"))).name
    return out


def load_exposure_breakdown() -> pd.DataFrame:
    pos = load_open_positions()
    if not pos.empty and "market_value" in pos.columns:
        by_asset = pos.groupby("symbol", dropna=False)["market_value"].sum().reset_index().rename(columns={"symbol": "asset", "market_value": "value_usdt"})
        by_asset["asset"] = by_asset["asset"].astype(str).str.replace("USDT", "", regex=False)
        cash = load_portfolio_summary().get("cash_usdt")
        if cash is not None and pd.notna(cash):
            by_asset = pd.concat([by_asset, pd.DataFrame([{"asset": "USDT", "value_usdt": float(cash)}])], ignore_index=True)
        return by_asset.dropna(subset=["value_usdt"])
    summary = load_portfolio_summary()
    cash = summary.get("cash_usdt")
    if cash is not None and pd.notna(cash):
        return pd.DataFrame([{"asset": "USDT", "value_usdt": float(cash)}])
    return pd.DataFrame(columns=["asset", "value_usdt"])


def load_model_comparison() -> pd.DataFrame:
    registry = load_model_registry()
    rows: list[dict[str, Any]] = []
    if not registry.empty:
        keep = ["model_id", "status", "acceptance_status", "symbol_scope", "timeframe", "train_start", "train_end", "test_start", "test_end", "training_ts_utc", "accuracy", "f1_macro", "strategy_return", "buy_hold_return", "sharpe", "max_drawdown", "profit_factor", "trade_count", "rejection_reasons_json_parsed"]
        rows += registry[[c for c in keep if c in registry.columns]].to_dict("records")
    for path in sorted(REPORTS_PATH.glob("backtest_oos_summary*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            econ = data.get("economic", {}) or {}
            cls = data.get("classification", {}) or {}
            rows.append({
                "model_id": data.get("model_id"), "status": "backtest_report",
                "acceptance_status": (data.get("acceptance") or {}).get("acceptance_status"),
                "symbol_scope": ",".join(data.get("symbols", [])), "timeframe": data.get("timeframe"),
                "test_start": data.get("start_datetime_utc"), "test_end": data.get("end_datetime_utc"),
                "accuracy": cls.get("accuracy"), "strategy_return": econ.get("strategy_return"),
                "buy_hold_return": econ.get("buy_hold_return"), "sharpe": econ.get("sharpe"),
                "max_drawdown": econ.get("max_drawdown"), "profit_factor": econ.get("profit_factor"),
                "trade_count": econ.get("trade_count"), "source": path.name,
            })
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_recent_logs(limit: int = 200) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table in ["bot_events", "risk_events"]:
        df = read_table(table, limit, "created_at_utc")
        if df.empty:
            continue
        for _, row in df.iterrows():
            severity = str(row.get("severity", "WARNING" if table == "risk_events" and not bool(row.get("approved", 1)) else "INFO")).upper()
            rows.append({
                "timestamp_utc": row.get("created_at_utc"), "source": table, "severity": severity,
                "component": row.get("component", row.get("model_id", "")),
                "message": row.get("message", row.get("reason", "")),
                "details": row.get("metadata_json", row.get("details_json", "")),
            })
    keywords = ("ERROR", "WARNING", "WARN", "rejected", "reject", "risk", "gap", "failed", "exception", "blocked")
    for path in sorted(LOGS_PATH.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]:
        try:
            text = path.read_bytes()[-120_000:].decode("utf-8", errors="replace")
        except Exception:
            continue
        for line in text.splitlines()[-1000:]:
            if any(k.lower() in line.lower() for k in keywords):
                rows.append({"timestamp_utc": "", "source": path.name, "severity": "ERROR" if "error" in line.lower() or "exception" in line.lower() else "WARNING", "component": "log_file", "message": line[:1200], "details": ""})
    if not rows:
        return pd.DataFrame(columns=["timestamp_utc", "source", "severity", "component", "message", "details"])
    df = pd.DataFrame(rows)
    df["_sort"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df.sort_values("_sort", ascending=False, na_position="last").drop(columns=["_sort"]).head(limit).reset_index(drop=True)


def load_latest_report_summary() -> dict[str, Any]:
    return {
        "backtest": _read_latest_json("backtest_oos_summary*.json"),
        "validation": _read_latest_json("validation_summary*.json"),
        "train": _read_latest_json("train_auto*.json") or _read_latest_json("train_lgbm*.json"),
    }


def load_data_inventory() -> dict[str, Any]:
    existing = set(list_tables())
    return {
        "db_path": str(DB_PATH), "db_exists": db_exists(), "existing_tables": sorted(existing),
        "missing_expected_tables": [t for t in sorted(ALLOWED_TABLES) if t not in existing],
        "reports_count": len(list(REPORTS_PATH.glob("*"))) if REPORTS_PATH.exists() else 0,
        "logs_count": len(list(LOGS_PATH.glob("*.log"))) if LOGS_PATH.exists() else 0,
    }


def get_risk_limits() -> dict[str, Any]:
    return {
        "MAX_EXPOSURE_PER_MODEL_USDT": MAX_EXPOSURE_PER_MODEL_USDT,
        "MAX_EXPOSURE_TOTAL_USDT": MAX_EXPOSURE_TOTAL_USDT,
        "MAX_DAILY_LOSS_USDT": MAX_DAILY_LOSS_USDT,
        "MAX_ORDER_NOTIONAL_USDT": MAX_ORDER_NOTIONAL_USDT,
        "MIN_ORDER_NOTIONAL_USDT": MIN_ORDER_NOTIONAL_USDT,
        "MAX_TRADES_PER_DAY_PER_MODEL": MAX_TRADES_PER_DAY_PER_MODEL,
        "KILL_SWITCH_ENABLED": KILL_SWITCH_ENABLED,
    }
