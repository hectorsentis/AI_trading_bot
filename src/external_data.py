"""External market-context data (Phase C): funding rate, open interest, fear/greed.

Opt-in and leakage-safe. Each external series is stored in its own timestamped table rows and
joined onto features with a **backward as-of merge**, so a row at time t only ever sees the last
external value known at or before t. Ingestion requires network access and is disabled by default
(`ENABLE_EXTERNAL_DATA=false`); the as-of join + storage layer are pure and testable offline.

CLI:
    python src/external_data.py --ingest fear_greed funding open_interest --symbols BTCUSDT ETHUSDT
"""
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from config import (
    DB_FILE,
    EXTERNAL_DATA_TABLE,
    ENABLE_EXTERNAL_DATA,
    FEAR_GREED_API_URL,
    BINANCE_FUTURES_BASE_URL,
)

_GLOBAL_SYMBOL = ""  # sentinel for metrics that are not symbol-specific (e.g. fear/greed)


def init_external_data_table() -> None:
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {EXTERNAL_DATA_TABLE} (
                provider TEXT NOT NULL,
                metric TEXT NOT NULL,
                symbol TEXT NOT NULL DEFAULT '',
                datetime_utc TEXT NOT NULL,
                value REAL,
                ingestion_ts_utc TEXT,
                PRIMARY KEY (provider, metric, symbol, datetime_utc)
            )
            """
        )
        conn.commit()


def save_external_metric(provider: str, metric: str, rows: list[dict], symbol: str | None = None) -> int:
    """Idempotently upsert external observations. ``rows`` are dicts with datetime_utc + value."""
    if not rows:
        return 0
    init_external_data_table()
    sym = (symbol or _GLOBAL_SYMBOL).upper() if symbol else _GLOBAL_SYMBOL
    now = pd.Timestamp.now(tz="UTC").isoformat()
    payload = []
    for r in rows:
        dt = r["datetime_utc"]
        dt = dt.isoformat() if hasattr(dt, "isoformat") else str(dt)
        value = r.get("value")
        payload.append((provider, metric, sym, dt, None if value is None else float(value), now))
    with sqlite3.connect(DB_FILE) as conn:
        conn.executemany(
            f"""
            INSERT OR REPLACE INTO {EXTERNAL_DATA_TABLE}
                (provider, metric, symbol, datetime_utc, value, ingestion_ts_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        conn.commit()
    return len(payload)


def load_external_metric(provider: str, metric: str, symbol: str | None = None) -> pd.DataFrame:
    """Return a sorted [datetime_utc, value] frame for one external metric."""
    init_external_data_table()
    sym = (symbol or _GLOBAL_SYMBOL).upper() if symbol else _GLOBAL_SYMBOL
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT datetime_utc, value
            FROM {EXTERNAL_DATA_TABLE}
            WHERE provider = ? AND metric = ? AND symbol = ?
            ORDER BY datetime_utc
            """,
            conn,
            params=(provider, metric, sym),
        )
    if df.empty:
        return df
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)


def attach_external_features_asof(
    df: pd.DataFrame,
    metric_frames: dict[str, pd.DataFrame],
    datetime_col: str = "datetime_utc",
) -> pd.DataFrame:
    """Attach external metrics as feature columns using a leakage-safe backward as-of merge.

    ``metric_frames`` maps the output column name to a [datetime_utc, value] frame. Each row of
    ``df`` receives only the most recent external value at or before its own timestamp. Rows with
    no prior external value get NaN (the model can treat that as missing).
    """
    out = df.sort_values(datetime_col).reset_index(drop=True)
    for column, frame in metric_frames.items():
        if frame is None or frame.empty:
            out[column] = float("nan")
            continue
        m = frame[[datetime_col, "value"]].dropna(subset=[datetime_col]).sort_values(datetime_col)
        merged = pd.merge_asof(
            out[[datetime_col]],
            m.rename(columns={"value": column}),
            on=datetime_col,
            direction="backward",
        )
        out[column] = merged[column].to_numpy()
    return out


# --------------------------------------------------------------------------------------------
# Ingestion (network; only invoked when ENABLE_EXTERNAL_DATA is true).
# --------------------------------------------------------------------------------------------
def ingest_fear_greed(limit: int = 365) -> int:
    import requests

    resp = requests.get(FEAR_GREED_API_URL, params={"limit": int(limit), "format": "json"}, timeout=20)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    rows = [
        {"datetime_utc": pd.to_datetime(int(d["timestamp"]), unit="s", utc=True), "value": float(d["value"])}
        for d in data
        if d.get("timestamp") and d.get("value") is not None
    ]
    return save_external_metric("alternative.me", "fear_greed", rows)


def ingest_funding_rate(symbol: str, limit: int = 1000) -> int:
    import requests

    resp = requests.get(
        f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate",
        params={"symbol": symbol.upper(), "limit": int(limit)},
        timeout=20,
    )
    resp.raise_for_status()
    rows = [
        {"datetime_utc": pd.to_datetime(int(d["fundingTime"]), unit="ms", utc=True), "value": float(d["fundingRate"])}
        for d in resp.json()
        if d.get("fundingTime") and d.get("fundingRate") is not None
    ]
    return save_external_metric("binance_futures", "funding_rate", rows, symbol=symbol)


def ingest_open_interest(symbol: str, period: str = "1h", limit: int = 500) -> int:
    import requests

    resp = requests.get(
        f"{BINANCE_FUTURES_BASE_URL}/futures/data/openInterestHist",
        params={"symbol": symbol.upper(), "period": period, "limit": int(limit)},
        timeout=20,
    )
    resp.raise_for_status()
    rows = [
        {"datetime_utc": pd.to_datetime(int(d["timestamp"]), unit="ms", utc=True), "value": float(d["sumOpenInterest"])}
        for d in resp.json()
        if d.get("timestamp") and d.get("sumOpenInterest") is not None
    ]
    return save_external_metric("binance_futures", "open_interest", rows, symbol=symbol)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest leakage-safe external context data.")
    parser.add_argument("--ingest", nargs="*", default=[], choices=["fear_greed", "funding", "open_interest"])
    parser.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--limit", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_external_data_table()
    if not ENABLE_EXTERNAL_DATA:
        print("ENABLE_EXTERNAL_DATA is false; refusing to make network calls. Set it to true to ingest.")
        return
    if "fear_greed" in args.ingest:
        print(f"fear_greed rows: {ingest_fear_greed(limit=min(args.limit, 2000))}")
    for symbol in args.symbols:
        if "funding" in args.ingest:
            print(f"[{symbol}] funding_rate rows: {ingest_funding_rate(symbol, limit=args.limit)}")
        if "open_interest" in args.ingest:
            print(f"[{symbol}] open_interest rows: {ingest_open_interest(symbol, limit=min(args.limit, 500))}")
    print("External data ingestion complete.")


if __name__ == "__main__":
    main()
