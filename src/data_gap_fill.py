import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import (
    RAW_DIR,
    DB_FILE,
    BINANCE_REST_BASE_URL,
    KLINES_LIMIT,
    API_SLEEP_SECONDS,
    HTTP_TIMEOUT_SECONDS,
    DATA_PROVIDER,
    DATA_GAPS_TABLE,
    RAW_FILE_TIMESTAMP_FORMAT,
    SAVE_RAW_AS_GZIP,
)


KLINES_ENDPOINT = "/api/v3/klines"

INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
    "1M": 30 * 24 * 60 * 60_000,
}


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def interval_to_ms(interval: str) -> int:
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Intervalo no soportado: {interval}")
    return INTERVAL_TO_MS[interval]


def read_gaps(limit: Optional[int] = None) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    try:
        query = f"""
        SELECT symbol, timeframe, gap_start_utc, gap_end_utc, missing_bars, detected_at_utc
        FROM {DATA_GAPS_TABLE}
        ORDER BY symbol, timeframe, gap_start_utc
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    if df.empty:
        return df

    df["gap_start_utc"] = pd.to_datetime(df["gap_start_utc"], utc=True, errors="coerce")
    df["gap_end_utc"] = pd.to_datetime(df["gap_end_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["gap_start_utc", "gap_end_utc"]).reset_index(drop=True)

    if limit is not None:
        df = df.head(limit).copy()

    return df


def fetch_klines_page(symbol: str, timeframe: str, start_ms: int, end_ms: int, limit: int) -> list:
    url = f"{BINANCE_REST_BASE_URL}{KLINES_ENDPOINT}"
    params = {
        "symbol": symbol,
        "interval": timeframe,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }

    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
    if r.status_code == 429:
        raise RuntimeError(f"[{symbol}] Binance devolvió 429 Too Many Requests")
    r.raise_for_status()
    return r.json()


def normalize_klines(symbol: str, timeframe: str, raw_klines: list) -> pd.DataFrame:
    if not raw_klines:
        return pd.DataFrame(columns=[
            "symbol",
            "timeframe",
            "datetime_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_utc",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "provider",
            "ingestion_ts_utc",
        ])

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    df = pd.DataFrame(raw_klines, columns=cols)

    df["datetime_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time_utc"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["symbol"] = symbol
    df["timeframe"] = timeframe
    df["provider"] = DATA_PROVIDER
    df["ingestion_ts_utc"] = pd.Timestamp.now(tz="UTC")

    df = df[
        [
            "symbol",
            "timeframe",
            "datetime_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_utc",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "provider",
            "ingestion_ts_utc",
        ]
    ].copy()

    df = df.dropna().sort_values("datetime_utc").reset_index(drop=True)
    return df


def fetch_klines_range(symbol: str, timeframe: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    tf_ms = interval_to_ms(timeframe)
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    all_pages = []
    cursor = start_ms

    while cursor <= end_ms:
        page = fetch_klines_page(
            symbol=symbol,
            timeframe=timeframe,
            start_ms=cursor,
            end_ms=end_ms,
            limit=KLINES_LIMIT,
        )

        if not page:
            break

        df_page = normalize_klines(symbol, timeframe, page)
        if not df_page.empty:
            all_pages.append(df_page)

        last_open_time_ms = page[-1][0]
        next_cursor = last_open_time_ms + tf_ms

        if next_cursor <= cursor:
            break

        cursor = next_cursor
        time.sleep(API_SLEEP_SECONDS)

        if len(page) < KLINES_LIMIT:
            break

    if not all_pages:
        return pd.DataFrame()

    df = pd.concat(all_pages, ignore_index=True)
    df = df.drop_duplicates(subset=["symbol", "timeframe", "datetime_utc"], keep="last")
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    df = df[(df["datetime_utc"] >= start_ts) & (df["datetime_utc"] <= end_ts)].copy()
    return df


def save_gap_snapshot(symbol: str, timeframe: str, df: pd.DataFrame) -> Optional[Path]:
    if df.empty:
        print(f"[{symbol}] No se descargó nada para el gap.")
        return None

    symbol_dir = RAW_DIR / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    snapshot_ts = datetime.now(timezone.utc).strftime(RAW_FILE_TIMESTAMP_FORMAT)
    suffix = ".csv.gz" if SAVE_RAW_AS_GZIP else ".csv"
    file_path = symbol_dir / f"{symbol}_{timeframe}_gapfill_{snapshot_ts}{suffix}"

    df = df.drop_duplicates(subset=["symbol", "timeframe", "datetime_utc"], keep="last")
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    if SAVE_RAW_AS_GZIP:
        df.to_csv(file_path, index=False, compression="gzip")
    else:
        df.to_csv(file_path, index=False)

    return file_path


def fill_gap_row(row: pd.Series) -> None:
    symbol = row["symbol"]
    timeframe = row["timeframe"]
    start_ts = row["gap_start_utc"]
    end_ts = row["gap_end_utc"]
    missing_bars = row.get("missing_bars", None)

    print(f"\n[{symbol}] Gap detectado")
    print(f"  timeframe: {timeframe}")
    print(f"  start:     {start_ts}")
    print(f"  end:       {end_ts}")
    print(f"  faltantes: {missing_bars}")

    df = fetch_klines_range(symbol, timeframe, start_ts, end_ts)

    print(f"  velas descargadas para el gap: {len(df)}")

    out = save_gap_snapshot(symbol, timeframe, df)
    if out is not None:
        print(f"  snapshot guardado en: {out}")


def main():
    ensure_directories()

    gaps_df = read_gaps()

    if gaps_df.empty:
        print("No hay gaps registrados en la base de datos.")
        return

    print(f"Gaps encontrados: {len(gaps_df)}")

    for _, row in gaps_df.iterrows():
        try:
            fill_gap_row(row)
        except Exception as exc:
            print(f"[{row['symbol']}] ERROR rellenando gap: {exc}")


if __name__ == "__main__":
    main()