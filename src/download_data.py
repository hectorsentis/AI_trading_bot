import argparse
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
    SYMBOLS,
    TIMEFRAME,
    DATA_PROVIDER,
    BINANCE_REST_BASE_URL,
    KLINES_LIMIT,
    INITIAL_BACKFILL_DAYS,
    OVERLAP_BARS,
    API_SLEEP_SECONDS,
    HTTP_TIMEOUT_SECONDS,
    RAW_FILE_TIMESTAMP_FORMAT,
    SAVE_RAW_AS_GZIP,
    DEFAULT_DOWNLOAD_MODE,
    DEFAULT_RECENT_BARS,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    PRICE_COLUMNS,
)

KLINES_ENDPOINT = "/api/v3/klines"

# Mapeo a milisegundos por vela
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
    "1M": 30 * 24 * 60 * 60_000,  # aproximación para uso interno
}


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    Path(DB_FILE).parent.mkdir(parents=True, exist_ok=True)
    for symbol in SYMBOLS:
        (RAW_DIR / symbol).mkdir(parents=True, exist_ok=True)


def interval_to_ms(interval: str) -> int:
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Intervalo no soportado en interval_to_ms: {interval}")
    return INTERVAL_TO_MS[interval]


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Descarga histórico de Binance Spot a snapshots raw.")
    parser.add_argument("--mode", choices=["incremental", "full", "recent", "range"], default=DEFAULT_DOWNLOAD_MODE)
    parser.add_argument("--symbols", nargs="*", default=None, help="Ej: BTCUSDT ETHUSDT")
    parser.add_argument("--timeframe", default=TIMEFRAME, help="Ej: 1m, 5m, 1h, 1d")
    parser.add_argument("--recent-bars", type=int, default=DEFAULT_RECENT_BARS)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE, help="Ej: 2024-01-01")
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE, help="Ej: 2025-12-31")
    parser.add_argument("--dry-run", action="store_true", help="Muestra qué haría sin escribir CSV")
    parser.add_argument("--force-empty-snapshot", action="store_true", help="Guarda snapshot aunque no haya filas nuevas")
    return parser.parse_args()


def get_latest_datetime_from_db(symbol: str, timeframe: str) -> Optional[pd.Timestamp]:
    if not Path(DB_FILE).exists():
        return None

    conn = sqlite3.connect(DB_FILE)
    try:
        table_check = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name='prices'
            """,
            conn,
        )
        if table_check.empty:
            return None

        row = pd.read_sql_query(
            """
            SELECT MAX(datetime_utc) AS max_dt
            FROM prices
            WHERE symbol = ? AND timeframe = ?
            """,
            conn,
            params=(symbol, timeframe),
        )

        if row.empty:
            return None

        max_dt = row.loc[0, "max_dt"]
        if pd.isna(max_dt):
            return None

        ts = pd.to_datetime(max_dt, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts
    finally:
        conn.close()


def get_binance_server_time() -> pd.Timestamp:
    url = f"{BINANCE_REST_BASE_URL}/api/v3/time"
    r = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    r.raise_for_status()
    payload = r.json()
    server_time_ms = payload["serverTime"]
    return pd.to_datetime(server_time_ms, unit="ms", utc=True)


def date_str_to_utc_ts(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str, tz="UTC")


def build_time_window(
    mode: str,
    symbol: str,
    timeframe: str,
    recent_bars: int,
    start_date: Optional[str],
    end_date: Optional[str],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    now_utc = get_binance_server_time()
    tf_ms = interval_to_ms(timeframe)

    if mode == "full":
        start_ts = now_utc - pd.Timedelta(days=INITIAL_BACKFILL_DAYS)
        end_ts = now_utc
        return start_ts, end_ts

    if mode == "recent":
        start_ts = now_utc - pd.Timedelta(milliseconds=recent_bars * tf_ms)
        end_ts = now_utc
        return start_ts, end_ts

    if mode == "range":
        if not start_date or not end_date:
            raise ValueError("En mode=range debes pasar --start-date y --end-date")
        start_ts = date_str_to_utc_ts(start_date)
        end_ts = date_str_to_utc_ts(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        return start_ts, end_ts

    # incremental
    latest_dt = get_latest_datetime_from_db(symbol, timeframe)
    if latest_dt is None:
        start_ts = now_utc - pd.Timedelta(days=INITIAL_BACKFILL_DAYS)
    else:
        start_ts = latest_dt - pd.Timedelta(milliseconds=OVERLAP_BARS * tf_ms)

    end_ts = now_utc
    return start_ts, end_ts


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
        return pd.DataFrame(columns=PRICE_COLUMNS)

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

    while cursor < end_ms:
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

        # avanzar al siguiente bloque usando el último open time + 1 vela
        last_open_time_ms = page[-1][0]
        next_cursor = last_open_time_ms + tf_ms

        if next_cursor <= cursor:
            break

        cursor = next_cursor
        time.sleep(API_SLEEP_SECONDS)

        # si la página vino incompleta, ya no suele haber más dentro del rango
        if len(page) < KLINES_LIMIT:
            break

    if not all_pages:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    df = pd.concat(all_pages, ignore_index=True)
    df = df.drop_duplicates(subset=["symbol", "timeframe", "datetime_utc"], keep="last")
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    # recorte final por seguridad
    df = df[(df["datetime_utc"] >= start_ts) & (df["datetime_utc"] <= end_ts)].copy()
    return df


def filter_incremental_new_rows(df: pd.DataFrame, symbol: str, timeframe: str, mode: str) -> pd.DataFrame:
    if mode != "incremental":
        return df

    latest_dt = get_latest_datetime_from_db(symbol, timeframe)
    if latest_dt is None:
        return df

    return df[df["datetime_utc"] > latest_dt].copy()


def save_raw_snapshot(symbol: str, timeframe: str, df: pd.DataFrame, dry_run: bool) -> Optional[Path]:
    symbol_dir = RAW_DIR / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    snapshot_ts = datetime.now(timezone.utc).strftime(RAW_FILE_TIMESTAMP_FORMAT)
    suffix = ".csv.gz" if SAVE_RAW_AS_GZIP else ".csv"
    file_path = symbol_dir / f"{symbol}_{timeframe}_{snapshot_ts}{suffix}"

    if dry_run:
        return file_path

    if SAVE_RAW_AS_GZIP:
        df.to_csv(file_path, index=False, compression="gzip")
    else:
        df.to_csv(file_path, index=False)

    return file_path


def download_symbol(
    symbol: str,
    timeframe: str,
    mode: str,
    recent_bars: int,
    start_date: Optional[str],
    end_date: Optional[str],
    dry_run: bool,
    force_empty_snapshot: bool,
) -> None:
    start_ts, end_ts = build_time_window(
        mode=mode,
        symbol=symbol,
        timeframe=timeframe,
        recent_bars=recent_bars,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"\n[{symbol}] modo={mode} timeframe={timeframe}")
    print(f"[{symbol}] rango UTC: {start_ts} -> {end_ts}")

    df = fetch_klines_range(symbol, timeframe, start_ts, end_ts)
    df = filter_incremental_new_rows(df, symbol, timeframe, mode)
    df = df.drop_duplicates(subset=["symbol", "timeframe", "datetime_utc"], keep="last")
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    print(f"[{symbol}] filas descargadas limpias: {len(df)}")

    if df.empty and not force_empty_snapshot:
        print(f"[{symbol}] no hay datos nuevos; no se guarda snapshot")
        return

    out = save_raw_snapshot(symbol, timeframe, df, dry_run=dry_run)
    if dry_run:
        print(f"[{symbol}] dry-run: guardaría snapshot en {out}")
    else:
        print(f"[{symbol}] snapshot guardado en: {out}")


def main():
    args = parse_cli_args()
    ensure_directories()

    symbols = args.symbols if args.symbols else SYMBOLS
    timeframe = args.timeframe

    for symbol in symbols:
        try:
            download_symbol(
                symbol=symbol,
                timeframe=timeframe,
                mode=args.mode,
                recent_bars=args.recent_bars,
                start_date=args.start_date,
                end_date=args.end_date,
                dry_run=args.dry_run,
                force_empty_snapshot=args.force_empty_snapshot,
            )
        except Exception as exc:
            print(f"[{symbol}] ERROR: {exc}")


if __name__ == "__main__":
    main()