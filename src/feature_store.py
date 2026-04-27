import argparse
import sqlite3
from typing import Iterable
import re

import pandas as pd

from config import (
    DB_FILE,
    PRICES_TABLE,
    FEATURES_TABLE,
    SYMBOLS,
    TIMEFRAME,
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    LABEL_VERSION,
    LOOKAHEAD_BARS,
    TP_MULTIPLIER,
    SL_MULTIPLIER,
    FEATURE_STORE_RECALC_OVERLAP_BARS,
    FEATURE_STORE_WARMUP_BARS,
)
from db_utils import init_research_tables, refresh_coverage_from_table
from features import compute_features
from labels import generate_triple_barrier_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Build and persist features + labels into SQLite feature store.")
    parser.add_argument("--symbols", nargs="*", default=None, help="Symbols to process. Default: config SYMBOLS")
    parser.add_argument("--timeframe", default=TIMEFRAME, help="Timeframe to process")
    parser.add_argument("--lookahead-bars", type=int, default=LOOKAHEAD_BARS)
    parser.add_argument("--tp-multiplier", type=float, default=TP_MULTIPLIER)
    parser.add_argument("--sl-multiplier", type=float, default=SL_MULTIPLIER)
    parser.add_argument(
        "--recalc-overlap-bars",
        type=int,
        default=FEATURE_STORE_RECALC_OVERLAP_BARS,
        help="Number of latest bars to recompute on each run to keep rolling features/labels consistent.",
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=FEATURE_STORE_WARMUP_BARS,
        help="Extra bars before the overlap window to compute rolling indicators correctly.",
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Recompute and upsert full history for selected symbols/timeframe.",
    )
    return parser.parse_args()


def available_symbols_for_timeframe(timeframe: str) -> list[str]:
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT DISTINCT symbol
            FROM {PRICES_TABLE}
            WHERE timeframe = ?
            ORDER BY symbol
            """,
            conn,
            params=(timeframe,),
        )
    finally:
        conn.close()

    if df.empty:
        return []
    return df["symbol"].astype(str).tolist()


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    match = re.fullmatch(r"(\d+)([mhdwM])", timeframe)
    if not match:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    if unit == "w":
        return pd.Timedelta(weeks=value)
    if unit == "M":
        return pd.Timedelta(days=30 * value)

    raise ValueError(f"Unsupported timeframe unit: {timeframe}")


def get_latest_feature_datetime(symbol: str, timeframe: str) -> pd.Timestamp | None:
    conn = sqlite3.connect(DB_FILE)
    try:
        row = pd.read_sql_query(
            f"""
            SELECT MAX(datetime_utc) AS max_dt
            FROM {FEATURES_TABLE}
            WHERE symbol = ? AND timeframe = ?
            """,
            conn,
            params=(symbol, timeframe),
        )
    finally:
        conn.close()

    if row.empty:
        return None

    raw = row.loc[0, "max_dt"]
    if pd.isna(raw):
        return None

    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def load_prices(symbol: str, timeframe: str, start_dt: pd.Timestamp | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    try:
        params = [symbol, timeframe]
        where_extra = ""
        if start_dt is not None:
            where_extra = " AND datetime_utc >= ?"
            params.append(start_dt.isoformat())

        df = pd.read_sql_query(
            f"""
            SELECT symbol, timeframe, datetime_utc, open, high, low, close, volume
            FROM {PRICES_TABLE}
            WHERE symbol = ? AND timeframe = ? {where_extra}
            ORDER BY datetime_utc
            """,
            conn,
            params=tuple(params),
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime_utc", "open", "high", "low", "close", "volume"]).copy()
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df


def _db_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _to_upsert_rows(df: pd.DataFrame) -> list[tuple]:
    rows = []
    for _, row in df.iterrows():
        values = [
            row["symbol"],
            row["timeframe"],
            row["datetime_utc"].isoformat(),
            _db_value(row["close"]),
            _db_value(row["fwd_return_1"]),
            _db_value(row["fwd_return_horizon"]),
        ]
        values.extend(_db_value(row[col]) for col in FEATURE_COLUMNS)
        values.extend(
            [
                _db_value(row["label_class"]),
                _db_value(row["label_name"]),
                _db_value(row["label_position"]),
                _db_value(row["feature_version"]),
                _db_value(row["label_version"]),
                _db_value(row["updated_at_utc"]),
            ]
        )
        rows.append(tuple(values))
    return rows


def upsert_feature_rows(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    feature_sql_columns = ", ".join(FEATURE_COLUMNS)
    placeholders = ", ".join(["?"] * (6 + len(FEATURE_COLUMNS) + 6))

    query = f"""
    INSERT OR REPLACE INTO {FEATURES_TABLE} (
        symbol,
        timeframe,
        datetime_utc,
        close,
        fwd_return_1,
        fwd_return_horizon,
        {feature_sql_columns},
        label_class,
        label_name,
        label_position,
        feature_version,
        label_version,
        updated_at_utc
    ) VALUES ({placeholders})
    """

    rows = _to_upsert_rows(df)
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.executemany(query, rows)
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def build_feature_frame(
    prices_df: pd.DataFrame,
    lookahead_bars: int,
    tp_multiplier: float,
    sl_multiplier: float,
) -> pd.DataFrame:
    feature_df = compute_features(prices_df)
    labeled_df = generate_triple_barrier_labels(
        feature_df,
        lookahead_bars=lookahead_bars,
        tp_multiplier=tp_multiplier,
        sl_multiplier=sl_multiplier,
    )
    labeled_df["feature_version"] = FEATURE_VERSION
    labeled_df["label_version"] = LABEL_VERSION
    labeled_df["updated_at_utc"] = pd.Timestamp.now(tz="UTC")
    return labeled_df


def run_feature_store(symbols: Iterable[str], timeframe: str, args) -> None:
    init_research_tables()
    timeframe_delta = timeframe_to_timedelta(timeframe)

    for symbol in symbols:
        latest_feature_dt = None if args.full_rebuild else get_latest_feature_datetime(symbol, timeframe)

        if latest_feature_dt is None:
            write_start_dt = None
            compute_start_dt = None
            mode = "full"
        else:
            write_start_dt = latest_feature_dt - (timeframe_delta * args.recalc_overlap_bars)
            compute_start_dt = write_start_dt - (timeframe_delta * args.warmup_bars)
            mode = "incremental"

        prices_df = load_prices(symbol, timeframe, start_dt=compute_start_dt)
        if prices_df.empty:
            print(f"[{symbol}] no data in prices table for timeframe={timeframe}.")
            continue

        full_df = build_feature_frame(
            prices_df=prices_df,
            lookahead_bars=args.lookahead_bars,
            tp_multiplier=args.tp_multiplier,
            sl_multiplier=args.sl_multiplier,
        )

        if write_start_dt is not None:
            full_df = full_df[full_df["datetime_utc"] >= write_start_dt].copy()

        written = upsert_feature_rows(full_df)
        usable_features = int(full_df[FEATURE_COLUMNS].dropna().shape[0]) if not full_df.empty else 0
        labeled_rows = int(full_df["label_class"].notna().sum()) if not full_df.empty else 0

        print(f"[{symbol}] mode={mode}")
        if mode == "incremental":
            print(f"[{symbol}] latest feature dt in DB: {latest_feature_dt}")
            print(f"[{symbol}] recomputed write window from: {write_start_dt}")
            print(f"[{symbol}] compute warmup start from: {compute_start_dt}")
        print(f"[{symbol}] rows upserted into feature store: {written}")
        print(f"[{symbol}] rows with complete features: {usable_features}")
        print(f"[{symbol}] rows with label_class not null: {labeled_rows}")

        refresh_coverage_from_table(PRICES_TABLE, "prices", symbol, timeframe)
        refresh_coverage_from_table(FEATURES_TABLE, "features", symbol, timeframe)


def main():
    args = parse_args()

    available = available_symbols_for_timeframe(args.timeframe)
    if not available:
        print("No data available in prices table for selected timeframe.")
        return

    if args.symbols:
        symbols = [s.upper().strip() for s in args.symbols if s.upper().strip() in set(available)]
    else:
        symbols = [s for s in SYMBOLS if s in set(available)]

    if not symbols:
        print("No valid symbols selected.")
        return

    print(f"Building feature store for symbols={symbols} timeframe={args.timeframe}")
    run_feature_store(symbols, args.timeframe, args)
    print("Feature store updated successfully.")


if __name__ == "__main__":
    main()
