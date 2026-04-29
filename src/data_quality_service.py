import argparse
import json
import sqlite3

import pandas as pd

from config import DATA_COVERAGE_TABLE, DATA_GAPS_TABLE, DB_FILE, PRICES_TABLE, SYMBOLS, TIMEFRAME
from data_loader import compute_gaps_for_symbol, replace_gaps_for_symbol, update_price_coverage
from db_utils import init_research_tables


def check_symbol_quality(symbol: str, timeframe: str) -> dict:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(
            f"SELECT datetime_utc FROM {PRICES_TABLE} WHERE symbol = ? AND timeframe = ? ORDER BY datetime_utc",
            conn,
            params=(symbol, timeframe),
        )
    finally:
        conn.close()
    if df.empty:
        return {"symbol": symbol, "timeframe": timeframe, "rows": 0, "status": "empty"}
    dt = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    duplicates = int(df["datetime_utc"].duplicated().sum())
    monotonic = bool(dt.is_monotonic_increasing)
    null_ts = int(dt.isna().sum())
    gaps = compute_gaps_for_symbol(symbol, timeframe)
    replace_gaps_for_symbol(symbol, timeframe, gaps)
    update_price_coverage(symbol, timeframe)
    latest = dt.max()
    latency_hours = float((pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 3600.0) if pd.notna(latest) else None
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": int(len(df)),
        "duplicates": duplicates,
        "monotonic": monotonic,
        "null_timestamps": null_ts,
        "latest_candle_utc": latest.isoformat() if pd.notna(latest) else None,
        "latency_hours": latency_hours,
        "gaps": int(len(gaps)),
        "missing_bars": int(gaps["missing_bars"].sum()) if not gaps.empty else 0,
        "status": "ok" if duplicates == 0 and monotonic and null_ts == 0 else "warning",
    }


def run_quality_checks(symbols: list[str] | None = None, timeframe: str = TIMEFRAME) -> list[dict]:
    symbols = [s.upper() for s in (symbols or SYMBOLS)]
    return [check_symbol_quality(symbol, timeframe) for symbol in symbols]


def main():
    parser = argparse.ArgumentParser(description="Data quality service: gaps, duplicates, UTC timestamps, coverage.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    args = parser.parse_args()
    print(json.dumps(run_quality_checks(args.symbols, args.timeframe), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
