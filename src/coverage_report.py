import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from config import DB_FILE, DATA_GAPS_TABLE, PRICES_TABLE, REPORTS_DIR, SYMBOLS, TIMEFRAME
from data_loader import timeframe_to_freq


def _parse_symbols(raw: list[str] | None) -> list[str]:
    if not raw:
        return [s.upper() for s in SYMBOLS]
    out: list[str] = []
    for item in raw:
        out.extend(s.strip().upper() for s in item.split(",") if s.strip())
    return sorted(set(out))


def coverage_for(symbol: str, timeframe: str) -> dict:
    conn = sqlite3.connect(DB_FILE)
    try:
        row = pd.read_sql_query(
            f"""
            SELECT MIN(datetime_utc) AS min_datetime_utc,
                   MAX(datetime_utc) AS max_datetime_utc,
                   COUNT(*) AS row_count
            FROM {PRICES_TABLE}
            WHERE symbol = ? AND timeframe = ?
            """,
            conn,
            params=(symbol, timeframe),
        )
        gaps = pd.read_sql_query(
            f"""
            SELECT COUNT(*) AS gaps_pending,
                   COALESCE(SUM(missing_bars), 0) AS pending_missing_bars
            FROM {DATA_GAPS_TABLE}
            WHERE symbol = ? AND timeframe = ?
            """,
            conn,
            params=(symbol, timeframe),
        )
    finally:
        conn.close()

    min_dt_raw = row.loc[0, "min_datetime_utc"]
    max_dt_raw = row.loc[0, "max_datetime_utc"]
    row_count = int(row.loc[0, "row_count"])

    expected_rows = 0
    coverage_pct = 0.0
    if pd.notna(min_dt_raw) and pd.notna(max_dt_raw) and row_count > 0:
        min_dt = pd.to_datetime(min_dt_raw, utc=True)
        max_dt = pd.to_datetime(max_dt_raw, utc=True)
        delta = pd.Timedelta(timeframe_to_freq(timeframe))
        expected_rows = int(((max_dt - min_dt) / delta) + 1)
        coverage_pct = float(row_count / expected_rows * 100.0) if expected_rows else 0.0

    gaps_pending = int(gaps.loc[0, "gaps_pending"]) if not gaps.empty else 0
    pending_missing_bars = int(gaps.loc[0, "pending_missing_bars"]) if not gaps.empty else 0

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "min_datetime_utc": min_dt_raw,
        "max_datetime_utc": max_dt_raw,
        "row_count": row_count,
        "expected_rows": expected_rows,
        "coverage_pct": round(coverage_pct, 6),
        "gaps_detected": gaps_pending,
        "gaps_resolved": 0 if gaps_pending else None,
        "gaps_pending": gaps_pending,
        "pending_missing_bars": pending_missing_bars,
    }


def build_report(symbols: list[str], timeframes: list[str]) -> pd.DataFrame:
    rows = []
    for timeframe in timeframes:
        for symbol in symbols:
            rows.append(coverage_for(symbol=symbol, timeframe=timeframe))
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SQLite market-data coverage report.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframes", nargs="*", default=[TIMEFRAME])
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    symbols = _parse_symbols(args.symbols)
    timeframes = [tf.strip() for item in args.timeframes for tf in item.split(",") if tf.strip()]
    report = build_report(symbols=symbols, timeframes=timeframes)

    if args.output:
        out = Path(args.output)
    else:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
        out = REPORTS_DIR / f"coverage_report_{stamp}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)

    print(report.to_string(index=False))
    print(f"coverage_report_csv={out}")
    print(json.dumps({"rows": report.to_dict(orient="records"), "csv": str(out)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
