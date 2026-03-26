import sqlite3

import pandas as pd

from config import (
    DB_FILE,
    PRICES_TABLE,
    DATA_GAPS_TABLE,
    QUALITY_LOGS_DIR,
)


INTERVAL_TO_FREQ = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1D",
}


def timeframe_to_freq(timeframe: str) -> str:
    if timeframe not in INTERVAL_TO_FREQ:
        raise ValueError(f"Timeframe no soportado para chequeo de gaps: {timeframe}")
    return INTERVAL_TO_FREQ[timeframe]


def get_available_symbol_timeframes() -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    try:
        query = f"""
        SELECT DISTINCT symbol, timeframe
        FROM {PRICES_TABLE}
        ORDER BY symbol, timeframe
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    return df


def compute_gaps_for_symbol(symbol: str, timeframe: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    try:
        query = f"""
        SELECT symbol, timeframe, datetime_utc
        FROM {PRICES_TABLE}
        WHERE symbol = ? AND timeframe = ?
        ORDER BY datetime_utc
        """
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
    finally:
        conn.close()

    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=[
            "symbol", "timeframe", "gap_start_utc", "gap_end_utc", "missing_bars", "detected_at_utc"
        ])

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna().sort_values("datetime_utc").reset_index(drop=True)

    freq = timeframe_to_freq(timeframe)
    expected_delta = pd.Timedelta(freq)

    gaps = []

    for i in range(1, len(df)):
        prev_dt = df.loc[i - 1, "datetime_utc"]
        curr_dt = df.loc[i, "datetime_utc"]
        delta = curr_dt - prev_dt

        if delta > expected_delta:
            missing_bars = int(delta / expected_delta) - 1
            gaps.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "gap_start_utc": prev_dt + expected_delta,
                "gap_end_utc": curr_dt - expected_delta,
                "missing_bars": missing_bars,
                "detected_at_utc": pd.Timestamp.now(tz="UTC"),
            })

    return pd.DataFrame(gaps)


def replace_gaps_for_symbol(symbol: str, timeframe: str, gaps_df: pd.DataFrame):
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(f"DELETE FROM {DATA_GAPS_TABLE} WHERE symbol = ? AND timeframe = ?", (symbol, timeframe))

        if not gaps_df.empty:
            rows = []
            for _, row in gaps_df.iterrows():
                rows.append((
                    row["symbol"],
                    row["timeframe"],
                    row["gap_start_utc"].isoformat(),
                    row["gap_end_utc"].isoformat(),
                    int(row["missing_bars"]),
                    row["detected_at_utc"].isoformat(),
                ))

            conn.executemany(f"""
            INSERT INTO {DATA_GAPS_TABLE} (
                symbol, timeframe, gap_start_utc, gap_end_utc, missing_bars, detected_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, rows)

        conn.commit()
    finally:
        conn.close()


def export_gap_report(symbol: str, timeframe: str, gaps_df: pd.DataFrame):
    QUALITY_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = QUALITY_LOGS_DIR / f"gaps_{symbol}_{timeframe}.csv"

    if gaps_df.empty:
        pd.DataFrame(columns=[
            "symbol", "timeframe", "gap_start_utc", "gap_end_utc", "missing_bars", "detected_at_utc"
        ]).to_csv(output_path, index=False)
    else:
        gaps_df.to_csv(output_path, index=False)

    print(f"  → gap report: {output_path}")


def ask_symbols(available_df: pd.DataFrame) -> list[str]:
    available_symbols = sorted(available_df["symbol"].unique().tolist())

    print("\nSímbolos disponibles en BD:")
    print(", ".join(available_symbols))

    raw = input("\n¿Qué símbolo quieres chequear? (uno o varios separados por coma, o 'all'): ").strip()

    if raw.lower() == "all":
        return available_symbols

    selected = [s.strip().upper() for s in raw.split(",") if s.strip()]
    selected = [s for s in selected if s in available_symbols]

    if not selected:
        print("No has introducido ningún símbolo válido.")
        return []

    return selected


def main():
    available_df = get_available_symbol_timeframes()

    if available_df.empty:
        print("No hay datos en la base de datos.")
        return

    selected_symbols = ask_symbols(available_df)
    if not selected_symbols:
        return

    print("\n=== DATA QUALITY: GAP CHECK ===")

    for symbol in selected_symbols:
        symbol_rows = available_df[available_df["symbol"] == symbol]

        for timeframe in sorted(symbol_rows["timeframe"].unique()):
            print(f"\nRevisando {symbol} {timeframe}...")

            gaps_df = compute_gaps_for_symbol(symbol, timeframe)
            replace_gaps_for_symbol(symbol, timeframe, gaps_df)
            export_gap_report(symbol, timeframe, gaps_df)

            if gaps_df.empty:
                print("  → sin gaps detectados")
            else:
                total_missing = int(gaps_df["missing_bars"].sum())
                print(f"  → gaps detectados: {len(gaps_df)}")
                print(f"  → velas faltantes totales: {total_missing}")
                print(gaps_df[["gap_start_utc", "gap_end_utc", "missing_bars"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()