import sqlite3
from pathlib import Path

import pandas as pd

from config import (
    RAW_DIR,
    DB_FILE,
    TIMEFRAME,
    PRICE_COLUMNS,
    PRICES_TABLE,
    INGESTION_LOG_TABLE,
    DATA_GAPS_TABLE,
    DATA_COVERAGE_TABLE,
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


def init_db():
    conn = sqlite3.connect(DB_FILE)

    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {PRICES_TABLE} (
        symbol TEXT,
        timeframe TEXT,
        datetime_utc TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        close_time_utc TEXT,
        quote_asset_volume REAL,
        number_of_trades REAL,
        taker_buy_base_volume REAL,
        taker_buy_quote_volume REAL,
        provider TEXT,
        ingestion_ts_utc TEXT,
        PRIMARY KEY (symbol, timeframe, datetime_utc)
    )
    """)

    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {INGESTION_LOG_TABLE} (
        file_path TEXT PRIMARY KEY,
        symbol TEXT,
        timeframe TEXT,
        rows_in_file INTEGER,
        rows_loaded INTEGER,
        loaded_at_utc TEXT
    )
    """)

    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {DATA_GAPS_TABLE} (
        symbol TEXT,
        timeframe TEXT,
        gap_start_utc TEXT,
        gap_end_utc TEXT,
        missing_bars INTEGER,
        detected_at_utc TEXT
    )
    """)

    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {DATA_COVERAGE_TABLE} (
        dataset TEXT NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        min_datetime_utc TEXT,
        max_datetime_utc TEXT,
        row_count INTEGER NOT NULL,
        updated_at_utc TEXT NOT NULL,
        PRIMARY KEY (dataset, symbol, timeframe)
    )
    """)

    conn.commit()
    conn.close()


def load_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    required = set(PRICE_COLUMNS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"El archivo {file_path.name} no tiene columnas requeridas: {missing}")

    datetime_cols = ["datetime_utc", "close_time_utc", "ingestion_ts_utc"]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

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

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["timeframe"] = df["timeframe"].astype(str).str.strip()
    df["provider"] = df["provider"].astype(str).str.strip()

    df = df.dropna(subset=[
        "symbol",
        "timeframe",
        "datetime_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]).copy()

    df = df.sort_values("datetime_utc").drop_duplicates(
        subset=["symbol", "timeframe", "datetime_utc"],
        keep="last"
    ).reset_index(drop=True)

    return df


def already_processed(file_path: Path) -> bool:
    conn = sqlite3.connect(DB_FILE)
    try:
        query = f"""
        SELECT COUNT(*) as n
        FROM {INGESTION_LOG_TABLE}
        WHERE file_path = ?
        """
        row = pd.read_sql_query(query, conn, params=(str(file_path),))
        return int(row.loc[0, "n"]) > 0
    finally:
        conn.close()


def log_ingestion(file_path: Path, symbol: str, timeframe: str, rows_in_file: int, rows_loaded: int):
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(f"""
        INSERT OR REPLACE INTO {INGESTION_LOG_TABLE} (
            file_path, symbol, timeframe, rows_in_file, rows_loaded, loaded_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(file_path),
            symbol,
            timeframe,
            rows_in_file,
            rows_loaded,
            pd.Timestamp.now(tz="UTC").isoformat(),
        ))
        conn.commit()
    finally:
        conn.close()


def upsert_prices(df: pd.DataFrame) -> int:
    conn = sqlite3.connect(DB_FILE)

    query = f"""
    INSERT OR REPLACE INTO {PRICES_TABLE} (
        symbol, timeframe, datetime_utc,
        open, high, low, close, volume,
        close_time_utc, quote_asset_volume, number_of_trades,
        taker_buy_base_volume, taker_buy_quote_volume,
        provider, ingestion_ts_utc
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    data = []
    for _, row in df.iterrows():
        data.append((
            row["symbol"],
            row["timeframe"],
            row["datetime_utc"].isoformat(),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"]),
            row["close_time_utc"].isoformat() if pd.notna(row["close_time_utc"]) else None,
            float(row["quote_asset_volume"]) if pd.notna(row["quote_asset_volume"]) else None,
            float(row["number_of_trades"]) if pd.notna(row["number_of_trades"]) else None,
            float(row["taker_buy_base_volume"]) if pd.notna(row["taker_buy_base_volume"]) else None,
            float(row["taker_buy_quote_volume"]) if pd.notna(row["taker_buy_quote_volume"]) else None,
            row["provider"],
            row["ingestion_ts_utc"].isoformat() if pd.notna(row["ingestion_ts_utc"]) else None,
        ))

    conn.executemany(query, data)
    conn.commit()
    conn.close()

    return len(data)


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
            gap_start = prev_dt + expected_delta
            gap_end = curr_dt - expected_delta

            gaps.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "gap_start_utc": gap_start,
                "gap_end_utc": gap_end,
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

    print(f"    → gap report: {output_path}")


def update_price_coverage(symbol: str, timeframe: str):
    conn = sqlite3.connect(DB_FILE)
    try:
        row = pd.read_sql_query(
            f"""
            SELECT
                MIN(datetime_utc) AS min_datetime_utc,
                MAX(datetime_utc) AS max_datetime_utc,
                COUNT(*) AS row_count
            FROM {PRICES_TABLE}
            WHERE symbol = ? AND timeframe = ?
            """,
            conn,
            params=(symbol, timeframe),
        )

        min_dt = row.loc[0, "min_datetime_utc"]
        max_dt = row.loc[0, "max_datetime_utc"]
        row_count = int(row.loc[0, "row_count"])

        conn.execute(
            f"""
            INSERT OR REPLACE INTO {DATA_COVERAGE_TABLE} (
                dataset, symbol, timeframe, min_datetime_utc, max_datetime_utc, row_count, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "prices",
                symbol,
                timeframe,
                min_dt,
                max_dt,
                row_count,
                pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def run_gap_checks_for_symbols(symbol_timeframes: set[tuple[str, str]]):
    if not symbol_timeframes:
        print("\nNo hay símbolos nuevos que revisar.")
        return

    print("\n=== DATA QUALITY: GAP CHECK ===")

    for symbol, timeframe in sorted(symbol_timeframes):
        print(f"\nRevisando {symbol} {timeframe}...")

        gaps_df = compute_gaps_for_symbol(symbol, timeframe)
        replace_gaps_for_symbol(symbol, timeframe, gaps_df)
        export_gap_report(symbol, timeframe, gaps_df)

        if gaps_df.empty:
            print("    → sin gaps detectados")
        else:
            total_missing = int(gaps_df["missing_bars"].sum())
            print(f"    → gaps detectados: {len(gaps_df)}")
            print(f"    → velas faltantes totales: {total_missing}")


def process_all_raw():
    total_files = 0
    total_rows_loaded = 0
    processed_symbols = set()

    for symbol_dir in RAW_DIR.iterdir():
        if not symbol_dir.is_dir():
            continue

        symbol = symbol_dir.name.upper().strip()
        print(f"\nProcesando símbolo: {symbol}")

        for file in sorted(symbol_dir.glob("*.csv")):
            if already_processed(file):
                print(f"  → {file.name} [ya procesado, se omite]")
                continue

            print(f"  → {file.name}")

            try:
                df = load_csv(file)
            except Exception as exc:
                print(f"     ERROR leyendo {file.name}: {exc}")
                continue

            if df.empty:
                print("     (vacío, se ignora)")
                log_ingestion(file, symbol, TIMEFRAME, 0, 0)
                continue

            rows_in_file = len(df)
            rows_loaded = upsert_prices(df)

            tf = df["timeframe"].iloc[0]
            processed_symbols.add((symbol, tf))

            log_ingestion(file, symbol, tf, rows_in_file, rows_loaded)

            total_files += 1
            total_rows_loaded += rows_loaded

    print("\n=== RESUMEN INGESTA ===")
    print(f"Archivos nuevos procesados: {total_files}")
    print(f"Filas cargadas/upsert: {total_rows_loaded}")

    for symbol, timeframe in sorted(processed_symbols):
        update_price_coverage(symbol, timeframe)

    return processed_symbols


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"s", "si", "sí", "y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Responde con s/n.")


def main():
    print("Inicializando base de datos...")
    init_db()

    print("Cargando snapshots raw a la BD...")
    processed_symbols = process_all_raw()

    if processed_symbols:
        do_check = ask_yes_no(
            "\n¿Quieres revisar si hay gaps en los datos que acabas de cargar? [s/n]: "
        )
        if do_check:
            run_gap_checks_for_symbols(processed_symbols)
    else:
        print("\nNo se han cargado archivos nuevos, así que no hay símbolos nuevos que revisar.")

    print("\nBase de datos actualizada correctamente.")


if __name__ == "__main__":
    main()
