import sqlite3
from typing import Optional

import pandas as pd

from config import (
    DB_FILE,
    DATA_DIR,
    RAW_DIR,
    DB_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    QUALITY_LOGS_DIR,
    FEATURES_TABLE,
    FEATURE_COLUMNS,
    DATA_COVERAGE_TABLE,
    MODEL_REGISTRY_TABLE,
    SIGNALS_TABLE,
)


def ensure_project_directories() -> None:
    for path in [DATA_DIR, RAW_DIR, DB_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, QUALITY_LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_FILE)


def init_research_tables() -> None:
    ensure_project_directories()

    feature_columns_sql = ",\n        ".join([f"{col} REAL" for col in FEATURE_COLUMNS])

    conn = get_connection()
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {FEATURES_TABLE} (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                datetime_utc TEXT NOT NULL,
                close REAL,
                fwd_return_1 REAL,
                fwd_return_horizon REAL,
                {feature_columns_sql},
                label_class INTEGER,
                label_name TEXT,
                label_position INTEGER,
                feature_version TEXT,
                label_version TEXT,
                updated_at_utc TEXT,
                PRIMARY KEY (symbol, timeframe, datetime_utc)
            )
            """
        )

        conn.execute(
            f"""
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
            """
        )

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MODEL_REGISTRY_TABLE} (
                model_id TEXT PRIMARY KEY,
                symbol_scope TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                train_start TEXT,
                train_end TEXT,
                test_start TEXT,
                test_end TEXT,
                feature_version TEXT,
                label_version TEXT,
                model_path TEXT NOT NULL,
                training_ts_utc TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                params_json TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SIGNALS_TABLE} (
                signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                datetime_utc TEXT NOT NULL,
                signal_position INTEGER NOT NULL,
                signal_label TEXT NOT NULL,
                prob_short REAL NOT NULL,
                prob_flat REAL NOT NULL,
                prob_long REAL NOT NULL,
                created_at_utc TEXT NOT NULL,
                UNIQUE(model_id, symbol, timeframe, datetime_utc)
            )
            """
        )

        conn.commit()
    finally:
        conn.close()


def upsert_data_coverage(
    dataset: str,
    symbol: str,
    timeframe: str,
    min_datetime_utc: Optional[str],
    max_datetime_utc: Optional[str],
    row_count: int,
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {DATA_COVERAGE_TABLE} (
                dataset,
                symbol,
                timeframe,
                min_datetime_utc,
                max_datetime_utc,
                row_count,
                updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset,
                symbol,
                timeframe,
                min_datetime_utc,
                max_datetime_utc,
                int(row_count),
                pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def refresh_coverage_from_table(table_name: str, dataset: str, symbol: str, timeframe: str) -> None:
    conn = get_connection()
    try:
        row = pd.read_sql_query(
            f"""
            SELECT
                MIN(datetime_utc) AS min_datetime_utc,
                MAX(datetime_utc) AS max_datetime_utc,
                COUNT(*) AS row_count
            FROM {table_name}
            WHERE symbol = ? AND timeframe = ?
            """,
            conn,
            params=(symbol, timeframe),
        )
    finally:
        conn.close()

    if row.empty:
        upsert_data_coverage(dataset, symbol, timeframe, None, None, 0)
        return

    upsert_data_coverage(
        dataset=dataset,
        symbol=symbol,
        timeframe=timeframe,
        min_datetime_utc=row.loc[0, "min_datetime_utc"],
        max_datetime_utc=row.loc[0, "max_datetime_utc"],
        row_count=int(row.loc[0, "row_count"]),
    )
