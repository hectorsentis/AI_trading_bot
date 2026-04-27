import json
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
    ORDERS_TABLE,
    POSITIONS_TABLE,
    PORTFOLIO_SNAPSHOTS_TABLE,
    VALIDATION_PREDICTIONS_TABLE,
)


def ensure_project_directories() -> None:
    for path in [DATA_DIR, RAW_DIR, DB_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, QUALITY_LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_FILE)


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r[1] for r in rows}


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_sql: str) -> None:
    cols = _table_columns(conn, table_name)
    if column_name in cols:
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")


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
        for feature_col in FEATURE_COLUMNS:
            _ensure_column(conn, FEATURES_TABLE, feature_col, "REAL")
        _ensure_column(conn, FEATURES_TABLE, "feature_version", "TEXT")
        _ensure_column(conn, FEATURES_TABLE, "label_version", "TEXT")

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
                status TEXT NOT NULL,
                acceptance_status TEXT NOT NULL DEFAULT 'candidate',
                rejection_reasons_json TEXT NOT NULL DEFAULT '[]',
                evaluation_scope TEXT NOT NULL DEFAULT 'holdout',
                is_active INTEGER NOT NULL DEFAULT 0,
                updated_at_utc TEXT
            )
            """
        )

        # Lightweight migration for legacy registries.
        _ensure_column(conn, MODEL_REGISTRY_TABLE, "acceptance_status", "TEXT NOT NULL DEFAULT 'candidate'")
        _ensure_column(conn, MODEL_REGISTRY_TABLE, "rejection_reasons_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, MODEL_REGISTRY_TABLE, "evaluation_scope", "TEXT NOT NULL DEFAULT 'holdout'")
        _ensure_column(conn, MODEL_REGISTRY_TABLE, "is_active", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, MODEL_REGISTRY_TABLE, "updated_at_utc", "TEXT")

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

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {VALIDATION_PREDICTIONS_TABLE} (
                pred_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                validation_run_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                datetime_utc TEXT NOT NULL,
                y_true INTEGER NOT NULL,
                y_pred INTEGER NOT NULL,
                prob_short REAL NOT NULL,
                prob_flat REAL NOT NULL,
                prob_long REAL NOT NULL,
                signal_position INTEGER NOT NULL,
                fold_id INTEGER NOT NULL,
                created_at_utc TEXT NOT NULL,
                UNIQUE(validation_run_id, symbol, timeframe, datetime_utc, fold_id)
            )
            """
        )

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {ORDERS_TABLE} (
                order_id TEXT PRIMARY KEY,
                model_id TEXT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal_datetime_utc TEXT,
                side TEXT NOT NULL,
                executable_action TEXT,
                order_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                requested_price REAL,
                fill_price REAL,
                notional REAL,
                status TEXT NOT NULL,
                reason TEXT,
                signal_position INTEGER,
                research_signal_label TEXT,
                decision_json TEXT,
                dry_run INTEGER NOT NULL,
                created_at_utc TEXT NOT NULL,
                filled_at_utc TEXT
            )
            """
        )
        _ensure_column(conn, ORDERS_TABLE, "signal_datetime_utc", "TEXT")
        _ensure_column(conn, ORDERS_TABLE, "executable_action", "TEXT")
        _ensure_column(conn, ORDERS_TABLE, "research_signal_label", "TEXT")
        _ensure_column(conn, ORDERS_TABLE, "decision_json", "TEXT")

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {POSITIONS_TABLE} (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_price REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                updated_at_utc TEXT NOT NULL,
                dry_run INTEGER NOT NULL,
                PRIMARY KEY (symbol, timeframe, dry_run)
            )
            """
        )

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {PORTFOLIO_SNAPSHOTS_TABLE} (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime_utc TEXT NOT NULL,
                cash REAL NOT NULL,
                equity REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                exposure_json TEXT NOT NULL,
                dry_run INTEGER NOT NULL
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


def save_validation_predictions(
    rows: list[dict],
    validation_run_id: str,
    replace_run: bool = True,
) -> int:
    if not rows:
        return 0

    conn = get_connection()
    try:
        if replace_run:
            conn.execute(
                f"DELETE FROM {VALIDATION_PREDICTIONS_TABLE} WHERE validation_run_id = ?",
                (validation_run_id,),
            )

        payload = []
        for row in rows:
            payload.append(
                (
                    row["model_id"],
                    validation_run_id,
                    row["symbol"],
                    row["timeframe"],
                    row["datetime_utc"],
                    int(row["y_true"]),
                    int(row["y_pred"]),
                    float(row["prob_short"]),
                    float(row["prob_flat"]),
                    float(row["prob_long"]),
                    int(row["signal_position"]),
                    int(row["fold_id"]),
                    row["created_at_utc"],
                )
            )

        conn.executemany(
            f"""
            INSERT OR REPLACE INTO {VALIDATION_PREDICTIONS_TABLE} (
                model_id,
                validation_run_id,
                symbol,
                timeframe,
                datetime_utc,
                y_true,
                y_pred,
                prob_short,
                prob_flat,
                prob_long,
                signal_position,
                fold_id,
                created_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        conn.commit()
    finally:
        conn.close()

    return len(payload)


def get_latest_validation_run_id(model_id: str, timeframe: Optional[str] = None) -> Optional[str]:
    conn = get_connection()
    try:
        where = ["model_id = ?"]
        params: list[object] = [model_id]
        if timeframe:
            where.append("timeframe = ?")
            params.append(timeframe)

        row = pd.read_sql_query(
            f"""
            SELECT validation_run_id, MAX(created_at_utc) AS latest_ts
            FROM {VALIDATION_PREDICTIONS_TABLE}
            WHERE {" AND ".join(where)}
            GROUP BY validation_run_id
            ORDER BY latest_ts DESC
            LIMIT 1
            """,
            conn,
            params=tuple(params),
        )
    finally:
        conn.close()

    if row.empty:
        return None
    return str(row.loc[0, "validation_run_id"])


def save_portfolio_snapshot(
    cash: float,
    equity: float,
    realized_pnl: float,
    unrealized_pnl: float,
    exposure_by_symbol: dict[str, float],
    dry_run: bool = True,
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            f"""
            INSERT INTO {PORTFOLIO_SNAPSHOTS_TABLE} (
                datetime_utc, cash, equity, realized_pnl, unrealized_pnl, exposure_json, dry_run
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pd.Timestamp.now(tz="UTC").isoformat(),
                float(cash),
                float(equity),
                float(realized_pnl),
                float(unrealized_pnl),
                json.dumps(exposure_by_symbol, ensure_ascii=True, sort_keys=True),
                1 if dry_run else 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()
