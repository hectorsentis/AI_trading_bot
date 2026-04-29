import json
import sqlite3
import argparse
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
    PRICES_TABLE,
    PRICE_COLUMNS,
    DATA_GAPS_TABLE,
    INGESTION_LOG_TABLE,
    FILLS_TABLE,
    PAPER_MODEL_METRICS_TABLE,
    MODEL_LIFECYCLE_EVENTS_TABLE,
    BOT_EVENTS_TABLE,
    RISK_EVENTS_TABLE,
    BOT_STATUS_TABLE,
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


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def init_research_tables() -> None:
    ensure_project_directories()

    feature_columns_sql = ",\n        ".join([f"{col} REAL" for col in FEATURE_COLUMNS])

    conn = get_connection()
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {PRICES_TABLE} (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                datetime_utc TEXT NOT NULL,
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
            """
        )
        for col in PRICE_COLUMNS:
            if col not in {"symbol", "timeframe", "datetime_utc"}:
                sql = "TEXT" if col in {"close_time_utc", "provider", "ingestion_ts_utc"} else "REAL"
                _ensure_column(conn, PRICES_TABLE, col, sql)

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {INGESTION_LOG_TABLE} (
                file_path TEXT PRIMARY KEY,
                symbol TEXT,
                timeframe TEXT,
                rows_in_file INTEGER,
                rows_loaded INTEGER,
                loaded_at_utc TEXT
            )
            """
        )

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {DATA_GAPS_TABLE} (
                gap_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                gap_start_utc TEXT,
                gap_end_utc TEXT,
                missing_bars INTEGER,
                detected_at_utc TEXT,
                severity TEXT DEFAULT 'warning',
                resolved INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        _ensure_column(conn, DATA_GAPS_TABLE, "severity", "TEXT DEFAULT 'warning'")
        _ensure_column(conn, DATA_GAPS_TABLE, "resolved", "INTEGER NOT NULL DEFAULT 0")

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
        _ensure_column(conn, MODEL_REGISTRY_TABLE, "paper_started_at_utc", "TEXT")
        _ensure_column(conn, MODEL_REGISTRY_TABLE, "paper_validated_at_utc", "TEXT")
        _ensure_column(conn, MODEL_REGISTRY_TABLE, "real_ready_at_utc", "TEXT")
        _ensure_column(conn, MODEL_REGISTRY_TABLE, "real_active_at_utc", "TEXT")
        _ensure_column(conn, MODEL_REGISTRY_TABLE, "account_mode", "TEXT")
        _ensure_column(conn, MODEL_REGISTRY_TABLE, "training_scope", "TEXT NOT NULL DEFAULT 'multi_symbol'")
        _ensure_column(conn, MODEL_REGISTRY_TABLE, "symbols_json", "TEXT NOT NULL DEFAULT '[]'")
        conn.execute(
            f"""
            UPDATE {MODEL_REGISTRY_TABLE}
            SET symbols_json = '[' || '"' || REPLACE(symbol_scope, ',', '","') || '"' || ']'
            WHERE (symbols_json IS NULL OR symbols_json = '[]' OR symbols_json = '')
              AND symbol_scope IS NOT NULL
              AND symbol_scope != ''
            """
        )

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MODEL_LIFECYCLE_EVENTS_TABLE} (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                from_status TEXT,
                to_status TEXT NOT NULL,
                reason TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{{}}',
                created_at_utc TEXT NOT NULL
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
                account_mode TEXT NOT NULL DEFAULT 'local_paper',
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
        _ensure_column(conn, SIGNALS_TABLE, "account_mode", "TEXT NOT NULL DEFAULT 'local_paper'")

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
        _ensure_column(conn, ORDERS_TABLE, "exchange_order_id", "TEXT")
        _ensure_column(conn, ORDERS_TABLE, "account_mode", "TEXT NOT NULL DEFAULT 'local_paper'")
        _ensure_column(conn, ORDERS_TABLE, "type", "TEXT")
        _ensure_column(conn, ORDERS_TABLE, "price_requested", "REAL")
        _ensure_column(conn, ORDERS_TABLE, "price_filled", "REAL")
        _ensure_column(conn, ORDERS_TABLE, "updated_at_utc", "TEXT")
        _ensure_column(conn, ORDERS_TABLE, "raw_exchange_response_json", "TEXT")

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {FILLS_TABLE} (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                exchange_trade_id TEXT,
                model_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT,
                account_mode TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                commission REAL NOT NULL DEFAULT 0,
                commission_asset TEXT,
                timestamp_utc TEXT NOT NULL,
                raw_exchange_response_json TEXT
            )
            """
        )

        if _table_exists(conn, POSITIONS_TABLE):
            cols = _table_columns(conn, POSITIONS_TABLE)
            if "model_id" not in cols or "account_mode" not in cols:
                legacy_name = f"{POSITIONS_TABLE}_legacy_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}"
                conn.execute(f"ALTER TABLE {POSITIONS_TABLE} RENAME TO {legacy_name}")
                conn.execute(
                    f"""
                    CREATE TABLE {POSITIONS_TABLE} (
                        model_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        account_mode TEXT NOT NULL,
                        quantity REAL NOT NULL DEFAULT 0,
                        avg_price REAL NOT NULL DEFAULT 0,
                        realized_pnl REAL NOT NULL DEFAULT 0,
                        unrealized_pnl REAL NOT NULL DEFAULT 0,
                        updated_at_utc TEXT NOT NULL,
                        dry_run INTEGER NOT NULL DEFAULT 1,
                        PRIMARY KEY (model_id, symbol, timeframe, account_mode)
                    )
                    """
                )
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {POSITIONS_TABLE} (
                        model_id, symbol, timeframe, account_mode, quantity, avg_price,
                        realized_pnl, unrealized_pnl, updated_at_utc, dry_run
                    )
                    SELECT 'legacy', symbol, timeframe,
                           CASE WHEN dry_run = 1 THEN 'local_paper' ELSE 'real' END,
                           quantity, avg_price, realized_pnl, unrealized_pnl, updated_at_utc, dry_run
                    FROM {legacy_name}
                    """
                )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {POSITIONS_TABLE} (
                model_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                account_mode TEXT NOT NULL,
                quantity REAL NOT NULL DEFAULT 0,
                avg_price REAL NOT NULL DEFAULT 0,
                realized_pnl REAL NOT NULL DEFAULT 0,
                unrealized_pnl REAL NOT NULL DEFAULT 0,
                updated_at_utc TEXT NOT NULL,
                dry_run INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (model_id, symbol, timeframe, account_mode)
            )
            """
        )

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {PORTFOLIO_SNAPSHOTS_TABLE} (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL DEFAULT 'legacy',
                account_mode TEXT NOT NULL DEFAULT 'local_paper',
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
        _ensure_column(conn, PORTFOLIO_SNAPSHOTS_TABLE, "model_id", "TEXT NOT NULL DEFAULT 'legacy'")
        _ensure_column(conn, PORTFOLIO_SNAPSHOTS_TABLE, "account_mode", "TEXT NOT NULL DEFAULT 'local_paper'")

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {PAPER_MODEL_METRICS_TABLE} (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                account_mode TEXT NOT NULL,
                timeframe TEXT,
                trades INTEGER NOT NULL DEFAULT 0,
                filled_trades INTEGER NOT NULL DEFAULT 0,
                win_rate REAL,
                realized_pnl REAL,
                unrealized_pnl REAL,
                total_pnl REAL,
                equity REAL,
                total_return REAL,
                max_drawdown REAL,
                profit_factor REAL,
                average_trade_return REAL,
                days_active REAL,
                last_signal_utc TEXT,
                last_order_utc TEXT,
                last_fill_utc TEXT,
                current_exposure REAL,
                validation_status TEXT NOT NULL,
                metrics_json TEXT NOT NULL DEFAULT '{{}}',
                evaluated_at_utc TEXT NOT NULL,
                UNIQUE(model_id, account_mode, evaluated_at_utc)
            )
            """
        )

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {BOT_EVENTS_TABLE} (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{{}}',
                created_at_utc TEXT NOT NULL
            )
            """
        )

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {BOT_STATUS_TABLE} (
                component TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                pid INTEGER,
                started_at_utc TEXT,
                last_heartbeat_utc TEXT NOT NULL,
                message TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{{}}'
            )
            """
        )

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RISK_EVENTS_TABLE} (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                symbol TEXT,
                account_mode TEXT,
                approved INTEGER NOT NULL,
                reason TEXT,
                details_json TEXT NOT NULL DEFAULT '{{}}',
                created_at_utc TEXT NOT NULL
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


def assert_required_schema() -> dict:
    init_research_tables()
    required_tables = [
        PRICES_TABLE,
        DATA_COVERAGE_TABLE,
        DATA_GAPS_TABLE,
        FEATURES_TABLE,
        MODEL_REGISTRY_TABLE,
        SIGNALS_TABLE,
        ORDERS_TABLE,
        FILLS_TABLE,
        POSITIONS_TABLE,
        PORTFOLIO_SNAPSHOTS_TABLE,
        PAPER_MODEL_METRICS_TABLE,
        MODEL_LIFECYCLE_EVENTS_TABLE,
        BOT_EVENTS_TABLE,
        BOT_STATUS_TABLE,
        RISK_EVENTS_TABLE,
    ]
    model_scoped = {
        SIGNALS_TABLE: "model_id",
        ORDERS_TABLE: "model_id",
        FILLS_TABLE: "model_id",
        POSITIONS_TABLE: "model_id",
        PORTFOLIO_SNAPSHOTS_TABLE: "model_id",
    }
    conn = get_connection()
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        missing_tables = [t for t in required_tables if t not in tables]
        missing_columns = {
            table: col
            for table, col in model_scoped.items()
            if table in tables and col not in _table_columns(conn, table)
        }
    finally:
        conn.close()
    return {
        "ok": not missing_tables and not missing_columns,
        "db_file": str(DB_FILE),
        "missing_tables": missing_tables,
        "missing_model_id_columns": missing_columns,
    }


def save_portfolio_snapshot(
    cash: float,
    equity: float,
    realized_pnl: float,
    unrealized_pnl: float,
    exposure_by_symbol: dict[str, float],
    dry_run: bool = True,
    model_id: str = "legacy",
    account_mode: str = "local_paper",
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            f"""
            INSERT INTO {PORTFOLIO_SNAPSHOTS_TABLE} (
                model_id, account_mode, datetime_utc, cash, equity, realized_pnl, unrealized_pnl, exposure_json, dry_run
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                account_mode,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="SQLite schema utilities for the trading platform.")
    parser.add_argument("--init", action="store_true", help="Create/migrate required SQLite tables.")
    parser.add_argument("--check-schema", action="store_true", help="Validate required tables and model_id columns.")
    args = parser.parse_args()

    if args.init or not args.check_schema:
        init_research_tables()
        print(json.dumps({"status": "initialized", "db_file": str(DB_FILE)}, ensure_ascii=True, indent=2))

    if args.check_schema:
        print(json.dumps(assert_required_schema(), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
