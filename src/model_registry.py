import argparse
import json
import sqlite3
from typing import Optional

import pandas as pd

from config import DB_FILE, MODEL_REGISTRY_TABLE
from db_utils import init_research_tables


def _json_dumps(payload) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _row_to_model_dict(row: pd.Series) -> dict:
    record = row.to_dict()
    for key in ["metrics_json", "params_json", "rejection_reasons_json"]:
        if key in record and isinstance(record[key], str):
            try:
                record[key] = json.loads(record[key])
            except Exception:
                pass
    return record


def register_model(
    model_id: str,
    symbol_scope: str,
    timeframe: str,
    train_start: Optional[str],
    train_end: Optional[str],
    test_start: Optional[str],
    test_end: Optional[str],
    feature_version: str,
    label_version: str,
    model_path: str,
    metrics: dict,
    params: dict,
    status: str = "candidate",
    acceptance_status: str = "candidate",
    rejection_reasons: Optional[list[str]] = None,
    evaluation_scope: str = "holdout",
    is_active: bool = False,
) -> None:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {MODEL_REGISTRY_TABLE} (
                model_id,
                symbol_scope,
                timeframe,
                train_start,
                train_end,
                test_start,
                test_end,
                feature_version,
                label_version,
                model_path,
                training_ts_utc,
                metrics_json,
                params_json,
                status,
                acceptance_status,
                rejection_reasons_json,
                evaluation_scope,
                is_active,
                updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                symbol_scope,
                timeframe,
                train_start,
                train_end,
                test_start,
                test_end,
                feature_version,
                label_version,
                model_path,
                pd.Timestamp.now(tz="UTC").isoformat(),
                _json_dumps(metrics),
                _json_dumps(params),
                status,
                acceptance_status,
                _json_dumps(rejection_reasons or []),
                evaluation_scope,
                1 if is_active else 0,
                pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_model_by_id(model_id: str) -> Optional[dict]:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM {MODEL_REGISTRY_TABLE}
            WHERE model_id = ?
            LIMIT 1
            """,
            conn,
            params=(model_id,),
        )
    finally:
        conn.close()

    if df.empty:
        return None
    return _row_to_model_dict(df.iloc[0])


def update_model_evaluation(
    model_id: str,
    metrics: Optional[dict] = None,
    params: Optional[dict] = None,
    status: Optional[str] = None,
    acceptance_status: Optional[str] = None,
    rejection_reasons: Optional[list[str]] = None,
    evaluation_scope: Optional[str] = None,
) -> None:
    init_research_tables()
    existing = get_model_by_id(model_id)
    if not existing:
        raise ValueError(f"Model not found in registry: {model_id}")

    merged_metrics = existing.get("metrics_json", {}) if isinstance(existing.get("metrics_json"), dict) else {}
    if metrics:
        merged_metrics.update(metrics)

    merged_params = existing.get("params_json", {}) if isinstance(existing.get("params_json"), dict) else {}
    if params:
        merged_params.update(params)

    final_status = status or existing.get("status", "candidate")
    final_acceptance_status = acceptance_status or existing.get("acceptance_status", "candidate")
    final_rejections = rejection_reasons if rejection_reasons is not None else existing.get("rejection_reasons_json", [])
    final_scope = evaluation_scope or existing.get("evaluation_scope", "holdout")

    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            f"""
            UPDATE {MODEL_REGISTRY_TABLE}
            SET metrics_json = ?,
                params_json = ?,
                status = ?,
                acceptance_status = ?,
                rejection_reasons_json = ?,
                evaluation_scope = ?,
                updated_at_utc = ?
            WHERE model_id = ?
            """,
            (
                _json_dumps(merged_metrics),
                _json_dumps(merged_params),
                final_status,
                final_acceptance_status,
                _json_dumps(final_rejections),
                final_scope,
                pd.Timestamp.now(tz="UTC").isoformat(),
                model_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def set_active_model(model_id: str, timeframe: Optional[str] = None) -> None:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        if timeframe:
            conn.execute(f"UPDATE {MODEL_REGISTRY_TABLE} SET is_active = 0 WHERE timeframe = ?", (timeframe,))
        else:
            conn.execute(f"UPDATE {MODEL_REGISTRY_TABLE} SET is_active = 0")
        conn.execute(
            f"""
            UPDATE {MODEL_REGISTRY_TABLE}
            SET is_active = 1,
                status = CASE WHEN acceptance_status = 'accepted' THEN 'active' ELSE status END,
                updated_at_utc = ?
            WHERE model_id = ?
            """,
            (pd.Timestamp.now(tz="UTC").isoformat(), model_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_latest_model(
    timeframe: Optional[str] = None,
    statuses: Optional[list[str]] = None,
    acceptance_statuses: Optional[list[str]] = None,
    prefer_active: bool = False,
) -> Optional[dict]:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        where_clauses = []
        params: list[object] = []

        if timeframe:
            where_clauses.append("timeframe = ?")
            params.append(timeframe)

        if statuses:
            placeholders = ", ".join(["?"] * len(statuses))
            where_clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)

        if acceptance_statuses:
            placeholders = ", ".join(["?"] * len(acceptance_statuses))
            where_clauses.append(f"acceptance_status IN ({placeholders})")
            params.extend(acceptance_statuses)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        order_sql = "is_active DESC, training_ts_utc DESC" if prefer_active else "training_ts_utc DESC"

        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM {MODEL_REGISTRY_TABLE}
            WHERE {where_sql}
            ORDER BY {order_sql}
            LIMIT 1
            """,
            conn,
            params=tuple(params),
        )
    finally:
        conn.close()

    if df.empty:
        return None
    return _row_to_model_dict(df.iloc[0])


def select_model_for_inference(
    timeframe: str,
    acceptance_order: list[str],
    prefer_active: bool = True,
) -> Optional[dict]:
    for acceptance_status in acceptance_order:
        model = get_latest_model(
            timeframe=timeframe,
            acceptance_statuses=[acceptance_status],
            prefer_active=prefer_active,
        )
        if model:
            return model
    return None


def list_accepted_models(timeframe: str, limit: int | None = None) -> list[dict]:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        limit_sql = "LIMIT ?" if limit is not None else ""
        params: list[object] = [timeframe]
        if limit is not None:
            params.append(int(limit))
        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM {MODEL_REGISTRY_TABLE}
            WHERE timeframe = ?
              AND acceptance_status = 'accepted'
              AND model_path IS NOT NULL
            ORDER BY is_active DESC, training_ts_utc DESC
            {limit_sql}
            """,
            conn,
            params=tuple(params),
        )
    finally:
        conn.close()
    return [_row_to_model_dict(row) for _, row in df.iterrows()]


def count_accepted_models(timeframe: str) -> int:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        row = pd.read_sql_query(
            f"""
            SELECT COUNT(*) AS n
            FROM {MODEL_REGISTRY_TABLE}
            WHERE timeframe = ? AND acceptance_status = 'accepted'
            """,
            conn,
            params=(timeframe,),
        )
    finally:
        conn.close()
    return int(row.loc[0, "n"]) if not row.empty else 0


def list_models(limit: int = 20) -> pd.DataFrame:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT
                model_id,
                symbol_scope,
                timeframe,
                train_start,
                train_end,
                test_start,
                test_end,
                training_ts_utc,
                status,
                acceptance_status,
                evaluation_scope,
                is_active
            FROM {MODEL_REGISTRY_TABLE}
            ORDER BY training_ts_utc DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
    finally:
        conn.close()
    return df


def main():
    parser = argparse.ArgumentParser(description="Inspect model registry.")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    df = list_models(limit=args.limit)
    if df.empty:
        print("No models found in registry.")
        return

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
