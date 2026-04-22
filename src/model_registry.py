import argparse
import json
import sqlite3
from typing import Optional

import pandas as pd

from config import DB_FILE, MODEL_REGISTRY_TABLE
from db_utils import init_research_tables


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
    status: str = "trained",
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
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(metrics, ensure_ascii=True, sort_keys=True),
                json.dumps(params, ensure_ascii=True, sort_keys=True),
                status,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_latest_model(timeframe: Optional[str] = None, status: str = "trained") -> Optional[dict]:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        where_clauses = ["status = ?"]
        params = [status]
        if timeframe:
            where_clauses.append("timeframe = ?")
            params.append(timeframe)

        where_sql = " AND ".join(where_clauses)
        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM {MODEL_REGISTRY_TABLE}
            WHERE {where_sql}
            ORDER BY training_ts_utc DESC
            LIMIT 1
            """,
            conn,
            params=tuple(params),
        )
    finally:
        conn.close()

    if df.empty:
        return None
    return df.iloc[0].to_dict()


def list_models(limit: int = 20) -> pd.DataFrame:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT model_id, symbol_scope, timeframe, train_start, train_end, test_start, test_end, training_ts_utc, status
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
