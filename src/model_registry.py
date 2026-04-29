import argparse
import json
import sqlite3
from typing import Optional

import pandas as pd

from config import (
    DB_FILE,
    MODEL_LIFECYCLE_EVENTS_TABLE,
    MODEL_LIFECYCLE_STATUSES,
    MODEL_REGISTRY_TABLE,
    ACCOUNT_MODE_TESTNET_PAPER,
)
from db_utils import init_research_tables


VALID_STATUSES = set(MODEL_LIFECYCLE_STATUSES)


def _json_dumps(payload) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _json_loads(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _row_to_model_dict(row: pd.Series) -> dict:
    record = row.to_dict()
    for key in ["metrics_json", "params_json", "rejection_reasons_json"]:
        if key in record and isinstance(record[key], str):
            record[key] = _json_loads(record[key], {} if key != "rejection_reasons_json" else [])
    return record


def _normalize_status(status: str, acceptance_status: str | None = None) -> str:
    status = (status or "candidate").strip()
    if status in VALID_STATUSES:
        return status
    # Backward compatibility with legacy accepted/rejected registry values.
    if status == "accepted" or acceptance_status == "accepted":
        return "validation_accepted"
    if status == "rejected" or acceptance_status == "rejected":
        return "validation_rejected"
    return "candidate"


def add_lifecycle_event(
    model_id: str,
    from_status: Optional[str],
    to_status: str,
    reason: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            f"""
            INSERT INTO {MODEL_LIFECYCLE_EVENTS_TABLE} (
                model_id, from_status, to_status, reason, metadata_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                from_status,
                to_status,
                reason,
                _json_dumps(metadata or {}),
                pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


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
    normalized_status = _normalize_status(status, acceptance_status)
    now = pd.Timestamp.now(tz="UTC").isoformat()
    conn = sqlite3.connect(DB_FILE)
    try:
        exists = conn.execute(
            f"SELECT 1 FROM {MODEL_REGISTRY_TABLE} WHERE model_id = ? LIMIT 1", (model_id,)
        ).fetchone()
        if exists:
            raise sqlite3.IntegrityError(f"model_id already exists; refusing to overwrite model history: {model_id}")
        conn.execute(
            f"""
            INSERT INTO {MODEL_REGISTRY_TABLE} (
                model_id, symbol_scope, timeframe, train_start, train_end, test_start, test_end,
                feature_version, label_version, model_path, training_ts_utc, metrics_json,
                params_json, status, acceptance_status, rejection_reasons_json,
                evaluation_scope, is_active, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                now,
                _json_dumps(metrics),
                _json_dumps(params),
                normalized_status,
                acceptance_status,
                _json_dumps(rejection_reasons or []),
                evaluation_scope,
                1 if is_active else 0,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    add_lifecycle_event(model_id, None, normalized_status, reason="registered", metadata={"evaluation_scope": evaluation_scope})


def get_model_by_id(model_id: str) -> Optional[dict]:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM {MODEL_REGISTRY_TABLE} WHERE model_id = ? LIMIT 1",
            conn,
            params=(model_id,),
        )
    finally:
        conn.close()
    if df.empty:
        return None
    return _row_to_model_dict(df.iloc[0])


def update_model_status(
    model_id: str,
    status: str,
    reason: Optional[str] = None,
    metadata: Optional[dict] = None,
    acceptance_status: Optional[str] = None,
    rejection_reasons: Optional[list[str]] = None,
    is_active: Optional[bool] = None,
    account_mode: Optional[str] = None,
) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid lifecycle status: {status}")
    init_research_tables()
    existing = get_model_by_id(model_id)
    if not existing:
        raise ValueError(f"Model not found in registry: {model_id}")
    from_status = str(existing.get("status") or "candidate")
    now = pd.Timestamp.now(tz="UTC").isoformat()

    assignments = ["status = ?", "updated_at_utc = ?"]
    params: list[object] = [status, now]
    if acceptance_status is not None:
        assignments.append("acceptance_status = ?")
        params.append(acceptance_status)
    if rejection_reasons is not None:
        assignments.append("rejection_reasons_json = ?")
        params.append(_json_dumps(rejection_reasons))
    if is_active is not None:
        assignments.append("is_active = ?")
        params.append(1 if is_active else 0)
    if account_mode is not None:
        assignments.append("account_mode = ?")
        params.append(account_mode)
    timestamp_columns = {
        "paper_active": "paper_started_at_utc",
        "paper_validated": "paper_validated_at_utc",
        "real_ready": "real_ready_at_utc",
        "real_active": "real_active_at_utc",
    }
    if status in timestamp_columns:
        assignments.append(f"{timestamp_columns[status]} = ?")
        params.append(now)
    params.append(model_id)

    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            f"UPDATE {MODEL_REGISTRY_TABLE} SET {', '.join(assignments)} WHERE model_id = ?",
            tuple(params),
        )
        conn.commit()
    finally:
        conn.close()
    add_lifecycle_event(model_id, from_status, status, reason=reason, metadata=metadata)


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

    final_status = _normalize_status(status or existing.get("status", "candidate"), acceptance_status)
    final_acceptance_status = acceptance_status or existing.get("acceptance_status", "candidate")
    final_rejections = rejection_reasons if rejection_reasons is not None else existing.get("rejection_reasons_json", [])
    final_scope = evaluation_scope or existing.get("evaluation_scope", "holdout")

    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            f"""
            UPDATE {MODEL_REGISTRY_TABLE}
            SET metrics_json = ?, params_json = ?, status = ?, acceptance_status = ?,
                rejection_reasons_json = ?, evaluation_scope = ?, updated_at_utc = ?
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
    if final_status != existing.get("status"):
        add_lifecycle_event(model_id, existing.get("status"), final_status, reason="evaluation_update", metadata={"scope": final_scope})


def list_models_by_status(statuses: str | list[str], timeframe: Optional[str] = None, limit: Optional[int] = None) -> list[dict]:
    init_research_tables()
    if isinstance(statuses, str):
        statuses = [statuses]
    placeholders = ",".join(["?"] * len(statuses))
    where = [f"status IN ({placeholders})"]
    params: list[object] = list(statuses)
    if timeframe:
        where.append("timeframe = ?")
        params.append(timeframe)
    limit_sql = "LIMIT ?" if limit is not None else ""
    if limit is not None:
        params.append(int(limit))
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM {MODEL_REGISTRY_TABLE} WHERE {' AND '.join(where)} ORDER BY training_ts_utc ASC {limit_sql}",
            conn,
            params=tuple(params),
        )
    finally:
        conn.close()
    return [_row_to_model_dict(row) for _, row in df.iterrows()]


def list_paper_active_models(timeframe: Optional[str] = None) -> list[dict]:
    return list_models_by_status("paper_active", timeframe=timeframe)


def list_backtest_accepted_models(timeframe: Optional[str] = None) -> list[dict]:
    return list_models_by_status("backtest_accepted", timeframe=timeframe)


def activate_model_for_paper(model_id: str, account_mode: str = ACCOUNT_MODE_TESTNET_PAPER) -> None:
    update_model_status(model_id, "paper_active", reason="activated_for_paper", acceptance_status="accepted", is_active=True, account_mode=account_mode)


def mark_model_paper_rejected(model_id: str, reason: str, metrics: Optional[dict] = None) -> None:
    existing = get_model_by_id(model_id) or {}
    rejections = existing.get("rejection_reasons_json", []) if isinstance(existing.get("rejection_reasons_json"), list) else []
    update_model_status(model_id, "paper_rejected", reason=reason, metadata=metrics or {}, rejection_reasons=rejections + [reason], is_active=False)


def mark_model_paper_validated(model_id: str, metrics: Optional[dict] = None) -> None:
    update_model_status(model_id, "paper_validated", reason="paper_criteria_passed", metadata=metrics or {})


def mark_model_real_ready(model_id: str, metrics: Optional[dict] = None) -> None:
    update_model_status(model_id, "real_ready", reason="paper_validated_real_ready", metadata=metrics or {})


def mark_model_real_active(model_id: str, reason: str = "explicit_real_activation") -> None:
    update_model_status(model_id, "real_active", reason=reason, is_active=True)


def pause_real_model(model_id: str, reason: str = "manual_or_safety_pause") -> None:
    update_model_status(model_id, "real_paused", reason=reason, is_active=False)


def archive_model(model_id: str, reason: str = "archived") -> None:
    update_model_status(model_id, "archived", reason=reason, is_active=False)


def set_active_model(model_id: str, timeframe: Optional[str] = None) -> None:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        if timeframe:
            conn.execute(f"UPDATE {MODEL_REGISTRY_TABLE} SET is_active = 0 WHERE timeframe = ?", (timeframe,))
        else:
            conn.execute(f"UPDATE {MODEL_REGISTRY_TABLE} SET is_active = 0")
        conn.execute(
            f"UPDATE {MODEL_REGISTRY_TABLE} SET is_active = 1, updated_at_utc = ? WHERE model_id = ?",
            (pd.Timestamp.now(tz="UTC").isoformat(), model_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_latest_model(timeframe: Optional[str] = None, statuses: Optional[list[str]] = None, acceptance_statuses: Optional[list[str]] = None, prefer_active: bool = False) -> Optional[dict]:
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
            f"SELECT * FROM {MODEL_REGISTRY_TABLE} WHERE {where_sql} ORDER BY {order_sql} LIMIT 1",
            conn,
            params=tuple(params),
        )
    finally:
        conn.close()
    if df.empty:
        return None
    return _row_to_model_dict(df.iloc[0])


def select_model_for_inference(timeframe: str, acceptance_order: list[str], prefer_active: bool = True) -> Optional[dict]:
    status_priority = ["paper_active", "paper_validated", "real_ready", "backtest_accepted", "validation_accepted"]
    for status in status_priority:
        model = get_latest_model(timeframe=timeframe, statuses=[status], prefer_active=prefer_active)
        if model:
            return model
    for acceptance_status in acceptance_order:
        model = get_latest_model(timeframe=timeframe, acceptance_statuses=[acceptance_status], prefer_active=prefer_active)
        if model:
            return model
    return None


def list_accepted_models(timeframe: str, limit: int | None = None) -> list[dict]:
    statuses = ["paper_active", "paper_validated", "real_ready", "backtest_accepted", "validation_accepted"]
    return list_models_by_status(statuses, timeframe=timeframe, limit=limit)


def count_accepted_models(timeframe: str) -> int:
    return len(list_accepted_models(timeframe=timeframe))


def count_paper_active_models(timeframe: Optional[str] = None) -> int:
    return len(list_paper_active_models(timeframe=timeframe))


def list_models(limit: int = 20) -> pd.DataFrame:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT model_id, symbol_scope, timeframe, train_start, train_end, test_start, test_end,
                   training_ts_utc, status, acceptance_status, evaluation_scope, is_active, account_mode
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
    parser.add_argument("--status", type=str, default=None)
    args = parser.parse_args()
    if args.status:
        print(json.dumps(list_models_by_status(args.status, limit=args.limit), ensure_ascii=True, indent=2))
        return
    df = list_models(limit=args.limit)
    if df.empty:
        print("No models found in registry.")
        return
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
