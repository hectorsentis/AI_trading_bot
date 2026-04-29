import argparse
import json
import os
import sqlite3
from typing import Optional

import pandas as pd

from config import BOT_EVENTS_TABLE, BOT_STATUS_TABLE, DB_FILE
from db_utils import init_research_tables


def now_utc() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def update_status(
    component: str,
    status: str,
    pid: Optional[int] = None,
    message: Optional[str] = None,
    metadata: Optional[dict] = None,
    started_at_utc: Optional[str] = None,
) -> None:
    init_research_tables()
    ts = now_utc()
    conn = sqlite3.connect(DB_FILE)
    try:
        existing = conn.execute(
            f"SELECT started_at_utc FROM {BOT_STATUS_TABLE} WHERE component = ?",
            (component,),
        ).fetchone()
        started = started_at_utc or (existing[0] if existing and existing[0] else ts)
        conn.execute(
            f"""
            INSERT INTO {BOT_STATUS_TABLE} (
                component, status, pid, started_at_utc, last_heartbeat_utc, message, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(component) DO UPDATE SET
                status = excluded.status,
                pid = excluded.pid,
                started_at_utc = COALESCE({BOT_STATUS_TABLE}.started_at_utc, excluded.started_at_utc),
                last_heartbeat_utc = excluded.last_heartbeat_utc,
                message = excluded.message,
                metadata_json = excluded.metadata_json
            """,
            (component, status, pid, started, ts, message, json.dumps(metadata or {}, ensure_ascii=True, sort_keys=True)),
        )
        conn.commit()
    finally:
        conn.close()


def record_event(component: str, severity: str, message: str, metadata: Optional[dict] = None) -> None:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            f"""
            INSERT INTO {BOT_EVENTS_TABLE} (component, severity, message, metadata_json, created_at_utc)
            VALUES (?, ?, ?, ?, ?)
            """,
            (component, severity, message, json.dumps(metadata or {}, ensure_ascii=True, sort_keys=True), now_utc()),
        )
        conn.commit()
    finally:
        conn.close()


def load_status(stale_after_seconds: int = 180) -> list[dict]:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        rows = pd.read_sql_query(
            f"SELECT * FROM {BOT_STATUS_TABLE} ORDER BY component",
            conn,
        )
    finally:
        conn.close()
    if rows.empty:
        return []
    now = pd.Timestamp.now(tz="UTC")
    out = []
    for record in rows.to_dict(orient="records"):
        hb = pd.to_datetime(record.get("last_heartbeat_utc"), utc=True, errors="coerce")
        age = None if pd.isna(hb) else float((now - hb).total_seconds())
        effective = record.get("status") or "unknown"
        if age is None or age > stale_after_seconds:
            effective = "stale" if effective == "running" else effective
        record["heartbeat_age_seconds"] = age
        record["effective_status"] = effective
        try:
            record["metadata_json"] = json.loads(record.get("metadata_json") or "{}")
        except Exception:
            pass
        out.append(record)
    return out


def platform_running(stale_after_seconds: int = 180) -> bool:
    statuses = load_status(stale_after_seconds=stale_after_seconds)
    required = {"autonomous_runner", "realtime_ingestor", "trading_bot", "paper_model_evaluator", "dashboard"}
    by_name = {row["component"]: row for row in statuses}
    return all(by_name.get(name, {}).get("effective_status") == "running" for name in required)


def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime status/heartbeat utility.")
    parser.add_argument("--component", default=None)
    parser.add_argument("--status", default=None)
    parser.add_argument("--message", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.component and args.status:
        update_status(args.component, args.status, pid=os.getpid(), message=args.message)
    if args.show or not (args.component and args.status):
        print(json.dumps({"platform_running": platform_running(), "services": load_status()}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
