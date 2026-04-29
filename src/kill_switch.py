import json
import sqlite3

import pandas as pd

from config import DB_FILE, KILL_SWITCH_ENABLED, RISK_EVENTS_TABLE, BOT_EVENTS_TABLE, ACCOUNT_MODE_REAL
from db_utils import init_research_tables


class KillSwitch:
    """Central kill switch guard used by paper and live engines."""

    def __init__(self, enabled: bool = KILL_SWITCH_ENABLED):
        init_research_tables()
        self.enabled = bool(enabled)

    def check(self, model_id: str | None = None, account_mode: str | None = None) -> dict:
        if not self.enabled:
            return {"ok": True, "reasons": ["kill_switch_disabled"]}
        reasons: list[str] = []
        conn = sqlite3.connect(DB_FILE)
        try:
            # Any recent explicit emergency bot event blocks execution.
            rows = pd.read_sql_query(
                f"""
                SELECT message, metadata_json
                FROM {BOT_EVENTS_TABLE}
                WHERE severity IN ('critical', 'emergency')
                ORDER BY created_at_utc DESC
                LIMIT 5
                """,
                conn,
            )
        finally:
            conn.close()
        if not rows.empty:
            reasons.append("critical_bot_event_present")
        return {"ok": len(reasons) == 0, "reasons": reasons}

    def record_event(self, component: str, severity: str, message: str, metadata: dict | None = None) -> None:
        conn = sqlite3.connect(DB_FILE)
        try:
            conn.execute(
                f"""
                INSERT INTO {BOT_EVENTS_TABLE} (component, severity, message, metadata_json, created_at_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (component, severity, message, json.dumps(metadata or {}, ensure_ascii=True, sort_keys=True), pd.Timestamp.now(tz="UTC").isoformat()),
            )
            conn.commit()
        finally:
            conn.close()


def assert_real_trading_not_default() -> dict:
    from config import DRY_RUN, ENABLE_LIVE_TRADING, ENABLE_REAL_ORDER_EXECUTION, ENABLE_REAL_BINANCE_ACCOUNT

    ok = DRY_RUN and not ENABLE_LIVE_TRADING and not ENABLE_REAL_ORDER_EXECUTION and not ENABLE_REAL_BINANCE_ACCOUNT
    return {
        "ok": ok,
        "DRY_RUN": DRY_RUN,
        "ENABLE_LIVE_TRADING": ENABLE_LIVE_TRADING,
        "ENABLE_REAL_ORDER_EXECUTION": ENABLE_REAL_ORDER_EXECUTION,
        "ENABLE_REAL_BINANCE_ACCOUNT": ENABLE_REAL_BINANCE_ACCOUNT,
        "real_orders_blocked_by_default": ok,
    }
