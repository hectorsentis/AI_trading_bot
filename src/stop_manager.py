from __future__ import annotations

import sqlite3

from config import DB_FILE, TRADES_TABLE
from db_utils import init_research_tables


def list_emergency_stops(account_mode: str | None = None) -> list[dict]:
    init_research_tables()
    where = "WHERE emergency_sl_price IS NOT NULL"
    params: tuple = ()
    if account_mode:
        where += " AND account_mode=?"
        params = (account_mode,)
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute(
            f"""
            SELECT trade_id, model_id, symbol, qty, emergency_sl_price, status, account_mode
            FROM {TRADES_TABLE} {where}
            ORDER BY updated_at_utc DESC
            """,
            params,
        ).fetchall()
    return [
        {
            "trade_id": r[0], "model_id": r[1], "symbol": r[2], "quantity": r[3],
            "emergency_sl_price": r[4], "status": r[5], "account_mode": r[6],
        }
        for r in rows
    ]
