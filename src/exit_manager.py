from __future__ import annotations

import sqlite3

import pandas as pd

from config import DB_FILE, TRADES_TABLE
from db_utils import init_research_tables


def evaluate_virtual_exits(price_by_symbol: dict[str, float]) -> list[dict]:
    """Evaluate model-owned virtual TP/SL/horizon exits without model authority.

    This function only records required exits. Central execution remains responsible
    for sending close orders in a later loop iteration.
    """
    init_research_tables()
    now = pd.Timestamp.now(tz="UTC")
    actions: list[dict] = []
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute(
            f"""
            SELECT trade_id, model_id, symbol, avg_entry_price, tp_price, sl_price,
                   valid_until_utc, status
            FROM {TRADES_TABLE}
            WHERE status='OPEN'
            """
        ).fetchall()
        for trade_id, model_id, symbol, entry, tp, sl, valid_until, status in rows:
            price = price_by_symbol.get(symbol)
            if price is None:
                continue
            reason = None
            if tp is not None and float(price) >= float(tp):
                reason = "virtual_take_profit_reached"
            elif sl is not None and float(price) <= float(sl):
                reason = "virtual_stop_loss_reached"
            elif valid_until and pd.Timestamp(valid_until) <= now:
                reason = "signal_horizon_expired"
            if reason:
                conn.execute(
                    f"UPDATE {TRADES_TABLE} SET status='CLOSING', exit_reason=?, updated_at_utc=? WHERE trade_id=?",
                    (reason, now.isoformat(), trade_id),
                )
                actions.append({"trade_id": trade_id, "model_id": model_id, "symbol": symbol, "reason": reason, "price": float(price)})
        conn.commit()
    return actions
