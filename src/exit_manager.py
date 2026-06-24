from __future__ import annotations

import sqlite3

import pandas as pd

from config import DB_FILE, TRADES_TABLE
from db_utils import init_research_tables


def evaluate_virtual_exits(
    price_by_symbol: dict[str, float],
    account_mode: str | None = None,
) -> list[dict]:
    """Evaluate model-owned virtual TP/SL/emergency/horizon exits without model authority.

    Marks each triggered trade as ``CLOSING`` and returns action rows carrying the full
    attribution chain so the central execution path can flatten them and book realized PnL.
    Central execution remains the only module allowed to place close orders; a model never
    closes its own (or another model's) trade directly.
    """
    init_research_tables()
    now = pd.Timestamp.now(tz="UTC")
    actions: list[dict] = []
    where = "WHERE status='OPEN'"
    params: list = []
    if account_mode:
        where += " AND account_mode=?"
        params.append(account_mode)
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute(
            f"""
            SELECT trade_id, model_id, prediction_id, proposal_id, allocation_id, symbol, timeframe,
                   qty, avg_entry_price, tp_price, sl_price, emergency_sl_price, valid_until_utc, account_mode
            FROM {TRADES_TABLE}
            {where}
            """,
            params,
        ).fetchall()
        for (trade_id, model_id, prediction_id, proposal_id, allocation_id, symbol, timeframe,
             qty, avg_entry, tp, sl, emergency_sl, valid_until, acct) in rows:
            price = price_by_symbol.get(symbol)
            if price is None:
                continue
            price = float(price)
            reason = None
            if tp is not None and price >= float(tp):
                reason = "virtual_take_profit_reached"
            elif emergency_sl is not None and price <= float(emergency_sl):
                reason = "emergency_stop_reached"
            elif sl is not None and price <= float(sl):
                reason = "virtual_stop_loss_reached"
            elif valid_until and pd.Timestamp(valid_until) <= now:
                reason = "signal_horizon_expired"
            if reason:
                conn.execute(
                    f"UPDATE {TRADES_TABLE} SET status='CLOSING', exit_reason=?, updated_at_utc=? WHERE trade_id=?",
                    (reason, now.isoformat(), trade_id),
                )
                actions.append(
                    {
                        "trade_id": trade_id,
                        "model_id": model_id,
                        "prediction_id": prediction_id,
                        "proposal_id": proposal_id,
                        "allocation_id": allocation_id,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "qty": float(qty or 0.0),
                        "avg_entry_price": float(avg_entry or 0.0),
                        "reason": reason,
                        "price": price,
                        "account_mode": acct,
                    }
                )
        conn.commit()
    return actions
