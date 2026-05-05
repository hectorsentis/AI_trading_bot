from __future__ import annotations

import json
import sqlite3

import pandas as pd

from config import DB_FILE, MODEL_PERFORMANCE_TABLE, TRADES_TABLE
from db_utils import init_research_tables
from trade_builder import BuiltTrade


def create_trade_record(built_trade: BuiltTrade, *, account_mode: str, status: str = "APPROVED") -> None:
    init_research_tables()
    t = built_trade.to_dict()
    now = pd.Timestamp.now(tz="UTC").isoformat()
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {TRADES_TABLE} (
                trade_id, proposal_id, allocation_id, prediction_id, model_id, symbol, timeframe,
                side, status, account_mode, requested_notional_usdt, approved_notional_usdt,
                qty, entry_price, avg_entry_price, tp_price, sl_price, emergency_sl_price,
                horizon_bars, valid_until_utc, realized_pnl_usdt, unrealized_pnl_usdt,
                fees_usdt, slippage_usdt, raw_json, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, ?, ?, ?)
            """,
            (
                t["trade_id"], t["proposal_id"], t["allocation_id"], t["prediction_id"], t["model_id"], t["symbol"], t["timeframe"],
                t["side"], status, account_mode, t["approved_notional_usdt"], t["approved_notional_usdt"], t["quantity"],
                t["entry_reference_price"], None, t["tp_price"], t["sl_price"], t["emergency_sl_price"], t["horizon_bars"],
                t["valid_until_utc"], json.dumps(t["raw_json"], ensure_ascii=True, sort_keys=True), now, now,
            ),
        )
        conn.commit()


def mark_trade_open(
    *,
    trade_id: str,
    avg_entry_price: float,
    qty: float,
    fees_usdt: float = 0.0,
    slippage_usdt: float = 0.0,
    raw_update: dict | None = None,
) -> None:
    now = pd.Timestamp.now(tz="UTC").isoformat()
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            f"""
            UPDATE {TRADES_TABLE}
            SET status='OPEN', avg_entry_price=?, entry_price=COALESCE(entry_price, ?), qty=?,
                opened_at_utc=COALESCE(opened_at_utc, ?), fees_usdt=COALESCE(fees_usdt,0)+?,
                slippage_usdt=COALESCE(slippage_usdt,0)+?, updated_at_utc=?, raw_json=?
            WHERE trade_id=?
            """,
            (
                float(avg_entry_price), float(avg_entry_price), float(qty), now, float(fees_usdt), float(slippage_usdt), now,
                json.dumps(raw_update or {}, ensure_ascii=True, sort_keys=True), trade_id,
            ),
        )
        conn.commit()


def mark_trade_failed(trade_id: str, reason: str, raw: dict | None = None) -> None:
    now = pd.Timestamp.now(tz="UTC").isoformat()
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            f"UPDATE {TRADES_TABLE} SET status='FAILED', exit_reason=?, raw_json=?, updated_at_utc=? WHERE trade_id=?",
            (reason, json.dumps(raw or {}, ensure_ascii=True, sort_keys=True), now, trade_id),
        )
        conn.commit()


def refresh_model_performance(account_mode: str | None = None) -> None:
    """Materialize model/trade attribution metrics for dashboard and audits."""
    init_research_tables()
    now = pd.Timestamp.now(tz="UTC").isoformat()
    with sqlite3.connect(DB_FILE) as conn:
        where = "WHERE account_mode=?" if account_mode else ""
        params = (account_mode,) if account_mode else ()
        rows = conn.execute(
            f"""
            SELECT model_id, symbol, timeframe, account_mode,
                   SUM(CASE WHEN status IN ('OPEN','REDUCING','CLOSING') THEN 1 ELSE 0 END) AS open_trades,
                   SUM(CASE WHEN status IN ('CLOSED','FORCE_CLOSED') THEN 1 ELSE 0 END) AS closed_trades,
                   SUM(COALESCE(realized_pnl_usdt,0)) AS realized_pnl,
                   SUM(COALESCE(unrealized_pnl_usdt,0)) AS unrealized_pnl,
                   SUM(CASE WHEN COALESCE(realized_pnl_usdt,0) > 0 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN status IN ('CLOSED','FORCE_CLOSED') THEN 1 ELSE 0 END) AS closed_n
            FROM {TRADES_TABLE}
            {where}
            GROUP BY model_id, symbol, timeframe, account_mode
            """,
            params,
        ).fetchall()
        for r in rows:
            closed_n = int(r[9] or 0)
            win_rate = (float(r[8] or 0) / closed_n) if closed_n else None
            conn.execute(
                f"""
                INSERT INTO {MODEL_PERFORMANCE_TABLE} (
                    model_id, symbol, timeframe, account_mode, timestamp_utc, open_trades,
                    closed_trades, realized_pnl_usdt, unrealized_pnl_usdt, win_rate,
                    metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r[0], r[1], r[2], r[3], now, int(r[4] or 0), int(r[5] or 0),
                    float(r[6] or 0), float(r[7] or 0), win_rate,
                    json.dumps({"source": "ledger.refresh_model_performance"}, ensure_ascii=True),
                ),
            )
        conn.commit()
