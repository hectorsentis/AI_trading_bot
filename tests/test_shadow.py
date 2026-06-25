"""Phase D: shadow-trade outcome resolution + analytics."""
from __future__ import annotations

import sqlite3

import pandas as pd

import config
from db_utils import init_research_tables
from shadow_evaluator import resolve_shadow_outcome, evaluate_open_shadow_trades, shadow_analytics


def test_resolve_tp_before_sl():
    ret, reason = resolve_shadow_outcome(100.0, 105.0, 95.0, [(101, 99, 100), (106, 99, 104)])
    assert reason == "tp"
    assert abs(ret - 0.05) < 1e-9


def test_resolve_sl():
    ret, reason = resolve_shadow_outcome(100.0, 110.0, 95.0, [(101, 94, 96)])
    assert reason == "sl"
    assert abs(ret - (-0.05)) < 1e-9


def test_resolve_expire_and_no_data():
    ret, reason = resolve_shadow_outcome(100.0, 110.0, 90.0, [(101, 99, 102)])
    assert reason == "expire" and abs(ret - 0.02) < 1e-9
    assert resolve_shadow_outcome(100.0, 1, 1, []) == (0.0, "no_data")


def test_evaluate_open_shadow_trades_books_outcome():
    init_research_tables()
    now = pd.Timestamp.now(tz="UTC")
    opened = (now - pd.Timedelta(hours=5)).isoformat()
    valid_until = (now - pd.Timedelta(hours=1)).isoformat()
    with sqlite3.connect(config.DB_FILE) as conn:
        # Price bars in the resolution window that hit TP (high >= 105).
        for i in range(1, 4):
            ts = (now - pd.Timedelta(hours=5) + pd.Timedelta(hours=i)).isoformat()
            high = 106.0 if i == 3 else 101.0
            conn.execute(
                f"INSERT OR REPLACE INTO {config.PRICES_TABLE} (symbol, timeframe, datetime_utc, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?)",
                ("SHADOWUSDT", "1h", ts, 100.0, high, 99.0, 100.0 + i, 10.0),
            )
        conn.execute(
            f"""INSERT OR REPLACE INTO {config.SHADOW_TRADES_TABLE}
                (shadow_trade_id, proposal_id, model_id, symbol, timeframe, side, status,
                 entry_reference_price, tp_price, sl_price, horizon_bars, valid_until_utc,
                 requested_notional_usdt, reason, opened_at_utc, created_at_utc)
                VALUES (?,?,?,?,?,'LONG','SHADOW_OPEN',?,?,?,?,?,?,?,?,?)""",
            ("sh1", "p1", "mX", "SHADOWUSDT", "1h", 100.0, 105.0, 95.0, 4, valid_until, 1000.0, "allocator_rejected", opened, opened),
        )
        conn.commit()

    closed = evaluate_open_shadow_trades()
    assert closed >= 1

    with sqlite3.connect(config.DB_FILE) as conn:
        row = conn.execute(
            f"SELECT status, outcome_pnl_usdt FROM {config.SHADOW_TRADES_TABLE} WHERE shadow_trade_id='sh1'"
        ).fetchone()
    assert row[0] == "SHADOW_CLOSED"
    assert abs(float(row[1]) - 50.0) < 1e-6  # +5% of 1000 notional

    analytics = shadow_analytics()
    assert analytics["closed"] >= 1
    assert analytics["would_have_won"] >= 1
