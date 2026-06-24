"""Phase A regression guard for the trustworthy paper loop.

Asserts the previously-broken behavior is fixed:
  * an OPEN trade transitions OPEN -> CLOSING -> CLOSED when its virtual TP triggers,
  * realized PnL is booked on close (was always 0 before),
  * a closing order + fill are written with full attribution,
  * portfolio equity persists across PortfolioManager instances (carries over runs).

Uses locally-simulated paper only (no network, no broker).
"""
from __future__ import annotations

import sqlite3

import pandas as pd

import config
from db_utils import init_research_tables
from portfolio_manager import PortfolioManager
from execution_engine import ExecutionEngine
from exit_manager import evaluate_virtual_exits
from ledger import refresh_model_performance, update_open_trade_unrealized

ACCOUNT_MODE = config.ACCOUNT_MODE_LOCAL_PAPER
SYMBOL = "BTCUSDT"
ENTRY = 100_000.0
QTY = 0.01


def _insert_open_trade(trade_id: str, *, tp_price: float, sl_price: float) -> None:
    now = pd.Timestamp.now(tz="UTC").isoformat()
    with sqlite3.connect(config.DB_FILE) as conn:
        conn.execute(
            f"""
            INSERT INTO {config.TRADES_TABLE} (
                trade_id, proposal_id, allocation_id, prediction_id, model_id, symbol, timeframe,
                side, status, account_mode, qty, entry_price, avg_entry_price, tp_price, sl_price,
                horizon_bars, valid_until_utc, opened_at_utc, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'LONG', 'OPEN', ?, ?, ?, ?, ?, ?, 6, ?, ?, ?, ?)
            """,
            (
                trade_id, "prop_1", "alloc_1", "pred_1", "model_test", SYMBOL, "1h",
                ACCOUNT_MODE, QTY, ENTRY, ENTRY, tp_price, sl_price,
                now, now, now, now,
            ),
        )
        conn.commit()


def _trade_status(trade_id: str):
    with sqlite3.connect(config.DB_FILE) as conn:
        return conn.execute(
            f"SELECT status, realized_pnl_usdt, exit_price, exit_reason FROM {config.TRADES_TABLE} WHERE trade_id=?",
            (trade_id,),
        ).fetchone()


def test_open_trade_closes_books_pnl_and_persists_equity():
    init_research_tables()

    pm = PortfolioManager(
        timeframe="1h", dry_run=True, initial_cash=10_000.0,
        model_id="shared_capital_pool", account_mode=ACCOUNT_MODE,
    )
    # Open a real position in the shared pool so the close has something to sell.
    pm.apply_fill(symbol=SYMBOL, side="BUY", quantity=QTY, fill_price=ENTRY, fee_rate=config.PAPER_FEE_RATE)
    pm.snapshot(price_by_symbol={SYMBOL: ENTRY})  # snapshot #1

    trade_id = "trade_smoke_1"
    _insert_open_trade(trade_id, tp_price=101_000.0, sl_price=98_000.0)

    # Price rises through TP -> exit should trigger and mark CLOSING.
    actions = evaluate_virtual_exits({SYMBOL: 102_000.0}, account_mode=ACCOUNT_MODE)
    assert any(a["trade_id"] == trade_id and a["reason"] == "virtual_take_profit_reached" for a in actions)
    assert _trade_status(trade_id)[0] == "CLOSING"

    engine = ExecutionEngine(
        portfolio_manager=pm, risk_manager=None, account_mode=ACCOUNT_MODE, broker_client=None,
    )
    result = engine.close_trade(action=actions[0], market_price=102_000.0)
    assert result["status"] == "CLOSED"

    status, realized, exit_price, exit_reason = _trade_status(trade_id)
    assert status == "CLOSED"
    assert exit_reason == "virtual_take_profit_reached"
    assert exit_price is not None and exit_price > ENTRY
    assert realized is not None and realized > 0  # was always 0 before Phase A

    # A closing SELL order + fill exist with attribution back to the trade.
    with sqlite3.connect(config.DB_FILE) as conn:
        order_rows = conn.execute(
            f"SELECT COUNT(*) FROM {config.ORDERS_TABLE} WHERE trade_id=? AND side='SELL' AND status='FILLED'",
            (trade_id,),
        ).fetchone()[0]
        fill_rows = conn.execute(
            f"SELECT COUNT(*) FROM {config.FILLS_TABLE} WHERE trade_id=?",
            (trade_id,),
        ).fetchone()[0]
    assert order_rows >= 1
    assert fill_rows >= 1

    # Persist post-close equity and confirm it carries to a fresh manager instance.
    update_open_trade_unrealized(ACCOUNT_MODE, {SYMBOL: 102_000.0})
    pm.snapshot(price_by_symbol={SYMBOL: 102_000.0})  # snapshot #2
    refresh_model_performance(account_mode=ACCOUNT_MODE)

    with sqlite3.connect(config.DB_FILE) as conn:
        snap_count = conn.execute(
            f"SELECT COUNT(*) FROM {config.PORTFOLIO_SNAPSHOTS_TABLE} WHERE account_mode=?",
            (ACCOUNT_MODE,),
        ).fetchone()[0]
    assert snap_count >= 2

    pm2 = PortfolioManager(
        timeframe="1h", dry_run=True, initial_cash=10_000.0,
        model_id="shared_capital_pool", account_mode=ACCOUNT_MODE,
    )
    # Realized PnL from the round trip must have carried over (not reset to a clean 10k).
    assert pm2.realized_pnl == pm.realized_pnl
    assert abs(pm2.realized_pnl) > 0
