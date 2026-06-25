"""Phase D: shadow-trade outcome evaluation + analytics.

Rejected/resized proposals are recorded as shadow trades (no orders sent). This module resolves
their hypothetical outcome against market data after the fact, so the dashboard can answer:
"which rejected proposals would have won?" and "is the allocator leaving money on the table?".

Resolution mirrors the bot's virtual TP/SL/horizon exit logic on OHLC candles. Long/exit only
(Binance Spot scope): a shadow is a hypothetical long entry at its reference price.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd

from config import DB_FILE, PRICES_TABLE, SHADOW_TRADES_TABLE
from db_utils import init_research_tables


def resolve_shadow_outcome(
    entry: float,
    tp: float | None,
    sl: float | None,
    bars: list[tuple[float, float, float]],
) -> tuple[float, str]:
    """Resolve a hypothetical long shadow trade to (return_pct, exit_reason).

    ``bars`` is the ordered list of (high, low, close) after entry. TP is checked before SL on a
    bar (optimistic); if neither barrier is hit the trade expires at the last close.
    """
    if entry is None or entry <= 0 or not bars:
        return 0.0, "no_data"
    for high, low, close in bars:
        if tp is not None and high >= tp:
            return float(tp) / entry - 1.0, "tp"
        if sl is not None and low <= sl:
            return float(sl) / entry - 1.0, "sl"
    return float(bars[-1][2]) / entry - 1.0, "expire"


def _load_bars(symbol: str, timeframe: str | None, start_utc: str, end_utc: str) -> list[tuple[float, float, float]]:
    with sqlite3.connect(DB_FILE) as conn:
        params: list = [symbol]
        tf_clause = ""
        if timeframe:
            tf_clause = "AND timeframe = ?"
            params.append(timeframe)
        params.extend([start_utc, end_utc])
        df = pd.read_sql_query(
            f"""
            SELECT high, low, close FROM {PRICES_TABLE}
            WHERE symbol = ? {tf_clause} AND datetime_utc > ? AND datetime_utc <= ?
            ORDER BY datetime_utc
            """,
            conn,
            params=tuple(params),
        )
    if df.empty:
        return []
    return [(float(r.high), float(r.low), float(r.close)) for r in df.itertuples(index=False)]


def evaluate_open_shadow_trades(limit: int = 1000) -> int:
    """Resolve SHADOW_OPEN trades whose validity window has passed; book outcome + close. Returns count closed."""
    init_research_tables()
    now = pd.Timestamp.now(tz="UTC")
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute(
            f"""
            SELECT shadow_trade_id, symbol, timeframe, entry_reference_price, tp_price, sl_price,
                   valid_until_utc, requested_notional_usdt, opened_at_utc, created_at_utc
            FROM {SHADOW_TRADES_TABLE}
            WHERE status = 'SHADOW_OPEN'
            ORDER BY created_at_utc
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

        closed = 0
        for (sid, symbol, timeframe, entry, tp, sl, valid_until, notional, opened_at, created_at) in rows:
            if not valid_until or pd.Timestamp(valid_until) > now:
                continue  # still within validity window; resolve later
            start = opened_at or created_at
            bars = _load_bars(symbol, timeframe, str(start), str(valid_until))
            if not bars:
                continue  # no market data yet to resolve
            ret_pct, reason = resolve_shadow_outcome(
                float(entry) if entry is not None else 0.0,
                float(tp) if tp is not None else None,
                float(sl) if sl is not None else None,
                bars,
            )
            outcome_pnl = float(ret_pct) * float(notional or 0.0)
            conn.execute(
                f"""
                UPDATE {SHADOW_TRADES_TABLE}
                SET status='SHADOW_CLOSED', outcome_pnl_usdt=?, closed_at_utc=?, updated_at_utc=?,
                    raw_json=?
                WHERE shadow_trade_id=?
                """,
                (
                    outcome_pnl, now.isoformat(), now.isoformat(),
                    json.dumps({"return_pct": ret_pct, "exit_reason": reason}, ensure_ascii=True, sort_keys=True),
                    sid,
                ),
            )
            closed += 1
        conn.commit()
    return closed


def shadow_analytics() -> dict:
    """Summary the dashboard surfaces: how the allocator's rejected/resized proposals played out."""
    init_research_tables()
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query(
            f"SELECT status, reason, outcome_pnl_usdt FROM {SHADOW_TRADES_TABLE}",
            conn,
        )
    total = int(len(df))
    if total == 0:
        return {"total": 0, "open": 0, "closed": 0, "would_have_won": 0, "would_have_won_rate": 0.0,
                "avg_outcome_pnl_usdt": 0.0, "missed_profit_usdt": 0.0, "avoided_loss_usdt": 0.0, "by_reason": {}}
    df["outcome_pnl_usdt"] = pd.to_numeric(df["outcome_pnl_usdt"], errors="coerce")
    closed = df[df["status"].astype(str) == "SHADOW_CLOSED"].copy()
    n_closed = int(len(closed))
    wins = closed[closed["outcome_pnl_usdt"] > 0]
    losses = closed[closed["outcome_pnl_usdt"] < 0]
    by_reason = {str(k): int(v) for k, v in df["reason"].fillna("unknown").value_counts().to_dict().items()}
    return {
        "total": total,
        "open": int((df["status"].astype(str) == "SHADOW_OPEN").sum()),
        "closed": n_closed,
        "would_have_won": int(len(wins)),
        "would_have_won_rate": float(len(wins) / n_closed) if n_closed else 0.0,
        "avg_outcome_pnl_usdt": float(closed["outcome_pnl_usdt"].mean()) if n_closed else 0.0,
        "missed_profit_usdt": float(wins["outcome_pnl_usdt"].sum()) if len(wins) else 0.0,
        "avoided_loss_usdt": float(losses["outcome_pnl_usdt"].sum()) if len(losses) else 0.0,
        "by_reason": by_reason,
    }


if __name__ == "__main__":
    n = evaluate_open_shadow_trades()
    print(json.dumps({"closed_shadow_trades": n, "analytics": shadow_analytics()}, ensure_ascii=True, indent=2))
