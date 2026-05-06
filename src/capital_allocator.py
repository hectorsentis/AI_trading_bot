from __future__ import annotations

import json
import sqlite3
import uuid

import pandas as pd

from config import (
    ALLOCATIONS_TABLE,
    ALLOCATOR_MIN_SCORE,
    DB_FILE,
    ENABLE_SHADOW_TRADES_FOR_REJECTED_PROPOSALS,
    MAX_MODEL_OPEN_EXPOSURE_USDT,
    MAX_OPEN_TRADES_PER_MODEL,
    MAX_OPEN_TRADES_PER_SYMBOL,
    MAX_OPEN_TRADES_TOTAL,
    MAX_ORDER_NOTIONAL_USDT,
    MAX_SYMBOL_EXPOSURE_USDT,
    MAX_TOTAL_EXPOSURE_USDT,
    MIN_CASH_RESERVE_USDT,
    MIN_ORDER_NOTIONAL_USDT,
    SHADOW_TRADES_TABLE,
    TRADES_TABLE,
)
from db_utils import init_research_tables
from trade_proposal_engine import TradeProposal, mark_proposal_decision


OPEN_TRADE_STATUSES = ("APPROVED", "ORDER_PENDING", "ORDER_SENT", "PARTIALLY_FILLED", "OPEN", "REDUCING", "CLOSING")


class CapitalAllocator:
    """Central shared-capital allocator for independent model proposals."""

    def __init__(self, account_mode: str = "local_paper"):
        init_research_tables()
        self.account_mode = account_mode

    def _portfolio_cash(self) -> float:
        with sqlite3.connect(DB_FILE) as conn:
            row = conn.execute(
                "SELECT cash FROM portfolio_snapshots WHERE account_mode=? ORDER BY snapshot_id DESC LIMIT 1",
                (self.account_mode,),
            ).fetchone()
        if row is None:
            return 10_000.0
        return float(row[0])

    def _open_trade_stats(self) -> dict:
        qs = ",".join("?" for _ in OPEN_TRADE_STATUSES)
        with sqlite3.connect(DB_FILE) as conn:
            rows = conn.execute(
                f"""
                SELECT model_id, symbol, COALESCE(approved_notional_usdt,0) AS n
                FROM {TRADES_TABLE}
                WHERE account_mode=? AND status IN ({qs})
                """,
                (self.account_mode, *OPEN_TRADE_STATUSES),
            ).fetchall()
        total = sum(float(r[2]) for r in rows)
        by_model: dict[str, float] = {}
        by_symbol: dict[str, float] = {}
        counts_by_model: dict[str, int] = {}
        counts_by_symbol: dict[str, int] = {}
        for model_id, symbol, notional in rows:
            by_model[str(model_id)] = by_model.get(str(model_id), 0.0) + float(notional)
            by_symbol[str(symbol)] = by_symbol.get(str(symbol), 0.0) + float(notional)
            counts_by_model[str(model_id)] = counts_by_model.get(str(model_id), 0) + 1
            counts_by_symbol[str(symbol)] = counts_by_symbol.get(str(symbol), 0) + 1
        return {
            "total_open_notional": total,
            "by_model": by_model,
            "by_symbol": by_symbol,
            "total_count": len(rows),
            "counts_by_model": counts_by_model,
            "counts_by_symbol": counts_by_symbol,
        }

    def allocate(self, proposal: TradeProposal, evaluation_timestamp_utc: str | None = None) -> dict:
        """Evaluate one model-owned proposal against the shared capital pool.

        ``evaluation_timestamp_utc`` is optional and defaults to wall-clock time
        for paper/live operation. Historical simulation passes the candle time so
        signal-validity checks use the same semantics without rejecting old OOS
        predictions merely because they are in the past today.
        """
        now = pd.Timestamp(evaluation_timestamp_utc) if evaluation_timestamp_utc else pd.Timestamp.now(tz="UTC")
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        else:
            now = now.tz_convert("UTC")
        reasons: list[str] = []
        decision = "ACCEPT"
        approved_notional = min(float(proposal.requested_notional_usdt), float(MAX_ORDER_NOTIONAL_USDT))
        cash = self._portfolio_cash()
        stats = self._open_trade_stats()

        if proposal.status == "REJECTED":
            reasons.append(proposal.rejection_reason or "proposal_pre_rejected")
        if proposal.side != "LONG":
            reasons.append("spot_allocator_only_opens_long_proposals")
        if pd.Timestamp(proposal.valid_until_utc) <= now:
            reasons.append("proposal_signal_expired")
        if float(proposal.proposal_score) < float(ALLOCATOR_MIN_SCORE):
            reasons.append("proposal_score_below_allocator_threshold")
        if float(proposal.expected_return_pct) <= 0:
            reasons.append("expected_return_after_costs_non_positive")
        if approved_notional < float(MIN_ORDER_NOTIONAL_USDT):
            reasons.append("requested_notional_below_minimum")
        if approved_notional > max(0.0, cash - float(MIN_CASH_RESERVE_USDT)):
            approved_notional = max(0.0, cash - float(MIN_CASH_RESERVE_USDT))
            if approved_notional < float(MIN_ORDER_NOTIONAL_USDT):
                reasons.append("insufficient_shared_cash_after_reserve")
            else:
                decision = "RESIZE"
        remaining_total = float(MAX_TOTAL_EXPOSURE_USDT) - float(stats["total_open_notional"])
        remaining_symbol = float(MAX_SYMBOL_EXPOSURE_USDT) - float(stats["by_symbol"].get(proposal.symbol, 0.0))
        remaining_model = float(MAX_MODEL_OPEN_EXPOSURE_USDT) - float(stats["by_model"].get(proposal.model_id, 0.0))
        cap = min(approved_notional, remaining_total, remaining_symbol, remaining_model)
        if cap < approved_notional:
            approved_notional = max(0.0, cap)
            decision = "RESIZE"
        if approved_notional < float(MIN_ORDER_NOTIONAL_USDT):
            reasons.append("exposure_limits_leave_no_minimum_notional")
        if int(stats["total_count"]) >= int(MAX_OPEN_TRADES_TOTAL):
            reasons.append("max_open_trades_total_reached")
        if int(stats["counts_by_model"].get(proposal.model_id, 0)) >= int(MAX_OPEN_TRADES_PER_MODEL):
            reasons.append("max_open_trades_per_model_reached")
        if int(stats["counts_by_symbol"].get(proposal.symbol, 0)) >= int(MAX_OPEN_TRADES_PER_SYMBOL):
            reasons.append("max_open_trades_per_symbol_reached")

        if reasons:
            decision = "SHADOW" if ENABLE_SHADOW_TRADES_FOR_REJECTED_PROPOSALS else "REJECT"
            approved_notional = 0.0
        allocation_id = f"alloc_{uuid.uuid4().hex}"
        shadow_trade_id = None
        if decision == "SHADOW":
            shadow_trade_id = self._create_shadow_trade(proposal, allocation_id, ";".join(reasons))
        allocation = {
            "allocation_id": allocation_id,
            "proposal_id": proposal.proposal_id,
            "prediction_id": proposal.prediction_id,
            "model_id": proposal.model_id,
            "symbol": proposal.symbol,
            "timestamp_utc": now.isoformat(),
            "decision": decision,
            "requested_notional_usdt": float(proposal.requested_notional_usdt),
            "approved_notional_usdt": float(approved_notional),
            "rejection_reason": ";".join(reasons) if reasons else None,
            "allocator_score": float(proposal.proposal_score),
            "expected_value_usdt": float(proposal.expected_value_usdt),
            "shadow_trade_id": shadow_trade_id,
            "risk_json": {
                "cash": cash,
                "open_trade_stats": stats,
                "limits": {
                    "max_total_exposure_usdt": MAX_TOTAL_EXPOSURE_USDT,
                    "max_symbol_exposure_usdt": MAX_SYMBOL_EXPOSURE_USDT,
                    "max_model_open_exposure_usdt": MAX_MODEL_OPEN_EXPOSURE_USDT,
                    "min_cash_reserve_usdt": MIN_CASH_RESERVE_USDT,
                },
            },
            "decision_json": {"notes": "Independent proposal evaluated against shared capital pool; signals are not averaged."},
        }
        self._persist_allocation(allocation)
        if decision in {"ACCEPT", "RESIZE"}:
            mark_proposal_decision(proposal.proposal_id, "ACCEPTED" if decision == "ACCEPT" else "RESIZED", None)
        else:
            mark_proposal_decision(proposal.proposal_id, decision, allocation["rejection_reason"])
        return allocation

    def _persist_allocation(self, allocation: dict) -> None:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {ALLOCATIONS_TABLE} (
                    allocation_id, proposal_id, prediction_id, model_id, symbol, timestamp_utc,
                    decision, requested_notional_usdt, approved_notional_usdt, rejection_reason,
                    allocator_score, expected_value_usdt, shadow_trade_id, risk_json,
                    decision_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    allocation["allocation_id"], allocation["proposal_id"], allocation["prediction_id"], allocation["model_id"], allocation["symbol"],
                    allocation["timestamp_utc"], allocation["decision"], allocation["requested_notional_usdt"], allocation["approved_notional_usdt"],
                    allocation["rejection_reason"], allocation["allocator_score"], allocation["expected_value_usdt"], allocation["shadow_trade_id"],
                    json.dumps(allocation["risk_json"], ensure_ascii=True, sort_keys=True),
                    json.dumps(allocation["decision_json"], ensure_ascii=True, sort_keys=True), pd.Timestamp.now(tz="UTC").isoformat(),
                ),
            )
            conn.commit()

    def _create_shadow_trade(self, proposal: TradeProposal, allocation_id: str, reason: str) -> str:
        shadow_trade_id = f"shadow_{uuid.uuid4().hex}"
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {SHADOW_TRADES_TABLE} (
                    shadow_trade_id, proposal_id, allocation_id, prediction_id, model_id, symbol,
                    timeframe, side, status, entry_reference_price, horizon_bars, valid_until_utc,
                    requested_notional_usdt, reason, raw_json, opened_at_utc, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    shadow_trade_id, proposal.proposal_id, allocation_id, proposal.prediction_id,
                    proposal.model_id, proposal.symbol, proposal.timeframe, proposal.side, "SHADOW_OPEN",
                    proposal.entry_reference_price, proposal.horizon_bars, proposal.valid_until_utc,
                    proposal.requested_notional_usdt, reason,
                    json.dumps({"proposal_score": proposal.proposal_score}, ensure_ascii=True, sort_keys=True),
                    pd.Timestamp.now(tz="UTC").isoformat(), pd.Timestamp.now(tz="UTC").isoformat(), pd.Timestamp.now(tz="UTC").isoformat(),
                ),
            )
            conn.commit()
        return shadow_trade_id
