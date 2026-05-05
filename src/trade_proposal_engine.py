from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, asdict

import pandas as pd

from config import (
    DB_FILE,
    DEFAULT_QUOTE_SIZE_USDT,
    MAX_ORDER_NOTIONAL_USDT,
    TRADE_PROPOSALS_TABLE,
    ENABLE_SHADOW_TRADES_FOR_REJECTED_PROPOSALS,
)
from db_utils import init_research_tables
from prediction_engine import ModelPrediction, prediction_is_tradeable


@dataclass(frozen=True)
class TradeProposal:
    proposal_id: str
    prediction_id: str
    model_id: str
    symbol: str
    timeframe: str
    timestamp_utc: str
    side: str
    entry_reference_price: float
    confidence: float
    expected_return_pct: float
    expected_move_pct: float
    expected_adverse_move_pct: float
    expected_value_usdt: float
    q05_return_pct: float
    q50_return_pct: float
    q95_return_pct: float
    expected_max_favorable_excursion_pct: float
    expected_max_adverse_excursion_pct: float
    horizon_bars: int
    valid_until_utc: str
    requested_notional_usdt: float
    proposal_score: float
    status: str
    rejection_reason: str | None
    raw_json: dict

    def to_dict(self) -> dict:
        return asdict(self)


def build_trade_proposal(prediction: ModelPrediction, entry_reference_price: float, requested_notional_usdt: float | None = None) -> TradeProposal:
    init_research_tables()
    requested = float(requested_notional_usdt if requested_notional_usdt is not None else DEFAULT_QUOTE_SIZE_USDT)
    requested = min(max(requested, 0.0), float(MAX_ORDER_NOTIONAL_USDT))
    expected_value = requested * float(prediction.expected_return_pct)
    downside = max(float(prediction.expected_adverse_move_pct), 1e-9)
    risk_reward = max(float(prediction.expected_return_pct), 0.0) / downside
    score = expected_value * float(prediction.confidence) * min(risk_reward, 5.0)
    tradeable, reasons = prediction_is_tradeable(prediction)
    status = "PROPOSED" if tradeable else "REJECTED"
    return TradeProposal(
        proposal_id=f"prop_{uuid.uuid4().hex}",
        prediction_id=prediction.prediction_id,
        model_id=prediction.model_id,
        symbol=prediction.symbol,
        timeframe=prediction.timeframe,
        timestamp_utc=prediction.timestamp_utc,
        side="LONG" if prediction.direction == "LONG" else prediction.direction,
        entry_reference_price=float(entry_reference_price),
        confidence=float(prediction.confidence),
        expected_return_pct=float(prediction.expected_return_pct),
        expected_move_pct=float(prediction.expected_move_pct),
        expected_adverse_move_pct=float(prediction.expected_adverse_move_pct),
        expected_value_usdt=float(expected_value),
        q05_return_pct=float(prediction.q05_return_pct),
        q50_return_pct=float(prediction.q50_return_pct),
        q95_return_pct=float(prediction.q95_return_pct),
        expected_max_favorable_excursion_pct=float(prediction.expected_max_favorable_excursion_pct),
        expected_max_adverse_excursion_pct=float(prediction.expected_max_adverse_excursion_pct),
        horizon_bars=int(prediction.horizon_bars),
        valid_until_utc=prediction.signal_valid_until_utc,
        requested_notional_usdt=float(requested),
        proposal_score=float(score),
        status=status,
        rejection_reason=";".join(reasons) if reasons else None,
        raw_json={
            "source": "prediction_engine",
            "prediction_direction": prediction.direction,
            "risk_reward_estimate": risk_reward,
            "shadow_enabled_if_rejected": ENABLE_SHADOW_TRADES_FOR_REJECTED_PROPOSALS,
        },
    )


def persist_trade_proposal(proposal: TradeProposal) -> str:
    p = proposal.to_dict()
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {TRADE_PROPOSALS_TABLE} (
                proposal_id, prediction_id, model_id, symbol, timeframe, timestamp_utc, side,
                entry_reference_price, confidence, expected_return_pct, expected_move_pct,
                expected_adverse_move_pct, expected_value_usdt, q05_return_pct, q50_return_pct,
                q95_return_pct, expected_max_favorable_excursion_pct,
                expected_max_adverse_excursion_pct, horizon_bars, valid_until_utc,
                requested_notional_usdt, proposal_score, status, rejection_reason, raw_json,
                created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                p["proposal_id"], p["prediction_id"], p["model_id"], p["symbol"], p["timeframe"], p["timestamp_utc"], p["side"],
                p["entry_reference_price"], p["confidence"], p["expected_return_pct"], p["expected_move_pct"],
                p["expected_adverse_move_pct"], p["expected_value_usdt"], p["q05_return_pct"], p["q50_return_pct"],
                p["q95_return_pct"], p["expected_max_favorable_excursion_pct"], p["expected_max_adverse_excursion_pct"],
                p["horizon_bars"], p["valid_until_utc"], p["requested_notional_usdt"], p["proposal_score"], p["status"],
                p["rejection_reason"], json.dumps(p["raw_json"], ensure_ascii=True, sort_keys=True),
                pd.Timestamp.now(tz="UTC").isoformat(), pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
        conn.commit()
    return proposal.proposal_id


def mark_proposal_decision(proposal_id: str, status: str, reason: str | None = None) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            f"UPDATE {TRADE_PROPOSALS_TABLE} SET status=?, rejection_reason=COALESCE(?, rejection_reason), updated_at_utc=? WHERE proposal_id=?",
            (status, reason, pd.Timestamp.now(tz="UTC").isoformat(), proposal_id),
        )
        conn.commit()
