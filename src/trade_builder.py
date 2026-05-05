from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, asdict

from config import (
    EMERGENCY_STOP_EXTRA_ADVERSE_MULTIPLIER,
    MAX_TRADE_LOSS_USDT,
    MIN_ORDER_NOTIONAL_USDT,
    PAPER_FEE_RATE,
    PAPER_SLIPPAGE_BPS,
)
from trade_proposal_engine import TradeProposal


@dataclass(frozen=True)
class BuiltTrade:
    trade_id: str
    proposal_id: str
    allocation_id: str
    prediction_id: str
    model_id: str
    symbol: str
    timeframe: str
    side: str
    entry_reference_price: float
    approved_notional_usdt: float
    quantity: float
    tp_price: float
    sl_price: float
    emergency_sl_price: float
    model_tp_return_pct: float
    model_sl_return_pct: float
    emergency_sl_return_pct: float
    horizon_bars: int
    valid_until_utc: str
    max_loss_usdt: float
    sizing_reason: str
    raw_json: dict

    def to_dict(self) -> dict:
        return asdict(self)


def build_trade_from_allocation(proposal: TradeProposal, allocation: dict, *, step_size: float = 0.0001) -> tuple[BuiltTrade | None, list[str]]:
    """Derive TP/SL/quantity from model distribution and hard risk limits.

    Returns dimensionally correct prices: entry_price * (1 +/- return). Return
    values are decimal returns (0.01 = 1%), not percentage points.
    """
    reasons: list[str] = []
    entry = float(proposal.entry_reference_price)
    if entry <= 0:
        return None, ["invalid_entry_reference_price"]
    notional = float(allocation.get("approved_notional_usdt") or 0.0)
    if notional < float(MIN_ORDER_NOTIONAL_USDT):
        return None, ["approved_notional_below_minimum"]

    fee_slip = float(PAPER_FEE_RATE) * 2.0 + float(PAPER_SLIPPAGE_BPS) / 10_000.0
    favorable_candidates = [
        float(proposal.expected_return_pct),
        float(proposal.expected_move_pct) * float(proposal.confidence),
        float(proposal.expected_max_favorable_excursion_pct) * 0.80,
        float(proposal.q95_return_pct) * 0.50,
        float(proposal.q50_return_pct),
    ]
    tp_return = max([x for x in favorable_candidates if math.isfinite(x)] + [fee_slip * 3.0])
    tp_return = max(tp_return, fee_slip * 3.0)

    adverse_candidates = [
        abs(float(proposal.expected_adverse_move_pct)),
        abs(float(proposal.expected_max_adverse_excursion_pct)) * 0.85,
        abs(min(float(proposal.q05_return_pct), 0.0)),
    ]
    sl_return = max([x for x in adverse_candidates if math.isfinite(x)] + [fee_slip * 3.0])
    sl_return = max(sl_return, fee_slip * 3.0)

    implied_loss = notional * sl_return
    sizing_reason = "model_distribution_size_ok"
    if implied_loss > float(MAX_TRADE_LOSS_USDT):
        capped = float(MAX_TRADE_LOSS_USDT) / max(sl_return, 1e-12)
        if capped < float(MIN_ORDER_NOTIONAL_USDT):
            return None, ["model_derived_sl_exceeds_max_trade_loss_and_cannot_resize"]
        notional = capped
        sizing_reason = "resized_to_max_trade_loss"

    raw_qty = notional / entry
    if step_size and step_size > 0:
        qty = math.floor(raw_qty / float(step_size)) * float(step_size)
    else:
        qty = raw_qty
    if qty <= 0:
        return None, ["quantity_rounds_to_zero"]

    # Recompute notional after step-size flooring so stored quantity and risk match.
    notional = qty * entry
    tp_price = entry * (1.0 + tp_return)
    sl_price = entry * (1.0 - sl_return)
    emergency_return = sl_return * float(EMERGENCY_STOP_EXTRA_ADVERSE_MULTIPLIER)
    emergency_sl_price = entry * (1.0 - emergency_return)
    if not (tp_price > entry > sl_price > 0 and emergency_sl_price > 0):
        reasons.append("invalid_model_derived_tp_sl_dimensions")
    if emergency_sl_price > sl_price:
        reasons.append("emergency_stop_not_wider_than_virtual_sl")
    if reasons:
        return None, reasons

    return BuiltTrade(
        trade_id=f"trade_{uuid.uuid4().hex}",
        proposal_id=proposal.proposal_id,
        allocation_id=str(allocation["allocation_id"]),
        prediction_id=proposal.prediction_id,
        model_id=proposal.model_id,
        symbol=proposal.symbol,
        timeframe=proposal.timeframe,
        side="LONG",
        entry_reference_price=entry,
        approved_notional_usdt=float(notional),
        quantity=float(qty),
        tp_price=float(tp_price),
        sl_price=float(sl_price),
        emergency_sl_price=float(emergency_sl_price),
        model_tp_return_pct=float(tp_return),
        model_sl_return_pct=float(sl_return),
        emergency_sl_return_pct=float(emergency_return),
        horizon_bars=int(proposal.horizon_bars),
        valid_until_utc=proposal.valid_until_utc,
        max_loss_usdt=float(notional * sl_return),
        sizing_reason=sizing_reason,
        raw_json={
            "fee_slippage_return_assumption": fee_slip,
            "tp_inputs": favorable_candidates,
            "sl_inputs": adverse_candidates,
            "dimension_rule": "price * dimensionless_return = price",
        },
    ), []
