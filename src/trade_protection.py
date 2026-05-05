from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LongProtection:
    entry_price: float
    take_profit_price: float
    stop_loss_price: float
    risk_reward: float
    tp_multiplier: float
    sl_multiplier: float


def _is_finite_positive(value: float) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0


def build_long_protection(
    entry_price: float,
    atr: float,
    tp_multiplier: float,
    sl_multiplier: float,
) -> LongProtection:
    """Build the same ATR-based long TP/SL bracket used by triple-barrier labels."""
    if not _is_finite_positive(entry_price):
        raise ValueError("entry_price must be finite and > 0")
    if not _is_finite_positive(atr):
        raise ValueError("atr must be finite and > 0")
    if not _is_finite_positive(tp_multiplier):
        raise ValueError("tp_multiplier must be finite and > 0")
    if not _is_finite_positive(sl_multiplier):
        raise ValueError("sl_multiplier must be finite and > 0")

    entry = float(entry_price)
    atr_value = float(atr)
    tp = entry + float(tp_multiplier) * atr_value
    sl = entry - float(sl_multiplier) * atr_value
    if sl <= 0:
        raise ValueError("stop_loss_price would be <= 0")
    rr = (tp - entry) / max(entry - sl, 1e-12)
    return LongProtection(
        entry_price=entry,
        take_profit_price=float(tp),
        stop_loss_price=float(sl),
        risk_reward=float(rr),
        tp_multiplier=float(tp_multiplier),
        sl_multiplier=float(sl_multiplier),
    )


def validate_long_protection(
    entry_price: float,
    take_profit_price: float | None,
    stop_loss_price: float | None,
    min_risk_reward: float = 1.0,
) -> tuple[bool, list[str], dict]:
    """Validate that a long entry has a protective TP above entry and SL below entry."""
    reasons: list[str] = []
    details: dict[str, float | None] = {
        "entry_price": float(entry_price) if _is_finite_positive(entry_price) else None,
        "take_profit_price": float(take_profit_price) if take_profit_price is not None else None,
        "stop_loss_price": float(stop_loss_price) if stop_loss_price is not None else None,
        "risk_reward": None,
    }

    if not _is_finite_positive(entry_price):
        reasons.append("invalid_entry_price_for_protection")
        return False, reasons, details
    if take_profit_price is None or not _is_finite_positive(take_profit_price):
        reasons.append("missing_or_invalid_take_profit")
    if stop_loss_price is None or not _is_finite_positive(stop_loss_price):
        reasons.append("missing_or_invalid_stop_loss")
    if reasons:
        return False, reasons, details

    entry = float(entry_price)
    tp = float(take_profit_price)
    sl = float(stop_loss_price)
    if tp <= entry:
        reasons.append("take_profit_not_above_entry")
    if sl >= entry:
        reasons.append("stop_loss_not_below_entry")
    if sl <= 0:
        reasons.append("stop_loss_not_positive")

    if not reasons:
        rr = (tp - entry) / max(entry - sl, 1e-12)
        details["risk_reward"] = float(rr)
        if rr < float(min_risk_reward):
            reasons.append("risk_reward_below_minimum")

    return len(reasons) == 0, reasons, details
