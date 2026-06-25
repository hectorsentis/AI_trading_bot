from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, asdict

import pandas as pd

from config import (
    DB_FILE,
    DEFAULT_SIGNAL_HORIZON_BARS,
    MODEL_PREDICTIONS_TABLE,
    PAPER_FEE_RATE,
    PAPER_SLIPPAGE_BPS,
    TRADE_PROPOSAL_MIN_CONFIDENCE,
    TRADE_PROPOSAL_MIN_EXPECTED_RETURN_PCT,
)
from db_utils import init_research_tables
from signal_engine import generate_signal_from_probabilities


@dataclass(frozen=True)
class ModelPrediction:
    prediction_id: str
    model_id: str
    symbol: str
    timeframe: str
    timestamp_utc: str
    direction: str
    confidence: float
    prob_up: float
    prob_down: float
    prob_flat: float
    expected_return_pct: float
    expected_move_pct: float
    expected_adverse_move_pct: float
    q05_return_pct: float
    q25_return_pct: float
    q50_return_pct: float
    q75_return_pct: float
    q95_return_pct: float
    expected_max_favorable_excursion_pct: float
    expected_max_adverse_excursion_pct: float
    horizon_bars: int
    signal_valid_until_utc: str
    raw_prediction_json: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def build_prediction_from_probabilities(
    *,
    model_id: str,
    symbol: str,
    timeframe: str,
    timestamp_utc,
    close_price: float,
    prob_short: float,
    prob_flat: float,
    prob_long: float,
    latest_features: dict | None = None,
    horizon_bars: int = DEFAULT_SIGNAL_HORIZON_BARS,
    raw: dict | None = None,
) -> ModelPrediction:
    """Convert legacy classifier probabilities into the structured prediction contract.

    The current trained artifacts are direction classifiers, not full return-distribution
    models. This adapter therefore derives conservative distribution fields from the
    model probabilities and contemporaneous volatility/ATR features. It preserves the
    final architecture while leaving room for native regression/quantile models.
    """
    init_research_tables()
    latest_features = latest_features or {}
    ts = pd.Timestamp(timestamp_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    signal = generate_signal_from_probabilities(prob_short=prob_short, prob_flat=prob_flat, prob_long=prob_long)
    direction = "LONG" if int(signal["final_signal_position"]) > 0 else ("EXIT" if prob_short > prob_flat and prob_short > prob_long else "FLAT")
    confidence = max(float(prob_long), float(prob_flat), float(prob_short))

    atr_pct = _float(latest_features.get("atr_pct"), 0.0)
    if atr_pct <= 0:
        atr = _float(latest_features.get("atr_14"), 0.0)
        atr_pct = atr / max(float(close_price), 1e-12)
    volatility = max(
        abs(atr_pct),
        _float(latest_features.get("volatility_20"), 0.0),
        _float(latest_features.get("volatility_10"), 0.0),
        0.001,
    )
    edge = float(prob_long) - float(prob_short)
    fee_slip = float(PAPER_FEE_RATE) * 2.0 + float(PAPER_SLIPPAGE_BPS) / 10_000.0
    expected_return = edge * volatility * max(1.0, horizon_bars ** 0.5) - fee_slip
    expected_move = max(abs(expected_return), volatility * max(float(prob_long), float(prob_short), 0.5))
    adverse = max(volatility * (float(prob_short) + 0.25 * float(prob_flat)), abs(min(expected_return, 0.0)), 0.0005)

    # Conservative synthetic quantiles around the expected value. These are in
    # return units, not percentages-as-100; 0.01 means +1%.
    q05 = expected_return - 1.65 * volatility
    q25 = expected_return - 0.67 * volatility
    q50 = expected_return
    q75 = expected_return + 0.67 * volatility
    q95 = expected_return + 1.65 * volatility
    valid_until = ts + pd.Timedelta(seconds=int(horizon_bars) * _timeframe_seconds(timeframe))

    payload = {
        "classifier_probabilities": {"short": prob_short, "flat": prob_flat, "long": prob_long},
        "legacy_signal": signal,
        "adapter_note": "Distribution fields are conservative estimates from classifier probabilities and current volatility until native quantile/MFE/MAE models are trained.",
    }
    if raw:
        payload.update(raw)

    return ModelPrediction(
        prediction_id=f"pred_{uuid.uuid4().hex}",
        model_id=model_id,
        symbol=symbol.upper(),
        timeframe=timeframe,
        timestamp_utc=ts.isoformat(),
        direction=direction,
        confidence=float(confidence),
        prob_up=float(prob_long),
        prob_down=float(prob_short),
        prob_flat=float(prob_flat),
        expected_return_pct=float(expected_return),
        expected_move_pct=float(expected_move),
        expected_adverse_move_pct=float(adverse),
        q05_return_pct=float(q05),
        q25_return_pct=float(q25),
        q50_return_pct=float(q50),
        q75_return_pct=float(q75),
        q95_return_pct=float(q95),
        expected_max_favorable_excursion_pct=float(max(q75, expected_move, volatility)),
        expected_max_adverse_excursion_pct=float(max(abs(q05), adverse, volatility * 0.75)),
        horizon_bars=int(horizon_bars),
        signal_valid_until_utc=valid_until.isoformat(),
        raw_prediction_json=payload,
    )


def _fee_slip() -> float:
    return float(PAPER_FEE_RATE) * 2.0 + float(PAPER_SLIPPAGE_BPS) / 10_000.0


def _direction_and_confidence(prob_short: float, prob_flat: float, prob_long: float) -> tuple[str, float, dict]:
    signal = generate_signal_from_probabilities(prob_short=prob_short, prob_flat=prob_flat, prob_long=prob_long)
    direction = (
        "LONG"
        if int(signal["final_signal_position"]) > 0
        else ("EXIT" if prob_short > prob_flat and prob_short > prob_long else "FLAT")
    )
    confidence = max(float(prob_long), float(prob_flat), float(prob_short))
    return direction, confidence, signal


def assemble_native_fields(
    *,
    raw_expected_return: float,
    quantiles_by_level: dict[float, float],
    raw_mfe: float,
    raw_mae: float,
) -> dict:
    """Turn raw native model outputs into the net, monotonic distribution fields.

    Shared by the live native path (`build_prediction_from_native`) and the per-fold validation
    path so both produce identical fields. Subtracts round-trip costs and enforces monotonic
    quantiles (quantile regressors can cross). Returns return-unit fractions (0.01 == +1%).
    """
    fee_slip = _fee_slip()
    levels = sorted(quantiles_by_level)
    q_net = sorted(_float(quantiles_by_level[a], 0.0) - fee_slip for a in levels)
    q_map = {a: q_net[i] for i, a in enumerate(levels)}
    last = len(q_net) - 1
    q05 = q_map.get(0.05, q_net[0])
    q25 = q_map.get(0.25, q_net[min(1, last)])
    q50 = q_map.get(0.5, q_net[last // 2])
    q75 = q_map.get(0.75, q_net[min(3, last)])
    q95 = q_map.get(0.95, q_net[last])

    expected_return = _float(raw_expected_return, 0.0) - fee_slip
    mfe = max(_float(raw_mfe, 0.0), 0.0)
    mae = min(_float(raw_mae, 0.0), 0.0)
    expected_move = max(abs(expected_return), q75, mfe, 0.0005)
    adverse = max(abs(mae), abs(q05), 0.0005)
    return {
        "expected_return_pct": float(expected_return),
        "expected_move_pct": float(expected_move),
        "expected_adverse_move_pct": float(adverse),
        "q05_return_pct": float(q05),
        "q25_return_pct": float(q25),
        "q50_return_pct": float(q50),
        "q75_return_pct": float(q75),
        "q95_return_pct": float(q95),
        "expected_max_favorable_excursion_pct": float(max(mfe, q95, abs(expected_return))),
        "expected_max_adverse_excursion_pct": float(max(abs(mae), abs(q05), 0.0005)),
    }


def build_prediction_from_native(
    *,
    model_id: str,
    symbol: str,
    timeframe: str,
    timestamp_utc,
    native_models: dict,
    feature_frame,
    prob_short: float,
    prob_flat: float,
    prob_long: float,
    horizon_bars: int | None = None,
    raw: dict | None = None,
) -> ModelPrediction:
    """Build a structured prediction from trained native return-distribution models.

    Direction/confidence still come from the calibrated classifier (the native models predict
    magnitude, not class), but expected return, quantiles and MFE/MAE are real model outputs
    rather than synthetic estimates. ``feature_frame`` is a 1-row DataFrame ordered by the
    native model's ``feature_columns``. Costs (fees + slippage) are subtracted so the reported
    distribution is net, consistent with the derived path and the proposal/allocator gates.
    """
    init_research_tables()
    ts = pd.Timestamp(timestamp_utc)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    horizon = int(horizon_bars or native_models.get("horizon_bars") or DEFAULT_SIGNAL_HORIZON_BARS)
    feature_cols = native_models.get("feature_columns")
    X = feature_frame[feature_cols] if feature_cols is not None else feature_frame

    direction, confidence, signal = _direction_and_confidence(prob_short, prob_flat, prob_long)

    q_levels = native_models.get("quantile_levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    q_models = native_models["quantiles"]
    raw_ret = _float(native_models["expected_return"].predict(X)[0], 0.0)
    raw_mfe = _float(native_models["mfe"].predict(X)[0], 0.0)
    raw_mae = _float(native_models["mae"].predict(X)[0], 0.0)
    quantiles_by_level = {float(a): _float(q_models[f"{a:.2f}"].predict(X)[0], 0.0) for a in q_levels}
    fields = assemble_native_fields(
        raw_expected_return=raw_ret,
        quantiles_by_level=quantiles_by_level,
        raw_mfe=raw_mfe,
        raw_mae=raw_mae,
    )

    valid_until = ts + pd.Timedelta(seconds=int(horizon) * _timeframe_seconds(timeframe))
    payload = {
        "native_models": True,
        "raw_expected_return": raw_ret,
        "raw_mfe": raw_mfe,
        "raw_mae": raw_mae,
        "classifier_probabilities": {"short": prob_short, "flat": prob_flat, "long": prob_long},
        "legacy_signal": signal,
        "trained_rows": native_models.get("trained_rows"),
    }
    if raw:
        payload.update(raw)

    return ModelPrediction(
        prediction_id=f"pred_{uuid.uuid4().hex}",
        model_id=model_id,
        symbol=symbol.upper(),
        timeframe=timeframe,
        timestamp_utc=ts.isoformat(),
        direction=direction,
        confidence=float(confidence),
        prob_up=float(prob_long),
        prob_down=float(prob_short),
        prob_flat=float(prob_flat),
        horizon_bars=int(horizon),
        signal_valid_until_utc=valid_until.isoformat(),
        raw_prediction_json=payload,
        **fields,
    )


def build_prediction_from_row_fields(
    row,
    *,
    model_id: str,
    symbol: str,
    timeframe: str,
    timestamp_utc,
    horizon_bars: int | None = None,
) -> ModelPrediction | None:
    """Build a structured prediction from already-computed (persisted) native fields.

    Used by the historical simulator so validation/backtest acceptance evaluates the same native,
    cost-adjusted, calibrated predictions the model produced per fold — not a recomputation that
    could mix models. Returns ``None`` when the row lacks native fields, so the caller falls back.
    """
    def _get(key):
        try:
            val = row[key]
        except Exception:
            return None
        return None if (val is None or pd.isna(val)) else val

    has_native = _get("has_native_prediction")
    if not has_native or _get("expected_return_pct") is None:
        return None

    ts = pd.Timestamp(timestamp_utc)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    horizon = int(horizon_bars or _get("native_horizon_bars") or DEFAULT_SIGNAL_HORIZON_BARS)
    prob_short = _float(_get("prob_short"), 0.0)
    prob_flat = _float(_get("prob_flat"), 0.0)
    prob_long = _float(_get("prob_long"), 0.0)
    direction, confidence, signal = _direction_and_confidence(prob_short, prob_flat, prob_long)
    valid_until = ts + pd.Timedelta(seconds=int(horizon) * _timeframe_seconds(timeframe))

    return ModelPrediction(
        prediction_id=f"pred_{uuid.uuid4().hex}",
        model_id=model_id,
        symbol=symbol.upper(),
        timeframe=timeframe,
        timestamp_utc=ts.isoformat(),
        direction=direction,
        confidence=float(confidence),
        prob_up=float(prob_long),
        prob_down=float(prob_short),
        prob_flat=float(prob_flat),
        expected_return_pct=_float(_get("expected_return_pct"), 0.0),
        expected_move_pct=_float(_get("expected_move_pct"), abs(_float(_get("expected_return_pct"), 0.0))),
        expected_adverse_move_pct=_float(_get("expected_adverse_move_pct"), 0.0005),
        q05_return_pct=_float(_get("q05_return_pct"), 0.0),
        q25_return_pct=_float(_get("q25_return_pct"), 0.0),
        q50_return_pct=_float(_get("q50_return_pct"), 0.0),
        q75_return_pct=_float(_get("q75_return_pct"), 0.0),
        q95_return_pct=_float(_get("q95_return_pct"), 0.0),
        expected_max_favorable_excursion_pct=_float(_get("expected_mfe_pct"), 0.0005),
        expected_max_adverse_excursion_pct=_float(_get("expected_mae_pct"), 0.0005),
        horizon_bars=int(horizon),
        signal_valid_until_utc=valid_until.isoformat(),
        raw_prediction_json={"native_from_persisted_fields": True, "legacy_signal": signal},
    )


def build_structured_prediction(
    *,
    model_id: str,
    symbol: str,
    timeframe: str,
    timestamp_utc,
    close_price: float,
    prob_short: float,
    prob_flat: float,
    prob_long: float,
    latest_features: dict | None = None,
    feature_frame=None,
    native_models: dict | None = None,
    horizon_bars: int = DEFAULT_SIGNAL_HORIZON_BARS,
    raw: dict | None = None,
) -> ModelPrediction:
    """Native-first dispatcher: use trained return-distribution models when available,
    otherwise fall back to the classifier-derived synthetic distribution."""
    if native_models and feature_frame is not None:
        try:
            return build_prediction_from_native(
                model_id=model_id,
                symbol=symbol,
                timeframe=timeframe,
                timestamp_utc=timestamp_utc,
                native_models=native_models,
                feature_frame=feature_frame,
                prob_short=prob_short,
                prob_flat=prob_flat,
                prob_long=prob_long,
                horizon_bars=native_models.get("horizon_bars", horizon_bars),
                raw=raw,
            )
        except Exception as exc:  # pragma: no cover - defensive; never break the loop on native failure
            raw = {**(raw or {}), "native_model_error": str(exc)}
    return build_prediction_from_probabilities(
        model_id=model_id,
        symbol=symbol,
        timeframe=timeframe,
        timestamp_utc=timestamp_utc,
        close_price=close_price,
        prob_short=prob_short,
        prob_flat=prob_flat,
        prob_long=prob_long,
        latest_features=latest_features,
        horizon_bars=horizon_bars,
        raw=raw,
    )


def _timeframe_seconds(timeframe: str) -> int:
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1] or "1")
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    return 3600


def persist_prediction(prediction: ModelPrediction) -> str:
    init_research_tables()
    p = prediction.to_dict()
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {MODEL_PREDICTIONS_TABLE} (
                prediction_id, model_id, symbol, timeframe, timestamp_utc, direction, confidence,
                prob_up, prob_down, prob_flat, expected_return_pct, expected_move_pct,
                expected_adverse_move_pct, q05_return_pct, q25_return_pct, q50_return_pct,
                q75_return_pct, q95_return_pct, expected_max_favorable_excursion_pct,
                expected_max_adverse_excursion_pct, horizon_bars, signal_valid_until_utc,
                raw_prediction_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                p["prediction_id"], p["model_id"], p["symbol"], p["timeframe"], p["timestamp_utc"], p["direction"],
                p["confidence"], p["prob_up"], p["prob_down"], p["prob_flat"], p["expected_return_pct"],
                p["expected_move_pct"], p["expected_adverse_move_pct"], p["q05_return_pct"], p["q25_return_pct"],
                p["q50_return_pct"], p["q75_return_pct"], p["q95_return_pct"],
                p["expected_max_favorable_excursion_pct"], p["expected_max_adverse_excursion_pct"], p["horizon_bars"],
                p["signal_valid_until_utc"], json.dumps(p["raw_prediction_json"], ensure_ascii=True, sort_keys=True),
                pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
        conn.commit()
    return prediction.prediction_id


def prediction_is_tradeable(prediction: ModelPrediction) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if prediction.direction != "LONG":
        reasons.append("prediction_direction_not_long_entry")
    if prediction.confidence < float(TRADE_PROPOSAL_MIN_CONFIDENCE):
        reasons.append("confidence_below_proposal_threshold")
    if prediction.expected_return_pct <= float(TRADE_PROPOSAL_MIN_EXPECTED_RETURN_PCT):
        reasons.append("expected_return_after_costs_not_positive_enough")
    if prediction.expected_adverse_move_pct <= 0:
        reasons.append("missing_expected_adverse_move")
    return not reasons, reasons
