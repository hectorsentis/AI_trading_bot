"""Phase B closure: native predictions flow into validation/backtest acceptance.

Covers the fields-based prediction builder used by the historical simulator, and the
persistence round-trip (validate_model persists native fields -> backtest OOS reloads them),
so the acceptance gate evaluates the same native, cost-adjusted predictions as paper.
"""
from __future__ import annotations

import pandas as pd

import config
from db_utils import init_research_tables, save_validation_predictions
from backtest import load_oos_predictions
from prediction_engine import build_prediction_from_row_fields


def _native_row(**overrides):
    row = {
        "has_native_prediction": 1,
        "expected_return_pct": 0.01,
        "expected_move_pct": 0.02,
        "expected_adverse_move_pct": 0.015,
        "q05_return_pct": -0.03,
        "q25_return_pct": -0.005,
        "q50_return_pct": 0.008,
        "q75_return_pct": 0.02,
        "q95_return_pct": 0.04,
        "expected_mfe_pct": 0.05,
        "expected_mae_pct": 0.03,
        "native_horizon_bars": 6,
        "prob_short": 0.1,
        "prob_flat": 0.3,
        "prob_long": 0.6,
    }
    row.update(overrides)
    return row


def test_build_prediction_from_row_fields_present():
    pred = build_prediction_from_row_fields(
        _native_row(), model_id="m", symbol="btcusdt", timeframe="1h",
        timestamp_utc="2026-01-01T00:00:00+00:00",
    )
    assert pred is not None
    assert pred.direction == "LONG"
    assert pred.expected_return_pct == 0.01
    assert pred.q95_return_pct == 0.04
    assert pred.expected_max_favorable_excursion_pct == 0.05
    assert pred.horizon_bars == 6
    assert pred.raw_prediction_json["native_from_persisted_fields"] is True


def test_build_prediction_from_row_fields_absent_falls_back_to_none():
    assert build_prediction_from_row_fields(
        {"has_native_prediction": 0}, model_id="m", symbol="b", timeframe="1h",
        timestamp_utc="2026-01-01T00:00:00+00:00",
    ) is None
    # Missing native fields entirely -> None (caller uses the derived path).
    assert build_prediction_from_row_fields(
        {}, model_id="m", symbol="b", timeframe="1h",
        timestamp_utc="2026-01-01T00:00:00+00:00",
    ) is None


def test_validation_predictions_native_persistence_roundtrip():
    init_research_tables()
    now = pd.Timestamp.now(tz="UTC").isoformat()
    rows = [
        _native_row(
            model_id="mtest", symbol="BTCUSDT", timeframe="1h",
            datetime_utc="2026-01-01T00:00:00+00:00", y_true=2, y_pred=2,
            signal_position=1, fold_id=1, created_at_utc=now, has_native_prediction=True,
        )
    ]
    n = save_validation_predictions(rows=rows, validation_run_id="run_phaseb_test", replace_run=True)
    assert n == 1

    df, run_id = load_oos_predictions(
        model_id="mtest", timeframe="1h", symbols=["BTCUSDT"],
        validation_run_id="run_phaseb_test", start_date=None, end_date=None,
    )
    assert run_id == "run_phaseb_test"
    assert len(df) == 1
    assert int(df.loc[0, "has_native_prediction"]) == 1
    assert abs(float(df.loc[0, "expected_return_pct"]) - 0.01) < 1e-9
    assert abs(float(df.loc[0, "q95_return_pct"]) - 0.04) < 1e-9
    assert int(df.loc[0, "native_horizon_bars"]) == 6

    # The reloaded OOS row must build a native prediction (the acceptance gate consumes this).
    pred = build_prediction_from_row_fields(
        df.iloc[0], model_id="mtest", symbol="BTCUSDT", timeframe="1h",
        timestamp_utc=df.loc[0, "datetime_utc"],
    )
    assert pred is not None and pred.expected_return_pct == 0.01
