"""Phase B: native return-distribution prediction path.

Uses lightweight fake regressors (no LightGBM training) so the test is fast and deterministic.
Verifies the native path produces real expected-return/quantile/MFE-MAE fields, enforces
monotonic quantiles even when the underlying regressors cross, and that the dispatcher falls
back to the derived path when no native models are present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prediction_engine import (
    build_prediction_from_native,
    build_structured_prediction,
    _fee_slip,
)

FEATURE_COLS = ["f1", "f2"]


class _FakeRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X):  # noqa: N803 - sklearn-style signature
        return np.array([self.value] * len(X))


def _native_models(quantile_values: dict[float, float]) -> dict:
    return {
        "horizon_bars": 6,
        "feature_columns": FEATURE_COLS,
        "quantile_levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "expected_return": _FakeRegressor(0.01),
        "quantiles": {f"{a:.2f}": _FakeRegressor(v) for a, v in quantile_values.items()},
        "mfe": _FakeRegressor(0.03),
        "mae": _FakeRegressor(-0.025),
        "trained_rows": 1234,
    }


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame([{"f1": 1.0, "f2": 2.0}])


def test_native_prediction_uses_real_fields_and_is_monotonic():
    # Deliberately crossing raw quantiles (q05 > q95) to prove monotonic enforcement.
    native = _native_models({0.05: 0.04, 0.25: -0.005, 0.5: 0.008, 0.75: 0.02, 0.95: -0.02})
    pred = build_prediction_from_native(
        model_id="m1", symbol="btcusdt", timeframe="1h",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        native_models=native, feature_frame=_feature_frame(),
        prob_short=0.1, prob_flat=0.3, prob_long=0.6,
    )
    assert pred.direction == "LONG"
    assert pred.raw_prediction_json.get("native_models") is True
    # Cost-adjusted expected return = raw(0.01) - round-trip costs.
    assert abs(pred.expected_return_pct - (0.01 - _fee_slip())) < 1e-9
    # Quantiles monotonic non-decreasing despite crossing inputs.
    qs = [pred.q05_return_pct, pred.q25_return_pct, pred.q50_return_pct, pred.q75_return_pct, pred.q95_return_pct]
    assert qs == sorted(qs)
    # MFE positive, adverse magnitude positive.
    assert pred.expected_max_favorable_excursion_pct > 0
    assert pred.expected_max_adverse_excursion_pct > 0


def test_dispatcher_prefers_native_then_falls_back():
    native = _native_models({0.05: -0.03, 0.25: -0.005, 0.5: 0.008, 0.75: 0.02, 0.95: 0.04})
    common = dict(
        model_id="m1", symbol="BTCUSDT", timeframe="1h",
        timestamp_utc="2026-01-01T00:00:00+00:00", close_price=100_000.0,
        prob_short=0.1, prob_flat=0.3, prob_long=0.6,
        latest_features={"atr_pct": 0.01, "volatility_20": 0.02},
    )
    native_pred = build_structured_prediction(native_models=native, feature_frame=_feature_frame(), **common)
    assert native_pred.raw_prediction_json.get("native_models") is True

    derived_pred = build_structured_prediction(native_models=None, feature_frame=None, **common)
    assert "adapter_note" in derived_pred.raw_prediction_json  # derived/synthetic path marker
