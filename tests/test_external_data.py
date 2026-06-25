"""Phase C: external data storage + leakage-safe as-of join.

No network: exercises the storage layer and the backward as-of merge that guarantees a row at
time t only sees external values known at or before t.
"""
from __future__ import annotations

import pandas as pd

from external_data import (
    init_external_data_table,
    save_external_metric,
    load_external_metric,
    attach_external_features_asof,
)


def test_external_metric_roundtrip():
    init_external_data_table()
    rows = [
        {"datetime_utc": "2026-01-01T00:00:00+00:00", "value": 30.0},
        {"datetime_utc": "2026-01-02T00:00:00+00:00", "value": 55.0},
    ]
    n = save_external_metric("alternative.me", "fear_greed", rows)
    assert n == 2
    df = load_external_metric("alternative.me", "fear_greed")
    assert len(df) == 2
    assert df["value"].tolist() == [30.0, 55.0]
    # Idempotent upsert (same keys) does not duplicate.
    save_external_metric("alternative.me", "fear_greed", rows)
    assert len(load_external_metric("alternative.me", "fear_greed")) == 2


def test_attach_external_features_asof_is_leakage_safe():
    # Feature rows hourly; external metric known only at 02:00 and 05:00.
    feat = pd.DataFrame({"datetime_utc": pd.date_range("2026-01-01 00:00", periods=8, freq="h", tz="UTC")})
    metric = pd.DataFrame({
        "datetime_utc": pd.to_datetime(["2026-01-01 02:00", "2026-01-01 05:00"], utc=True),
        "value": [10.0, 20.0],
    })
    out = attach_external_features_asof(feat, {"funding": metric})

    vals = out.set_index("datetime_utc")["funding"]
    # Before the first external observation -> NaN (no future peeking).
    assert pd.isna(vals.loc["2026-01-01 00:00+00:00"])
    assert pd.isna(vals.loc["2026-01-01 01:00+00:00"])
    # From 02:00 until just before 05:00 -> 10.0 (last known value).
    assert vals.loc["2026-01-01 02:00+00:00"] == 10.0
    assert vals.loc["2026-01-01 04:00+00:00"] == 10.0
    # From 05:00 onward -> 20.0.
    assert vals.loc["2026-01-01 05:00+00:00"] == 20.0
    assert vals.loc["2026-01-01 07:00+00:00"] == 20.0


def test_attach_handles_empty_metric():
    feat = pd.DataFrame({"datetime_utc": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")})
    out = attach_external_features_asof(feat, {"oi": pd.DataFrame(columns=["datetime_utc", "value"])})
    assert out["oi"].isna().all()
