"""Phase C: expanded feature set (volatility/regime/momentum/microstructure/cross-asset).

Verifies the full feature contract is produced, microstructure + cross-asset fall back to neutral
values when their inputs are absent, and features are strictly leakage-safe (a past row's features
never change when a future bar changes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import compute_features
from config import FEATURE_COLUMNS


def _ohlcv(n=320, seed=1, with_taker=True):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    vol = rng.uniform(50, 150, n)
    df = pd.DataFrame({
        "symbol": "ETHUSDT", "timeframe": "1h", "datetime_utc": t,
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * 1.005, "low": close * 0.995, "close": close, "volume": vol,
    })
    if with_taker:
        df["quote_asset_volume"] = vol * close
        df["number_of_trades"] = rng.integers(100, 500, n)
        df["taker_buy_base_volume"] = vol * rng.uniform(0.3, 0.7, n)
        df["taker_buy_quote_volume"] = vol * close * 0.5
    return df


def _context(n=320, seed=2):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"datetime_utc": t, "ref_close": 100 * np.cumprod(1 + rng.normal(0.0002, 0.012, n))})


def test_full_feature_contract_present():
    out = compute_features(_ohlcv(), context=_context())
    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    assert missing == []
    # The most recent rows must be fully populated (no trailing NaN holes after warmup).
    assert int(out[FEATURE_COLUMNS].tail(30).dropna().shape[0]) == 30


def test_microstructure_neutral_fallback_without_taker():
    out = compute_features(_ohlcv(with_taker=False), context=None)
    assert (out["taker_buy_ratio"] == 0.5).all()
    assert (out["taker_imbalance"] == 0.0).all()
    assert (out["taker_imbalance_zscore_20"] == 0.0).all()
    assert (out["avg_trade_size_zscore_20"] == 0.0).all()


def test_cross_asset_neutral_without_context():
    out = compute_features(_ohlcv(), context=None)
    assert (out["btc_ret_24"] == 0.0).all()
    assert (out["rel_strength_vs_btc_24"] == 0.0).all()
    assert (out["corr_btc_50"] == 0.0).all()
    assert (out["beta_btc_50"] == 1.0).all()


def test_features_are_leakage_safe():
    df = _ohlcv()
    ctx = _context()
    base = compute_features(df, context=ctx)
    # Mutate only the final (future) bar.
    df2 = df.copy()
    df2.loc[len(df2) - 1, "close"] *= 1.5
    df2.loc[len(df2) - 1, "high"] *= 1.5
    mutated = compute_features(df2, context=ctx)
    row = 250
    a = base[FEATURE_COLUMNS].iloc[row].fillna(0).to_numpy()
    b = mutated[FEATURE_COLUMNS].iloc[row].fillna(0).to_numpy()
    assert np.allclose(a, b)
