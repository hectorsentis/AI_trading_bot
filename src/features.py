import argparse
import sqlite3
from typing import Iterable

import numpy as np
import pandas as pd

from config import DB_FILE, PRICES_TABLE, SYMBOLS, TIMEFRAME, LOOKAHEAD_BARS, FEATURE_COLUMNS


def parse_args():
    parser = argparse.ArgumentParser(description="Compute market features from prices in SQLite.")
    parser.add_argument("--symbols", nargs="*", default=None, help="Symbols to process. Default: config SYMBOLS")
    parser.add_argument("--timeframe", default=TIMEFRAME, help="Timeframe to process")
    parser.add_argument("--head", type=int, default=5, help="Rows to print from each symbol")
    return parser.parse_args()


# Optional microstructure columns. Present in modern Binance ingests; absent in legacy/synthetic
# data, in which case the microstructure features fall back to neutral values.
TAKER_COLUMNS = ["quote_asset_volume", "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]


def _load_prices(symbol: str, timeframe: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT symbol, timeframe, datetime_utc, open, high, low, close, volume,
                   quote_asset_volume, number_of_trades, taker_buy_base_volume, taker_buy_quote_volume
            FROM {PRICES_TABLE}
            WHERE symbol = ? AND timeframe = ?
            ORDER BY datetime_utc
            """,
            conn,
            params=(symbol, timeframe),
        )
    finally:
        conn.close()

    if df.empty:
        return df

    for col in ["open", "high", "low", "close", "volume", *TAKER_COLUMNS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime_utc", "open", "high", "low", "close", "volume"]).copy()
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df


def _attach_cross_asset_features(df: pd.DataFrame, context: pd.DataFrame | None) -> pd.DataFrame:
    """Leakage-safe cross-asset (BTC) context, aligned by backward as-of merge.

    ``context`` is a frame with ``datetime_utc`` and ``ref_close`` (reference-asset close, e.g.
    BTCUSDT). Each row uses only the reference close known at or before its own timestamp. When
    no context is supplied the features fall back to neutral constants (harmless to the model).
    """
    if context is not None and not getattr(context, "empty", True) and "ref_close" in context.columns:
        ctx = context[["datetime_utc", "ref_close"]].dropna().sort_values("datetime_utc")
    else:
        ctx = None

    if ctx is None or ctx.empty:
        df["btc_ret_24"] = 0.0
        df["rel_strength_vs_btc_24"] = 0.0
        df["corr_btc_50"] = 0.0
        df["beta_btc_50"] = 1.0
        return df

    merged = pd.merge_asof(
        df[["datetime_utc"]].sort_values("datetime_utc"),
        ctx,
        on="datetime_utc",
        direction="backward",
    )
    ref_close = pd.to_numeric(merged["ref_close"], errors="coerce").reset_index(drop=True)
    sym_close = df["close"].reset_index(drop=True)
    sym_ret = sym_close.pct_change()
    ref_ret = ref_close.pct_change()

    df["btc_ret_24"] = ref_close.pct_change(24).to_numpy()
    df["rel_strength_vs_btc_24"] = (sym_close.pct_change(24) - ref_close.pct_change(24)).to_numpy()
    df["corr_btc_50"] = sym_ret.rolling(50, min_periods=50).corr(ref_ret).to_numpy()
    cov = sym_ret.rolling(50, min_periods=50).cov(ref_ret)
    var = ref_ret.rolling(50, min_periods=50).var()
    df["beta_btc_50"] = (cov / var.replace(0, np.nan)).fillna(1.0).to_numpy()
    return df


# Neutral values used when a higher timeframe's data is unavailable.
_HTF_NEUTRAL = {"rsi_14": 50.0, "trend_strength": 0.0, "volatility": 0.0}


def _attach_higher_timeframe_features(df: pd.DataFrame, higher_timeframes: dict | None) -> pd.DataFrame:
    """Attach context from up to two higher timeframes (htf1, htf2) via leakage-safe as-of merge.

    ``higher_timeframes`` maps the slot label (``"htf1"``/``"htf2"``) to a frame with an
    ``available_at`` column (the higher-TF candle's CLOSE time) plus the slot's feature columns
    (``{slot}_rsi_14``, ``{slot}_trend_strength``, ``{slot}_volatility``). Each base row only sees
    higher-TF candles that have already closed at or before its timestamp. Missing slots fall back
    to neutral constants.
    """
    higher_timeframes = higher_timeframes or {}
    for slot in ("htf1", "htf2"):
        cols = [f"{slot}_rsi_14", f"{slot}_trend_strength", f"{slot}_volatility"]
        frame = higher_timeframes.get(slot)
        if frame is None or getattr(frame, "empty", True) or "available_at" not in getattr(frame, "columns", []):
            for c in cols:
                df[c] = _HTF_NEUTRAL[c.split(f"{slot}_", 1)[1]]
            continue
        m = frame.dropna(subset=["available_at"]).sort_values("available_at")
        merged = pd.merge_asof(
            df[["datetime_utc"]].sort_values("datetime_utc"),
            m,
            left_on="datetime_utc",
            right_on="available_at",
            direction="backward",
        )
        for c in cols:
            df[c] = merged[c].to_numpy() if c in merged.columns else _HTF_NEUTRAL[c.split(f"{slot}_", 1)[1]]
    return df


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_features(
    prices_df: pd.DataFrame,
    context: pd.DataFrame | None = None,
    higher_timeframes: dict | None = None,
) -> pd.DataFrame:
    if prices_df.empty:
        return prices_df

    df = prices_df.copy()
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    returns_1 = close.pct_change(1)
    returns_3 = close.pct_change(3)
    returns_6 = close.pct_change(6)
    returns_12 = close.pct_change(12)

    df["ret_1"] = returns_1
    df["ret_3"] = returns_3
    df["ret_6"] = returns_6
    df["ret_12"] = returns_12

    df["hl_range"] = (high - low) / close.replace(0, np.nan)
    df["oc_range"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.rolling(14, min_periods=14).mean()

    df["volatility_10"] = returns_1.rolling(10, min_periods=10).std()
    df["volatility_20"] = returns_1.rolling(20, min_periods=20).std()

    vol_ma_10 = volume.rolling(10, min_periods=10).mean()
    vol_ma_20 = volume.rolling(20, min_periods=20).mean()
    vol_std_20 = volume.rolling(20, min_periods=20).std()

    df["vol_ratio_10"] = volume / vol_ma_10.replace(0, np.nan)
    df["vol_zscore_20"] = (volume - vol_ma_20) / vol_std_20.replace(0, np.nan)

    ma_5 = close.rolling(5, min_periods=5).mean()
    ma_10 = close.rolling(10, min_periods=10).mean()
    ma_20 = close.rolling(20, min_periods=20).mean()
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()

    df["dist_ma_5"] = (close / ma_5) - 1.0
    df["dist_ma_10"] = (close / ma_10) - 1.0
    df["dist_ma_20"] = (close / ma_20) - 1.0
    df["atr_pct"] = df["atr_14"] / close.replace(0, np.nan)
    df["ema_12_dist"] = (close / ema_12) - 1.0
    df["ema_26_dist"] = (close / ema_26) - 1.0
    df["ema_12_26_spread"] = (ema_12 / ema_26.replace(0, np.nan)) - 1.0
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["slope_ma_5"] = ma_5.pct_change()
    df["slope_ma_10"] = ma_10.pct_change()

    rolling_max_20 = high.rolling(20, min_periods=20).max()
    rolling_min_20 = low.rolling(20, min_periods=20).min()
    df["rolling_max_dist_20"] = (close / rolling_max_20) - 1.0
    df["rolling_min_dist_20"] = (close / rolling_min_20) - 1.0

    df["rsi_14"] = _compute_rsi(close, period=14)
    ret_mean_20 = returns_1.rolling(20, min_periods=20).mean()
    ret_std_20 = returns_1.rolling(20, min_periods=20).std()
    df["ret_zscore_20"] = (returns_1 - ret_mean_20) / ret_std_20.replace(0, np.nan)
    df["volatility_ratio_10_20"] = df["volatility_10"] / df["volatility_20"].replace(0, np.nan)

    bb_std_20 = close.rolling(20, min_periods=20).std()
    bb_upper_20 = ma_20 + (2.0 * bb_std_20)
    bb_lower_20 = ma_20 - (2.0 * bb_std_20)
    bb_width = bb_upper_20 - bb_lower_20
    df["bb_width_20"] = bb_width / ma_20.replace(0, np.nan)
    df["bb_percent_b_20"] = (close - bb_lower_20) / bb_width.replace(0, np.nan)

    low_14 = low.rolling(14, min_periods=14).min()
    high_14 = high.rolling(14, min_periods=14).max()
    stoch_range = high_14 - low_14
    df["stoch_k_14"] = (close - low_14) / stoch_range.replace(0, np.nan)
    df["stoch_d_3"] = df["stoch_k_14"].rolling(3, min_periods=3).mean()

    df["volume_change_1"] = volume.pct_change(1)
    df["volume_trend_5"] = volume.rolling(5, min_periods=5).mean() / vol_ma_20.replace(0, np.nan) - 1.0

    candle_range = (high - low).replace(0, np.nan)
    body = (close - df["open"]).abs()
    upper_wick = high - pd.concat([df["open"], close], axis=1).max(axis=1)
    lower_wick = pd.concat([df["open"], close], axis=1).min(axis=1) - low

    df["body_ratio"] = body / candle_range
    df["upper_wick_ratio"] = upper_wick / candle_range
    df["lower_wick_ratio"] = lower_wick / candle_range
    range_mean_20 = candle_range.rolling(20, min_periods=20).mean()
    range_std_20 = candle_range.rolling(20, min_periods=20).std()
    df["range_zscore_20"] = (candle_range - range_mean_20) / range_std_20.replace(0, np.nan)
    df["trend_strength_20"] = (close - ma_20) / (20.0 * df["atr_14"].replace(0, np.nan))
    df["consecutive_up_3"] = (
        (close > close.shift(1))
        & (close.shift(1) > close.shift(2))
        & (close.shift(2) > close.shift(3))
    ).astype(float)
    df["consecutive_down_3"] = (
        (close < close.shift(1))
        & (close.shift(1) < close.shift(2))
        & (close.shift(2) < close.shift(3))
    ).astype(float)

    # Candlestick / chart-pattern proxy features. They use only current and past bars.
    df["is_doji"] = (df["body_ratio"] <= 0.10).astype(float)
    df["is_hammer"] = (
        (df["lower_wick_ratio"] >= 0.55)
        & (df["upper_wick_ratio"] <= 0.20)
        & (df["body_ratio"] <= 0.35)
    ).astype(float)
    df["is_shooting_star"] = (
        (df["upper_wick_ratio"] >= 0.55)
        & (df["lower_wick_ratio"] <= 0.20)
        & (df["body_ratio"] <= 0.35)
    ).astype(float)

    prev_open = df["open"].shift(1)
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    df["bullish_engulfing"] = (
        (prev_close < prev_open)
        & (close > df["open"])
        & (df["open"] <= prev_close)
        & (close >= prev_open)
    ).astype(float)
    df["bearish_engulfing"] = (
        (prev_close > prev_open)
        & (close < df["open"])
        & (df["open"] >= prev_close)
        & (close <= prev_open)
    ).astype(float)
    df["inside_bar"] = ((high < prev_high) & (low > prev_low)).astype(float)
    df["outside_bar"] = ((high > prev_high) & (low < prev_low)).astype(float)

    prior_high_20 = high.shift(1).rolling(20, min_periods=20).max()
    prior_low_20 = low.shift(1).rolling(20, min_periods=20).min()
    df["breakout_20"] = (close > prior_high_20).astype(float)
    df["breakdown_20"] = (close < prior_low_20).astype(float)
    df["ma_cross_5_20"] = ((ma_5 > ma_20) & (ma_5.shift(1) <= ma_20.shift(1))).astype(float)

    prior_high_50 = high.shift(1).rolling(50, min_periods=50).max()
    prior_low_50 = low.shift(1).rolling(50, min_periods=50).min()
    atr_pct = (df["atr_14"] / close.replace(0, np.nan)).replace(0, np.nan)
    near_prior_high = ((close / prior_high_50.replace(0, np.nan)) - 1.0).abs()
    near_prior_low = ((close / prior_low_50.replace(0, np.nan)) - 1.0).abs()
    df["double_top_proxy"] = (
        (near_prior_high <= (1.5 * atr_pct))
        & (df["upper_wick_ratio"] > df["lower_wick_ratio"])
        & (df["rsi_14"] >= 60)
    ).astype(float)
    df["double_bottom_proxy"] = (
        (near_prior_low <= (1.5 * atr_pct))
        & (df["lower_wick_ratio"] > df["upper_wick_ratio"])
        & (df["rsi_14"] <= 40)
    ).astype(float)

    hour = df["datetime_utc"].dt.hour.astype(float)
    df["hour_sin"] = np.sin((2.0 * np.pi * hour) / 24.0)
    df["hour_cos"] = np.cos((2.0 * np.pi * hour) / 24.0)

    # ===== Phase C (v4) expansion: volatility / regime / momentum =====
    ma_50 = close.rolling(50, min_periods=50).mean()
    ma_200 = close.rolling(200, min_periods=200).mean()

    df["ret_24"] = close.pct_change(24)
    df["roc_10"] = close.pct_change(10)
    df["rsi_7"] = _compute_rsi(close, period=7)

    df["volatility_50"] = returns_1.rolling(50, min_periods=50).std()
    df["volatility_ratio_20_50"] = df["volatility_20"] / df["volatility_50"].replace(0, np.nan)
    downside = returns_1.where(returns_1 < 0, 0.0)
    df["downside_volatility_20"] = downside.rolling(20, min_periods=20).std()
    vol20_mean_100 = df["volatility_20"].rolling(100, min_periods=100).mean()
    vol20_std_100 = df["volatility_20"].rolling(100, min_periods=100).std()
    df["volatility_regime_score"] = (df["volatility_20"] - vol20_mean_100) / vol20_std_100.replace(0, np.nan)

    df["dist_ma_50"] = (close / ma_50) - 1.0
    df["dist_ma_200"] = (close / ma_200) - 1.0
    df["price_above_sma_50"] = (close > ma_50).astype(float)
    df["trend_strength_50"] = (close - ma_50) / (50.0 * df["atr_14"].replace(0, np.nan))

    rolling_max_close_50 = close.rolling(50, min_periods=50).max()
    df["rolling_drawdown_50"] = (close / rolling_max_close_50.replace(0, np.nan)) - 1.0
    rolling_high_50 = high.rolling(50, min_periods=50).max()
    df["dist_from_high_50"] = (close / rolling_high_50.replace(0, np.nan)) - 1.0

    # ===== Phase C (v4): microstructure from Binance taker data (current bar only) =====
    if "taker_buy_base_volume" in df.columns:
        taker_buy_base = pd.to_numeric(df["taker_buy_base_volume"], errors="coerce")
        df["taker_buy_ratio"] = (taker_buy_base / volume.replace(0, np.nan)).clip(0.0, 1.0).fillna(0.5)
        df["taker_imbalance"] = ((2.0 * taker_buy_base - volume) / volume.replace(0, np.nan)).fillna(0.0)
        ti_mean_20 = df["taker_imbalance"].rolling(20, min_periods=20).mean()
        ti_std_20 = df["taker_imbalance"].rolling(20, min_periods=20).std()
        df["taker_imbalance_zscore_20"] = (df["taker_imbalance"] - ti_mean_20) / ti_std_20.replace(0, np.nan)
    else:
        df["taker_buy_ratio"] = 0.5
        df["taker_imbalance"] = 0.0
        df["taker_imbalance_zscore_20"] = 0.0

    if "quote_asset_volume" in df.columns and "number_of_trades" in df.columns:
        trades = pd.to_numeric(df["number_of_trades"], errors="coerce").replace(0, np.nan)
        avg_trade_size = pd.to_numeric(df["quote_asset_volume"], errors="coerce") / trades
        ats_mean_20 = avg_trade_size.rolling(20, min_periods=20).mean()
        ats_std_20 = avg_trade_size.rolling(20, min_periods=20).std()
        df["avg_trade_size_zscore_20"] = (avg_trade_size - ats_mean_20) / ats_std_20.replace(0, np.nan)
    else:
        df["avg_trade_size_zscore_20"] = 0.0

    # ===== Phase C (v4): cross-asset BTC context (leakage-safe as-of) =====
    df = _attach_cross_asset_features(df, context)

    # ===== Phase C (v5): multi-timeframe context (closed higher-TF candles only) =====
    df = _attach_higher_timeframe_features(df, higher_timeframes)

    df["fwd_return_1"] = close.shift(-1) / close - 1.0
    df["fwd_return_horizon"] = close.shift(-LOOKAHEAD_BARS) / close - 1.0

    # Keep the declared feature contract explicit.
    feature_missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if feature_missing:
        raise ValueError(f"Features missing from compute_features output: {feature_missing}")

    return df


def compute_features_for_symbols(symbols: Iterable[str], timeframe: str) -> dict[str, pd.DataFrame]:
    result = {}
    for symbol in symbols:
        prices = _load_prices(symbol, timeframe)
        if prices.empty:
            result[symbol] = prices
            continue
        result[symbol] = compute_features(prices)
    return result


def main():
    args = parse_args()
    symbols = args.symbols if args.symbols else SYMBOLS

    outputs = compute_features_for_symbols(symbols, args.timeframe)
    for symbol, df in outputs.items():
        if df.empty:
            print(f"[{symbol}] no rows available in prices table.")
            continue
        non_null_rows = int(df[FEATURE_COLUMNS].dropna().shape[0])
        print(f"\n[{symbol}] rows={len(df)} rows_with_full_features={non_null_rows}")
        cols = ["datetime_utc", "close"] + FEATURE_COLUMNS[:5]
        print(df[cols].tail(args.head).to_string(index=False))


if __name__ == "__main__":
    main()
