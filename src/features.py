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


def _load_prices(symbol: str, timeframe: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT symbol, timeframe, datetime_utc, open, high, low, close, volume
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

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime_utc", "open", "high", "low", "close", "volume"]).copy()
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_features(prices_df: pd.DataFrame) -> pd.DataFrame:
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
