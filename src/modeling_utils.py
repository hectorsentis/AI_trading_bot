from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


CLASS_SHORT = 0
CLASS_FLAT = 1
CLASS_LONG = 2

CLASS_TO_NAME = {
    CLASS_SHORT: "SHORT",
    CLASS_FLAT: "FLAT",
    CLASS_LONG: "LONG",
}

CLASS_TO_POSITION = {
    CLASS_SHORT: -1,
    CLASS_FLAT: 0,
    CLASS_LONG: 1,
}

POSITION_TO_NAME = {
    -1: "SHORT",
    0: "FLAT",
    1: "LONG",
}


@dataclass
class SplitMetrics:
    accuracy: float
    f1_macro: float
    strategy_return: float
    buy_hold_return: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    trade_count: int
    active_ratio: float
    periods: int


def timeframe_periods_per_year(timeframe: str) -> float:
    unit = timeframe[-1]
    value = int(timeframe[:-1])

    if unit == "m":
        return (60 / value) * 24 * 365
    if unit == "h":
        return (24 / value) * 365
    if unit == "d":
        return 365 / value
    if unit == "w":
        return 52 / value
    if unit == "M":
        return 12 / value

    # fallback conservative
    return 365


def probabilities_to_signal(
    probas: np.ndarray,
    short_threshold: float,
    long_threshold: float,
) -> np.ndarray:
    p_short = probas[:, CLASS_SHORT]
    p_long = probas[:, CLASS_LONG]

    signal = np.zeros(len(probas), dtype=int)
    long_mask = (p_long >= long_threshold) & (p_long > p_short)
    short_mask = (p_short >= short_threshold) & (p_short > p_long)

    signal[long_mask] = 1
    signal[short_mask] = -1
    return signal


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro")),
    )


def _to_float(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def compute_economic_metrics(
    frame: pd.DataFrame,
    timeframe: str,
    cost_per_trade: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, SplitMetrics]:
    required = {"symbol", "datetime_utc", "signal_position", "fwd_return_1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for economic metrics: {sorted(missing)}")

    df = frame.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime_utc"]).sort_values(["symbol", "datetime_utc"]).reset_index(drop=True)

    df["fwd_return_1"] = pd.to_numeric(df["fwd_return_1"], errors="coerce").fillna(0.0)
    df["signal_position"] = pd.to_numeric(df["signal_position"], errors="coerce").fillna(0).astype(int)

    df["prev_signal"] = df.groupby("symbol")["signal_position"].shift(1).fillna(0).astype(int)
    df["turnover"] = (df["signal_position"] - df["prev_signal"]).abs()
    df["transaction_cost"] = df["turnover"] * float(cost_per_trade)
    df["gross_return"] = df["signal_position"] * df["fwd_return_1"]
    df["strategy_return"] = df["gross_return"] - df["transaction_cost"]

    portfolio = (
        df.groupby("datetime_utc", as_index=False)
        .agg(
            strategy_return=("strategy_return", "mean"),
            market_return=("fwd_return_1", "mean"),
            active_positions=("signal_position", lambda s: float((s != 0).mean())),
            turnover=("turnover", "sum"),
        )
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )

    portfolio["strategy_equity"] = (1.0 + portfolio["strategy_return"]).cumprod()
    portfolio["market_equity"] = (1.0 + portfolio["market_return"]).cumprod()
    running_max = portfolio["strategy_equity"].cummax().replace(0, np.nan)
    portfolio["drawdown"] = (portfolio["strategy_equity"] / running_max) - 1.0

    periods = len(portfolio)
    strategy_total = _to_float(portfolio["strategy_equity"].iloc[-1] - 1.0) if periods else 0.0
    market_total = _to_float(portfolio["market_equity"].iloc[-1] - 1.0) if periods else 0.0
    max_drawdown = _to_float(portfolio["drawdown"].min()) if periods else 0.0

    mu = portfolio["strategy_return"].mean() if periods else 0.0
    sigma = portfolio["strategy_return"].std(ddof=0) if periods else 0.0
    annual_factor = np.sqrt(timeframe_periods_per_year(timeframe))
    sharpe = float((mu / sigma) * annual_factor) if sigma and not pd.isna(sigma) else 0.0

    gains = portfolio.loc[portfolio["strategy_return"] > 0, "strategy_return"].sum()
    losses = portfolio.loc[portfolio["strategy_return"] < 0, "strategy_return"].sum()
    profit_factor = float(gains / abs(losses)) if losses < 0 else float("inf")

    metrics = SplitMetrics(
        accuracy=0.0,
        f1_macro=0.0,
        strategy_return=strategy_total,
        buy_hold_return=market_total,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        trade_count=int((df["turnover"] > 0).sum()),
        active_ratio=float((df["signal_position"] != 0).mean()) if len(df) else 0.0,
        periods=periods,
    )
    return df, portfolio, metrics
