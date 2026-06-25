import argparse
from typing import Optional

import numpy as np
import pandas as pd

from config import LOOKAHEAD_BARS, TP_MULTIPLIER, SL_MULTIPLIER
from modeling_utils import CLASS_FLAT, CLASS_LONG, CLASS_SHORT, CLASS_TO_NAME, CLASS_TO_POSITION
from trade_protection import build_long_protection


def compute_native_regression_targets(
    frame: pd.DataFrame,
    horizon_bars: int = LOOKAHEAD_BARS,
    *,
    symbol_col: str = "symbol",
    time_col: str = "datetime_utc",
    close_col: str = "close",
) -> pd.DataFrame:
    """Leakage-safe forward targets for native regression/quantile/MFE-MAE models.

    For each row at time t (per symbol, in time order) over the next ``horizon_bars`` bars:
      - ``native_ret``  = close[t+h] / close[t] - 1            (horizon return)
      - ``native_mfe``  = max(close[t+1..t+h]) / close[t] - 1  (favorable excursion, >= 0)
      - ``native_mae``  = min(close[t+1..t+h]) / close[t] - 1  (adverse excursion, <= 0)

    Excursions use the forward close path (the features table does not persist intrabar
    high/low); this slightly understates true intrabar excursion but is leakage-safe and
    a sound first native target. Rows without a full forward horizon get NaN and must be
    dropped before training. Returns a frame keyed by ``[symbol, datetime_utc]`` plus the
    three target columns, so it can be merged back onto the training frame.
    """
    h = int(horizon_bars)
    parts: list[pd.DataFrame] = []
    for symbol, group in frame.sort_values(time_col).groupby(symbol_col):
        g = group.sort_values(time_col).reset_index(drop=True)
        close = g[close_col].astype(float).to_numpy()
        n = len(close)
        native_ret = np.full(n, np.nan)
        native_mfe = np.full(n, np.nan)
        native_mae = np.full(n, np.nan)
        for i in range(n):
            j = i + h
            base = close[i]
            if j >= n or not np.isfinite(base) or base <= 0:
                continue
            window = close[i + 1 : j + 1]
            if window.size == 0 or not np.isfinite(window).all() or not np.isfinite(close[j]):
                continue
            native_ret[i] = close[j] / base - 1.0
            native_mfe[i] = float(window.max()) / base - 1.0
            native_mae[i] = float(window.min()) / base - 1.0
        out = pd.DataFrame(
            {
                # Keep the Series (not .values) so tz-aware datetimes survive the round-trip
                # and merge cleanly against the tz-aware training frame.
                symbol_col: g[symbol_col],
                time_col: g[time_col],
                "native_ret": native_ret,
                "native_mfe": native_mfe,
                "native_mae": native_mae,
            }
        )
        parts.append(out)
    if not parts:
        return pd.DataFrame(columns=[symbol_col, time_col, "native_ret", "native_mfe", "native_mae"])
    return pd.concat(parts, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate triple-barrier labels for a feature dataframe.")
    parser.add_argument("--lookahead-bars", type=int, default=LOOKAHEAD_BARS)
    parser.add_argument("--tp-multiplier", type=float, default=TP_MULTIPLIER)
    parser.add_argument("--sl-multiplier", type=float, default=SL_MULTIPLIER)
    parser.add_argument("--head", type=int, default=10)
    return parser.parse_args()


def _resolve_outcome(
    high: float,
    low: float,
    tp_level: float,
    sl_level: float,
    direction: str,
) -> Optional[str]:
    if direction == "long":
        tp_hit = high >= tp_level
        sl_hit = low <= sl_level
    else:
        tp_hit = low <= tp_level
        sl_hit = high >= sl_level

    if tp_hit and sl_hit:
        return "ambiguous"
    if tp_hit:
        return "tp"
    if sl_hit:
        return "sl"
    return None


def generate_triple_barrier_labels(
    frame: pd.DataFrame,
    lookahead_bars: int = LOOKAHEAD_BARS,
    tp_multiplier: float = TP_MULTIPLIER,
    sl_multiplier: float = SL_MULTIPLIER,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    required = {"close", "high", "low", atr_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for labels: {sorted(missing)}")

    df = frame.copy().reset_index(drop=True)
    n = len(df)

    label_class = [pd.NA] * n
    label_name = [None] * n
    label_position = [pd.NA] * n
    label_take_profit_price = [pd.NA] * n
    label_stop_loss_price = [pd.NA] * n
    label_risk_reward = [pd.NA] * n

    for i in range(n):
        if i + lookahead_bars >= n:
            continue

        entry = float(df.loc[i, "close"])
        atr = float(df.loc[i, atr_col]) if pd.notna(df.loc[i, atr_col]) else float("nan")
        if not pd.notna(atr) or atr <= 0:
            label_class[i] = CLASS_FLAT
            label_name[i] = CLASS_TO_NAME[CLASS_FLAT]
            label_position[i] = CLASS_TO_POSITION[CLASS_FLAT]
            continue

        long_tp = entry + tp_multiplier * atr
        long_sl = entry - sl_multiplier * atr
        short_tp = entry - tp_multiplier * atr
        short_sl = entry + sl_multiplier * atr

        long_outcome = None
        short_outcome = None
        long_step = None
        short_step = None

        for step in range(1, lookahead_bars + 1):
            high = float(df.loc[i + step, "high"])
            low = float(df.loc[i + step, "low"])

            if long_outcome is None:
                event = _resolve_outcome(high, low, long_tp, long_sl, direction="long")
                if event is not None:
                    long_outcome = event
                    long_step = step

            if short_outcome is None:
                event = _resolve_outcome(high, low, short_tp, short_sl, direction="short")
                if event is not None:
                    short_outcome = event
                    short_step = step

            if long_outcome is not None and short_outcome is not None:
                break

        long_success = long_outcome == "tp"
        short_success = short_outcome == "tp"

        if long_success and not short_success:
            cls = CLASS_LONG
        elif short_success and not long_success:
            cls = CLASS_SHORT
        elif long_success and short_success:
            if long_step is not None and short_step is not None:
                if long_step < short_step:
                    cls = CLASS_LONG
                elif short_step < long_step:
                    cls = CLASS_SHORT
                else:
                    cls = CLASS_FLAT
            else:
                cls = CLASS_FLAT
        else:
            cls = CLASS_FLAT

        label_class[i] = cls
        label_name[i] = CLASS_TO_NAME[cls]
        label_position[i] = CLASS_TO_POSITION[cls]
        if cls == CLASS_LONG and long_sl > 0:
            protection = build_long_protection(
                entry_price=entry,
                atr=atr,
                tp_multiplier=tp_multiplier,
                sl_multiplier=sl_multiplier,
            )
            label_take_profit_price[i] = protection.take_profit_price
            label_stop_loss_price[i] = protection.stop_loss_price
            label_risk_reward[i] = protection.risk_reward
        elif cls == CLASS_SHORT:
            # SHORT remains a research label for market regime learning. Binance
            # Spot execution is long/flat only, but recording the symmetric
            # barrier keeps the training target auditable.
            label_take_profit_price[i] = short_tp
            label_stop_loss_price[i] = short_sl
            label_risk_reward[i] = (entry - short_tp) / max(short_sl - entry, 1e-12)

    df["label_class"] = pd.Series(label_class, dtype="Int64")
    df["label_name"] = label_name
    df["label_position"] = pd.Series(label_position, dtype="Int64")
    df["label_take_profit_price"] = pd.Series(label_take_profit_price, dtype="Float64")
    df["label_stop_loss_price"] = pd.Series(label_stop_loss_price, dtype="Float64")
    df["label_risk_reward"] = pd.Series(label_risk_reward, dtype="Float64")
    df["label_lookahead_bars"] = int(lookahead_bars)
    df["label_tp_multiplier"] = float(tp_multiplier)
    df["label_sl_multiplier"] = float(sl_multiplier)
    return df


def main():
    args = parse_args()
    print(
        "labels.py is a library module. "
        "Use it from feature_store.py; no standalone DB execution is performed here."
    )
    print(
        f"Configured defaults: lookahead={args.lookahead_bars}, "
        f"tp_multiplier={args.tp_multiplier}, sl_multiplier={args.sl_multiplier}"
    )


if __name__ == "__main__":
    main()
