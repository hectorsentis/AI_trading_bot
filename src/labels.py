import argparse
from typing import Optional

import pandas as pd

from config import LOOKAHEAD_BARS, TP_MULTIPLIER, SL_MULTIPLIER
from modeling_utils import CLASS_FLAT, CLASS_LONG, CLASS_SHORT, CLASS_TO_NAME, CLASS_TO_POSITION


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

    df["label_class"] = pd.Series(label_class, dtype="Int64")
    df["label_name"] = label_name
    df["label_position"] = pd.Series(label_position, dtype="Int64")
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
