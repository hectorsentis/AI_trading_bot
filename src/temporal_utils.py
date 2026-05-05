"""Temporal split helpers for leakage-safe market model training.

All labels in this project can depend on future bars through the configured
lookahead horizon.  A plain split at ``test_start`` is therefore not enough:
the final training labels before ``test_start`` may have observed validation
prices.  These helpers centralize the required embargo.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplitMetadata:
    split_method: str
    reference_now_utc: str
    validation_start_utc: str
    validation_end_utc: str
    train_cutoff_utc: str
    embargo_bars: int
    embargo_duration: str
    stale_dataset_anchor_used: bool

    def to_dict(self) -> dict:
        return asdict(self)


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    match = re.fullmatch(r"(\d+)([mhdwM])", str(timeframe).strip())
    if not match:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    if unit == "w":
        return pd.Timedelta(weeks=value)
    if unit == "M":
        return pd.Timedelta(days=30 * value)
    raise ValueError(f"Unsupported timeframe unit: {timeframe}")


def embargo_timedelta(timeframe: str, lookahead_bars: int) -> pd.Timedelta:
    return timeframe_to_timedelta(timeframe) * max(0, int(lookahead_bars))


def apply_label_embargo_cutoff(
    test_start: pd.Timestamp,
    timeframe: str,
    lookahead_bars: int,
) -> pd.Timestamp:
    test_start = pd.Timestamp(test_start)
    if test_start.tzinfo is None:
        test_start = test_start.tz_localize("UTC")
    else:
        test_start = test_start.tz_convert("UTC")
    return test_start - embargo_timedelta(timeframe, lookahead_bars)


def strict_train_validation_split(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    timeframe: str,
    training_cutoff_hours_before_now: int,
    validation_window_hours: int,
    fallback_test_size_dates: int,
    lookahead_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, TemporalSplitMetadata]:
    """Split into train/validation with a label-lookahead embargo.

    The preferred split uses the configured validation window before "now".
    If the local dataset is historical/stale, the same window is anchored to
    the dataset maximum so research can continue without using future rows.
    If that produces empty segments, the latest ``fallback_test_size_dates``
    unique timestamps are reserved as validation and the same embargo is
    applied before that validation block.
    """
    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")

    ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    if ts.isna().all():
        raise ValueError(f"Timestamp column {timestamp_col} has no valid UTC timestamps.")

    dataset_max = ts.max()
    reference_now = pd.Timestamp.now(tz="UTC")
    stale_anchor_used = False
    stale_threshold = pd.Timedelta(hours=int(training_cutoff_hours_before_now) + int(validation_window_hours))
    if dataset_max < reference_now - stale_threshold:
        reference_now = dataset_max + pd.Timedelta(hours=int(training_cutoff_hours_before_now))
        stale_anchor_used = True

    validation_end = reference_now - pd.Timedelta(hours=int(training_cutoff_hours_before_now))
    validation_start = validation_end - pd.Timedelta(hours=int(validation_window_hours))
    train_cutoff = apply_label_embargo_cutoff(validation_start, timeframe, lookahead_bars)

    train_df = df[ts < train_cutoff].copy()
    test_df = df[(ts >= validation_start) & (ts < validation_end)].copy()
    split_method = "configured_window"

    if train_df.empty or test_df.empty:
        unique_dates = sorted(pd.Series(ts.dropna().unique()).tolist())
        if len(unique_dates) <= int(fallback_test_size_dates):
            raise ValueError(
                f"Not enough unique datetimes ({len(unique_dates)}) for test_size={fallback_test_size_dates}."
            )
        validation_start = pd.Timestamp(unique_dates[-int(fallback_test_size_dates)]).tz_convert("UTC")
        validation_end = pd.Timestamp(unique_dates[-1]).tz_convert("UTC") + timeframe_to_timedelta(timeframe)
        train_cutoff = apply_label_embargo_cutoff(validation_start, timeframe, lookahead_bars)
        train_df = df[ts < train_cutoff].copy()
        test_df = df[ts >= validation_start].copy()
        split_method = "latest_dates_fallback"

    if train_df.empty or test_df.empty:
        raise ValueError("Temporal split with label embargo produced empty train/test segment.")

    embargo = embargo_timedelta(timeframe, lookahead_bars)
    meta = TemporalSplitMetadata(
        split_method=split_method,
        reference_now_utc=reference_now.isoformat(),
        validation_start_utc=validation_start.isoformat(),
        validation_end_utc=validation_end.isoformat(),
        train_cutoff_utc=train_cutoff.isoformat(),
        embargo_bars=int(lookahead_bars),
        embargo_duration=str(embargo),
        stale_dataset_anchor_used=stale_anchor_used,
    )
    return train_df, test_df, meta
