from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modeling_utils import compute_economic_metrics
from temporal_utils import strict_train_validation_split
from labels import generate_triple_barrier_labels
from trade_protection import build_long_protection, validate_long_protection


class TemporalTrainingSafetyTests(unittest.TestCase):
    def test_split_applies_label_embargo_before_validation(self) -> None:
        now = pd.Timestamp.now(tz="UTC").floor("h")
        df = pd.DataFrame(
            {
                "datetime_utc": [now - pd.Timedelta(hours=500 - i) for i in range(500)],
                "value": range(500),
            }
        )

        train, validation, meta = strict_train_validation_split(
            df,
            timestamp_col="datetime_utc",
            timeframe="1h",
            training_cutoff_hours_before_now=24,
            validation_window_hours=48,
            fallback_test_size_dates=50,
            lookahead_bars=6,
        )

        self.assertFalse(train.empty)
        self.assertFalse(validation.empty)
        self.assertLess(train["datetime_utc"].max(), validation["datetime_utc"].min() - pd.Timedelta(hours=6))
        self.assertEqual(meta.embargo_bars, 6)

    def test_spot_economics_do_not_profit_from_short_signals_by_default(self) -> None:
        frame = pd.DataFrame(
            {
                "symbol": ["BTCUSDT", "BTCUSDT"],
                "datetime_utc": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
                "signal_position": [-1, -1],
                "fwd_return_1": [-0.10, -0.10],
            }
        )
        _, _, spot_metrics = compute_economic_metrics(frame, timeframe="1h", cost_per_trade=0.0)
        _, _, research_short_metrics = compute_economic_metrics(frame, timeframe="1h", cost_per_trade=0.0, allow_short=True)

        self.assertEqual(spot_metrics.strategy_return, 0.0)
        self.assertGreater(research_short_metrics.strategy_return, 0.0)

    def test_triple_barrier_labels_record_tp_sl_geometry(self) -> None:
        frame = pd.DataFrame(
            {
                "close": [100.0, 100.0, 100.0, 100.0],
                "high": [100.0, 103.0, 100.0, 100.0],
                "low": [100.0, 99.5, 100.0, 100.0],
                "atr_14": [1.0, 1.0, 1.0, 1.0],
            }
        )
        labeled = generate_triple_barrier_labels(frame, lookahead_bars=2, tp_multiplier=2.0, sl_multiplier=1.0)

        self.assertEqual(int(labeled.loc[0, "label_position"]), 1)
        self.assertAlmostEqual(float(labeled.loc[0, "label_take_profit_price"]), 102.0)
        self.assertAlmostEqual(float(labeled.loc[0, "label_stop_loss_price"]), 99.0)
        self.assertAlmostEqual(float(labeled.loc[0, "label_risk_reward"]), 2.0)

    def test_long_protection_requires_tp_above_and_sl_below_entry(self) -> None:
        protection = build_long_protection(entry_price=100.0, atr=2.0, tp_multiplier=1.5, sl_multiplier=1.0)
        ok, reasons, details = validate_long_protection(
            entry_price=100.0,
            take_profit_price=protection.take_profit_price,
            stop_loss_price=protection.stop_loss_price,
            min_risk_reward=1.0,
        )
        self.assertTrue(ok, reasons)
        self.assertAlmostEqual(float(details["risk_reward"]), 1.5)

        ok, reasons, _ = validate_long_protection(
            entry_price=100.0,
            take_profit_price=None,
            stop_loss_price=99.0,
        )
        self.assertFalse(ok)
        self.assertIn("missing_or_invalid_take_profit", reasons)


if __name__ == "__main__":
    unittest.main()
