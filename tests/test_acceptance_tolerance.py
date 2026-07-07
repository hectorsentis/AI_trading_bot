"""Honest baseline-tolerance acceptance logic (trade-friendly, not fudged)."""
from __future__ import annotations

from strategy_evaluator import evaluate_model_acceptance


def _bundle(total_return, buy_hold, sharpe=0.5, pf=1.2, dd=0.10, trades=12):
    return {
        "backtest_oos": {
            "trade_lifecycle": {
                "metrics": {
                    "total_return": total_return,
                    "buy_hold_return": buy_hold,
                    "sharpe": sharpe,
                    "profit_factor": pf,
                    "max_drawdown": dd,
                    "trade_count": trades,
                }
            }
        },
        "walk_forward": {"classification": {"overall_accuracy": 0.5, "overall_f1_macro": 0.4}},
    }


def test_accepts_net_positive_within_baseline_tolerance():
    # +5% vs HODL +8% -> excess -3% (within 5% tolerance) and net-positive -> accepted.
    res = evaluate_model_acceptance(_bundle(total_return=0.05, buy_hold=0.08))
    assert res["accepted"], res["rejection_reasons"]


def test_rejects_far_below_hodl():
    # +1% vs HODL +20% -> excess -19% -> underperform baseline -> rejected.
    res = evaluate_model_acceptance(_bundle(total_return=0.01, buy_hold=0.20))
    assert not res["accepted"]
    assert "underperform_baseline" in res["rejection_reasons"]


def test_rejects_net_negative_return():
    res = evaluate_model_acceptance(_bundle(total_return=-0.01, buy_hold=-0.05))
    assert not res["accepted"]
    assert "strategy_return_below_threshold" in res["rejection_reasons"]


def test_rejects_too_few_trades():
    res = evaluate_model_acceptance(_bundle(total_return=0.05, buy_hold=0.06, trades=3))
    assert not res["accepted"]
    assert "trade_count_below_threshold" in res["rejection_reasons"]
