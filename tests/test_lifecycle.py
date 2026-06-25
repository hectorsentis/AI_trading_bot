"""Phase D: paper model degradation/quarantine lifecycle assessment."""
from __future__ import annotations

from paper_model_evaluator import assess_lifecycle


def _m(**kw):
    base = {
        "filled_trades": 10, "enough_sample": True, "max_drawdown": 0.01,
        "profit_factor": 1.3, "total_return": 0.02, "validation_status": "passed",
    }
    base.update(kw)
    return base


def test_pass_when_healthy_and_enough_sample():
    assert assess_lifecycle(_m()) == "pass"


def test_fail_on_enough_sample_failure():
    assert assess_lifecycle(_m(validation_status="failed", profit_factor=0.8)) == "fail"


def test_quarantine_on_severe_drawdown():
    assert assess_lifecycle(_m(validation_status="failed", max_drawdown=0.12)) == "quarantine"


def test_quarantine_on_severe_loss():
    assert assess_lifecycle(_m(validation_status="failed", total_return=-0.08)) == "quarantine"


def test_degrade_on_soft_breach_before_full_sample():
    # Not enough sample to pass/fail, but enough trades and a soft drawdown breach -> degrade.
    m = _m(enough_sample=False, validation_status="insufficient_sample", max_drawdown=0.07, total_return=-0.01)
    assert assess_lifecycle(m) == "degrade"


def test_hold_when_insufficient_evidence():
    m = _m(filled_trades=2, enough_sample=False, validation_status="insufficient_sample")
    assert assess_lifecycle(m) == "hold"
