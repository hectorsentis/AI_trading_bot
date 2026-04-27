from dataclasses import asdict, dataclass
from typing import Any

from config import (
    MIN_ACCEPTABLE_SHARPE,
    MIN_ACCEPTABLE_PROFIT_FACTOR,
    MAX_ACCEPTABLE_DRAWDOWN,
    MIN_ACCEPTABLE_TRADES,
    MIN_ACCEPTABLE_F1_MACRO,
    MIN_ACCEPTABLE_ACCURACY,
    MIN_ACCEPTABLE_STRATEGY_RETURN,
    REQUIRE_OUTPERFORM_BASELINE,
    MAX_TRAIN_VALIDATION_DRIFT,
    REQUIRE_OOS_FOR_ACCEPTANCE,
)


@dataclass
class ModelAcceptanceCriteria:
    min_acceptable_sharpe: float = MIN_ACCEPTABLE_SHARPE
    min_acceptable_profit_factor: float = MIN_ACCEPTABLE_PROFIT_FACTOR
    max_acceptable_drawdown: float = MAX_ACCEPTABLE_DRAWDOWN
    min_acceptable_trades: int = MIN_ACCEPTABLE_TRADES
    min_acceptable_f1_macro: float = MIN_ACCEPTABLE_F1_MACRO
    min_acceptable_accuracy: float = MIN_ACCEPTABLE_ACCURACY
    min_acceptable_strategy_return: float = MIN_ACCEPTABLE_STRATEGY_RETURN
    require_outperform_baseline: bool = REQUIRE_OUTPERFORM_BASELINE
    max_train_validation_drift: float = MAX_TRAIN_VALIDATION_DRIFT
    require_oos_for_acceptance: bool = REQUIRE_OOS_FOR_ACCEPTANCE


def _nested_get(payload: dict[str, Any], keys: list[str]) -> Any:
    node: Any = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _record_check(
    checks: list[dict[str, Any]],
    key: str,
    scope: str,
    value: Any,
    threshold: Any,
    passed: bool,
    comparator: str,
) -> None:
    checks.append(
        {
            "key": key,
            "scope": scope,
            "value": value,
            "threshold": threshold,
            "comparator": comparator,
            "passed": bool(passed),
        }
    )


def evaluate_model_acceptance(
    metrics_bundle: dict[str, Any],
    criteria: ModelAcceptanceCriteria | None = None,
) -> dict[str, Any]:
    criteria = criteria or ModelAcceptanceCriteria()
    checks: list[dict[str, Any]] = []
    rejection_reasons: list[str] = []

    holdout = metrics_bundle.get("holdout", {}) or {}
    walk_forward = metrics_bundle.get("walk_forward", {}) or {}
    backtest_oos = metrics_bundle.get("backtest_oos", {}) or {}

    oos_available = bool(walk_forward) or bool(backtest_oos)
    if criteria.require_oos_for_acceptance and not oos_available:
        rejection_reasons.append("missing_oos_evidence")

    # Classification scope preference: walk-forward > holdout.
    cls_accuracy = _to_float(
        _nested_get(walk_forward, ["classification", "overall_accuracy"])
        or _nested_get(holdout, ["classification", "accuracy"])
    )
    cls_f1 = _to_float(
        _nested_get(walk_forward, ["classification", "overall_f1_macro"])
        or _nested_get(holdout, ["classification", "f1_macro"])
    )

    # Economic scope preference: OOS backtest > walk-forward > holdout.
    econ_strategy_return = _to_float(
        _nested_get(backtest_oos, ["economic", "strategy_return"])
        or _nested_get(walk_forward, ["economic", "overall_strategy_return"])
        or _nested_get(holdout, ["economic", "strategy_return"])
    )
    econ_buy_hold = _to_float(
        _nested_get(backtest_oos, ["economic", "buy_hold_return"])
        or _nested_get(walk_forward, ["economic", "overall_buy_hold_return"])
        or _nested_get(holdout, ["economic", "buy_hold_return"])
    )
    econ_sharpe = _to_float(
        _nested_get(backtest_oos, ["economic", "sharpe"])
        or _nested_get(walk_forward, ["economic", "overall_sharpe"])
        or _nested_get(holdout, ["economic", "sharpe"])
    )
    econ_drawdown = _to_float(
        _nested_get(backtest_oos, ["economic", "max_drawdown"])
        or _nested_get(walk_forward, ["economic", "overall_max_drawdown"])
        or _nested_get(holdout, ["economic", "max_drawdown"])
    )
    econ_pf = _to_float(
        _nested_get(backtest_oos, ["economic", "profit_factor"])
        or _nested_get(walk_forward, ["economic", "overall_profit_factor"])
        or _nested_get(holdout, ["economic", "profit_factor"])
    )
    econ_trades = _to_int(
        _nested_get(backtest_oos, ["economic", "trade_count"])
        or _nested_get(walk_forward, ["economic", "overall_trade_count"])
        or _nested_get(holdout, ["economic", "trade_count"])
    )

    if cls_accuracy is None:
        rejection_reasons.append("missing_accuracy_metric")
    else:
        passed = cls_accuracy >= criteria.min_acceptable_accuracy
        _record_check(checks, "accuracy", "classification", cls_accuracy, criteria.min_acceptable_accuracy, passed, ">=")
        if not passed:
            rejection_reasons.append("accuracy_below_threshold")

    if cls_f1 is None:
        rejection_reasons.append("missing_f1_metric")
    else:
        passed = cls_f1 >= criteria.min_acceptable_f1_macro
        _record_check(checks, "f1_macro", "classification", cls_f1, criteria.min_acceptable_f1_macro, passed, ">=")
        if not passed:
            rejection_reasons.append("f1_macro_below_threshold")

    if econ_strategy_return is None:
        rejection_reasons.append("missing_strategy_return_metric")
    else:
        passed = econ_strategy_return >= criteria.min_acceptable_strategy_return
        _record_check(
            checks,
            "strategy_return",
            "economic",
            econ_strategy_return,
            criteria.min_acceptable_strategy_return,
            passed,
            ">=",
        )
        if not passed:
            rejection_reasons.append("strategy_return_below_threshold")

    if econ_sharpe is None:
        rejection_reasons.append("missing_sharpe_metric")
    else:
        passed = econ_sharpe >= criteria.min_acceptable_sharpe
        _record_check(checks, "sharpe", "economic", econ_sharpe, criteria.min_acceptable_sharpe, passed, ">=")
        if not passed:
            rejection_reasons.append("sharpe_below_threshold")

    if econ_pf is None:
        rejection_reasons.append("missing_profit_factor_metric")
    else:
        passed = econ_pf >= criteria.min_acceptable_profit_factor
        _record_check(
            checks,
            "profit_factor",
            "economic",
            econ_pf,
            criteria.min_acceptable_profit_factor,
            passed,
            ">=",
        )
        if not passed:
            rejection_reasons.append("profit_factor_below_threshold")

    if econ_drawdown is None:
        rejection_reasons.append("missing_drawdown_metric")
    else:
        drawdown_abs = abs(float(econ_drawdown))
        passed = drawdown_abs <= criteria.max_acceptable_drawdown
        _record_check(
            checks,
            "max_drawdown_abs",
            "economic",
            drawdown_abs,
            criteria.max_acceptable_drawdown,
            passed,
            "<=",
        )
        if not passed:
            rejection_reasons.append("drawdown_above_threshold")

    if econ_trades is None:
        rejection_reasons.append("missing_trade_count_metric")
    else:
        passed = econ_trades >= criteria.min_acceptable_trades
        _record_check(checks, "trade_count", "economic", econ_trades, criteria.min_acceptable_trades, passed, ">=")
        if not passed:
            rejection_reasons.append("trade_count_below_threshold")

    if criteria.require_outperform_baseline:
        if econ_strategy_return is None or econ_buy_hold is None:
            rejection_reasons.append("missing_baseline_comparison_metric")
        else:
            passed = econ_strategy_return > econ_buy_hold
            _record_check(
                checks,
                "outperform_baseline",
                "economic",
                econ_strategy_return - econ_buy_hold,
                0.0,
                passed,
                ">",
            )
            if not passed:
                rejection_reasons.append("underperform_baseline")

    holdout_return = _to_float(_nested_get(holdout, ["economic", "strategy_return"]))
    oos_return = _to_float(
        _nested_get(backtest_oos, ["economic", "strategy_return"])
        or _nested_get(walk_forward, ["economic", "overall_strategy_return"])
    )
    if holdout_return is not None and oos_return is not None:
        drift = abs(holdout_return - oos_return)
        passed = drift <= criteria.max_train_validation_drift
        _record_check(
            checks,
            "train_validation_drift",
            "consistency",
            drift,
            criteria.max_train_validation_drift,
            passed,
            "<=",
        )
        if not passed:
            rejection_reasons.append("train_validation_drift_above_threshold")

    unique_reasons = sorted(set(rejection_reasons))
    hard_fail = any(
        reason
        for reason in unique_reasons
        if reason
        and reason
        not in {
            "missing_oos_evidence",
        }
    )

    if hard_fail:
        acceptance_status = "rejected"
    elif unique_reasons:
        acceptance_status = "candidate"
    else:
        acceptance_status = "accepted"

    accepted = acceptance_status == "accepted"

    summary = {
        "accepted": accepted,
        "acceptance_status": acceptance_status,
        "rejection_reasons": unique_reasons,
        "checks": checks,
        "criteria": asdict(criteria),
        "evaluation_scope": ",".join(
            scope
            for scope, present in [
                ("holdout", bool(holdout)),
                ("walk_forward", bool(walk_forward)),
                ("backtest_oos", bool(backtest_oos)),
            ]
            if present
        ),
    }
    return summary
