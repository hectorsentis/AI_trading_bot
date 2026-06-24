from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from config import REPORTS_DIR


def _clean_key(value: Any) -> str:
    if value is None:
        return "NULL"
    try:
        if pd.isna(value):
            return "NULL"
    except Exception:
        pass
    text = str(value)
    return text if text else "EMPTY"


def _split_reasons(reason: Any) -> list[str]:
    if reason is None:
        return ["none"]
    try:
        if pd.isna(reason):
            return ["none"]
    except Exception:
        pass
    parts = [p.strip() for p in str(reason).split(";") if p.strip()]
    return parts or ["none"]


def _counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda item: str(item[0]))}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ValidationFunnelRecorder:
    """Runtime audit of the strict validation/backtest trade lifecycle funnel.

    The recorder intentionally observes the existing pipeline; it does not change
    allocator, risk-manager or trade-builder decisions.
    """

    def __init__(self, *, model_id: str, validation_run_id: str | None, simulation_run_id: str, timeframe: str):
        self.model_id = str(model_id)
        self.validation_run_id = str(validation_run_id) if validation_run_id else None
        self.simulation_run_id = str(simulation_run_id)
        self.timeframe = str(timeframe)
        self.generated_at_utc = pd.Timestamp.now(tz="UTC").isoformat()

        self.validation_prediction_rows = 0
        self.signal_position_counts: Counter = Counter()
        self.model_prediction_rows = 0
        self.direction_counts: Counter = Counter()
        self.long_prediction_count = 0

        self.proposal_count = 0
        self.proposal_by_side_status: Counter = Counter()
        self.initial_proposal_by_side_status: Counter = Counter()
        self.proposal_rejection_reasons: Counter = Counter()

        self.allocation_count = 0
        self.allocation_by_decision_rejection_reason: Counter = Counter()
        self.allocation_rejection_reasons: Counter = Counter()

        self.trade_builder_success_count = 0
        self.trade_builder_failure_count = 0
        self.trade_builder_failure_reasons: Counter = Counter()
        self.trade_builder_sizing_reasons: Counter = Counter()

        self.risk_manager_approval_count = 0
        self.risk_manager_rejection_count = 0
        self.risk_manager_rejection_reasons: Counter = Counter()

        self.historical_entries_scheduled = 0
        self.historical_entries_filled = 0
        self.historical_entries_unfilled = 0
        self.historical_entry_fill_fail_reasons: Counter = Counter()

        self.trades_opened = 0
        self.trades_closed = 0
        self.exit_reasons: Counter = Counter()
        self.orders_count = 0
        self.orders_by_side_status: Counter = Counter()
        self.fills_count = 0
        self.artifact_path: str | None = None

    def observe_validation_predictions(self, predictions_df: pd.DataFrame) -> None:
        self.validation_prediction_rows = int(len(predictions_df)) if predictions_df is not None else 0
        if predictions_df is None or predictions_df.empty:
            return
        if "signal_position" in predictions_df.columns:
            signal_names = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
            series = pd.to_numeric(predictions_df["signal_position"], errors="coerce").fillna(0).astype(int)
            self.signal_position_counts.update(signal_names.get(int(v), str(int(v))) for v in series.tolist())

    def observe_model_prediction(self, prediction: Any) -> None:
        self.model_prediction_rows += 1
        direction = _clean_key(getattr(prediction, "direction", None)).upper()
        self.direction_counts.update([direction])
        if direction == "LONG":
            self.long_prediction_count += 1

    def observe_proposal_initial(self, proposal: Any) -> None:
        self.proposal_count += 1
        side = _clean_key(getattr(proposal, "side", None)).upper()
        status = _clean_key(getattr(proposal, "status", None)).upper()
        self.initial_proposal_by_side_status.update([f"{side}|{status}"])
        for reason in _split_reasons(getattr(proposal, "rejection_reason", None)):
            if reason != "none":
                self.proposal_rejection_reasons.update([reason])

    def observe_allocation(self, proposal: Any, allocation: dict) -> None:
        self.allocation_count += 1
        decision = _clean_key(allocation.get("decision")).upper()
        reason_text = allocation.get("rejection_reason")
        reason_key = ";".join(_split_reasons(reason_text))
        self.allocation_by_decision_rejection_reason.update([f"{decision}|{reason_key}"])
        for reason in _split_reasons(reason_text):
            if reason != "none":
                self.allocation_rejection_reasons.update([reason])

        if decision in {"ACCEPT", "RESIZE"}:
            final_status = "ACCEPTED" if decision == "ACCEPT" else "RESIZED"
        else:
            final_status = decision
        side = _clean_key(getattr(proposal, "side", None)).upper()
        self.proposal_by_side_status.update([f"{side}|{final_status}"])

    def observe_trade_builder(self, built_trade: Any | None, reasons: list[str]) -> None:
        if built_trade is None:
            self.trade_builder_failure_count += 1
            for reason in reasons or ["unknown_trade_builder_failure"]:
                self.trade_builder_failure_reasons.update([str(reason)])
            return
        self.trade_builder_success_count += 1
        sizing = _clean_key(getattr(built_trade, "sizing_reason", None))
        self.trade_builder_sizing_reasons.update([sizing])

    def observe_entry_scheduled(self) -> None:
        self.historical_entries_scheduled += 1

    def observe_risk(self, risk: dict) -> None:
        if risk.get("approved"):
            self.risk_manager_approval_count += 1
            return
        self.risk_manager_rejection_count += 1
        for reason in risk.get("reasons") or ["unknown_risk_rejection"]:
            self.risk_manager_rejection_reasons.update([str(reason)])

    def observe_entry_filled(self) -> None:
        self.historical_entries_filled += 1
        self.trades_opened += 1

    def observe_entry_unfilled(self, reason: str) -> None:
        self.historical_entries_unfilled += 1
        self.historical_entry_fill_fail_reasons.update([reason])

    def observe_order(self, *, side: str, status: str) -> None:
        self.orders_count += 1
        self.orders_by_side_status.update([f"{_clean_key(side).upper()}|{_clean_key(status).upper()}"])

    def observe_fill(self) -> None:
        self.fills_count += 1

    def observe_trade_closed(self, exit_reason: str) -> None:
        self.trades_closed += 1
        self.exit_reasons.update([_clean_key(exit_reason)])

    def final_report(self, *, lifecycle_metrics: dict | None = None) -> dict:
        lifecycle_metrics = lifecycle_metrics or {}
        report = {
            "model_id": self.model_id,
            "validation_run_id": self.validation_run_id,
            "simulation_run_id": self.simulation_run_id,
            "timeframe": self.timeframe,
            "generated_at_utc": self.generated_at_utc,
            "validation_predictions": {
                "total_prediction_rows": int(self.validation_prediction_rows),
                "signal_position_counts": _counter_to_dict(self.signal_position_counts),
            },
            "model_predictions": {
                "total_prediction_rows": int(self.model_prediction_rows),
                "direction_counts": _counter_to_dict(self.direction_counts),
                "long_prediction_count": int(self.long_prediction_count),
            },
            "trade_proposals": {
                "proposal_count": int(self.proposal_count),
                "proposal_count_by_side_status": _counter_to_dict(self.proposal_by_side_status),
                "initial_proposal_count_by_side_status": _counter_to_dict(self.initial_proposal_by_side_status),
                "proposal_rejection_reasons": _counter_to_dict(self.proposal_rejection_reasons),
            },
            "allocations": {
                "allocation_count": int(self.allocation_count),
                "allocation_count_by_decision_rejection_reason": _counter_to_dict(self.allocation_by_decision_rejection_reason),
                "rejection_reason_counts": _counter_to_dict(self.allocation_rejection_reasons),
            },
            "trade_builder": {
                "success_count": int(self.trade_builder_success_count),
                "failure_count": int(self.trade_builder_failure_count),
                "failure_reasons": _counter_to_dict(self.trade_builder_failure_reasons),
                "sizing_reasons": _counter_to_dict(self.trade_builder_sizing_reasons),
            },
            "risk_manager": {
                "approval_count": int(self.risk_manager_approval_count),
                "rejection_count": int(self.risk_manager_rejection_count),
                "rejection_reasons": _counter_to_dict(self.risk_manager_rejection_reasons),
            },
            "historical_simulator": {
                "historical_entries_scheduled": int(self.historical_entries_scheduled),
                "historical_entries_filled": int(self.historical_entries_filled),
                "historical_entries_unfilled": int(self.historical_entries_unfilled),
                "historical_entry_fill_fail_reasons": _counter_to_dict(self.historical_entry_fill_fail_reasons),
                "trades_opened": int(self.trades_opened),
                "trades_closed": int(self.trades_closed),
                "exit_reasons": _counter_to_dict(self.exit_reasons),
                "orders_count": int(self.orders_count),
                "orders_by_side_status": _counter_to_dict(self.orders_by_side_status),
                "fills_count": int(self.fills_count),
            },
            "final_trade_lifecycle_metrics": lifecycle_metrics,
        }
        report["diagnosis"] = self._diagnose(lifecycle_metrics)
        if self.artifact_path:
            report["artifact_path"] = self.artifact_path
        return _json_safe(report)

    def _diagnose(self, lifecycle_metrics: dict) -> dict:
        signal_long = int(self.signal_position_counts.get("LONG", 0))
        signal_short = int(self.signal_position_counts.get("SHORT", 0))
        signal_flat = int(self.signal_position_counts.get("FLAT", 0))
        active_signals = signal_long + signal_short
        accepted_allocations = sum(
            count for key, count in self.allocation_by_decision_rejection_reason.items() if str(key).startswith(("ACCEPT|", "RESIZE|"))
        )
        trade_count = int(lifecycle_metrics.get("trade_count") or self.trades_closed or 0)

        if trade_count > 0:
            return {"status": "trades_created", "message": "Lifecycle simulation opened and closed historical trades."}
        if active_signals > 0 and signal_long == 0:
            return {
                "status": "expected_spot_no_long_entries",
                "message": "All active diagnostic signals are SHORT/FLAT. In Binance Spot scope SHORT means avoid/exit, not open a short trade.",
                "active_signal_count": active_signals,
                "signal_position_counts": _counter_to_dict(self.signal_position_counts),
            }
        if self.long_prediction_count == 0:
            return {
                "status": "no_long_model_predictions",
                "message": "No structured LONG predictions reached the proposal layer; no Spot entries should be opened.",
                "signal_position_counts": _counter_to_dict(self.signal_position_counts),
                "direction_counts": _counter_to_dict(self.direction_counts),
            }
        if self.allocation_count and accepted_allocations == 0:
            return {
                "status": "allocator_rejected_or_shadowed_all_proposals",
                "message": "LONG predictions existed, but the allocator did not accept/resize any proposal. Do not loosen thresholds blindly; inspect rejection reasons.",
                "allocation_rejection_reason_counts": _counter_to_dict(self.allocation_rejection_reasons),
            }
        if self.trade_builder_failure_count and self.trade_builder_success_count == 0:
            return {
                "status": "trade_builder_failed_all_accepted_allocations",
                "message": "Allocator accepted/resized proposals, but trade_builder produced no executable trade plans.",
                "trade_builder_failure_reasons": _counter_to_dict(self.trade_builder_failure_reasons),
            }
        if self.risk_manager_rejection_count and self.risk_manager_approval_count == 0:
            return {
                "status": "risk_manager_rejected_all_trade_plans",
                "message": "Trade plans were built, but the risk manager rejected all entries.",
                "risk_rejection_reason_counts": _counter_to_dict(self.risk_manager_rejection_reasons),
            }
        if self.historical_entries_scheduled and self.historical_entries_filled == 0:
            return {
                "status": "historical_entries_scheduled_but_not_filled",
                "message": "Entries were scheduled but no next-bar fill occurred. Inspect market coverage or simulator scheduling.",
                "historical_entry_fill_fail_reasons": _counter_to_dict(self.historical_entry_fill_fail_reasons),
            }
        if self.long_prediction_count > 0 and self.proposal_count and self.trades_opened == 0:
            return {
                "status": "valid_long_path_no_trades_opened",
                "message": "LONG predictions/proposals existed but no trades opened; inspect allocator, builder and risk sections for the blocking stage.",
            }
        return {"status": "no_tradeable_entries", "message": "No tradeable Spot LONG entries reached execution under current strict criteria."}


def write_funnel_report(report: dict, *, prefix: str = "validation_funnel") -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_id = str(report.get("model_id") or "unknown").replace("/", "_").replace("\\", "_").replace(":", "_")
    run_id = str(report.get("validation_run_id") or report.get("simulation_run_id") or "no_run").replace("/", "_").replace("\\", "_").replace(":", "_")
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    path = Path(REPORTS_DIR) / f"{prefix}_{model_id}_{run_id}_{stamp}.json"
    path.write_text(json.dumps(_json_safe(report), ensure_ascii=True, indent=2), encoding="utf-8")
    return path
