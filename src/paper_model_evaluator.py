import argparse
import json
import os
import sqlite3
import time

import pandas as pd

from config import (
    ACCOUNT_MODE_LOCAL_PAPER,
    ACCOUNT_MODE_TESTNET_PAPER,
    AUTO_REPLACE_REJECTED_MODELS,
    DB_FILE,
    FILLS_TABLE,
    MIN_PAPER_VALIDATION_DAYS,
    MIN_PAPER_VALIDATION_TRADES,
    ORDERS_TABLE,
    PAPER_MAX_DRAWDOWN,
    PAPER_MIN_PROFIT_FACTOR,
    PAPER_MIN_TOTAL_RETURN,
    PAPER_MIN_WIN_RATE,
    PAPER_MODEL_METRICS_TABLE,
    PORTFOLIO_SNAPSHOTS_TABLE,
    POSITIONS_TABLE,
    SIGNALS_TABLE,
    TIMEFRAME,
    TIMEFRAMES,
    MODEL_EVALUATION_INTERVAL_SECONDS,
    PAPER_DEGRADE_MIN_TRADES,
    PAPER_DEGRADE_DRAWDOWN,
    PAPER_DEGRADE_PROFIT_FACTOR,
    PAPER_DEGRADE_MIN_RETURN,
    PAPER_QUARANTINE_DRAWDOWN,
    PAPER_QUARANTINE_MIN_RETURN,
)
from db_utils import init_research_tables
from runtime_status import record_event, update_status
from model_registry import (
    get_model_by_id,
    list_paper_active_models,
    list_paper_degraded_models,
    mark_model_paper_rejected,
    mark_model_paper_validated,
    mark_model_paper_degraded,
    mark_model_quarantined,
    reactivate_paper_model,
    mark_model_real_ready,
)


def assess_lifecycle(metrics: dict) -> str:
    """Map paper metrics to a lifecycle decision (Phase D).

    Returns one of: ``pass`` (validate + real-ready), ``fail`` (hard reject),
    ``quarantine`` (severe degradation, stop), ``degrade`` (soft pause, recoverable),
    or ``hold`` (insufficient evidence; leave as-is).
    """
    filled = int(metrics.get("filled_trades", 0))
    enough = bool(metrics.get("enough_sample", False))
    max_dd = float(metrics.get("max_drawdown", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    total_return = float(metrics.get("total_return", 0.0))
    passes = metrics.get("validation_status") == "passed"

    # Severe breaches quarantine immediately (even before the full sample window).
    if filled >= PAPER_DEGRADE_MIN_TRADES and (max_dd >= PAPER_QUARANTINE_DRAWDOWN or total_return <= PAPER_QUARANTINE_MIN_RETURN):
        return "quarantine"
    if enough and passes:
        return "pass"
    if enough and metrics.get("validation_status") == "failed":
        return "fail"
    if filled >= PAPER_DEGRADE_MIN_TRADES and (
        max_dd >= PAPER_DEGRADE_DRAWDOWN
        or pf < PAPER_DEGRADE_PROFIT_FACTOR
        or total_return <= PAPER_DEGRADE_MIN_RETURN
    ):
        return "degrade"
    return "hold"


def _read(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax().replace(0, pd.NA)
    dd = (equity / roll_max - 1.0).fillna(0.0)
    return float(abs(dd.min()))


def evaluate_model_paper(model_id: str, account_mode: str = ACCOUNT_MODE_LOCAL_PAPER, timeframe: str | None = None) -> dict:
    init_research_tables()
    orders = _read(
        f"SELECT * FROM {ORDERS_TABLE} WHERE model_id = ? AND account_mode = ? ORDER BY created_at_utc",
        (model_id, account_mode),
    )
    fills = _read(
        f"SELECT * FROM {FILLS_TABLE} WHERE model_id = ? AND account_mode = ? ORDER BY timestamp_utc",
        (model_id, account_mode),
    )
    snaps = _read(
        f"SELECT * FROM {PORTFOLIO_SNAPSHOTS_TABLE} WHERE model_id = ? AND account_mode = ? ORDER BY datetime_utc",
        (model_id, account_mode),
    )
    pos = _read(
        f"SELECT * FROM {POSITIONS_TABLE} WHERE model_id = ? AND account_mode = ?",
        (model_id, account_mode),
    )
    last_signal = _read(
        f"SELECT datetime_utc FROM {SIGNALS_TABLE} WHERE model_id = ? AND account_mode = ? ORDER BY datetime_utc DESC LIMIT 1",
        (model_id, account_mode),
    )

    trades = int(len(orders))
    filled_trades = int(len(orders[orders["status"].astype(str).str.upper() == "FILLED"])) if not orders.empty else 0
    if not snaps.empty:
        for col in ["equity", "realized_pnl", "unrealized_pnl"]:
            snaps[col] = pd.to_numeric(snaps[col], errors="coerce").fillna(0.0)
        equity = float(snaps["equity"].iloc[-1])
        initial_equity = float(snaps["equity"].iloc[0]) or 1e-9
        realized_pnl = float(snaps["realized_pnl"].iloc[-1])
        unrealized_pnl = float(snaps["unrealized_pnl"].iloc[-1])
        total_pnl = equity - initial_equity
        total_return = total_pnl / initial_equity
        max_dd = _max_drawdown(snaps["equity"])
        first_ts = pd.to_datetime(snaps["datetime_utc"].iloc[0], utc=True, errors="coerce")
        last_ts = pd.to_datetime(snaps["datetime_utc"].iloc[-1], utc=True, errors="coerce")
        days_active = max(0.0, float((last_ts - first_ts).total_seconds() / 86400.0)) if pd.notna(first_ts) and pd.notna(last_ts) else 0.0
    else:
        equity = realized_pnl = unrealized_pnl = total_pnl = total_return = max_dd = days_active = 0.0

    # Conservative trade quality metrics from realized deltas inferred by sell/close fills are not always available.
    # Use order notional changes as a rough audit metric and keep full fills for external inspection.
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    if not orders.empty and "reason" in orders:
        # If the portfolio has positive realized PnL, count it as one aggregate win for gating visibility.
        if realized_pnl > 0:
            wins, gross_profit = 1, realized_pnl
        elif realized_pnl < 0:
            losses, gross_loss = 1, abs(realized_pnl)
    win_rate = float(wins / max(1, wins + losses)) if (wins + losses) else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    avg_trade_return = float(total_return / max(1, filled_trades))

    current_exposure = 0.0
    if not pos.empty:
        qty = pd.to_numeric(pos.get("quantity"), errors="coerce").fillna(0.0).abs()
        avg = pd.to_numeric(pos.get("avg_price"), errors="coerce").fillna(0.0)
        current_exposure = float((qty * avg).sum())

    enough_sample = days_active >= MIN_PAPER_VALIDATION_DAYS or filled_trades >= MIN_PAPER_VALIDATION_TRADES
    passes = (
        enough_sample
        and profit_factor >= PAPER_MIN_PROFIT_FACTOR
        and max_dd <= PAPER_MAX_DRAWDOWN
        and total_return >= PAPER_MIN_TOTAL_RETURN
        and win_rate >= PAPER_MIN_WIN_RATE
    )
    fails = enough_sample and not passes
    validation_status = "insufficient_sample"
    if passes:
        validation_status = "passed"
    elif fails:
        validation_status = "failed"

    metrics = {
        "model_id": model_id,
        "account_mode": account_mode,
        "trades": trades,
        "filled_trades": filled_trades,
        "win_rate": win_rate,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "equity": equity,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "average_trade_return": avg_trade_return,
        "days_active": days_active,
        "last_signal": None if last_signal.empty else str(last_signal.iloc[0]["datetime_utc"]),
        "last_order": None if orders.empty else str(orders.iloc[-1].get("created_at_utc")),
        "last_fill": None if fills.empty else str(fills.iloc[-1].get("timestamp_utc")),
        "current_exposure": current_exposure,
        "validation_status": validation_status,
        "enough_sample": enough_sample,
    }
    _persist_metrics(metrics, timeframe=timeframe)
    return metrics


def _persist_metrics(metrics: dict, timeframe: str | None = None) -> None:
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            f"""
            INSERT INTO {PAPER_MODEL_METRICS_TABLE} (
                model_id, account_mode, timeframe, trades, filled_trades, win_rate,
                realized_pnl, unrealized_pnl, total_pnl, equity, total_return,
                max_drawdown, profit_factor, average_trade_return, days_active,
                last_signal_utc, last_order_utc, last_fill_utc, current_exposure,
                validation_status, metrics_json, evaluated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metrics["model_id"], metrics["account_mode"], timeframe, metrics["trades"], metrics["filled_trades"],
                metrics["win_rate"], metrics["realized_pnl"], metrics["unrealized_pnl"], metrics["total_pnl"], metrics["equity"],
                metrics["total_return"], metrics["max_drawdown"], metrics["profit_factor"], metrics["average_trade_return"],
                metrics["days_active"], metrics["last_signal"], metrics["last_order"], metrics["last_fill"], metrics["current_exposure"],
                metrics["validation_status"], json.dumps(metrics, ensure_ascii=True, sort_keys=True), pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def evaluate_active_models(account_mode: str | None = None, timeframe: str = TIMEFRAME) -> list[dict]:
    init_research_tables()
    # Resolve any matured shadow trades so allocator opportunity-cost analytics stay current.
    try:
        from shadow_evaluator import evaluate_open_shadow_trades
        evaluate_open_shadow_trades()
    except Exception as exc:  # pragma: no cover - shadow resolution must never block evaluation
        record_event("paper_model_evaluator", "warning", f"shadow evaluation failed: {exc}")
    results: list[dict] = []

    # Active models: promote, reject, degrade or quarantine.
    for model in list_paper_active_models(timeframe=timeframe):
        mode = account_mode or model.get("account_mode") or ACCOUNT_MODE_TESTNET_PAPER
        metrics = evaluate_model_paper(model["model_id"], account_mode=mode, timeframe=timeframe)
        decision = assess_lifecycle(metrics)
        metrics["lifecycle_decision"] = decision
        results.append(metrics)
        if decision == "pass":
            mark_model_paper_validated(model["model_id"], metrics=metrics)
            mark_model_real_ready(model["model_id"], metrics=metrics)
        elif decision == "fail":
            mark_model_paper_rejected(model["model_id"], reason="paper_validation_failed", metrics=metrics)
        elif decision == "quarantine":
            mark_model_quarantined(model["model_id"], reason="paper_severe_degradation", metrics=metrics)
            record_event("paper_model_evaluator", "warning", f"quarantined {model['model_id']}", metadata={"max_drawdown": metrics["max_drawdown"], "total_return": metrics["total_return"]})
        elif decision == "degrade":
            mark_model_paper_degraded(model["model_id"], reason="paper_degradation_detected", metrics=metrics)
            record_event("paper_model_evaluator", "warning", f"degraded {model['model_id']}", metadata={"max_drawdown": metrics["max_drawdown"], "profit_factor": metrics["profit_factor"]})

    # Degraded models: recover to active, quarantine if worse, else stay degraded.
    for model in list_paper_degraded_models(timeframe=timeframe):
        mode = account_mode or model.get("account_mode") or ACCOUNT_MODE_TESTNET_PAPER
        metrics = evaluate_model_paper(model["model_id"], account_mode=mode, timeframe=timeframe)
        decision = assess_lifecycle(metrics)
        metrics["lifecycle_decision"] = f"degraded::{decision}"
        results.append(metrics)
        if decision in {"pass", "hold"} and metrics["max_drawdown"] < PAPER_DEGRADE_DRAWDOWN:
            reactivate_paper_model(model["model_id"], account_mode=mode, metrics=metrics)
            record_event("paper_model_evaluator", "info", f"recovered {model['model_id']} to paper_active")
        elif decision == "quarantine":
            mark_model_quarantined(model["model_id"], reason="degraded_then_severe", metrics=metrics)
        elif decision == "fail":
            mark_model_paper_rejected(model["model_id"], reason="degraded_then_failed", metrics=metrics)
    return results


def evaluate_active_models_many(account_mode: str | None = None, timeframes: list[str] | None = None) -> dict[str, list[dict]]:
    selected = [tf.strip() for tf in (timeframes or TIMEFRAMES) if str(tf).strip()]
    return {timeframe: evaluate_active_models(account_mode=account_mode, timeframe=timeframe) for timeframe in selected}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate paper trading performance per model_id/account_mode.")
    parser.add_argument("--evaluate-active", action="store_true")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--account-mode", default=None)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--timeframes", nargs="*", default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=MODEL_EVALUATION_INTERVAL_SECONDS)
    return parser.parse_args()


def main():
    args = parse_args()
    timeframes = [tf.strip() for tf in (args.timeframes or []) if str(tf).strip()]
    if args.loop:
        update_status("paper_model_evaluator", "running", pid=os.getpid(), message="loop starting")
        while True:
            try:
                results = evaluate_active_models_many(account_mode=args.account_mode, timeframes=timeframes) if timeframes else evaluate_active_models(account_mode=args.account_mode, timeframe=args.timeframe)
                update_status(
                    "paper_model_evaluator",
                    "running",
                    pid=os.getpid(),
                    message="evaluated active paper models",
                    metadata={"timeframes": timeframes or [args.timeframe]},
                )
                print(json.dumps({"component": "paper_model_evaluator", "results": results}, ensure_ascii=True), flush=True)
            except KeyboardInterrupt:
                update_status("paper_model_evaluator", "stopped", pid=os.getpid(), message="keyboard interrupt")
                raise
            except Exception as exc:
                update_status("paper_model_evaluator", "error", pid=os.getpid(), message=str(exc))
                record_event("paper_model_evaluator", "error", str(exc))
                print(json.dumps({"component": "paper_model_evaluator", "error": str(exc)}, ensure_ascii=True), flush=True)
            time.sleep(max(10, int(args.interval_seconds)))

    if args.evaluate_active:
        results = evaluate_active_models_many(account_mode=args.account_mode, timeframes=timeframes) if timeframes else evaluate_active_models(account_mode=args.account_mode, timeframe=args.timeframe)
        print(json.dumps(results, ensure_ascii=True, indent=2))
        return
    if not args.model_id:
        raise SystemExit("Use --evaluate-active or --model-id")
    print(json.dumps(evaluate_model_paper(args.model_id, account_mode=args.account_mode or ACCOUNT_MODE_LOCAL_PAPER, timeframe=args.timeframe), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

