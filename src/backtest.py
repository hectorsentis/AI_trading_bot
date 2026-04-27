import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd

from config import (
    DB_FILE,
    FEATURES_TABLE,
    FEATURE_COLUMNS,
    VALIDATION_PREDICTIONS_TABLE,
    TIMEFRAME,
    SYMBOLS,
    COST_PER_TRADE,
    REPORTS_DIR,
)
from db_utils import (
    init_research_tables,
    ensure_project_directories,
    get_latest_validation_run_id,
)
from model_registry import (
    get_latest_model,
    get_model_by_id,
    update_model_evaluation,
)
from modeling_utils import POSITION_TO_NAME, compute_economic_metrics, probabilities_to_signal
from strategy_evaluator import evaluate_model_acceptance


def parse_args():
    parser = argparse.ArgumentParser(description="Economic backtest from feature store or strict OOS predictions.")
    parser.add_argument("--mode", choices=["in_sample", "oos"], default="in_sample")
    parser.add_argument("--model-path", type=str, default=None, help="Path to .joblib model artifact (in_sample mode)")
    parser.add_argument("--model-id", type=str, default=None, help="Model id for OOS prediction backtest")
    parser.add_argument("--validation-run-id", type=str, default=None, help="Specific OOS validation run id")
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--start-date", type=str, default=None, help="UTC date filter, e.g. 2025-01-01")
    parser.add_argument("--end-date", type=str, default=None, help="UTC date filter, e.g. 2026-01-01")
    return parser.parse_args()


def resolve_model_path(model_path_arg: str | None, timeframe: str) -> Path:
    if model_path_arg:
        path = Path(model_path_arg)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        return path

    latest = get_latest_model(timeframe=timeframe, prefer_active=True)
    if latest and Path(latest["model_path"]).exists():
        return Path(latest["model_path"])

    candidates = sorted(Path("models").glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No trained model found in registry or models directory.")
    return candidates[0]


def load_backtest_frame(
    timeframe: str,
    symbols: list[str],
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    try:
        where = ["timeframe = ?", "fwd_return_1 IS NOT NULL"]
        params = [timeframe]
        if symbols:
            placeholders = ", ".join(["?"] * len(symbols))
            where.append(f"symbol IN ({placeholders})")
            params.extend(symbols)

        df = pd.read_sql_query(
            f"""
            SELECT
                symbol,
                timeframe,
                datetime_utc,
                close,
                fwd_return_1,
                {", ".join(FEATURE_COLUMNS)}
            FROM {FEATURES_TABLE}
            WHERE {" AND ".join(where)}
            ORDER BY datetime_utc, symbol
            """,
            conn,
            params=tuple(params),
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df["fwd_return_1"] = pd.to_numeric(df["fwd_return_1"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["datetime_utc", "fwd_return_1"] + FEATURE_COLUMNS).copy()

    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        df = df[df["datetime_utc"] >= start_ts].copy()
    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC")
        df = df[df["datetime_utc"] <= end_ts].copy()

    df = df.sort_values(["datetime_utc", "symbol"]).reset_index(drop=True)
    return df


def apply_symbol_mapping(df: pd.DataFrame, mapping: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    out["symbol_code"] = out["symbol"].map(mapping)
    out = out.dropna(subset=["symbol_code"]).copy()
    out["symbol_code"] = out["symbol_code"].astype(int)
    return out


def _hit_ratio(frame: pd.DataFrame) -> float:
    active = frame[frame["signal_position"] != 0].copy()
    if active.empty:
        return 0.0
    active["hit"] = (active["signal_position"] * active["fwd_return_1"]) > 0
    return float(active["hit"].mean())


def load_oos_predictions(
    model_id: str,
    timeframe: str,
    symbols: list[str],
    validation_run_id: str | None,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.DataFrame, str]:
    selected_run_id = validation_run_id or get_latest_validation_run_id(model_id=model_id, timeframe=timeframe)
    if not selected_run_id:
        raise ValueError(f"No validation_predictions found for model_id={model_id} timeframe={timeframe}")

    conn = sqlite3.connect(DB_FILE)
    try:
        where = ["model_id = ?", "timeframe = ?", "validation_run_id = ?"]
        params: list[object] = [model_id, timeframe, selected_run_id]
        if symbols:
            placeholders = ", ".join(["?"] * len(symbols))
            where.append(f"symbol IN ({placeholders})")
            params.extend(symbols)

        df = pd.read_sql_query(
            f"""
            SELECT
                model_id,
                validation_run_id,
                symbol,
                timeframe,
                datetime_utc,
                y_true,
                y_pred,
                prob_short,
                prob_flat,
                prob_long,
                signal_position,
                fold_id
            FROM {VALIDATION_PREDICTIONS_TABLE}
            WHERE {" AND ".join(where)}
            ORDER BY datetime_utc, symbol
            """,
            conn,
            params=tuple(params),
        )
    finally:
        conn.close()

    if df.empty:
        return df, selected_run_id

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    for col in ["y_true", "y_pred", "signal_position", "fold_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["prob_short", "prob_flat", "prob_long"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(
        subset=["datetime_utc", "y_true", "y_pred", "signal_position", "prob_short", "prob_flat", "prob_long"]
    ).copy()

    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        df = df[df["datetime_utc"] >= start_ts].copy()
    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC")
        df = df[df["datetime_utc"] <= end_ts].copy()

    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)
    df["signal_position"] = df["signal_position"].astype(int)
    df["fold_id"] = df["fold_id"].astype(int)
    df = df.sort_values(["datetime_utc", "symbol"]).reset_index(drop=True)
    return df, selected_run_id


def _run_in_sample(args) -> dict:
    model_path = resolve_model_path(args.model_path, args.timeframe)
    artifact = joblib.load(model_path)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    symbol_mapping = artifact["symbol_mapping"]
    short_threshold = float(artifact.get("short_threshold", 0.55))
    long_threshold = float(artifact.get("long_threshold", 0.55))
    model_id = str(artifact.get("model_id", model_path.stem))

    symbols = [s.upper().strip() for s in args.symbols] if args.symbols else [s.upper() for s in SYMBOLS]
    frame = load_backtest_frame(
        timeframe=args.timeframe,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if frame.empty:
        raise ValueError("No backtest rows found with complete features.")

    frame = apply_symbol_mapping(frame, symbol_mapping)
    if frame.empty:
        raise ValueError("No symbols in backtest frame match model symbol mapping.")

    X = frame[feature_columns]
    probas = model.predict_proba(X)
    pred_class = model.predict(X)
    signal_position = probabilities_to_signal(
        probas=probas,
        short_threshold=short_threshold,
        long_threshold=long_threshold,
    )

    result = frame[["symbol", "timeframe", "datetime_utc", "close", "fwd_return_1"]].copy()
    result["pred_class"] = pred_class
    result["signal_position"] = signal_position
    result["signal_label"] = result["signal_position"].map(POSITION_TO_NAME)
    result["prob_short"] = probas[:, 0]
    result["prob_flat"] = probas[:, 1]
    result["prob_long"] = probas[:, 2]

    detailed_frame, equity_curve, econ = compute_economic_metrics(
        frame=result[["symbol", "datetime_utc", "signal_position", "fwd_return_1"]],
        timeframe=args.timeframe,
        cost_per_trade=COST_PER_TRADE,
    )

    merged_result = result.merge(
        detailed_frame[
            [
                "symbol",
                "datetime_utc",
                "prev_signal",
                "turnover",
                "transaction_cost",
                "gross_return",
                "strategy_return",
            ]
        ],
        on=["symbol", "datetime_utc"],
        how="left",
    )

    hit_ratio = _hit_ratio(merged_result)

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    details_path = REPORTS_DIR / f"backtest_signals_{model_id}_{stamp}.csv"
    curve_path = REPORTS_DIR / f"backtest_equity_{model_id}_{stamp}.csv"
    summary_path = REPORTS_DIR / f"backtest_summary_{model_id}_{stamp}.json"

    merged_result.to_csv(details_path, index=False)
    equity_curve.to_csv(curve_path, index=False)

    summary = {
        "mode": "in_sample",
        "model_id": model_id,
        "model_path": str(model_path),
        "timeframe": args.timeframe,
        "symbols": sorted(merged_result["symbol"].unique().tolist()),
        "start_datetime_utc": merged_result["datetime_utc"].min().isoformat(),
        "end_datetime_utc": merged_result["datetime_utc"].max().isoformat(),
        "rows": int(len(merged_result)),
        "economic": {
            "strategy_return": econ.strategy_return,
            "buy_hold_return": econ.buy_hold_return,
            "sharpe": econ.sharpe,
            "max_drawdown": econ.max_drawdown,
            "profit_factor": econ.profit_factor,
            "trade_count": econ.trade_count,
            "active_ratio": econ.active_ratio,
            "periods": econ.periods,
            "hit_ratio": hit_ratio,
        },
        "artifacts": {
            "details_csv": str(details_path),
            "equity_csv": str(curve_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _run_oos(args) -> dict:
    if args.model_id:
        model_id = args.model_id
    else:
        latest = get_latest_model(timeframe=args.timeframe, prefer_active=True)
        if not latest:
            raise ValueError("No model_id provided and no model found in registry.")
        model_id = str(latest["model_id"])

    symbols = [s.upper().strip() for s in args.symbols] if args.symbols else [s.upper() for s in SYMBOLS]
    preds, selected_run_id = load_oos_predictions(
        model_id=model_id,
        timeframe=args.timeframe,
        symbols=symbols,
        validation_run_id=args.validation_run_id,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if preds.empty:
        raise ValueError("No OOS prediction rows found for selected filters.")

    rows_before_dedup = int(len(preds))
    preds = (
        preds.sort_values(["datetime_utc", "symbol", "fold_id"])
        .drop_duplicates(subset=["symbol", "timeframe", "datetime_utc"], keep="first")
        .reset_index(drop=True)
    )
    rows_after_dedup = int(len(preds))

    # Need fwd_return_1 to build economic curve; join from features table by key.
    conn = sqlite3.connect(DB_FILE)
    try:
        returns_df = pd.read_sql_query(
            f"""
            SELECT symbol, timeframe, datetime_utc, fwd_return_1
            FROM {FEATURES_TABLE}
            WHERE timeframe = ?
            """,
            conn,
            params=(args.timeframe,),
        )
    finally:
        conn.close()
    returns_df["datetime_utc"] = pd.to_datetime(returns_df["datetime_utc"], utc=True, errors="coerce")
    returns_df["fwd_return_1"] = pd.to_numeric(returns_df["fwd_return_1"], errors="coerce")
    returns_df = returns_df.dropna(subset=["datetime_utc", "fwd_return_1"]).copy()

    merged = preds.merge(
        returns_df[["symbol", "timeframe", "datetime_utc", "fwd_return_1"]],
        on=["symbol", "timeframe", "datetime_utc"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("Could not join OOS predictions with fwd returns from features table.")

    detailed_frame, equity_curve, econ = compute_economic_metrics(
        frame=merged[["symbol", "datetime_utc", "signal_position", "fwd_return_1"]],
        timeframe=args.timeframe,
        cost_per_trade=COST_PER_TRADE,
    )
    result = merged.merge(
        detailed_frame[
            [
                "symbol",
                "datetime_utc",
                "prev_signal",
                "turnover",
                "transaction_cost",
                "gross_return",
                "strategy_return",
            ]
        ],
        on=["symbol", "datetime_utc"],
        how="left",
    )
    result["signal_label"] = result["signal_position"].map(POSITION_TO_NAME)

    hit_ratio = _hit_ratio(result)
    accuracy = float((result["y_true"] == result["y_pred"]).mean()) if len(result) else 0.0

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    details_path = REPORTS_DIR / f"backtest_oos_signals_{model_id}_{stamp}.csv"
    curve_path = REPORTS_DIR / f"backtest_oos_equity_{model_id}_{stamp}.csv"
    summary_path = REPORTS_DIR / f"backtest_oos_summary_{model_id}_{stamp}.json"

    result.to_csv(details_path, index=False)
    equity_curve.to_csv(curve_path, index=False)

    summary = {
        "mode": "oos",
        "model_id": model_id,
        "validation_run_id": selected_run_id,
        "timeframe": args.timeframe,
        "symbols": sorted(result["symbol"].unique().tolist()),
        "start_datetime_utc": result["datetime_utc"].min().isoformat(),
        "end_datetime_utc": result["datetime_utc"].max().isoformat(),
        "rows": int(len(result)),
        "rows_before_dedup": rows_before_dedup,
        "rows_after_dedup": rows_after_dedup,
        "classification": {
            "accuracy": accuracy,
        },
        "economic": {
            "strategy_return": econ.strategy_return,
            "buy_hold_return": econ.buy_hold_return,
            "sharpe": econ.sharpe,
            "max_drawdown": econ.max_drawdown,
            "profit_factor": econ.profit_factor,
            "trade_count": econ.trade_count,
            "active_ratio": econ.active_ratio,
            "periods": econ.periods,
            "hit_ratio": hit_ratio,
        },
        "artifacts": {
            "details_csv": str(details_path),
            "equity_csv": str(curve_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)

    model_entry = get_model_by_id(model_id)
    if model_entry:
        holdout_metrics = {}
        walk_forward_metrics = {}
        if isinstance(model_entry.get("metrics_json"), dict):
            holdout_metrics = model_entry["metrics_json"].get("holdout", {})
            walk_forward_metrics = model_entry["metrics_json"].get("walk_forward", {})

        acceptance = evaluate_model_acceptance(
            metrics_bundle={
                "holdout": holdout_metrics,
                "walk_forward": walk_forward_metrics,
                "backtest_oos": summary,
            }
        )
        update_model_evaluation(
            model_id=model_id,
            metrics={"backtest_oos": summary},
            status=acceptance["acceptance_status"],
            acceptance_status=acceptance["acceptance_status"],
            rejection_reasons=acceptance["rejection_reasons"],
            evaluation_scope=acceptance.get("evaluation_scope", "backtest_oos"),
        )
        summary["acceptance"] = acceptance
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    return summary


def main():
    args = parse_args()
    ensure_project_directories()
    init_research_tables()

    if args.mode == "in_sample":
        summary = _run_in_sample(args)
    else:
        summary = _run_oos(args)

    print(f"Backtest completed in mode={args.mode}")
    print(f"Summary: {summary['summary_path']}")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
