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
    TIMEFRAME,
    SYMBOLS,
    COST_PER_TRADE,
    REPORTS_DIR,
)
from db_utils import init_research_tables, ensure_project_directories
from model_registry import get_latest_model
from modeling_utils import POSITION_TO_NAME, compute_economic_metrics, probabilities_to_signal


def parse_args():
    parser = argparse.ArgumentParser(description="Economic backtest from feature store using a trained model.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to .joblib model artifact")
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

    latest = get_latest_model(timeframe=timeframe)
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


def main():
    args = parse_args()
    ensure_project_directories()
    init_research_tables()

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
        print("No backtest rows found with complete features.")
        return

    frame = apply_symbol_mapping(frame, symbol_mapping)
    if frame.empty:
        print("No symbols in backtest frame match model symbol mapping.")
        return

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

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    details_path = REPORTS_DIR / f"backtest_signals_{model_id}_{stamp}.csv"
    curve_path = REPORTS_DIR / f"backtest_equity_{model_id}_{stamp}.csv"
    summary_path = REPORTS_DIR / f"backtest_summary_{model_id}_{stamp}.json"

    merged_result.to_csv(details_path, index=False)
    equity_curve.to_csv(curve_path, index=False)

    summary = {
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
        },
        "artifacts": {
            "details_csv": str(details_path),
            "equity_csv": str(curve_path),
        },
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Backtest completed for model_id={model_id}")
    print(f"Signals detail: {details_path}")
    print(f"Equity curve: {curve_path}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
