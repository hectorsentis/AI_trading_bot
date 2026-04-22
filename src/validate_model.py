import argparse
import json
import sqlite3

import pandas as pd
from lightgbm import LGBMClassifier

from config import (
    DB_FILE,
    FEATURES_TABLE,
    FEATURE_COLUMNS,
    MODEL_PARAMS,
    LONG_THRESHOLD,
    SHORT_THRESHOLD,
    TIMEFRAME,
    SYMBOLS,
    TRAIN_SIZE,
    TEST_SIZE,
    RETRAIN_STEP,
    COST_PER_TRADE,
    REPORTS_DIR,
    MIN_TRAIN_ROWS,
)
from db_utils import init_research_tables, ensure_project_directories
from modeling_utils import (
    classification_metrics,
    compute_economic_metrics,
    probabilities_to_signal,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward validation (strict temporal) for multiclass model.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE, help="Train window size in unique datetimes")
    parser.add_argument("--test-size", type=int, default=TEST_SIZE, help="Test window size in unique datetimes")
    parser.add_argument("--step-size", type=int, default=RETRAIN_STEP, help="Step between folds in unique datetimes")
    parser.add_argument("--max-folds", type=int, default=25, help="Limit fold count to keep runtime reasonable")
    parser.add_argument("--min-train-rows", type=int, default=MIN_TRAIN_ROWS)
    return parser.parse_args()


def load_dataset(symbols: list[str], timeframe: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    try:
        where = ["timeframe = ?", "label_class IS NOT NULL"]
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
                fwd_return_1,
                label_class,
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
    df["label_class"] = pd.to_numeric(df["label_class"], errors="coerce")
    df["fwd_return_1"] = pd.to_numeric(df["fwd_return_1"], errors="coerce")
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["datetime_utc", "label_class", "fwd_return_1"] + FEATURE_COLUMNS).copy()
    df["label_class"] = df["label_class"].astype(int)
    df = df.sort_values(["datetime_utc", "symbol"]).reset_index(drop=True)
    return df


def build_symbol_mapping(df: pd.DataFrame) -> dict[str, int]:
    symbols = sorted(df["symbol"].unique().tolist())
    return {symbol: idx for idx, symbol in enumerate(symbols)}


def add_symbol_code(df: pd.DataFrame, mapping: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    out["symbol_code"] = out["symbol"].map(mapping)
    out = out.dropna(subset=["symbol_code"]).copy()
    out["symbol_code"] = out["symbol_code"].astype(int)
    return out


def walk_forward_splits(
    unique_dates: list[pd.Timestamp],
    train_size: int,
    test_size: int,
    step_size: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    splits = []
    start_idx = train_size
    end_limit = len(unique_dates) - test_size + 1
    for idx in range(start_idx, end_limit, step_size):
        test_start = unique_dates[idx]
        test_end = unique_dates[idx + test_size - 1]
        splits.append((test_start, test_end))
    return splits


def main():
    args = parse_args()
    ensure_project_directories()
    init_research_tables()

    symbols = [s.upper().strip() for s in args.symbols] if args.symbols else [s.upper() for s in SYMBOLS]
    df = load_dataset(symbols=symbols, timeframe=args.timeframe)
    if df.empty:
        print("No usable rows found in features table for validation.")
        return

    unique_dates = sorted(df["datetime_utc"].unique())
    if len(unique_dates) < args.train_size + args.test_size:
        raise ValueError(
            f"Not enough unique datetimes ({len(unique_dates)}) for train_size={args.train_size} and test_size={args.test_size}."
        )

    splits = walk_forward_splits(unique_dates, args.train_size, args.test_size, args.step_size)
    if args.max_folds and len(splits) > args.max_folds:
        splits = splits[-args.max_folds:]

    symbol_mapping = build_symbol_mapping(df)
    feature_cols = FEATURE_COLUMNS + ["symbol_code"]

    fold_rows = []
    prediction_rows = []

    print(
        f"Running walk-forward validation: folds={len(splits)} "
        f"train_size={args.train_size} test_size={args.test_size} step={args.step_size}"
    )

    for fold_id, (test_start, test_end) in enumerate(splits, start=1):
        train_df = df[df["datetime_utc"] < test_start].copy()
        test_df = df[(df["datetime_utc"] >= test_start) & (df["datetime_utc"] <= test_end)].copy()

        if len(train_df) < args.min_train_rows or test_df.empty:
            continue

        train_df = add_symbol_code(train_df, symbol_mapping)
        test_df = add_symbol_code(test_df, symbol_mapping)
        if train_df.empty or test_df.empty:
            continue

        model = LGBMClassifier(**MODEL_PARAMS)
        model.fit(train_df[feature_cols], train_df["label_class"])

        y_true = test_df["label_class"].values
        y_pred = model.predict(test_df[feature_cols])
        probas = model.predict_proba(test_df[feature_cols])
        signal_position = probabilities_to_signal(
            probas=probas,
            short_threshold=SHORT_THRESHOLD,
            long_threshold=LONG_THRESHOLD,
        )

        accuracy, f1_macro = classification_metrics(y_true=y_true, y_pred=y_pred)

        econ_input = test_df[["symbol", "datetime_utc", "fwd_return_1"]].copy()
        econ_input["signal_position"] = signal_position
        _, _, econ = compute_economic_metrics(
            frame=econ_input,
            timeframe=args.timeframe,
            cost_per_trade=COST_PER_TRADE,
        )

        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_start": train_df["datetime_utc"].min().isoformat(),
                "train_end": train_df["datetime_utc"].max().isoformat(),
                "test_start": test_df["datetime_utc"].min().isoformat(),
                "test_end": test_df["datetime_utc"].max().isoformat(),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "strategy_return": econ.strategy_return,
                "buy_hold_return": econ.buy_hold_return,
                "sharpe": econ.sharpe,
                "max_drawdown": econ.max_drawdown,
                "profit_factor": econ.profit_factor,
                "trade_count": econ.trade_count,
                "active_ratio": econ.active_ratio,
                "periods": econ.periods,
            }
        )

        pred_frame = test_df[["symbol", "datetime_utc", "fwd_return_1", "label_class"]].copy()
        pred_frame["fold_id"] = fold_id
        pred_frame["pred_class"] = y_pred
        pred_frame["signal_position"] = signal_position
        pred_frame["prob_short"] = probas[:, 0]
        pred_frame["prob_flat"] = probas[:, 1]
        pred_frame["prob_long"] = probas[:, 2]
        prediction_rows.append(pred_frame)

        print(
            f"fold={fold_id} test=[{test_start} -> {test_end}] "
            f"acc={accuracy:.4f} f1={f1_macro:.4f} "
            f"strat_ret={econ.strategy_return:.4f} sharpe={econ.sharpe:.4f}"
        )

    if not fold_rows:
        print("No valid folds generated. Check data sufficiency and parameters.")
        return

    folds_df = pd.DataFrame(fold_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)

    overall_accuracy, overall_f1 = classification_metrics(
        y_true=predictions_df["label_class"].values,
        y_pred=predictions_df["pred_class"].values,
    )
    _, portfolio_curve, overall_econ = compute_economic_metrics(
        frame=predictions_df[["symbol", "datetime_utc", "signal_position", "fwd_return_1"]],
        timeframe=args.timeframe,
        cost_per_trade=COST_PER_TRADE,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    folds_path = REPORTS_DIR / f"validation_folds_{args.timeframe}_{stamp}.csv"
    preds_path = REPORTS_DIR / f"validation_predictions_{args.timeframe}_{stamp}.csv"
    curve_path = REPORTS_DIR / f"validation_equity_{args.timeframe}_{stamp}.csv"
    report_path = REPORTS_DIR / f"validation_summary_{args.timeframe}_{stamp}.json"

    folds_df.to_csv(folds_path, index=False)
    predictions_df.to_csv(preds_path, index=False)
    portfolio_curve.to_csv(curve_path, index=False)

    summary = {
        "timeframe": args.timeframe,
        "symbols": sorted(df["symbol"].unique().tolist()),
        "folds": int(len(folds_df)),
        "train_size": args.train_size,
        "test_size": args.test_size,
        "step_size": args.step_size,
        "max_folds": args.max_folds,
        "classification": {
            "overall_accuracy": overall_accuracy,
            "overall_f1_macro": overall_f1,
            "fold_accuracy_mean": float(folds_df["accuracy"].mean()),
            "fold_accuracy_median": float(folds_df["accuracy"].median()),
            "fold_f1_mean": float(folds_df["f1_macro"].mean()),
            "fold_f1_median": float(folds_df["f1_macro"].median()),
        },
        "economic": {
            "overall_strategy_return": overall_econ.strategy_return,
            "overall_buy_hold_return": overall_econ.buy_hold_return,
            "overall_sharpe": overall_econ.sharpe,
            "overall_max_drawdown": overall_econ.max_drawdown,
            "overall_profit_factor": overall_econ.profit_factor,
            "overall_trade_count": overall_econ.trade_count,
            "fold_strategy_return_mean": float(folds_df["strategy_return"].mean()),
            "fold_strategy_return_median": float(folds_df["strategy_return"].median()),
            "fold_sharpe_mean": float(folds_df["sharpe"].mean()),
            "fold_max_drawdown_worst": float(folds_df["max_drawdown"].min()),
        },
        "rows": {
            "dataset_rows": int(len(df)),
            "prediction_rows": int(len(predictions_df)),
        },
        "artifacts": {
            "folds_csv": str(folds_path),
            "predictions_csv": str(preds_path),
            "equity_csv": str(curve_path),
        },
    }

    report_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Validation summary saved to: {report_path}")
    print(f"Folds metrics saved to: {folds_path}")
    print(f"Predictions saved to: {preds_path}")
    print(f"Equity curve saved to: {curve_path}")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
