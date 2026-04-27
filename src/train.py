import argparse
import base64
import json
import sqlite3

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from config import (
    DB_FILE,
    FEATURES_TABLE,
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    LABEL_VERSION,
    MODEL_PARAMS,
    LONG_THRESHOLD,
    SHORT_THRESHOLD,
    TIMEFRAME,
    SYMBOLS,
    TEST_SIZE,
    COST_PER_TRADE,
    MODELS_DIR,
    REPORTS_DIR,
    MIN_TRAIN_ROWS,
)
from db_utils import init_research_tables, ensure_project_directories
from model_registry import register_model
from modeling_utils import (
    classification_metrics,
    compute_economic_metrics,
    probabilities_to_signal,
)
from strategy_evaluator import evaluate_model_acceptance


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM multiclass model from SQLite feature store.")
    parser.add_argument("--symbols", nargs="*", default=None, help="Symbols to include")
    parser.add_argument("--timeframe", default=TIMEFRAME, help="Timeframe filter")
    parser.add_argument("--test-size", type=int, default=TEST_SIZE, help="Number of latest datetimes for holdout")
    parser.add_argument("--model-id", type=str, default=None, help="Optional model id override")
    parser.add_argument("--min-train-rows", type=int, default=MIN_TRAIN_ROWS)
    parser.add_argument("--short-threshold", type=float, default=None, help="Override short probability threshold")
    parser.add_argument("--long-threshold", type=float, default=None, help="Override long probability threshold")
    parser.add_argument(
        "--model-params-json",
        type=str,
        default=None,
        help="JSON object to override LightGBM params, e.g. '{\"n_estimators\":500}'.",
    )
    parser.add_argument(
        "--model-params-b64",
        type=str,
        default=None,
        help="Base64-encoded JSON object for LightGBM params (PowerShell-safe alternative).",
    )
    return parser.parse_args()


def _resolve_model_params(model_params_json: str | None, model_params_b64: str | None) -> dict:
    params = dict(MODEL_PARAMS)
    payload = model_params_json
    if model_params_b64:
        payload = base64.b64decode(model_params_b64.encode("utf-8")).decode("utf-8")

    if not payload:
        return params
    overrides = json.loads(payload)
    if not isinstance(overrides, dict):
        raise ValueError("--model-params-json must decode to a JSON object.")
    params.update(overrides)
    return params


def _load_dataset(symbols: list[str], timeframe: str) -> pd.DataFrame:
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
                close,
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


def _split_train_test(df: pd.DataFrame, test_size_dates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(df["datetime_utc"].unique())
    if len(unique_dates) <= test_size_dates:
        raise ValueError(
            f"Not enough unique datetimes ({len(unique_dates)}) for test_size={test_size_dates}."
        )

    test_start = unique_dates[-test_size_dates]
    train_df = df[df["datetime_utc"] < test_start].copy()
    test_df = df[df["datetime_utc"] >= test_start].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Temporal split produced empty train/test segment.")
    return train_df, test_df


def _build_symbol_mapping(df: pd.DataFrame) -> dict[str, int]:
    symbols = sorted(df["symbol"].unique().tolist())
    return {symbol: idx for idx, symbol in enumerate(symbols)}


def _apply_symbol_code(df: pd.DataFrame, mapping: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    out["symbol_code"] = out["symbol"].map(mapping)
    out = out.dropna(subset=["symbol_code"]).copy()
    out["symbol_code"] = out["symbol_code"].astype(int)
    return out


def main():
    args = parse_args()
    ensure_project_directories()
    init_research_tables()

    symbols = [s.upper().strip() for s in args.symbols] if args.symbols else [s.upper() for s in SYMBOLS]
    df = _load_dataset(symbols=symbols, timeframe=args.timeframe)
    if df.empty:
        print("No rows found in features table with valid labels and complete feature values.")
        return

    train_df, test_df = _split_train_test(df, test_size_dates=args.test_size)
    if len(train_df) < args.min_train_rows:
        raise ValueError(
            f"Train rows too low ({len(train_df)}). Increase data or reduce filtering; min_train_rows={args.min_train_rows}."
        )

    symbol_mapping = _build_symbol_mapping(df)
    train_df = _apply_symbol_code(train_df, symbol_mapping)
    test_df = _apply_symbol_code(test_df, symbol_mapping)

    feature_cols = FEATURE_COLUMNS + ["symbol_code"]
    X_train = train_df[feature_cols]
    y_train = train_df["label_class"]
    X_test = test_df[feature_cols]
    y_test = test_df["label_class"]

    model_params = _resolve_model_params(args.model_params_json, args.model_params_b64)
    short_threshold = float(args.short_threshold) if args.short_threshold is not None else float(SHORT_THRESHOLD)
    long_threshold = float(args.long_threshold) if args.long_threshold is not None else float(LONG_THRESHOLD)

    model = LGBMClassifier(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    probas = model.predict_proba(X_test)
    signal_position = probabilities_to_signal(
        probas=probas,
        short_threshold=short_threshold,
        long_threshold=long_threshold,
    )

    accuracy, f1_macro = classification_metrics(y_true=y_test.values, y_pred=y_pred)

    econ_input = test_df[["symbol", "datetime_utc", "fwd_return_1"]].copy()
    econ_input["signal_position"] = signal_position
    _, portfolio_curve, econ = compute_economic_metrics(
        frame=econ_input,
        timeframe=args.timeframe,
        cost_per_trade=COST_PER_TRADE,
    )

    training_ts = pd.Timestamp.now(tz="UTC")
    model_id = args.model_id or f"lgbm_{args.timeframe}_{training_ts.strftime('%Y%m%d_%H%M%S')}"
    model_path = MODELS_DIR / f"{model_id}.joblib"

    artifact = {
        "model_id": model_id,
        "trained_at_utc": training_ts.isoformat(),
        "timeframe": args.timeframe,
        "symbols": sorted(df["symbol"].unique().tolist()),
        "feature_columns": feature_cols,
        "base_feature_columns": FEATURE_COLUMNS,
        "symbol_mapping": symbol_mapping,
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "feature_version": FEATURE_VERSION,
        "label_version": LABEL_VERSION,
        "model_params": model_params,
        "model": model,
    }
    joblib.dump(artifact, model_path)

    metrics = {
        "classification": {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
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
        },
        "rows": {
            "all": int(len(df)),
            "train": int(len(train_df)),
            "test": int(len(test_df)),
        },
        "split": {
            "train_start": train_df["datetime_utc"].min().isoformat(),
            "train_end": train_df["datetime_utc"].max().isoformat(),
            "test_start": test_df["datetime_utc"].min().isoformat(),
            "test_end": test_df["datetime_utc"].max().isoformat(),
        },
    }

    acceptance = evaluate_model_acceptance(metrics_bundle={"holdout": metrics})
    acceptance_status = acceptance["acceptance_status"]
    registry_status = acceptance_status

    report_path = REPORTS_DIR / f"train_{model_id}.json"
    report_payload = {
        "model_id": model_id,
        "model_path": str(model_path),
        "metrics": metrics,
        "acceptance": acceptance,
        "params": {
            "model_params": model_params,
            "short_threshold": short_threshold,
            "long_threshold": long_threshold,
            "cost_per_trade": COST_PER_TRADE,
        },
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    curve_path = REPORTS_DIR / f"train_holdout_curve_{model_id}.csv"
    portfolio_curve.to_csv(curve_path, index=False)

    register_model(
        model_id=model_id,
        symbol_scope=",".join(sorted(df["symbol"].unique().tolist())),
        timeframe=args.timeframe,
        train_start=train_df["datetime_utc"].min().isoformat(),
        train_end=train_df["datetime_utc"].max().isoformat(),
        test_start=test_df["datetime_utc"].min().isoformat(),
        test_end=test_df["datetime_utc"].max().isoformat(),
        feature_version=FEATURE_VERSION,
        label_version=LABEL_VERSION,
        model_path=str(model_path),
        metrics={"holdout": metrics},
        params={
            "model_params": model_params,
            "short_threshold": short_threshold,
            "long_threshold": long_threshold,
            "cost_per_trade": COST_PER_TRADE,
            "feature_columns": feature_cols,
        },
        status=registry_status,
        acceptance_status=acceptance_status,
        rejection_reasons=acceptance["rejection_reasons"],
        evaluation_scope=acceptance.get("evaluation_scope", "holdout"),
    )

    print(f"Model trained successfully: {model_id}")
    print(f"Model saved to: {model_path}")
    print(f"Report saved to: {report_path}")
    print(f"Holdout curve saved to: {curve_path}")
    print("Holdout metrics:")
    print(json.dumps(metrics, ensure_ascii=True, indent=2))
    print("Acceptance gating:")
    print(json.dumps(acceptance, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
