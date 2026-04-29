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
    TRAINING_CUTOFF_HOURS_BEFORE_NOW,
    VALIDATION_WINDOW_HOURS,
    TRAINING_SCOPE,
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
    parser.add_argument(
        "--training-scope",
        choices=["multi-symbol", "multi_symbol", "per-symbol", "per_symbol", "both"],
        default=TRAINING_SCOPE,
        help="multi-symbol trains one model across all symbols with symbol_code; per-symbol trains one model per crypto; both runs both scopes.",
    )
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


def _normalize_training_scope(raw: str) -> str:
    return (raw or "multi_symbol").replace("-", "_")


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
    reference_now = pd.Timestamp.now(tz="UTC")
    dataset_max = df["datetime_utc"].max()
    # If the local dataset is historical/stale, anchor the temporal holdout to dataset_max
    # so research remains usable without ever training on rows after the validation window.
    if dataset_max < reference_now - pd.Timedelta(hours=TRAINING_CUTOFF_HOURS_BEFORE_NOW + VALIDATION_WINDOW_HOURS):
        reference_now = dataset_max + pd.Timedelta(hours=TRAINING_CUTOFF_HOURS_BEFORE_NOW)
    validation_end = reference_now - pd.Timedelta(hours=TRAINING_CUTOFF_HOURS_BEFORE_NOW)
    validation_start = validation_end - pd.Timedelta(hours=VALIDATION_WINDOW_HOURS)
    train_df = df[df["datetime_utc"] < validation_start].copy()
    test_df = df[(df["datetime_utc"] >= validation_start) & (df["datetime_utc"] < validation_end)].copy()
    if not train_df.empty and not test_df.empty:
        return train_df, test_df

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



def train_one_scope(args, symbols: list[str], model_id_override: str | None = None) -> dict:
    df = _load_dataset(symbols=symbols, timeframe=args.timeframe)
    if df.empty:
        raise ValueError(f"No rows found in features table for symbols={symbols} timeframe={args.timeframe}.")

    train_df, test_df = _split_train_test(df, test_size_dates=args.test_size)
    if len(train_df) < args.min_train_rows:
        raise ValueError(
            f"Train rows too low ({len(train_df)}) for symbols={symbols}. "
            f"Increase data or reduce filtering; min_train_rows={args.min_train_rows}."
        )

    training_scope = _normalize_training_scope(args.training_scope)
    included_symbols = sorted(df["symbol"].unique().tolist())
    symbol_mapping = _build_symbol_mapping(df)
    train_df = _apply_symbol_code(train_df, symbol_mapping)
    test_df = _apply_symbol_code(test_df, symbol_mapping)

    # Both scopes keep symbol_code for artifact compatibility. In per_symbol it is a
    # constant 0, while multi_symbol uses it to let one model learn cross-symbol effects.
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
    if model_id_override:
        model_id = model_id_override
    elif args.model_id:
        model_id = args.model_id
    else:
        scope_tag = "multi" if training_scope == "multi_symbol" else included_symbols[0].lower()
        model_id = f"lgbm_{scope_tag}_{args.timeframe}_{training_ts.strftime('%Y%m%d_%H%M%S')}"
    model_path = MODELS_DIR / f"{model_id}.joblib"

    artifact = {
        "model_id": model_id,
        "trained_at_utc": training_ts.isoformat(),
        "training_scope": training_scope,
        "timeframe": args.timeframe,
        "symbols": included_symbols,
        "symbol_scope": ",".join(included_symbols),
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
        "training_scope": training_scope,
        "symbols": included_symbols,
    }

    acceptance = evaluate_model_acceptance(metrics_bundle={"holdout": metrics})
    acceptance_status = "candidate"
    registry_status = "candidate"

    report_path = REPORTS_DIR / f"train_{model_id}.json"
    report_payload = {
        "model_id": model_id,
        "model_path": str(model_path),
        "training_scope": training_scope,
        "symbol_scope": ",".join(included_symbols),
        "symbols": included_symbols,
        "metrics": metrics,
        "acceptance": acceptance,
        "params": {
            "model_params": model_params,
            "short_threshold": short_threshold,
            "long_threshold": long_threshold,
            "cost_per_trade": COST_PER_TRADE,
            "training_scope": training_scope,
        },
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    curve_path = REPORTS_DIR / f"train_holdout_curve_{model_id}.csv"
    portfolio_curve.to_csv(curve_path, index=False)

    register_model(
        model_id=model_id,
        symbol_scope=",".join(included_symbols),
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
            "training_scope": training_scope,
            "symbols": included_symbols,
        },
        status=registry_status,
        acceptance_status=acceptance_status,
        rejection_reasons=[],
        evaluation_scope=acceptance.get("evaluation_scope", "holdout"),
        training_scope=training_scope,
        symbols=included_symbols,
    )

    return {
        "model_id": model_id,
        "model_path": str(model_path),
        "report_path": str(report_path),
        "curve_path": str(curve_path),
        "training_scope": training_scope,
        "symbol_scope": ",".join(included_symbols),
        "symbols": included_symbols,
        "metrics": metrics,
        "acceptance": acceptance,
    }


def main():
    args = parse_args()
    ensure_project_directories()
    init_research_tables()

    symbols = [s.upper().strip() for s in args.symbols] if args.symbols else [s.upper() for s in SYMBOLS]
    training_scope = _normalize_training_scope(args.training_scope)

    if training_scope == "both":
        results = []
        multi_args = argparse.Namespace(**vars(args))
        multi_args.training_scope = "multi_symbol"
        results.append(train_one_scope(args=multi_args, symbols=symbols))
        per_args = argparse.Namespace(**vars(args))
        per_args.training_scope = "per_symbol"
        for symbol in symbols:
            model_id = None
            if args.model_id:
                model_id = f"{args.model_id}_{symbol.lower()}"
            result = train_one_scope(args=per_args, symbols=[symbol], model_id_override=model_id)
            results.append(result)
        print(json.dumps({"training_scope": "both", "models": results}, ensure_ascii=True, indent=2))
        return

    if training_scope == "per_symbol":
        results = []
        if args.model_id and len(symbols) > 1:
            print("--model-id with --training-scope per-symbol and multiple symbols will be suffixed per symbol.")
        for symbol in symbols:
            model_id = None
            if args.model_id:
                model_id = args.model_id if len(symbols) == 1 else f"{args.model_id}_{symbol.lower()}"
            result = train_one_scope(args=args, symbols=[symbol], model_id_override=model_id)
            results.append(result)
            print(f"Model trained successfully: {result['model_id']} scope=per_symbol symbol={symbol}")
            print(f"Model saved to: {result['model_path']}")
        print(json.dumps({"training_scope": training_scope, "models": results}, ensure_ascii=True, indent=2))
        return

    result = train_one_scope(args=args, symbols=symbols)
    print(f"Model trained successfully: {result['model_id']} scope=multi_symbol")
    print(f"Model saved to: {result['model_path']}")
    print(f"Report saved to: {result['report_path']}")
    print(f"Holdout curve saved to: {result['curve_path']}")
    print("Holdout metrics:")
    print(json.dumps(result["metrics"], ensure_ascii=True, indent=2))
    print("Acceptance gating:")
    print(json.dumps(result["acceptance"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
