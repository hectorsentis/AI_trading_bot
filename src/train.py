import argparse
import base64
import json
import math
import sqlite3

import joblib
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV

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
    LOOKAHEAD_BARS,
    TP_MULTIPLIER,
    SL_MULTIPLIER,
    REQUIRE_TP_SL_ON_ENTRY,
    ENABLE_NATIVE_PREDICTION_MODELS,
    NATIVE_PREDICTION_HORIZON_BARS,
    NATIVE_PREDICTION_MIN_ROWS,
    NATIVE_QUANTILE_LEVELS,
    NATIVE_MODEL_PARAMS,
    ENABLE_PROBABILITY_CALIBRATION,
    PROBABILITY_CALIBRATION_METHOD,
    PROBABILITY_CALIBRATION_CV,
    PROBABILITY_CALIBRATION_MIN_ROWS,
    MODEL_FAMILY_PARAMS,
)
from labels import compute_native_regression_targets
from db_utils import init_research_tables, ensure_project_directories
from model_registry import register_model
from modeling_utils import (
    classification_metrics,
    compute_economic_metrics,
    probabilities_to_signal,
    predict_class_probabilities,
)


def instantiate_classifier(family: str, params: dict):
    """Factory for a sklearn-compatible classifier by family. Lazy imports keep optional
    dependencies (xgboost/catboost) out of the import path when a family isn't used."""
    fam = (family or "lgbm").lower()
    if fam == "lgbm":
        return LGBMClassifier(**params)
    if fam in ("xgb", "xgboost"):
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    if fam in ("catboost", "cat"):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params)
    if fam in ("rf", "random_forest", "randomforest"):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    if fam in ("et", "extra_trees", "extratrees"):
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**params)
    if fam in ("lr", "logreg", "logistic"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        # Scale-sensitive; wrap in a pipeline that still exposes predict_proba/classes_.
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**params))])
    raise ValueError(f"Unknown model family: {family}")


def _family_base_params(family: str) -> dict:
    return dict(MODEL_FAMILY_PARAMS.get((family or "lgbm").lower(), MODEL_PARAMS))
from strategy_evaluator import evaluate_model_acceptance
from temporal_utils import apply_label_embargo_cutoff, strict_train_validation_split


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
        help="Base64-encoded JSON object for model params (PowerShell-safe alternative).",
    )
    parser.add_argument(
        "--model-family",
        default="lgbm",
        help="Classifier family: lgbm | xgb | catboost | rf | et | lr. Base params come from MODEL_FAMILY_PARAMS.",
    )
    return parser.parse_args()


def _normalize_training_scope(raw: str) -> str:
    return (raw or "multi_symbol").replace("-", "_")


def _resolve_model_params(model_params_json: str | None, model_params_b64: str | None, family: str = "lgbm") -> dict:
    params = _family_base_params(family)
    payload = model_params_json
    if model_params_b64:
        payload = base64.b64decode(model_params_b64.encode("utf-8")).decode("utf-8")

    if not payload:
        return params
    overrides = json.loads(payload)
    # A stray 'model_family' key may ride along in the overrides payload; it is not a hyperparam.
    if isinstance(overrides, dict):
        overrides = {k: v for k, v in overrides.items() if k != "model_family"}
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


def _split_train_test(df: pd.DataFrame, test_size_dates: int, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_df, test_df, split_meta = strict_train_validation_split(
        df,
        timestamp_col="datetime_utc",
        timeframe=timeframe,
        training_cutoff_hours_before_now=TRAINING_CUTOFF_HOURS_BEFORE_NOW,
        validation_window_hours=VALIDATION_WINDOW_HOURS,
        fallback_test_size_dates=test_size_dates,
        lookahead_bars=LOOKAHEAD_BARS,
    )
    return train_df, test_df, split_meta.to_dict()


def _build_symbol_mapping(df: pd.DataFrame) -> dict[str, int]:
    symbols = sorted(df["symbol"].unique().tolist())
    return {symbol: idx for idx, symbol in enumerate(symbols)}


def _apply_symbol_code(df: pd.DataFrame, mapping: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    out["symbol_code"] = out["symbol"].map(mapping)
    out = out.dropna(subset=["symbol_code"]).copy()
    out["symbol_code"] = out["symbol_code"].astype(int)
    return out


def _has_all_classes(y: pd.Series) -> bool:
    return set(pd.Series(y).dropna().astype(int).unique().tolist()) == {0, 1, 2}


def _has_min_classes(y: pd.Series, n: int = 2) -> bool:
    return len(set(pd.Series(y).dropna().astype(int).unique().tolist())) >= int(n)


def _threshold_score(econ) -> float:
    pf = min(float(econ.profit_factor), 10.0) if econ.profit_factor != float("inf") else 10.0
    return float(
        econ.strategy_return
        + 0.02 * econ.sharpe
        + 0.01 * math.log1p(pf)
        - abs(econ.max_drawdown)
        + min(0.03, econ.active_ratio * 0.03)
    )


def _calibrate_thresholds(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    model_params: dict,
    timeframe: str,
    user_short_threshold: float | None,
    user_long_threshold: float | None,
    family: str = "lgbm",
) -> tuple[float, float, dict]:
    default_short = float(user_short_threshold) if user_short_threshold is not None else float(SHORT_THRESHOLD)
    default_long = float(user_long_threshold) if user_long_threshold is not None else float(LONG_THRESHOLD)
    if user_short_threshold is not None or user_long_threshold is not None:
        return default_short, default_long, {"enabled": False, "reason": "manual_threshold_override"}

    unique_dates = sorted(train_df["datetime_utc"].dropna().unique().tolist())
    min_dates = 120
    if len(unique_dates) < min_dates:
        return default_short, default_long, {"enabled": False, "reason": "not_enough_train_dates"}

    calib_dates = max(30, min(240, len(unique_dates) // 5))
    calib_start = pd.Timestamp(unique_dates[-calib_dates])
    if calib_start.tzinfo is None:
        calib_start = calib_start.tz_localize("UTC")
    else:
        calib_start = calib_start.tz_convert("UTC")
    fit_cutoff = apply_label_embargo_cutoff(calib_start, timeframe, LOOKAHEAD_BARS)
    fit_df = train_df[train_df["datetime_utc"] < fit_cutoff].copy()
    calib_df = train_df[train_df["datetime_utc"] >= calib_start].copy()
    if len(fit_df) < 300 or len(calib_df) < 30 or not _has_all_classes(fit_df["label_class"]):
        return default_short, default_long, {
            "enabled": False,
            "reason": "calibration_split_too_small_or_missing_classes",
            "fit_rows": int(len(fit_df)),
            "calibration_rows": int(len(calib_df)),
        }

    tmp_model = instantiate_classifier(family, model_params)
    tmp_model.fit(fit_df[feature_cols], fit_df["label_class"])
    probas = predict_class_probabilities({"model": tmp_model}, calib_df[feature_cols])

    grid = [0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.55, 0.60]
    best = {
        "short_threshold": default_short,
        "long_threshold": default_long,
        "score": float("-inf"),
        "economic": None,
    }
    for short_thr in grid:
        for long_thr in grid:
            signal_position = probabilities_to_signal(probas, short_threshold=short_thr, long_threshold=long_thr)
            econ_input = calib_df[["symbol", "datetime_utc", "fwd_return_1"]].copy()
            econ_input["signal_position"] = signal_position
            _, _, econ = compute_economic_metrics(
                frame=econ_input,
                timeframe=timeframe,
                cost_per_trade=COST_PER_TRADE,
            )
            if econ.trade_count < 3:
                continue
            score = _threshold_score(econ)
            if score > float(best["score"]):
                best = {
                    "short_threshold": float(short_thr),
                    "long_threshold": float(long_thr),
                    "score": float(score),
                    "economic": {
                        "strategy_return": econ.strategy_return,
                        "sharpe": econ.sharpe,
                        "max_drawdown": econ.max_drawdown,
                        "profit_factor": econ.profit_factor,
                        "trade_count": econ.trade_count,
                        "active_ratio": econ.active_ratio,
                    },
                }

    if best["economic"] is None:
        return default_short, default_long, {"enabled": False, "reason": "no_threshold_with_min_trades"}
    return float(best["short_threshold"]), float(best["long_threshold"]), {
        "enabled": True,
        "fit_rows": int(len(fit_df)),
        "calibration_rows": int(len(calib_df)),
        "calibration_start": calib_start.isoformat(),
        "fit_train_cutoff_after_embargo": fit_cutoff.isoformat(),
        "selected": best,
    }


def _per_symbol_economic_metrics(test_df: pd.DataFrame, signal_position, timeframe: str) -> dict:
    frame = test_df[["symbol", "datetime_utc", "fwd_return_1"]].copy()
    frame["signal_position"] = signal_position
    out: dict[str, dict] = {}
    for symbol, part in frame.groupby("symbol"):
        _, _, econ = compute_economic_metrics(part, timeframe=timeframe, cost_per_trade=COST_PER_TRADE)
        out[str(symbol)] = {
            "strategy_return": econ.strategy_return,
            "buy_hold_return": econ.buy_hold_return,
            "sharpe": econ.sharpe,
            "max_drawdown": econ.max_drawdown,
            "profit_factor": econ.profit_factor,
            "trade_count": econ.trade_count,
            "active_ratio": econ.active_ratio,
            "periods": econ.periods,
        }
    return out


def _train_native_models(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    precomputed_targets: pd.DataFrame | None = None,
) -> dict | None:
    """Train native return-distribution models alongside the direction classifier.

    Produces real (not synthetic) estimates the allocator/trade-builder consume:
      - expected return regression
      - quantile regressions (q05..q95)
      - MFE / MAE excursion regressions

    Targets are computed leakage-safe from the forward close path (see
    ``labels.compute_native_regression_targets``). Returns ``None`` (and the system falls
    back to the derived path) if there is not enough labeled training data.
    """
    targets = precomputed_targets if precomputed_targets is not None else compute_native_regression_targets(df, horizon_bars=horizon)
    train_keyed = train_df.merge(targets, on=["symbol", "datetime_utc"], how="left")
    labeled = train_keyed.dropna(subset=["native_ret", "native_mfe", "native_mae"]).copy()
    if len(labeled) < int(NATIVE_PREDICTION_MIN_ROWS):
        return None

    X = labeled[feature_cols]
    quantile_models: dict[str, object] = {}
    for alpha in NATIVE_QUANTILE_LEVELS:
        q_params = dict(NATIVE_MODEL_PARAMS)
        q_params.update({"objective": "quantile", "alpha": float(alpha)})
        q_model = LGBMRegressor(**q_params)
        q_model.fit(X, labeled["native_ret"])
        quantile_models[f"{alpha:.2f}"] = q_model

    expected_return_model = LGBMRegressor(**NATIVE_MODEL_PARAMS)
    expected_return_model.fit(X, labeled["native_ret"])
    mfe_model = LGBMRegressor(**NATIVE_MODEL_PARAMS)
    mfe_model.fit(X, labeled["native_mfe"])
    mae_model = LGBMRegressor(**NATIVE_MODEL_PARAMS)
    mae_model.fit(X, labeled["native_mae"])

    return {
        "horizon_bars": int(horizon),
        "feature_columns": list(feature_cols),
        "quantile_levels": [float(a) for a in NATIVE_QUANTILE_LEVELS],
        "expected_return": expected_return_model,
        "quantiles": quantile_models,
        "mfe": mfe_model,
        "mae": mae_model,
        "trained_rows": int(len(labeled)),
        "target_note": "native_ret/mfe/mae from forward close path; leakage-safe under embargo",
    }


def _fit_probability_calibrator(X_train, y_train, model_params: dict, family: str = "lgbm"):
    """Fit a calibrated multiclass classifier so reported probabilities are trustworthy.

    Uses internal cross-validated calibration (no leakage). Returns ``None`` (and the system
    uses raw model probabilities) when disabled, when there are too few rows, when not all three
    classes are present, or if calibration fails. Handles both the modern (`estimator=`) and
    legacy (`base_estimator=`) scikit-learn signatures.
    """
    if not ENABLE_PROBABILITY_CALIBRATION:
        return None
    if len(X_train) < int(PROBABILITY_CALIBRATION_MIN_ROWS) or not _has_all_classes(y_train):
        return None
    for kwargs in (
        {"estimator": instantiate_classifier(family, model_params)},
        {"base_estimator": instantiate_classifier(family, model_params)},
    ):
        try:
            calibrator = CalibratedClassifierCV(
                method=PROBABILITY_CALIBRATION_METHOD,
                cv=int(PROBABILITY_CALIBRATION_CV),
                **kwargs,
            )
            calibrator.fit(X_train, y_train)
            return calibrator
        except TypeError:
            continue  # try the other signature
        except Exception:
            return None
    return None


def train_one_scope(args, symbols: list[str], model_id_override: str | None = None) -> dict:
    df = _load_dataset(symbols=symbols, timeframe=args.timeframe)
    if df.empty:
        raise ValueError(f"No rows found in features table for symbols={symbols} timeframe={args.timeframe}.")

    train_df, test_df, split_meta = _split_train_test(df, test_size_dates=args.test_size, timeframe=args.timeframe)
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

    family = (getattr(args, "model_family", None) or "lgbm").lower()
    model_params = _resolve_model_params(args.model_params_json, args.model_params_b64, family=family)
    short_threshold, long_threshold, threshold_calibration = _calibrate_thresholds(
        train_df=train_df,
        feature_cols=feature_cols,
        model_params=model_params,
        timeframe=args.timeframe,
        user_short_threshold=args.short_threshold,
        user_long_threshold=args.long_threshold,
        family=family,
    )
    if not _has_min_classes(y_train, 2):
        raise ValueError(
            f"Training labels for symbols={symbols} contain fewer than two classes; "
            "refusing to train a brittle model."
        )

    model = instantiate_classifier(family, model_params)
    model.fit(X_train, y_train)

    native_models = None
    if ENABLE_NATIVE_PREDICTION_MODELS:
        native_models = _train_native_models(
            df=df,
            train_df=train_df,
            feature_cols=feature_cols,
            horizon=int(NATIVE_PREDICTION_HORIZON_BARS),
        )

    calibrator = _fit_probability_calibrator(X_train, y_train, model_params, family=family)

    # Canonical [SHORT, FLAT, LONG] probabilities regardless of family/classes_ ordering.
    import numpy as _np
    y_pred = _np.asarray(model.predict(X_test)).ravel()
    probas = predict_class_probabilities({"model": model}, X_test)
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
        model_id = f"{family}_{scope_tag}_{args.timeframe}_{training_ts.strftime('%Y%m%d_%H%M%S')}"
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
        "model_family": family,
        "model_params": model_params,
        "label_policy": {
            "type": "atr_triple_barrier_with_mandatory_entry_tp_sl",
            "lookahead_bars": int(LOOKAHEAD_BARS),
            "tp_multiplier": float(TP_MULTIPLIER),
            "sl_multiplier": float(SL_MULTIPLIER),
            "require_tp_sl_on_entry": bool(REQUIRE_TP_SL_ON_ENTRY),
        },
        "model": model,
        "native_models": native_models,
        "has_native_models": native_models is not None,
        "calibrator": calibrator,
        "has_calibrator": calibrator is not None,
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
            **split_meta,
        },
        "training_scope": training_scope,
        "symbols": included_symbols,
        "per_symbol_economic": _per_symbol_economic_metrics(test_df, signal_position, args.timeframe),
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
            "threshold_calibration": threshold_calibration,
            "label_policy": artifact["label_policy"],
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
            "model_family": family,
            "model_params": model_params,
            "short_threshold": short_threshold,
            "long_threshold": long_threshold,
            "cost_per_trade": COST_PER_TRADE,
            "feature_columns": feature_cols,
            "training_scope": training_scope,
            "symbols": included_symbols,
            "label_policy": artifact["label_policy"],
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
