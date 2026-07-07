import argparse
import base64
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import (
    MODEL_PARAMS,
    MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE,
    MODEL_POOL_VALIDATION_MAX_FOLDS,
    SYMBOLS,
    TARGET_ACCEPTED_MODELS,
    TIMEFRAME,
    TIMEFRAMES,
    MODEL_MAINTENANCE_INTERVAL_SECONDS,
    TRAINING_SCOPE,
    ENABLED_MODEL_FAMILIES,
)
from model_registry import count_accepted_models, get_model_by_id, list_accepted_models, set_active_model
from runtime_status import record_event, update_status


@dataclass
class TrainingTrial:
    short_threshold: float
    long_threshold: float
    params: dict
    family: str = "lgbm"


def _python_env() -> dict:
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parent)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not current else f"{src_path}{os.pathsep}{current}"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LOKY_MAX_CPU_COUNT", "4")
    return env


def _run_command(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parent.parent,
        env=_python_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def _b64_json(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


# Per-family candidate trials. `params` are OVERRIDES only; train.py merges them onto the family
# base from MODEL_FAMILY_PARAMS. Thresholds default trade-friendly (0.40); a couple of variants
# per family give diversity without exploding runtime.
_FAMILY_TRIALS: dict[str, list[tuple[float, float, dict]]] = {
    "lgbm": [
        (0.40, 0.40, {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 40}),
        (0.34, 0.34, {"n_estimators": 800, "learning_rate": 0.02, "num_leaves": 63, "min_child_samples": 80, "colsample_bytree": 0.75}),
    ],
    "xgb": [
        (0.40, 0.40, {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.03}),
        (0.34, 0.34, {"n_estimators": 600, "max_depth": 8, "learning_rate": 0.02, "subsample": 0.80}),
    ],
    "catboost": [
        (0.40, 0.40, {"iterations": 400, "depth": 6, "learning_rate": 0.03}),
        (0.34, 0.34, {"iterations": 600, "depth": 8, "learning_rate": 0.02}),
    ],
    "rf": [
        (0.40, 0.40, {"n_estimators": 400, "min_samples_leaf": 40}),
    ],
    "et": [
        (0.40, 0.40, {"n_estimators": 500, "min_samples_leaf": 40}),
    ],
    "lr": [
        (0.40, 0.40, {"C": 1.0}),
    ],
}


def build_trial_plan(max_attempts: int) -> list[TrainingTrial]:
    """Round-robin across enabled families so a diverse pool is tried even with a small budget."""
    families = [f for f in ENABLED_MODEL_FAMILIES if f in _FAMILY_TRIALS] or ["lgbm"]
    per_family = [
        [(fam, s, l, ov) for (s, l, ov) in _FAMILY_TRIALS[fam]]
        for fam in families
    ]
    trials: list[TrainingTrial] = []
    for round_idx in range(max((len(col) for col in per_family), default=0)):
        for col in per_family:
            if round_idx < len(col):
                fam, short_thr, long_thr, overrides = col[round_idx]
                trials.append(TrainingTrial(short_threshold=short_thr, long_threshold=long_thr, params=dict(overrides), family=fam))
    return trials[: max(0, int(max_attempts))]


def _normalize_training_scope(raw: str) -> str:
    scope = (raw or "both").replace("-", "_")
    return scope if scope in {"multi_symbol", "per_symbol", "both"} else "both"


def _eligible_existing_accepted(timeframe: str, training_scope: str | None = None, symbols: list[str] | None = None) -> list[dict]:
    models = []
    for model in list_accepted_models(timeframe=timeframe, training_scope=training_scope, symbols=symbols):
        path = Path(str(model.get("model_path", "")))
        metrics = model.get("metrics_json") if isinstance(model.get("metrics_json"), dict) else {}
        has_lifecycle = bool(
            metrics.get("backtest_oos", {}).get("trade_lifecycle", {}).get("metrics")
            or metrics.get("walk_forward", {}).get("trade_lifecycle", {}).get("metrics")
        )
        if path.exists() and has_lifecycle:
            models.append(model)
    return models


def maintain_model_pool(
    symbols: list[str],
    timeframe: str,
    target_accepted_models: int = TARGET_ACCEPTED_MODELS,
    max_attempts: int = MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE,
    validation_max_folds: int = MODEL_POOL_VALIDATION_MAX_FOLDS,
    training_scope: str = TRAINING_SCOPE,
) -> dict:
    training_scope = _normalize_training_scope(training_scope)
    if training_scope == "both":
        multi = maintain_model_pool(
            symbols=symbols,
            timeframe=timeframe,
            target_accepted_models=target_accepted_models,
            max_attempts=max_attempts,
            validation_max_folds=validation_max_folds,
            training_scope="multi_symbol",
        )
        per = maintain_model_pool(
            symbols=symbols,
            timeframe=timeframe,
            target_accepted_models=target_accepted_models,
            max_attempts=max_attempts,
            validation_max_folds=validation_max_folds,
            training_scope="per_symbol",
        )
        return {
            "training_scope": "both",
            "multi_symbol": multi,
            "per_symbol": per,
            "target_met": bool(multi.get("target_met")) and bool(per.get("target_met")),
        }
    if training_scope == "per_symbol" and len(symbols) > 1:
        per_symbol = {}
        for symbol in symbols:
            per_symbol[symbol] = maintain_model_pool(
                symbols=[symbol],
                timeframe=timeframe,
                target_accepted_models=target_accepted_models,
                max_attempts=max_attempts,
                validation_max_folds=validation_max_folds,
                training_scope=training_scope,
            )
        return {
            "training_scope": training_scope,
            "per_symbol": per_symbol,
            "target_met": all(item.get("target_met") for item in per_symbol.values()),
        }
    accepted_before = _eligible_existing_accepted(timeframe, training_scope=training_scope, symbols=symbols)
    needed = max(0, int(target_accepted_models) - len(accepted_before))
    summary = {
        "target_accepted_models": int(target_accepted_models),
        "accepted_before": [m["model_id"] for m in accepted_before],
        "attempts": [],
        "accepted_after": [],
        "needed_before_attempts": needed,
        "training_scope": training_scope,
        "symbols": symbols,
    }
    if needed <= 0:
        summary["accepted_after"] = summary["accepted_before"]
        return summary

    for idx, trial in enumerate(build_trial_plan(max_attempts), start=1):
        if count_accepted_models(timeframe, training_scope=training_scope, symbols=symbols) >= int(target_accepted_models):
            break

        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
        scope_tag = "multi" if training_scope == "multi_symbol" else symbols[0].lower()
        model_id = f"auto_{trial.family}_{scope_tag}_{timeframe}_{stamp}_{idx:02d}"
        params_b64 = _b64_json(trial.params)
        symbols_args = [s.upper().strip() for s in symbols]

        attempt = {
            "attempt": idx,
            "model_id": model_id,
            "short_threshold": trial.short_threshold,
            "long_threshold": trial.long_threshold,
            "params": trial.params,
            "stages": {},
        }

        train_cmd = [
            sys.executable,
            "src/train.py",
            "--symbols",
            *symbols_args,
            "--timeframe",
            timeframe,
            "--model-id",
            model_id,
            "--training-scope",
            training_scope,
            "--short-threshold",
            str(trial.short_threshold),
            "--long-threshold",
            str(trial.long_threshold),
            "--model-params-b64",
            params_b64,
            "--model-family",
            str(trial.family),
        ]
        code, out = _run_command(train_cmd)
        attempt["stages"]["train"] = {"returncode": code, "tail": out[-4000:]}
        if code != 0:
            summary["attempts"].append(attempt)
            continue

        validate_cmd = [
            sys.executable,
            "src/validate_model.py",
            "--symbols",
            *symbols_args,
            "--timeframe",
            timeframe,
            "--model-id",
            model_id,
            "--max-folds",
            str(validation_max_folds),
            "--short-threshold",
            str(trial.short_threshold),
            "--long-threshold",
            str(trial.long_threshold),
            "--model-params-b64",
            params_b64,
            "--model-family",
            str(trial.family),
        ]
        code, out = _run_command(validate_cmd)
        attempt["stages"]["validate"] = {"returncode": code, "tail": out[-4000:]}
        if code != 0:
            summary["attempts"].append(attempt)
            continue

        backtest_cmd = [
            sys.executable,
            "src/backtest.py",
            "--mode",
            "oos",
            "--timeframe",
            timeframe,
            "--model-id",
            model_id,
            "--symbols",
            *symbols_args,
        ]
        code, out = _run_command(backtest_cmd)
        attempt["stages"]["backtest_oos"] = {"returncode": code, "tail": out[-4000:]}

        record = get_model_by_id(model_id)
        attempt["final_acceptance_status"] = record.get("acceptance_status") if record else None
        attempt["final_status"] = record.get("status") if record else None
        attempt["rejection_reasons"] = record.get("rejection_reasons_json") if record else None

        if record and record.get("acceptance_status") == "accepted":
            set_active_model(model_id, timeframe=timeframe)
            attempt["set_active"] = True

        summary["attempts"].append(attempt)

    accepted_after = _eligible_existing_accepted(timeframe, training_scope=training_scope, symbols=symbols)
    summary["accepted_after"] = [m["model_id"] for m in accepted_after]
    summary["accepted_count_after"] = len(accepted_after)
    summary["target_met"] = len(accepted_after) >= int(target_accepted_models)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train/validate/backtest models until accepted-model pool target is met.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--timeframes", nargs="*", default=None, help="Multiple timeframes to maintain. Overrides --timeframe when provided.")
    parser.add_argument("--target-accepted-models", type=int, default=TARGET_ACCEPTED_MODELS)
    parser.add_argument("--max-attempts", type=int, default=MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE)
    parser.add_argument("--validation-max-folds", type=int, default=MODEL_POOL_VALIDATION_MAX_FOLDS)
    parser.add_argument("--training-scope", choices=["multi-symbol", "multi_symbol", "per-symbol", "per_symbol", "both"], default=TRAINING_SCOPE)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=MODEL_MAINTENANCE_INTERVAL_SECONDS)
    return parser.parse_args()


def main():
    args = parse_args()
    symbols = [s.upper().strip() for s in args.symbols] if args.symbols else [s.upper() for s in SYMBOLS]
    timeframes = [tf.strip() for tf in (args.timeframes or []) if str(tf).strip()] or [args.timeframe] or TIMEFRAMES

    training_scope = _normalize_training_scope(args.training_scope)

    def _cycle_for_timeframe(timeframe: str):
        if training_scope == "per_symbol" and len(symbols) > 1:
            per_symbol = {}
            for symbol in symbols:
                per_symbol[symbol] = maintain_model_pool(
                    symbols=[symbol],
                    timeframe=timeframe,
                    target_accepted_models=args.target_accepted_models,
                    max_attempts=args.max_attempts,
                    validation_max_folds=args.validation_max_folds,
                    training_scope=training_scope,
                )
            return {
                "training_scope": training_scope,
                "per_symbol": per_symbol,
                "target_met": all(item.get("target_met") for item in per_symbol.values()),
            }
        return maintain_model_pool(
            symbols=symbols,
            timeframe=timeframe,
            target_accepted_models=args.target_accepted_models,
            max_attempts=args.max_attempts,
            validation_max_folds=args.validation_max_folds,
            training_scope=training_scope,
        )

    def _cycle():
        results = {timeframe: _cycle_for_timeframe(timeframe) for timeframe in timeframes}
        if len(results) == 1:
            return next(iter(results.values()))
        return {
            "timeframes": results,
            "target_met": all(bool(item.get("target_met")) for item in results.values()),
        }

    if args.loop:
        update_status("model_maintenance", "running", pid=os.getpid(), message="loop starting")
        while True:
            try:
                summary = _cycle()
                update_status(
                    "model_maintenance",
                    "running",
                    pid=os.getpid(),
                    message="maintenance cycle completed",
                    metadata={"target_met": summary.get("target_met"), "accepted_count_after": summary.get("accepted_count_after")},
                )
                print(json.dumps(summary, ensure_ascii=True), flush=True)
            except KeyboardInterrupt:
                update_status("model_maintenance", "stopped", pid=os.getpid(), message="keyboard interrupt")
                raise
            except Exception as exc:
                update_status("model_maintenance", "error", pid=os.getpid(), message=str(exc))
                record_event("model_maintenance", "error", str(exc))
                print(json.dumps({"component": "model_maintenance", "error": str(exc)}, ensure_ascii=True), flush=True)
            time.sleep(max(60, int(args.interval_seconds)))

    summary = _cycle()
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
