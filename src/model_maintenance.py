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
    MODEL_MAINTENANCE_INTERVAL_SECONDS,
)
from model_registry import count_accepted_models, get_model_by_id, list_accepted_models, set_active_model
from runtime_status import record_event, update_status


@dataclass
class TrainingTrial:
    short_threshold: float
    long_threshold: float
    params: dict


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


def build_trial_plan(max_attempts: int) -> list[TrainingTrial]:
    base = dict(MODEL_PARAMS)
    candidates = [
        (0.55, 0.55, {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31}),
        (0.45, 0.45, {"n_estimators": 400, "learning_rate": 0.025, "num_leaves": 31}),
        (0.40, 0.40, {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 63}),
        (0.36, 0.36, {"n_estimators": 350, "learning_rate": 0.03, "num_leaves": 15}),
        (0.34, 0.34, {"n_estimators": 600, "learning_rate": 0.015, "num_leaves": 31}),
        (0.32, 0.32, {"n_estimators": 450, "learning_rate": 0.02, "num_leaves": 63, "min_child_samples": 50}),
        (0.38, 0.38, {"n_estimators": 300, "learning_rate": 0.04, "num_leaves": 15, "min_child_samples": 100}),
        (0.42, 0.42, {"n_estimators": 700, "learning_rate": 0.015, "num_leaves": 31, "feature_fraction": 0.85}),
    ]
    trials: list[TrainingTrial] = []
    for short_thr, long_thr, overrides in candidates:
        params = dict(base)
        params.update(overrides)
        trials.append(TrainingTrial(short_threshold=short_thr, long_threshold=long_thr, params=params))
    return trials[: max(0, int(max_attempts))]


def _eligible_existing_accepted(timeframe: str) -> list[dict]:
    models = []
    for model in list_accepted_models(timeframe=timeframe):
        path = Path(str(model.get("model_path", "")))
        if path.exists():
            models.append(model)
    return models


def maintain_model_pool(
    symbols: list[str],
    timeframe: str,
    target_accepted_models: int = TARGET_ACCEPTED_MODELS,
    max_attempts: int = MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE,
    validation_max_folds: int = MODEL_POOL_VALIDATION_MAX_FOLDS,
) -> dict:
    accepted_before = _eligible_existing_accepted(timeframe)
    needed = max(0, int(target_accepted_models) - len(accepted_before))
    summary = {
        "target_accepted_models": int(target_accepted_models),
        "accepted_before": [m["model_id"] for m in accepted_before],
        "attempts": [],
        "accepted_after": [],
        "needed_before_attempts": needed,
    }
    if needed <= 0:
        summary["accepted_after"] = summary["accepted_before"]
        return summary

    for idx, trial in enumerate(build_trial_plan(max_attempts), start=1):
        if count_accepted_models(timeframe) >= int(target_accepted_models):
            break

        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
        model_id = f"auto_lgbm_{timeframe}_{stamp}_{idx:02d}"
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
            "--short-threshold",
            str(trial.short_threshold),
            "--long-threshold",
            str(trial.long_threshold),
            "--model-params-b64",
            params_b64,
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
        ]
        code, out = _run_command(backtest_cmd)
        attempt["stages"]["backtest_oos"] = {"returncode": code, "tail": out[-4000:]}

        record = get_model_by_id(model_id)
        attempt["final_acceptance_status"] = record.get("acceptance_status") if record else None
        attempt["final_status"] = record.get("status") if record else None
        attempt["rejection_reasons"] = record.get("rejection_reasons_json") if record else None

        if record and record.get("acceptance_status") == "accepted" and not _eligible_existing_accepted(timeframe):
            set_active_model(model_id, timeframe=timeframe)
            attempt["set_active"] = True

        summary["attempts"].append(attempt)

    accepted_after = _eligible_existing_accepted(timeframe)
    summary["accepted_after"] = [m["model_id"] for m in accepted_after]
    summary["accepted_count_after"] = len(accepted_after)
    summary["target_met"] = len(accepted_after) >= int(target_accepted_models)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train/validate/backtest models until accepted-model pool target is met.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--target-accepted-models", type=int, default=TARGET_ACCEPTED_MODELS)
    parser.add_argument("--max-attempts", type=int, default=MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE)
    parser.add_argument("--validation-max-folds", type=int, default=MODEL_POOL_VALIDATION_MAX_FOLDS)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=MODEL_MAINTENANCE_INTERVAL_SECONDS)
    return parser.parse_args()


def main():
    args = parse_args()
    symbols = [s.upper().strip() for s in args.symbols] if args.symbols else [s.upper() for s in SYMBOLS]

    def _cycle():
        return maintain_model_pool(
            symbols=symbols,
            timeframe=args.timeframe,
            target_accepted_models=args.target_accepted_models,
            max_attempts=args.max_attempts,
            validation_max_folds=args.validation_max_folds,
        )

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
