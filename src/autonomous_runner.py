import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from config import (
    BASE_DIR,
    BOT_POLL_SECONDS,
    DASHBOARD_REFRESH_SECONDS,
    DB_FILE,
    FEATURES_TABLE,
    FEATURE_VERSION,
    LOGS_DIR,
    MIN_TRAIN_ROWS,
    MODEL_EVALUATION_INTERVAL_SECONDS,
    MODEL_MAINTENANCE_INTERVAL_SECONDS,
    PRICES_TABLE,
    SYMBOLS,
    TARGET_ACCEPTED_MODELS,
    TIMEFRAME,
    TIMEFRAMES,
    TRAINING_SCOPE,
)
from db_utils import ensure_project_directories, init_research_tables
from runtime_status import load_status, record_event, update_status


@dataclass
class ServiceSpec:
    name: str
    cmd: list[str]
    log_name: str
    restart: bool = True


class AutonomousRunner:
    def __init__(self, symbols: list[str], timeframe: str, timeframes: list[str] | None = None, no_dashboard: bool = False, no_maintenance: bool = False, bootstrap: bool = True):
        self.symbols = symbols
        self.timeframe = timeframe
        self.timeframes = [tf for tf in (timeframes or [timeframe]) if tf]
        self.no_dashboard = no_dashboard
        self.no_maintenance = no_maintenance
        self.bootstrap = bootstrap
        self.procs: dict[str, subprocess.Popen] = {}
        self.started_at: dict[str, str] = {}
        self.logs_dir = LOGS_DIR / "services"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        init_research_tables()

    def _env(self) -> dict:
        env = os.environ.copy()
        src = str(BASE_DIR / "src")
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = src if not current else f"{src}{os.pathsep}{current}"
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("LOKY_MAX_CPU_COUNT", "4")
        return env

    def specs(self) -> list[ServiceSpec]:
        sym_args = self.symbols
        specs = [
            ServiceSpec(
                "realtime_ingestor",
                [sys.executable, "src/realtime_ingestor.py", "--symbols", *sym_args, "--timeframes", *self.timeframes, "--loop", "--poll-seconds", str(BOT_POLL_SECONDS)],
                "realtime_ingestor.log",
            ),
            ServiceSpec(
                "trading_bot",
                [
                    sys.executable,
                    "src/trading_bot.py",
                    "--mode",
                    "paper",
                    "--paper-mode",
                    "per-model",
                    "--loop",
                    "--symbols",
                    *sym_args,
                    "--timeframes",
                    *self.timeframes,
                    "--sync-latest-from-binance",
                    "--refresh-features",
                    "--skip-model-maintenance",
                    "--target-accepted-models",
                    str(TARGET_ACCEPTED_MODELS),
                    "--training-scope",
                    TRAINING_SCOPE,
                    "--log-level",
                    "INFO",
                ],
                "trading_bot.log",
            ),
            ServiceSpec(
                "paper_model_evaluator",
                [sys.executable, "src/paper_model_evaluator.py", "--evaluate-active", "--loop", "--timeframe", self.timeframe, "--interval-seconds", str(MODEL_EVALUATION_INTERVAL_SECONDS)],
                "paper_model_evaluator.log",
            ),
        ]
        if not self.no_maintenance:
            specs.append(
                ServiceSpec(
                    "model_maintenance",
                    [sys.executable, "src/model_maintenance.py", "--symbols", *sym_args, "--timeframes", *self.timeframes, "--target-accepted-models", str(TARGET_ACCEPTED_MODELS), "--loop", "--interval-seconds", str(MODEL_MAINTENANCE_INTERVAL_SECONDS)],
                    "model_maintenance.log",
                )
            )
        if not self.no_dashboard:
            specs.append(
                ServiceSpec(
                    "dashboard",
                    [sys.executable, "-m", "streamlit", "run", "src/dashboard.py", "--server.headless", "true"],
                    "dashboard.log",
                    restart=True,
                )
            )
        return specs

    def start_service(self, spec: ServiceSpec) -> None:
        stdout_path = self.logs_dir / spec.log_name
        stderr_path = self.logs_dir / spec.log_name.replace(".log", ".err.log")
        stdout = open(stdout_path, "a", encoding="utf-8")
        stderr = open(stderr_path, "a", encoding="utf-8")
        proc = subprocess.Popen(spec.cmd, cwd=BASE_DIR, env=self._env(), stdout=stdout, stderr=stderr, text=True)
        self.procs[spec.name] = proc
        ts = __import__("pandas").Timestamp.now(tz="UTC").isoformat()
        self.started_at[spec.name] = ts
        update_status(spec.name, "running", pid=proc.pid, started_at_utc=ts, message="started by autonomous_runner", metadata={"cmd": spec.cmd, "log": str(stdout_path)})
        record_event("autonomous_runner", "info", f"started {spec.name}", metadata={"pid": proc.pid, "cmd": spec.cmd})

    def stop_all(self) -> None:
        for name, proc in list(self.procs.items()):
            if proc.poll() is None:
                update_status(name, "stopping", pid=proc.pid, message="runner stopping")
                proc.terminate()
        time.sleep(3)
        for name, proc in list(self.procs.items()):
            if proc.poll() is None:
                proc.kill()
            update_status(name, "stopped", pid=proc.pid, message="runner stopped")

    def print_status(self) -> None:
        statuses = load_status(stale_after_seconds=max(180, BOT_POLL_SECONDS * 3))
        by_name = {s["component"]: s for s in statuses}
        print("\n=== AUTONOMOUS BOT STATUS ===", flush=True)
        print(f"Platform: {'RUNNING' if by_name.get('autonomous_runner', {}).get('effective_status') == 'running' else 'OFF/STALE'}", flush=True)
        for name, proc in self.procs.items():
            row = by_name.get(name, {})
            state = row.get("effective_status", "unknown")
            pid = proc.pid
            rc = proc.poll()
            print(f"- {name:22s} status={state:8s} pid={pid} returncode={rc}", flush=True)

    # ---- one-click bootstrap: get data + features ready before services start ----
    def _price_rows(self, symbol: str, timeframe: str) -> int:
        try:
            with sqlite3.connect(DB_FILE) as conn:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM {PRICES_TABLE} WHERE symbol=? AND timeframe=?",
                    (symbol, timeframe),
                ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.OperationalError:
            return 0

    def _features_current(self, symbol: str, timeframe: str) -> bool:
        try:
            with sqlite3.connect(DB_FILE) as conn:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM {FEATURES_TABLE} WHERE symbol=? AND timeframe=? AND feature_version=?",
                    (symbol, timeframe, FEATURE_VERSION),
                ).fetchone()
            return bool(row and int(row[0]) > 0)
        except sqlite3.OperationalError:
            return False

    def _run_bootstrap_cmd(self, name: str, cmd: list[str], log) -> None:
        log.write(f"\n[bootstrap] {name}: {' '.join(cmd)}\n")
        log.flush()
        try:
            proc = subprocess.run(cmd, cwd=BASE_DIR, env=self._env(), stdout=log, stderr=subprocess.STDOUT, text=True, timeout=60 * 60)
            log.write(f"[bootstrap] {name} rc={proc.returncode}\n")
        except Exception as exc:  # network/timeout must not abort the run — continue on existing data
            log.write(f"[bootstrap] {name} FAILED: {exc}\n")
            record_event("autonomous_runner", "warning", f"bootstrap {name} failed: {exc}")
        log.flush()

    def _bootstrap(self) -> None:
        """Idempotent: ensure history + feature store (current version) exist before trading.
        Skips heavy work when data is already sufficient and features are at the current version."""
        ensure_project_directories()
        init_research_tables()
        log_path = self.logs_dir / "bootstrap.log"
        update_status("autonomous_runner", "running", pid=os.getpid(), message="bootstrapping data + features")
        with open(log_path, "a", encoding="utf-8") as log:
            for tf in self.timeframes:
                need_full = any(self._price_rows(s, tf) < int(MIN_TRAIN_ROWS) for s in self.symbols)
                mode = "full" if need_full else "incremental"
                self._run_bootstrap_cmd(
                    f"download_{tf}_{mode}",
                    [sys.executable, "src/download_data.py", "--symbols", *self.symbols, "--timeframe", tf, "--mode", mode],
                    log,
                )
                self._run_bootstrap_cmd(
                    f"gapcheck_{tf}",
                    [sys.executable, "src/data_loader.py", "--gap-check", "--no-prompt"],
                    log,
                )
                need_rebuild = any(not self._features_current(s, tf) for s in self.symbols)
                fs_cmd = [sys.executable, "src/feature_store.py", "--symbols", *self.symbols, "--timeframes", tf]
                if need_rebuild:
                    fs_cmd.append("--full-rebuild")
                self._run_bootstrap_cmd(f"feature_store_{tf}{'_full' if need_rebuild else ''}", fs_cmd, log)
        record_event("autonomous_runner", "info", "bootstrap complete", metadata={"timeframes": self.timeframes})

    def run(self) -> int:
        if self.bootstrap:
            self._bootstrap()
        specs = self.specs()
        update_status("autonomous_runner", "running", pid=os.getpid(), message="runner starting", metadata={"symbols": self.symbols, "timeframes": self.timeframes})
        for spec in specs:
            self.start_service(spec)
        def _request_shutdown(signum, frame):
            raise KeyboardInterrupt

        try:
            signal.signal(signal.SIGTERM, _request_shutdown)
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, _request_shutdown)
        except Exception:
            pass
        try:
            while True:
                update_status("autonomous_runner", "running", pid=os.getpid(), message="supervising services", metadata={"services": list(self.procs.keys())})
                for spec in specs:
                    proc = self.procs.get(spec.name)
                    if proc is None:
                        self.start_service(spec)
                        continue
                    rc = proc.poll()
                    if rc is None:
                        update_status(spec.name, "running", pid=proc.pid, message="process alive")
                    else:
                        update_status(spec.name, "stopped", pid=proc.pid, message=f"process exited rc={rc}")
                        record_event("autonomous_runner", "warning", f"{spec.name} exited", metadata={"returncode": rc})
                        if spec.restart:
                            self.start_service(spec)
                self.print_status()
                time.sleep(15)
        except KeyboardInterrupt:
            record_event("autonomous_runner", "info", "keyboard interrupt; stopping all")
            self.stop_all()
            update_status("autonomous_runner", "stopped", pid=os.getpid(), message="keyboard interrupt")
            return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full autonomous local trading platform and dashboard.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--timeframes", nargs="*", default=None)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--no-maintenance", action="store_true")
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip the data/feature bootstrap phase (assume already prepared).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.upper().strip() for s in args.symbols] if args.symbols else [s.upper() for s in SYMBOLS]
    timeframes = [tf.strip() for tf in (args.timeframes or []) if str(tf).strip()] or [args.timeframe] or TIMEFRAMES
    runner = AutonomousRunner(
        symbols=symbols,
        timeframe=timeframes[0],
        timeframes=timeframes,
        no_dashboard=args.no_dashboard,
        no_maintenance=args.no_maintenance,
        bootstrap=not args.no_bootstrap,
    )
    raise SystemExit(runner.run())


if __name__ == "__main__":
    main()
