import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from config import (
    BASE_DIR,
    BOT_POLL_SECONDS,
    DASHBOARD_REFRESH_SECONDS,
    LOGS_DIR,
    MODEL_EVALUATION_INTERVAL_SECONDS,
    MODEL_MAINTENANCE_INTERVAL_SECONDS,
    SYMBOLS,
    TARGET_ACCEPTED_MODELS,
    TIMEFRAME,
    TRAINING_SCOPE,
)
from db_utils import init_research_tables
from runtime_status import load_status, record_event, update_status


@dataclass
class ServiceSpec:
    name: str
    cmd: list[str]
    log_name: str
    restart: bool = True


class AutonomousRunner:
    def __init__(self, symbols: list[str], timeframe: str, no_dashboard: bool = False, no_maintenance: bool = False):
        self.symbols = symbols
        self.timeframe = timeframe
        self.no_dashboard = no_dashboard
        self.no_maintenance = no_maintenance
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
                [sys.executable, "src/realtime_ingestor.py", "--symbols", *sym_args, "--timeframe", self.timeframe, "--loop", "--poll-seconds", str(BOT_POLL_SECONDS)],
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
                    "--timeframe",
                    self.timeframe,
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
                    [sys.executable, "src/model_maintenance.py", "--symbols", *sym_args, "--timeframe", self.timeframe, "--target-accepted-models", str(TARGET_ACCEPTED_MODELS), "--loop", "--interval-seconds", str(MODEL_MAINTENANCE_INTERVAL_SECONDS)],
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

    def run(self) -> int:
        specs = self.specs()
        update_status("autonomous_runner", "running", pid=os.getpid(), message="runner starting", metadata={"symbols": self.symbols, "timeframe": self.timeframe})
        for spec in specs:
            self.start_service(spec)
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
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--no-maintenance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.upper().strip() for s in args.symbols] if args.symbols else [s.upper() for s in SYMBOLS]
    runner = AutonomousRunner(symbols=symbols, timeframe=args.timeframe, no_dashboard=args.no_dashboard, no_maintenance=args.no_maintenance)
    raise SystemExit(runner.run())


if __name__ == "__main__":
    main()
