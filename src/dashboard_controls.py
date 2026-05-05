"""Audited control/config write layer for the Streamlit dashboard.

The dashboard never sends exchange orders and never starts heavy jobs inline.
Operator intent is persisted as pending rows in ``bot_control_actions`` so an
external orchestrator/worker can process it safely.
"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from config import BASE_DIR, DB_FILE, DRY_RUN, ENABLE_LIVE_TRADING, ENABLE_REAL_BINANCE_ACCOUNT, ENABLE_REAL_ORDER_EXECUTION
except Exception:  # pragma: no cover
    BASE_DIR = Path(__file__).resolve().parent.parent
    DB_FILE = BASE_DIR / "data" / "db" / "market_data.sqlite"
    DRY_RUN = True
    ENABLE_LIVE_TRADING = False
    ENABLE_REAL_BINANCE_ACCOUNT = False
    ENABLE_REAL_ORDER_EXECUTION = False

try:
    from runtime_status import record_event, update_status
except Exception:  # pragma: no cover
    def record_event(component: str, severity: str, message: str, metadata: dict | None = None) -> None:
        return None

    def update_status(*args, **kwargs) -> None:
        return None


DB_PATH = Path(DB_FILE).expanduser()
if not DB_PATH.is_absolute():
    DB_PATH = (Path(BASE_DIR) / DB_PATH).resolve()

BOT_ACTIONS_TABLE = "bot_control_actions"
MODEL_CONTROL_TABLE = "model_control"
RUNTIME_CONFIG_TABLE = "runtime_config"
RUNTIME_CONFIG_AUDIT_TABLE = "runtime_config_audit"

ACTION_TYPES = {
    "START_BOT",
    "STOP_BOT",
    "PAUSE_SIGNALS",
    "RESUME_SIGNALS",
    "ENABLE_PAPER",
    "DISABLE_PAPER",
    "REQUEST_RETRAIN",
    "REFRESH_DATA",
    "RUN_DATA_CHECK",
    "ACTIVATE_MODEL",
    "DEACTIVATE_MODEL",
    "ENABLE_MODEL_PAPER",
    "DISABLE_MODEL_PAPER",
    "MARK_MODEL_CANDIDATE",
    "MARK_MODEL_REJECTED",
    "KILL_SWITCH",
    "ENABLE_LIVE_RUNTIME",
    "DISABLE_LIVE_RUNTIME",
}

RUNTIME_CONFIG_SCHEMA: dict[str, dict[str, Any]] = {
    "bot.signal_generation_enabled": {"type": "bool", "default": "true", "description": "Allow signal generation workers to act."},
    "market.symbols": {"type": "list[str]", "default": os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT"), "description": "Comma-separated trading universe."},
    "market.timeframe": {"type": "str", "default": os.getenv("TIMEFRAME", "1h"), "description": "Primary candle timeframe."},
    "market.quote_asset": {"type": "str", "default": "USDT", "description": "Quote asset for portfolio reporting."},
    "model.active_model_selection_policy": {"type": "str", "default": "risk_adjusted", "description": "Policy consumed by an orchestrator."},
    "model.min_confidence": {"type": "float", "default": "0.50", "min": 0.0, "max": 1.0, "description": "Minimum signal confidence threshold."},
    "model.retrain_enabled": {"type": "bool", "default": "true", "description": "Whether scheduled retraining may run."},
    "model.retrain_interval_hours": {"type": "int", "default": "24", "min": 1, "max": 720, "description": "Requested retrain cadence."},
    "risk.max_position_size_usdt": {"type": "float", "default": os.getenv("MAX_ORDER_NOTIONAL_USDT", "50"), "min": 0.0, "description": "Max notional for one new order."},
    "risk.max_total_exposure_pct": {"type": "float", "default": "0.50", "min": 0.0, "max": 1.0, "description": "Max total portfolio exposure fraction."},
    "risk.max_symbol_exposure_pct": {"type": "float", "default": os.getenv("MAX_POSITION_PCT_PER_SYMBOL", "0.20"), "min": 0.0, "max": 1.0, "description": "Max exposure fraction per symbol."},
    "risk.daily_loss_limit_pct": {"type": "float", "default": "0.05", "min": 0.0, "max": 1.0, "description": "Daily loss guardrail fraction."},
    "risk.max_trades_per_day": {"type": "int", "default": os.getenv("MAX_TRADES_PER_DAY_PER_MODEL", "10"), "min": 0, "max": 1000, "description": "Per-model max trades per day."},
    "risk.allow_short": {"type": "bool", "default": "false", "description": "Spot trading should normally keep this false."},
    "execution.dry_run": {"type": "bool", "default": str(DRY_RUN).lower(), "description": "Dry-run execution flag mirrored from environment."},
    "execution.paper_trading_enabled": {"type": "bool", "default": "true", "description": "Allow paper/testnet workers if configured."},
    "execution.live_trading_enabled": {"type": "bool", "default": "false", "dangerous": True, "description": "Read-only/blocked unless server env gates permit real trading."},
    "execution.order_type": {"type": "str", "default": "MARKET", "description": "Default order type for workers."},
    "execution.slippage_assumption": {"type": "float", "default": "0.0002", "min": 0.0, "max": 0.05, "description": "Backtest/paper slippage assumption."},
    "execution.fee_assumption": {"type": "float", "default": "0.0005", "min": 0.0, "max": 0.05, "description": "Fee assumption."},
    "data.ingestion_enabled": {"type": "bool", "default": "true", "description": "Whether ingestion workers may run."},
    "data.data_refresh_interval": {"type": "int", "default": "60", "min": 5, "max": 86400, "description": "Requested data refresh interval in seconds."},
    "data.gap_check_enabled": {"type": "bool", "default": "true", "description": "Enable data gap checks."},
}


def utc_now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or DB_PATH)
    if not path.exists():
        raise FileNotFoundError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _json_dumps(payload: dict[str, Any] | None) -> str:
    return json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)


def ensure_dashboard_tables(db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {BOT_ACTIONS_TABLE} (
                action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                model_id TEXT,
                requested_at_utc TEXT NOT NULL,
                requested_by TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                params_json TEXT NOT NULL DEFAULT '{{}}',
                result_message TEXT,
                processed_at_utc TEXT
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MODEL_CONTROL_TABLE} (
                model_id TEXT PRIMARY KEY,
                signal_enabled INTEGER NOT NULL DEFAULT 0,
                paper_enabled INTEGER NOT NULL DEFAULT 0,
                live_enabled INTEGER NOT NULL DEFAULT 0,
                updated_at_utc TEXT NOT NULL,
                updated_by TEXT NOT NULL,
                notes TEXT
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RUNTIME_CONFIG_TABLE} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                value_type TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                updated_by TEXT NOT NULL,
                description TEXT
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RUNTIME_CONFIG_AUDIT_TABLE} (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT NOT NULL,
                value_type TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                updated_by TEXT NOT NULL,
                reason TEXT
            )
            """
        )
        now = utc_now_iso()
        for key, spec in RUNTIME_CONFIG_SCHEMA.items():
            conn.execute(
                f"""
                INSERT INTO {RUNTIME_CONFIG_TABLE} (key, value, value_type, updated_at_utc, updated_by, description)
                VALUES (?, ?, ?, ?, 'system_default', ?)
                ON CONFLICT(key) DO NOTHING
                """,
                (key, str(spec["default"]), str(spec["type"]), now, str(spec.get("description", ""))),
            )
        conn.commit()


def validate_runtime_value(key: str, value: Any) -> tuple[str, str]:
    if key not in RUNTIME_CONFIG_SCHEMA:
        raise ValueError(f"Unsupported runtime config key: {key}")
    spec = RUNTIME_CONFIG_SCHEMA[key]
    typ = str(spec["type"])
    if key == "execution.live_trading_enabled" and str(value).strip().lower() in {"true", "1", "yes", "on"}:
        allowed = (
            os.getenv("LIVE_TRADING_ALLOWED", "false").lower() == "true"
            and ENABLE_LIVE_TRADING
            and ENABLE_REAL_ORDER_EXECUTION
            and ENABLE_REAL_BINANCE_ACCOUNT
            and not DRY_RUN
        )
        if not allowed:
            raise ValueError("Live trading cannot be enabled from the dashboard with current environment safety gates.")
    if typ == "bool":
        if isinstance(value, bool):
            parsed = value
        else:
            raw = str(value).strip().lower()
            if raw not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
                raise ValueError(f"{key} must be boolean")
            parsed = raw in {"true", "1", "yes", "on"}
        return ("true" if parsed else "false", typ)
    if typ == "int":
        parsed_i = int(value)
        if "min" in spec and parsed_i < int(spec["min"]):
            raise ValueError(f"{key} must be >= {spec['min']}")
        if "max" in spec and parsed_i > int(spec["max"]):
            raise ValueError(f"{key} must be <= {spec['max']}")
        return (str(parsed_i), typ)
    if typ == "float":
        parsed_f = float(value)
        if "min" in spec and parsed_f < float(spec["min"]):
            raise ValueError(f"{key} must be >= {spec['min']}")
        if "max" in spec and parsed_f > float(spec["max"]):
            raise ValueError(f"{key} must be <= {spec['max']}")
        return (str(parsed_f), typ)
    if typ == "list[str]":
        parts = [p.strip().upper() for p in str(value).replace(";", ",").split(",") if p.strip()]
        if not parts:
            raise ValueError(f"{key} must contain at least one item")
        return (",".join(parts), typ)
    text = str(value).strip()
    if not text:
        raise ValueError(f"{key} must not be empty")
    return (text, typ)


def request_bot_action(
    action_type: str,
    params: dict[str, Any] | None = None,
    requested_by: str = "unknown",
    model_id: str | None = None,
    db_path: Path | None = None,
) -> int:
    action_type = action_type.upper()
    if action_type not in ACTION_TYPES:
        raise ValueError(f"Unsupported action type: {action_type}")
    ensure_dashboard_tables(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"""
            INSERT INTO {BOT_ACTIONS_TABLE}
                (action_type, model_id, requested_at_utc, requested_by, status, params_json)
            VALUES (?, ?, ?, ?, 'pending', ?)
            """,
            (action_type, model_id, utc_now_iso(), requested_by, _json_dumps(params)),
        )
        conn.commit()
        return int(cur.lastrowid)


def _set_action_result(action_id: int, status: str, message: str) -> None:
    try:
        with _connect() as conn:
            conn.execute(
                f"""
                UPDATE {BOT_ACTIONS_TABLE}
                SET status = ?, result_message = ?, processed_at_utc = ?
                WHERE action_id = ?
                """,
                (status, message[:4000], utc_now_iso(), int(action_id)),
            )
            conn.commit()
    except Exception:
        return None


def server_actions_enabled() -> bool:
    return os.getenv("DASHBOARD_ALLOW_SERVER_ACTIONS", "true").strip().lower() in {"1", "true", "yes", "y", "on"}


def _python_cmd(script: str, args: list[str]) -> list[str]:
    return [sys.executable, str(Path("src") / script), *args]


def _log_handles(name: str):
    log_dir = Path(BASE_DIR) / "logs" / "dashboard" / "actions"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    stdout = open(log_dir / f"{stamp}_{name}.out.log", "a", encoding="utf-8")
    stderr = open(log_dir / f"{stamp}_{name}.err.log", "a", encoding="utf-8")
    return stdout, stderr


def _run_background(name: str, cmd: list[str]) -> int:
    env = os.environ.copy()
    src = str(Path(BASE_DIR) / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else f"{src}{os.pathsep}{env['PYTHONPATH']}"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    stdout, stderr = _log_handles(name)
    try:
        proc = subprocess.Popen(cmd, cwd=BASE_DIR, env=env, stdout=stdout, stderr=stderr, text=True)
        return int(proc.pid)
    finally:
        stdout.close()
        stderr.close()


def _read_bot_status_pids() -> list[tuple[str, int]]:
    if not DB_PATH.exists():
        return []
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT component, pid FROM bot_status WHERE pid IS NOT NULL AND status IN ('running','stopping','unknown')"
            ).fetchall()
        return [(str(component), int(pid)) for component, pid in rows if pid]
    except Exception:
        return []


def _terminate_pid(pid: int) -> str:
    if pid <= 0 or pid == os.getpid():
        return "skipped"
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True, text=True)
        else:
            os.kill(pid, signal.SIGTERM)
        return "terminated"
    except ProcessLookupError:
        return "not_running"
    except Exception as exc:
        return f"failed: {exc}"


def _runtime_set_if_exists(key: str, value: Any, requested_by: str) -> None:
    try:
        update_runtime_config(key, value, requested_by, reason="dashboard_direct_action")
    except Exception:
        pass


def execute_bot_action(
    action_type: str,
    params: dict[str, Any] | None = None,
    requested_by: str = "unknown",
    model_id: str | None = None,
) -> int:
    """Queue and immediately execute a safe authenticated dashboard action.

    This is intended for VPS operation from the authenticated panel. It still
    never sends market orders and never enables live trading.
    """
    action_id = request_bot_action(action_type, params, requested_by, model_id)
    action_type = action_type.upper()
    params = params or {}
    if not server_actions_enabled():
        _set_action_result(action_id, "pending", "Server-side dashboard actions disabled; action queued only.")
        return action_id
    try:
        if action_type == "START_BOT":
            symbols_raw = str(params.get("symbols") or os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT"))
            symbols = [s.strip().upper() for s in symbols_raw.replace(";", ",").split(",") if s.strip()]
            timeframe = str(params.get("timeframe") or os.getenv("TIMEFRAME", "1h"))
            cmd = _python_cmd("autonomous_runner.py", ["--symbols", *symbols, "--timeframe", timeframe, "--no-dashboard"])
            pid = _run_background("start_bot", cmd)
            update_status("dashboard_control", "running", pid=os.getpid(), message="started bot from dashboard", metadata={"child_pid": pid, "cmd": cmd})
            record_event("dashboard", "info", "START_BOT executed from dashboard", {"requested_by": requested_by, "pid": pid})
            _set_action_result(action_id, "completed", f"Started autonomous_runner --no-dashboard pid={pid}")
        elif action_type == "STOP_BOT":
            results = []
            for component, pid in _read_bot_status_pids():
                if component == "dashboard":
                    continue
                result = _terminate_pid(pid)
                results.append({"component": component, "pid": pid, "result": result})
                try:
                    update_status(component, "stopped", pid=pid, message="stopped from dashboard")
                except Exception:
                    pass
            record_event("dashboard", "warning", "STOP_BOT executed from dashboard", {"requested_by": requested_by, "results": results})
            _set_action_result(action_id, "completed", json.dumps(results, ensure_ascii=False))
        elif action_type == "PAUSE_SIGNALS":
            update_runtime_config("bot.signal_generation_enabled", "false", requested_by, reason="dashboard_pause_signals")
            _set_action_result(action_id, "completed", "Signal generation disabled in runtime_config.")
        elif action_type == "RESUME_SIGNALS":
            update_runtime_config("bot.signal_generation_enabled", "true", requested_by, reason="dashboard_resume_signals")
            _set_action_result(action_id, "completed", "Signal generation enabled in runtime_config.")
        elif action_type == "ENABLE_PAPER":
            update_runtime_config("execution.paper_trading_enabled", "true", requested_by, reason="dashboard_enable_paper")
            _set_action_result(action_id, "completed", "Paper trading enabled in runtime_config.")
        elif action_type == "DISABLE_PAPER":
            update_runtime_config("execution.paper_trading_enabled", "false", requested_by, reason="dashboard_disable_paper")
            _set_action_result(action_id, "completed", "Paper trading disabled in runtime_config.")
        elif action_type == "KILL_SWITCH":
            update_runtime_config("bot.signal_generation_enabled", "false", requested_by, reason="dashboard_kill_switch")
            update_runtime_config("execution.paper_trading_enabled", "false", requested_by, reason="dashboard_kill_switch")
            update_runtime_config("execution.live_trading_enabled", "false", requested_by, reason="dashboard_kill_switch")
            results = []
            for component, pid in _read_bot_status_pids():
                if component == "dashboard":
                    continue
                result = _terminate_pid(pid)
                results.append({"component": component, "pid": pid, "result": result})
                try:
                    update_status(component, "stopped", pid=pid, message="dashboard kill switch")
                except Exception:
                    pass
            record_event("dashboard", "critical", "KILL_SWITCH executed from dashboard", {"requested_by": requested_by, "results": results})
            _set_action_result(action_id, "completed", "Kill switch executed: signals/paper/live runtime disabled and bot processes stopped.")
        elif action_type == "ENABLE_LIVE_RUNTIME":
            # Guarded: this only flips runtime_config and only when deployment env flags already permit real trading.
            update_runtime_config("execution.live_trading_enabled", "true", requested_by, reason="dashboard_reauthenticated_live_enable")
            record_event("dashboard", "critical", "Live runtime enabled from dashboard after re-auth", {"requested_by": requested_by})
            _set_action_result(action_id, "completed", "Live runtime enabled. Real order path still depends on env flags, RiskManager and KillSwitch.")
        elif action_type == "DISABLE_LIVE_RUNTIME":
            update_runtime_config("execution.live_trading_enabled", "false", requested_by, reason="dashboard_live_disable")
            record_event("dashboard", "warning", "Live runtime disabled from dashboard", {"requested_by": requested_by})
            _set_action_result(action_id, "completed", "Live runtime disabled.")
        elif action_type == "REQUEST_RETRAIN":
            symbols_raw = str(params.get("symbols") or os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT"))
            symbols = [s.strip().upper() for s in symbols_raw.replace(";", ",").split(",") if s.strip()]
            timeframe = str(params.get("timeframe") or os.getenv("TIMEFRAME", "1h"))
            scope = str(params.get("training_scope") or os.getenv("TRAINING_SCOPE", "both"))
            max_attempts = str(int(params.get("max_attempts") or os.getenv("MAX_TRAINING_ATTEMPTS_PER_CYCLE", "50")))
            target = str(int(params.get("target_accepted_models") or os.getenv("TARGET_ACCEPTED_MODELS", "5")))
            cmd = _python_cmd("model_maintenance.py", ["--symbols", *symbols, "--timeframe", timeframe, "--training-scope", scope, "--target-accepted-models", target, "--max-attempts", max_attempts])
            pid = _run_background("request_retrain", cmd)
            _set_action_result(action_id, "completed", f"Started model maintenance pid={pid}")
        elif action_type == "REFRESH_DATA":
            symbols_raw = str(params.get("symbols") or os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT"))
            symbols = [s.strip().upper() for s in symbols_raw.replace(";", ",").split(",") if s.strip()]
            timeframe = str(params.get("timeframe") or os.getenv("TIMEFRAME", "1h"))
            cmd = _python_cmd("realtime_ingestor.py", ["--symbols", *symbols, "--timeframe", timeframe])
            pid = _run_background("refresh_data", cmd)
            _set_action_result(action_id, "completed", f"Started one-shot data refresh pid={pid}")
        elif action_type == "RUN_DATA_CHECK":
            symbols_raw = str(params.get("symbols") or os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT"))
            symbols = [s.strip().upper() for s in symbols_raw.replace(";", ",").split(",") if s.strip()]
            timeframe = str(params.get("timeframe") or os.getenv("TIMEFRAME", "1h"))
            cmd = _python_cmd("data_quality_service.py", ["--symbols", *symbols, "--timeframe", timeframe])
            pid = _run_background("data_quality", cmd)
            _set_action_result(action_id, "completed", f"Started data quality check pid={pid}")
        else:
            _set_action_result(action_id, "pending", "Action recorded; no direct executor implemented for this action type.")
        return action_id
    except Exception as exc:
        _set_action_result(action_id, "failed", str(exc))
        raise


def _upsert_model_control(conn: sqlite3.Connection, model_id: str, requested_by: str, *, signal_enabled: int | None = None, paper_enabled: int | None = None) -> None:
    now = utc_now_iso()
    conn.execute(
        f"""
        INSERT INTO {MODEL_CONTROL_TABLE} (model_id, signal_enabled, paper_enabled, live_enabled, updated_at_utc, updated_by, notes)
        VALUES (?, 0, 0, 0, ?, ?, 'created by dashboard')
        ON CONFLICT(model_id) DO NOTHING
        """,
        (model_id, now, requested_by),
    )
    sets = ["updated_at_utc = ?", "updated_by = ?"]
    params: list[Any] = [now, requested_by]
    if signal_enabled is not None:
        sets.append("signal_enabled = ?")
        params.append(int(signal_enabled))
    if paper_enabled is not None:
        sets.append("paper_enabled = ?")
        params.append(int(paper_enabled))
    params.append(model_id)
    conn.execute(f"UPDATE {MODEL_CONTROL_TABLE} SET {', '.join(sets)} WHERE model_id = ?", params)


def activate_model(model_id: str, requested_by: str, db_path: Path | None = None) -> int:
    if not model_id:
        raise ValueError("model_id is required")
    ensure_dashboard_tables(db_path)
    with _connect(db_path) as conn:
        _upsert_model_control(conn, model_id, requested_by, signal_enabled=1)
        if _table_exists(conn, "model_registry") and "is_active" in _table_columns(conn, "model_registry"):
            conn.execute("UPDATE model_registry SET is_active = 1, updated_at_utc = ? WHERE model_id = ?", (utc_now_iso(), model_id))
        conn.commit()
    return request_bot_action("ACTIVATE_MODEL", {"signal_enabled": True}, requested_by, model_id, db_path)


def deactivate_model(model_id: str, requested_by: str, db_path: Path | None = None) -> int:
    if not model_id:
        raise ValueError("model_id is required")
    ensure_dashboard_tables(db_path)
    with _connect(db_path) as conn:
        _upsert_model_control(conn, model_id, requested_by, signal_enabled=0)
        if _table_exists(conn, "model_registry") and "is_active" in _table_columns(conn, "model_registry"):
            conn.execute("UPDATE model_registry SET is_active = 0, updated_at_utc = ? WHERE model_id = ?", (utc_now_iso(), model_id))
        conn.commit()
    return request_bot_action("DEACTIVATE_MODEL", {"signal_enabled": False}, requested_by, model_id, db_path)


def set_model_paper(model_id: str, enabled: bool, requested_by: str, db_path: Path | None = None) -> int:
    if not model_id:
        raise ValueError("model_id is required")
    ensure_dashboard_tables(db_path)
    with _connect(db_path) as conn:
        _upsert_model_control(conn, model_id, requested_by, paper_enabled=1 if enabled else 0)
        conn.commit()
    return request_bot_action("ENABLE_MODEL_PAPER" if enabled else "DISABLE_MODEL_PAPER", {"paper_enabled": enabled}, requested_by, model_id, db_path)


def request_model_training(params: dict[str, Any] | None, requested_by: str, db_path: Path | None = None) -> int:
    return request_bot_action("REQUEST_RETRAIN", params or {}, requested_by, db_path=db_path)


def update_runtime_config(key: str, value: Any, requested_by: str, reason: str = "dashboard_update", db_path: Path | None = None) -> None:
    normalized, value_type = validate_runtime_value(key, value)
    ensure_dashboard_tables(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(f"SELECT value FROM {RUNTIME_CONFIG_TABLE} WHERE key = ?", (key,)).fetchone()
        old_value = row[0] if row else None
        now = utc_now_iso()
        spec = RUNTIME_CONFIG_SCHEMA[key]
        conn.execute(
            f"""
            INSERT INTO {RUNTIME_CONFIG_TABLE} (key, value, value_type, updated_at_utc, updated_by, description)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                value_type=excluded.value_type,
                updated_at_utc=excluded.updated_at_utc,
                updated_by=excluded.updated_by,
                description=excluded.description
            """,
            (key, normalized, value_type, now, requested_by, str(spec.get("description", ""))),
        )
        conn.execute(
            f"""
            INSERT INTO {RUNTIME_CONFIG_AUDIT_TABLE}
                (key, old_value, new_value, value_type, updated_at_utc, updated_by, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (key, old_value, normalized, value_type, now, requested_by, reason),
        )
        conn.commit()


def load_control_actions(limit: int = 200, db_path: Path | None = None) -> pd.DataFrame:
    try:
        ensure_dashboard_tables(db_path)
        with _connect(db_path) as conn:
            return pd.read_sql_query(
                f"SELECT * FROM {BOT_ACTIONS_TABLE} ORDER BY requested_at_utc DESC LIMIT ?",
                conn,
                params=(int(limit),),
            )
    except Exception as exc:
        df = pd.DataFrame()
        df.attrs["message"] = str(exc)
        return df


def load_model_control(db_path: Path | None = None) -> pd.DataFrame:
    try:
        ensure_dashboard_tables(db_path)
        with _connect(db_path) as conn:
            return pd.read_sql_query(f"SELECT * FROM {MODEL_CONTROL_TABLE} ORDER BY updated_at_utc DESC", conn)
    except Exception as exc:
        df = pd.DataFrame()
        df.attrs["message"] = str(exc)
        return df


def load_runtime_config(db_path: Path | None = None) -> pd.DataFrame:
    try:
        ensure_dashboard_tables(db_path)
        with _connect(db_path) as conn:
            return pd.read_sql_query(f"SELECT * FROM {RUNTIME_CONFIG_TABLE} ORDER BY key", conn)
    except Exception as exc:
        df = pd.DataFrame()
        df.attrs["message"] = str(exc)
        return df


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()
    return bool(row)


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()}


def live_trading_locked_reason() -> str:
    if os.getenv("LIVE_TRADING_ALLOWED", "false").lower() != "true":
        return "LIVE_TRADING_ALLOWED is not true."
    if DRY_RUN:
        return "DRY_RUN is true."
    missing = [
        name
        for name, ok in {
            "ENABLE_LIVE_TRADING": ENABLE_LIVE_TRADING,
            "ENABLE_REAL_ORDER_EXECUTION": ENABLE_REAL_ORDER_EXECUTION,
            "ENABLE_REAL_BINANCE_ACCOUNT": ENABLE_REAL_BINANCE_ACCOUNT,
        }.items()
        if not ok
    ]
    if missing:
        return "Missing real-trading env gates: " + ", ".join(missing)
    return "Unlocked by environment, but dashboard still does not send real orders."
