import argparse
import json
import os
import sys
from pathlib import Path

from config import BASE_DIR, DATA_DIR, DB_FILE, LOGS_DIR, MODELS_DIR, RAW_DIR, REPORTS_DIR
from db_utils import assert_required_schema, ensure_project_directories, init_research_tables
from runtime_status import record_event, update_status


def ensure_env_file() -> dict:
    env_path = BASE_DIR / ".env"
    example = BASE_DIR / ".env.example"
    if env_path.exists():
        return {"created": False, "path": str(env_path)}
    if example.exists():
        env_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        return {"created": True, "path": str(env_path), "source": str(example)}
    env_path.write_text("DRY_RUN=true\nENABLE_LIVE_TRADING=false\n", encoding="utf-8")
    return {"created": True, "path": str(env_path), "source": "minimal_safe_defaults"}


def ensure_runtime_dirs() -> list[str]:
    ensure_project_directories()
    extra = [LOGS_DIR / "services", LOGS_DIR / "dashboard", BASE_DIR / ".tools"]
    for path in [DATA_DIR, RAW_DIR, DB_FILE.parent, MODELS_DIR, REPORTS_DIR, LOGS_DIR, *extra]:
        path.mkdir(parents=True, exist_ok=True)
    return [str(p) for p in [DATA_DIR, RAW_DIR, DB_FILE.parent, MODELS_DIR, REPORTS_DIR, LOGS_DIR, *extra]]


def run_install() -> dict:
    dirs = ensure_runtime_dirs()
    env = ensure_env_file()
    init_research_tables()
    schema = assert_required_schema()
    update_status("installer", "completed", pid=os.getpid(), message="install completed", metadata={"schema_ok": schema["ok"]})
    record_event("installer", "info", "install completed", metadata={"schema": schema})
    return {
        "ok": schema["ok"],
        "python": sys.executable,
        "db_file": str(DB_FILE),
        "directories": dirs,
        "env": env,
        "schema": schema,
        "next": ".tools\\run.cmd or powershell -ExecutionPolicy Bypass -File .tools\\run.ps1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Install/bootstrap local autonomous trading platform state.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_install()
    print(json.dumps(result, ensure_ascii=True, indent=2) if args.json else result)


if __name__ == "__main__":
    main()
