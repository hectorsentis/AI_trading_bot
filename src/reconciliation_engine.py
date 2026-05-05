from __future__ import annotations

import argparse
import json
import sqlite3
import uuid

import pandas as pd

from broker_client import BinanceCredentialsError, BinanceSpotClient
from config import (
    ACCOUNT_MODE_LOCAL_PAPER,
    ACCOUNT_MODE_REAL,
    ACCOUNT_MODE_TESTNET_PAPER,
    ACCOUNT_SNAPSHOTS_TABLE,
    BALANCE_SNAPSHOTS_TABLE,
    DB_FILE,
    ENABLE_REAL_BINANCE_ACCOUNT,
    RECONCILIATION_EVENTS_TABLE,
    SYSTEM_STATUS_TABLE,
)
from db_utils import init_research_tables


def _client_for_mode(mode: str):
    if mode == ACCOUNT_MODE_TESTNET_PAPER:
        return BinanceSpotClient.testnet_account_client()
    if mode == ACCOUNT_MODE_REAL and ENABLE_REAL_BINANCE_ACCOUNT:
        return BinanceSpotClient.real_account_client()
    return None


def record_system_status(component: str, status: str, *, mode: str | None = None, message: str | None = None, metadata: dict | None = None) -> None:
    init_research_tables()
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {SYSTEM_STATUS_TABLE} (
                component, status, mode, reconciliation_status, message, metadata_json, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                component, status, mode, status if component == "reconciliation" else None,
                message, json.dumps(metadata or {}, ensure_ascii=True, sort_keys=True), pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
        conn.commit()


def synchronize_account(mode: str = ACCOUNT_MODE_LOCAL_PAPER) -> dict:
    """Persist account/balance snapshots. Local paper records DB-derived state."""
    init_research_tables()
    now = pd.Timestamp.now(tz="UTC").isoformat()
    snapshot_id = f"acct_{uuid.uuid4().hex}"
    client = _client_for_mode(mode)
    raw = {}
    balances: list[dict] = []
    status = "OK"
    message = "local_paper_snapshot"
    free_usdt = locked_usdt = total_equity = 0.0

    if client is None:
        with sqlite3.connect(DB_FILE) as conn:
            row = conn.execute(
                "SELECT cash, equity FROM portfolio_snapshots WHERE account_mode=? ORDER BY snapshot_id DESC LIMIT 1",
                (mode,),
            ).fetchone()
        free_usdt = float(row[0]) if row else 10_000.0
        total_equity = float(row[1]) if row else free_usdt
        balances = [{"asset": "USDT", "free": free_usdt, "locked": 0.0}]
    else:
        try:
            raw = client.account_info()
            balances = raw.get("balances", [])
            for bal in balances:
                if str(bal.get("asset", "")).upper() == "USDT":
                    free_usdt = float(bal.get("free", 0.0))
                    locked_usdt = float(bal.get("locked", 0.0))
                    total_equity = free_usdt + locked_usdt
            message = "binance_account_snapshot"
        except BinanceCredentialsError as exc:
            status = "PAUSED_NO_CREDENTIALS"
            message = str(exc)
        except Exception as exc:
            status = "PAUSED_NO_CONNECTION"
            message = str(exc)

    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {ACCOUNT_SNAPSHOTS_TABLE} (
                snapshot_id, account_mode, source, total_equity_usdt, free_usdt, locked_usdt,
                raw_json, timestamp_utc, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (snapshot_id, mode, "binance" if client else "local", total_equity, free_usdt, locked_usdt, json.dumps(raw, ensure_ascii=True, sort_keys=True), now, now),
        )
        for bal in balances:
            asset = str(bal.get("asset", "USDT")).upper()
            free = float(bal.get("free", 0.0))
            locked = float(bal.get("locked", 0.0))
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {BALANCE_SNAPSHOTS_TABLE} (
                    balance_snapshot_id, account_snapshot_id, account_mode, asset, free, locked, total,
                    usdt_value, timestamp_utc, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (f"bal_{uuid.uuid4().hex}", snapshot_id, mode, asset, free, locked, free + locked, free + locked if asset == "USDT" else None, now, json.dumps(bal, ensure_ascii=True, sort_keys=True)),
            )
        conn.commit()
    record_system_status("account_sync", status, mode=mode, message=message, metadata={"snapshot_id": snapshot_id})
    return {"status": status, "snapshot_id": snapshot_id, "mode": mode, "message": message, "free_usdt": free_usdt, "locked_usdt": locked_usdt}


def run_reconciliation(mode: str = ACCOUNT_MODE_LOCAL_PAPER) -> dict:
    init_research_tables()
    sync = synchronize_account(mode)
    status = "OK" if sync["status"] == "OK" else sync["status"]
    severity = "info" if status == "OK" else "error"
    differences = {}
    with sqlite3.connect(DB_FILE) as conn:
        open_trades = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE account_mode=? AND status IN ('APPROVED','ORDER_PENDING','ORDER_SENT','PARTIALLY_FILLED','OPEN','REDUCING','CLOSING')",
            (mode,),
        ).fetchone()[0]
        db_state = {"open_trades": int(open_trades)}
        conn.execute(
            f"""
            INSERT INTO {RECONCILIATION_EVENTS_TABLE} (
                account_mode, status, severity, message, db_state_json, exchange_state_json,
                differences_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mode, status, severity, sync.get("message"), json.dumps(db_state, ensure_ascii=True),
                json.dumps(sync, ensure_ascii=True), json.dumps(differences, ensure_ascii=True), pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
        conn.commit()
    record_system_status("reconciliation", status, mode=mode, message=sync.get("message"), metadata={"differences": differences})
    return {"status": status, "severity": severity, "mode": mode, "db_state": db_state, "sync": sync, "differences": differences}


def parse_args():
    parser = argparse.ArgumentParser(description="Reconcile Binance/account state with local SQLite ledger.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--mode", choices=[ACCOUNT_MODE_LOCAL_PAPER, ACCOUNT_MODE_TESTNET_PAPER, ACCOUNT_MODE_REAL], default=ACCOUNT_MODE_LOCAL_PAPER)
    return parser.parse_args()


def main():
    args = parse_args()
    print(json.dumps(run_reconciliation(mode=args.mode), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
