"""One-click bootstrap idempotency guards (no subprocesses spawned)."""
from __future__ import annotations

import sqlite3

import pandas as pd

import config
from db_utils import init_research_tables
from autonomous_runner import AutonomousRunner


def _runner():
    return AutonomousRunner(
        symbols=["BTCUSDT"], timeframe="1h", timeframes=["1h"],
        no_dashboard=True, no_maintenance=True, bootstrap=False,
    )


def test_price_rows_and_features_current_guards():
    init_research_tables()
    r = _runner()
    # Unknown symbol -> no price rows, features not current.
    assert r._price_rows("NOPEUSDT", "1h") == 0
    assert r._features_current("NOPEUSDT", "1h") is False

    now = pd.Timestamp.now(tz="UTC").isoformat()
    with sqlite3.connect(config.DB_FILE) as conn:
        # Feature row at the CURRENT version -> features_current True (bootstrap would skip rebuild).
        conn.execute(
            f"INSERT OR REPLACE INTO {config.FEATURES_TABLE} (symbol, timeframe, datetime_utc, feature_version, updated_at_utc) VALUES (?,?,?,?,?)",
            ("BOOTUSDT", "1h", now, config.FEATURE_VERSION, now),
        )
        # A stale-version row for another symbol -> features_current False (bootstrap would rebuild).
        conn.execute(
            f"INSERT OR REPLACE INTO {config.FEATURES_TABLE} (symbol, timeframe, datetime_utc, feature_version, updated_at_utc) VALUES (?,?,?,?,?)",
            ("OLDUSDT", "1h", now, "v0_ancient", now),
        )
        conn.commit()

    assert r._features_current("BOOTUSDT", "1h") is True
    assert r._features_current("OLDUSDT", "1h") is False


def test_bootstrap_flag_wiring():
    assert _runner().bootstrap is False
    on = AutonomousRunner(symbols=["BTCUSDT"], timeframe="1h", timeframes=["1h"], no_dashboard=True, no_maintenance=True, bootstrap=True)
    assert on.bootstrap is True
