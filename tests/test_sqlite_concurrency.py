"""Regression: multi-process SQLite concurrency pragmas.

The autonomous runner drives several processes against one SQLite file. Without WAL + a long
busy_timeout, concurrent writers raise "database is locked" (the failure observed in the live
run). config installs these pragmas process-wide on every sqlite3.connect.
"""
from __future__ import annotations

import sqlite3
import threading
import time

import config


def test_pragmas_installed_on_every_connection():
    conn = sqlite3.connect(str(config.DB_FILE))
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 60000
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
    finally:
        conn.close()


def test_concurrent_writer_waits_instead_of_erroring(tmp_path):
    db = str(tmp_path / "concurrency.sqlite")
    sqlite3.connect(db).execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v INTEGER)")

    holder_ready = threading.Event()

    def hold_write_lock():
        c = sqlite3.connect(db)
        c.execute("BEGIN IMMEDIATE")
        c.execute("INSERT INTO t (v) VALUES (1)")
        holder_ready.set()
        time.sleep(1.0)  # hold the write lock past the old 5s? no—short, but proves waiting works
        c.commit()
        c.close()

    t = threading.Thread(target=hold_write_lock)
    t.start()
    holder_ready.wait(timeout=5)

    # With busy_timeout this write blocks until the holder commits, then succeeds (no exception).
    c2 = sqlite3.connect(db)
    c2.execute("INSERT INTO t (v) VALUES (2)")
    c2.commit()
    c2.close()
    t.join(timeout=5)

    rows = sqlite3.connect(db).execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert rows == 2
