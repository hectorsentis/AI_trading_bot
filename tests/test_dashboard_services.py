from __future__ import annotations

import sqlite3
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import dashboard_auth
import dashboard_controls


class DashboardAuthTests(unittest.TestCase):
    def test_password_hash_roundtrip(self) -> None:
        stored = dashboard_auth.generate_password_hash("correct horse battery staple", salt=b"1234567890123456")
        self.assertTrue(dashboard_auth.verify_password("correct horse battery staple", stored))
        self.assertFalse(dashboard_auth.verify_password("wrong", stored))
        self.assertFalse(dashboard_auth.verify_password("", stored))


class DashboardControlsTests(unittest.TestCase):
    def _db(self) -> Path:
        tmp_root = ROOT / "_test_runtime"
        tmp_root.mkdir(exist_ok=True)
        tmp_dir = tmp_root / f"case_{uuid.uuid4().hex}"
        tmp_dir.mkdir()
        db_path = tmp_dir / "dashboard.sqlite"
        sqlite3.connect(db_path).close()
        return db_path

    def test_action_and_runtime_config_are_audited(self) -> None:
        db_path = self._db()
        dashboard_controls.ensure_dashboard_tables(db_path)
        action_id = dashboard_controls.request_model_training({"symbols": "BTCUSDT", "timeframe": "1h"}, "tester", db_path)
        self.assertGreater(action_id, 0)
        dashboard_controls.update_runtime_config("model.min_confidence", "0.61", "tester", db_path=db_path)

        with sqlite3.connect(db_path) as conn:
            action = conn.execute("SELECT action_type, status, requested_by FROM bot_control_actions WHERE action_id=?", (action_id,)).fetchone()
            cfg = conn.execute("SELECT value, updated_by FROM runtime_config WHERE key='model.min_confidence'").fetchone()
            audit_count = conn.execute("SELECT COUNT(*) FROM runtime_config_audit WHERE key='model.min_confidence'").fetchone()[0]

        self.assertEqual(action, ("REQUEST_RETRAIN", "pending", "tester"))
        self.assertEqual(cfg, ("0.61", "tester"))
        self.assertEqual(audit_count, 1)

    def test_live_trading_cannot_be_enabled_by_default(self) -> None:
        with self.assertRaises(ValueError):
            dashboard_controls.validate_runtime_value("execution.live_trading_enabled", "true")

    def test_kill_switch_action_type_is_supported(self) -> None:
        db_path = self._db()
        dashboard_controls.ensure_dashboard_tables(db_path)
        action_id = dashboard_controls.request_bot_action("KILL_SWITCH", {"source": "test"}, "tester", db_path=db_path)
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT action_type, status FROM bot_control_actions WHERE action_id=?", (action_id,)).fetchone()
        self.assertEqual(row, ("KILL_SWITCH", "pending"))

    def test_activate_model_updates_registry_and_queues_action(self) -> None:
        db_path = self._db()
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE model_registry (model_id TEXT PRIMARY KEY, is_active INTEGER DEFAULT 0, updated_at_utc TEXT)")
            conn.execute("INSERT INTO model_registry (model_id, is_active) VALUES ('m1', 0)")
            conn.commit()

        action_id = dashboard_controls.activate_model("m1", "tester", db_path)
        with sqlite3.connect(db_path) as conn:
            is_active = conn.execute("SELECT is_active FROM model_registry WHERE model_id='m1'").fetchone()[0]
            signal_enabled = conn.execute("SELECT signal_enabled FROM model_control WHERE model_id='m1'").fetchone()[0]
            action = conn.execute("SELECT action_type FROM bot_control_actions WHERE action_id=?", (action_id,)).fetchone()[0]

        self.assertEqual(is_active, 1)
        self.assertEqual(signal_enabled, 1)
        self.assertEqual(action, "ACTIVATE_MODEL")


if __name__ == "__main__":
    unittest.main()
