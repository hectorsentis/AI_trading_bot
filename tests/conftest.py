"""Test harness setup.

Sets an isolated temporary SQLite DB via SQLITE_DB_PATH BEFORE any `src` module imports
`config` (which resolves `DB_FILE` at import time), and puts `src/` on the import path.
This guarantees tests never touch the operational database.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Must be set before `import config` anywhere in the test process.
_TMP_DB = Path(tempfile.mkdtemp(prefix="trading_v02_test_")) / "test_market_data.sqlite"
os.environ.setdefault("SQLITE_DB_PATH", str(_TMP_DB))
