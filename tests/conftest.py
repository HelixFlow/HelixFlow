"""
Root pytest configuration for HelixFlow.

Fixture layering (aligned with 组会纪要 §3.3):
    tests/_fixtures/app.py  — FastAPI TestClient fixtures
    tests/_fixtures/db.py   — SQLite in-memory sqlmodel Session fixture
    tests/integration/conftest.py — pytest-docker compose fixture

Unit / api / smoke / concurrent tests use SQLite in-memory; integration tests
use pytest-docker to spin up the full docker-compose stack.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of how pytest is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Environment defaults for tests — keeps us away from the production MySQL URL
# that `config.arg_settings` reads at import time.
# -----------------------------------------------------------------------------
os.environ.setdefault("HELIXFLOW_ENV", "test")
os.environ.setdefault("APP_TABLE_URL", "sqlite:///:memory:")

# -----------------------------------------------------------------------------
# `config.arg_settings` calls `parser.parse_args()` at import time, which would
# fail under pytest because pytest's own CLI args leak into sys.argv. We stash
# the real argv and feed argparse an empty list so the module loads cleanly.
# Fixing config.arg_settings itself is coder-1's territory (config bugfix is
# listed as P0'-b cleanup); for the P0'-a test skeleton this shim is enough.
# -----------------------------------------------------------------------------
_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
try:
    import config.arg_settings  # noqa: F401 — warm the argparse-at-import module
finally:
    sys.argv = _ORIG_ARGV

# Fixture plugins — these are `pytest_plugins` imported from this root
# conftest so the nested per-folder conftests stay thin.
pytest_plugins = [
    "tests._fixtures.app",
    "tests._fixtures.db",
]
