"""
Case #13 — LangGraph PostgresSaver write/read/delete roundtrip.

Owner: coder-2 (T14)
Status: skeleton — enabled once P0'-b installs `langgraph-checkpoint-postgres`.
"""

from __future__ import annotations

import pytest

# When the optional dependency ships (P0'-b), this importorskip becomes a
# no-op and the real test body runs.
pytest.importorskip(
    "langgraph.checkpoint.postgres",
    reason="PostgresSaver is a P0'-b deliverable; not installed in P0'-a",
)


pytestmark = pytest.mark.persistence


def test_postgres_saver_roundtrip() -> None:
    """Write → read → delete checkpoint via LangGraph PostgresSaver."""
    # Real body once P0'-b lands:
    #   1. Spin up ephemeral PG via pytest-postgresql or compose
    #   2. saver.put({'thread_id': 't1', ...})
    #   3. saver.get_tuple({'configurable': {'thread_id': 't1'}})
    #   4. saver.delete_thread('t1') — verify empty
    raise AssertionError("PostgresSaver roundtrip body is P0'-b Sprint 2")
