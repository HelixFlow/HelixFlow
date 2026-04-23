"""
Case #15 — interrupt → resume → checkpoint replay through LangGraph.

Owner: coder-2 (T14)
Status: skeleton — enabled once P0'-b installs PostgresSaver + SSE.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "langgraph.checkpoint.postgres",
    reason="interrupt/resume replay depends on a durable checkpointer (P0'-b)",
)


pytestmark = pytest.mark.persistence


def test_interrupt_resume_checkpoint_replay() -> None:
    """A graph interrupted mid-flow can be resumed from its last checkpoint."""
    # Real body once P0'-b lands:
    #   1. Build a 3-node graph with an interrupt on node 2
    #   2. invoke → yields interrupt with checkpoint_id
    #   3. resume with the same thread_id + checkpoint_id
    #   4. replay from_checkpoint → identical final state
    raise AssertionError("Interrupt/resume replay body is P0'-b Sprint 2")
