"""
Case #12 — SSE event-order contract for `POST /flows/process`.

Owner: coder-2 (T14)
Status: skeleton — skipped until P0'-b Sprint 2 ships the streaming endpoint.

When SSE lands, replace the skip with a real event-stream assertion:
    1.  `event: start` is first
    2.  zero or more `event: node` frames in graph topological order
    3.  `event: end` is last (with `data: {...final state...}`)
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="P0'-b Sprint 2: SSE endpoint not yet implemented")
def test_flow_process_sse_event_order(client) -> None:
    """POST /helixflow/flows/process (SSE) must emit start → nodes → end in order."""
    # Skeleton only. When SSE lands, use httpx streaming or starlette testclient
    # stream=True and assert the frame sequence.
    assert True
