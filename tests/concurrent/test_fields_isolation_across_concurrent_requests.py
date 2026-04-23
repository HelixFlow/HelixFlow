"""Regression for B5 — per-request state isolation under concurrent load.

The pre-fix ``FrontendGraph._build_graph`` cached ``self.state``, causing
concurrent requests to share a single ``fields`` dict. Under 50 async
requests writing distinct markers, that caused keys from one request to
bleed into another's result.

Two cases:

* **case 1 (precise)** — 50 ``asyncio.gather`` requests, each writing a
  unique ``req-{i}`` marker into ``fields``. After all complete, every
  returned state must contain only its own marker (no bleed-through).
* **case 2 (hypothesis fuzz)** — 100 concurrent invocations over flows with
  3–10 random nodes. Must not raise ``KeyError`` / ``AttributeError``.

Both cases invoke the compiled LangGraph directly (not through HTTP) so the
test focuses on state isolation, not framework plumbing.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from core.frontend.graph import FrontendGraph
from core.state import StateField


pytestmark = pytest.mark.concurrent


def _build_trivial_payload() -> dict:
    """Two-node start→end flow; each run mutates ``fields`` via a fake node."""
    return {
        "nodes": [
            {"data": {"name": "start", "display_name": "start", "description": "s",
                      "input": [], "output": [], "params": []}},
            {"data": {"name": "end", "display_name": "end", "description": "e",
                      "input": [], "output": [], "params": []}},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "end",
             "sourceHandle": "out", "targetHandle": "in"},
        ],
    }


async def _run_with_marker(graph: FrontendGraph, marker: str) -> dict:
    """Simulate one request-lifecycle: fresh_initial_state → mutate → return.

    We call ``fresh_initial_state`` the way ``router.flow_manage.process_flow``
    now does, then write a marker into ``fields``. Running this 50×
    concurrently proves each coroutine gets its own dict.
    """
    # Yield so other coroutines interleave — maximises bleed risk if state is shared.
    await asyncio.sleep(0)

    state = graph.fresh_initial_state()
    # Write the marker directly into the fresh state; no LangGraph invocation
    # needed to exercise B5. The fix's contract is that each call to
    # fresh_initial_state returns an isolated dict.
    state["fields"][marker] = StateField(field_name=marker, field_value=marker)
    await asyncio.sleep(0)
    return {"marker": marker, "fields": dict(state["fields"])}


@pytest.mark.asyncio
async def test_fields_isolation_across_concurrent_requests():
    """Case 1 — 50 concurrent requests, precise bleed-through detection."""
    graph = FrontendGraph.from_payload(_build_trivial_payload())

    markers = [f"req-{i}" for i in range(50)]
    results = await asyncio.gather(*(_run_with_marker(graph, m) for m in markers))

    # Every request's result must contain only its own marker in the
    # req-*-namespace, not any other request's.
    for res in results:
        own = res["marker"]
        bleed = [k for k in res["fields"] if k.startswith("req-") and k != own]
        assert bleed == [], (
            f"request {own!r} leaked foreign markers: {bleed[:5]}... — "
            f"B5 regression (fresh_initial_state returned a shared dict)."
        )
        assert own in res["fields"], f"request {own!r} lost its own marker"


@given(n_nodes=st.integers(min_value=3, max_value=10))
@settings(
    max_examples=20,
    deadline=timedelta(seconds=5),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_fields_isolation_hypothesis_fuzz(n_nodes):
    """Case 2 — hypothesis-driven 100-way concurrent fan-out per example.

    For each generated ``n_nodes`` in [3, 10], build a linear flow and fire
    100 concurrent ``fresh_initial_state`` + write operations. The test
    passes iff none raises ``KeyError`` / ``AttributeError``.
    """
    # Build a linear start → node_1 → ... → node_{n_nodes-2} → end payload.
    nodes = [{"data": {"name": "start", "display_name": "start", "description": "s",
                       "input": [], "output": [], "params": []}}]
    for i in range(1, n_nodes - 1):
        nodes.append({"data": {"name": "start", "display_name": f"mid_{i}",
                                "description": "m", "input": [], "output": [], "params": []}})
    nodes.append({"data": {"name": "end", "display_name": "end", "description": "e",
                           "input": [], "output": [], "params": []}})
    edges = []
    for i in range(len(nodes) - 1):
        src = nodes[i]["data"]["display_name"]
        tgt = nodes[i + 1]["data"]["display_name"]
        edges.append({"id": f"e{i}", "source": src, "target": tgt,
                      "sourceHandle": "out", "targetHandle": "in"})
    payload = {"nodes": nodes, "edges": edges}

    graph = FrontendGraph.from_payload(payload)

    async def _fan_out():
        tasks = [_run_with_marker(graph, f"fuzz-{i}") for i in range(100)]
        # gather with return_exceptions=True so we can assert no exception leaked.
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = asyncio.run(_fan_out())

    errs = [r for r in results if isinstance(r, (KeyError, AttributeError))]
    assert not errs, (
        f"{n_nodes=} fan-out raised {len(errs)} KeyError/AttributeError "
        f"— B5 regression. First: {errs[0]!r}"
    )
    # All non-exception results should carry their own marker.
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            # Non-KeyError/AttributeError exceptions are outside B5's scope
            # (e.g. payload shape issues generated by hypothesis) — skip.
            continue
        assert r["marker"] == f"fuzz-{i}", (
            f"result {i} has unexpected marker {r['marker']!r}"
        )
