"""Regression tests for B1 — module- and instance-level ``compile_graph`` must
accept a default ``checkpointer_type`` so callers (e.g. ``update_flow``
online-validation path) don't need to supply one.
"""

from __future__ import annotations

import pytest

from core.frontend.graph import FrontendGraph, _CHECKPOINTER_FACTORY, compile_graph


def _trivial_payload() -> dict:
    """Smallest payload accepted by ``FrontendGraph.from_payload``."""
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


def test_compile_graph_default_checkpointer():
    """Module-level ``compile_graph(data)`` without ``checkpointer_type`` returns
    a compiled graph with an ``invoke`` method — covering B1's main case."""
    data = _trivial_payload()

    compiled = compile_graph(data)  # no checkpointer_type argument

    assert compiled is not None, "compile_graph should return a compiled graph"
    assert hasattr(compiled, "invoke"), (
        "compiled graph must expose .invoke() so downstream callers can run it"
    )

    # Instance method default should also work
    graph = FrontendGraph.from_payload(data)
    compiled_inst = graph.compile_graph()  # no explicit type
    assert compiled_inst is not None
    assert hasattr(compiled_inst, "invoke")

    # Factory must contain the three documented keys (memory real, sqlite/postgres stubs)
    assert set(_CHECKPOINTER_FACTORY.keys()) >= {"memory", "sqlite", "postgres"}


def test_compile_graph_rejects_unknown_checkpointer():
    """Unknown checkpointer types fail fast with a clear ValueError."""
    data = _trivial_payload()
    graph = FrontendGraph.from_payload(data)

    with pytest.raises(ValueError, match="not supported"):
        graph.compile_graph(checkpointer_type="redis")
