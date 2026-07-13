"""Engine tests: multi-target edges (fan-out) and if_condition back-edges (loops)."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from core.frontend.graph import FrontendGraph
from core.frontend.node import FrontendNode
from core.initial import NODE_FUNCTIONS


def _node_data(name, display_name=None, params=None):
    return {"data": {"name": name, "display_name": display_name or name, "description": "",
                     "input": [], "output": [], "params": params or []}}


def _edge(source, target, handle="out", idx=[0]):
    idx[0] += 1
    return {"id": f"e{idx[0]}", "source": source, "target": target,
            "sourceHandle": handle, "targetHandle": "in"}


@pytest.fixture()
def inject_node():
    """Temporarily register a custom node function in NODE_FUNCTIONS."""
    injected = []

    def _inject(name, fn):
        node = FrontendNode(name=name, display_name=name, description="",
                            input=[], output=[], params=[], function=fn)
        NODE_FUNCTIONS[name] = node
        injected.append(name)
        return node

    yield _inject
    for name in injected:
        NODE_FUNCTIONS.pop(name, None)


def _val(value):
    """end_node 的 update_state_by_relation 会把裸值包成 StateField。"""
    return getattr(value, "field_value", value)


def _invoke(payload, thread_id="t-loop"):
    graph = FrontendGraph.from_payload(payload)
    compiled = graph.compile_graph(checkpointer=InMemorySaver())
    config = graph.config
    config["configurable"]["thread_id"] = thread_id
    return compiled.invoke(input=graph.fresh_initial_state(), config=config)


def test_multi_target_fanout_runs_all_branches(inject_node):
    """One source with two outgoing edges executes both branches (fan-out)."""
    inject_node("brancha", lambda state: {"fields": {"brancha/hit": True}})
    inject_node("branchb", lambda state: {"fields": {"branchb/hit": True}})

    payload = {
        "nodes": [_node_data("start"), _node_data("brancha"), _node_data("branchb"), _node_data("end")],
        "edges": [
            _edge("start", "brancha"),
            _edge("start", "branchb"),
            _edge("brancha", "end"),
            _edge("branchb", "end"),
        ],
    }
    result = _invoke(payload, thread_id="t-fanout")
    assert _val(result["fields"].get("brancha/hit")) is True
    assert _val(result["fields"].get("branchb/hit")) is True


def test_if_condition_back_edge_loops_until_condition_flips(inject_node):
    """counter → if_condition → counter (back edge) loops until count == 3."""

    def counter(state):
        current = state["fields"].get("counter/count") or 0
        return {"fields": {"counter/count": current + 1}}

    inject_node("counter", counter)

    loop_param = {"name": "loop", "value": {
        "reference": "counter/count", "compare": "shorter than",
        "compare_reference": False, "compare_value": 3}}
    else_param = {"name": "else", "value": {}}

    payload = {
        "nodes": [
            _node_data("start"),
            _node_data("counter"),
            _node_data("if_condition", "if_condition_1", params=[loop_param, else_param]),
            _node_data("end"),
        ],
        "edges": [
            _edge("start", "counter"),
            _edge("counter", "if_condition_1"),
            _edge("if_condition_1", "counter", handle="loop"),  # 回边：构成环
            _edge("if_condition_1", "end", handle="else"),
        ],
    }
    result = _invoke(payload, thread_id="t-loop-3")
    assert _val(result["fields"]["counter/count"]) == 3


def test_build_edges_returns_target_lists():
    payload = {
        "nodes": [_node_data("start"), _node_data("end")],
        "edges": [_edge("start", "end")],
    }
    graph = FrontendGraph.from_payload(payload)
    assert graph.edges == {"start": ["end"]}
