"""Multi-turn memory: reusing a thread_id restores AppState.messages history."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from core.frontend.graph import FrontendGraph
from core.frontend.node import FrontendNode
from core.initial import NODE_FUNCTIONS


def _payload():
    def node(name):
        return {"data": {"name": name, "display_name": name, "description": "",
                         "input": [], "output": [], "params": []}}
    return {
        "nodes": [node("start"), node("echo"), node("end")],
        "edges": [
            {"id": "e1", "source": "start", "target": "echo", "sourceHandle": "out", "targetHandle": "in"},
            {"id": "e2", "source": "echo", "target": "end", "sourceHandle": "out", "targetHandle": "in"},
        ],
    }


def test_same_thread_id_accumulates_messages():
    def echo(state):
        seen = len(state.get("messages") or [])
        return {"fields": {"echo/seen": seen},
                "messages": [AIMessage(content=f"turn-{seen}")]}

    NODE_FUNCTIONS["echo"] = FrontendNode(name="echo", display_name="echo", description="",
                                          input=[], output=[], params=[], function=echo)
    try:
        graph = FrontendGraph.from_payload(_payload())
        shared_saver = InMemorySaver()
        compiled = graph.compile_graph(checkpointer=shared_saver)
        config = graph.config
        config["configurable"]["thread_id"] = "conversation-42"

        def seen(result):
            value = result["fields"]["echo/seen"]
            return getattr(value, "field_value", value)

        first = compiled.invoke(input=graph.fresh_initial_state(), config=config)
        assert seen(first) == 0
        assert len(first["messages"]) == 1

        # 第二轮：同 thread_id，checkpointer 恢复上一轮的 messages
        second = compiled.invoke(input=graph.fresh_initial_state(), config=config)
        assert seen(second) == 1
        assert [m.content for m in second["messages"]] == ["turn-0", "turn-1"]

        # 不同 thread_id 则从零开始，互不串扰
        config_other = dict(config)
        config_other["configurable"] = {**config["configurable"], "thread_id": "conversation-other"}
        other = compiled.invoke(input=graph.fresh_initial_state(), config=config_other)
        assert seen(other) == 0
    finally:
        NODE_FUNCTIONS.pop("echo", None)
