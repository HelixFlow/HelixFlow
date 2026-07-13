"""SqliteSaver checkpointer: durable, real (no more warn-and-fallback)."""

from __future__ import annotations

import core.frontend.graph as graph_module
from core.frontend.graph import FrontendGraph, get_sqlite_saver


def _trivial_payload():
    return {
        "nodes": [
            {"data": {"name": "start", "display_name": "start", "description": "",
                      "input": [], "output": [], "params": []}},
            {"data": {"name": "end", "display_name": "end", "description": "",
                      "input": [], "output": [], "params": []}},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "end",
             "sourceHandle": "out", "targetHandle": "in"},
        ],
    }


def test_sqlite_checkpointer_persists_thread_state(tmp_path, monkeypatch):
    db_path = tmp_path / "checkpoints.db"
    monkeypatch.setenv("HELIXFLOW_CHECKPOINT_DB", str(db_path))
    monkeypatch.setattr(graph_module, "_sqlite_saver", None)

    graph = FrontendGraph.from_payload(_trivial_payload())
    compiled = graph.compile_graph(checkpointer_type="sqlite")
    config = graph.config
    config["configurable"]["thread_id"] = "durable-thread"
    compiled.invoke(input=graph.fresh_initial_state(), config=config)

    assert db_path.exists(), "sqlite checkpointer must create the DB file"
    saver = get_sqlite_saver()
    checkpoint = saver.get({"configurable": {"thread_id": "durable-thread"}})
    assert checkpoint is not None, "checkpoint for the thread must be retrievable"

    # 单例：同一进程内重复获取是同一个 saver
    assert get_sqlite_saver() is saver
