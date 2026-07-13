"""
Case #12 — SSE event-order contract for `POST /flows/process?stream=true`.

Frames must arrive as: `event: start` first, zero or more `event: node`
frames, `event: end` last (data carries the parsed flow output + thread_id).
"""

from __future__ import annotations

import json
import re
import uuid


def _create_runnable_flow(client) -> str:
    payload = {
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
    created = client.post("/helixflow/flows/", json={
        "name": f"sse-flow-{uuid.uuid4().hex[:8]}", "description": "sse test",
        "data": payload, "status": 1,
    })
    assert created.status_code == 201, created.text
    return created.json()["data"]["id"]


def _parse_sse(text: str):
    events = []
    for block in re.split(r"\n\n+", text.strip()):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event:
            events.append((event, data))
    return events


def test_flow_process_sse_event_order(client) -> None:
    """POST /helixflow/flows/process (SSE) must emit start → nodes → end in order."""
    flow_id = _create_runnable_flow(client)

    response = client.post(f"/helixflow/flows/process?id={flow_id}&stream=true")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    names = [event for event, _ in events]

    assert names[0] == "start"
    assert names[-1] == "end"
    assert set(names[1:-1]) <= {"node"}

    start_data = events[0][1]
    assert start_data["flow_id"] == str(flow_id)
    assert start_data["thread_id"]

    end_data = events[-1][1]
    assert end_data["thread_id"] == start_data["thread_id"]


def test_flow_process_sse_conversation_id_reused(client) -> None:
    """Passing conversation_id pins the LangGraph thread across calls."""
    flow_id = _create_runnable_flow(client)

    response = client.post(
        f"/helixflow/flows/process?id={flow_id}&stream=true&conversation_id=conv-1")
    events = _parse_sse(response.text)
    assert events[0][1]["thread_id"] == "conv-1"
    assert events[-1][1]["thread_id"] == "conv-1"
