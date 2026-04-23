"""
Factory Boy factories for HelixFlow test data.

All test objects that touch the DB should go through a factory here so the
schema change surface is centralised. Added per 组会纪要 §3.3 (task T03).
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict

import factory

try:  # pragma: no cover — only executed when DB models are importable
    from database.model.flow import Flow
except Exception:  # pragma: no cover — fallback for unit tests without DB
    Flow = None  # type: ignore[assignment]


def _minimal_flow_payload() -> Dict[str, Any]:
    """Return a minimal valid flow graph payload (nodes + edges)."""
    start_id = "start"
    end_id = "end"
    return {
        "nodes": [
            {"id": start_id, "name": "start", "type": "StartNode", "params": []},
            {"id": end_id, "name": "end", "type": "EndNode", "params": []},
        ],
        "edges": [
            {"source": start_id, "target": end_id},
        ],
    }


class FlowDataFactory(factory.DictFactory):
    """Produces a valid dict payload suitable for `Flow.data`."""

    nodes = factory.LazyFunction(lambda: _minimal_flow_payload()["nodes"])
    edges = factory.LazyFunction(lambda: _minimal_flow_payload()["edges"])


if Flow is not None:  # pragma: no branch

    class FlowFactory(factory.Factory):
        """Build a `Flow` sqlmodel instance for API/DB tests.

        Usage::

            flow = FlowFactory.build(name="my-flow")

        The `data` field is serialised to JSON (matching the router's
        `json_serialization` contract).
        """

        class Meta:
            model = Flow

        id = factory.LazyFunction(uuid.uuid4)
        name = factory.Sequence(lambda n: f"flow-{n}")
        user_id = 1
        description = factory.Faker("sentence", nb_words=4)
        logo = None
        status = 1
        data = factory.LazyFunction(lambda: json.dumps(_minimal_flow_payload()))

else:  # pragma: no cover

    class FlowFactory:  # type: ignore[no-redef]
        """Placeholder used when DB deps are unavailable in the current env."""

        @staticmethod
        def build(**kwargs: Any) -> Dict[str, Any]:
            base = {
                "id": str(uuid.uuid4()),
                "name": f"flow-{uuid.uuid4().hex[:8]}",
                "user_id": 1,
                "description": "placeholder",
                "logo": None,
                "status": 1,
                "data": json.dumps(_minimal_flow_payload()),
            }
            base.update(kwargs)
            return base


__all__ = ["FlowFactory", "FlowDataFactory", "_minimal_flow_payload"]
