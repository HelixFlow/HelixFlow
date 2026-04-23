"""Smoke test — LangGraph 1.x import and trivial invoke.

* **P0'-a**: assert ``langgraph.__version__.startswith("1.")`` (the codebase
  currently pins 1.0.2).
* **P0'-b**: tighten to ``startswith("1.1.9")`` once that upgrade lands.

The trailing "119" in the filename is an intentional TODO marker so the
test name matches RFC §8's DB-2 contract without misleading the P0'-a
assertion strength.
"""

from __future__ import annotations

import importlib.metadata as _md
import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

pytestmark = pytest.mark.smoke


# Defined at module level so ``langgraph.graph.state._get_channels`` can
# resolve forward refs via ``typing.get_type_hints`` against our module globals.
class _SmokeState(TypedDict):
    count: Annotated[int, operator.add]


def _increment(state: _SmokeState) -> _SmokeState:
    return {"count": 1}


def test_langgraph_119_import_and_invoke():
    """Import langgraph 1.x and trivially invoke a 1-node StateGraph."""
    # P0'-a: accept any 1.x; P0'-b will tighten to "1.1.9".
    # langgraph doesn't expose ``__version__`` as a module attr — use metadata.
    version = _md.version("langgraph")
    assert version.startswith("1."), f"expected langgraph 1.x; got {version}"

    # Trivial invocation: one-node graph that mutates ``count``.
    sg = StateGraph(_SmokeState)
    sg.add_node("inc", _increment)
    sg.add_edge(START, "inc")
    sg.add_edge("inc", END)
    compiled = sg.compile()

    result = compiled.invoke({"count": 0})

    assert result == {"count": 1}, f"unexpected result: {result}"
