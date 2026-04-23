"""Regression for B3 — the 'is not empty' branch must use AND, not OR.

The pre-fix tautology meant the condition always evaluated to True for any
variable value, silently routing every flow down the 'is not empty' branch.
We parametrize across 6 representative values × 2 branches (empty / not empty).

Tests drive ``if_condition`` through a synthesised ``appstate + config``
mimicking what ``create_dynamic_state_graph`` builds at runtime.
"""

from __future__ import annotations

import pytest

from core.builtin.if_condition import if_condition


def _build_appstate(value):
    """appstate['fields'] maps names to values directly (StateField wrapping
    is B3.5, slated for P0'-b — see 组会纪要 §4 D15).

    ``if_condition`` reads ``appstate['fields'][param.value['reference']]``
    verbatim, so at runtime this position is whatever the upstream graph put
    there. For the purposes of testing the boolean correction we plug in the
    raw value, which matches the path that fires in production today.
    """
    return {"messages": [], "fields": {"var": value}}


class _Param:
    """Lightweight stand-in for the condition parameter the graph builds."""

    def __init__(self, name: str, compare: str):
        self.name = name
        self.value = {
            "reference": "var",
            "compare": compare,
            "compare_reference": False,
            "compare_value": None,
        }


def _build_config(compare: str, target: str = "target-A"):
    """Build the nested config layout that ``get_current_if_condition`` walks."""
    return {
        "metadata": {"langgraph_node": "ifc"},
        "configurable": {
            "_edges": {"ifc": "ifc"},  # points back to the same node
            "ifc": [
                {"param": _Param("if", compare), "target": target},
                {"param": _Param("else", compare), "target": "target-else"},
            ],
        },
    }


# (value, compare, expected_target)
#   - 'is empty' should match '' and None (return target-A); others fall through
#     to the else branch -> target-else.
#   - 'is not empty' is the B3 fix: only 'x' / 0 / False / [] etc. should match.
#
# NOTE: 0, False, [] are truthy-false but *not* '' or None, so under the
# corrected AND logic they count as 'not empty'. That matches the semantic
# of "string-or-null empty" used by the frontend's operator dropdown.
@pytest.mark.parametrize(
    "value,compare,expected_target",
    [
        # is empty — True for '' and None only
        ("", "is empty", "target-A"),
        (None, "is empty", "target-A"),
        ("x", "is empty", "target-else"),
        (0, "is empty", "target-else"),
        (False, "is empty", "target-else"),
        ([], "is empty", "target-else"),
        # is not empty — True for everything except '' and None
        ("", "is not empty", "target-else"),
        (None, "is not empty", "target-else"),
        ("x", "is not empty", "target-A"),
        (0, "is not empty", "target-A"),
        (False, "is not empty", "target-A"),
        ([], "is not empty", "target-A"),
    ],
)
def test_if_condition_and_logic(value, compare, expected_target):
    appstate = _build_appstate(value)
    config = _build_config(compare)

    result = if_condition(appstate, config)

    assert result == expected_target, (
        f"value={value!r} compare={compare!r} → got {result!r}, "
        f"expected {expected_target!r}"
    )
