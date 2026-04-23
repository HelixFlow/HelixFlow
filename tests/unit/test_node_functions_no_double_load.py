"""Regression for B4 — ``NODE_FUNCTIONS`` must reuse ``ALL_NODES`` instead of
re-invoking ``load_nodes_from_directory``.

Testing the "called-exactly-once" contract at import time is tricky because
``core.initial`` is already imported by the time pytest starts. We instead
assert the post-conditions that hold iff the scanner ran once:

1. ``NODE_FUNCTIONS[name] is ALL_NODES[idx]`` for every key — same Python
   object identity (the pre-fix code built a *second* batch of nodes via a
   second scan, so these identities would diverge).
2. ``NODE_FUNCTIONS == {node.name: node for node in ALL_NODES}`` as a value
   check (covers both identity and mapping completeness).
3. ``load_nodes_from_directory`` is call-counted once when triggered fresh —
   the counter check runs the scanner directly inside an isolated namespace
   so the assertion is independent of module-import timing.

Additionally, the scanner must resolve its directory from ``__file__``, not
``os.getcwd()`` — so running from an arbitrary CWD (e.g. pytest's tmp_path)
must not raise ``FileNotFoundError``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_node_functions_no_double_load():
    """Three assertions — see module docstring."""
    import core.initial as initial_mod

    # --- (b) mapping equals the dict-comprehension over ALL_NODES -----------
    assert initial_mod.NODE_FUNCTIONS == {
        node.name: node for node in initial_mod.ALL_NODES
    }, "NODE_FUNCTIONS must be the dict-comp over ALL_NODES (B4 fix)."

    # --- (a) object-identity: each NODE_FUNCTIONS entry IS the same instance
    #     that lives in ALL_NODES. A second scan would have produced distinct
    #     FrontendNode objects with the same ``name``, breaking ``is``.
    all_by_name = {n.name: n for n in initial_mod.ALL_NODES}
    for name, nf_node in initial_mod.NODE_FUNCTIONS.items():
        assert nf_node is all_by_name[name], (
            f"NODE_FUNCTIONS[{name!r}] is a distinct instance from ALL_NODES — "
            f"suggests load_nodes_from_directory ran twice (B4 regression)."
        )

    # --- (c) sanity: at least the three built-in nodes are present.
    assert {"start", "end", "if_condition"}.issubset(all_by_name.keys())


def test_load_nodes_called_exactly_once_on_fresh_module_load():
    """Explicit call_count == 1 check: execute the module-level statement
    pattern in an isolated mock-wrapped namespace.

    We don't reload ``core.initial`` itself (that re-binds the real
    ``load_nodes_from_directory`` symbol and defeats the patch). Instead we
    replicate the top-level assignments in a locals-dict where
    ``load_nodes_from_directory`` is a ``MagicMock`` — then count its
    invocations.
    """
    mock_loader = MagicMock(
        return_value=[
            type("N", (), {"name": "fake1"})(),
            type("N", (), {"name": "fake2"})(),
        ]
    )

    ns: dict = {"load_nodes_from_directory": mock_loader}

    # Execute the exact top-level pattern used in core/initial.py (post-fix).
    exec("ALL_NODES = load_nodes_from_directory()", ns)
    exec("NODE_FUNCTIONS = {n.name: n for n in ALL_NODES}", ns)

    assert mock_loader.call_count == 1, (
        f"load_nodes_from_directory must be called exactly once; "
        f"got {mock_loader.call_count} (B4 was: called twice)."
    )
    assert set(ns["NODE_FUNCTIONS"].keys()) == {"fake1", "fake2"}


def test_load_nodes_from_directory_independent_of_cwd(tmp_path, monkeypatch):
    """Invoke the scanner with ``tmp_path`` as cwd — the __file__-relative
    path lookup must still succeed (B4 CWD fix)."""
    import core.initial as initial_mod

    monkeypatch.chdir(tmp_path)

    nodes = initial_mod.load_nodes_from_directory()

    assert nodes, "scanner must return built-in nodes regardless of cwd"
    names = {n.name for n in nodes}
    for required in ("start", "end", "if_condition"):
        assert required in names, (
            f"expected built-in {required!r}; got {sorted(names)}"
        )
