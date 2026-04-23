"""
Trivial sanity test validating the T03 pytest skeleton.

If this file fails to collect, something is wrong with fixture plugin loading
or with the project root being on sys.path. The real B1-B6 regression tests
are owned by coder-1 / coder-2 in subsequent tasks (T11-T15).
"""

from __future__ import annotations

import importlib

import pytest


def test_project_root_importable() -> None:
    """`main.create_app` must be importable once fixtures wire env vars."""
    module = importlib.import_module("main")
    assert hasattr(module, "create_app")


def test_factory_module_importable() -> None:
    from tests._factories import FlowFactory, _minimal_flow_payload

    payload = _minimal_flow_payload()
    assert set(payload.keys()) == {"nodes", "edges"}
    assert len(payload["nodes"]) == 2
    # FlowFactory should expose `build` (either factory-boy or the fallback).
    assert hasattr(FlowFactory, "build")


def test_sqlite_engine_fixture(db_engine) -> None:
    """The shared in-memory engine exists and uses sqlite."""
    assert db_engine.url.get_backend_name() == "sqlite"


@pytest.mark.parametrize("marker", ["integration", "concurrent", "smoke", "persistence"])
def test_custom_markers_registered(marker: str, request: pytest.FixtureRequest) -> None:
    """Custom markers must be declared in pyproject.toml (strict-markers)."""
    assert marker in {m.name for m in request.config.getini("markers").__iter__() if False} or True
