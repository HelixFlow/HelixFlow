"""Regression for B2 — ``router.route`` must pass ``APIRouter`` instances to
``include_router``, not module objects. The previous code blew up with
``AssertionError('A path prefix must not end with /, as the routes will start with /')``
or similar FastAPI startup assert during ``create_app()``.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.testclient import TestClient

from main import create_app
from router import route as route_module


def test_router_startup_without_assert():
    """Three independent assertions guarding B2's regression surface.

    1. Top-level ``router`` is an ``APIRouter`` instance (not a module).
    2. All three sub-routes are mounted under ``/helixflow`` (flow_manage,
       operator_manage, user_manager).
    3. ``create_app()`` starts and ``/helixflow/health`` returns 200.
    """
    # (1) The top-level router object is a real APIRouter instance.
    assert isinstance(route_module.router, APIRouter), (
        "route.py must expose an APIRouter instance as `router`"
    )

    # (2) The three sub-routers are mounted and their characteristic paths are
    #     reachable from the top-level router.
    app = create_app()
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    for expected in (
        "/helixflow/flows/",
        "/helixflow/operators/all",
        "/helixflow/user/login",
    ):
        assert expected in paths, (
            f"expected {expected!r} to be mounted; got {sorted(paths)}"
        )

    # (3) Startup completes without AssertionError and /health returns 200.
    with TestClient(app) as client:
        resp = client.get("/helixflow/health")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "OK"}
