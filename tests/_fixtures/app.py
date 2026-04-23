"""
FastAPI TestClient fixtures.

These fixtures import `main.create_app()` lazily so the test collection phase
does not crash even if a partially-refactored module raises at import time.
Sessions are overridden to use the SQLite in-memory engine from
`tests._fixtures.db`.
"""

from __future__ import annotations

from typing import Iterator

import pytest


@pytest.fixture()
def app(db_session_factory):  # noqa: D401 — fixture, not a function
    """Return a FastAPI app instance with DB overridden to SQLite in-memory."""
    # Import lazily so that environment overrides in tests/conftest.py take
    # effect before `config.arg_settings` is evaluated.
    from main import create_app

    try:
        from database.base import get_table_session
    except Exception:  # pragma: no cover
        get_table_session = None  # type: ignore[assignment]

    application = create_app()

    if get_table_session is not None:

        def _override() -> Iterator:
            with db_session_factory() as session:
                yield session

        application.dependency_overrides[get_table_session] = _override

    return application


@pytest.fixture()
def client(app):
    """Synchronous FastAPI TestClient.

    Requires ``httpx<0.28`` — httpx 0.28 removed the ``app=`` kwarg from
    ``httpx.Client`` that Starlette 0.27.0 still relies on. See
    ``requirements-dev.txt`` for the pin.
    """
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
async def async_client(app):
    """Async HTTPX client against the in-process ASGI app.

    Used by concurrent / SSE / streaming tests.
    """
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
