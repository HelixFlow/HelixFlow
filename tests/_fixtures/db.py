"""
SQLite in-memory session fixture.

All unit / api tests run against an ephemeral SQLite database so we do not
need a MySQL container for the fast test tier. Integration tests that require
the full MySQL stack live under tests/integration/ and use pytest-docker.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine


@pytest.fixture(scope="session")
def db_engine():
    """A shared in-memory SQLite engine (single connection, usable across threads)."""
    # StaticPool + check_same_thread=False keeps the in-memory DB alive across
    # every Session opened during the test session.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Ensure all sqlmodel-declared tables are present.
    try:
        import database.model  # noqa: F401 — registers tables with SQLModel.metadata
    except Exception:  # pragma: no cover — in case imports are partial
        pass

    SQLModel.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine) -> Iterator[Session]:
    """Per-test session with rollback isolation."""
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()


@pytest.fixture()
def db_session_factory(db_engine):
    """Factory that yields sessions bound to the shared in-memory engine.

    Used by the FastAPI `dependency_overrides` path — FastAPI's DI expects a
    callable that yields a session (matching `get_table_session`).
    """

    @contextmanager
    def _factory() -> Iterator[Session]:
        with Session(db_engine) as session:
            yield session

    return _factory
