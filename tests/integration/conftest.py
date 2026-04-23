"""
pytest-docker fixtures for integration tests.

Integration tests bring up the full docker-compose stack (helixflow + mysql)
and hit the running API. They are marked with ``@pytest.mark.integration`` and
skipped automatically in the default unit run.

Usage::

    @pytest.mark.integration
    def test_something(helix_api_url):
        r = httpx.get(f"{helix_api_url}/helixflow/health")
        assert r.status_code == 200
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def docker_compose_file() -> str:
    """Point pytest-docker at the repo-root docker-compose.yaml."""
    return str(_PROJECT_ROOT / "docker-compose.yaml")


@pytest.fixture(scope="session")
def docker_compose_project_name() -> str:
    """Isolate the compose project so concurrent CI runs do not collide."""
    return os.environ.get("HELIXFLOW_COMPOSE_PROJECT", "helixflow-it")


def _is_responsive(url: str, timeout: float = 2.0) -> bool:
    import httpx

    try:
        response = httpx.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def helix_api_url(docker_ip, docker_services) -> Iterator[str]:
    """Bring up the stack and return the API base URL once /health answers 200."""
    port = docker_services.port_for("helixflow-api", 11110)
    url = f"http://{docker_ip}:{port}"
    docker_services.wait_until_responsive(
        timeout=120.0,
        pause=1.0,
        check=lambda: _is_responsive(f"{url}/helixflow/health"),
    )
    yield url


@pytest.fixture(scope="session")
def wait_for_mysql(docker_ip, docker_services):
    """Ensure mysql is reachable before tests that talk to it directly."""
    port = docker_services.port_for("mysql", 3306)
    deadline = time.time() + 120

    def _mysql_ready() -> bool:
        try:
            import pymysql  # pragma: no cover — only in integration env
        except Exception:
            return False
        try:
            conn = pymysql.connect(host=docker_ip, port=port, user="root", password="123123")
            conn.close()
            return True
        except Exception:
            return False

    while time.time() < deadline:
        if _mysql_ready():
            return f"{docker_ip}:{port}"
        time.sleep(1)
    raise RuntimeError("mysql did not become ready within 120s")
