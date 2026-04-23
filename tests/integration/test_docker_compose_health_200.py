"""Integration test — docker-compose brings up helixflow-api + mysql, and
``GET /helixflow/health`` returns 200.

Uses ``pytest-docker`` (v3+) to manage the compose lifecycle so tests clean
up after themselves. The fixtures below follow the plugin's
``docker_compose_file`` / ``docker_services`` contract.

This test is skipped automatically in environments without docker (e.g. the
``unit`` CI job); it runs in the ``integration`` CI job (push-to-main /
nightly / PR with ``run-integration`` label).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("pytest_docker", reason="pytest-docker not installed")


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    """Point pytest-docker at the repo-root docker-compose.yaml."""
    return str(Path(__file__).resolve().parents[2] / "docker-compose.yaml")


def _is_responsive(url: str) -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def helix_api_url(docker_services, docker_ip):
    """Wait for helixflow-api to expose a healthy /helixflow/health."""
    port = docker_services.port_for("helixflow-api", 11110)
    url = f"http://{docker_ip}:{port}/helixflow/health"
    docker_services.wait_until_responsive(
        timeout=120.0,
        pause=1.0,
        check=lambda: _is_responsive(url),
    )
    return url


def test_docker_compose_health_200(helix_api_url):
    """GET /helixflow/health returns 200 after compose is up."""
    import urllib.request

    with urllib.request.urlopen(helix_api_url, timeout=5) as resp:
        assert resp.status == 200, f"expected 200 from {helix_api_url}; got {resp.status}"
