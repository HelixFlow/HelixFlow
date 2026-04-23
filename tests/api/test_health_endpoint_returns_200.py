"""
Case #16 (补测) — `GET /helixflow/health` returns 200 at the FastAPI layer.

Owner: coder-2 (T15)
Rationale: #6 covers the same endpoint but through docker-compose; this variant
runs in the fast unit tier (no container) so we catch regressions during PR
review without waiting for the integration job.
"""

from __future__ import annotations


def test_health_endpoint_returns_200(client) -> None:
    """`GET /helixflow/health` returns HTTP 200 with the expected body."""
    response = client.get("/helixflow/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload == {"status": "OK"}
