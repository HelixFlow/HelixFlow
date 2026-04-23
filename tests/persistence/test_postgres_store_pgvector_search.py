"""
Case #14 — LangGraph Store + pgvector similarity search.

Owner: coder-2 (T14)
Status: skeleton — enabled once P0'-b installs `langgraph-store-postgres` +
pgvector extension in the PG image.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "langgraph.store.postgres",
    reason="LangGraph Store + pgvector is a P0'-b deliverable",
)


pytestmark = pytest.mark.persistence


def test_postgres_store_pgvector_search() -> None:
    """Upsert 3 namespace-scoped vectors, search top_k=2 by cosine similarity."""
    # Real body once P0'-b lands:
    #   1. store.put(('ns',), 'k1', {'text': 'hello'}, embedding=[...])
    #   2. store.put(('ns',), 'k2', {'text': 'world'}, embedding=[...])
    #   3. hits = store.search(('ns',), query_embedding=[...], limit=2)
    #   4. assert ordered by distance ASC
    raise AssertionError("Store pgvector search body is P0'-b Sprint 2")
