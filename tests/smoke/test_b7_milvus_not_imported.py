"""
Case #17 (补测) — B7 regression guard.

Owner: coder-2 (T15)
Verifies that no runtime code path under `core/`, `router/`, `service/`, or
`utils/` imports `pymilvus` or `langchain_milvus`. These packages have been
moved to `requirements-optional.txt` (see 组会纪要 §3.2 B7); if a future PR
re-introduces the import the optional extras install would silently become
mandatory for anyone running the app.

Also asserts that `requirements.txt` carries the explanatory comment pointing
to `requirements-optional.txt`.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FORBIDDEN_MODULES = {"pymilvus", "langchain_milvus"}
_SCAN_DIRS = ("core", "router", "service", "utils")


def _iter_python_files():
    for top in _SCAN_DIRS:
        root = _PROJECT_ROOT / top
        if not root.exists():
            continue
        yield from root.rglob("*.py")


def _imported_modules(path: pathlib.Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # pragma: no cover — we want to know about this
        pytest.fail(f"Failed to parse {path}: {exc}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module.split(".")[0]


def test_b7_milvus_not_imported() -> None:
    """No module under core/ router/ service/ utils/ imports pymilvus / langchain_milvus."""
    offenders: list[str] = []
    for path in _iter_python_files():
        for mod in _imported_modules(path):
            if mod in _FORBIDDEN_MODULES:
                offenders.append(f"{path.relative_to(_PROJECT_ROOT)} imports {mod!r}")
    assert offenders == [], (
        "B7 regression: pymilvus / langchain_milvus must not be imported from "
        "runtime code. Offenders:\n  " + "\n  ".join(offenders)
    )


def test_requirements_txt_points_at_optional() -> None:
    """`requirements.txt` must advertise `requirements-optional.txt` for retrieval deps."""
    req = (_PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "requirements-optional.txt" in req, (
        "requirements.txt must reference requirements-optional.txt so users " "know where retrieval deps moved (B7)."
    )
    # And the file must NOT list pymilvus / langchain-milvus directly any more.
    for forbidden in ("pymilvus==", "langchain-milvus=="):
        assert (
            forbidden not in req
        ), f"requirements.txt still pins {forbidden!r} — should live in requirements-optional.txt"


def test_optional_requirements_lists_retrieval_deps() -> None:
    """`requirements-optional.txt` must carry pymilvus + langchain-milvus."""
    path = _PROJECT_ROOT / "requirements-optional.txt"
    assert path.exists(), "requirements-optional.txt must exist for B7"
    content = path.read_text(encoding="utf-8").lower()
    assert "pymilvus" in content
    assert "langchain-milvus" in content
