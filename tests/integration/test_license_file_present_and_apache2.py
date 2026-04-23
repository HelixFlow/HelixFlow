"""Sanity check: LICENSE file is checked in and contains Apache 2.0 full text.

Runs as "integration" only for tagging purposes (no docker required). Kept in
``tests/integration/`` to match the RFC §8 DB-2 contract path.
"""

from __future__ import annotations

from pathlib import Path


def test_license_file_present_and_apache2():
    """Four checks:

    1. LICENSE file exists at repo root.
    2. Contains the "Apache License" banner.
    3. Contains "Version 2.0".
    4. Is the full standard text (> 100 lines).
    """
    repo_root = Path(__file__).resolve().parents[2]
    license_path = repo_root / "LICENSE"

    assert license_path.is_file(), f"LICENSE must exist at {license_path}"

    text = license_path.read_text(encoding="utf-8")
    assert "Apache License" in text, "LICENSE must contain 'Apache License' banner"
    assert "Version 2.0" in text, "LICENSE must reference 'Version 2.0'"

    num_lines = len(text.splitlines())
    assert num_lines > 100, (
        f"LICENSE must contain the full Apache-2.0 text (>100 lines); got {num_lines}"
    )
