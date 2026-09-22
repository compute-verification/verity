"""Repository invariants (AGENTS.md "Working principles"): the tree holds executable code, concise operational guidance,
and small test fixtures.  Experimental evidence, run outputs, and model weights live in artifact storage, never in Git.

Enforced here rather than remembered: every tracked blob stays under ``BLOB_LIMIT`` bytes.  Directories that only ever
hold run outputs are ignored by ``.gitignore`` as well; this test is the backstop for anything that slips past it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOB_LIMIT = 5 * 1024 * 1024  # 5 MiB; the largest fixture today is 4.5 MB


def _tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    return [ROOT / p.decode() for p in out.split(b"\0") if p]


def test_no_tracked_blob_exceeds_limit() -> None:
    offenders = []
    for path in _tracked_files():
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:  # tracked but deleted in the working tree
            continue
        if size > BLOB_LIMIT:
            offenders.append(f"{path.relative_to(ROOT)}: {size / 1048576:.1f} MiB")
    assert not offenders, (
        "tracked files over the blob limit (evidence and run outputs belong in artifact storage, not Git):\n"
        + "\n".join(offenders)
    )
