"""lean-lock's runner: recompute the records of the guarantees a package's `lean-audit.json` lists from the modules it reads
(`families/lean_lock.py`), compare them with the committed lock, and fail a stale one with the command that regenerates
it. Until it is written every such task is an error, which no executor takes for a result."""
from __future__ import annotations

from pathlib import Path


def run(task: dict, work: Path) -> dict:
    return {"status": "error", "detail": {"error": "the lean-lock runner isn't written yet"}}
