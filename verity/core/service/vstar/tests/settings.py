"""The environment variables vstar's difftests read, each declared once."""

from __future__ import annotations

import os
from pathlib import Path


def recursive_data() -> str:
    """``VERITY_VSTAR_RECURSIVE_DATA``: a recursion_e2e run's kept data/TAG (KEEP_DATA=1, after STEP=lean), which the
    whole-session recursive tests read and skip without; else empty."""
    return os.environ.get("VERITY_VSTAR_RECURSIVE_DATA", "")


def lean_jobs() -> int:
    """``LEAN_JOBS``: how many flock-verify processes a whole-session recursive test runs at once (default 4)."""
    return int(os.environ.get("LEAN_JOBS") or 4)


def flock_trace() -> Path | None:
    """``FLOCK_TRACE``, a build of ``flock-trace``; else the ``flock-trace`` beside ``FLOCK_CIRCUIT``'s ``flock-circuit``."""
    if os.environ.get("FLOCK_TRACE"):
        return Path(os.environ["FLOCK_TRACE"])
    if os.environ.get("FLOCK_CIRCUIT"):
        return Path(os.environ["FLOCK_CIRCUIT"]).with_name("flock-trace")
    return None


def irdep_tree() -> Path | None:
    """``VERITY_IRDEP_TREE``, a deployed recursive run's tree (`stage/`, `inner/s0/`, `v/`)."""
    return Path(os.environ["VERITY_IRDEP_TREE"]) if os.environ.get("VERITY_IRDEP_TREE") else None


def irdep_lean() -> Path | None:
    """``VERITY_IRDEP_LEAN``, the output of `IrDepSession.lean` at that tree."""
    return Path(os.environ["VERITY_IRDEP_LEAN"]) if os.environ.get("VERITY_IRDEP_LEAN") else None
