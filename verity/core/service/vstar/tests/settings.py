"""The environment variables vstar's difftests read, each declared once."""

from __future__ import annotations

import os
from pathlib import Path

#: a recursion_e2e run's kept data/TAG (KEEP_DATA=1, after STEP=lean); empty, the deployed recursive tests skip
VSTAR_RECURSIVE_DATA = os.environ.get("VERITY_VSTAR_RECURSIVE_DATA", "")
#: how many flock-verify processes the deployed recursive difftest runs at once
LEAN_JOBS = int(os.environ.get("LEAN_JOBS", "4"))


def flock_trace() -> Path | None:
    """``FLOCK_TRACE``, a build of ``flock-trace``; else the ``flock-trace`` beside ``FLOCK_CIRCUIT``'s ``flock-circuit``."""
    if os.environ.get("FLOCK_TRACE"):
        return Path(os.environ["FLOCK_TRACE"])
    if os.environ.get("FLOCK_CIRCUIT"):
        return Path(os.environ["FLOCK_CIRCUIT"]).with_name("flock-trace")
    return None
