"""The environment variables vstar's difftests read, each declared once."""

from __future__ import annotations

import os
from pathlib import Path


def flock_trace() -> Path | None:
    """``FLOCK_TRACE``, a build of ``flock-trace``; else the ``flock-trace`` beside ``FLOCK_CIRCUIT``'s ``flock-circuit``."""
    if os.environ.get("FLOCK_TRACE"):
        return Path(os.environ["FLOCK_TRACE"])
    if os.environ.get("FLOCK_CIRCUIT"):
        return Path(os.environ["FLOCK_CIRCUIT"]).with_name("flock-trace")
    return None
