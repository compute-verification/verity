"""Flock's tests' settings registry (`tests/lint/test_repo_env_reads.py`): each variable a test reads, declared once here,
where a test's new read goes."""

from __future__ import annotations

import os
from pathlib import Path


def flock_circuit() -> Path | None:
    """``FLOCK_CIRCUIT``: a proving build of `flock-circuit`, which `test_stream_zk_server`'s sessions run and skip without;
    else None."""
    return Path(p) if (p := os.environ.get("FLOCK_CIRCUIT")) and Path(p).is_file() else None


def served_reads() -> Path | None:
    """``FLOCK_READ_SERVED``: a directory holding the served reads' circuit files under the names
    ``fixtures/served_reads.json`` gives them, which `test_read_tmpl`'s served-file tests check and skip without; else
    None."""
    return Path(p) if (p := os.environ.get("FLOCK_READ_SERVED")) else None
