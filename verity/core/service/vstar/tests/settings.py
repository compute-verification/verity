"""V*'s tests' settings registry (`tests/lint/test_repo_env_reads.py`): each variable a test reads, declared once here,
where a test's new read goes."""

from __future__ import annotations

import os

RECURSIVE_DATA = os.environ.get("VERITY_VSTAR_RECURSIVE_DATA", "")
"""`VERITY_VSTAR_RECURSIVE_DATA`: a recursion_e2e run's kept data/TAG (KEEP_DATA=1, after STEP=lean), which the whole-session
tests read; they skip without it."""

LEAN_JOBS = int(os.environ.get("LEAN_JOBS", "4"))
"""`LEAN_JOBS`: how many flock-verify processes a whole-session test runs at once."""
