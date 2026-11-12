"""PoUS's tests' settings registry (`tests/lint/test_repo_env_reads.py`): each variable a test reads, declared once here,
where a test's new read goes."""

from __future__ import annotations

import os

SLOW = bool(os.environ.get("VERITY_POUS_SLOW"))
"""`VERITY_POUS_SLOW`: run the minutes-long checks, which skip otherwise (the testing skill, "Keep it fast")."""
