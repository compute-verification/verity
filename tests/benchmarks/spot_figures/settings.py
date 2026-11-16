"""The SPOT figures' tests' settings registry (`tests/lint/test_repo_env_reads.py`): each variable a test reads, declared once
here, where a test's new read goes."""

from __future__ import annotations

import os

SLOW = bool(os.environ.get("VERITY_SPOT_FIGURES_SLOW"))
"""`VERITY_SPOT_FIGURES_SLOW`: run the minutes-long tests, which skip otherwise (the testing skill, "Keep it fast")."""
