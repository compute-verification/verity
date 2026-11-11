"""verity/protocols/spot/verifier/lean's tests' settings registry (`tests/lint/test_repo_env_reads.py`): each variable a test
reads, declared once here, where a test's new read goes."""

from __future__ import annotations

import os

REAL_TEMPLATES = bool(os.environ.get("VERITY_SPOT_REAL_TEMPLATES"))
"""`VERITY_SPOT_REAL_TEMPLATES`: hold the `unit_gates` piece to its walk on a served window's templates too
(test_difftest's `REAL_TEMPLATES`); without it those tests skip."""
