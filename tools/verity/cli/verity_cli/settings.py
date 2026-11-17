"""The environment variables `verity run` reads, declared once (`tests/lint/test_repo_env_reads.py`: reads belong here)."""
from __future__ import annotations

import os


def run_id() -> str:
    """The `research run` this one is part of (``RESEARCH_RUN_ID``), which each verdict line names; ``local`` outside one."""
    return os.environ.get("RESEARCH_RUN_ID") or "local"


def hold_placement() -> bool:
    """Unless ``VERITY_HOLD_PLACEMENT=0``, the refusals resting only on where coins or composites are made hold."""
    return os.environ.get("VERITY_HOLD_PLACEMENT") != "0"
