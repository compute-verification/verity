"""The environment variables `verity run` reads, declared once (`tests/lint/test_repo_env_reads.py`: reads belong here)."""
from __future__ import annotations

import os


def run_id() -> str:
    """The `research run` this one is part of (``RESEARCH_RUN_ID``), which each verdict line names; ``local`` outside one."""
    return os.environ.get("RESEARCH_RUN_ID") or "local"


#: the setting that restores this week's lifted placement holds (``1``)
HOLD_PLACEMENT = "VERITY_HOLD_PLACEMENT"


def hold_placement() -> bool:
    """Whether the placement holds stand (``VERITY_HOLD_PLACEMENT=1``): `served_table.onsite_check`'s on a record the
    workload's node composed. Unset, they are lifted."""
    return os.environ.get(HOLD_PLACEMENT) == "1"
