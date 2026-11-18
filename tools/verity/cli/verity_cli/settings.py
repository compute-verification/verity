"""The environment variables `verity run` reads, declared once (`tests/lint/test_repo_env_reads.py`: reads belong here)."""
from __future__ import annotations

import os


def run_id() -> str:
    """The `research run` this one is part of (``RESEARCH_RUN_ID``), which each verdict line names; ``local`` outside one."""
    return os.environ.get("RESEARCH_RUN_ID") or "local"


def hold_placement() -> bool:
    """``VERITY_HOLD_PLACEMENT=1``: the refusals that rest only on where coins are flipped or a composite made hold."""
    return os.environ.get("VERITY_HOLD_PLACEMENT") == "1"


#: the runner's variable that carries the application file's SPOT levels to it
SPOT_LEVELS = "SPOT_LEVELS"


def spot_levels() -> str | None:
    """`SPOT_LEVELS` as this environment sets it, ``None`` where unset: it only carries the application file's levels to
    the runner, so `served_table.levels` refuses one that differs."""
    return os.environ.get(SPOT_LEVELS) or None
