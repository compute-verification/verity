"""The environment variables `verity run` reads, declared once (`tests/lint/test_repo_env_reads.py`: reads belong here)."""
from __future__ import annotations

import os
from collections.abc import Mapping


def run_id() -> str:
    """The `research run` this one is part of (``RESEARCH_RUN_ID``), which each verdict line names; ``local`` outside one."""
    return os.environ.get("RESEARCH_RUN_ID") or "local"


def hold_placement(env: Mapping[str, str] | None = None) -> bool:
    """``VERITY_HOLD_PLACEMENT=1`` in ``env`` (this process's environment by default): the refusals that rest only on where
    coins are flipped or a composite made hold. Unset, empty or any other value, they are lifted. The one reader:
    `benchmarks/one_stage/placement.sh`'s `hold_placement` is the shell's, held to this one by a test."""
    return (os.environ if env is None else env).get("VERITY_HOLD_PLACEMENT") == "1"
