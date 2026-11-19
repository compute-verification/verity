"""The environment variables `verity run` reads, declared once (`tests/lint/test_repo_env_reads.py`: reads belong here)."""
from __future__ import annotations

import os
from collections.abc import Mapping


def run_id() -> str:
    """The `research run` this one is part of (``RESEARCH_RUN_ID``), which each verdict line names; ``local`` outside one."""
    return os.environ.get("RESEARCH_RUN_ID") or "local"


#: ``1`` lifts the refusals that rest only on where coins are flipped or a composite made; a lifted run's record says so
LIFT_PLACEMENT_HOLD = "VERITY_LIFT_PLACEMENT_HOLD"
#: the hold's old switch, whose ``1`` held: refused while set, since unset now holds
RETIRED_HOLD = "VERITY_HOLD_PLACEMENT"


def hold_placement(env: Mapping[str, str] | None = None) -> bool:
    """Whether the refusals that rest only on where coins are flipped or a composite made hold, in ``env`` (this process's
    environment by default): they do unless ``VERITY_LIFT_PLACEMENT_HOLD=1``. ValueError while the retired
    ``VERITY_HOLD_PLACEMENT`` is set. The one reader: `benchmarks/one_stage/placement.sh`'s `hold_placement` is the
    shell's, held to this one by a test."""
    env = os.environ if env is None else env
    if RETIRED_HOLD in env:
        raise ValueError(f"{RETIRED_HOLD} is retired: the hold is the default, and {LIFT_PLACEMENT_HOLD}=1 lifts it; "
                         f"unset {RETIRED_HOLD}")
    return env.get(LIFT_PLACEMENT_HOLD) != "1"
