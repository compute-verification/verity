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


def placement_hold_record(env: Mapping[str, str] | None = None) -> str | None:
    """What a record and a VERDICT line say of a run that lifted the placement hold (``lifted (VERITY_LIFT_PLACEMENT_HOLD=1)``),
    None while it is held. The one place that text is made; ValueError while the retired switch is set."""
    return None if hold_placement(env) else f"lifted ({LIFT_PLACEMENT_HOLD}=1)"


#: what follows ``accept`` where a lifted run's accept is printed
LIFTED_NOTE = "placement hold lifted"


def printed_verdict(verdict: str, env: Mapping[str, str] | None = None) -> str:
    """The word a reader sees for ``verdict``: an accept while the placement hold is lifted reads ``accept (placement
    hold lifted)``; every other word, and every word of a held run, as it is. Printed text only: a JSON value code
    compares keeps its plain word. `benchmarks/one_stage/placement.sh`'s `printed_verdict_lines` is the shell's, held to
    this by a test."""
    return f"{verdict} ({LIFTED_NOTE})" if verdict == "accept" and not hold_placement(env) else verdict
