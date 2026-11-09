"""The cap on a wait an agent's tool call makes: no tool call may block more than 20 minutes, since an agent that
blocks in one long call looks silent and misses its wake-ups.  A wait that reaches the cap stops waiting, leaves what it waited
on going (nothing is cancelled), says on stderr how to check back, and exits STILL_GOING: `research run --wait`, `research msg
ask-daniel ... --wait` and `research queue ready --on`."""
from __future__ import annotations

WAIT_CAP_S = 1200
#: the exit of a wait that reached the cap with its subject still going.  Not sysexits' 75 (EX_TEMPFAIL): slot.py and gpu-lease
#: exit 75 when their command never started, and `run --wait` exits with the run's own rc
STILL_GOING = 79
RUN_END = "End your turn now; the run's end will ring your lane."


class OverCap(ValueError):
    pass


def max_s(given: float | None, flag: str = "--wait-max") -> float:
    """The wait's bound in seconds: `given` (a `--wait-max S`), which may lower the cap and never raise it, else the cap."""
    if given is None:
        return WAIT_CAP_S
    if not 0 < given <= WAIT_CAP_S:
        raise OverCap(f"{flag} {given:g}: a wait may last 0 to {WAIT_CAP_S} s (WAIT_CAP_S, 20 minutes: no tool call may block "
                      f"longer); drop it to wait the whole {WAIT_CAP_S} s")
    return given


def still_going(what: str, check: str, waited_s: float, end: str = RUN_END) -> str:
    """The line a wait prints when it stops at its bound: what is still going, the command that reads it later, and `end`."""
    return (f"research: {what} is still going; stopped waiting after {waited_s:g} s (the cap is {WAIT_CAP_S} s: no tool call may "
            f"block more than 20 minutes). Nothing was cancelled. Check back with `{check}`. {end}")
