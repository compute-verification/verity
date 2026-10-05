"""Disk admission for a queued run (`research run --queue --disk-gb N`), on the machine, ahead of the run's allocation.

`research disk-hold --gb N --path P -- CMD...` admits CMD at once when the filesystem holding P would stay at or under HOLD_PCT
(80) of its size with N more GB used, and otherwise holds it until that falls to RELEASE_PCT (75), the thresholds node 1's
`vy-disk-guard` uses; then it execs CMD in its own process, so the run's pid is unchanged. The launch doesn't wait: the hold
happens in the detached process the machine starts, and its lines go to the run's launcher.log. Requests other runs have
declared but not yet used aren't counted (a follow-up). A machine whose guard holds at another level sets
RESEARCH_DISK_HOLD_PCT in the run's environment.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

HOLD_PCT = 80
RELEASE_PCT = 75
POLL_S = 30
GB = 1e9


def hold(gb: float, path: Path, *, hold_pct: float = HOLD_PCT, release_pct: float = RELEASE_PCT, poll_s: float = POLL_S,
         usage: Callable[[Path], tuple[int, int, int]] = shutil.disk_usage, sleep: Callable[[float], None] = time.sleep,
         log: Callable[[str], None] = lambda s: print(s, file=sys.stderr, flush=True)) -> int:
    """Return once `gb` more GB fits on `path`'s filesystem by the rule above; the number of polls it was held for."""
    held = 0
    while True:
        total, used, _ = usage(path)
        after = 100 * (used + gb * GB) / total
        if after <= (release_pct if held else hold_pct):
            if held:
                log(f"research disk-hold: released after {held} poll(s): {after:.1f}% with {gb:g} GB more")
            return held
        if not held:
            log(f"research disk-hold: holding: {gb:g} GB more would fill {path}'s filesystem to {after:.1f}% (over {hold_pct:g}%); "
                f"released at {release_pct:g}% or less, polled every {poll_s:g} s")
        held += 1
        sleep(poll_s)


def main(argv: list[str], *, usage: Callable[[Path], tuple[int, int, int]] = shutil.disk_usage,
         sleep: Callable[[float], None] = time.sleep, execvp: Callable[[str, list[str]], None] = os.execvp) -> int:
    ap = argparse.ArgumentParser(prog="research disk-hold")
    ap.add_argument("--gb", type=float, required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    ns = ap.parse_args(argv)
    cmd = ns.cmd[1:] if ns.cmd[:1] == ["--"] else ns.cmd
    if not cmd:
        print("research disk-hold: give the command after --", file=sys.stderr)
        return 2
    pct = float(os.environ.get("RESEARCH_DISK_HOLD_PCT", HOLD_PCT))
    hold(ns.gb, Path(ns.path), hold_pct=pct, release_pct=min(RELEASE_PCT, pct), usage=usage, sleep=sleep)
    execvp(cmd[0], cmd)
    return 127


__all__ = ["HOLD_PCT", "RELEASE_PCT", "hold", "main"]
