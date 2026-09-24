"""What counts as work on a pod: one definition for the watcher's pod probe (research.notes POD_PROBE) and the pod-side guard
(pods/sh/pod_guard.sh daemon, which `research launch-request` starts with `guard_env()` in its environment).

A pod is busy while any of these holds: a work process runs (a `ps -eo args=` line matching WORK_RE and not NOT_WORK_RE), a live
verifier serves (SERVE_RE, not an observer's own line), a GPU compute app exists, GPU utilisation is above 0.  NOT_WORK_RE names the
observers themselves (the probe's mark, the guard), shells that only wait (sleep) and the pod's own services (sshd, jupyter, nginx,
RunPod's start.sh).  How long an idle pod has been idle is the age of its newest file under /workspace, pruning the directory names
in PRUNE (venvs, caches, build trees and the guards' own state change without anyone working).
"""
from __future__ import annotations

import re

PROBE_MARK = "research-notes-probe"          # the watcher's probe script carries it (and the regexes below) in its own argv
GUARD_MARK = "research-pod-guard"            # the guard's helper processes carry it
MARKS_RE = rf"{PROBE_MARK}|{GUARD_MARK}|pod_guard"
WORK_RE = r"python|cargo|rustc|pytest|ligero|nvcc|\.sh( |$)"
NOT_WORK_RE = rf"grep|{MARKS_RE}|compile_worker|nginx|jupyter|sshd|start\.sh|live serve|sleep"
SERVE_RE = "live serve"
PRUNE = ("venv312", ".venv", "target", "__pycache__", ".cache", "node_modules", "guard")
FIND_PRUNE = r"\( " + " -o ".join(f"-name {n}" for n in PRUNE) + r" \) -prune"


def is_work(args: str) -> bool:
    return bool(re.search(WORK_RE, args)) and not re.search(NOT_WORK_RE, args)


def is_serve(args: str) -> bool:
    return SERVE_RE in args and not re.search(MARKS_RE, args)


def guard_env() -> dict[str, str]:
    """The definition as the guard reads it (grep -E patterns; PRUNE space-separated)."""
    return {"GUARD_WORK_RE": WORK_RE, "GUARD_NOT_WORK_RE": NOT_WORK_RE, "GUARD_SERVE_RE": SERVE_RE, "GUARD_MARKS_RE": MARKS_RE,
            "GUARD_PRUNE": " ".join(PRUNE)}
