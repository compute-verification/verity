"""`research run --on auto`'s size for a suites.py run of named suites, from what each declares in its pyproject.toml's
`[tool.verity.tests]`: `worker_gb`, one xdist worker's peak, and `timeout_s`, the bound `--timeouts` puts on its run.

The run's memory is its test processes at the largest worker_gb, one per CPU (suites.py's `--jobs` is the CPUs it may use) or
its `--jobs`; its time is the suites' timeouts one after another (an undeclared one's SUITE_TIMEOUT_S) plus SYNC_S for the
environment's `uv sync`. Neither drops below the kind's own (AUTO_*). Only what a suite declares is used, so a run of suites
that declare neither keeps the kind's size, as does a run that names no suite (all of them) or a name that matches none or
several (suites.py refuses it). Reads only the checkout's tracked pyproject.toml files; imports nothing from Verity.
"""
from __future__ import annotations

import math
import re
import subprocess
import tomllib
from pathlib import Path

SUITES_PY = "tools/verity/check/suites.py"
#: suites.py's options that take a value
VALUED = {"--changed", "--skip", "--jobs", "-j", "--mem-gb", "--logs", "--json", "--run", "--per-test-min-s", "--explain"}
#: suites.py's SUITE_TIMEOUT_S: an undeclared suite's bound
SUITE_TIMEOUT_S = 3600
SYNC_S = 600
#: kind agent-test's shape (infra/cluster/src/cluster/kinds.py, phase cpu-test, and kinds/infra.toml's max_wall_min)
AUTO_CPUS, AUTO_MEM_GB, AUTO_MAX_S = 4, 16, 3600


def named(command: list[str]) -> tuple[list[str], int | None] | None:
    """The suites a suites.py command names and its `--jobs`, or None when it isn't one or names none."""
    at = next((i for i, a in enumerate(command) if a == SUITES_PY or a.endswith("/" + SUITES_PY)), None)
    if at is None:
        return None
    names, jobs, rest = [], None, iter(command[at + 1:])
    for a in rest:
        if a == "--":
            break
        if a in VALUED:
            v = next(rest, "")
            jobs = int(v) if a in ("--jobs", "-j") and v.isdigit() else jobs
        elif not a.startswith("-"):
            names.append(a)
    return (names, jobs) if names else None


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def declared(source: Path) -> list[dict]:
    """Each suite of the checkout (a pyproject.toml with [tool.pytest], named as suites.py names it) and what it declares."""
    files = subprocess.run(["git", "-C", str(source), "ls-files", "-z", "--", "pyproject.toml", "*/pyproject.toml"],
                           capture_output=True, text=True).stdout.split("\0")
    out = []
    for f in filter(None, files):
        try:
            data = tomllib.loads((source / f).read_text())
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if "pytest" not in data.get("tool", {}):
            continue
        path = str(Path(f).parent)
        project = data.get("project", {})
        tests = data["tool"].get("verity", {}).get("tests", {})
        out.append({"name": "repository" if path == "." else _norm(project["name"]) if project.get("name") else path, "path": path,
                    "worker_gb": float(tests.get("worker_gb", 0)), "timeout_s": float(tests.get("timeout_s", 0))})
    return out


def pick(suites: list[dict], names: list[str]) -> list[dict] | None:
    """suites.py's `select`: each name exactly a suite's name or path, else in exactly one; None when one doesn't resolve."""
    chosen = []
    for n in names:
        hits = [s for s in suites if n in (s["name"], s["path"], s["path"].rstrip("/"))] or \
               [s for s in suites if n in s["name"] or n in s["path"]]
        if len(hits) != 1:
            return None
        chosen += hits
    return list({s["path"]: s for s in chosen}.values())


def size(source: Path, command: list[str], cpus: int | None) -> dict | None:
    """`{suites, mem_gb?, max_s?}` for a suites.py run of named suites whose declarations say more than the kind's size."""
    got = named(command)
    chosen = pick(declared(source), got[0]) if got else None
    if not chosen:
        return None
    out: dict = {"suites": [s["name"] for s in chosen]}
    if gb := max(s["worker_gb"] for s in chosen):
        out["mem_gb"] = max(AUTO_MEM_GB, math.ceil((got[1] or cpus or AUTO_CPUS) * gb))
    if any(s["timeout_s"] for s in chosen):
        out["max_s"] = max(AUTO_MAX_S, int(sum(s["timeout_s"] or SUITE_TIMEOUT_S for s in chosen)) + SYNC_S)
    return out if len(out) > 1 else None


__all__ = ["named", "declared", "pick", "size"]
