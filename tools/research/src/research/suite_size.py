"""`research run --on auto`'s size for a suites.py run, from what each suite it runs declares in its pyproject.toml's
`[tool.verity.tests]`: `worker_gb`, one xdist worker's peak, and `timeout_s`, the bound `--timeouts` puts on its run.

The run's memory is its test processes at the largest worker_gb, one per CPU (suites.py's `--jobs` is the CPUs it may use) or
its `--jobs`; a run of many suites at no less than TEST_PROCESS_GB each. That is the budget suites.py divides among its test
processes (its `--mem-gb`: `serial_slots` and `wide_workers` start no more than it holds at each one's figure), so a run stays
inside it unless a process passes its own figure, and a larger one buys processes beyond the CPUs asked. A run of named suites
reserves it whole. A run of many suites has processes at their figure only while the few suites that reach it run, and in a
pool's slot the reservation caps nothing and is held against the node, whether or not the run reaches it, for as long as the run
holds the slot: there it reserves MANY_RESERVED of the budget (`reserve_gb`) and plans its processes on all of it (slot.py's
BUDGET_OPT). A run
past its reservation meets the node's own limits, as one past its budget does, and where the node orders its OOM kills by
reservation (pods/nebius/oom_order.py) it is picked before a run within its own. Its time, for named suites, is their timeouts one after another (an undeclared one's SUITE_TIMEOUT_S) plus SYNC_S
for the environment's `uv sync`. A run of many suites, `--changed REV` (the suites the change touches, as suites.py's `--list`
names them) or none named (all of them; a change that deletes a pyproject.toml reruns every suite), runs them side by side
on its jobs: its time is the list-scheduling bound, their times summed over the jobs plus the longest, with an undeclared suite at
UNDECLARED_S, since at SUITE_TIMEOUT_S each the tree's 79 would ask for a day. Neither drops below the kind's own (AUTO_*). Only
what a suite declares is used, so named suites that declare neither keep the kind's size, as does a name that matches none or
several (suites.py refuses it). Reads the checkout's tracked pyproject.toml files, and for `--changed` runs its suites.py
`--list`; imports nothing from Verity.
"""
from __future__ import annotations

import math
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

SUITES_PY = "tools/verity/check/suites.py"
#: slot.py's option that gives a run its budget where it reserves less: the launcher sends it only to a tree whose slot.py
#: names it (one from before the option refuses it, so that tree's run reserves its budget whole)
SLOT_PY, BUDGET_OPT = "tools/verity/check/slot.py", "--budget-gb"
#: suites.py's options that take a value
VALUED = {"--changed", "--skip", "--jobs", "-j", "--mem-gb", "--logs", "--json", "--run", "--per-test-min-s", "--explain"}
#: suites.py's SUITE_TIMEOUT_S: an undeclared suite's bound
SUITE_TIMEOUT_S = 3600
SYNC_S = 600
#: an undeclared suite's time in a run of many: a node's suites.py durations.json, 32 suites run whole, had a median of 45 s
#: and a mean of 181 s; the slow ones (circuit-check's 63 min) declare their timeout_s
UNDECLARED_S = 300
#: suites.py's TEST_PROCESS_GB: what a test process of a suite that declares no worker_gb may reach
TEST_PROCESS_GB = 4.5
#: the share of its budget a run of many suites reserves in a pool's slot (the module docstring)
MANY_RESERVED = 0.5
#: kind agent-test's shape (infra/cluster/src/cluster/kinds.py, phase cpu-test, and kinds/infra.toml's max_wall_min)
AUTO_CPUS, AUTO_MEM_GB, AUTO_MAX_S = 4, 16, 3600


def _words(command: list[str]) -> list[str]:
    """``command``'s words, a shell string that runs suites.py split into its own (`bash -lc "uv sync && uv run
    tools/verity/check/suites.py --changed REV"`, the lane run `research queue` names), so that run is sized as well."""
    out: list[str] = []
    for a in command:
        if SUITES_PY in a and a != SUITES_PY and not a.endswith("/" + SUITES_PY):
            try:
                out += [w for w in shlex.split(a) if w not in ("&&", "||", ";")]
                continue
            except ValueError as e:
                print(f"suite_size: {a[:120]!r} doesn't split as a shell string ({e}); it is sized by its words as given",
                      file=sys.stderr)
        out.append(a)
    return out


def selection(command: list[str]) -> tuple[list[str], int | None, str | None] | None:
    """The suites a suites.py command names, its `--jobs` and its `--changed` REV, or None when it isn't one."""
    command = _words(command)
    at = next((i for i, a in enumerate(command) if a == SUITES_PY or a.endswith("/" + SUITES_PY)), None)
    if at is None:
        return None
    names, jobs, rev, rest = [], None, None, iter(command[at + 1:])
    for a in rest:
        if a == "--":
            break
        if a.startswith("--changed="):
            rev = a.split("=", 1)[1]
        elif a in VALUED:
            v = next(rest, "")
            jobs = int(v) if a in ("--jobs", "-j") and v.isdigit() else jobs
            rev = v if a == "--changed" else rev
        elif not a.startswith("-"):
            names.append(a)
    return names, jobs, rev


def touched(source: Path, names: list[str], rev: str) -> list[str] | None:
    """The suites (of `names`, or all) the change from `rev` touches, as the checkout's suites.py says; None where it can't say."""
    try:
        r = subprocess.run([sys.executable, SUITES_PY, *names, "--changed", rev, "--list"], cwd=source, capture_output=True, text=True,
                           timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = next((x for x in r.stdout.splitlines() if x.startswith("suites: the change from ") and " touches " in x), None)
    if r.returncode != 0 or line is None:
        return None
    listed = line.split(": ", 2)[2] if line.count(": ") >= 2 else ""
    return [] if listed == "none" else [n.strip() for n in listed.split(",") if n.strip()]


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


def _takes_budget(source: Path) -> bool:
    try:
        return BUDGET_OPT in (source / SLOT_PY).read_text()
    except (OSError, UnicodeDecodeError):
        return False


def size(source: Path, command: list[str], cpus: int | None) -> dict | None:
    """`{suites, mem_gb?, reserve_gb?, max_s?}` for a suites.py run whose suites' declarations, or number, say more than the
    kind's size: `mem_gb` the budget its processes are counted on, `reserve_gb` what a run of many suites reserves of it in a
    pool's slot (never under the kind's own memory; none where the tree's slot.py takes no BUDGET_OPT)."""
    if not (got := selection(command)):
        return None
    names, jobs, rev = got
    suites = declared(source)
    if rev is not None:
        hit = touched(source, names, rev)
        chosen = None if hit is None else [s for s in suites if s["name"] in hit or s["path"] in hit]
    else:
        chosen = pick(suites, names) if names else suites
    if not chosen:
        return None
    out: dict = {"suites": [s["name"] for s in chosen]}
    workers = jobs or cpus or AUTO_CPUS
    many = rev is not None or not names
    if many:
        out["mem_gb"] = max(AUTO_MEM_GB, math.ceil(workers * max(TEST_PROCESS_GB, *(s["worker_gb"] for s in chosen))))
        if _takes_budget(source):
            out["reserve_gb"] = max(AUTO_MEM_GB, math.ceil(out["mem_gb"] * MANY_RESERVED))
    elif gb := max(s["worker_gb"] for s in chosen):
        out["mem_gb"] = max(AUTO_MEM_GB, math.ceil(workers * gb))
    if many:
        ts = [s["timeout_s"] or UNDECLARED_S for s in chosen]
        if (t := math.ceil(sum(ts) / workers + max(ts)) + SYNC_S) > AUTO_MAX_S:
            out["max_s"] = t
    elif any(s["timeout_s"] for s in chosen):
        out["max_s"] = max(AUTO_MAX_S, int(sum(s["timeout_s"] or SUITE_TIMEOUT_S for s in chosen)) + SYNC_S)
    return out if len(out) > 1 else None


__all__ = ["selection", "touched", "declared", "pick", "size"]
