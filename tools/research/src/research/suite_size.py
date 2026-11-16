"""`research run --on auto`'s size for a suites.py run, from each suite's `[tool.verity.tests]`."""
from __future__ import annotations

import math
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

from . import suite_layout

SUITES_PY = "tools/check/suites.py"
SLOT_PY, BUDGET_OPT = "tools/check/slot.py", "--budget-gb"
VALUED = {"--changed", "--skip", "--jobs", "-j", "--mem-gb", "--logs", "--json", "--run", "--per-test-min-s", "--explain"}
SUITE_TIMEOUT_S = 3600
SYNC_S = 600
UNDECLARED_S = 300
TEST_PROCESS_GB = 4.5
MANY_RESERVED = 0.5
AUTO_CPUS, AUTO_MEM_GB, AUTO_MAX_S = 4, 16, 3600


def _words(command: list[str]) -> list[str]:
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
    from . import launcher

    if launcher.tree(source):
        print(f"suite_size: the suites a --changed run touches are not listed under ${launcher.ENV} (the source's suites.py "
              "doesn't run here), so it takes the kind's size unless it gives --mem-gb and --timeout", file=sys.stderr, flush=True)
        return None
    try:
        r = subprocess.run([sys.executable, SUITES_PY, *names, "--changed", rev, "--list"], cwd=source, capture_output=True, text=True,
                           timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"suite_size: suites.py --list did not run, so the suites touched are not known ({type(e).__name__})", file=sys.stderr, flush=True)
        return None
    line = next((x for x in r.stdout.splitlines() if x.startswith("suites: the change from ") and " touches " in x), None)
    if r.returncode != 0 or line is None:
        return None
    listed = line.split(": ", 2)[2] if line.count(": ") >= 2 else ""
    return [] if listed == "none" else [n.strip() for n in listed.split(",") if n.strip()]


def declared(source: Path) -> list[dict]:
    from . import launcher

    files = subprocess.run(launcher.git("-C", str(source), "ls-files", "-z", "--", "pyproject.toml", "*/pyproject.toml"),
                           capture_output=True, text=True, env=launcher.git_env()).stdout.split("\0")
    read = {}
    for f in filter(None, files):
        try:
            read[str(Path(f).parent)] = tomllib.loads((source / f).read_text())
        except (OSError, tomllib.TOMLDecodeError) as e:
            print(f"suite_size: {f} unread, skipped ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
    out, dirs = [], suite_layout.member_dirs(lambda d: read.get(d, {}))
    for path, data in read.items():
        if "pytest" not in data.get("tool", {}):
            continue
        tests = data["tool"].get("verity", {}).get("tests", {})
        out.append({"name": suite_layout.suite_name(path, data.get("project", {}), dirs), "path": path,
                    "worker_gb": float(tests.get("worker_gb", 0)), "timeout_s": float(tests.get("timeout_s", 0)),
                    "timeout_workers": int(tests.get("timeout_workers", 0))})
    return out


def _bound(s: dict, workers: int, undeclared: float) -> float:
    return (s["timeout_s"] or undeclared) * max(1.0, s["timeout_workers"] / max(1, workers))


def pick(suites: list[dict], names: list[str]) -> list[dict] | None:
    chosen = []
    for n in names:
        hits = [s for s in suites if n in suite_layout.names(s["path"], s["name"])] or \
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
        ts = [_bound(s, workers, UNDECLARED_S) for s in chosen]
        if (t := math.ceil(sum(ts) / workers + max(ts)) + SYNC_S) > AUTO_MAX_S:
            out["max_s"] = t
    elif any(s["timeout_s"] for s in chosen):
        out["max_s"] = max(AUTO_MAX_S, int(sum(_bound(s, workers, SUITE_TIMEOUT_S) for s in chosen)) + SYNC_S)
    return out if len(out) > 1 else None


__all__ = ["selection", "touched", "declared", "pick", "size"]
