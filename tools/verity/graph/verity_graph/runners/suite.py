"""suite: `tools/verity/check/suites.py` on the suite the task names (its var `suites`), as check's pytest step runs it: each
suite's timeout, and the task's cores and memory as `--jobs` and `--mem-gb`.  A task of a suite's part (its var `files`, K/M)
runs `--files K/M` with `--timings` a record that holds none of the suite's files, so suites.py splits it alike on every
machine.  `suites` runs it, and the lean-suites runner's suites with `--lean-built` after the shell prefix it gives.

Its status is from suites.py's exit and its report.  Every suite passed, a pass of its key reused among them, is a pass.  Suites
that failed on their tests, on reads outside their inputs or on their timeout are a fail, each named in `detail.failed` (its name,
directory and failed tests), by which drive names, allows and blames each as the suite's own red; so is a run whose tests changed
the repository.  A suite task's fail on nothing but the tests it names also lists them as `detail.failed_tests`, relative to its
var `suite`, as a test file's, by which the loop finds them among its base's.  A suite the report lacks or reports skipped (not
run), a failure the report gives no cause of (a kill, fixtures that couldn't be fetched), a usage error and a signal are an
error.  `detail.suites` is each suite's status."""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from .. import ROOT
from ..graph import LOG
from . import call

SUITES = "tools/verity/check/suites.py"
REPORT, LOGS, TIMINGS = "suites.json", "suites", "suites-timings.json"
#: suites.py's line when tests wrote into the repository, which fails its run with no suite failed
WROTE = "the tests changed"
#: the most failed tests a row names: a longer list is cut, and then failed for more than the tests it names
NAMED = 100
PASSED = ("passed", "cached")


def run(task: dict, work: Path) -> dict:
    res = suites(task, work, REPORT)
    if res["status"] == "fail" and len(rows := res["detail"].get("failed") or ()) == 1 and rows[0]["failed_tests"] and not (
            rows[0].get("beyond_tests") or rows[0].get("timed_out")):
        res["detail"]["failed_tests"] = rows[0]["failed_tests"]
    return res


def suites(task: dict, work: Path, report: str, lean: str | None = None) -> dict:
    """suites.py on ``task``'s suites, its report at ``report`` in ``work``; with ``lean``, a shell prefix that puts lake on the
    PATH, `--lean-built` after it, as check's lean-suites step runs them."""
    names = task["vars"]["suites"].split()
    r = task["resources"]
    part = []
    if files := task["vars"].get("files"):
        (work / TIMINGS).write_text(json.dumps({n: {} for n in names}))
        part = ["--files", files, "--timings", str(work / TIMINGS)]
    argv = [sys.executable, str(ROOT / SUITES), *names, *(["--lean-built"] if lean is not None else []), "--json",
            str(work / report), "--logs", str(work / LOGS), "--timeouts", "--jobs", str(max(1, int(r["cores"]))), "--mem-gb",
            str(r["mem_gb"]), *part]
    rc = call(argv if lean is None else ["bash", "-c", lean + shlex.join(argv)], work, ROOT)
    if rc < 0 or rc > 128:
        return {"status": "error", "detail": {"error": f"suites.py was killed (exit {rc})"}}
    if rc not in (0, 1) or not (work / report).is_file():
        return {"status": "error", "detail": {"error": f"suites.py exited {rc}" + ("" if (work / report).is_file() else
                                                                                   " and wrote no report")}}
    rows = json.loads((work / report).read_text())
    status = {r["name"]: r["status"] for r in rows}
    detail: dict = {"exit": rc, "suites": status}
    unrun = [n for n in names if not any(n in (r.get("name"), r.get("suite")) and r["status"] != "skipped" for r in rows)]
    if unrun:
        return {"status": "error", "detail": {**detail, "error": f"not run: {', '.join(unrun)}"}}
    failed = [r for r in rows if r["status"] not in PASSED]
    named = [r for r in failed if r.get("failed_tests") or r.get("outside") or r.get("child_outside") or r.get("timed_out")]
    if rc == 0 and not failed:
        return {"status": "pass", "detail": detail}
    if named:
        return {"status": "fail", "detail": {**detail, "failed": [row(r) for r in named]}}
    if rc == 1 and not failed and WROTE in (work / LOG).read_text(errors="replace"):
        return {"status": "fail", "detail": {**detail, "why": "its tests changed files in the repository"}}
    return {"status": "error", "detail": {**detail, "error": "suites.py failed on nothing its report names a cause of: "
                                                             + ", ".join(r["name"] for r in failed) if failed else
                                                             f"suites.py exited {rc} with every suite passed"}}


def row(r: dict) -> dict:
    """What drive reads of a failed suite's report row: its name and directory, its failed tests, whether it failed for more than
    them (`beyond_tests`, as suites.py gives it, and when the list is cut), and its timeout."""
    tests = [str(t) for t in r.get("failed_tests") or ()]
    out = {"name": r["name"], "suite": r.get("suite") or ".", "failed_tests": tests[:NAMED]}
    if len(tests) > NAMED:
        out["beyond_tests"] = True
    elif "beyond_tests" in r:
        out["beyond_tests"] = r["beyond_tests"]
    if r.get("timed_out"):
        out["timed_out"] = r["timed_out"]
    return out
