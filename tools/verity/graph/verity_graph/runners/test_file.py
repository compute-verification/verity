"""A test-file task (`families/test_file.py`): pytest on its one test file as `tools/verity/check/suites.py` runs the file's suite,
from the suite's directory (the task's `vars` `suite`) with the suite's configuration, the read guard (`-p verity_suite_guard`),
no cache provider and a fresh `--basetemp`, in one process (`-p no:xdist`), after fetching the registered fixtures `suites.py`
would fetch for the suite.

The guard runs with no inputs declared to it, so it records every file of the tree the run opens, the test process's (a cached
module as its source) and the Pythons its tests start; the runner holds them to the task's inputs. A read of none of them, of no
file its env's spec names and, when the task reads the fixture registry, of no registered fixture is an undeclared read, which
makes the run an error. A directory opened is no read, nor is a name that is no file of the tree, nor a file under the work
directory. A file git ignores that no fixture is (a build's lock or output, under `.lake`) is none the graph lists, so no
input a key could hold: it is `detail.ignored`. Programs that aren't Python, or a Python that doesn't trace itself, read what no
tracer sees: they are `detail.untraced`, and an error unless the task is keyed on its suite's whole tree (`vars` `untraced` =
`suite`: its suite's `[tool.verity.tests] untraced` names the file), or they are compilers (`COMPILERS`) and its closure builds
native code (`compilers`).

Its status: pytest's exit 0, or 5 when the file holds no test to run, is a pass; 1 (a test failed) and 2 (collecting failed, or
a `pytest.exit`) are a fail; death by a signal, any other exit, no guard report, a Python started that couldn't trace itself, an
untraced program in a task not keyed on its suite's tree, and fixtures that couldn't be fetched are an error. `detail` holds the guard's counts (passed, failed, skipped and errors, a
setup's or teardown's), pytest's exit, the failed tests' node ids and the untraced programs."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from .. import ROOT, graph
from . import call

GUARD = ROOT / "tools" / "verity" / "check" / "guard"
#: on the tests' PYTHONPATH: its `sitecustomize` makes every Python a test starts trace what it reads
CHILD = GUARD / "child"
REGISTRY = "fixtures/artifacts.json"
#: pytest with the guard's directory on `sys.path` (`suites._PYTEST`)
PYTEST = "import sys, pytest; sys.path.append(sys.argv.pop(1)); sys.exit(pytest.main(sys.argv[1:]))"
#: what a test never inherits: a machine runner's `RESEARCH_*` and PYTHONPYCACHEPREFIX (`suites.RUNNER_PREFIX`, `RUNNER_ENV`), and
#: the guard's own variables, which a suite this runs inside of sets
DROP_PREFIXES, DROP = ("RESEARCH_", "VERITY_SUITE_"), ("PYTHONPYCACHEPREFIX",)
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
#: pytest's exits that are results: every test passed, or none was collected; a test failed, or collecting stopped
PASS, FAIL = (0, 5), (1, 2)
REPORT, READS = "guard.json", "guard.reads"
#: what a task whose closure builds native code may start (`vars` `untraced` = `compilers`): it reads the sources beside the module
COMPILERS = frozenset({"g++", "gcc", "c++", "cc", "clang", "clang++", "nvcc", "ninja"})

_spec = importlib.util.spec_from_file_location("verity_suite_reads", CHILD / "verity_suite_reads.py")
_reads = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reads)


def run(task: dict, work: Path) -> dict:
    suite = ROOT / task["vars"]["suite"]
    tests = tomllib.loads((suite / "pyproject.toml").read_text()).get("tool", {}).get("verity", {}).get("tests", {})
    if (need := fixtures(task, tests)) and call([sys.executable, "-P", "-m", "research", "data", "fetch-fixtures", "--repo",
                                                str(ROOT), *need], work, ROOT):
        return {"status": "error", "detail": {"error": f"fixtures could not be fetched: {', '.join(need)[:400]}"}}
    for f in (REPORT, READS):
        (work / f).unlink(missing_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="verity-test-file-"))
    ooms = oom_kills()
    try:
        rc = call([sys.executable, "-P", "-c", PYTEST, str(GUARD), "-p", "verity_suite_guard", "-p", "no:cacheprovider",
                   "-p", "no:xdist", f"--basetemp={tmp / 'b'}", f"--rootdir={suite}", "-c", str(suite / "pyproject.toml"),
                   os.path.relpath(ROOT / task["id"].partition(":")[2], suite)], work, suite,
                  environment(task, tests, work, suite))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return verdict(task, rc, work, tmp, (oom_kills() or 0) > ooms if ooms is not None else False)


def fixtures(task: dict, tests: dict) -> list[str]:
    """The registered fixtures git doesn't hold under the suite's directory or its declared inputs (the root's aside), which
    `suites.py` fetches before it runs the suite, when the task reads the registry."""
    if REGISTRY not in task["inputs"]:
        return []
    dirs = [d for d in (p.strip("/") for p in (task["vars"]["suite"], *tests.get("inputs", []))) if d not in ("", ".")]
    reg = json.loads((ROOT / REGISTRY).read_text())
    under = sorted({v for k, v in reg.items() if k.startswith("art:") and isinstance(v, str)
                    and any(v == d or v.startswith(d + "/") for d in dirs)})
    if not under:
        return []
    held = set(subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", *under], capture_output=True,
                              text=True).stdout.split("\0"))
    return [p for p in under if p not in held]


def environment(task: dict, tests: dict, work: Path, suite: Path) -> dict[str, str]:
    """The test process's environment: this one's less `DROP`, this Python's directory first on the PATH (a test's `python3` is the
    workspace's, as under `uv run`, which runs `suites.py`), thread pools sized to the task's cores, a git identity (tests commit
    in throwaway repositories), and the guard's variables, which declare no inputs to it."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(DROP_PREFIXES) and k not in DROP}
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), *filter(None, [env.get("PATH")])])
    for v in THREAD_VARS:
        env.setdefault(v, str(max(1, int(task["resources"]["cores"]))))
    for k in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        env.setdefault(k, "check")
    for k in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        env.setdefault(k, "check@localhost")
    inside = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and Path(p).resolve().is_relative_to(ROOT)]
    env.update(VERITY_SUITE_ROOT=str(ROOT), VERITY_SUITE_ALLOW="[]", VERITY_SUITE_DENY="[]", VERITY_SUITE_WATCH="[]",
               VERITY_SUITE_MARKDOWN="1", VERITY_SUITE_PROGRAMS=json.dumps(tests.get("artifacts", [])),
               VERITY_SUITE_REPORT=str(work / REPORT), VERITY_SUITE_READS=str(work / READS), VERITY_SUITE_DIR=str(suite),
               PYTHONPATH=os.pathsep.join([str(CHILD), *inside]))
    return env


def verdict(task: dict, rc: int, work: Path, tmp: Path, oom: bool) -> dict:
    """The run's status, detail and undeclared reads, from pytest's exit and what the guard recorded."""
    rep = json.loads((work / REPORT).read_text()) if (work / REPORT).is_file() else None
    opened = set((rep or {}).get("outside", {}))
    untraced = {p for progs in (rep or {}).get("untraced", {}).values() for p in progs}
    blind = []
    for line in (work / READS).read_text().splitlines() if (work / READS).is_file() else ():
        kind, _module, rel = (line.split("\t") + ["", ""])[:3]
        if kind == "O":
            opened.add(os.path.relpath(_reads._source(str(ROOT / rel)), ROOT))
        elif kind == "X":
            untraced.add(rel)
        elif kind == "E":
            blind.append(rel)
    counts = (rep or {}).get("counts") or {}
    detail = {"passed": counts.get("passed", 0), "failed": counts.get("failed", 0), "skipped": counts.get("skipped", 0),
              "errors": counts.get("error", 0), "exit": rc}
    if (rep or {}).get("failed"):
        detail["failed_tests"] = rep["failed"]
    if untraced:
        detail["untraced"] = sorted({p.replace(str(tmp), "$BASETEMP", 1) for p in untraced})
    reads, ignored = undeclared(task, opened, [work, tmp])
    if ignored:
        detail["ignored"] = ignored
    why = (f"pytest died by {signal.Signals(-rc).name}" + (" when the kernel killed a process for memory" if oom else "")
           if rc < 0 else "the read guard wrote no report" if rep is None
           else f"a Python its tests started couldn't trace its reads: {blind[0]}" if blind
           else (f"its tests started programs the read guard can't trace ({', '.join(detail['untraced'][:4])}): name the file "
                 "under its suite's `[tool.verity.tests] untraced`, which keys it on the suite's whole tree, or give the test a "
                 "stand-in") if untraced and not allowed(task, untraced)
           else None if rc in PASS + FAIL else f"pytest exited {rc}")
    if why:
        detail["error"] = why
    status = "error" if why else "pass" if rc in PASS else "fail"
    return {"status": status, "detail": detail, "undeclared_reads": reads}


def allowed(task: dict, untraced: set[str]) -> bool:
    """Whether the task may start the untraced programs ``untraced``: any when it is keyed on its suite's whole tree, compilers
    when its closure builds native code."""
    keyed = task["vars"].get("untraced")
    return keyed == "suite" or keyed == "compilers" and all(Path(p).name in COMPILERS for p in untraced)


def undeclared(task: dict, opened: set[str], scratch: list[Path]) -> tuple[list[str], list[str]]:
    """(undeclared reads, ignored files): the files of ``opened`` (paths from the root) outside the task's inputs, its env's spec
    files and, when it reads the registry, the registered fixtures, split into those the graph lists (`graph.tracked`) or the
    registry names and the rest. Files under ``scratch`` aside, and what is no file of the tree: a directory, or a name the
    guard joined to the working directory that was opened relative to another (`shutil.rmtree`'s walk)."""
    inputs = set(task["inputs"])
    spec = graph.spec(ROOT)["env"][task["env"]]["spec"]
    registered = ([v for k, v in json.loads((ROOT / REGISTRY).read_text()).items() if k.startswith("art:") and isinstance(v, str)]
                  if (ROOT / REGISTRY).is_file() else [])
    fetched = registered if REGISTRY in inputs else []
    away = [str(d.resolve().relative_to(ROOT)) for d in scratch if d.resolve().is_relative_to(ROOT)]
    out = sorted(p for p in opened if p not in inputs and not any(graph.matches(p, s) for s in spec)
                 and not any(p == f or p.startswith(f + "/") for f in [*fetched, *away]) and (ROOT / p).is_file())
    listed = set(graph.tracked(ROOT)) if out else set()
    keyed = [p for p in out if p in listed or any(p == f or p.startswith(f + "/") for f in registered)]
    return keyed, [p for p in out if p not in keyed]


def oom_kills() -> int | None:
    """The kernel's count of OOM kills since boot (`/proc/vmstat`), None where it isn't there (`suites.oom_kills`)."""
    vmstat = Path("/proc/vmstat")
    if not vmstat.is_file():
        return None
    return next((int(ln.split()[1]) for ln in vmstat.read_text().splitlines() if ln.startswith("oom_kill ")), None)
