"""lean-suites: `tools/verity/check/suites.py` on suites check runs after its lean-build (`suites.AFTER_LEAN`; the task's var
`suites`), as check's lean-suites step runs them: `--lean-built`, lake on the PATH through elan (`lean_build.ELAN`), each suite's
timeout, and the task's cores and memory as `--jobs` and `--mem-gb`.  A task of a suite's part (its var `files`, K/M) runs
`--files K/M` with `--timings` a record that holds none of the suite's files, so suites.py packs them by path alone and every
machine splits one tree alike.  First each verifier executable the task reads goes to its path in the tree (its var `bin`,
lean-build's `.lake/build/bin`), where the suites look, unless the tree holds those bytes there already (lean-build built them
on this machine); then the rest of lean-build's build (its var `tree`, `lean_build.TREE`) is unpacked into the tree and `lake
build` run on it there, under check's lock, unless the tree holds that build already (`lean_build.STAMP`), so the tree is as
check's lean-build step leaves it for its lean-suites step: the tests' own `lake build`s of the verifier, at once in every
test process, find every module built.  A tree that holds only the executables has each of them build the whole verifier in
one `.lake` together, which breaks their links.  That `lake build` failing on the build lean-build passed is an error.

Its status is from suites.py's exit and its report.  Every suite passed, a pass of its key reused among them, is a pass.  Suites
that failed on their tests, on reads outside their inputs or on their timeout are a fail, each named in `detail.failed` (its name,
directory and failed tests), by which drive names, allows and blames each as the suite's own red; so is a run whose tests changed
the repository.  A suite the report lacks or reports skipped (not run), a failure the report gives no cause of (a kill, fixtures
that couldn't be fetched), a usage error and a signal are an error.  `detail.suites` is each suite's status."""
from __future__ import annotations

import fcntl
import json
import os
import shlex
import shutil
import sys
import tarfile
from pathlib import Path

from .. import ROOT, contract
from ..graph import LOG
from . import call
from .lean_build import ELAN, LAKE_BUILD, STAMP, TREE

SUITES = "tools/verity/check/suites.py"
REPORT, LOGS, TIMINGS = "suites-lean.json", "suites", "suites-timings.json"
#: in the package's `.lake`: the tree's tasks stage lean-build's build one at a time
STAGE_LOCK = "graph-stage.lock"
#: suites.py's line when tests wrote into the repository, which fails its run with no suite failed
WROTE = "the tests changed"
#: the most failed tests a row names: a longer list is cut, and then failed for more than the tests it names
NAMED = 100
PASSED = ("passed", "cached")


def place(task: dict, work: Path) -> str | None:
    """What ``task`` reads, from its work directory into the tree: the executables to their paths, executable; then the build
    they were linked in (`stage`).  None, or why the tree couldn't be made so."""
    for r in task["reads"]:
        if r["output"] == TREE:
            continue
        src, dst = work / task["vars"]["bin"] / r["output"], ROOT / task["vars"]["bin"] / r["output"]
        if not (dst.is_file() and contract.file_digest(dst) == contract.file_digest(src)):
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.graph-{os.getpid()}")
            shutil.copyfile(src, tmp)
            tmp.chmod(0o755)
            os.replace(tmp, dst)
        dst.chmod(0o755)
    return stage(task, work) if task["vars"].get("tree") else None


def stage(task: dict, work: Path) -> str | None:
    """lean-build's build (the var `tree`) unpacked into the tree and `lake build` run on it in the package whose `.lake` holds
    the executables, one task of the tree at a time; none of it where the package's `STAMP` names that build, which lean-build
    in this tree, or a task that staged it here, wrote.  None, or why the build didn't build here."""
    lake = (ROOT / task["vars"]["bin"]).parent.parent
    want = contract.file_digest(work / task["vars"]["tree"])
    lake.mkdir(parents=True, exist_ok=True)
    with open(lake / STAGE_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (lake / STAMP).is_file() and (lake / STAMP).read_text() == want:
            return None
        (lake / STAMP).unlink(missing_ok=True)
        unpack(work / task["vars"]["tree"], ROOT)
        if rc := call(["bash", "-c", LAKE_BUILD], work, lake.parent):
            return f"`lake build` in {lake.parent.relative_to(ROOT)}, on the build lean-build passed, unpacked here, exited {rc}"
        (lake / STAMP).write_text(want)
    return None


def unpack(archive: Path, root: Path) -> None:
    """``archive`` (`lean_build.pack`'s) into ``root``: each file to its path, under a package's `.lake/build`, by a rename, so a
    process that has the file it replaces open or mapped keeps reading those bytes; its mode and time as packed."""
    with tarfile.open(archive) as tf:
        for m in tf:
            parts = Path(m.name).parts
            if (not m.isfile() or Path(m.name).is_absolute() or ".." in parts
                    or not any(parts[i:i + 2] == (".lake", "build") for i in range(len(parts) - 2))):
                raise ValueError(f"{archive.name}: {m.name} is no file under a package's .lake/build")
            dst = root / m.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.graph-{os.getpid()}")
            with tf.extractfile(m) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out)
            tmp.chmod(m.mode & 0o777)
            os.utime(tmp, (m.mtime, m.mtime))
            os.replace(tmp, dst)


def run(task: dict, work: Path) -> dict:
    if why := place(task, work):
        return {"status": "error", "detail": {"error": why}}
    names = task["vars"]["suites"].split()
    r = task["resources"]
    part = []
    if files := task["vars"].get("files"):
        (work / TIMINGS).write_text(json.dumps({n: {} for n in names}))
        part = ["--files", files, "--timings", str(work / TIMINGS)]
    argv = [sys.executable, str(ROOT / SUITES), *names, "--lean-built", "--json", str(work / REPORT), "--logs", str(work / LOGS),
            "--timeouts", "--jobs", str(max(1, int(r["cores"]))), "--mem-gb", str(r["mem_gb"]), *part]
    rc = call(["bash", "-c", ELAN + shlex.join(argv)], work, ROOT)
    if rc < 0 or rc > 128:
        return {"status": "error", "detail": {"error": f"suites.py was killed (exit {rc})"}}
    if rc not in (0, 1) or not (work / REPORT).is_file():
        return {"status": "error", "detail": {"error": f"suites.py exited {rc}" + ("" if (work / REPORT).is_file() else
                                                                                   " and wrote no report")}}
    rows = json.loads((work / REPORT).read_text())
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
