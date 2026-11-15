"""The runners: each module here, `<kind>.py` with a kind's - written _, runs the tasks of that kind as `run(task: dict, work:
Path) -> dict`, returning result/v1's `status` (pass or fail; error when it couldn't run) and optionally its `detail` and
`undeclared_reads`.  `execute` gives it the task's work directory, its declared reads already there at their producers'
paths, and makes the rest of result/v1: the key, the measured seconds, CPU and peak memory, the outputs' digests and the log's.
A runner's module and this one are inputs of every task of its kind (`graph.runner_sources`)."""
from __future__ import annotations

import importlib
import os
import resource
import signal
import subprocess
import time
from pathlib import Path
from types import ModuleType

from .. import contract
from ..graph import LOG


class Timeout(Exception):
    """A command `call` started outlived the task's timeout_s."""


def module(kind: str) -> ModuleType:
    return importlib.import_module(f"{__name__}.{kind.replace('-', '_')}")


def run(task: dict, work: Path) -> dict:
    return module(task["kind"]).run(task, work)


#: the monotonic time by which the running task must end (`execute` sets it from the task's timeout_s)
_deadline: list[float] = []


def call(argv: list[str], work: Path, cwd: Path, env: dict[str, str] | None = None) -> int:
    """Run ``argv`` in ``cwd``, its output appended to the task's log, killed with its process group when the task's timeout_s
    runs out (`Timeout`); its exit status."""
    left = (_deadline[-1] - time.monotonic()) if _deadline else None
    if left is not None and left <= 0:
        raise Timeout(argv[0])
    with open(work / LOG, "ab") as log:
        log.write(f"$ {' '.join(argv)}  (in {cwd})\n".encode())
        log.flush()
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        try:
            return p.wait(timeout=left)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            raise Timeout(argv[0]) from None


def read_digests(task: dict, graph: dict, work: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    """({read task: {output: digest}} of the reads in ``work``, at their producers' paths; the reads missing there)."""
    by = {t["id"]: t for t in graph["tasks"]}
    got: dict[str, dict[str, str]] = {}
    missing = []
    for r in task["reads"]:
        path = next(o["path"] for o in by[r["task"]]["outputs"] if o["name"] == r["output"])
        if (work / path).is_file():
            got.setdefault(r["task"], {})[r["output"]] = contract.file_digest(work / path)
        else:
            missing.append(f"{r['task']}'s {r['output']} ({path})")
    return got, missing


def execute(task: dict, graph: dict, work: Path, runner=None) -> dict:
    """Run ``task`` (of ``graph``) in ``work`` with its kind's runner (or ``runner``), and its result/v1."""
    work.mkdir(parents=True, exist_ok=True)
    (work / LOG).write_bytes(b"")
    reads, missing = read_digests(task, graph, work)
    env = graph["envs"][task["env"]]["digest"]
    key = None if missing else contract.key(task, graph["files"], reads, env)
    t0, self0, kids0 = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    detail: dict = {}
    undeclared: list[str] = []
    if missing:
        status, detail = "error", {"error": f"reads missing from the work directory: {'; '.join(missing)}"}
    else:
        _deadline.append(t0 + task["timeout_s"])
        try:
            got = (runner or run)(task, work)
        except Timeout as e:  # a slow machine established nothing about the task: an error, never cached as its verdict
            got = {"status": "error", "detail": {"timeout_s": task["timeout_s"], "error": f"{e} outlived the task's timeout"}}
        except Exception as e:  # noqa: BLE001 - a runner that raises established nothing about the task: an error, never a result
            got = {"status": "error", "detail": {"error": f"the runner raised {type(e).__name__}: {e}"}}
        finally:
            _deadline.pop()
        if not isinstance(got, dict) or (extra := set(got) - {"status", "detail", "undeclared_reads"}):
            got = {"status": "error", "detail": {"error": f"the runner returned {got!r}: status, detail and undeclared_reads"
                                                           " are its fields" if not isinstance(got, dict) else
                                                           f"the runner returned fields {sorted(extra)} that are the core's"}}
        status, detail, undeclared = got.get("status"), dict(got.get("detail") or {}), sorted(got.get("undeclared_reads") or [])
        if undeclared:
            status = "error"
    self1, kids1 = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    outputs = {o["name"]: contract.file_digest(work / o["path"]) for o in task["outputs"] if (work / o["path"]).is_file()}
    if status == "pass" and (gone := [o["name"] for o in task["outputs"] if o["name"] not in outputs]):
        status, detail = "fail", {**detail, "error": f"passed without its outputs {', '.join(gone)}"}
    cpu = sum(getattr(b, f) - getattr(a, f) for a, b in ((self0, self1), (kids0, kids1)) for f in ("ru_utime", "ru_stime"))
    res = {"schema": contract.SCHEMA_RESULT, "task": task["id"], "key": key, "status": status,
           "seconds": round(time.monotonic() - t0, 3), "cpu_s": round(cpu, 3),
           "peak_rss_gb": round(max(self1.ru_maxrss, kids1.ru_maxrss) / (1 << 20), 3), "outputs": outputs,
           "undeclared_reads": undeclared, "detail": detail, "log": contract.file_digest(work / LOG)}
    if errs := contract.result_errors(res, task):
        res.update(status="error", undeclared_reads=[], detail={"error": f"no result/v1: {'; '.join(errs)}", "runner": detail})
    return res
