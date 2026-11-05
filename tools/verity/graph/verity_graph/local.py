"""`verity graph check --local`: the serial reference executor.  No cache and no parallelism: the graph listed once, and every
task of it in `order`, each in a fresh work directory holding the artifacts it reads at their producers' output paths, run as
its argv with `--work DIR --graph PART` (its part of that listing, so that no task's run lists again), its result/v1 checked
(the schema, its key against the one computed here from the reads' digests as they appear, its outputs' bytes against their
digests).  An error (no result, a key that isn't the executor's, an undeclared read) is never a result and is retried; a fail
is a result and is never retried; a task whose read didn't pass is cancelled.  It writes an events/v1 line for each event,
purpose `adhoc` (`verity graph check --local`: to its work directory's `events.jsonl`)."""
from __future__ import annotations

import datetime
import json
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import IO

from . import contract, graph

#: what `verity graph run` gets past a task's own timeout_s before the executor gives up on it
GRACE_S = 60


def verity() -> str:
    """The `verity` command of this interpreter's environment."""
    exe = Path(sys.executable).with_name("verity")
    if not exe.is_file():
        raise graph.GraphError(f"no `verity` beside {sys.executable}: `uv sync --all-packages` installs it")
    return str(exe)


def launcher(root: Path, doc: dict, spawn: Callable[..., subprocess.CompletedProcess] = subprocess.run
             ) -> Callable[[dict, Path], dict]:
    """Run a task of the listing ``doc`` as its graph argv with `--work DIR --graph PART` from ``root``, PART its `graph.part`
    written beside DIR, so that its run lists nothing, and parse the result/v1 it prints."""
    by = {t["id"]: t for t in doc["tasks"]}

    def launch(task: dict, work: Path) -> dict:
        part = work.with_name(f"{work.name}.graph.json")
        part.write_bytes(contract.canonical(graph.part(doc, task["id"], by)))
        argv = [verity() if task["argv"][0] == "verity" else task["argv"][0], *task["argv"][1:], "--work", str(work),
                "--graph", str(part)]
        try:
            p = spawn(argv, cwd=root, capture_output=True, text=True, timeout=task["timeout_s"] + GRACE_S)
        except subprocess.TimeoutExpired:
            return {"error": f"`verity graph run` outlived timeout_s + {GRACE_S} s"}
        lines = p.stdout.strip().splitlines()
        try:
            return json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError):
            return {"error": f"exit {p.returncode}, no result/v1 on stdout: {(p.stderr or p.stdout).strip()[-400:]}"}
    return launch


class Events:
    """events/v1 lines, one per event, to ``out`` (nothing without it)."""

    def __init__(self, out: IO[str] | None, commit: str | None, purpose: str = "adhoc") -> None:
        self.out, self.commit, self.purpose = out, commit, purpose
        self.submission = f"local-{uuid.uuid4().hex[:16]}"
        self.queued: dict[str, float] = {}

    def __call__(self, event: str, task: dict, key: str | None, res: dict | None = None) -> None:
        now = time.monotonic()
        if event == "queued":
            self.queued[task["id"]] = now
        if self.out is None:
            return
        line = {"schema": contract.SCHEMA_EVENTS, "ts": datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds"),
                "submission": self.submission, "purpose": self.purpose, "commit": self.commit, "task": task["id"], "key": key,
                "event": event, "worker": "local", "wait_s": round(now - self.queued.get(task["id"], now), 3),
                "seconds": (res or {}).get("seconds"), "status": (res or {}).get("status"),
                "peak_rss_gb": (res or {}).get("peak_rss_gb"), "hit_from": None}
        self.out.write(json.dumps(line, sort_keys=True) + "\n")
        self.out.flush()


def _error(task: dict, key: str | None, why: str) -> dict:
    return {"schema": contract.SCHEMA_RESULT, "task": task["id"], "key": key, "status": "error", "seconds": 0, "cpu_s": 0,
            "peak_rss_gb": 0, "outputs": {}, "undeclared_reads": [], "detail": {"error": why}, "log": None}


def judged(task: dict, key: str, got: dict, work: Path) -> dict:
    """``got``, a launch's result of ``task`` under ``key``, if it is one this executor takes; else an error saying why."""
    if errs := contract.result_errors(got, task):
        return _error(task, key, got.get("error") if isinstance(got, dict) and "error" in got and len(got) == 1
                      else f"no result/v1: {'; '.join(errs)}")
    if got["undeclared_reads"]:
        return {**got, "status": "error", "detail": {**got["detail"], "error": "undeclared reads: "
                                                     + ", ".join(got["undeclared_reads"])}}
    if got["key"] != key:
        return _error(task, key, f"the run's key {got['key']} isn't the executor's {key}")
    paths = {o["name"]: o["path"] for o in task["outputs"]}
    for name, d in got["outputs"].items():
        if not (work / paths[name]).is_file() or contract.file_digest(work / paths[name]) != d:
            return _error(task, key, f"output {name} at {paths[name]} isn't the bytes its digest names")
    return got


def check(root: Path, only: set[str] | None, work_root: Path, launch: Callable[[dict, Path], dict] | None = None,
          retries: int = 1, events: IO[str] | None = None, say: IO[str] = sys.stderr, doc: dict | None = None) -> dict:
    """Run the graph of ``root`` (``only`` those families and what they read; or ``doc``) serially under ``work_root``:
    {"results": {id: its pass or fail}, "errors": {id: its last error}, "cancelled": [ids], "ok": every task passed}."""
    doc = doc or graph.listing(root, only)
    tasks = graph.order(doc["tasks"])
    store = work_root / "artifacts"
    store.mkdir(parents=True, exist_ok=True)
    launch = launch or launcher(root, doc)
    ev = Events(events, doc["commit"])
    for t in tasks:
        ev("queued", t, t["key"])
    results: dict[str, dict] = {}
    errors: dict[str, dict] = {}
    cancelled: list[str] = []
    by = {t["id"]: t for t in tasks}
    for n, t in enumerate(tasks, 1):
        if blocked := sorted({r["task"] for r in t["reads"] if results.get(r["task"], {}).get("status") != "pass"}):
            cancelled.append(t["id"])
            ev("cancelled", t, None)
            print(f"[{n}/{len(tasks)}] {t['id']}: cancelled, its reads of {', '.join(blocked)} didn't pass", file=say, flush=True)
            continue
        reads = {r["task"]: results[r["task"]]["outputs"] for r in t["reads"]}
        key = t["key"] or contract.key(t, doc["files"], reads, doc["envs"][t["env"]]["digest"])
        for attempt in range(retries + 1):
            work = work_root / "work" / f"{n:05d}.{attempt}"
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            for r in t["reads"]:
                path = next(o["path"] for o in by[r["task"]]["outputs"] if o["name"] == r["output"])
                (work / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(store / results[r["task"]]["outputs"][r["output"]].partition(":")[2], work / path)
            ev("started", t, key)
            res = judged(t, key, launch(t, work), work)
            if res["status"] != "error":
                break
            ev("errored", t, key, res)
        if res["status"] == "error":
            print(f"[{n}/{len(tasks)}] {t['id']}: error after {retries + 1} attempts: {res['detail'].get('error')}", file=say,
                  flush=True)
            errors[t["id"]] = res
            continue
        paths = {o["name"]: o["path"] for o in t["outputs"]}
        for name, d in res["outputs"].items():
            shutil.copyfile(work / paths[name], store / d.partition(":")[2])
        results[t["id"]] = res
        ev("finished", t, key, res)
        print(f"[{n}/{len(tasks)}] {t['id']}: {res['status']} in {res['seconds']:.1f} s", file=say, flush=True)
    ok = not cancelled and not errors and all(r["status"] == "pass" for r in results.values())
    return {"results": results, "errors": errors, "cancelled": cancelled, "ok": ok}
