"""Drive's side of the task executor: a check as a `tasks/v1` graph (`verity tasks list --json`, `tools/verity/check/tasks.py`)
put on the executor's queue, and each task's verdict read back by its key.

    Executor(transport).enqueue(graph, priority)       -> {task id: key}
    Executor(transport).enqueue_check(check, priority) -> None: the executor lists the check's graph on a worker
    Executor(transport).graph(check)                   -> (its art: id, {task id: key}) once labelled, else None
    Executor(transport).verdicts(keys)                 -> {key: pass | fail | pending}

A graph's keys hash the lister's environment and its suite splits come from the lister's caches, so only a graph listed where
the tasks run has the keys their verdicts come under: the executor lists a check enqueued with no graph on a worker, puts it
in the store (`tasks-graph/v1`) and labels `commit:<sha> tasks_graph art:<id>` (`vocab.TASKS_GRAPH`), then queues its tasks.

A task's verdict is the `verdict` labels on `task:<key>` (pass or fail, written by the worker that ran `verity tasks run ID`):
fail when any says fail (a duplicate that disagrees is a red, never a green), else pass when one says pass, else pending.  A
task with no key of its own (a build, a circuit-check slice, a job that keeps no verdict: tasks/v1's `unkeyed`) gets one of
its check alone (`keyed`), so it runs in every check and no verdict of it is reused across checks.

The transport is how the rows reach the queue and the graphs and verdicts come back: `QueueTransport`, infra's queue (its
enqueue command, drive.toml's `executor_enqueue`, and the labels and graphs from the store's remote), or `FileTransport`, a
directory, for tests.

Drive's `executor` flag (drive.toml) uses it after landing (`drive.Driver`): `shadow` enqueues the check of each `checked` tip
the after-landing run checks at `AFTER_LANDING_PRIORITY` with no graph, reads the graph the executor listed for it once
labelled, and once both have finished records per tip how the run's verdicts and the executor's compare (`compare` by that
graph's keys, an `executor_shadow` label on `commit:<tip>`; every part either side has, so a part only one side ran
disagrees), changing nothing else (a shadow that fails, the store's or the transport's error, or no graph within
`SHADOW_WAIT_S`, is recorded incomplete, and landing goes on); `research queue shadow-report`
prints the newest of them and whether `SWITCH_STREAK` tips in a row agreed fully, the gate for `on`, where no run is launched
and the executor's verdicts alone verify the tip, `main` following once every task passed.  A failed task marks the tip failed
and is told to `CI`; no culprit is named or reverted from it yet."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

from .store import vocab

PASS, FAIL, PENDING = "pass", "fail", "pending"
#: a shadow comparison's side with no counterpart: the run has no verdict of an executor-only task, and the executor has no
#: task of a part the run records (missing in executor)
NONE, MISSING = "none", "missing"
#: the priority of an after-landing check of `checked`'s tip on the queue, its highest
AFTER_LANDING_PRIORITY = 1
#: the store kind of a graph the executor listed
TASKS_GRAPH_KIND = "tasks-graph/v1"
#: the newest tips in a row whose shadow records agree fully before drive may switch to `executor = "on"`
SWITCH_STREAK = 3
#: a shadowed tip with no graph from the executor this long after its check was enqueued is recorded incomplete, and one whose
#: verdicts are still pending this long after is recorded with them pending
SHADOW_WAIT_S = 6 * 3600
#: what a check's step and suite records say, as a task's verdict: a step skipped by name, or a pass reused, is a pass
OLD_PASS = ("passed", "skipped", "reused")
OLD_FAIL = ("failed",)
#: a lean-audit log's verdict on one Lake package: its audit (`AUDIT PKG: PASS|FAIL`, PKG under the audit's scratch tree on a
#: check's run) or a pass reused (`lean-audit: PKG: PASS reused`)
AUDIT_RE = re.compile(r"^AUDIT (?:\S*?/lean-audit-scratch-[^/\s]+/)?(\S+?): (PASS|FAIL)\b", re.M)
REUSED_RE = re.compile(r"^lean-audit: (\S+?): PASS reused\b", re.M)


def task_target(key: str) -> str:
    return f"task:{key}"


def check_id(graph: dict) -> str:
    """The check a graph is of: its kind and commit (`after-landing:SHA`)."""
    return f"{graph.get('kind') or 'after-landing'}:{graph['commit']}"


def _digest(doc) -> str:
    return hashlib.sha512(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def keyed(graph: dict) -> dict:
    """``graph`` with each task's `key` the one its verdict is labelled under: its own, or for an unkeyed task, SHA-512 over its
    check and id (its `unkeyed` reason kept), which no other check shares."""
    check = check_id(graph)
    return {**graph, "tasks": [t if t.get("key") else {**t, "key": _digest({"check": check, "id": t["id"]})}
                               for t in graph["tasks"]]}


def fold(values: Iterable[str]) -> str:
    """A key's verdict from its labels' values: fail when any is fail, else pass when any is pass, else pending."""
    vs = set(values)
    return FAIL if FAIL in vs else PASS if PASS in vs else PENDING


class Transport(Protocol):
    def enqueue(self, graph_file: Path | None, priority: int, check: str) -> None:
        """``check`` on the queue at ``priority``: ``graph_file``'s tasks, or with None the graph the executor lists."""
        ...

    def graph(self, check: str) -> tuple[str, dict] | None:
        """(its id, the tasks/v1 graph the executor listed for ``check``) once it labelled one, the newest; None until then."""
        ...

    def labels(self, keys: list[str]) -> dict[str, list[str]]:
        """{key: the values of the `verdict` labels on `task:<key>`}."""
        ...


class Executor:
    def __init__(self, transport: Transport):
        self.transport = transport

    def enqueue(self, graph: dict, priority: int) -> dict[str, str]:
        """Put ``graph``'s tasks on the queue at ``priority``; {task id: the key its verdict comes under}."""
        g = keyed(graph)
        with tempfile.TemporaryDirectory(prefix="executor-") as d:
            f = Path(d) / "graph.json"
            f.write_text(json.dumps(g))
            self.transport.enqueue(f, priority, check_id(g))
        return {t["id"]: t["key"] for t in g["tasks"]}

    def enqueue_check(self, check: str, priority: int) -> None:
        """Put ``check`` (`after-landing:SHA`) on the queue at ``priority`` with no graph: the executor lists it on a worker,
        and `graph` reads what it listed."""
        self.transport.enqueue(None, priority, check)

    def graph(self, check: str) -> tuple[str, dict[str, str]] | None:
        """(its id, {task id: the key its verdict comes under}) of the graph the executor listed for ``check``, each unkeyed
        task keyed as `keyed` keys it; None until the executor labelled one.  A graph that isn't tasks/v1 or is of another
        check is refused (ValueError)."""
        if (got := self.transport.graph(check)) is None:
            return None
        gid, g = got
        if not isinstance(g, dict) or g.get("schema") != "tasks/v1" or "commit" not in g or check_id(g) != check:
            what = check_id(g) if isinstance(g, dict) and "commit" in g else "no tasks/v1 check"
            raise ValueError(f"the executor's graph {gid} of {check} is of {what}")
        return gid, {t["id"]: t["key"] for t in keyed(g)["tasks"]}

    def verdicts(self, keys: Iterable[str]) -> dict[str, str]:
        ks = list(dict.fromkeys(keys))
        got = self.transport.labels(ks)
        return {k: fold(got.get(k) or ()) for k in ks}


class FileTransport:
    """The executor as a directory: `enqueue` appends {check, priority, graph} to `enqueued.jsonl` (graph None when enqueued
    without one), the graph the executor listed for a check is `graphs/<check>.json` (`list_graph` writes one, as the
    executor's worker would), and a key's verdicts are the lines {key, verdict} of `verdicts.jsonl` (`finish` writes one)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def enqueue(self, graph_file: Path | None, priority: int, check: str) -> None:
        graph = json.loads(graph_file.read_text()) if graph_file is not None else None
        with (self.root / "enqueued.jsonl").open("a") as f:
            f.write(json.dumps({"check": check, "priority": priority, "graph": graph}) + "\n")

    def enqueued(self) -> list[dict]:
        p = self.root / "enqueued.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else []

    def _graph_file(self, check: str) -> Path:
        return self.root / "graphs" / f"{check.replace(':', '_')}.json"

    def list_graph(self, check: str, graph: dict) -> None:
        p = self._graph_file(check)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(graph))

    def graph(self, check: str) -> tuple[str, dict] | None:
        p = self._graph_file(check)
        return (str(p), json.loads(p.read_text())) if p.is_file() else None

    def finish(self, key: str, verdict: str) -> None:
        with (self.root / "verdicts.jsonl").open("a") as f:
            f.write(json.dumps({"key": key, "verdict": verdict}) + "\n")

    def labels(self, keys: list[str]) -> dict[str, list[str]]:
        p, want, out = self.root / "verdicts.jsonl", set(keys), {}
        for line in p.read_text().splitlines() if p.is_file() else ():
            rec = json.loads(line)
            if rec["key"] in want:
                out.setdefault(rec["key"], []).append(rec["verdict"])
        return out


class QueueTransport:
    """infra's queue: a check goes in through infra's enqueue command (``command``, drive.toml's `executor_enqueue`: research
    imports nothing from `infra/`), run with `--priority N --check ID` and, for a graph drive listed, `--graph FILE` (a tasks/v1
    document, each task's `key` the one its verdict is labelled under), when it inserts one row {key, task, priority, check} per
    task and exits 0 once they are in; with no `--graph`, it exits 0 at once and the executor lists the graph on a worker.  The
    graph the executor listed for a check comes back as the `tasks-graph/v1` artifact a `tasks_graph` label on `commit:<sha>`
    names, and verdicts as `verdict` labels on `task:<key>`, each target's labels pulled from the store's remote while it has
    none here yet."""

    def __init__(self, store, command: tuple[str, ...] = (), cwd: Path | None = None):
        self.store, self.command, self.cwd = store, tuple(command), cwd

    def enqueue(self, graph_file: Path | None, priority: int, check: str) -> None:
        if not self.command:
            raise RuntimeError("drive.toml names no `executor_enqueue` command: set it to infra's enqueue command, the words "
                               "of its argv")
        graph = ["--graph", str(graph_file)] if graph_file is not None else []
        r = subprocess.run([*self.command, *graph, "--priority", str(priority), "--check", check],
                           cwd=self.cwd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode:
            raise RuntimeError(f"{' '.join(self.command)} exited {r.returncode}: {(r.stderr or r.stdout).strip()[-400:]}")

    def graph(self, check: str) -> tuple[str, dict] | None:
        from .store import labels_sync

        target = f"commit:{check.partition(':')[2]}"
        if not (labs := self.store.labels(target, key=vocab.TASKS_GRAPH)) and getattr(self.store, "remote", None) is not None:
            labels_sync.pull_targets(self.store, [target])
            labs = self.store.labels(target, key=vocab.TASKS_GRAPH)
        if not labs:
            return None
        art = labs[-1].value
        if (kind := self.store.get_manifest(art).kind) != TASKS_GRAPH_KIND:
            raise ValueError(f"{target}'s {vocab.TASKS_GRAPH} label names {art}, a {kind}, not a {TASKS_GRAPH_KIND}")
        return art, json.loads(Path(self.store.fetch(art)).read_text())

    def labels(self, keys: list[str]) -> dict[str, list[str]]:
        from .queue import PREFETCH_JOBS
        from .store import labels_sync

        have = {k: [lab.value for lab in self.store.labels(task_target(k), key=vocab.TASK_VERDICT)] for k in keys}
        if (todo := [task_target(k) for k, v in have.items() if not v]) and getattr(self.store, "remote", None) is not None:
            labels_sync.pull_targets(self.store, todo, jobs=PREFETCH_JOBS)
            have.update({k: [lab.value for lab in self.store.labels(task_target(k), key=vocab.TASK_VERDICT)]
                         for k, v in have.items() if not v})
        return have


# -- the shadow comparison: an after-landing run's verdicts beside the executor's, per task where the run records one, else per
#    job (a run records circuit-check, the builds and the other whole steps once, however many tasks the graph cuts them into)

def old_verdict(status: str | None) -> str | None:
    return PASS if status in OLD_PASS else FAIL if status in OLD_FAIL else None


def old_verdicts(run_dirs: Iterable[Path]) -> dict[str, str]:
    """{part: pass | fail} of an after-landing run's records, over its shards: each step (steps.jsonl), each suite as its task
    id (`pytest:NAME`, `lean-suites:NAME`), each Lake package its lean-audit log names (`lean-audit:PKG`).  A part failed in any
    shard fails; one stopped or never run is absent."""
    from .drive import step_records, suite_rows

    got: dict[str, set[str]] = {}

    def put(part: str, v: str | None) -> None:
        if v is not None:
            got.setdefault(part, set()).add(v)

    for d in run_dirs:
        for st in step_records(d):
            put(st.get("step") or "", old_verdict(st.get("status")))
            if st.get("step") == "lean-audit" and st.get("log"):
                try:
                    log = (d / Path(st["log"]).name).read_text(errors="replace")
                except OSError:
                    log = ""
                for pkg, said in AUDIT_RE.findall(log):
                    put(f"lean-audit:{pkg}", PASS if said == "PASS" else FAIL)
                for pkg in REUSED_RE.findall(log):
                    put(f"lean-audit:{pkg}", PASS)
        for name, s in suite_rows(d).items():
            put(f"{s['step']}:{name}", old_verdict(s.get("status")))
    return {p: fold(vs) for p, vs in got.items() if p}


def compare(ids: dict[str, str], old: dict[str, str], new: dict[str, str]) -> dict:
    """How the run's verdicts (``old``, `old_verdicts`) and the executor's (``new``, by key, of the tasks ``ids`` {task id:
    key}) compare, over the union of both sides' parts:

    - each task the run records on its own (`pytest:SUITE`, `lean-audit:PKG`, a job run whole): its verdict beside the task's;
    - each job the run records as a step: the step's verdict beside its tasks' folded (fail when one fails, pending when one
      is, else pass), so a task of a job the run records only whole (a circuit-check slice) is compared through it;
    - a part the run records that no task matches, by id or by job: "missing in executor" (`MISSING`), a disagreement;
    - a task the run has no counterpart of (its id unrecorded, and its job unrecorded or recorded part by part without it):
      "executor only", a disagreement whatever its verdict.

    So a graph smaller than the run never agrees fully, nor one that ran something the run didn't."""
    units: dict[str, tuple[str, str]] = {}
    jobs: dict[str, list[str]] = {}
    only = []
    by_part = {p.partition(":")[0] for p in old if ":" in p}
    for tid, key in ids.items():
        v, job = new.get(key, PENDING), tid.partition(":")[0]
        jobs.setdefault(job, []).append(v)
        if tid in old:
            units[tid] = (old[tid], v)
        elif tid == job or job in by_part or job not in old:
            units[tid] = (NONE, v)
            only.append(tid)
    for job, vs in jobs.items():
        if job in old:
            units[job] = (old[job], FAIL if FAIL in vs else PENDING if PENDING in vs else PASS)
    missing = sorted(p for p in old if p not in units)
    units.update({p: (old[p], MISSING) for p in missing})
    rows = sorted(units.items())
    return {"units": len(rows), "agree": sum(o == n for _, (o, n) in rows),
            "pending": sorted(u for u, (o, n) in rows if n == PENDING and o != NONE),
            "disagree": [{"unit": u, "old": o, "executor": n} for u, (o, n) in rows
                         if o != n and (n != PENDING or o == NONE)],
            "missing": missing, "executor_only": sorted(only)}


def agrees(rec: dict) -> bool:
    """A tip's shadow record agrees fully: every unit agrees, nothing pending, missing in the executor or executor only, and
    no error and nothing left incomplete.  A record without the coverage fields (`compare`'s "missing", "executor_only")
    never agrees."""
    return (not rec.get("error") and not rec.get("incomplete") and rec.get("units", 0) > 0
            and rec.get("agree") == rec.get("units") and not rec.get("pending") and not rec.get("disagree")
            and rec.get("missing") == [] and rec.get("executor_only") == [])


# -- `research queue shadow-report`

@dataclass
class Shadowed:
    tip: str
    ts: str
    rec: dict


def shadow_records(store, pull: bool = True, log: Callable[[str], None] = print) -> list[Shadowed]:
    """Every tip's newest `executor_shadow` record, newest first, its labels pulled from the store's remote first."""
    from .store import ids, labels_sync

    if pull and getattr(store, "remote", None) is not None:
        try:
            labels_sync.pull_prefix(store, "commit:")
        except Exception as e:  # noqa: BLE001  (the local labels still answer, said so)
            log(f"research queue shadow-report: the store's remote wasn't read ({type(e).__name__}: {e}); local records only")
    out = []
    for d in sorted((Path(store.root) / "labels").glob(f"{ids.target_dirname('commit:')}*")):
        tip = d.name.partition("_")[2]
        if labs := store.labels(f"commit:{tip}", key=vocab.EXECUTOR_SHADOW):
            out.append(Shadowed(tip, labs[-1].ts, labs[-1].value))
    return sorted(out, key=lambda s: s.ts, reverse=True)


def shadow_report(records: list[Shadowed], tips: int) -> tuple[str, bool]:
    """(the report of the newest ``tips`` shadowed tips, whether the switch gate holds: the newest `SWITCH_STREAK` agree fully)."""
    shown = records[:tips]
    streak = next((i for i, s in enumerate(records) if not agrees(s.rec)), len(records))
    lines = [f"executor shadow: the newest {len(shown)} of {len(records)} checked tips compared"]
    for s in shown:
        r = s.rec
        what = (f"error: {r['error']}" if r.get("error") else f"incomplete: {r['incomplete']}" if r.get("incomplete") else
                f"agree {r.get('agree', 0)}/{r.get('units', 0)}, disagree {len(r.get('disagree') or ())}"
                + (f", pending {len(r['pending'])}" if r.get("pending") else "")
                + ("" if "missing" in r else ", no coverage counts")
                + (f", missing in executor {len(r['missing'])}" if r.get("missing") else "")
                + (f", executor only {len(r['executor_only'])}" if r.get("executor_only") else ""))
        lines.append(f"{s.tip[:12]}  {s.ts}  {what}  (run {r.get('run') or '-'})")
        for x in r.get("disagree") or ():
            lines.append(f"    {x['unit']}: " + ("missing in executor (run " + x["old"] + ")" if x["executor"] == MISSING else
                                                 f"executor only (executor {x['executor']})" if x["old"] == NONE else
                                                 f"run {x['old']}, executor {x['executor']}"))
        lines += [f"    {u}: executor pending" for u in r.get("pending") or ()]
    ok = streak >= SWITCH_STREAK
    lines.append(f"switch gate ({SWITCH_STREAK} tips in a row agreeing fully, none missing in executor or executor only, none "
                 f"incomplete): {'met' if ok else 'not met'} "
                 f"(the newest {streak} agree)")
    return "\n".join(lines), ok

