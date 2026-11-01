"""Drive's side of the task executor: a check as a `tasks/v1` graph (`verity tasks list --json`, `tools/verity/check/tasks.py`)
put on the executor's queue, and each task's verdict read back by its key.

    Executor(transport).enqueue(graph, priority) -> {task id: key}
    Executor(transport).verdicts(keys)           -> {key: pass | fail | pending}

A task's verdict is the `verdict` labels on `task:<key>` (pass or fail, written by the worker that ran `verity tasks run ID`):
fail when any says fail (a duplicate that disagrees is a red, never a green), else pass when one says pass, else pending.  A
task with no key of its own (a build, a circuit-check slice, a job that keeps no verdict: tasks/v1's `unkeyed`) gets one of
its check alone (`keyed`), so it runs in every check and no verdict of it is reused across checks.

The transport is how the rows reach the queue and the verdicts come back: `QueueTransport`, infra's queue (its `ENQUEUE`
command, and the labels from the store's remote), or `FileTransport`, a directory, for tests.

Drive's `executor` flag (drive.toml) uses it after landing (`drive.Driver`): `shadow` lists the graph of each `checked` tip the
after-landing run checks (detached, in a worktree of the tip: `drive.list_tasks`), enqueues it at `AFTER_LANDING_PRIORITY`, and
once both have finished records per tip how the run's verdicts and the executor's compare (`compare`, an `executor_shadow` label
on `commit:<tip>`), changing nothing else; `research queue shadow-report` prints the newest of them and whether `SWITCH_STREAK`
tips in a row agreed fully, the gate for `on`, where no run is launched and the executor's verdicts alone verify the tip, `main`
following once every task passed.  A failed task marks the tip failed and is told to `CI`; no culprit is named or reverted from
it yet."""
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
#: the priority of an after-landing check of `checked`'s tip on the queue, its highest
AFTER_LANDING_PRIORITY = 1
#: infra's command that puts a graph's rows on the queue: `--graph FILE` (a tasks/v1 document, each task's `key` the one its
#: verdict is labelled under), `--priority N`, `--check ID`; it inserts one row {key, task, priority, check} per task and
#: exits 0 once they are in
ENQUEUE = ("research", "executor", "enqueue")
#: the newest tips in a row whose shadow records agree fully before drive may switch to `executor = "on"`
SWITCH_STREAK = 3
#: a tip whose executor verdicts are still pending this long after its graph was enqueued is recorded with them pending
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
    def enqueue(self, graph_file: Path, priority: int, check: str) -> None: ...

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

    def verdicts(self, keys: Iterable[str]) -> dict[str, str]:
        ks = list(dict.fromkeys(keys))
        got = self.transport.labels(ks)
        return {k: fold(got.get(k) or ()) for k in ks}


class FileTransport:
    """The executor as a directory: `enqueue` appends {check, priority, graph} to `enqueued.jsonl`, and a key's verdicts are
    the lines {key, verdict} of `verdicts.jsonl` (`finish` writes one)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def enqueue(self, graph_file: Path, priority: int, check: str) -> None:
        with (self.root / "enqueued.jsonl").open("a") as f:
            f.write(json.dumps({"check": check, "priority": priority, "graph": json.loads(graph_file.read_text())}) + "\n")

    def enqueued(self) -> list[dict]:
        p = self.root / "enqueued.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else []

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
    """infra's queue: rows go in through `ENQUEUE`, and verdicts come back as `verdict` labels on `task:<key>` in the store,
    pulled from its remote for each key with no verdict here yet."""

    def __init__(self, store, command: tuple[str, ...] = ENQUEUE, cwd: Path | None = None):
        self.store, self.command, self.cwd = store, command, cwd

    def enqueue(self, graph_file: Path, priority: int, check: str) -> None:
        r = subprocess.run([*self.command, "--graph", str(graph_file), "--priority", str(priority), "--check", check],
                           cwd=self.cwd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode:
            raise RuntimeError(f"{' '.join(self.command)} exited {r.returncode}: {(r.stderr or r.stdout).strip()[-400:]}")

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
    key}) compare: each task the run records on its own, and each job the run records only whole, its tasks' verdicts folded
    (fail when one fails, pending when one is, else pass).  "none" for a unit the run has no verdict of."""
    units: dict[str, tuple[str, str]] = {}
    jobs: dict[str, list[str]] = {}
    for tid, key in ids.items():
        if tid in old:
            units[tid] = (old[tid], new.get(key, PENDING))
        else:
            jobs.setdefault(tid.partition(":")[0], []).append(new.get(key, PENDING))
    for job, vs in jobs.items():
        units[job] = (old.get(job, "none"), FAIL if FAIL in vs else PENDING if PENDING in vs else PASS)
    rows = sorted(units.items())
    return {"units": len(rows), "agree": sum(o == n for _, (o, n) in rows),
            "pending": sorted(u for u, (_, n) in rows if n == PENDING),
            "disagree": [{"unit": u, "old": o, "executor": n} for u, (o, n) in rows if o != n and n != PENDING]}


def agrees(rec: dict) -> bool:
    """A tip's shadow record agrees fully: every unit agrees, nothing pending, no error."""
    return not rec.get("error") and rec.get("units", 0) > 0 and rec.get("agree") == rec.get("units")


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
        what = (f"error: {r['error']}" if r.get("error") else
                f"agree {r.get('agree', 0)}/{r.get('units', 0)}, disagree {len(r.get('disagree') or ())}"
                + (f", pending {len(r['pending'])}" if r.get("pending") else ""))
        lines.append(f"{s.tip[:12]}  {s.ts}  {what}  (run {r.get('run') or '-'})")
        lines += [f"    {x['unit']}: run {x['old']}, executor {x['executor']}" for x in r.get("disagree") or ()]
        lines += [f"    {u}: executor pending" for u in r.get("pending") or ()]
    ok = streak >= SWITCH_STREAK
    lines.append(f"switch gate ({SWITCH_STREAK} tips in a row agreeing fully): {'met' if ok else 'not met'} "
                 f"(the newest {streak} agree)")
    return "\n".join(lines), ok
