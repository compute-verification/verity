"""Drive's side of the task executor: a check put on the executor's queue, and each task's verdict read back by its key.

    Executor(transport).enqueue(graph, priority)       -> {task id: key}
    Executor(transport).enqueue_check(check, priority) -> None: the executor lists the check's tasks/v1 graph on a worker
    Executor(transport).graph(check)                   -> (its art: id, {task id: key}) once labelled, else None
    Executor(transport).verdicts(keys)                 -> {key: pass | fail | pending}
    Executor(transport).enqueue_graph(check, priority) -> None: the executor lists the check's graph/v2 on a worker
    Executor(transport).graph_v2(check)                -> (its art: id, the graph/v2 document) once labelled, else None
    Executor(transport).settle(graph)                  -> {task id: (pass | fail | pending | blocked, its result or None)}

Two graph schemas.  `tasks/v1` (`verity tasks list --json`, `tools/verity/check/tasks.py`), which `executor = "shadow"`
compares with the after-landing run, and graph/v2 (`verity graph`, `tools/verity/graph`), which `executor = "on"` verifies a tip
by, beside a train.sh run of the steps drive still takes from one (`drive.SLIM_STEPS`).

A graph's keys hash the lister's environment and its suite splits come from the lister's caches, so only a graph listed where
the tasks run has the keys their verdicts come under: the executor lists a check enqueued with no graph on a worker, puts it
in the store (`tasks-graph/v1`) and labels `commit:<sha> tasks_graph art:<id>` (`vocab.TASKS_GRAPH`), then queues its tasks.

A task's verdict comes from the `verdict` labels on `task:<key>`, written by the executor from its run of `verity tasks run
ID`: pass or fail by that run's result, or `ERROR` when no run gave one, which is pending (not run), so a later graph runs the
key again; a step that skipped gets no label and stays pending.  A fail stands until an error label is written after it (a
false fail relabelled), so a rerun that passes never turns a fail green (a test that flakes is deleted, not rerun); past
that, the newest of pass and error is the key's.  A task with no key of its own (a build, a circuit-check slice, a job that keeps no
verdict: tasks/v1's `unkeyed`) gets one of its check alone (`keyed`), so it runs in every check and no verdict of it is reused
across checks.

The transport is how the rows reach the queue and the graphs and verdicts come back: `QueueTransport`, infra's queue (its
enqueue command, drive.toml's `executor_enqueue`, and the labels and graphs from the store's remote), or `FileTransport`, a
directory, for tests.

Drive's `executor` flag (drive.toml) uses it after landing (`drive.Driver`): `shadow` enqueues the check of each `checked` tip
the after-landing run checks at `AFTER_LANDING_PRIORITY` with no graph, reads the graph the executor listed for it once
labelled, and once both have finished records per tip how the run's verdicts and the executor's compare (`compare` by that
graph's keys, an `executor_shadow` label on `commit:<tip>`; every part either side has, so a part only one side ran
disagrees), changing nothing else (a shadow that fails, the store's or the transport's error, or no graph within
`SHADOW_WAIT_S`, is recorded incomplete, and landing goes on); `research queue shadow-report`
prints the newest of them and whether `SWITCH_STREAK` tips in a row agreed fully, the gate for `on`.

`on` launches no run of every step: it enqueues the tip's check with `--schema verity-graph/v2` and no graph, the executor lists
graph/v2 on a worker, puts it in the store (`verity-graph/v2`) and labels `commit:<sha> verity_graph art:<id>`
(`vocab.VERITY_GRAPH`), and each task's result/v1 document is a `result` label on `task:<key>` (`vocab.TASK_RESULT`; an error is
never a result, so only pass and fail appear).  A graph/v2 task that reads none has its key in the graph; one that reads others
has none there, and `settle` computes it by the contract's key rule (`task_key`, held to
`tools/verity/graph/tests/key_vectors.json`) once each task it reads passed, from their results' `outputs`; it is blocked, never
run, once one of them failed or is blocked.  Drive judges the results (`drive.Driver._follow_tasks`) beside a train.sh run of
`drive.SLIM_STEPS` alone (lean-agreement and ratchet-gate), whose verdicts on those steps are the only ones it counts."""
from __future__ import annotations

import hashlib
import heapq
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

from .store import vocab

PASS, FAIL, PENDING = "pass", "fail", "pending"
#: the label of a task whose runs all ended without a result (the machine lost, the run refused or cancelled): no verdict of
#: the tree, so `fold` reads it as pending
ERROR = "error"
#: a graph/v2 task that never runs: a task it reads failed, is blocked, or passed without the output it reads
BLOCKED = "blocked"
#: a shadow comparison's side with no counterpart: the run has no verdict of an executor-only task, and the executor has no
#: task of a part the run records (missing in executor)
NONE, MISSING = "none", "missing"
#: the priority of an after-landing check of `checked`'s tip on the queue, its highest
AFTER_LANDING_PRIORITY = 1
#: the store kind of a tasks/v1 graph the executor listed
TASKS_GRAPH_KIND = "tasks-graph/v1"
#: graph/v2's schema (`verity_graph.contract.SCHEMA_GRAPH`), which is also the store kind of one the executor listed
GRAPH_V2 = "verity-graph/v2"
TASKS_V1 = "tasks/v1"
#: {schema: (the label on `commit:<sha>` naming the graph the executor listed of it, the graph's store kind)}
LISTED = {TASKS_V1: (vocab.TASKS_GRAPH, TASKS_GRAPH_KIND), GRAPH_V2: (vocab.VERITY_GRAPH, GRAPH_V2)}
#: the key rule's tag (`verity_graph.contract.KEY_TAG`)
KEY_TAG = b"verity-key/v1\n"
#: the fields of a graph/v2 task `settle` reads
V2_FIELDS = {"id", "kind", "env", "argv", "vars", "inputs", "reads", "outputs"}
#: the newest tips in a row whose shadow records agree fully before drive may switch to `executor = "on"`
SWITCH_STREAK = 3
#: a shadowed tip with no graph from the executor this long after its check was enqueued is recorded incomplete, and one whose
#: verdicts are still pending this long after is recorded with them pending
SHADOW_WAIT_S = 6 * 3600
#: what a check's step and suite records say, as a task's verdict: a step skipped by name, or a pass reused, is a pass
OLD_PASS = ("passed", "skipped", "reused", "cached")
OLD_FAIL = ("failed",)
#: a suite's record that is no verdict of it: a step skips by name the suites another step runs (`pytest` skips the Lean suites)
SUITE_NO_VERDICT = ("skipped",)
#: a task of a suite split by test file (`pytest:SUITE:K/M`): compared through its suite when the run records it whole
SPLIT_RE = re.compile(r":\d+/\d+$")
#: a lean-audit log's verdict on one Lake package: its audit (`AUDIT PKG: PASS|FAIL`, PKG under the audit's scratch tree on a
#: check's run) or a pass reused (`lean-audit: PKG: PASS reused`)
AUDIT_RE = re.compile(r"^AUDIT (?:\S*?/lean-audit-scratch-[^/\s]+/)?(\S+?): (PASS|FAIL)\b", re.M)
REUSED_RE = re.compile(r"^lean-audit: (\S+?): PASS reused\b", re.M)


def task_target(key: str) -> str:
    """The store target of a task's key: `task:<128 hex>`, a graph/v2 key without its `sha512:`."""
    return f"task:{key.removeprefix('sha512:')}"


def canonical(x) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def task_key(task: dict, files: dict[str, str], env: str, reads: dict[str, dict[str, str]]) -> str:
    """A graph/v2 task's key by the contract's rule (`verity_graph.contract.key`): ``files`` the graph's file digests, ``env``
    its env's digest, ``reads`` {task id: {output name: digest}} of the tasks it reads, their results' `outputs`."""
    doc = {"env": env, "kind": task["kind"], "argv": list(task["argv"]), "vars": dict(task["vars"]),
           "inputs": sorted([p, files[p]] for p in task["inputs"]),
           "reads": sorted([r["task"], r["output"], reads[r["task"]][r["output"]]] for r in task["reads"]),
           "outputs": sorted([o["name"], o["path"]] for o in task["outputs"])}
    return "sha512:" + hashlib.sha512(KEY_TAG + canonical(doc)).hexdigest()


def fold_results(docs: Iterable[object], key: str) -> tuple[str, dict | None]:
    """A graph/v2 key's state from the result/v1 documents labelled on it, those of another key ignored: fail (with the first
    failed) when any failed, else pass (with the first passed, whose `outputs` its readers' keys take) when one passed, else
    pending."""
    got = [d for d in docs if isinstance(d, dict) and d.get("key") == key and d.get("status") in (PASS, FAIL)]
    first = next((d for s in (FAIL, PASS) for d in got if d["status"] == s), None)
    return (first["status"], first) if first else (PENDING, None)


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
    """A key's verdict from its labels' values, oldest first: fail while a fail has no `ERROR` after it, else pass when the
    newest is pass, else pending (no label, or `ERROR` newest)."""
    vs = list(values)
    if FAIL in vs and ERROR not in vs[len(vs) - vs[::-1].index(FAIL):]:
        return FAIL
    return PASS if vs and vs[-1] == PASS else PENDING


class Transport(Protocol):
    def enqueue(self, graph_file: Path | None, priority: int, check: str, schema: str | None = None) -> None:
        """``check`` on the queue at ``priority``: ``graph_file``'s tasks, or with None the graph the executor lists, of
        ``schema`` when given (tasks/v1 when not)."""
        ...

    def graph(self, check: str, schema: str = TASKS_V1) -> tuple[str, dict] | None:
        """(its id, the ``schema`` graph the executor listed for ``check``) once it labelled one, the newest; None until then."""
        ...

    def labels(self, keys: list[str]) -> dict[str, list[str]]:
        """{key: the values of the `verdict` labels on `task:<key>`, oldest first}."""
        ...

    def results(self, keys: list[str]) -> dict[str, list[object]]:
        """{key: the values of the `result` labels on `task:<key>`, result/v1 documents}."""
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

    def enqueue_graph(self, check: str, priority: int) -> None:
        """Put ``check`` (`after-landing:SHA`) on the queue at ``priority`` with no graph, as graph/v2: the executor lists it on
        a worker, and `graph_v2` reads what it listed."""
        self.transport.enqueue(None, priority, check, GRAPH_V2)

    def graph_v2(self, check: str) -> tuple[str, dict] | None:
        """(its id, the graph/v2 document) the executor listed for ``check``; None until it labelled one.  One that isn't
        graph/v2, is of another commit, or has a task whose key `settle` couldn't compute, is refused (ValueError)."""
        if (got := self.transport.graph(check, GRAPH_V2)) is None:
            return None
        gid, g = got
        if not isinstance(g, dict) or g.get("schema") != GRAPH_V2 or not isinstance(g.get("tasks"), list):
            raise ValueError(f"the executor's graph {gid} of {check} is no {GRAPH_V2} graph")
        if g.get("commit") != check.partition(":")[2]:
            raise ValueError(f"the executor's graph {gid} of {check} is of commit {g.get('commit')}")
        files, envs = g.get("files") or {}, g.get("envs") or {}
        ids = {t.get("id") for t in g["tasks"] if isinstance(t, dict)}
        for t in g["tasks"]:
            if not (isinstance(t, dict) and V2_FIELDS <= t.keys() and t["env"] in envs and set(t["inputs"]) <= files.keys()
                    and all(isinstance(r, dict) and r.get("task") in ids for r in t["reads"])):
                raise ValueError(f"the executor's graph {gid} of {check} has a task with no key to compute: "
                                 f"{t.get('id') if isinstance(t, dict) else repr(t)}")
        return gid, g

    def settle(self, graph: dict, keyed: dict[str, str] | None = None) -> dict[str, tuple[str, dict | None]]:
        """{task id: (its state, its result)} of graph/v2 ``graph``, and into ``keyed`` when given, each task's key, the graph's
        or the one made from what it reads.  A task with a key in the graph is pass, fail or pending by
        its `result` labels (`fold_results`).  A task that reads others is blocked once one of them failed, is blocked, or
        passed without the output it reads; pending while one is; and once each passed, keyed by `task_key` from their
        results' `outputs` and read as the others are.  Read in rounds, each of the keys the last one made known, a round
        taking the graph's tasks in its order, so one settled there is seen by those after it; a task whose reads never settle
        (a task the graph lacks) stays pending.  A task is looked at only once what it reads can settle it, so the cost is the
        graph's reads, not its reads times its rounds: a Lean graph reads each module's transitive imports, millions of reads
        over hundreds of rounds."""
        tasks = {t["id"]: t for t in graph["tasks"]}
        ids = list(tasks)
        order = {i: n for n, i in enumerate(ids)}
        files, envs = graph.get("files") or {}, graph.get("envs") or {}
        out: dict[str, tuple[str, dict | None]] = {}
        keys = {i: t["key"] for i, t in tasks.items() if t.get("key")}
        #: of each task that settles by what it reads: how many of its reads haven't settled, and whether a settled one
        #: blocks it or leaves it pending
        unread = {i: len(t["reads"]) for i, t in tasks.items() if i not in keys}
        blocked: set[str] = set()
        pending: set[str] = set()
        readers: dict[str, list[tuple[str, dict]]] = {}
        for i in unread:
            for r in tasks[i]["reads"]:
                readers.setdefault(r["task"], []).append((i, r))
        fetch = dict(keys)
        later = {i for i, n in unread.items() if not n}
        now: list[int] = []
        at = -1

        def settled(j: str) -> None:
            """``j`` settled: each task that reads it and can now settle is looked at this round if it comes after ``j``,
            else in the next."""
            s = out[j]
            for i, r in readers.get(j, ()):
                if i in out or i in keys:
                    continue
                unread[i] -= 1
                if s[0] in (FAIL, BLOCKED) or s[0] == PASS and r["output"] not in (s[1].get("outputs") or {}):
                    blocked.add(i)
                elif s[0] == PENDING:
                    pending.add(i)
                if i in blocked or i in pending or not unread[i]:
                    if order[i] > at:
                        heapq.heappush(now, order[i])
                    else:
                        later.add(i)

        while fetch or later:
            if fetch:
                got = self.transport.results(list(dict.fromkeys(fetch.values())))
                out.update({i: fold_results(got.get(k) or (), k) for i, k in fetch.items()})
                for i in fetch:
                    settled(i)
                fetch = {}
            for i in later:
                heapq.heappush(now, order[i])
            later = set()
            while now:
                i = ids[at := heapq.heappop(now)]
                if i in out or i in keys:
                    continue
                if i in blocked or i in pending:
                    out[i] = (BLOCKED if i in blocked else PENDING, None)
                    settled(i)
                else:
                    t = tasks[i]
                    keys[i] = fetch[i] = task_key(t, files, envs[t["env"]]["digest"],
                                                  {r["task"]: out[r["task"]][1]["outputs"] for r in t["reads"]})
            at = -1
        if keyed is not None:
            keyed.update(keys)
        return {i: out.get(i, (PENDING, None)) for i in tasks}


class FileTransport:
    """The executor as a directory: `enqueue` appends {check, priority, graph} to `enqueued.jsonl` (graph None when enqueued
    without one; with its schema when given), the graph of a schema the executor listed for a check is
    `graphs/<schema>/<check>.json` (`list_graph` writes one, as the executor's worker would), a key's verdicts are the lines
    {key, verdict} of `verdicts.jsonl` (`finish` writes one), and its results the lines {key, result} of `results.jsonl`
    (`finish_result`)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def enqueue(self, graph_file: Path | None, priority: int, check: str, schema: str | None = None) -> None:
        graph = json.loads(graph_file.read_text()) if graph_file is not None else None
        with (self.root / "enqueued.jsonl").open("a") as f:
            f.write(json.dumps({"check": check, "priority": priority, "graph": graph, **({"schema": schema} if schema else {})})
                    + "\n")

    def enqueued(self) -> list[dict]:
        p = self.root / "enqueued.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else []

    def _graph_file(self, check: str, schema: str) -> Path:
        return self.root / "graphs" / schema.replace("/", "_") / f"{check.replace(':', '_')}.json"

    def list_graph(self, check: str, graph: dict) -> None:
        p = self._graph_file(check, graph.get("schema") or TASKS_V1)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(graph))

    def graph(self, check: str, schema: str = TASKS_V1) -> tuple[str, dict] | None:
        p = self._graph_file(check, schema)
        return (str(p), json.loads(p.read_text())) if p.is_file() else None

    def finish(self, key: str, verdict: str) -> None:
        with (self.root / "verdicts.jsonl").open("a") as f:
            f.write(json.dumps({"key": key, "verdict": verdict}) + "\n")

    def finish_result(self, result: dict) -> None:
        with (self.root / "results.jsonl").open("a") as f:
            f.write(json.dumps({"key": result["key"], "result": result}) + "\n")

    def _lines(self, name: str, field: str, keys: list[str]) -> dict[str, list]:
        p, want, out = self.root / name, set(keys), {}
        for line in p.read_text().splitlines() if p.is_file() else ():
            rec = json.loads(line)
            if rec["key"] in want:
                out.setdefault(rec["key"], []).append(rec[field])
        return out

    def labels(self, keys: list[str]) -> dict[str, list[str]]:
        return self._lines("verdicts.jsonl", "verdict", keys)

    def results(self, keys: list[str]) -> dict[str, list[object]]:
        return self._lines("results.jsonl", "result", keys)


class QueueTransport:
    """infra's queue: a check goes in through infra's enqueue command (``command``, drive.toml's `executor_enqueue`: research
    imports nothing from `infra/`), run with `--priority N --check ID` and, for a graph drive listed, `--graph FILE` (a tasks/v1
    document, each task's `key` the one its verdict is labelled under), when it inserts one row {key, task, priority, check} per
    task and exits 0 once they are in; with no `--graph`, it exits 0 at once and the executor lists the graph on a worker, a
    graph/v2 with `--schema verity-graph/v2`, else tasks/v1.  The graph the executor listed for a check comes back as the
    artifact a label on `commit:<sha>` names (`LISTED`: `tasks_graph` a `tasks-graph/v1`, `verity_graph` a `verity-graph/v2`),
    verdicts as `verdict` labels on `task:<key>` and results as `result` labels there.  A key's verdicts are pulled from the
    store's remote while it isn't passed here (a fail here may since have been relabelled error there, and an error rerun), its
    results while it has none here yet.  With ``by``, only labels written by it count (`research data label --by`)."""

    #: the writer whose labels count, anyone's when None
    by: str | None = None

    def __init__(self, store, command: tuple[str, ...] = (), cwd: Path | None = None):
        self.store, self.command, self.cwd = store, tuple(command), cwd

    def _labels(self, target: str, key: str) -> list:
        return [lab for lab in self.store.labels(target, key=key) if self.by is None or lab.by == self.by]

    def enqueue(self, graph_file: Path | None, priority: int, check: str, schema: str | None = None) -> None:
        if not self.command:
            raise RuntimeError("drive.toml names no `executor_enqueue` command: set it to infra's enqueue command, the words "
                               "of its argv")
        graph = ["--graph", str(graph_file)] if graph_file is not None else []
        r = subprocess.run([*self.command, *graph, "--priority", str(priority), "--check", check,
                            *(["--schema", schema] if schema else [])],
                           cwd=self.cwd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode:
            raise RuntimeError(f"{' '.join(self.command)} exited {r.returncode}: {(r.stderr or r.stdout).strip()[-400:]}")

    def graph(self, check: str, schema: str = TASKS_V1) -> tuple[str, dict] | None:
        from .store import labels_sync

        label, want = LISTED[schema]
        target = f"commit:{check.partition(':')[2]}"
        if not (labs := self._labels(target, label)) and getattr(self.store, "remote", None) is not None:
            labels_sync.pull_targets(self.store, [target])
            labs = self._labels(target, label)
        if not labs:
            return None
        art = labs[-1].value
        if (kind := self.store.get_manifest(art).kind) != want:
            raise ValueError(f"{target}'s {label} label names {art}, a {kind}, not a {want}")
        return art, json.loads(Path(self.store.fetch(art)).read_text())

    def _values(self, keys: list[str], label: str) -> dict[str, list]:
        from .queue import PREFETCH_JOBS
        from .store import labels_sync

        have = {k: [lab.value for lab in self._labels(task_target(k), label)] for k in keys}
        stale = (lambda v: fold(v) != PASS) if label == vocab.TASK_VERDICT else (lambda v: not v)
        if (todo := [k for k, v in have.items() if stale(v)]) and getattr(self.store, "remote", None) is not None:
            labels_sync.pull_targets(self.store, [task_target(k) for k in todo], jobs=PREFETCH_JOBS)
            have.update({k: [lab.value for lab in self._labels(task_target(k), label)] for k in todo})
        return have

    def labels(self, keys: list[str]) -> dict[str, list[str]]:
        return self._values(keys, vocab.TASK_VERDICT)

    def results(self, keys: list[str]) -> dict[str, list[object]]:
        return self._values(keys, vocab.TASK_RESULT)


# -- the shadow comparison: an after-landing run's verdicts beside the executor's, per task where the run records one, else per
#    job (a run records circuit-check, the builds and the other whole steps once, however many tasks the graph cuts them into)

def old_verdict(status: str | None) -> str | None:
    return PASS if status in OLD_PASS else FAIL if status in OLD_FAIL else None


def old_verdicts(run_dirs: Iterable[Path]) -> dict[str, str]:
    """{part: pass | fail} of an after-landing run's records, over its shards: each step (steps.jsonl), each suite as its task
    id (`pytest:NAME`, `lean-suites:NAME`), each Lake package its lean-audit log names (`lean-audit:PKG`).  A part failed in any
    shard fails; one stopped or never run is absent, and so are the preflight steps, which no after-landing graph holds
    (`tasks.py` runs a part without them), and a suite a step skipped by name (`SUITE_NO_VERDICT`)."""
    from .drive import PREFLIGHT, step_records, suite_rows

    got: dict[str, set[str]] = {}

    def put(part: str, v: str | None) -> None:
        if v is not None:
            got.setdefault(part, set()).add(v)

    for d in run_dirs:
        for st in step_records(d):
            if st.get("step") in PREFLIGHT:
                continue
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
            if s.get("status") not in SUITE_NO_VERDICT:
                put(f"{s['step']}:{name}", old_verdict(s.get("status")))
    return {p: FAIL if FAIL in vs else PASS for p, vs in got.items() if p}


def compare(ids: dict[str, str], old: dict[str, str], new: dict[str, str]) -> dict:
    """How the run's verdicts (``old``, `old_verdicts`) and the executor's (``new``, by key, of the tasks ``ids`` {task id:
    key}) compare, over the union of both sides' parts:

    - each task the run records on its own (`pytest:SUITE`, `lean-audit:PKG`, a job run whole): its verdict beside the task's;
    - each job the run records as a step: the step's verdict beside its tasks' folded (fail when one fails, pending when one
      is, else pass), so a task of a job the run records only whole (a circuit-check slice) is compared through it;
    - each suite the run records whole that the graph splits by test file (`SPLIT_RE`): its verdict beside its parts' folded;
    - a part the run records that no task matches, by id or by job: "missing in executor" (`MISSING`), a disagreement;
    - a task the run has no counterpart of (its id unrecorded, and its job unrecorded or recorded part by part without it):
      "executor only", a disagreement whatever its verdict.

    So a graph smaller than the run never agrees fully, nor one that ran something the run didn't."""
    units: dict[str, tuple[str, str]] = {}
    jobs: dict[str, list[str]] = {}
    split: dict[str, list[str]] = {}
    only = []
    by_part = {p.partition(":")[0] for p in old if ":" in p}
    for tid, key in ids.items():
        v, job = new.get(key, PENDING), tid.partition(":")[0]
        jobs.setdefault(job, []).append(v)
        if tid in old:
            units[tid] = (old[tid], v)
        elif (whole := SPLIT_RE.sub("", tid)) != tid and whole != job and whole in old:
            split.setdefault(whole, []).append(v)
        elif tid == job or job in by_part or job not in old:
            units[tid] = (NONE, v)
            only.append(tid)

    def folded(vs: list[str]) -> str:
        return FAIL if FAIL in vs else PENDING if PENDING in vs else PASS

    for job, vs in jobs.items():
        if job in old:
            units[job] = (old[job], folded(vs))
    units.update({whole: (old[whole], folded(vs)) for whole, vs in split.items()})
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

