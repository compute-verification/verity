"""The queue's side of the task executor: a check put on the executor's queue, and each task's verdict read back by its key.

    Executor(transport).enqueue(graph, priority)       -> {task id: key}
    Executor(transport).enqueue_check(check, priority) -> None: the executor lists the check's tasks/v1 graph on a worker
    Executor(transport).graph(check)                   -> (its art: id, {task id: key}) once labelled, else None
    Executor(transport).verdicts(keys)                 -> {key: pass | fail | pending}
    Executor(transport).enqueue_graph(check, priority) -> None: the executor lists the check's graph/v2 on a worker
    Executor(transport).graph_v2(check)                -> (its art: id, the graph/v2 document) once labelled, else None
    Executor(transport).settle(graph)                  -> {task id: (pass | fail | pending | blocked, its result or None)}

Two graph schemas.  `tasks/v1` (`verity tasks list --json`, `tools/check/tasks.py`), and graph/v2 (`verity graph`,
`tools/verity/graph`), which the verified queue's loop verifies a candidate by (`vqueue_loop`, graph2).

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

A graph/v2 check is enqueued with `--schema verity-graph/v2` and no graph: the executor lists
graph/v2 on a worker, puts it in the store (`verity-graph/v2`) and labels `commit:<sha> verity_graph art:<id>`
(`vocab.VERITY_GRAPH`), and each task's result/v1 document is a `result` label on `task:<key>` (`vocab.TASK_RESULT`; an error is
never a result, so only pass and fail appear).  A graph/v2 task that reads none has its key in the graph; one that reads others
has none there, and `settle` computes it by the contract's key rule (`task_key`, held to
`tests/tools/verity/graph/key_vectors.json`) once each task it reads passed, from their results' `outputs`; it is blocked, never
run, once one of them failed or is blocked.  The loop reads them as a candidate's verdict (`vqueue_loop`)."""
from __future__ import annotations

import hashlib
import heapq
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Protocol

from .store import vocab

PASS, FAIL, PENDING = "pass", "fail", "pending"
#: the label of a task whose runs all ended without a result (the machine lost, the run refused or cancelled): no verdict of
#: the tree, so `fold` reads it as pending
ERROR = "error"
#: a graph/v2 task that never runs: a task it reads failed, is blocked, or passed without the output it reads
BLOCKED = "blocked"
#: the priority of an after-landing check on the queue, its highest
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
    imports nothing from `infra/`), run with `--priority N --check ID` and, for a graph listed here, `--graph FILE` (a tasks/v1
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
