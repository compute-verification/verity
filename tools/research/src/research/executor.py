"""The verified queue's side of the task executor (graph2): a check put on the executor's queue, its graph read back once
the executor listed it, and each task's result read back by its key.

    Executor(transport).enqueue_graph(check, priority) -> None: the executor lists the check's graph/v2 on a worker
    Executor(transport).graph_v2(check)                -> (its art: id, the graph/v2 document) once labelled, else None
    Executor(transport).settle(graph)                  -> {task id: (pass | fail | pending | blocked, its result or None)}

A graph's keys hash the lister's environment and its suite splits come from the lister's caches, so only a graph listed where
the tasks run has the keys their results come under: the executor lists a check enqueued with no graph on a worker, puts it in
the store (`verity-graph/v2`, `verity graph`, `tools/verity/graph`) and labels `commit:<sha> verity_graph art:<id>`
(`vocab.VERITY_GRAPH`), then queues its tasks, and each task's result/v1 document is a `result` label on `task:<key>`
(`vocab.TASK_RESULT`; an error is never a result, so only pass and fail appear).  A graph/v2 task that reads none has its key in
the graph; one that reads others has none there, and `settle` computes it by the contract's key rule (`task_key`, held to
`tools/verity/graph/tests/key_vectors.json`) once each task it reads passed, from their results' `outputs`; it is blocked,
never run, once one of them failed or is blocked.

The transport is how the checks reach the queue and the graphs and results come back: `QueueTransport`, infra's queue (its
enqueue command, and the labels and graphs from the store's remote), or `FileTransport`, a directory, for tests.  The verified
queue's loop (`research.vqueue_loop`) judges each candidate by these results, beside the run of what graph2 doesn't check
(`research.vqueue_slim`)."""
from __future__ import annotations

import hashlib
import heapq
import json
import subprocess
from pathlib import Path
from typing import Iterable, Protocol

from .store import vocab

PASS, FAIL, PENDING = "pass", "fail", "pending"
#: a graph/v2 task that never runs: a task it reads failed, is blocked, or passed without the output it reads
BLOCKED = "blocked"
#: graph/v2's schema (`verity_graph.contract.SCHEMA_GRAPH`), which is also the store kind of one the executor listed
GRAPH_V2 = "verity-graph/v2"
#: {schema: (the label on `commit:<sha>` naming the graph the executor listed of it, the graph's store kind)}
LISTED = {GRAPH_V2: (vocab.VERITY_GRAPH, GRAPH_V2)}
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


class Transport(Protocol):
    def enqueue(self, graph_file: Path | None, priority: int, check: str, schema: str | None = None) -> None:
        """``check`` on the queue at ``priority``: ``graph_file``'s tasks, or with None the graph the executor lists, of
        ``schema`` when given."""
        ...

    def graph(self, check: str, schema: str = GRAPH_V2) -> tuple[str, dict] | None:
        """(its id, the ``schema`` graph the executor listed for ``check``) once it labelled one, the newest; None until then."""
        ...

    def results(self, keys: list[str]) -> dict[str, list[object]]:
        """{key: the values of the `result` labels on `task:<key>`, result/v1 documents}."""
        ...


class Executor:
    def __init__(self, transport: Transport):
        self.transport = transport

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
    `graphs/<schema>/<check>.json` (`list_graph` writes one, as the executor's worker would), and a key's results are the lines
    {key, result} of `results.jsonl` (`finish_result`)."""

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
        p = self._graph_file(check, graph.get("schema") or GRAPH_V2)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(graph))

    def graph(self, check: str, schema: str = GRAPH_V2) -> tuple[str, dict] | None:
        p = self._graph_file(check, schema)
        return (str(p), json.loads(p.read_text())) if p.is_file() else None

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

    def results(self, keys: list[str]) -> dict[str, list[object]]:
        return self._lines("results.jsonl", "result", keys)


class QueueTransport:
    """infra's queue: a check goes in through infra's enqueue command (``command``: graph2's `enqueue`, as the verified queue's
    loop runs it; research imports nothing from `infra/`), run with `--priority N --check ID --schema verity-graph/v2`, which
    exits 0 at once while the executor lists the graph on a worker.  The graph the executor listed for a check comes back as
    the artifact a label on `commit:<sha>` names (`LISTED`: `verity_graph`, a `verity-graph/v2`), and results as `result`
    labels on `task:<key>`, pulled from the store's remote while a key has none here yet.  With ``by``, only labels written by
    it count (`research data label --by`)."""

    #: the writer whose labels count, anyone's when None
    by: str | None = None

    def __init__(self, store, command: tuple[str, ...] = (), cwd: Path | None = None):
        self.store, self.command, self.cwd = store, tuple(command), cwd

    def _labels(self, target: str, key: str) -> list:
        return [lab for lab in self.store.labels(target, key=key) if self.by is None or lab.by == self.by]

    def enqueue(self, graph_file: Path | None, priority: int, check: str, schema: str | None = None) -> None:
        if not self.command:
            raise RuntimeError("no enqueue command: give the transport infra's enqueue command, the words of its argv")
        graph = ["--graph", str(graph_file)] if graph_file is not None else []
        r = subprocess.run([*self.command, *graph, "--priority", str(priority), "--check", check,
                            *(["--schema", schema] if schema else [])],
                           cwd=self.cwd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode:
            raise RuntimeError(f"{' '.join(self.command)} exited {r.returncode}: {(r.stderr or r.stdout).strip()[-400:]}")

    def graph(self, check: str, schema: str = GRAPH_V2) -> tuple[str, dict] | None:
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
        if (todo := [k for k, v in have.items() if not v]) and getattr(self.store, "remote", None) is not None:
            labels_sync.pull_targets(self.store, [task_target(k) for k in todo], jobs=PREFETCH_JOBS)
            have.update({k: [lab.value for lab in self._labels(task_target(k), label)] for k in todo})
        return have

    def results(self, keys: list[str]) -> dict[str, list[object]]:
        return self._values(keys, vocab.TASK_RESULT)
