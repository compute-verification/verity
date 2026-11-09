"""The verified queue's side of the task executor (`research.executor`), on the file-backed fake transport and the store's
remote: a graph/v2 task that reads others is keyed by the contract's rule once each it reads passed, and blocked behind one that
failed; settling incrementally is the rounds over the whole graph; `QueueTransport` enqueues a check as graph/v2 and reads the
executor's graph and results through the store's remote."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.executor import BLOCKED, GRAPH_V2, Executor, FileTransport, QueueTransport, fold_results, task_key, task_target
from research.landing import JOBS
from research.store import vocab
from research.store.local import LocalStore

#: the graph/v2 envs of the fakes here; only their digests enter a key
ENVS = {"py": {"spec": ["uv.lock"], "digest": "sha512:" + "e" * 128},
        "lean": {"spec": ["lean-toolchain"], "digest": "sha512:" + "f" * 128}}
X, A, B, FP8_TASK = "test-file:pkg/tests/test_x.py", "lean-module:A.lean", "lean-module:B.lean", "circuit-binding:fp8"
LS = "lean-suites:after-lean"


def _digest(text: str) -> str:
    return "sha512:" + hashlib.sha512(text.encode()).hexdigest()


def _task(tid: str, inputs: tuple[str, ...], reads: tuple[str, ...] = (), out: bool = False, **vars) -> dict:
    """A graph/v2 task, its kind its id's prefix, in the lean env when it is Lean: ``reads`` the `olean` of each task named,
    ``out``, it builds one itself."""
    kind = tid.partition(":")[0]
    return {"id": tid, "kind": kind, "about": tid, "env": "lean" if kind.startswith("lean") else "py",
            "argv": ["verity", "graph", "run", tid], "vars": vars, "inputs": list(inputs),
            "reads": [{"task": r, "output": "olean"} for r in reads],
            "outputs": [{"name": "olean", "path": f"out/{tid.partition(':')[2]}.olean"}] if out else [],
            "resources": {"cores": 1, "mem_gb": 1, "disk_gb": 1}, "timeout_s": 60, "key": None}


def _v2(tip: str, *tasks: dict) -> dict:
    """``tip``'s graph/v2 of ``tasks``, each that reads none keyed by the contract's rule, as `verity graph` keys it."""
    files = {p: _digest(p) for t in tasks for p in t["inputs"]}
    return {"schema": GRAPH_V2, "commit": tip, "envs": ENVS, "files": files,
            "tasks": [{**t, "key": None if t["reads"] else task_key(t, files, ENVS[t["env"]]["digest"], {})} for t in tasks]}


#: a task of each other job of the check, its kind the job's name, so that a graph has a task of every job `main` waits for
WHOLE_V2 = tuple(_task(f"{j}:all", ()) for j in JOBS if j not in ("pytest", "lean-audit", "circuit-check"))


def _graph_v2(tip: str, whole: bool = True, lean_suites: tuple[str, ...] = ()) -> dict:
    """A test file of suite `pkg/`, Lean module B reading A's olean, a circuit-check binding, and (``whole``) `WHOLE_V2`; with
    ``lean_suites`` (the tree), its lean-suites task is `LS`, one coarse task of suites `tools/lsuite` and `other` reading it."""
    whole_v2 = [t for t in WHOLE_V2 if not (lean_suites and t["kind"] == "lean-suites")] if whole else []
    return _v2(tip, _task(X, ("pkg/tests/test_x.py", "pkg/x.py"), suite="pkg"), _task(A, ("A.lean",), out=True),
               _task(B, ("B.lean",), reads=(A,), out=True), _task(FP8_TASK, ("pkg/fp8.circuit-check.py",), binding="fp8"),
               *whole_v2, *([_task(LS, lean_suites, suites="tools/lsuite other")] if lean_suites else ()))


def _result(task: dict, key: str, status: str = "pass", **detail) -> dict:
    """A result/v1 of ``task`` under ``key``, each output's digest one of the key's own."""
    return {"schema": "verity-result/v1", "task": task["id"], "key": key, "status": status, "seconds": 1.0, "cpu_s": 1.0,
            "peak_rss_gb": 0.1, "outputs": {o["name"]: _digest(f"{key}/{o['name']}") for o in task["outputs"]},
            "undeclared_reads": [], "detail": detail, "log": None}


def _reader_key(g: dict, reader: str, producer: str) -> str:
    """The key the contract's rule gives ``reader`` once ``producer`` passed with `_result`'s outputs."""
    by = {t["id"]: t for t in g["tasks"]}
    p = by[producer]
    return task_key(by[reader], g["files"], ENVS[by[reader]["env"]]["digest"], {producer: _result(p, p["key"])["outputs"]})


def test_the_key_rule_reproduces_the_contracts_vectors():
    """The key the loop computes for a task that reads others is the one `verity graph` and the executor compute
    (`tools/verity/graph/tests/key_vectors.json`, which both sides of the contract reproduce)."""
    vectors = json.loads((Path(__file__).resolve().parents[3] / "tools/verity/graph/tests/key_vectors.json").read_text())
    files = {f["path"]: f["digest"] for f in vectors["files"]}
    assert len(vectors["keys"]) >= 10
    for v in vectors["keys"]:
        assert task_key(v["task"], files, v["env_digest"], v["read_digests"]) == v["key"], v["name"]
    assert task_target(v["key"]) == f"task:{v['key'].removeprefix('sha512:')}" and task_target("a" * 128) == f"task:{'a' * 128}"


def test_settle_keys_a_reader_once_what_it_reads_passed_and_blocks_it_behind_a_failure(tmp_path):
    tip = "1" * 40
    g = _v2(tip, _task(A, ("A.lean",), out=True), _task(B, ("B.lean",), reads=(A,), out=True),
            _task("lean-module:C.lean", ("C.lean",), reads=(B,)), _task("lean-module:D.lean", ("D.lean",)))
    by = {t["id"]: t for t in g["tasks"]}
    fake = FileTransport(tmp_path)
    ex = Executor(fake)
    assert {i: s for i, (s, _) in ex.settle(g).items()} == dict.fromkeys(by, "pending")
    fake.finish_result(_result(by[A], by[A]["key"]))
    fake.finish_result(_result(by[B], "sha512:" + "0" * 128))   # under a key that isn't B's: not B's result
    got = ex.settle(g)
    assert got[A] == ("pass", _result(by[A], by[A]["key"])) and got[B][0] == "pending" and got["lean-module:C.lean"][0] == "pending"
    kb = _reader_key(g, B, A)
    fake.finish_result(_result(by[B], kb, "fail"))
    got = ex.settle(g)
    assert got[B] == ("fail", _result(by[B], kb, "fail")) and got["lean-module:C.lean"] == (BLOCKED, None)
    assert got["lean-module:D.lean"] == ("pending", None)
    fake.finish_result(_result(by[B], kb))                       # a pass beside a fail: still failed
    assert ex.settle(g)[B][0] == "fail"
    other = FileTransport(tmp_path / "other")
    other.finish_result({**_result(by[A], by[A]["key"]), "outputs": {}})
    assert Executor(other).settle(g)[B] == (BLOCKED, None), "A passed without the olean B reads"
    errors = FileTransport(tmp_path / "errors")
    errors.finish_result({**_result(by[A], by[A]["key"]), "status": "error"})
    assert Executor(errors).settle(g)[A][0] == "pending", "an error is never a result"
    fake.list_graph(f"ready:{tip}", {**g, "files": {p: d for p, d in g["files"].items() if p != "B.lean"}})
    with pytest.raises(ValueError, match=f"has a task with no key to compute: {B}"):
        ex.graph_v2(f"ready:{tip}")


def _settle_by_rounds(transport, graph: dict, keyed: dict[str, str]) -> dict:
    """`Executor.settle` as each round looked at every task the graph has yet to settle, each task's key into ``keyed``: the
    reference its incremental form is held to, which costs a Lean graph's millions of reads times its hundreds of rounds."""
    tasks = {t["id"]: t for t in graph["tasks"]}
    files, envs = graph.get("files") or {}, graph.get("envs") or {}
    out, keys = {}, {i: t["key"] for i, t in tasks.items() if t.get("key")}
    fetch = dict(keys)
    while True:
        if fetch:
            got = transport.results(list(dict.fromkeys(fetch.values())))
            out.update({i: fold_results(got.get(k) or (), k) for i, k in fetch.items()})
            fetch = {}
        moved = False
        for i, t in tasks.items():
            if i in out or i in keys:
                continue
            reads = [(r, out.get(r["task"])) for r in t["reads"]]
            if any(s and (s[0] in ("fail", BLOCKED) or s[0] == "pass" and r["output"] not in (s[1].get("outputs") or {}))
                   for r, s in reads):
                out[i] = (BLOCKED, None)
            elif any(s and s[0] == "pending" for _, s in reads):
                out[i] = ("pending", None)
            elif all(s for _, s in reads):
                keys[i] = fetch[i] = task_key(t, files, envs[t["env"]]["digest"], {r["task"]: s[1]["outputs"] for r, s in reads})
            else:
                continue
            moved = True
        if not moved and not fetch:
            break
    keyed.update(keys)
    return {i: out.get(i, ("pending", None)) for i in tasks}


class _Dealt:
    """Results dealt by key: each key's documents drawn from ``seed`` and the key alone (none, an error, a pass with or without
    the olean its readers read, a fail, a pass beside a fail), and each batch `results` was asked, in order."""

    def __init__(self, seed: int):
        self.seed, self.asked = seed, []

    def results(self, keys: list[str]) -> dict[str, list[object]]:
        import random

        self.asked.append(list(keys))
        out = {}
        for k in keys:
            rng = random.Random(f"{self.seed}:{k}")
            doc = {"key": k, "outputs": {"olean": _digest(k)}}
            out[k] = [[], [{**doc, "status": "error"}], [{**doc, "status": "pass"}], [{**doc, "status": "pass"}],
                      [{**doc, "status": "pass", "outputs": {}}], [{**doc, "status": "fail"}],
                      [{**doc, "status": "pass"}, {**doc, "status": "fail"}]][rng.randrange(7)]
        return out


@pytest.mark.parametrize("seed", range(300))
def test_settle_settles_each_task_as_rounds_over_the_whole_graph_do(seed):
    """Incremental settling is the same function as the rounds over the whole graph, the same results read in the same batches
    and the same keys made, each the one a task whose reads all passed takes from their outputs: on random graphs whose tasks
    read those before and after them, a task the graph lacks, one twice, an output no task makes, and a task keyed in the graph
    that reads others."""
    import random

    rng = random.Random(seed)
    n = rng.randrange(1, 30)
    names = [f"lean-module:M{k}.lean" if rng.random() < 0.7 else f"test-file:t/test_{k}.py" for k in range(n)]
    tasks = []
    for k, tid in enumerate(names):
        reads = [names[j] for j in rng.sample(range(n), rng.randrange(min(n, 6))) if j != k]
        if rng.random() < 0.1:
            reads.append("lean-module:Missing.lean")
        if reads and rng.random() < 0.1:
            reads.append(reads[0])
        t = _task(tid, (f"src/{k}",), reads=tuple(reads), out=True)
        if t["reads"] and rng.random() < 0.15:
            t["reads"][-1] = {**t["reads"][-1], "output": "ilean"}
        tasks.append(t)
    g = _v2("1" * 40, *tasks)
    for t in g["tasks"]:
        if t["reads"] and rng.random() < 0.1:
            t["key"] = _digest(f"{seed}/{t['id']}")
    new, old = _Dealt(seed), _Dealt(seed)
    made, ref = {}, {}
    states = Executor(new).settle(g, made)
    assert states == _settle_by_rounds(old, g, ref)
    assert new.asked == old.asked and made == ref
    for t in g["tasks"]:
        reads = [states[r["task"]] for r in t["reads"] if r["task"] in states]
        rule = t["key"] or (task_key(t, g["files"], ENVS[t["env"]]["digest"], {r["task"]: s[1]["outputs"] for r, s in
                                                                               zip(t["reads"], reads)})
                            if len(reads) == len(t["reads"]) and states[t["id"]][0] != BLOCKED
                            and all(s[0] == "pass" for s in reads) else "")
        assert made.get(t["id"], "") == rule, t["id"]


def test_the_queue_transport_enqueues_graph_v2_and_reads_its_graph_and_results_through_the_remote(tmp_path):
    """A check goes in with `--schema verity-graph/v2`; the executor's store put the `verity-graph/v2` artifact and its
    `verity_graph` label on `commit:<sha>`, and each result/v1 as a `result` label on `task:<key>` without `sha512:`, and
    the loop's store pulls them from the remote."""
    import sys

    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    executor_store = LocalStore(tmp_path / "executor-store", config_path=tmp_path / "store.toml")
    loop_store = LocalStore(tmp_path / "loop-store", config_path=tmp_path / "store.toml")
    seen, stub = tmp_path / "argv.txt", tmp_path / "enqueue.py"
    stub.write_text(f"import sys\nopen({str(seen)!r}, 'w').write(' '.join(sys.argv[1:]))\n")
    tip = "1" * 40
    check = f"ready:{tip}"
    ex = Executor(QueueTransport(loop_store, command=(sys.executable, str(stub))))
    ex.enqueue_graph(check, 1)
    assert seen.read_text() == f"--priority 1 --check {check} --schema {GRAPH_V2}"
    assert ex.graph_v2(check) is None

    g = _graph_v2(tip)
    (f := tmp_path / "graph.json").write_text(json.dumps(g))
    art = executor_store.put_artifact(GRAPH_V2, {}, file=f)
    executor_store.push(art)
    executor_store.add_label(f"commit:{tip}", vocab.VERITY_GRAPH, art, "executor")
    executor_store.push_labels([f"commit:{tip}"])
    assert ex.graph_v2(check) == (art, g)

    by = {t["id"]: t for t in g["tasks"]}
    done = [(by[X], by[X]["key"], "fail"), (by[A], by[A]["key"], "pass"), (by[B], _reader_key(g, B, A), "pass")]
    for task, key, status in done:
        executor_store.add_label(task_target(key), vocab.TASK_RESULT, _result(task, key, status), "executor")
    executor_store.push_labels([task_target(k) for _, k, _ in done])
    got = ex.settle(g)
    assert {i: got[i][0] for i in (X, A, B, FP8_TASK)} == {X: "fail", A: "pass", B: "pass", FP8_TASK: "pending"}, got
    assert got[B][1] == _result(by[B], _reader_key(g, B, A))

