"""The task executor's client (`research.executor`), on the file-backed fake transport: it enqueues a check's graph, keys each
task by the contract's rule (a task that reads others by the key the rule gives it), folds each key's verdicts and settles a
graph/v2 as rounds over the whole graph would, compares the executor's verdicts with a run's records, and `QueueTransport`
reads the executor's graphs and results through the store's remote."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.executor import (AFTER_LANDING_PRIORITY, BLOCKED, GRAPH_V2, TASKS_GRAPH_KIND, Executor, FileTransport, QueueTransport,
                               Shadowed, agrees, compare, fold_results, keyed, old_verdicts, shadow_report, task_key, task_target)
from research.landing import AFTER_LANDING
from research.store import vocab
from research.store.local import LocalStore

K_SUITE, K_LEAN = "a" * 128, "b" * 128
#: the jobs the graph lists whole, unkeyed (a build, or a job that keeps no verdict of its own)
WHOLE = tuple(j for j in AFTER_LANDING if j not in ("pytest", "lean-audit", "flock-rust-tests", "circuit-check"))
N_TASKS = 5 + len(WHOLE)


def _graph(tip: str) -> dict:
    """A tip's tasks/v1 graph: a suite and a Lake package keyed by their inputs (the same at every tip here), a job keyed by
    the tip, circuit-check's two slices, unkeyed, and every other after-landing job whole, unkeyed."""
    return {"schema": "tasks/v1", "commit": tip, "kind": "after-landing", "tasks": [
        {"id": "pytest:pkg", "key": K_SUITE}, {"id": "lean-audit:Security", "key": K_LEAN},
        {"id": "flock-rust-tests", "key": tip[:32] * 4},
        {"id": "circuit-check:1/2", "key": None, "unkeyed": "per target"},
        {"id": "circuit-check:2/2", "key": None, "unkeyed": "per target"},
        *({"id": j, "key": None, "unkeyed": "no verdict of its own"} for j in WHOLE)]}


def _keys(fake: FileTransport, i: int) -> dict[str, str]:
    return {t["id"]: t["key"] for t in fake.enqueued()[i]["graph"]["tasks"]}


def test_the_executor_enqueues_a_graph_with_a_key_for_each_task_and_folds_each_keys_verdicts(tmp_path):
    fake = FileTransport(tmp_path)
    ex = Executor(fake)
    tip, other = "1" * 40, "2" * 40
    keys = ex.enqueue(_graph(tip), AFTER_LANDING_PRIORITY)
    (row,) = fake.enqueued()
    assert row["check"] == f"after-landing:{tip}" and row["priority"] == 1
    assert keys == _keys(fake, 0) and keys["pytest:pkg"] == K_SUITE and len(set(keys.values())) == N_TASKS
    assert keyed(_graph(other))["tasks"][3]["key"] != keys["circuit-check:1/2"], "an unkeyed task's key is its check's alone"
    assert keyed(_graph(other))["tasks"][3]["unkeyed"] == "per target"
    fake.finish(K_SUITE, "pass")
    fake.finish(K_LEAN, "pass")
    fake.finish(K_LEAN, "fail")
    assert ex.verdicts([K_SUITE, K_LEAN, keys["flock-rust-tests"]]) == {K_SUITE: "pass", K_LEAN: "fail",
                                                                      keys["flock-rust-tests"]: "pending"}


def test_error_is_not_run_and_clears_the_fail_before_it(tmp_path):
    """A false fail (a run that gave no result, labelled fail) written over as error reads as not run, and the key's rerun
    that passes is its verdict; error never reads as pass or fail."""
    from research.executor import ERROR

    fake = FileTransport(tmp_path)
    ex = Executor(fake)
    for v in ("fail", ERROR):
        fake.finish(K_SUITE, v)
    fake.finish(K_LEAN, ERROR)
    assert ex.verdicts([K_SUITE, K_LEAN]) == {K_SUITE: "pending", K_LEAN: "pending"}
    fake.finish(K_SUITE, "pass")
    fake.finish(K_LEAN, "fail")
    assert ex.verdicts([K_SUITE, K_LEAN]) == {K_SUITE: "pass", K_LEAN: "fail"}
    assert ERROR in vocab.TASK_VERDICTS, "the store takes the executor's error label"


def test_a_fail_stands_when_a_rerun_passes():
    """A flake must not go green on a rerun: a fail with no error label after it is the key's verdict whatever came later,
    and only an error written after it (a false fail relabelled) lets a later pass count."""
    from research.executor import ERROR, fold

    assert fold(["fail", "pass"]) == "fail"
    assert fold(["pass", "fail", "pass"]) == "fail"
    assert fold(["fail", ERROR, "pass"]) == "pass"
    assert fold(["fail", ERROR, "pass", "fail", "pass"]) == "fail"
    assert fold(["pass", ERROR]) == "pending"
    assert fold(["fail", "pass", ERROR]) == "pending"
    assert fold([]) == "pending"


def test_the_queue_transport_pulls_every_key_not_passed_here(monkeypatch):
    """A key failed here may since have been written over as error or rerun on the remote, so only a passed key (which the
    executor never reruns) is read from the local store alone."""
    from types import SimpleNamespace

    from research.executor import ERROR, QueueTransport
    from research.store import labels_sync

    new = "c" * 128
    here = {f"task:{K_SUITE}": ["pass"], f"task:{K_LEAN}": ["fail"]}
    remote = {f"task:{K_LEAN}": ["fail", ERROR], f"task:{new}": ["pass"]}

    class Store:
        def __init__(self):
            self.remote = object()

        def labels(self, target, key=None):
            return [SimpleNamespace(value=v) for v in here.get(target, [])]

    pulled = []
    monkeypatch.setattr(labels_sync, "pull_targets",
                        lambda store, targets, jobs: pulled.extend(targets) or here.update({t: remote[t] for t in targets}))
    assert Executor(QueueTransport(Store())).verdicts([K_SUITE, K_LEAN, new]) == {K_SUITE: "pass", K_LEAN: "pending", new: "pass"}
    assert pulled == [f"task:{K_LEAN}", f"task:{new}"]


def test_the_comparison_is_per_task_where_the_run_records_one_and_per_job_where_it_records_the_job_whole():
    ids = {"pytest:pkg": "k1", "circuit-check:1/2": "k2", "circuit-check:2/2": "k3", "lean-unit-cut": "k4"}
    old = {"pytest": "pass", "pytest:pkg": "pass", "circuit-check": "fail", "lean-unit-cut": "pass"}
    got = compare(ids, old, {"k1": "pass", "k2": "pass", "k3": "fail", "k4": "pass"})
    assert got == {"units": 4, "agree": 4, "pending": [], "disagree": [], "missing": [], "executor_only": []}
    assert agrees(got)


def test_a_graph_missing_a_part_the_run_has_never_agrees_fully():
    ids = {"pytest:pkg": "k1", "lean-unit-cut": "k4"}
    old = {"pytest": "pass", "pytest:pkg": "pass", "pytest:other": "pass", "lean-unit-cut": "pass", "ratchet-gate": "pass"}
    got = compare(ids, old, {"k1": "pass", "k4": "pass"})
    assert got["missing"] == ["pytest:other", "ratchet-gate"] and got["executor_only"] == [] and not agrees(got)
    assert {"unit": "ratchet-gate", "old": "pass", "executor": "missing"} in got["disagree"] and got["agree"] == 3
    text, ok = shadow_report([Shadowed("1" * 40, "2026-11-01T01:00:00Z", got)] * 3, 10)
    assert not ok and "missing in executor 2" in text and "    ratchet-gate: missing in executor (run pass)" in text, text


def test_an_executor_only_task_is_listed_and_disagrees():
    ids = {"pytest:pkg": "k1", "pytest:new": "k2", "lean-unit-cut": "k3", "circuit-check:1/1": "k4"}
    old = {"pytest": "pass", "pytest:pkg": "pass", "circuit-check": "pass"}
    got = compare(ids, old, {"k1": "pass", "k2": "pass", "k3": "pending", "k4": "pass"})
    # pytest's suites are recorded one by one, so pytest:new has no counterpart; lean-unit-cut the run didn't run at all
    assert got["executor_only"] == ["lean-unit-cut", "pytest:new"] and got["missing"] == [] and not agrees(got)
    assert {"unit": "lean-unit-cut", "old": "none", "executor": "pending"} in got["disagree"] and got["pending"] == []
    text, _ = shadow_report([Shadowed("1" * 40, "2026-11-01T01:00:00Z", got)], 10)
    assert "executor only 2" in text and "    pytest:new: executor only (executor pass)" in text, text
    assert not agrees({k: v for k, v in compare({"a": "k"}, {"a": "pass"}, {"k": "pass"}).items()
                       if k not in ("missing", "executor_only")}), "a record without coverage counts never agrees"


def test_the_runs_records_read_as_the_graph_cuts_them(tmp_path):
    """A suite reused from the cache passes; one a step skipped by name, and the preflight, have no verdict; a suite the graph
    splits by file is compared whole through its parts."""
    d = tmp_path / "r"
    d.mkdir()
    (d / "steps.jsonl").write_text(json.dumps({"step": "preflight-lock", "status": "passed"}) + "\n"
                                   + json.dumps({"step": "pytest", "status": "failed"}) + "\n")
    (d / "suites.json").write_text(json.dumps([{"name": "kept", "status": "cached"}, {"name": "lean", "status": "skipped"},
                                               {"name": "big", "status": "failed"}]))
    old = old_verdicts([d])
    assert old == {"pytest": "fail", "pytest:kept": "pass", "pytest:big": "fail"}, old
    ids = {"pytest:kept": "k1", "pytest:big:1/2": "k2", "pytest:big:2/2": "k3"}
    got = compare(ids, old, {"k1": "pass", "k2": "pass", "k3": "fail"})
    assert got == {"units": 3, "agree": 3, "pending": [], "disagree": [], "missing": [], "executor_only": []}, got
    assert agrees(got)
    assert compare(ids, old, {"k1": "pass", "k2": "pass"})["pending"] == ["pytest", "pytest:big"]


def test_shadow_reports_the_switch_gate_once_three_tips_in_a_row_agree():
    full = {"units": 2, "agree": 2, "pending": [], "disagree": [], "missing": [], "executor_only": []}
    off = {"units": 2, "agree": 1, "pending": [], "disagree": [{"unit": "x", "old": "pass", "executor": "fail"}],
           "missing": [], "executor_only": []}
    recs = [Shadowed(f"{i}" * 40, f"2026-11-01T0{i}:00:00Z", r) for i, r in ((4, full), (3, full), (2, full), (1, off))]
    assert shadow_report(recs, 2)[1] and "the newest 3 agree" in shadow_report(recs, 2)[0]
    assert not shadow_report(recs[:2] + recs[3:], 10)[1]


#: the graph/v2 envs of the fakes here; only their digests enter a key
ENVS = {"py": {"spec": ["uv.lock"], "digest": "sha512:" + "e" * 128},
        "lean": {"spec": ["lean-toolchain"], "digest": "sha512:" + "f" * 128}}


X, A, B, FP8_TASK = "test-file:pkg/tests/test_x.py", "lean-module:A.lean", "lean-module:B.lean", "circuit-binding:fp8"


LS, L_TEST = "lean-suites:after-lean", "tests/test_l.py::test_z"


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


#: a task of each other after-landing job, its kind the job's name, so that a graph has a task of every job `main` waits for
WHOLE_V2 = tuple(_task(f"{j}:all", ()) for j in AFTER_LANDING if j not in ("pytest", "lean-audit", "circuit-check"))


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
    """The key drive computes for a task that reads others is the one `verity graph` and the executor compute
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
    fake.list_graph(f"after-landing:{tip}", {**g, "files": {p: d for p, d in g["files"].items() if p != "B.lean"}})
    with pytest.raises(ValueError, match=f"has a task with no key to compute: {B}"):
        ex.graph_v2(f"after-landing:{tip}")


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
    """Results dealt by key: each key's documents generated from ``seed`` and the key alone (none, an error, a pass with or without
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


def test_the_queue_transport_enqueues_a_check_with_no_graph_and_reads_the_graph_the_executor_labelled(tmp_path):
    """Drive's store reads the executor's graph only through the remote: the executor's store put the `tasks-graph/v1`
    artifact and its `tasks_graph` label on `commit:<sha>` there, and drive's pulls the label, then fetches the graph."""
    import sys

    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    executor_store = LocalStore(tmp_path / "executor-store", config_path=tmp_path / "store.toml")
    drive_store = LocalStore(tmp_path / "drive-store", config_path=tmp_path / "store.toml")
    seen, stub = tmp_path / "argv.txt", tmp_path / "enqueue.py"
    stub.write_text(f"import sys\nopen({str(seen)!r}, 'w').write(' '.join(sys.argv[1:]))\n")
    tip = "1" * 40
    ex = Executor(QueueTransport(drive_store, command=(sys.executable, str(stub))))
    ex.enqueue_check(f"after-landing:{tip}", AFTER_LANDING_PRIORITY)
    assert seen.read_text() == f"--priority 1 --check after-landing:{tip}"
    assert ex.graph(f"after-landing:{tip}") is None

    (f := tmp_path / "graph.json").write_text(json.dumps(_graph(tip)))
    art = executor_store.put_artifact(TASKS_GRAPH_KIND, {}, file=f)
    executor_store.push(art)
    executor_store.add_label(f"commit:{tip}", vocab.TASKS_GRAPH, art, "executor")
    executor_store.push_labels([f"commit:{tip}"])
    assert drive_store.labels(f"commit:{tip}") == []
    gid, keys = ex.graph(f"after-landing:{tip}")
    assert gid == art and keys == {t["id"]: t["key"] for t in keyed(_graph(tip))["tasks"]} and keys["pytest:pkg"] == K_SUITE

    (g := tmp_path / "other.json").write_text("{}")
    other = executor_store.put_artifact("fixture/v1", {}, file=g)
    executor_store.push(other)
    executor_store.add_label(f"commit:{'2' * 40}", vocab.TASKS_GRAPH, other, "executor")
    executor_store.push_labels([f"commit:{'2' * 40}"])
    with pytest.raises(ValueError, match=f"names {other}, a fixture/v1, not a tasks-graph/v1"):
        ex.graph(f"after-landing:{'2' * 40}")


def test_the_queue_transport_enqueues_graph_v2_and_reads_its_graph_and_results_through_the_remote(tmp_path):
    """`on`'s check goes in with `--schema verity-graph/v2`; the executor's store put the `verity-graph/v2` artifact and its
    `verity_graph` label on `commit:<sha>`, and each result/v1 as a `result` label on `task:<key>` without `sha512:`, and
    drive's store pulls them from the remote."""
    import sys

    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    executor_store = LocalStore(tmp_path / "executor-store", config_path=tmp_path / "store.toml")
    drive_store = LocalStore(tmp_path / "drive-store", config_path=tmp_path / "store.toml")
    seen, stub = tmp_path / "argv.txt", tmp_path / "enqueue.py"
    stub.write_text(f"import sys\nopen({str(seen)!r}, 'w').write(' '.join(sys.argv[1:]))\n")
    tip = "1" * 40
    check = f"after-landing:{tip}"
    ex = Executor(QueueTransport(drive_store, command=(sys.executable, str(stub))))
    ex.enqueue_graph(check, AFTER_LANDING_PRIORITY)
    assert seen.read_text() == f"--priority 1 --check {check} --schema {GRAPH_V2}"
    assert ex.graph_v2(check) is None
    with pytest.raises(RuntimeError, match="drive.toml names no `executor_enqueue` command"):
        Executor(QueueTransport(drive_store)).enqueue_graph(check, AFTER_LANDING_PRIORITY)

    g = _graph_v2(tip)
    (f := tmp_path / "graph.json").write_text(json.dumps(g))
    art = executor_store.put_artifact(GRAPH_V2, {}, file=f)
    executor_store.push(art)
    executor_store.add_label(f"commit:{tip}", vocab.VERITY_GRAPH, art, "executor")
    executor_store.push_labels([f"commit:{tip}"])
    assert ex.graph(check) is None, "a graph/v2 is no tasks/v1 graph"
    assert ex.graph_v2(check) == (art, g)

    by = {t["id"]: t for t in g["tasks"]}
    done = [(by[X], by[X]["key"], "fail"), (by[A], by[A]["key"], "pass"), (by[B], _reader_key(g, B, A), "pass")]
    for task, key, status in done:
        executor_store.add_label(task_target(key), vocab.TASK_RESULT, _result(task, key, status), "executor")
    executor_store.push_labels([task_target(k) for _, k, _ in done])
    got = ex.settle(g)
    assert {i: got[i][0] for i in (X, A, B, FP8_TASK)} == {X: "fail", A: "pass", B: "pass", FP8_TASK: "pending"}, got
    assert got[B][1] == _result(by[B], _reader_key(g, B, A))
