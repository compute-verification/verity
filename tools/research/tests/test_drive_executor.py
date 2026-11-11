"""Drive's side of the task executor (`research.executor`, drive.toml's `executor`), on the file-backed fake transport: off
changes nothing; shadow enqueues each checked tip's check with no graph beside its after-landing run, reads the graph the
executor listed, and records how their verdicts compare by its keys; on enqueues each tip as graph/v2, reads its tasks'
results by key (a task that reads others by the key the contract's rule gives it), judges a failure by policy.toml's
allowances, names the landing whose change touches what a red task depends on (a coarse lean-suites task's, per failed suite
as a run's), and moves `main` only once every task on the gate passed, never on a run that ran nothing, and once the run of
`SLIM_STEPS` it launches beside the check passed its preflights and its steps on the gate, whose red ratchet-gate, off it,
holds nothing and names its culprit as a run's.  `QueueTransport`
reads the executor's graphs and results through the store's remote."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import research.drive as D
from research import merge
from research.drive import AFTER_LANDING, AFTER_LANDING_OPTIONAL, SLIM_JUDGED, SLIM_STEPS, task_files
from research.executor import (AFTER_LANDING_PRIORITY, BLOCKED, GRAPH_V2, SHADOW_WAIT_S, TASKS_GRAPH_KIND, Executor, FileTransport,
                               QueueTransport, Shadowed, agrees, compare, fold_results, keyed, old_verdicts, shadow_records,
                               shadow_report, task_key, task_target)
from research.store.local import LocalStore
from research.store import vocab
from test_drive_policy import EXPIRED, LIVE, _policy
from test_drive_review import GATED_TREE, Review, _main, _move_main, _owning, _rings
from test_queue import World, _git

K_SUITE, K_LEAN = "a" * 128, "b" * 128
#: the preflights a check runs before any step (check.py's `preflight`), which the run beside the executor's check judges
PREFLIGHTS = tuple(s for s in SLIM_JUDGED if s not in SLIM_STEPS)
#: the jobs the graph lists whole, unkeyed (a build, or a job that keeps no verdict of its own)
WHOLE = tuple(j for j in AFTER_LANDING if j not in ("pytest", "lean-audit", "flock-rust-tests", "circuit-check"))
N_TASKS = 5 + len(WHOLE)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _graph(tip: str) -> dict:
    """A tip's tasks/v1 graph: a suite and a Lake package keyed by their inputs (the same at every tip here), a job keyed by
    the tip, circuit-check's two slices, unkeyed, and every other after-landing job whole, unkeyed."""
    return {"schema": "tasks/v1", "commit": tip, "kind": "after-landing", "tasks": [
        {"id": "pytest:pkg", "key": K_SUITE}, {"id": "lean-audit:Security", "key": K_LEAN},
        {"id": "flock-rust-tests", "key": tip[:32] * 4},
        {"id": "circuit-check:1/2", "key": None, "unkeyed": "per target"},
        {"id": "circuit-check:2/2", "key": None, "unkeyed": "per target"},
        *({"id": j, "key": None, "unkeyed": "no verdict of its own"} for j in WHOLE)]}


def _executor(h: Review, tmp_path, mode: str) -> FileTransport:
    fake = FileTransport(tmp_path / "executor")
    h.d.executor, h.d._tasks = mode, Executor(fake)
    return fake


def _checked(world: World) -> str:
    return _git(world.origin, "rev-parse", "refs/heads/checked")


def _land(h: Review, n: int) -> str:
    h.world.gh.put(n, h.world.change({f"f{n}.txt": f"{n}\n"}))
    h.review(n)
    assert h.tick().startswith(f"landed #{n}"), h.world.logs
    return _checked(h.world)


def _passed_run(h: Review, tmp_path) -> None:
    """The newest after-landing run passes every step, suite `pkg` and the Lake package `Security`."""
    rid = f"r-post-{len(h.posted) - 1}"
    h.run_state[rid] = ("done", 0, None)
    d = tmp_path / rid
    d.mkdir(parents=True)
    (d / "steps.jsonl").write_text("".join(json.dumps({"step": s, "status": "passed", **({"log": "lean-audit.log"}
                                                                                          if s == "lean-audit" else {})}) + "\n"
                                           for s in AFTER_LANDING))
    (d / "lean-audit.log").write_text("AUDIT /c/lean-audit-scratch-1-ab/Security: PASS  12 declarations\n")
    (d / "suites.json").write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "passed"}]))
    h.evidence = [d]


def _keys(fake: FileTransport, i: int) -> dict[str, str]:
    return {t["id"]: t["key"] for t in fake.enqueued()[i]["graph"]["tasks"]}


def _list(fake: FileTransport, tip: str, graph: dict | None = None) -> dict[str, str]:
    """The fake executor lists ``tip``'s graph on its worker (``graph``, else `_graph`); {task id: the key its verdict is under}."""
    fake.list_graph(f"after-landing:{tip}", g := graph or _graph(tip))
    return {t["id"]: t["key"] for t in keyed(g)["tasks"]}


def _graph_id(fake: FileTransport, tip: str) -> str:
    return fake.graph(f"after-landing:{tip}")[0]


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
    ids = {"pytest:pkg": "k1", "circuit-check:1/2": "k2", "circuit-check:2/2": "k3", "lean-changed": "k4"}
    old = {"pytest": "pass", "pytest:pkg": "pass", "circuit-check": "fail", "lean-changed": "pass"}
    got = compare(ids, old, {"k1": "pass", "k2": "pass", "k3": "fail", "k4": "pass"})
    assert got == {"units": 4, "agree": 4, "pending": [], "disagree": [], "missing": [], "executor_only": []}
    assert agrees(got)


def test_a_graph_missing_a_part_the_run_has_never_agrees_fully():
    ids = {"pytest:pkg": "k1", "lean-changed": "k4"}
    old = {"pytest": "pass", "pytest:pkg": "pass", "pytest:other": "pass", "lean-changed": "pass", "ratchet-gate": "pass"}
    got = compare(ids, old, {"k1": "pass", "k4": "pass"})
    assert got["missing"] == ["pytest:other", "ratchet-gate"] and got["executor_only"] == [] and not agrees(got)
    assert {"unit": "ratchet-gate", "old": "pass", "executor": "missing"} in got["disagree"] and got["agree"] == 3
    text, ok = shadow_report([Shadowed("1" * 40, "2026-11-01T01:00:00Z", got)] * 3, 10)
    assert not ok and "missing in executor 2" in text and "    ratchet-gate: missing in executor (run pass)" in text, text


def test_an_executor_only_task_is_listed_and_disagrees():
    ids = {"pytest:pkg": "k1", "pytest:new": "k2", "lean-changed": "k3", "circuit-check:1/1": "k4"}
    old = {"pytest": "pass", "pytest:pkg": "pass", "circuit-check": "pass"}
    got = compare(ids, old, {"k1": "pass", "k2": "pass", "k3": "pending", "k4": "pass"})
    # pytest's suites are recorded one by one, so pytest:new has no counterpart; lean-changed the run didn't run at all
    assert got["executor_only"] == ["lean-changed", "pytest:new"] and got["missing"] == [] and not agrees(got)
    assert {"unit": "lean-changed", "old": "none", "executor": "pending"} in got["disagree"] and got["pending"] == []
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


def test_off_changes_nothing(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "off")
    main = _main(world)
    tip = _land(h, 1)
    _passed_run(h, tmp_path)
    h.tick()
    assert _main(world) == tip != main and len(h.posted) == 1
    assert fake.enqueued() == [] and "executor" not in h.q.load_state()["drive"]
    assert world.store.labels(f"commit:{tip}") == []


def test_shadow_enqueues_each_tips_check_without_a_graph_and_compares_by_the_executors_graph(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "shadow")
    one = _land(h, 1)
    assert len(h.posted) == 1, "the run launches as before"
    assert fake.enqueued() == [{"check": f"after-landing:{one}", "priority": AFTER_LANDING_PRIORITY, "graph": None}]
    for k in _list(fake, one).values():
        fake.finish(k, "pass")
    h.tick()
    assert world.store.labels(f"commit:{one}") == [], "the run hasn't ended: nothing to compare yet"
    _passed_run(h, tmp_path / "1")
    h.tick()
    assert _main(world) == one                                  # the run alone moves `main`
    (rec,) = world.store.labels(f"commit:{one}", key=vocab.EXECUTOR_SHADOW)
    n = len(AFTER_LANDING) + 2                                  # every step, suite `pkg` and the Lake package `Security`
    assert rec.value == {"check": f"after-landing:{one}", "graph": _graph_id(fake, one), "run": "r-post-0", "units": n,
                         "agree": n, "pending": [], "disagree": [], "missing": [], "executor_only": []}
    assert one not in h.q.load_state()["drive"].get("executor", {})

    h.clock += 1                                                # the next record is the newer by drive's clock
    two = _land(h, 2)
    h.tick()
    keys = _list(fake, two)
    assert keys["pytest:pkg"] == K_SUITE and keys["flock-rust-tests"] != _list(fake, one)["flock-rust-tests"]
    for t, k in keys.items():
        if t not in ("pytest:pkg", "lean-audit:Security"):
            fake.finish(k, "fail" if t == "flock-rust-tests" else "pass")
    _passed_run(h, tmp_path / "2")
    h.tick()
    assert _main(world) == two                                  # a shadow's disagreement holds nothing
    (rec,) = world.store.labels(f"commit:{two}", key=vocab.EXECUTOR_SHADOW)
    assert rec.value["disagree"] == [{"unit": "flock-rust-tests", "old": "pass", "executor": "fail"}]
    assert rec.value["agree"] == n - 1
    text, ok = shadow_report(shadow_records(world.store, pull=False), 10)
    assert not ok and f"{two[:12]}" in text and "flock-rust-tests: run pass, executor fail" in text, text
    assert f"agree {n}/{n}, disagree 0" in text and "the newest 0 agree" in text
    assert [r["graph"] for r in fake.enqueued()] == [None, None] and all("schema" not in r for r in fake.enqueued())


def test_a_shadow_tip_is_pending_until_the_executor_labels_its_graph_then_agrees_by_its_keys(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "shadow")
    one = _land(h, 1)
    _passed_run(h, tmp_path)
    for _ in range(2):
        h.clock += SHADOW_WAIT_S // 3
        h.tick()
        assert world.store.labels(f"commit:{one}") == [], "no graph from the executor yet: the tip's shadow is pending"
        assert "keys" not in h.q.load_state()["drive"]["executor"][one]
    # the executor's worker lists the graph with keys of its own environment, none of them a key `_graph` has
    worker = {**_graph(one), "tasks": [{**t, "key": f"{i:x}" * 128} for i, t in enumerate(_graph(one)["tasks"][:3], 1)]
              + _graph(one)["tasks"][3:]}
    keys = _list(fake, one, worker)
    assert not {K_SUITE, K_LEAN} & set(keys.values())
    h.tick()
    assert world.store.labels(f"commit:{one}") == [] and h.q.load_state()["drive"]["executor"][one]["keys"] == keys
    for k in keys.values():
        fake.finish(k, "pass")
    fake.finish(K_SUITE, "fail")                                # a verdict under a key not in the executor's graph is unread
    h.tick()
    (rec,) = world.store.labels(f"commit:{one}", key=vocab.EXECUTOR_SHADOW)
    assert agrees(rec.value) and rec.value["graph"] == _graph_id(fake, one), rec.value


def test_a_graph_the_executor_never_labels_is_recorded_incomplete_after_the_wait(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, land_into="checked")
    _executor(h, tmp_path, "shadow")
    one = _land(h, 1)
    _passed_run(h, tmp_path)
    h.clock += SHADOW_WAIT_S
    h.tick()
    assert world.store.labels(f"commit:{one}") == [] and one in h.q.load_state()["drive"]["executor"]
    h.clock += 1
    h.tick()
    (rec,) = world.store.labels(f"commit:{one}", key=vocab.EXECUTOR_SHADOW)
    assert rec.value["incomplete"] == (f"the executor labelled no task graph of it (tasks_graph on commit:{one}) "
                                       f"{SHADOW_WAIT_S // 3600} h after its enqueue") and not agrees(rec.value)
    assert rec.value["check"] == f"after-landing:{one}" and rec.value["run"] == "r-post-0"
    assert one not in h.q.load_state()["drive"].get("executor", {}) and _main(world) == one
    text, ok = shadow_report(shadow_records(world.store, pull=False), 10)
    assert not ok and "incomplete: the executor labelled no task graph" in text, text


def test_a_graph_of_another_commit_is_refused_and_the_tip_recorded_incomplete(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "shadow")
    one = _land(h, 1)
    _passed_run(h, tmp_path)
    _list(fake, one, _graph("9" * 40))
    h.tick()
    (rec,) = world.store.labels(f"commit:{one}", key=vocab.EXECUTOR_SHADOW)
    assert rec.value["incomplete"].startswith("ValueError: the executor's graph ") and rec.value["incomplete"].endswith(
        f" of after-landing:{one} is of after-landing:{'9' * 40}"), rec.value


def test_a_shadow_tip_still_pending_is_recorded_pending_once_it_waited_long_enough(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "shadow")
    one = _land(h, 1)
    _list(fake, one)
    h.tick()
    _passed_run(h, tmp_path)
    fake.finish(K_SUITE, "pass")
    h.tick()
    assert world.store.labels(f"commit:{one}") == []
    h.clock += SHADOW_WAIT_S + 1
    h.tick()
    (rec,) = world.store.labels(f"commit:{one}", key=vocab.EXECUTOR_SHADOW)
    assert rec.value["pending"] == sorted({*AFTER_LANDING, "lean-audit:Security"} - {"pytest"}) and rec.value["agree"] == 2


def test_an_enqueue_that_fails_in_shadow_is_recorded_incomplete_and_the_tip_still_lands(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "shadow")

    def refused(graph_file, priority, check):
        raise RuntimeError("research executor enqueue exited 2: no command")

    fake.enqueue = refused
    one = _land(h, 1)
    assert len(h.posted) == 1
    (rec,) = world.store.labels(f"commit:{one}", key=vocab.EXECUTOR_SHADOW)
    assert rec.value["incomplete"] == "enqueueing its check: RuntimeError: research executor enqueue exited 2: no command"
    assert one not in h.q.load_state()["drive"].get("executor", {})


def test_a_transport_error_in_shadow_is_logged_marks_the_tip_incomplete_and_the_tick_still_lands(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "shadow")
    one = _land(h, 1)
    for k in _list(fake, one).values():
        fake.finish(k, "pass")
    h.tick()
    _passed_run(h, tmp_path)

    def down(keys):
        raise ConnectionError("the store's remote is down")

    fake.labels = down
    two = _land(h, 2)                                           # the tick that reads the verdicts lands #2 all the same
    assert _main(world) == one and two != one
    (rec,) = world.store.labels(f"commit:{one}", key=vocab.EXECUTOR_SHADOW)
    assert rec.value["incomplete"] == "ConnectionError: the store's remote is down" and rec.value["run"] == "r-post-0"
    assert not agrees(rec.value)
    assert any(f"{one[:12]}: failed (ConnectionError" in x and "recorded incomplete" in x for x in world.logs), world.logs
    assert one not in h.q.load_state()["drive"].get("executor", {})
    text, ok = shadow_report(shadow_records(world.store, pull=False), 10)
    assert not ok and "incomplete: ConnectionError" in text, text


def test_shadow_reports_the_switch_gate_once_three_tips_in_a_row_agree():
    full = {"units": 2, "agree": 2, "pending": [], "disagree": [], "missing": [], "executor_only": []}
    off = {"units": 2, "agree": 1, "pending": [], "disagree": [{"unit": "x", "old": "pass", "executor": "fail"}],
           "missing": [], "executor_only": []}
    recs = [Shadowed(f"{i}" * 40, f"2026-11-01T0{i}:00:00Z", r) for i, r in ((4, full), (3, full), (2, full), (1, off))]
    assert shadow_report(recs, 2)[1] and "the newest 3 agree" in shadow_report(recs, 2)[0]
    assert not shadow_report(recs[:2] + recs[3:], 10)[1]


# -- on: graph/v2, its results read by key

#: the graph/v2 envs of the fakes here; only their digests enter a key
ENVS = {"py": {"spec": ["uv.lock"], "digest": "sha512:" + "e" * 128},
        "lean": {"spec": ["lean-toolchain"], "digest": "sha512:" + "f" * 128}}
TEST = "tests/test_x.py::test_y"
FP8 = "partition/gate-recomputed ScaledMmFp8Block_v1"
X, A, B, FP8_TASK = "test-file:pkg/tests/test_x.py", "lean-module:A.lean", "lean-module:B.lean", "circuit-binding:fp8"
LS, L_TEST = "lean-suites:after-lean", "tests/test_l.py::test_z"
#: `GATED_TREE` with suite `tools/lsuite` (`l-suite`), which declares `shared` an input
L_TREE = {**GATED_TREE, "tools/lsuite/pyproject.toml": '[project]\nname = "L_Suite"\n[tool.verity.tests]\ninputs = ["shared"]\n'}


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


def _slim(h: Review, tmp_path, **logs: str) -> Path:
    """The newest run beside the executor's check (`r-post-N`, of `SLIM_STEPS` but one it skips) ended: each preflight and each
    of its steps passed, but each ``logs`` names (``ratchet_gate=LOG``), which failed with that log, its exit 1 then; after a
    failed preflight it ran no step, as check.py starts none."""
    rid, failed = f"r-post-{len(h.posted) - 1}", {k.replace("_", "-"): v for k, v in logs.items()}
    d = tmp_path / rid
    d.mkdir(parents=True)
    rows = []
    for s in (*PREFLIGHTS, *(() if failed.keys() & set(PREFLIGHTS) else h.posted[-1][1].split(","))):
        rows.append({"step": s, "status": "failed", "rc": 1, "log": f"{s}.log"} if s in failed else {"step": s, "status": "passed"})
        if s in failed:
            (d / f"{s}.log").write_text(failed[s])
    (d / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    h.run_state[rid], h.evidence = ("done", 1 if failed else 0, None), [d]
    return d


def _results(fake: FileTransport, g: dict, fails: dict[str, dict] | None = None, reader: str | None = "pass") -> None:
    """Each keyed task of ``g`` finished: those ``fails`` names failed with their detail, the rest passed; and B, which reads A,
    under the key A's result gives it (``reader``: its status, None for no result yet)."""
    fails = fails or {}
    for t in g["tasks"]:
        if t["key"]:
            fake.finish_result(_result(t, t["key"], "fail" if t["id"] in fails else "pass", **fails.get(t["id"], {})))
    if reader:
        fake.finish_result(_result(next(t for t in g["tasks"] if t["id"] == B), _reader_key(g, B, A), reader))


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
    assert task_files(g, ["lean-module:C.lean"]) == {"C.lean", "B.lean", "A.lean", "lean-toolchain"}
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


def test_on_enqueues_graph_v2_and_verifies_the_tip_once_every_task_passed_by_key(world, monkeypatch, tmp_path):
    """No run of every step: the tip's check goes on the queue as graph/v2 with no graph, beside a run of ratchet-gate alone
    (lean-agreement skipped: nothing it checks changed) with the base a run of every step takes, and the tip is verified, and
    `main` follows, once every task's result under its key passed, B's under the key A's result gives it, and that run
    passed: until then `main` waits for it."""
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    main = _main(world)
    tip = _land(h, 1)
    check = f"after-landing:{tip}"
    assert fake.enqueued() == [{"check": check, "priority": AFTER_LANDING_PRIORITY, "graph": None, "schema": GRAPH_V2}]
    assert h.posted == [(tip, "ratchet-gate", {"base": main, "agreement": False, "no_cache": False, "circuit_cache": False,
                                               "no_regenerate": True})]
    assert h.postland()["slim"] == {"ids": ["r-post-0"], "tip": tip, "at": h.clock, "steps": ["ratchet-gate"],
                                    "skip": ["lean-agreement"], "ratchet_base": main}
    h.tick()
    assert _main(world) == main and h.postland()["run"]["tasks"] == {"check": check}, "no graph labelled yet"
    g = _graph_v2(tip)
    fake.list_graph(check, g)
    _results(fake, g, reader=None)
    fake.finish_result(_result(g["tasks"][2], "sha512:" + "0" * 128))   # a result of B under a key that isn't its own
    h.tick()
    pl = h.postland()
    assert _main(world) == main and pl["run"]["tasks"] == {"check": check, "graph": fake.graph(check, GRAPH_V2)[0]}, \
        "B is pending: `main` stays"
    _results(fake, g)
    h.tick()
    pl = h.postland()
    assert _main(world) == main and pl["run"]["tip"] == tip and pl.get("gated") != tip and pl["verified"] != tip, \
        "every task passed, but the run beside them runs: `main` waits for it"
    _slim(h, tmp_path)
    h.tick()
    pl = h.postland()
    assert _main(world) == tip and pl["verified"] == tip and pl["run"] is None and len(h.posted) == 1 and not pl.get("red")
    assert pl["verified_as"] == {"steps": list(AFTER_LANDING), "no_cache": False, "executor": check, "slim": "r-post-0"}
    assert "slim" not in pl and pl["green"]["step ratchet-gate"] == tip and pl["ratchet_judged"]["tip"] == tip
    assert len(fake.enqueued()) == 1 and any(f"(the executor's {check})" in x for x in world.logs), world.logs
    assert any(", and its run r-post-0 passed ratchet-gate" in x for x in world.logs), world.logs


def test_on_a_graph_with_no_task_of_a_job_main_waits_for_gates_nothing_and_is_told(world, monkeypatch, tmp_path):
    """The check stands for the jobs its graph has tasks of and the steps of the run beside it: with none of some job `main`
    waits for (one graph/v2 has no family of), every task passing verifies the tip on its jobs and that run's steps alone (and
    lean-agreement, which it skipped by name), `main` stays, and `CI` is told what the graph lacks, ratchet-gate not among
    them though it gates here: that run checks it."""
    _move_main(world, GATED_TREE)
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    main = _main(world)
    tip = _land(h, 1)
    g = _graph_v2(tip, whole=False)
    fake.list_graph(f"after-landing:{tip}", g)
    _results(fake, g)
    _slim(h, tmp_path)
    h.tick()
    pl = h.postland()
    assert pl["verified"] == tip and pl["verified_as"]["steps"] == ["lean-agreement", "lean-audit", "pytest", "circuit-check",
                                                                    "ratchet-gate"]
    assert _main(world) == main and pl.get("gated") != tip and pl["run"] is None
    lacks = [j for j in AFTER_LANDING if j not in (*AFTER_LANDING_OPTIONAL, *SLIM_STEPS, "pytest", "lean-audit", "circuit-check")]
    (told,) = [p for p in h.posts if "its graph has no task of" in p]
    assert f"has no task of {', '.join(lacks)}, which `main` waits for there" in told, told


def test_on_names_the_one_landing_of_the_batch_whose_change_touches_the_failed_tasks_inputs(world, monkeypatch, tmp_path):
    """Three landings in one check; the failed test file's inputs are `pkg/tests/test_x.py` and `pkg/x.py`, which only #2
    changes (#3 changes `pkg/y.py`, in the suite's directory, which a run's rule would blame too).  The red is its suite's,
    names #2 and rings #2's lane and the suite's owner at once, while the check goes on; once it ended and `FIX_FORWARD_S`
    passed with no fix marked, #2 is reverted on `checked` and the revert's tip is checked."""
    _owning(world, {"lane-2": [], "pkg-owner": ["pkg/"]})
    _move_main(world, {**GATED_TREE, "pkg/pyproject.toml": '[project]\nname = "My_Pkg"\n'})
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    main = _main(world)
    for n, f in ((1, "a.txt"), (2, "pkg/x.py"), (3, "pkg/y.py")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
        h.review(n)
    assert h.tick().startswith("landed #3"), world.logs
    tip = _checked(world)
    (row,) = fake.enqueued()
    g = _graph_v2(tip)
    fake.list_graph(row["check"], g)
    _results(fake, g, {X: {"exit": 1, "failed_tests": [TEST], "passed": 2, "failed": 1}}, reader=None)
    h.tick()
    (ring,) = _rings(h)
    assert ring.startswith("@lane-2 @pkg-owner queue: suite my-pkg is red on `checked`"), ring
    assert f"{X}: failed {TEST}" in ring and f"(run {row['check']})" in ring, ring
    assert f"Culprit: #2 (its change touches pkg/x.py, which {X} depends on), of the 3 landing(s) since" in ring, ring
    r = h.postland()["red"]["suite my-pkg"]
    assert r["culprit"][0] == 2 and r["base"] == main and r["paths"] == ["pkg/x.py"] and r["step"] == "pytest"
    assert h.postland()["run"]["tip"] == tip == _checked(world), "B is still pending: the check goes on, red named at once"
    _results(fake, g, {X: {"exit": 1, "failed_tests": [TEST]}})
    _slim(h, tmp_path)
    h.tick()
    pl = h.postland()
    assert pl["failed"] == tip and _main(world) == main and pl.get("gated") != tip and _checked(world) == tip
    assert len(_rings(h)) == 1, "named once per tip"
    h.clock += D.FIX_FORWARD_S
    h.tick()
    reverted = _checked(world)
    assert _git(world.origin, "rev-parse", f"{reverted}^") == tip and _main(world) == main
    assert "pkg/x.py" not in _git(world.origin, "ls-tree", "-r", "--name-only", reverted)
    assert "pkg/y.py" in _git(world.origin, "ls-tree", "-r", "--name-only", reverted)
    assert any(p.startswith("@lane-2 @pkg-owner queue: #2 reverted on `checked`") for p in h.posts), h.posts
    pl = h.postland()
    assert pl["run"]["tip"] == reverted, "the revert's tip is checked next"
    assert fake.enqueued()[-1]["check"] == f"after-landing:{reverted}"
    assert [c for c, _, _ in h.posted] == [tip, reverted], "beside a run of its own"


@pytest.mark.parametrize("expires, exit_, red", [(LIVE, 1, False), (EXPIRED, 1, True), (LIVE, 2, True)])
def test_on_a_failed_task_an_unexpired_allowance_covers_is_not_red(world, monkeypatch, tmp_path, expires, exit_, red):
    """policy.toml's `suite NAME::TEST` covers a test-file task that failed on that test alone (pytest's exit 1): the tip is
    verified.  Expired, it is as if absent and the ring names it; and a file that failed beyond its tests (exit 2, collecting
    failed) isn't covered by its tests' allowances."""
    _move_main(world, {**GATED_TREE, "pkg/pyproject.toml": '[project]\nname = "My_Pkg"\n',
                       "tools/verity/check/policy.toml": _policy((f"suite my-pkg::{TEST}", "ci", expires))})
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    main = _main(world)
    tip = _land(h, 1)
    g = _graph_v2(tip)
    fake.list_graph(f"after-landing:{tip}", g)
    _results(fake, g, {X: {"exit": exit_, "failed_tests": [TEST]}})
    _slim(h, tmp_path)
    h.tick()
    pl = h.postland()
    rings = _rings(h)
    if not red:
        assert _main(world) == tip and pl["verified"] == tip and rings == [] and not pl.get("red"), (h.posts, world.logs)
        assert any("or failed only on what an allowance covers (1)" in x for x in world.logs), world.logs
        return
    assert _main(world) == main and pl["failed"] == tip and list(pl["red"]) == ["suite my-pkg"]
    (ring,) = rings
    assert "None of the 1 landing(s)" in ring and "Reverted nothing" in ring, ring
    assert ("Its allowance in policy.toml expired" in ring) == (expires == EXPIRED), ring


@pytest.mark.parametrize("allowed", [False, True])
def test_on_a_red_task_off_the_gate_holds_no_promotion_and_names_its_culprit(world, monkeypatch, tmp_path, allowed):
    """A circuit-check task is off the merge gate (policy.toml's `off_gate`): every task on the gate passed, so `main` follows
    the tip though the binding failed; its red is circuit-check's, whose culprit is reverted `FIX_FORWARD_S` after with no fix
    marked.  An allowance of each failure it names covers it, and the tip is verified."""
    _move_main(world, {**GATED_TREE, "tools/verity/check/policy.toml": _policy(*([(f"circuit-check {FP8}", "ci", LIVE)] * allowed))})
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    world.gh.put(1, world.change({"pkg/fp8.circuit-check.py": "X = 1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    tip = _checked(world)
    g = _graph_v2(tip)
    fake.list_graph(f"after-landing:{tip}", g)
    _results(fake, g, {FP8_TASK: {"failures": [f"{FP8}{{N=4}}: 3 mismatches"]}})
    _slim(h, tmp_path)
    h.tick()
    pl = h.postland()
    assert _main(world) == tip and any(f"(the executor's after-landing:{tip})" in x for x in world.logs), world.logs
    if allowed:
        assert pl["verified"] == tip and not pl.get("red") and _checked(world) == tip and pl.get("failed") != tip
        return
    assert pl["gated"] == tip and pl["gated_by"] == f"after-landing:{tip}" and pl["failed"] == tip
    assert list(pl["red"]) == ["step circuit-check"]
    (ring,) = [p for p in h.posts if "Culprit: #1" in p]
    assert "step circuit-check is red on `checked`" in ring and "or drive reverts #1 then." in ring, ring
    assert _checked(world) == tip
    h.clock += D.FIX_FORWARD_S
    h.tick()
    reverted = _checked(world)
    assert _git(world.origin, "rev-parse", f"{reverted}^") == tip
    assert "pkg/fp8.circuit-check.py" not in _git(world.origin, "ls-tree", "-r", "--name-only", reverted)


def _suite_row(name: str, suite: str, *tests: str, **more) -> dict:
    """A failed suite's row in a lean-suites task's `detail.failed` (`runners/lean_suites.py`'s `row`)."""
    return {"name": name, "suite": suite, "failed_tests": list(tests), **more}


@pytest.mark.parametrize("files, why", [
    (("a.txt", "tools/lsuite/tests/test_l.py", "tools/lsuite/y.py"),
     f"its change touches tools/lsuite/tests/test_l.py, which {LS} depends on"),
    (("a.txt", "shared/z.py", "pkg/y.py"), "its change touches shared/z.py, which is under suite l-suite's declared inputs")])
def test_on_a_lean_suites_task_is_red_per_failed_suite_and_names_its_culprit_as_a_run_does(world, monkeypatch, tmp_path, files,
                                                                                          why):
    """One coarse lean-suites task, its inputs the whole tree (each landing touches them), fails suite `l-suite` on one test.
    The red is `suite l-suite`, as a run's is, rung to the suite's owner; its culprit is the one landing touching the failed
    test's file (#3, in the suite's directory, isn't), else, none does, the one touching the suite's declared inputs."""
    _owning(world, {"lane-2": [], "ls-owner": ["tools/lsuite/"]})
    _move_main(world, L_TREE)
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    main = _main(world)
    for n, f in enumerate(files, 1):
        world.gh.put(n, world.change({f: f"{n}\n"}))
        h.review(n)
    assert h.tick().startswith("landed #3"), world.logs
    tip = _checked(world)
    g = _graph_v2(tip, lean_suites=files)
    fake.list_graph(f"after-landing:{tip}", g)
    _results(fake, g, {LS: {"exit": 1, "suites": {"l-suite": "failed", "other": "passed"},
                            "failed": [_suite_row("l-suite", "tools/lsuite", L_TEST)]}})
    _slim(h, tmp_path)
    h.tick()
    pl = h.postland()
    assert list(pl["red"]) == ["suite l-suite"] and pl["failed"] == tip and _main(world) == main
    r = pl["red"]["suite l-suite"]
    assert r["culprit"][0] == 2 and r["step"] == "lean-suites" and r["paths"] == [files[1]] and r["why"] == why, r
    (ring,) = _rings(h)
    assert ring.startswith("@lane-2 @ls-owner queue: suite l-suite is red on `checked`"), ring
    assert f"{LS}: suite l-suite failed {L_TEST}" in ring and f"Culprit: #2 ({why})" in ring, ring


@pytest.mark.parametrize("allow, expires, more, red", [
    ("suite l-suite", LIVE, {}, False), ("suite l-suite", EXPIRED, {}, True),
    (f"suite l-suite::{L_TEST}", LIVE, {"beyond_tests": False}, False), (f"suite l-suite::{L_TEST}", LIVE, {}, True)])
def test_on_a_lean_suites_tasks_failed_suite_is_judged_by_its_own_allowance(world, monkeypatch, tmp_path, allow, expires, more,
                                                                            red):
    """policy.toml judges each failed suite of a lean-suites task by its report row, as a run's (`Policy.suite`): `suite NAME`
    covers it and the tip is verified; expired, it is red and the ring names it; `suite NAME::TEST` covers it only when its row
    says it failed for nothing else (suites.py says so of none, so on either path only a whole suite's allowance covers it)."""
    _move_main(world, {**L_TREE, "tools/verity/check/policy.toml": _policy((allow, "ci", expires))})
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    main = _main(world)
    tip = _land(h, 1)
    g = _graph_v2(tip, lean_suites=("f1.txt",))
    fake.list_graph(f"after-landing:{tip}", g)
    _results(fake, g, {LS: {"exit": 1, "suites": {"l-suite": "failed", "other": "passed"},
                            "failed": [_suite_row("l-suite", "tools/lsuite", L_TEST, **more)]}})
    _slim(h, tmp_path)
    h.tick()
    pl = h.postland()
    rings = _rings(h)
    if not red:
        assert _main(world) == tip and pl["verified"] == tip and rings == [] and not pl.get("red"), (h.posts, world.logs)
        return
    assert _main(world) == main and pl["failed"] == tip and list(pl["red"]) == ["suite l-suite"]
    (ring,) = rings
    assert ("Its allowance in policy.toml expired" in ring) == (expires == EXPIRED), ring


def test_on_a_suite_a_lean_suites_task_passes_ends_its_red_and_a_run_that_ran_nothing_holds_main(world, monkeypatch, tmp_path):
    """Tip 1's lean-suites task fails suites `l-suite` and `other`: each is red.  Tip 2's fails `other` alone: `l-suite`'s red
    ends there though the tip failed.  Tip 3's runner ran nothing it was to run (a skip): it gives an error, which is no result,
    so the task stays pending, nothing is gated or verified, and `main` stays, though the run beside the check passed."""
    _move_main(world, L_TREE)
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    main = _main(world)
    other = _suite_row("other", "other", "tests/test_o.py::test_o")
    for n, failed in ((1, [_suite_row("l-suite", "tools/lsuite", L_TEST), other]), (2, [other])):
        tip = _land(h, n)
        g = _graph_v2(tip, lean_suites=tuple(f"f{k}.txt" for k in range(1, n + 1)))
        fake.list_graph(f"after-landing:{tip}", g)
        _results(fake, g, {LS: {"exit": 1, "failed": failed,
                                "suites": {"l-suite": "failed" if n == 1 else "passed", "other": "failed"}}})
        _slim(h, tmp_path)
        h.tick()
        pl = h.postland()
        assert pl["failed"] == tip and _main(world) == main
        assert sorted(pl["red"]) == (["suite l-suite", "suite other"] if n == 1 else ["suite other"]), (n, pl["red"])
    assert pl["green"]["suite l-suite"] == tip and any("suite l-suite passed on `checked`" in x for x in world.logs), world.logs
    tip = _land(h, 3)
    g = _graph_v2(tip, lean_suites=("f1.txt", "f2.txt", "f3.txt"))
    fake.list_graph(f"after-landing:{tip}", g)
    _results(fake, {**g, "tasks": [t for t in g["tasks"] if t["id"] != LS]})
    ls = next(t for t in g["tasks"] if t["id"] == LS)
    fake.finish_result({**_result(ls, ls["key"]), "status": "error", "detail": {"error": "not run: tools/lsuite"}})
    _slim(h, tmp_path)
    h.tick()
    pl = h.postland()
    assert _main(world) == main and pl["run"]["tip"] == tip and pl.get("gated") != tip and pl["verified"] != tip
    assert pl.get("failed") != tip and sorted(pl["red"]) == ["suite other"]


def test_on_a_graph_of_another_commit_is_told_and_the_check_enqueued_again(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    tip = _land(h, 1)
    fake.list_graph(f"after-landing:{tip}", _graph_v2("9" * 40))
    h.tick()
    assert any("is of commit " + "9" * 40 in p and "the next tick enqueues" in p for p in h.posts), h.posts
    assert [r["check"] for r in fake.enqueued()] == [f"after-landing:{tip}"] * 2 and h.postland()["run"]["tasks"] == {
        "check": f"after-landing:{tip}"}, "the tick that refused it enqueues it again"
    assert len(h.posted) == 1 and h.postland()["slim"]["ids"] == ["r-post-0"], "beside the same run"


def test_on_main_waits_for_the_run_of_lean_agreement_and_ratchet_gate_beside_the_executors_check(world, monkeypatch, tmp_path):
    """A landing that changes what lean-agreement checks: the run beside the executor's check is of lean-agreement and
    ratchet-gate (`train.sh --deferred lean-agreement,ratchet-gate --agreement`).  The graph's own tasks of those two fail,
    which counts for nothing, and every other task passes: nothing is gated, verified or red while that run runs, and once it
    passed both, the tip is verified on every step and `main` follows."""
    monkeypatch.setattr(merge, "required_steps", lambda repo, base, commit, tool: {"lean-agreement": ["verity/protocols/flock/prover/"]})
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    main = _main(world)
    tip = _land(h, 1)
    assert h.posted == [(tip, "lean-agreement,ratchet-gate", {"base": main, "agreement": True, "no_cache": False,
                                                              "circuit_cache": False, "no_regenerate": True})]
    g = _graph_v2(tip)
    fake.list_graph(f"after-landing:{tip}", g)
    _results(fake, g, {"lean-agreement:all": {"exit": 1}, "ratchet-gate:all": {"exit": 1}})
    for _ in range(2):
        h.tick()
        pl = h.postland()
        assert _main(world) == main and pl["run"]["tip"] == tip and pl.get("gated") != tip and not pl.get("red"), pl
    _slim(h, tmp_path)
    h.tick()
    pl = h.postland()
    assert _main(world) == tip == pl["verified"] and not pl.get("red") and pl.get("failed") != tip, world.logs
    assert pl["verified_as"]["steps"] == list(AFTER_LANDING) and pl["verified_as"]["slim"] == "r-post-0"
    assert any(", and its run r-post-0 passed lean-agreement, ratchet-gate" in x for x in world.logs), world.logs


def test_on_a_red_ratchet_gate_in_the_run_beside_the_check_holds_nothing_and_names_its_culprit_as_a_run_does(world, monkeypatch,
                                                                                                         tmp_path):
    """Every task of the executor's check passes, but the run beside it fails ratchet-gate, its log naming the merge the gate
    fails.  ratchet-gate is off the gate (policy.toml's `off_gate`, as in the tree), so it holds nothing, as in a run of every
    step: the tip is `gated` and `main` follows it.  The red names that merge's PR as its culprit and rings as a run's does,
    and drive reverts it on `checked` once `FIX_FORWARD_S` passed with no fix marked.  The revert's check, launched in the tick
    that reverts it, runs beside a run whose gate judges above the red tip, where the revert lies; that passing, `main`
    follows the revert and the red ends."""
    _owning(world, {"lane-2": []})
    _move_main(world, {**GATED_TREE, "tools/verity/check/policy.toml": _policy(off_gate=("circuit-check", "ratchet-gate"))})
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    main = _main(world)
    assert "ratchet-gate" not in h.d._gating(main)
    for n, f in ((1, "a.txt"), (2, "c.py"), (3, "b.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
        h.review(n)
    assert h.tick().startswith("landed #3"), world.logs
    tip, verified = _checked(world), h.postland()["verified"]
    merge2 = _git(world.origin, "rev-parse", f"{tip}^1")
    assert _git(world.origin, "log", "-1", "--format=%s", merge2).startswith("Queue: #2")
    g = _graph_v2(tip)
    fake.list_graph(f"after-landing:{tip}", g)
    _results(fake, g)
    _slim(h, tmp_path / "1", ratchet_gate=f"  #2 ({merge2[:12]}): failed: grows p01.json 2 -> 3 entries\n")
    h.tick()
    pl = h.postland()
    (ring,) = _rings(h)
    assert ring.startswith(f"@lane-2 @ci queue: step ratchet-gate is red on `checked` {tip[:12]} (run r-post-0): "
                           f"`#2 ({merge2[:12]}): failed: grows p01.json 2 -> 3 entries`"), ring
    assert "Culprit: #2 (the ratchet gate fails its merge), of the 3 landing(s)" in ring and "or drive reverts #2 then." in ring, ring
    assert pl["red"]["step ratchet-gate"]["culprit"][:1] == [2]
    assert pl["ratchet_judged"] == {"tip": tip, "base": verified, "failed": [merge2]}
    assert pl["failed"] == tip
    assert pl["verified"] == verified and pl["gated"] == tip == _main(world) == _checked(world) != main, world.logs
    h.clock += D.FIX_FORWARD_S
    h.tick()
    pl = h.postland()
    reverted = _checked(world)
    assert _git(world.origin, "rev-parse", f"{reverted}^") == tip and _main(world) == tip
    assert f"This reverts commit {merge2}" in _git(world.origin, "log", "-1", "--format=%B", reverted)
    assert any(p.startswith("@lane-2 @ci queue: #2 reverted on `checked`") for p in h.posts), h.posts
    assert pl["run"]["tip"] == pl["slim"]["tip"] == reverted, "the revert's check and its run, launched in the tick reverting"
    got, steps, kw = h.posted[-1]
    assert pl["verified"] == tip, "`main` reached the red tip, past every step on the gate, before the revert"
    assert got == reverted and steps == "ratchet-gate" and kw["base"] == tip and "ratchet_base" not in kw, kw
    g = _graph_v2(reverted)
    fake.list_graph(f"after-landing:{reverted}", g)
    _results(fake, g)
    _slim(h, tmp_path / "2")
    h.tick()
    pl = h.postland()
    assert _main(world) == reverted == pl["verified"] and "step ratchet-gate" not in (pl.get("red") or {}), world.logs


@pytest.mark.parametrize("fails, off_gate, holds", [
    ("ratchet-gate", ("circuit-check", "ratchet-gate"), False), ("ratchet-gate", ("circuit-check",), True),
    ("lean-agreement", ("circuit-check", "ratchet-gate"), True), ("preflight-lints", ("circuit-check", "ratchet-gate"), True)])
def test_on_a_red_in_the_run_beside_the_check_holds_main_only_on_the_gate(world, monkeypatch, tmp_path, fails, off_gate,
                                                                          holds):
    """Every task of the executor's check passes and the run beside it fails one thing: the tip is `gated` and `main` follows
    it unless that is on the gate by the `off_gate` a run of every step reads (`_gating`), or is a preflight, which no graph/v2
    task runs, so it holds `main` though the run's one step after it, ratchet-gate (lean-agreement skipped), is off the gate."""
    if fails == "lean-agreement":
        monkeypatch.setattr(merge, "required_steps", lambda repo, base, commit, tool: {"lean-agreement": ["verity/protocols/flock/prover/"]})
    _move_main(world, {**GATED_TREE, "tools/verity/check/policy.toml": _policy(off_gate=off_gate)})
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    main = _main(world)
    assert ("ratchet-gate" in h.d._gating(main)) is ("ratchet-gate" not in off_gate)
    tip = _land(h, 1)
    assert h.posted[-1][1] == ("lean-agreement,ratchet-gate" if fails == "lean-agreement" else "ratchet-gate")
    g = _graph_v2(tip)
    fake.list_graph(f"after-landing:{tip}", g)
    _results(fake, g)
    _slim(h, tmp_path, **{fails.replace("-", "_"): f"{fails} failed\n"})
    h.tick()
    pl = h.postland()
    assert pl["failed"] == tip and pl["run"] is None and pl["verified"] != tip, world.logs
    assert (pl.get("gated") == tip) is not holds and _main(world) == (main if holds else tip), world.logs


def test_on_a_run_beside_the_check_killed_by_its_machine_runs_again_once_and_killed_again_is_told(world, monkeypatch, tmp_path):
    """The run beside the executor's check, killed (a signal's exit, a machine's fault), launches again once beside the same
    check; killed again, `CI` is told, nothing is reverted, and `main` stays below the tip though every task passed."""
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    main = _main(world)
    tip = _land(h, 1)
    g = _graph_v2(tip)
    fake.list_graph(f"after-landing:{tip}", g)
    _results(fake, g)
    h.run_state["r-post-0"] = ("done", 137, None)
    h.tick()
    pl = h.postland()
    assert [c for c, _, _ in h.posted] == [tip, tip] and pl["slim"]["ids"] == ["r-post-1"] and pl["slim"]["resend"], world.logs
    assert _main(world) == main and pl["run"]["tip"] == tip and len(fake.enqueued()) == 1
    h.run_state["r-post-1"] = ("done", 137, None)
    h.tick()
    pl = h.postland()
    (told,) = [p for p in h.posts if "killed or lost again on its resend" in p]
    assert f"the run of ratchet-gate beside the executor's check of `checked` {tip[:12]}, r-post-1, failed" in told, told
    assert len(h.posted) == 2 and _main(world) == main and pl["failed"] == tip and pl["run"] is None and not pl.get("red")


def test_drive_runs_the_enqueue_command_drive_toml_names(world, monkeypatch, tmp_path):
    """Drive's default transport runs drive.toml's `executor_enqueue`, infra's command (research imports nothing from `infra/`),
    with drive's arguments."""
    import sys

    from research.queue import DriveFlags

    assert DriveFlags.parse('executor_enqueue = ["python3", "enqueue.py"]\n').executor_enqueue == ("python3", "enqueue.py")
    assert DriveFlags.parse("").executor_enqueue == ()
    for bad in ("executor_enqueue = []", 'executor_enqueue = "python3 enqueue.py"', 'executor_enqueue = [""]'):
        with pytest.raises(Exception, match="executor_enqueue = .*: a command as a non-empty list of its words"):
            DriveFlags.parse(bad + "\n")
    seen, stub = tmp_path / "argv.txt", tmp_path / "enqueue.py"
    stub.write_text(f"import sys\nopen({str(seen)!r}, 'w').write(' '.join(sys.argv[1:]))\n")
    h = Review(world, monkeypatch, land_into="checked")
    h.d.executor, h.d.executor_enqueue = "shadow", (sys.executable, str(stub), "--db", "q b.db")
    one = _land(h, 1)
    assert seen.read_text() == f"--db q b.db --priority 1 --check after-landing:{one}"
    assert one in h.q.load_state()["drive"]["executor"]


def test_with_no_enqueue_command_a_shadowed_tip_is_recorded_incomplete_and_lands(world, monkeypatch):
    h = Review(world, monkeypatch, land_into="checked")
    h.d.executor = "shadow"
    tip = _land(h, 1)
    assert len(h.posted) == 1
    (rec,) = world.store.labels(f"commit:{tip}", key=vocab.EXECUTOR_SHADOW)
    assert rec.value["incomplete"] == ("enqueueing its check: RuntimeError: drive.toml names no `executor_enqueue` command: "
                                       "set it to infra's enqueue command, the words of its argv")


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
