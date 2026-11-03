"""Drive's side of the task executor (`research.executor`, drive.toml's `executor`), on the file-backed fake transport: off
changes nothing; shadow enqueues each checked tip's check with no graph beside its after-landing run, reads the graph the
executor listed, and records how their verdicts compare by its keys; on verifies a tip, and moves `main`, only once every
task's verdict is pass.  `QueueTransport` reads the executor's graph through the store's remote."""
from __future__ import annotations

import json

import pytest

from research.drive import AFTER_LANDING
from research.executor import (AFTER_LANDING_PRIORITY, SHADOW_WAIT_S, TASKS_GRAPH_KIND, Executor, FileTransport, QueueTransport,
                               Shadowed, agrees, compare, keyed, old_verdicts, shadow_records, shadow_report)
from research.store.local import LocalStore
from research.store import vocab
from test_drive_review import Review, _main
from test_queue import World, _git

K_SUITE, K_LEAN = "a" * 128, "b" * 128
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
    h.listings: list[str] = []
    h.d.list_tasks = lambda repo, tip: h.listings.append(tip) or {"tip": tip}
    h.d.listed_tasks = lambda rec: (_graph(rec["tip"]), None)
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
    assert h.listings == [] and fake.enqueued() == [] and "executor" not in h.q.load_state()["drive"]
    assert world.store.labels(f"commit:{tip}") == []


def test_shadow_enqueues_each_tips_check_without_a_graph_and_compares_by_the_executors_graph(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "shadow")
    one = _land(h, 1)
    assert len(h.posted) == 1 and h.listings == [], "the run launches as before, and drive's host lists no graph"
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
    assert h.listings == [] and [r["graph"] for r in fake.enqueued()] == [None, None]


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
    assert h.listings == []


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
    assert h.listings == []


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


def test_on_launches_no_run_and_promotes_main_only_once_every_key_passed(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    main = _main(world)
    tip = _land(h, 1)
    assert h.posted == [] and h.listings == [tip]
    h.tick()
    keys = _keys(fake, 0)
    assert fake.enqueued()[0]["priority"] == 1 and _main(world) == main
    for k in list(keys.values())[:-1]:
        fake.finish(k, "pass")
    h.tick()
    assert _main(world) == main, "a task still pending: `main` stays"
    fake.finish(list(keys.values())[-1], "pass")
    h.tick()
    pl = h.postland()
    assert _main(world) == tip and pl["verified"] == tip and pl["run"] is None and h.posted == []
    assert pl["verified_as"]["executor"] == f"after-landing:{tip}"
    assert any(f"(the executor's after-landing:{tip})" in x for x in world.logs), world.logs


def test_on_refuses_main_on_a_failed_task_and_tells_ci(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, land_into="checked")
    fake = _executor(h, tmp_path, "on")
    main = _main(world)
    tip = _land(h, 1)
    h.tick()
    keys = _keys(fake, 0)
    for t, k in keys.items():
        fake.finish(k, "fail" if t == "lean-audit:Security" else "pass")
    h.tick()
    pl = h.postland()
    assert _main(world) == main and pl["failed"] == tip and pl["verified"] != tip and pl["run"] is None
    assert any(f"failed 1 of {N_TASKS} tasks: lean-audit:Security" in p for p in h.posts), h.posts
    h.tick()
    assert h.listings == [tip] and h.posted == [], "a failed tip isn't checked again"


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
