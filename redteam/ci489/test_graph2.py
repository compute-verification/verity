"""graph2's verdict, enqueue, listing, cache and leases (infra's executor), as the verified queue reads them.

Run against infra's branch: `VQRED_GRAPH2=<infra checkout>/infra/executor/graph2.py pytest redteam/ci489/test_graph2.py`.
Each test drives graph2's own functions on a fresh SQLite database, the way serve's threads and the CLI call them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

A, B = "a" * 40, "b" * 40
QUIET = {"log": lambda *a, **k: None}


def doc(g2, sha: str, tasks: list[dict]) -> dict:
    """A graph/v2 document of ``tasks`` at ``sha``: each a test-file task on src/x.py unless it says otherwise."""
    files = {"src/x.py": "sha512:" + "1" * 128, "src/y.py": "sha512:" + "2" * 128, "uv.lock": "sha512:" + "3" * 128}
    envs = {"py": {"spec": ["uv.lock"], "digest": g2.env_digest(["uv.lock"], files)}}
    full = [{"kind": "test-file", "env": "py", "argv": ["verity", "graph", "run", t["id"]], "vars": {}, "inputs": ["src/x.py"],
             "reads": [], "outputs": [], "resources": {"cores": 1, "mem_gb": 1}, "timeout_s": 900, **t} for t in tasks]
    return {"schema": g2.GRAPH, "commit": sha, "envs": envs, "files": files, "tasks": full}


@pytest.fixture
def db(g2, tmp_path):
    return g2.connect(tmp_path / "graph2.db"), g2.Events(tmp_path / "events")


def list_once(g2, c, ev, lister) -> int:
    return g2.list_requested(c, ev, "node", lambda sha: Path("/trees") / sha, lister=lister,
                             publisher=lambda sha, graph: ("art:graph", ""), **QUIET)


def listed(g2, c, ev, check: str, graph: dict) -> str:
    """Enqueue ``check`` and let the check lister list it as ``graph`` and publish it, as serve does; its submission."""
    g2.request(c, check, 5)
    list_once(g2, c, ev, lambda node, tree: (graph, "run-list", "ok"))
    list_once(g2, c, ev, lambda node, tree: pytest.fail("listed twice"))
    return c.execute("select submission from checks where id = ?", (check,)).fetchone()[0]


def lease(g2, c, sid: str, tid: str, slot: str = "s1") -> str | None:
    c.execute("update tasks set state = 'leased', worker = ?, started = ?, attempts = attempts + 1 where submission = ? and id = ?",
              (slot, g2.now(), sid, tid))
    return c.execute("select key from tasks where submission = ? and id = ?", (sid, tid)).fetchone()[0]


def end(g2, c, ev, sid: str, tid: str, status: str, outputs: dict | None = None, slot: str = "s1", **detail) -> str:
    """Lease (``sid``, ``tid``) to ``slot`` and finish it with a result/v1 of ``status``, as a dispatcher does; its new state."""
    key = lease(g2, c, sid, tid, slot)
    res = {"schema": g2.RESULT, "task": tid, "key": key, "status": status, "outputs": outputs or {}, "detail": detail,
           "undeclared_reads": detail.pop("undeclared_reads", [])}
    return g2.finish(c, ev, sid, tid, key, "run-task", res, f"run on {slot}")


def check_state(c, check: str) -> str:
    return c.execute("select state from checks where id = ?", (check,)).fetchone()[0]


def labels(g2, c) -> list[dict]:
    got: list[dict] = []
    g2.label_verdicts_pending(c, labeler=lambda sha, d: got.append({"sha": sha, **d}) or True, **QUIET)
    return got


# --- soundness ---------------------------------------------------------------------------------------------------------------

def test_a_listing_with_no_tasks_passes_and_is_labelled_pass(g2, db):
    """FINDING (soundness): the verdict is `pass` when no task failed or is pending, so a candidate whose own tree lists no
    task (or fewer: a family dropped) passes. Nothing compares the task set with the base's. FIX: W3 holds each gating job to
    at least one task (drive's `lacks`) and the candidate's task ids to the base's, less tasks whose files it deleted."""
    c, ev = db
    listed(g2, c, ev, f"ready:{A}", doc(g2, A, []))
    v = g2.verdict(c, f"ready:{A}")
    assert (v["state"], v["tasks"], v["pending"]) == ("pass", 0, 0)
    assert [x["state"] for x in labels(g2, c)] == ["pass"]


def test_enqueue_with_a_graph_checks_the_graphs_commit_under_the_checks_id(g2, db, tmp_path, monkeypatch):
    """FINDING (soundness, latent): `enqueue --check ready:A --graph FILE` takes FILE's commit as the tree to run (`tree_at`
    of graph["commit"]) without comparing it with A, and the verdict is labelled on `commit:A`. FIX: refuse a graph whose
    commit isn't the check's sha; W3 never passes --graph."""
    c, _ = db
    f = tmp_path / "graph.json"
    f.write_text(json.dumps(doc(g2, B, [{"id": "test-file:t.py"}])))
    monkeypatch.setattr(g2.X, "tree_at", lambda repo, trees, commit: Path("/trees") / commit)
    assert g2.main(["--db", str(tmp_path / "graph2.db"), "--events", str(tmp_path / "events"), "enqueue", "--check",
                    f"ready:{A}", "--graph", str(f), "--schema", g2.GRAPH]) == 0
    sid = c.execute("select submission from checks where id = ?", (f"ready:{A}",)).fetchone()[0]
    assert c.execute("select commit_sha, source from submissions where id = ?", (sid,)).fetchone() == (B, f"/trees/{B}")


def test_fail_arrives_while_tasks_are_pending_and_blocked_tasks_are_unnamed(g2, db):
    """FINDING (soundness, at W3's gate filter): `fail` is returned as soon as one task failed, with others still pending, and a
    task blocked behind a failure is only counted. ci's W3 drops failures of off-gate jobs (circuit-check) itself, so a
    candidate whose only failure so far is circuit-check's (red on the base today, so answered from the cache in seconds) reads
    `fail` with nothing gating named. W3 may take it as pass only once `pending` and `blocked` are 0, and it cannot tell
    whether a blocked task was gating. FIX: `verdict` names blocked tasks with the task they wait on; W3 requires positive
    evidence: every gating task passed."""
    c, ev = db
    sid = listed(g2, c, ev, f"ready:{A}", doc(g2, A, [
        {"id": "circuit-catalog:all", "kind": "circuit-catalog", "outputs": [{"name": "records", "path": "out/records"}]},
        {"id": "circuit-binding:Foo", "kind": "circuit-binding", "inputs": ["src/y.py"],
         "reads": [{"task": "circuit-catalog:all", "output": "records"}]},
        {"id": "test-file:gating_reads_catalog.py", "inputs": ["src/y.py"],
         "reads": [{"task": "circuit-catalog:all", "output": "records"}]},
        {"id": "test-file:gating.py"}]))
    assert end(g2, c, ev, sid, "circuit-catalog:all", "fail") == "done"
    v = g2.verdict(c, f"ready:{A}")
    assert v["state"] == "fail" and v["pending"] == 1 and v["blocked"] == 2
    assert [x["task"] for x in v["failed"]] == ["circuit-catalog:all"]
    assert "test-file:gating_reads_catalog.py" not in json.dumps(v)


def test_a_cached_fail_answers_a_retest_of_the_same_key_without_running(g2, db):
    """FINDING (innocent ejects): a fail is cached by key like a pass, so W3's "a single PR is retested once, to rule out a
    flake" reruns nothing when the task's key didn't change: the retest candidate (main plus that PR) is answered from the
    cache and fails at once, and a flaky first run ejects the PR as a "possible flake". FIX: graph2 gains a rerun of one
    failed key, uncached, recording both outcomes (two outcomes: nondeterministic, no PR blamed); or W3 drops the retest."""
    c, ev = db
    sid = listed(g2, c, ev, f"ready:{A}", doc(g2, A, [{"id": "test-file:flaky.py"}]))
    end(g2, c, ev, sid, "test-file:flaky.py", "fail")
    retest = listed(g2, c, ev, f"ready:{B}", doc(g2, B, [{"id": "test-file:flaky.py"}]))
    v = g2.verdict(c, f"ready:{B}")
    assert v["state"] == "fail" and v["pending"] == 0
    assert c.execute("select state from tasks where submission = ?", (retest,)).fetchone() == ("hit",)
    assert g2.claim(c, "s1") is None


# --- liveness ----------------------------------------------------------------------------------------------------------------

def test_fixed_an_undeclared_read_is_a_failure_and_an_outages_errors_arent_the_trees(g2, db, monkeypatch):
    """FIXED: at 40e08bd3556 an undeclared read or a race is the task's own fault (`faulted`, a failed entry W3 blames by
    inputs; at ffa125a3756 it was infra forever); at 3afee5d3402 an exhausted error is `tree` only when the same error came
    on two nodes that each passed another task after it (`trees`), so an outage's errors (each a different why, or on
    nodes that pass nothing meanwhile) aren't; at 40e08bd3556 any error on two nodes was."""
    clock = iter(range(1, 10**6))
    monkeypatch.setattr(g2, "now", lambda: float(next(clock)))
    c, ev = db
    check = f"ready:{A}"
    sid = listed(g2, c, ev, check, doc(g2, A, [{"id": "test-file:reads_undeclared.py"}, {"id": "test-file:t2.py"},
                                               {"id": "test-file:t3.py"}]))
    assert end(g2, c, ev, sid, "test-file:reads_undeclared.py", "error", undeclared_reads=["src/y.py"],
               error="an undeclared read") == "faulted"
    for slot, why in (("c4x0", "the CAS refused an artifact"), ("n2x0", "No space left on device"),
                      ("c4x1", "the store timed out")):
        end(g2, c, ev, sid, "test-file:t2.py", "error", slot=slot, error=why)
    v = g2.verdict(c, check)
    assert [(f["task"], f["tests"]) for f in v["failed"]] == [("test-file:reads_undeclared.py", [])]
    assert [(f["task"], f["tree"]) for f in v["infra"]] == [("test-file:t2.py", False)]
    for slot in ("c4x0", "n2x0", "n3x0"):
        end(g2, c, ev, sid, "test-file:t3.py", "error", slot=slot, error="Timeout after 1800 s")
    assert [(f["task"], f["tree"]) for f in g2.verdict(c, check)["infra"]] == [("test-file:t2.py", False),
                                                                             ("test-file:t3.py", False)]


def test_an_outages_errors_become_the_trees_once_it_ends(g2, db, monkeypatch):
    """FINDING (after the cut-over; it matters only once `--tree-ejects` is on): `trees` reads each node's newest pass at
    the time of the verdict, with no bound, so the same error two nodes gave in an outage (a full disk on both) becomes the
    tree's as soon as each passes any other task after the outage, and the loop reads every verdict again at each tick. With
    `--tree-ejects` on, that bisects and ejects lone members for the outage after it ends. FIX: count a node's pass only
    before the task's next attempt (or within its run's window), stored with the error when it is logged."""
    clock = iter(range(1, 10**6))
    monkeypatch.setattr(g2, "now", lambda: float(next(clock)))
    c, ev = db
    check = f"ready:{A}"
    sid = listed(g2, c, ev, check, doc(g2, A, [{"id": "test-file:t2.py"}, {"id": "test-file:t3.py"},
                                               {"id": "test-file:t4.py"}]))
    for slot in ("c4x0", "n2x0", "c4x1"):
        end(g2, c, ev, sid, "test-file:t2.py", "error", slot=slot, error="OSError: [Errno 28] No space left on device")
    assert [(f["task"], f["tree"]) for f in g2.verdict(c, check)["infra"]] == [("test-file:t2.py", False)]
    end(g2, c, ev, sid, "test-file:t3.py", "pass", slot="c4x0")
    end(g2, c, ev, sid, "test-file:t4.py", "pass", slot="n2x0")
    assert [(f["task"], f["tree"]) for f in g2.verdict(c, check)["infra"]] == [("test-file:t2.py", True)]


def test_a_listing_submit_refuses_wedges_the_check_in_listing(g2, db):
    """FINDING (liveness): `list_requested` sets the check `listing`, then calls `submit` outside its try. A graph `submit`
    refuses (an id listed twice, an env digest that doesn't match, a read of an unlisted task: what a PR's broken lister
    makes) raises out of it; the check lister logs it and goes on, and the check stays `listing`: its verdict is `pending`
    forever, enqueue answers `listing`, and nothing lists it again. W3 waits on the lowest candidate, so the queue stops.
    FIX: submit inside the try, the refusal `list-failed` with its why; W3 treats a list-failed that repeats as the batch's."""
    c, ev = db
    check = f"ready:{A}"
    g2.request(c, check, 5)
    twice = doc(g2, A, [{"id": "test-file:t.py"}, {"id": "test-file:t.py"}])
    with pytest.raises(ValueError, match="listed twice"):
        list_once(g2, c, ev, lambda node, tree: (twice, "run-list", "ok"))
    assert check_state(c, check) == "listing"
    assert g2.verdict(c, check)["state"] == "pending"
    assert g2.request(c, check, 5) == "listing"
    assert list_once(g2, c, ev, lambda node, tree: pytest.fail("listed again")) == 0


def test_a_serve_restart_while_listing_leaves_the_check_listing(g2, db):
    """FINDING (liveness, restarts): serve's start requeues leased tasks (`requeue_leased`) but not a check left `listing` by
    the restart, so that check is `pending` forever, as above. FIX: serve's start sets `listing` checks back to `requested`."""
    c, ev = db
    check = f"ready:{A}"
    g2.request(c, check, 5)
    c.execute("update checks set state = 'listing', attempts = 1 where id = ?", (check,))
    g2.requeue_leased(c)
    assert g2.request(c, check, 5) == "listing"
    assert list_once(g2, c, ev, lambda node, tree: pytest.fail("listed again")) == 0
    assert g2.verdict(c, check)["state"] == "pending"


def test_a_list_failed_check_listed_again_keeps_its_list_failed_label(g2, db):
    """FINDING (stale label): enqueue of a `list-failed` check resets its state but not `checks.verdict`, and
    `label_verdicts_pending` labels only checks whose verdict is null, so after it is listed and passes, `commit:<sha>
    ready_verdict` still says `list-failed`: the restart-safe copy W3 reads after a restart is stale. FIX: reset `verdict` in
    `request`'s list-failed branch, as its cancelled and infra branches do."""
    c, ev = db
    check = f"ready:{A}"
    g2.request(c, check, 5)
    for _ in range(g2.ATTEMPTS):
        list_once(g2, c, ev, lambda node, tree: (None, None, "the listing run died"))
    assert check_state(c, check) == "list-failed"
    assert [x["state"] for x in labels(g2, c)] == ["list-failed"]
    assert g2.request(c, check, 5) == "requested"
    list_once(g2, c, ev, lambda node, tree: (doc(g2, A, [{"id": "test-file:t.py"}]), "run-list", "ok"))
    list_once(g2, c, ev, lambda node, tree: pytest.fail("listed twice"))
    sid = c.execute("select submission from checks where id = ?", (check,)).fetchone()[0]
    end(g2, c, ev, sid, "test-file:t.py", "pass")
    assert g2.verdict(c, check)["state"] == "pass"
    assert labels(g2, c) == []


def test_a_healthy_batch_run_loses_its_short_tasks_leases(g2, db, monkeypatch):
    """FINDING (liveness, after the cut-over): one run takes a closure (`claim_batch`): a task, more one-core tasks of its
    submission (test files beside a root Lean module), then its readers level by level (the modules that import it). Every
    lease starts at the claim, and the dispatcher ends the run's tasks only when the whole run returns. `expire_leases`
    holds each lease to its own task's timeout (a test file's: 1950 s), so a run whose Lean chain takes longer loses the test
    files' leases while it is healthy: at its end their results are ignored ("its lease was taken back"), they run again, and
    each loss counts an attempt, so the third makes the verdict `infra`. Node-side waits for the slot count against every
    lease too. FIX: hold a run's leases to the run's budget (the sum of its levels' timeouts), renewed while it heartbeats."""
    c, ev = db
    t0 = 1_000_000.0
    monkeypatch.setattr(g2, "now", lambda: t0)
    lean = {"kind": "lean-module", "timeout_s": 3600}
    sid = listed(g2, c, ev, f"ready:{A}", doc(g2, A, [
        {"id": "lean-module:Root", **lean, "outputs": [{"name": "olean", "path": "out/Root.olean"}]},
        {"id": "lean-module:Mid", **lean, "inputs": ["src/y.py"], "reads": [{"task": "lean-module:Root", "output": "olean"}],
         "outputs": [{"name": "olean", "path": "out/Mid.olean"}]},
        {"id": "lean-module:Top", **lean, "inputs": ["src/y.py"], "reads": [{"task": "lean-module:Mid", "output": "olean"}]},
        {"id": "test-file:t.py"}]))
    got = g2.claim_batch(c, "s1", cores=16, mem_gb=64, most=64)
    assert [t for _, t, _ in got] == ["lean-module:Root", "test-file:t.py", "lean-module:Mid", "lean-module:Top"]
    test_file_limit = max(900, g2.LEASE_MIN_S) * g2.LEASE_FACTOR + g2.LEASE_GRACE_S
    assert test_file_limit == 1950
    # each of the run's three levels takes 700 s, well within every task's timeout; the run is in its third level
    monkeypatch.setattr(g2, "now", lambda: t0 + 2000)
    assert g2.expire_leases(c, ev, **QUIET) == 1
    assert not g2.still_leased(c, sid, "test-file:t.py", "s1")
    assert all(g2.still_leased(c, sid, t, "s1") for t in ("lean-module:Root", "lean-module:Mid", "lean-module:Top"))
    assert c.execute("select state, attempts from tasks where submission = ? and id = 'test-file:t.py'", (sid,)).fetchone() == \
        ("queued", 1)
