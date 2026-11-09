"""`research.vqueue_loop` (ci's W3b) with `research.vqueue.decide` (W3a) on ci's own harness: a throwaway origin, a fake
GitHub, graph2 as files (`test_vqueue_loop.Env`).

It runs inside ci's tests directory, whose conftest isolates the store and git: in a checkout of `cursor/ci-vqueue-w3` at
3b9df5d6dc or later (earlier heads: `cursor/ci-vqueue-loop` at 7c2e6589f1, which merges W3a's `decide`),

    cp redteam/ci489/test_loop.py tools/research/tests/test_vqred_loop.py
    PYTHONPATH=tools/research/src:tools/research/tests pytest tools/research/tests/test_vqred_loop.py
"""
from __future__ import annotations

import os
import subprocess

import pytest

import research.queue as Q
from research.executor import GRAPH_V2, Executor
from research.vqueue import decide
from test_queue import _git
from test_vqueue_loop import _acts, env  # noqa: F401  (env is ci's fixture)

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def git_in(repo, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(["git", *args], cwd=repo, input=stdin, capture_output=True, text=True, check=True, env=GIT_ENV).stdout.strip()


def test_a_failed_enqueue_after_the_push_is_enqueued_by_the_next_tick(env, monkeypatch):
    """FIXED at 702a8b73ac; a finding at 32e8a92f5e and 7c2e6589f1 (crash safety): Build pushes `queue/<sha12>`, then enqueues
    `ready:<sha>`. When the enqueue failed (graph2's database locked, a restart between the two), every later tick read the
    ref `pending` and nothing enqueued it: the queue stopped, silently. Now a candidate the log holds no enqueue of is
    list-failed, and the next tick retries it, once."""
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(decide)
    calls, real = [], loop.world.enqueue

    def enqueue(sha: str, priority: int) -> None:
        calls.append(sha)
        if len(calls) == 1:
            raise Q.QueueError("enqueue: graph2 exited 1: database is locked")
        real(sha, priority)

    monkeypatch.setattr(loop.world, "enqueue", enqueue)
    (failed,) = loop.tick()
    assert failed["action"] == "build" and "database is locked" in failed["error"]
    (sha,) = env.queued().values()
    (retry,) = loop.tick()
    assert (retry["action"], retry["sha"], retry["enqueued"]) == ("retry", sha, True)
    for _ in range(30):
        assert loop.tick() == []
    assert calls == [sha, sha] and [e["check"] for e in env.graph2.enqueued()] == [f"ready:{sha}"]


def test_the_shadows_ejects_keep_prs_out_of_the_live_queue_after_the_cut_over(env):
    """FINDING (cut-over, fix before it): the live loop reads the shadow's records as its own. A PR the shadow ejected (its mark
    never withdrawn, its owner never told) stays out of the live queue until it is marked again, for the log's 48 h window:
    nobody knows to mark it. The same holds for the shadow's rings and skip notices, which count as sent. FIX: the live loop
    reads only records with `shadow: false` (or starts on a new log) at the cut-over."""
    three = env.put(3, {"c.txt": "3\n"})
    shadow = env.loop(decide, shadow=True)
    (built,) = shadow.tick()
    keys = env.listed(built["sha"], {"pytest:c": ["c.txt"]})
    env.result(keys["pytest:c"], "pytest:c", "fail")
    assert _acts(shadow.tick()) == ["eject", "drop"]
    assert (3, three) in env.o.gh.ready and env.posts == [] and env.withdrawn == []
    live = env.loop(decide)
    for _ in range(5):
        assert live.tick() == []
    assert env.queued() == {} and env.posts == []


def forge(env, like: str, parents: list[str], extra: dict[str, str]) -> str:
    """A commit with ``like``'s message and tree plus ``extra``'s files, on ``parents``, pushed to origin's `queue/<sha12>`."""
    o = env.o.origin
    entries = _git(o, "ls-tree", f"{like}^{{tree}}").splitlines()
    for path, text in extra.items():
        entries.append(f"100644 blob {git_in(o, 'hash-object', '-w', '--stdin', stdin=text)}\t{path}")
    tree = git_in(o, "mktree", stdin="\n".join(entries) + "\n")
    msg = _git(o, "log", "-1", "--format=%B", like)
    sha = git_in(o, "commit-tree", tree, *(x for p in parents for x in ("-p", p)), "-m", msg)
    _git(o, "update-ref", f"refs/heads/queue/{sha[:12]}", sha)
    return sha


def test_a_queue_ref_the_loop_didnt_build_is_ignored_and_never_lands(env):
    """FIXED at 3b9df5d6dc; a finding (soundness) at 702a8b73ac and before: the loop took every `queue/*` branch whose
    first-parent commits parse as `Queue: #N` merges as a candidate, so a ref pushed by anyone that claimed a better-ranked
    member displaced the loop's own candidate, was enqueued by the loop, and once its check passed `main` fast-forwarded to a
    tree holding a file no PR has. Now a ref without the loop's own build record is logged `ignore`, and its own candidate
    stays."""
    one = env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    main = env.main()
    loop = env.loop(decide)
    (built,) = loop.tick()
    assert built["members"] == [1, 2]
    forged = forge(env, f"{built['sha']}^1", [main, one], {"evil.txt": "no PR adds this\n"})
    acts = [(r["action"], r["sha"]) for r in loop.tick()]
    assert ("ignore", forged) in acts and ("drop", built["sha"]) not in acts
    keys = env.listed(forged, {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    for _ in range(3):
        loop.tick()
    assert env.main() != forged
    assert f"ready:{forged}" not in [e["check"] for e in env.graph2.enqueued()]


def _agreement_graph(env, sha: str, agreement: str) -> None:
    """``sha``'s graph: a passing test file and a lean-agreement task of the verifier the PR changes, ended ``agreement``."""
    task = {"env": "py", "argv": ["t"], "vars": {}, "reads": [], "outputs": []}
    env.graph2.list_graph(f"ready:{sha}", {
        "schema": GRAPH_V2, "commit": sha, "envs": {"py": {"digest": "sha512:env"}},
        "files": {"verity/ml/flock/verifier/check.py": "sha512:v", "t/test_a.py": "sha512:t"},
        "tasks": [{"id": "test-file:t/test_a.py", "kind": "test-file", "inputs": ["t/test_a.py"], "key": "sha512:a", **task},
                  {"id": "lean-agreement:verity/ml/flock/verifier", "kind": "lean-agreement",
                   "inputs": ["verity/ml/flock/verifier/check.py"], "key": "sha512:agree", **task}]})
    env.result("sha512:a", "test-file:t/test_a.py", "pass")
    env.result("sha512:agree", "lean-agreement:verity/ml/flock/verifier", agreement)


def test_fixed_a_failing_run_beside_the_check_holds_a_candidate(env):
    """FIXED at a2a1368d98 (W3b's c3931ab8); a finding (coverage) at 3b9df5d6dc and before, when the loop left every
    `drive.SLIM_STEPS` task (lean-agreement, ratchet-gate) out of the gate and folded in no run that judged them, so a
    candidate whose lean-agreement failed advanced. Now the run beside each candidate's check starts at its build, and the
    verdict is `vqueue_slim.combine` of graph2's and the run's: while the run runs the candidate waits, and a failing run
    (a preflight, ratchet-gate) holds it though graph2's part passed. `--no-slim` is refused outside --shadow and
    --dry-run. Since b85656625c graph2's lean-agreement gates on its own too (the next test)."""
    from research import vqueue_slim
    from test_vqueue_loop import Slim

    class Beside(Slim):
        combine = staticmethod(vqueue_slim.combine)

    env.put(1, {"verity/ml/flock/verifier/check.py": "1\n"})
    slim = Beside()
    loop = env.loop(decide, slim=slim)
    loop.world.slim_driver = lambda: "drive"
    built, started = loop.tick()
    sha = built["sha"]
    assert (started["action"], started["sha"], started["why"]) == ("slim", sha, "started")
    _agreement_graph(env, sha, "pass")
    assert loop.world.verdict(sha).verdict.state == "pass"
    for _ in range(3):
        assert "advance" not in _acts(loop.tick())
    slim.verdicts["run-1"] = "fail"
    after = [a for _ in range(5) for a in _acts(loop.tick())]
    assert "advance" not in after and env.main() != sha


def test_fixed_a_failing_lean_agreement_task_holds_a_candidate_without_the_run_beside(env):
    """FIXED at b85656625c (0763fe0a7c): graph2's lean-agreement task is on the gate, so a candidate whose lean-agreement
    failed never advances even with no run beside its check; only ratchet-gate and policy's `off_gate` stay out."""
    env.put(1, {"verity/ml/flock/verifier/check.py": "1\n"})
    loop = env.loop(decide)
    (built,) = loop.tick()
    sha = built["sha"]
    _agreement_graph(env, sha, "fail")
    v = loop.world.verdict(sha).verdict
    assert v.state == "fail" and [f.task for f in v.failed] == ["lean-agreement:verity/ml/flock/verifier"]
    after = [a for _ in range(5) for a in _acts(loop.tick())]
    assert "advance" not in after and env.main() != sha


def test_fixed_a_result_label_another_writer_writes_doesnt_count_but_by_executor_is_self_asserted(tmp_path):
    """FIXED at b85656625c (709fe37587); a finding (soundness) at a2a1368d98 and before, when every `result` label on
    `task:<key>` counted, by whoever wrote it. Now only labels `--by executor` (`GRAPH2_BY`) count. That name is still
    self-asserted: `research data label --by` is a required free string and no `label_auth` companion is asked of a result,
    so anyone with store credentials writes as graph2. graph2's own verdict, which the loop also reads, overrides a label
    at graph2's own key; the next test is where it doesn't."""
    from research.store.local import LocalStore
    from research.vqueue_loop import Transport

    graph = {"schema": GRAPH_V2, "commit": "c" * 40, "envs": {"py": {"digest": "sha512:env"}}, "files": {"t.py": "sha512:t"},
             "tasks": [{"id": "test-file:t.py", "kind": "test-file", "env": "py", "argv": ["t"], "vars": {}, "inputs": ["t.py"],
                        "reads": [], "outputs": [], "key": "sha512:" + "a" * 128}]}
    for by, state in (("not-graph2", "pending"), ("executor", "pass")):
        store = LocalStore(tmp_path / by)
        store.add_label(f"task:{'a' * 128}", "result", {"schema": "verity-result/v1", "key": "sha512:" + "a" * 128,
                                                         "task": "test-file:t.py", "status": "pass", "outputs": {}}, by=by)
        assert Executor(Transport(store)).settle(graph)["test-file:t.py"][0] == state


def test_a_gating_task_graph2_names_failed_passes_where_the_loop_keys_it_otherwise(env):
    """FINDING (soundness, after the cut-over; it takes a deliberate writer): `_said` lets graph2's verdict override a label
    only at graph2's own key, and `_held` accepts graph2's `fail` (it must: graph2 counts circuit-check's tasks, which the
    lister never marks `"gate": false`). A task that reads another's outputs the loop keys itself, from the results it
    reads. So a pass written `--by executor` for the task it reads, with outputs graph2 never made, before graph2's own,
    and a pass at the key those make, pass the task while graph2's verdict names it failed, and the candidate advances on
    "graph2 says fail". FIX: hold while graph2's verdict names a task on the gate, by id, failed or infra that the loop
    reads as passed (or sign graph2's result labels)."""
    import json
    from pathlib import Path

    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(decide)
    script = env.graph2_script(loop)
    (built,) = loop.tick()
    sha, w = built["sha"], loop.world
    keys = env.listed(sha, {"lean-build:x": ["x.lean"], "pytest:a": ["a.txt"]}, reads={"pytest:a": "lean-build:x"})
    env.result(keys["lean-build:x"], "lean-build:x", "pass")
    _, made, _ = w._settled(w.tasks.graph_v2(f"ready:{sha}")[1])
    env.result(made["pytest:a"], "pytest:a", "pass")
    Path(f"{script}.verdict.ready:{sha}").write_text(json.dumps({"state": "fail", "pending": 0, "failed": [
        {"task": "pytest:a", "key": "sha512:graph2s-own", "tests": ["t.py::test_a"], "inputs": ["a.txt"], "reads": []}]}))
    j = w.verdict(sha, built["base"])
    assert j.verdict.state == "pass" and "graph2 says fail" in j.note
    assert "advance" in _acts(loop.tick()) and env.main() == sha


def _move_main(env) -> str:
    """`main` moved to a commit graph2 has no listing of, as at the cut-over."""
    _git(env.o.origin, "update-ref", "refs/heads/main", env.o.change({"m.txt": "m\n"}))
    return env.main()


def test_a_base_whose_listing_failed_makes_mains_red_each_prs_own_and_ejects_every_pr(env):
    """FINDING (liveness and false blame, fix before the cut-over): `main` is red today (W5's ready check: 4 real fails and
    45 faulted tasks). Only the base's listing says a failure is `main`'s. When graph2 says that listing failed, or the
    graph can't be read, `World._base` returns no keys ("each is its candidate's own"), `_gone` skips the kinds check, and
    nothing enqueues `ready:<base>` again (`_bases` enqueues only an unlisted base, once). `main`'s red is then every
    candidate's own: a batch bisects to lone members, each is rerun, fails again and is ejected by the flake rule, its
    owner told it broke a test it never touched. Here neither PR touches the red task and both are ejected. FIX: a base
    whose listing failed or can't be read waits as an unlisted one does, re-enqueued and rung naming the base; never "no
    failure on it"."""
    import json
    from pathlib import Path

    main = _move_main(env)
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    loop = env.loop(decide)
    script = env.graph2_script(loop, {"state": "fail", "pending": 0})
    Path(f"{script}.verdict.ready:{main}").write_text(json.dumps({"state": "list-failed", "pending": 0, "why": "node lost"}))
    red = {"pytest:r": "sha512:red"}
    env.result("sha512:red", "pytest:r", "fail")                        # main's red, which its listing would have named
    listed: set[str] = set()
    acts: list[str] = []
    for _ in range(40):
        for sha in env.queued().values():
            if sha not in listed:
                listed.add(sha)
                env.listed(sha, {"pytest:r": ["red.txt"]}, keys=red)
        acts += (now := _acts(loop.tick()))
        if "rerun" in now:
            env.result("sha512:red", "pytest:r", "fail")
    assert sorted(n for n, _ in env.withdrawn) == [1, 2] and env.main() == main and "advance" not in acts
    assert f"ready:{main}" not in [e["check"] for e in env.graph2.enqueued()]


def test_a_base_that_never_lists_holds_every_candidate_and_infra_is_rung_at_8_h_about_the_candidate(env):
    """FINDING (liveness, fix before the cut-over or run-book it): every candidate's verdict waits for its base's listing
    (`_gone` is asked of every candidate, failing or not). The loop enqueues `ready:<base>` once (`_bases`), and a check
    graph2 leaves `listing` (a serve restart while it lists, or a refused submit: `test_graph2.py`'s two wedges, undated)
    answers every enqueue `listing` and is never listed. So nothing lands, ever: decide retries the candidate at 4 h, which
    re-enqueues the candidate, not the base, and rings @infra once at 8 h, naming the candidate's PRs, not the base. FIX:
    re-enqueue an unlisted base at each retry and name it in the ring; graph2 puts `listing` checks back to `requested` at
    serve's start."""
    import json
    from pathlib import Path

    main = _move_main(env)
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(decide)
    script = env.graph2_script(loop, {"state": "pass", "pending": 0})
    Path(f"{script}.verdict.ready:{main}").write_text(json.dumps({"state": "pending", "pending": 0, "why": None}))
    (built,) = loop.tick()
    sha = built["sha"]
    keys = env.listed(sha, {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    acts: list[str] = []
    for _ in range(48):
        acts += _acts(loop.tick())
        env.clock += 1800
    assert env.main() == main and "advance" not in acts
    assert [e["check"] for e in env.graph2.enqueued()].count(f"ready:{main}") == 1
    rings = [r for r in env.lines() if r.get("action") == "posting" and r.get("posting") == "ring"]
    assert [r["to"] for r in rings] == [["@infra"]], rings
    assert sha[:12] in rings[0]["text"] and main[:12] not in rings[0]["text"] and "#1" in rings[0]["text"]


def test_fixed_a_fault_main_has_too_is_inherited_through_the_loop_and_one_the_change_adds_is_blamed(env):
    """Condition 2 through the loop's verdict, at b85656625c (df1fcef29c, b85656625c): a PR that touches a task's inputs,
    where `main` faults on that task with the same undeclared read, lands (the fault is rung as main's red); one whose
    change adds an undeclared read is ejected. What the inherited fault covers: graph2 keeps no test of a faulted run
    (its entry's `tests` is always []), and a task blocked behind it never runs, so the PR lands though `pytest:u`, whose
    input it changed, ran nothing. That is the accepted cost of advancing on a red `main`; its size is how many tasks
    fault on `main` (W5's ready check: 45, with 240 blocked behind them)."""
    import json
    from pathlib import Path

    def faulted(script, at: str, key: str, fault: str, blocked: int = 0) -> None:
        Path(f"{script}.verdict.ready:{at}").write_text(json.dumps({"state": "fail", "pending": 0, "blocked": blocked,
            "failed": [{"task": "pytest:t", "key": key, "tests": [], "inputs": ["t.txt"], "reads": [],
                        "why": f"run r20260101-000000-aaaa on n1 exit 1; error: {{}}; {fault}"}]}))

    undeclared = "read 1 file(s) outside its declared inputs: x.txt"
    for adds, pr in ((False, 1), (True, 2)):
        env.put(pr, {"t.txt": f"{pr}\n", "u.txt": f"{pr}\n"})
        loop = env.loop(decide, clone=f"clone{pr}")
        script = env.graph2_script(loop)
        built = next(r for r in loop.tick() if r["action"] == "build")
        sha, main = built["sha"], built["base"]
        env.listed(main, {"pytest:t": ["t.txt"], "pytest:u": ["u.txt"]}, reads={"pytest:u": "pytest:t"},
                   keys={"pytest:t": f"sha512:t0-{pr}"})
        env.listed(sha, {"pytest:t": ["t.txt"], "pytest:u": ["u.txt"]}, reads={"pytest:u": "pytest:t"},
                   keys={"pytest:t": f"sha512:t1-{pr}"})
        faulted(script, main, f"sha512:t0-{pr}", undeclared, blocked=1)
        faulted(script, sha, f"sha512:t1-{pr}", undeclared + (", z.txt" if adds else ""), blocked=1)
        acts = [a for _ in range(4) for a in _acts(loop.tick())]
        if adds:
            assert "eject" in acts and env.main() != sha
        else:
            assert "advance" in acts and "eject" not in acts and env.main() == sha


def test_a_family_removal_holds_every_pr_batched_with_it_and_rings_infra_not_its_owner(env):
    """Cost 3 at b85656625c: a PR that removes a family (here lean-lock's lister) makes its candidate list-failed, `lacks kind
    lean-lock`. decide reads list-failed as infra: retried at every reading and rung once, as "check failed to list", to
    @infra alone (`Loop._ring`), never to the PR's owner or ci and without the kind it lacks; the PR batched with it is held
    too, since nothing splits a list-failed candidate. A run-book "lands it by hand" must still be a candidate whose check
    passed, never a push to `main`."""
    env.gates = ["pytest", "lean-audit"]
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"tools/verity/graph/verity_graph/families/lean_lock.py": ""})
    loop = env.loop(decide)
    (built,) = loop.tick()
    sha, main = built["sha"], built["base"]
    assert built["members"] == [1, 2]
    from test_vqueue_loop import _graph

    lean = [{"id": "lean-module:M", "kind": "lean-module", "inputs": ["M.lean"], "key": "sha512:m"},
            {"id": "lean-lock:L", "kind": "lean-lock", "inputs": ["L.json"], "key": "sha512:l"}]
    env.graph2.list_graph(f"ready:{main}", _graph(main, [{"id": "pytest:a", "kind": "test-file", "inputs": ["a.txt"],
                                                          "key": "sha512:a0"}, *lean]))
    env.graph2.list_graph(f"ready:{sha}", _graph(sha, [{"id": "pytest:a", "kind": "test-file", "inputs": ["a.txt"],
                                                        "key": "sha512:a1"}, lean[0]]))
    env.result("sha512:a1", "pytest:a", "pass")
    env.result("sha512:m", "lean-module:M", "pass")
    acts = [a for _ in range(12) for a in _acts(loop.tick())]
    assert "advance" not in acts and "eject" not in acts and env.main() == main and acts.count("ring") == 1
    rings = [r for r in env.lines() if r.get("action") == "posting" and r.get("posting") == "ring"]
    assert [r["to"] for r in rings] == [["@infra"]] and "failed to list" in rings[0]["text"]
    assert "lean-lock" not in rings[0]["text"]
