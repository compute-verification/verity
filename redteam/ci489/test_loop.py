"""`research.vqueue_loop` (ci's W3b) with `research.vqueue.decide` (W3a) on ci's own harness: a throwaway origin, a fake
GitHub, graph2 as files (`test_vqueue_loop.Env`).

It runs inside ci's tests directory, whose conftest isolates the store and git: in a checkout of `cursor/ci-vqueue-loop` with
`cursor/ci-vqueue-core`'s vqueue.py in place (the merge ci plans),

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


def test_a_failed_enqueue_after_the_push_leaves_its_candidate_pending_forever(env, monkeypatch):
    """FINDING (crash safety, fix before the cut-over): Build pushes `queue/<sha12>`, then enqueues `ready:<sha>`. When the
    enqueue fails (graph2's database locked, the CLI missing, a restart between the two), the tick stops with the ref on
    origin and no build record. Every later tick finds that ref, reads it `pending` (graph2: "not requested"), and `decide`
    waits on a pending candidate without acting, so nothing ever enqueues it: the queue stops, silently. FIX: each tick
    enqueues every live candidate that isn't terminal (graph2's enqueue is idempotent), or a candidate graph2 says is "not
    requested" is a Retry."""
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
    for _ in range(30):
        assert loop.tick() == []
    assert calls == [sha] and env.graph2.enqueued() == []
    assert loop.world.verdict(sha).verdict.state == "pending"


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


def test_a_queue_ref_the_loop_didnt_build_displaces_its_candidate_and_lands_what_no_pr_has(env):
    """FINDING (soundness, fix before the cut-over): the loop takes every `queue/*` branch on origin as a candidate whose
    first-parent commits parse as `Queue: #N` merges (`queue_commit` reads only the message's trailers: not that the second
    parent is `Head`, nor the tree). A ref pushed by anyone (an agent, a stale clone, a test against the real origin) that
    claims a better-ranked member displaces the loop's own candidate ("a second candidate on main"); never enqueued, it
    stalls the queue; once its check passes (anyone can enqueue `ready:<sha>` on the pod, or label results: see
    `test_a_result_label_anyone_writes_counts_as_the_tasks_pass`), `main` fast-forwards to a tree holding a file no PR has.
    FIX: adopt only candidates with the loop's own build record (sha, base, members, heads), else drop them; check that each
    `Queue: #N` merge's second parent is its `Head` and its tree is `merge-tree`'s; a ruleset gives `refs/heads/queue/*` to
    drive's App alone."""
    one = env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    main = env.main()
    loop = env.loop(decide)
    (built,) = loop.tick()
    assert built["members"] == [1, 2]
    forged = forge(env, f"{built['sha']}^1", [main, one], {"evil.txt": "no PR adds this\n"})
    acts = loop.tick()
    assert ("drop", built["sha"]) in [(r["action"], r["sha"]) for r in acts]
    assert [e["check"] for e in env.graph2.enqueued()] == [f"ready:{built['sha']}", *(f"ready:{r['sha']}" for r in acts
                                                                                     if r["action"] == "build")]
    keys = env.listed(forged, {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    assert "advance" in _acts(loop.tick())
    assert env.main() == forged and "evil.txt" in _git(env.o.origin, "ls-tree", "--name-only", forged).split()


def test_a_failing_lean_agreement_task_doesnt_hold_a_candidate(env):
    """FINDING (coverage, blocks the cut-over): the loop leaves out of the gate every task of a `drive.SLIM_STEPS` job
    (lean-agreement, ratchet-gate), as drive's `executor = "on"` path does, but it doesn't start the slim train.sh run that
    path judges them and the preflights by (`Driver._start_slim`/`_follow_slim`; nothing in vqueue_loop.py). So a candidate
    whose lean-agreement task fails against upstream's Rust verifier advances, and `uv lock --check` and fixture fetchability
    run nowhere before `main`. FIX: start the slim run at each candidate's sha and fold its steps into the verdict (pass only
    when both pass), as ci's 07:08Z comment says; or list lean-agreement and the preflights as graph tasks that gate."""
    env.put(1, {"verity/ml/flock/verifier/check.py": "1\n"})
    loop = env.loop(decide)
    (built,) = loop.tick()
    sha = built["sha"]
    task = {"env": "py", "argv": ["t"], "vars": {}, "reads": [], "outputs": []}
    env.graph2.list_graph(f"ready:{sha}", {
        "schema": GRAPH_V2, "commit": sha, "envs": {"py": {"digest": "sha512:env"}},
        "files": {"verity/ml/flock/verifier/check.py": "sha512:v", "t/test_a.py": "sha512:t"},
        "tasks": [{"id": "test-file:t/test_a.py", "kind": "test-file", "inputs": ["t/test_a.py"], "key": "sha512:a", **task},
                  {"id": "lean-agreement:verity/ml/flock/verifier", "kind": "lean-agreement",
                   "inputs": ["verity/ml/flock/verifier/check.py"], "key": "sha512:agree", **task}]})
    env.result("sha512:a", "test-file:t/test_a.py", "pass")
    env.result("sha512:agree", "lean-agreement:verity/ml/flock/verifier", "fail")
    assert loop.world.verdict(sha).verdict.state == "pass"
    assert _acts(loop.tick()) == ["advance", "drop"] and env.main() == sha


def test_a_result_label_anyone_writes_counts_as_the_tasks_pass(tmp_path):
    """FINDING (soundness, fix before the cut-over): the loop's verdict reads graph2's results from the store: every `result`
    label on `task:<key>`, by whoever wrote it (`research data label --by` is self-asserted), folded pass once any says pass
    and none fail; and the graph a candidate is judged by is the newest `verity_graph` label on its commit, by anyone. A pass
    written before graph2's own result lands is enough for `main` to move. FIX: the loop runs on graph2's pod: read the listed
    graph and the results from graph2's database (or its `verdict` with per-task states), not the store; at least, count only
    labels by graph2's asserter."""
    from research.store.local import LocalStore
    from research.vqueue_loop import Transport

    store = LocalStore(tmp_path / "store")
    key = "sha512:" + "a" * 128
    store.add_label(f"task:{'a' * 128}", "result", {"schema": "verity-result/v1", "key": key, "task": "test-file:t.py",
                                                     "status": "pass", "outputs": {}}, by="not-graph2")
    graph = {"schema": GRAPH_V2, "commit": "c" * 40, "envs": {"py": {"digest": "sha512:env"}}, "files": {"t.py": "sha512:t"},
             "tasks": [{"id": "test-file:t.py", "kind": "test-file", "env": "py", "argv": ["t"], "vars": {}, "inputs": ["t.py"],
                        "reads": [], "outputs": [], "key": key}]}
    assert Executor(Transport(store)).settle(graph)["test-file:t.py"][0] == "pass"
