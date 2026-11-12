"""`research queue vqueue` (`research.vqueue_loop`): the verified queue's loop on a throwaway origin and a clone of it, graph2's
listings and results as `FileTransport` files, a fake GitHub and fake labels, and a small policy in place of `vqueue.decide`.
Every test reads the clock it injects."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import research.merge as M
import research.queue as Q
import research.vqueue_loop as L
from research import carry, vqueue, vqueue_records, vqueue_slim
from research.executor import GRAPH_V2, Executor, FileTransport
from research.landing import PREFLIGHT
from research.queue import DRIVE_FLAGS
from research.store import vocab
from research.vqueue import (FAIL, INFRA, LIST_FAILED, PASS, PENDING, Advance, Build, Drop, Eject, Outcome, RecordsRun, Red,
                             Rerun, Retry, Ring, Snapshot, Verdict)
from research.vqueue_loop import Loop, World, _Clone, _said, branch
from research.vqueue_records import lock_message
from test_queue import World as Origin, _git, _landed_move, _merge

T0 = 1_900_000_000.0
#: the env's spec files in each listed graph
SPEC = ("pyproject.toml", "uv.lock")


@dataclass(frozen=True)
class Lab:
    by: str
    value: object
    ts: str
    ref: str | None = None


class Labels:
    """A PR's ready mark is the fake GitHub's (`ready`), and each marked head has a land verdict by `reviewer` unless ``held``,
    but a head ``verdicts`` gives its own ({(PR, head): its `reviewed` labels}), which are its PR's other heads' too."""

    def __init__(self, gh):
        self.gh, self.held, self.verdicts = gh, set(), {}

    def ready(self, n: int, head: str) -> bool:
        return (n, head) in self.gh.ready

    def reviews(self, n: int, head: str) -> list[Lab]:
        if (n, head) in self.verdicts:
            return list(self.verdicts[n, head])
        return [Lab("reviewer", "hold" if n in self.held else "land", "2026-01-01T00:00:00Z")]

    def _other_heads(self, n: int, head: str, key: str) -> dict[str, list[Lab]]:
        return {h: list(labs) for (m, h), labs in self.verdicts.items()
                if key == vocab.REVIEWED and m == n and h != head and labs}

    def ready_by(self, n: int, head: str) -> str:
        return f"lane-{n}"

    marked_by = ready_by

    def marked_at(self, n: int, head: str) -> str | None:
        return f"2026-01-01T00:00:{n:02d}Z" if (n, head) in self.gh.ready else None

    def before_cutoff(self, n: int, head: str) -> str | None:
        return None

    def prefetch(self, heads) -> None:
        return None


class Fake(World):
    """The world with labels faked: an eject withdraws the fake mark, and a run's cancel is kept in ``cancelled``."""

    withdrawn: list
    cancelled: list

    def withdraw(self, number: int, head: str, why: str):
        self.withdrawn.append((number, head))
        self.q.github.ready.discard((number, head))

    def cancel_runs(self, ids, why: str) -> dict[str, str]:
        self.cancelled.append((list(ids), why))
        return {}


class Env:
    """`main` waits for `pytest` alone (`gating`, which ``gates`` sets), and the first `main`'s check is listed with no task, so
    a candidate's failures there are its own."""

    def __init__(self, tmp: Path, monkeypatch):
        self.o, self.tmp = Origin(tmp), tmp
        self.labels, self.graph2 = Labels(self.o.gh), FileTransport(tmp / "graph2")
        self.log, self.clock, self.posts, self.out, self.withdrawn = tmp / "vqueue" / "decisions.jsonl", T0, [], [], []
        self.cancelled: list = []
        self.gates = ["pytest"]
        monkeypatch.setattr(M, "caught_up", lambda *a, **k: None)
        monkeypatch.setattr(M, "_sweep_after", lambda *a, **k: None)
        monkeypatch.setattr(L, "gating", lambda repo, sha: list(self.gates))
        self.graph2.list_graph(f"ready:{self.main()}", {"schema": GRAPH_V2, "commit": self.main(), "tasks": []})

    def loop(self, decide, clone: str = "clone", **kw) -> Loop:
        path = self.tmp / clone
        if not path.exists():
            _git(self.tmp, "clone", "-q", str(self.o.origin), str(path))
        q = _Clone(path, remote=str(self.o.origin), github=self.o.gh, ready=self.labels.ready, log=self.out.append,
                   marked_by=self.labels.marked_by, marked_at=self.labels.marked_at, prefetch=self.labels.prefetch,
                   land_into="main")
        q.branch = "main"
        w = Fake(q, self.labels, Executor(self.graph2), post=lambda text, to=(): self.posts.append(text) or True,
                 now=lambda: self.clock)
        w.withdrawn, w.cancelled = self.withdrawn, self.cancelled
        return Loop(w, self.log, decide=decide, now=lambda: self.clock, out=self.out.append, **kw)

    def put(self, n: int, files: dict[str, str]) -> str:
        self.o.gh.put(n, head := self.o.change(files))
        return head

    def main(self) -> str:
        return _git(self.o.origin, "rev-parse", "refs/heads/main")

    def queued(self) -> dict[str, str]:
        out = _git(self.o.origin, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads/queue/")
        return dict(line.split() for line in out.splitlines())

    def lines(self) -> list[dict]:
        return [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.is_file() else []

    def listed(self, sha: str, tasks: dict[str, list[str]], reads: dict[str, str] | None = None,
               keys: dict[str, str] | None = None, suites: dict[str, str] | None = None) -> dict[str, str]:
        """graph2's listing of `ready:<sha>`: a task of each ``tasks`` with those inputs (and the suite of ``suites``), in an env
        whose spec is `SPEC`, keyed (by ``keys``, else by ``sha``) unless it reads another (``reads``: {task: the task it
        reads, whose output `out` it reads})."""
        reads, keys, suites = reads or {}, keys or {}, suites or {}
        graph = {"schema": GRAPH_V2, "commit": sha,
                 "files": {f: f"sha512:{f}" for f in [*SPEC, *(f for ins in tasks.values() for f in ins)]},
                 "envs": {"py": {"digest": "sha512:env", "spec": list(SPEC)}},
                 "tasks": [{"id": i, "kind": "test-file", "env": "py", "argv": ["t", i],
                            "vars": {"suite": suites[i]} if i in suites else {}, "inputs": ins,
                            "reads": [{"task": reads[i], "output": "out"}] if i in reads else [],
                            "outputs": [{"name": "out", "path": f"{i}.out"}],
                            **({} if i in reads else {"key": keys.get(i, f"sha512:{sha[:8]}-{i}")})} for i, ins in tasks.items()]}
        self.graph2.list_graph(f"ready:{sha}", graph)
        return {t["id"]: t.get("key", "") for t in graph["tasks"]}

    def result(self, key: str, task: str, status: str, infra: bool = False, tests: list[str] | None = None) -> None:
        detail = {**({"infra": True} if infra else {}), **({"failed_tests": tests} if tests is not None else {})}
        self.graph2.finish_result({"key": key, "task": task, "status": status, "outputs": {"out": "sha512:o"},
                                   **({"detail": detail} if detail else {})})

    def graph2_script(self, loop: Loop, verdict: dict | None = None) -> Path:
        """A graph2 with `verdict`, `cancel` and `rerun` for ``loop``: its verdict on a check `<script>.verdict.<check>`'s,
        else ``verdict`` (default a fail), each check asked appended to `<script>.asked`, and each cancel and rerun to
        `<script>.cancel` and `<script>.rerun`."""
        (script := self.tmp / "graph2.py").write_text(GRAPH2)
        if verdict is not None:
            Path(f"{script}.verdict").write_text(json.dumps(verdict))
        loop.world.graph2 = (sys.executable, str(script))
        return script


@pytest.fixture
def env(tmp_path, monkeypatch) -> Env:
    return Env(tmp_path, monkeypatch)


def policy(s: Snapshot) -> list:
    """A small policy in `vqueue.decide`'s place: drop a candidate on `main`, advance one that passed on it, eject the one member
    a failure's inputs meet and drop the candidate, retry infra until `rings` enqueues and then ring, and with no candidate
    build the ready PRs that merge onto `main`."""
    acts: list = []
    files = {x.number: x.files for x in s.ready}
    for c in s.candidates:
        if c.sha == s.main:
            acts.append(Drop(c.sha, "on main", Outcome(c.members, c.heads, PASS)))
        elif c.verdict.state == PASS and c.base == s.main:
            acts.append(Advance(c.sha))
        elif c.verdict.state == FAIL:
            bad = {p for f in c.verdict.failed for p in f.inputs}
            suspects = tuple(n for n in c.members if files.get(n, frozenset()) & bad)
            acts += [Eject(n, h, "its change fails the check", c.verdict.failed)
                     for n, h in zip(c.members, c.heads) if (n,) == suspects]
            acts.append(Drop(c.sha, "failed", Outcome(c.members, c.heads, FAIL, suspects)))
        elif c.verdict.state in (INFRA, LIST_FAILED):
            acts.append(Retry(c.sha) if c.tries < s.rings else Ring(c.sha, "out of tries"))
    if not s.candidates and (members := tuple(x.number for x in s.ready if not x.conflict)[:s.batch]):
        acts.append(Build(members, s.main, 5))
    return acts


def _acts(recs: list[dict]) -> list[str]:
    return [r["action"] for r in recs]


def test_a_candidate_that_passed_fast_forwards_main_to_itself_and_its_branch_is_deleted(env):
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    loop = env.loop(policy)
    (built,) = loop.tick()
    sha = built["sha"]
    assert (built["action"], built["members"], built["base"], built["priority"]) == ("build", [1, 2], main, 5), built
    assert env.queued() == {branch(sha): sha} and env.main() == main
    assert env.graph2.enqueued() == [{"check": f"ready:{sha}", "priority": 5, "graph": None, "schema": GRAPH_V2}]
    keys = env.listed(sha, {"pytest:a": ["a.txt"], "pytest:b": ["b.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    assert loop.tick() == []                                    # pytest:b pending
    env.result(keys["pytest:b"], "pytest:b", "pass")
    assert _acts(loop.tick()) == ["advance", "on_main"]                     # drive's steps once it is on `main`, logged after
    assert env.main() == sha and _git(env.o.origin, "rev-parse", f"{sha}^1^1") == main      # fast-forwarded to it, only
    (drop,) = loop.tick()
    assert drop["action"] == "drop" and drop["outcome"] == {"members": [1, 2], "heads": built["heads"], "state": PASS,
                                                            "suspects": []}
    assert env.queued() == {} and env.main() == sha and loop.tick() == []
    assert _git(env.o.origin, "for-each-ref", "--format=%(refname)", "refs/heads/") == "refs/heads/main"
    assert [r["tick"] for r in env.lines()] == [1, 3, 3, 4] and not any(r["shadow"] for r in env.lines())
    assert env.posts == [] and env.withdrawn == []


def test_a_failed_candidate_ejects_the_member_its_failed_inputs_meet_and_the_rest_are_built_again(env):
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    two = env.put(2, {"b.txt": "2\n"})
    env.put(3, {"c.txt": "3\n"})
    loop = env.loop(policy)
    (built,) = loop.tick()
    keys = env.listed(built["sha"], {"pytest:a": ["a.txt"], "pytest:b": ["b.txt"], "pytest:c": ["c.txt"]})
    for task, status in (("pytest:a", "pass"), ("pytest:b", "fail"), ("pytest:c", "pass")):
        env.result(keys[task], task, status)
    eject, drop = loop.tick()
    assert (eject["action"], eject["number"], eject["heads"]) == ("eject", 2, [two])
    assert eject["failed"] == [{"task": "pytest:b", "key": keys["pytest:b"]}] and env.withdrawn == [(2, two)]
    (post,) = env.posts
    assert "#2 at" in post and "pytest:b" in post and keys["pytest:b"] in post, post
    assert drop["action"] == "drop" and drop["outcome"]["state"] == FAIL and drop["outcome"]["suspects"] == [2]
    assert env.queued() == {} and env.main() == main
    (again,) = loop.tick()
    assert again["action"] == "build" and again["members"] == [1, 3] and again["base"] == main
    assert env.queued() == {branch(again["sha"]): again["sha"]}


def test_a_failed_candidate_blames_a_member_its_owner_took_out_in_flight_by_its_files_as_built_and_ejects_nobody(env):
    env.put(1, {"a.txt": "1\n"})
    two = env.put(2, {"b.txt": "2\n"})
    loop = env.loop(vqueue.decide)
    (built,) = loop.tick()
    env.o.gh.ready.discard((2, two))                                # its owner withdrew its mark
    keys = env.listed(built["sha"], {"pytest:a": ["a.txt"], "pytest:b": ["b.txt"]})
    for task, status in (("pytest:a", "pass"), ("pytest:b", "fail")):
        env.result(keys[task], task, status)
    recs = loop.tick()
    (drop,) = [r for r in recs if r["action"] == "drop"]
    assert drop["outcome"]["state"] == FAIL and drop["outcome"]["suspects"] == [2], drop
    assert "eject" not in _acts(recs) and env.withdrawn == [] and env.posts == []
    assert [r["members"] for r in recs if r["action"] == "build"] == [[1]], recs


def test_only_another_s_hold_or_eject_pulls_a_pr_and_a_pr_holding_a_pulled_pr_s_head_is_pulled_with_it(env):
    """`World.pulled`: a hold by its owner or its owner's eject leaves a candidate in flight to land; a reviewer's hold or a
    red's eject (`EJECTOR`, even of a PR `EJECTOR` owns, whose own withdrawn mark doesn't pull it) takes the PR, and a PR whose
    head holds it with it, as it carries its change."""
    one = env.put(1, {"a.txt": "1\n"})
    env.o.gh.put(2, env.o.change({"b.txt": "2\n"}, base=one))
    by: dict[tuple[int, str], str] = {}
    env.labels.marked_by = lambda n, h: by.get((n, h), f"lane-{n}")
    loop = env.loop(lambda s: [])
    w, main, reviews = loop.world, env.main(), env.labels.reviews
    loop.tick()
    env.labels.reviews = lambda n, h: [*reviews(n, h), *([Lab("@Lane-1", "hold", "2026-01-02T00:00:00Z")] if n == 1 else [])]
    assert w.ready(main) == [] and w.left[1][1] == "held by Lane-1 (no reason)" and w.pulled == set()
    env.labels.held.add(1)
    assert w.ready(main) == [] and w.pulled == {1, 2}
    env.labels.held.clear()
    env.labels.reviews = reviews
    w.q.eject(1, "mine to fix", "lane-1")
    assert 1 not in [r.number for r in w.ready(main)] and w.pulled == set()
    w.q.eject(1, "pytest:a failed", L.EJECTOR)
    assert w.ready(main) == [] and w.pulled == {1, 2}
    env.o.gh.put(3, three := env.o.change({"c.txt": "3\n"}), body=f"Lane `{L.EJECTOR}`.")
    by[(3, three)] = L.EJECTOR
    env.o.gh.ready.discard((3, three))
    assert w.fetch() == main and w.ready(main) == [] and w.left[3][1] == f"its ready mark was withdrawn by {L.EJECTOR}" and w.pulled == {1, 2}
    env.o.gh.ready.add((3, three))
    w.q.eject(3, "pytest:c failed", L.EJECTOR)
    assert w.ready(main) == [] and w.pulled == {1, 2, 3}


def test_an_ejection_for_a_failed_lean_lock_task_is_told_as_any_failure_with_no_word_of_its_own_on_regenerating_records(env):
    """The records run before a Lean candidate's check regenerates its records, so a lean-lock failure is the change's own,
    told as any failure is: the task, its key and where its log is."""
    env.put(1, {"a.lean": "1\n"})
    two = env.put(2, {"b.txt": "2\n"})
    loop = env.loop(policy)
    (built,) = loop.tick()
    keys = env.listed(built["sha"], {"lean-lock:Security": ["a.lean"], "pytest:b": ["b.txt"]})
    env.result(keys["lean-lock:Security"], "lean-lock:Security", "fail")
    env.result(keys["pytest:b"], "pytest:b", "pass")
    eject, _ = loop.tick()
    assert (eject["action"], eject["number"]) == ("eject", 1)
    (post,) = env.posts
    assert f"Failed: lean-lock:Security (key {keys['lean-lock:Security']})" in post, post
    assert post.endswith("Fix it and mark it ready again") and "regenerate" not in post, post
    env.put(3, {"c.txt": "3\n"})
    (again,) = loop.tick()
    keys = env.listed(again["sha"], {"pytest:b": ["b.txt"], "pytest:c": ["c.txt"]})
    env.result(keys["pytest:b"], "pytest:b", "fail")
    env.result(keys["pytest:c"], "pytest:c", "pass")
    loop.tick()
    assert "lean-lock" not in env.posts[-1] and two[:12] in env.posts[-1], env.posts[-1]


def test_a_retry_sets_its_infra_results_aside_so_the_check_is_pending_until_a_newer_one_and_infra_is_rung_once(env):
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy)
    (built,) = loop.tick()
    sha = built["sha"]
    keys = env.listed(sha, {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "error", infra=True)
    (retry,) = loop.tick()
    assert (retry["action"], retry["enqueued"], retry["counts"]) == ("retry", True, {keys["pytest:a"]: 1})
    assert retry["failed"] == [{"task": "pytest:a", "key": keys["pytest:a"]}]
    assert loop.tick() == [] and len(env.graph2.enqueued()) == 2       # the error it retried is set aside: pending
    env.result(keys["pytest:a"], "pytest:a", "error", infra=True)
    (ring,) = loop.tick()                                       # infra again after `rings` enqueues: the policy rings
    assert ring["action"] == "ring" and ring["done"] and env.posts[-1].startswith("@infra vqueue:"), env.posts
    assert loop.tick() == [] and len(env.posts) == 1 and len(env.graph2.enqueued()) == 2       # once for each why
    env.result(keys["pytest:a"], "pytest:a", "pass")
    assert _acts(loop.tick()) == ["advance", "on_main"] and env.main() == sha


@pytest.mark.parametrize("said", [INFRA, PENDING, FAIL])
def test_a_retry_reruns_its_infra_keys_itself_while_graph2_s_own_verdict_isnt_infra(env, said):
    """graph2's enqueue reruns a check's infra keys only while its own verdict is infra: with a task still pending, or one
    failed (here at its base's key, so the loop's reading is infra), the retry reruns them."""
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide)
    script = env.graph2_script(loop, {"state": said, "failed": [{"task": "pytest:r", "key": "sha512:red"}] if said == FAIL
                                      else []})
    (built,) = loop.tick()
    sha, red = built["sha"], {"pytest:r": "sha512:red"}
    keys = env.listed(sha, {"pytest:a": ["a.txt"], **({"pytest:b": ["b.txt"]} if said == PENDING else {}),
                            **({"pytest:r": ["red.txt"]} if said == FAIL else {})}, keys=red)
    env.listed(main, {"pytest:r": ["red.txt"]}, keys=red)
    env.result("sha512:red", "pytest:r", "fail")
    env.result(a := keys["pytest:a"], "pytest:a", "error", infra=True)
    (retry,) = loop.tick()
    assert (retry["action"], retry["counts"], len(env.graph2.enqueued())) == ("retry", {a: 1}, 2)
    if said == INFRA:
        assert not Path(f"{script}.rerun").exists() and "rerun" not in retry
    else:
        assert Path(f"{script}.rerun").read_text() == f"ready:{sha} {a}\n" and retry["rerun"] == f"ready:{sha} requeued 1"
    assert loop.tick() == []


def test_a_member_that_conflicts_with_one_merged_before_it_is_left_out_naming_it_and_its_owner_told_once(env):
    """#2 merges onto `main` but not after #1, which the build merged first: its skip names #1, not `main`, nor #3, merged
    before it too in a file #2 doesn't touch."""
    env.put(1, {"a.txt": "1\n"})
    env.put(3, {"d.txt": "3\n"})
    two = env.put(2, {"a.txt": "2\n"})
    loop = env.loop(lambda s: [Drop(c.sha, "rebuilt") for c in s.candidates] or [Build((1, 3, 2), s.main, 5)])
    skip, built = loop.tick()
    assert (skip["action"], skip["number"], skip["heads"], skip["done"], skip["by"]) == ("skip", 2, [two], True, [1])
    assert skip["why"] == "conflicts with #1, merged before it, in a.txt (it merges onto `main`)", skip
    assert built["members"] == [1, 3], built
    (post,) = env.posts
    assert "#2 at" in post and "doesn't merge" in post and "conflicts with `main`" not in post, post
    assert post.endswith("It is left out until it merges: merge `main` into it once #1 is on it"), post
    assert _acts(loop.tick()) == ["drop"]
    again, built = loop.tick()                                      # the same head onto the same base: told once
    assert (again["action"], again["text"], again["done"], built["action"]) == ("skip", None, False, "build")
    assert len(env.posts) == 1


@pytest.mark.parametrize("merges", [True, False])
def test_a_rebuild_on_a_candidate_the_same_tick_rebuilt_goes_on_what_that_rebuild_pushed(env, merges):
    """A red drops its candidate and the one above it, and the tick builds each again (`vqueue.Build.replaces`): the upper
    rebuild names the lower candidate as its base and goes on what the lower rebuild pushed, or, with none of the lower one's
    members merging, on what that one went on.  Each build's line names the candidate it replaces, and a member that
    doesn't merge onto what the upper one goes on (#5, on #1) is skipped there."""
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    env.put(3, {"c.txt": "3\n"})
    env.o.gh.put(4, env.o.change({"m.txt": "4\n"}, base=main))
    five = env.put(5, {"a.txt": "5\n"})
    _main_to(env, {"m.txt": "main\n"})
    main = env.main()
    lower = (1,) if merges else (4,)

    def policy(s: Snapshot) -> list:
        on = {c.members: c.sha for c in s.candidates}
        if not on:
            return [Build((1, 2), s.main, 5)]
        if (1, 2) not in on:
            return []
        if (3,) not in on:
            return [Build((3,), on[1, 2], 4)]
        return [Drop(on[1, 2], "failed"), Drop(on[3,], "its base failed"), Build(lower, s.main, 5, replaces=on[1, 2]),
                Build((3, 5), on[1, 2], 4, replaces=on[3,])]

    loop = env.loop(policy)
    low, up = loop.tick()[-1], loop.tick()[-1]
    logged = loop.tick()
    again, top = [r for r in logged if r["action"] == "build"]
    on = again["sha"] if merges else main
    assert (again["base"], again["replaces"]) == (main, low["sha"]), again
    assert (top["base"], top["members"], top["replaces"]) == (on, [3] if merges else [3, 5], up["sha"]), top
    assert loop.world._changed(main, top["sha"]) == {"a.txt", "c.txt"}             # #1's a.txt, or #5's
    skips = [(r["number"], r["base"], f"candidate on {on[:12]}:" in r["text"]) for r in logged
             if r["action"] == "skip" and r["heads"] == [five]]
    assert skips == ([(5, on, True)] if merges else [])


def test_a_member_that_conflicts_with_a_candidate_below_names_its_member_and_one_that_conflicts_with_main_says_main(env):
    """Built on a candidate, a PR that merges onto `main` but not onto the candidate conflicts with the candidate's member,
    named; one that conflicts with `main` itself is told so, whatever was merged before it."""
    base = env.main()
    _main_to(env, {"c.txt": "main\n"})
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"a.txt": "2\n"})
    three = env.o.change({"c.txt": "3\n"}, base=base)
    env.o.gh.put(3, three)
    loop = env.loop(lambda s: [Build((2, 3), s.candidates[0].sha, 5)] if s.candidates else [Build((1,), s.main, 5)])
    told, low = loop.tick()                                         # #3 doesn't merge onto `main`: its owner is told so
    assert (told["action"], told["number"], low["members"]) == ("conflict", 3, [1]), (told, low)
    skip2, skip3, none = loop.tick()
    assert (skip2["number"], skip2["by"], none["action"], none["base"]) == (2, [1], "build", low["sha"]), (skip2, none)
    assert skip2["why"] == "conflicts with #1, merged before it, in a.txt (it merges onto `main`)", skip2
    assert (skip3["number"], skip3["why"], "by" in skip3) == (3, "conflicts with `main` in c.txt", False), skip3
    assert [p.rsplit(": ", 1)[1] for p in env.posts[1:]] == ["merge `main` into it once #1 is on it", "merge `main` into it"]


def test_a_member_whose_head_what_it_merges_onto_holds_is_left_out_untold_and_then_logged_once_as_riding(env):
    """A member whose head an earlier member's holds merges nothing: it reaches `main` with the build, so its owner isn't told
    it doesn't merge (`held`), and while that candidate is on `main`'s chain the PR is no ready PR: no build takes it, so
    nobody is told to merge `main` into it, and it is logged once a head and candidate (`riding`)."""
    one = env.put(1, {"a.txt": "1\n"})
    env.o.gh.put(2, env.o.change({"b.txt": "2\n"}, base=one))
    loop = env.loop(lambda s: [] if s.candidates else [Build((2, 1), s.main, 5)])
    skip, built = loop.tick()
    assert (skip["action"], skip["number"], skip["heads"], skip["held"], skip["done"]) == ("skip", 1, [one], True, False)
    assert skip["why"] == f"its head {one[:12]} is already in what it merges onto: #2's head holds it" and "text" not in skip
    assert built["members"] == [2] and env.posts == []
    (riding,) = loop.tick()
    assert (riding["action"], riding["number"], riding["heads"], riding["sha"]) == ("riding", 1, [one], built["sha"])
    assert riding["why"] == f"{built['sha'][:12]} holds its head without it as a member: it reaches `main` with it"
    assert loop.tick() == [] and env.posts == []
    assert [r.number for r in loop.snapshot(L.Past.read(env.log, 0), 9).snap.ready] == [2]


def test_a_pr_riding_in_a_candidate_after_its_own_mark_lands_with_it_and_is_never_built_or_told(env):
    """A PR stacked on an unmarked PR of the same owner carries that PR's change (`Queue.entries`), so it is built alone; once
    the PR below is marked, its head is a ready PR the candidate holds: no build takes it, nobody is told, and it reaches
    `main` with the candidate."""
    one = env.put(1, {"a.txt": "1\n"})
    env.o.gh.put(2, two := env.o.change({"b.txt": "2\n"}, base=one), body="Lane `lane-1`.")
    env.o.gh.ready.discard((1, one))
    loop = env.loop(vqueue.decide)
    (built,) = loop.tick()
    assert (built["members"], built["heads"]) == ([2], [two]), built
    env.o.gh.ready.add((1, one))
    (riding,) = loop.tick()
    assert (riding["action"], riding["number"], riding["sha"]) == ("riding", 1, built["sha"]), riding
    assert loop.tick() == [] and env.posts == []
    keys = env.listed(built["sha"], {"pytest:b": ["b.txt"]})
    env.result(keys["pytest:b"], "pytest:b", "pass")
    assert _acts(loop.tick()) == ["advance", "on_main", "drop"] and env.main() == built["sha"]
    assert loop.tick() == [] and env.posts == []


def test_a_skip_for_a_head_main_holds_when_fetched_again_tells_nobody(env):
    """`main` moved to #2's head after the tick read it: #2's skip tells nobody, and is logged with the `main` that holds it."""
    env.put(1, {"a.txt": "1\n"})
    two = env.put(2, {"a.txt": "2\n"})

    def moved(s: Snapshot) -> list:
        if s.candidates:
            return []
        _git(env.o.origin, "update-ref", "refs/heads/main", two)
        return [Build((1, 2), s.main, 5)]

    skip, built = env.loop(moved).tick()
    assert (skip["action"], skip["number"], skip["landed"], skip["done"], "text" in skip) == ("skip", 2, two, False, False)
    assert skip["why"] == ("conflicts with #1, merged before it, in a.txt (it merges onto `main`); "
                           f"`main` {two[:12]}, fetched again, holds its head"), skip
    assert built["members"] == [1] and env.posts == []


def test_a_skip_whose_fetch_of_main_again_fails_reads_the_tick_s_main_and_is_told(env):
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"a.txt": "2\n"})
    loop = env.loop(lambda s: [] if s.candidates else [Build((1, 2), s.main, 5)])

    def unreachable() -> str:
        raise L.QueueError("git fetch: origin can't be reached")

    loop.world.main_now = unreachable
    skip, built = loop.tick()
    assert (skip["number"], skip["done"], len(env.posts), built["members"]) == (2, True, 1, [1]), (skip, built)
    assert any("fetching `main` again failed" in x for x in env.out), env.out


def test_a_conflict_notice_for_a_head_main_holds_when_fetched_again_tells_nobody(env):
    """A ready PR that doesn't merge onto the `main` the tick read, whose head `main` holds once fetched again (it landed
    meanwhile), is told nothing, and is no ready PR the next tick."""
    base = env.main()
    _main_to(env, {"c.txt": "main\n"})
    four = env.o.change({"c.txt": "4\n"}, base=base)
    env.o.gh.put(4, four)
    resolved = env.o.change({"c.txt": "resolved\n"}, base=four)

    def landed(s: Snapshot) -> list:
        _git(env.o.origin, "update-ref", "refs/heads/main", resolved)
        return []

    loop = env.loop(landed)
    (conflict,) = loop.tick()
    assert (conflict["action"], conflict["number"], conflict["landed"], conflict["done"]) == ("conflict", 4, resolved, False)
    assert "text" not in conflict and env.posts == []
    assert loop.tick() == [] and loop.world.left == {4: (four, "its head is on `main`")} and env.posts == []


def _move(env: Env, n: int, files: dict[str, str | None], base: str = "origin/main") -> str:
    """Move PR ``n`` (its body names a word map: `PR.move`), its head writing ``files`` on ``base``; returns the head."""
    env.o.gh.put(n, head := env.o.change(files, base=base), body=f"Lane `lane-{n}`.\nMove: tools/move/x_map.toml\n")
    return head


def _main_to(env: Env, files: dict[str, str]) -> str:
    """`main` moved to a commit writing ``files``, its check listed with no task as `Env` lists the first `main`'s."""
    _git(env.o.origin, "update-ref", "refs/heads/main", main := env.o.change(files))
    env.listed(main, {})
    return main


def test_a_move_pr_with_a_land_verdict_is_built_by_its_merge_and_advances_main_as_any_pr(env):
    """A move PR with a land verdict whose move can't be regenerated (its map isn't on `main`) has its head merged onto `main`
    as it is: the loop builds it as it builds any PR."""
    main = env.main()
    head = _move(env, 1, {"c.txt": None, "moved/c.txt": "c\n", "a.txt": "moved/c\n"})
    loop = env.loop(policy)
    (built,) = loop.tick()
    sha = built["sha"]
    assert (built["action"], built["members"], built["heads"]) == ("build", [1], [head]), built
    assert _git(env.o.origin, "rev-parse", f"{sha}^1", f"{sha}^2").split() == [main, head]
    keys = env.listed(sha, {"pytest:c": ["moved/c.txt"]})
    env.result(keys["pytest:c"], "pytest:c", "pass")
    assert _acts(loop.tick()) == ["advance", "on_main"] and env.main() == sha


@pytest.mark.parametrize("shadow", [False, True])
def test_a_head_below_a_move_main_landed_is_taken_unless_its_change_names_what_the_move_retires_and_that_one_is_told_once(
        env, shadow):
    """A merge of a head based below a layout move leaves its own lines as they are, however clean: one whose own change names
    none of the names the move's map retires crosses by that merge and is built, and one that names one, in a changed path or
    an added line, is left out, its owner told the move and the fix once a head and move."""
    base, move = _landed_move(env.o, '[paths]\n"c.txt" = "moved/c.txt"\n', {"c.txt": None, "moved/c.txt": "c\n"})
    env.listed(move, {})
    below = {n: env.o.change(files, base=base) for n, files in ((1, {"b.txt": "below\n"}), (2, {"e.txt": "reads c.txt\n"}),
                                                                (3, {"c.txt": "edited\n"}))}
    for n, head in below.items():
        env.o.gh.put(n, head)
    crossed = env.put(4, {"d.txt": "crossed\n"})
    loop = env.loop(policy, shadow=shadow)
    *told, built = loop.tick()
    assert (built["action"], built["members"], built["heads"]) == ("build", [1, 4], [below[1], crossed]), built
    assert [(r["action"], r["number"], r["heads"], r["base"], r["done"]) for r in told] == [
        ("lacks", n, [below[n]], move, not shadow) for n in (2, 3)], told
    assert all(r["why"].startswith(f"it lacks the move {move[:12]} `main` landed (tools/move/x_map.toml), whose retired "
                                   "names its own change names (`c.txt`): merge `main` into it") for r in told), told
    assert {n: why for n, (_, why) in loop.world.left.items()} == {r["number"]: r["why"] for r in told}, loop.world.left
    assert env.posts == ([] if shadow else [f"{' '.join(r['to'])} vqueue: {r['text']}" for r in told]), env.posts
    assert loop.tick() == []
    env.o.gh.put(2, again := env.o.change({"e.txt": "reads c.txt again\n"}, base=base))
    (told2,) = loop.tick()
    assert (told2["action"], told2["number"], told2["heads"]) == ("lacks", 2, [again]), told2
    assert len(env.posts) == (0 if shadow else 3)


@pytest.mark.parametrize("lowrisk", [False, True])
def test_a_move_pr_without_a_land_verdict_is_not_built_even_where_the_low_risk_rule_builds_another_on_its_mark(env, lowrisk):
    if lowrisk:
        _main_to(env, {Q.DRIVE_FLAGS: 'lowrisk_excludes = ["*.lean"]\n'})
    one = env.put(1, {"a.txt": "1\n"})
    two = _move(env, 2, {"c.txt": None, "moved/c.txt": "c\n"})
    env.labels.reviews = lambda n, head: []                     # both marked ready, neither reviewed
    recs = (loop := env.loop(policy)).tick()
    assert [r["members"] for r in recs if r["action"] == "build"] == ([[1]] if lowrisk else []), recs
    assert loop.world.left == {**({} if lowrisk else {1: (one, "no land verdict, and it isn't low-risk")}),
                               2: (two, "no land verdict, and a move lands on its mark alone only once its map "
                                       "tools/move/x_map.toml is on `main`")}, loop.world.left


@pytest.mark.parametrize("shadow", [False, True])
def test_a_ready_pr_that_doesnt_merge_onto_main_is_told_its_files_once_a_head_keeping_its_mark_and_with_shadow_only_logged(
        env, shadow):
    """A ready PR whose head doesn't merge onto `main` (`Ready.conflict`), a move's as any other's while its move isn't
    regenerated, is in no build: its owner is told the files and to rebase it, once a head and again at a new head that
    doesn't merge either."""
    base = env.main()
    _main_to(env, {"a.txt": "main\n"})
    one = env.o.change({"a.txt": "one\n"}, base=base)
    env.o.gh.put(1, one)
    env.put(2, {"b.txt": "2\n"})
    three = _move(env, 3, {"c.txt": None, "moved/c.txt": "c\n", "a.txt": "moved/c\n"}, base=base)
    loop = env.loop(policy, shadow=shadow)
    *told, built = loop.tick()
    assert [(r["action"], r["number"], r["heads"], r["files"], r["done"], r["shadow"]) for r in told] == [
        ("conflict", 1, [one], ["a.txt"], not shadow, shadow), ("conflict", 3, [three], ["a.txt"], not shadow, shadow)], told
    assert (built["action"], built["members"]) == ("build", [2]), built
    assert all(x in told[0]["text"] for x in ("conflicts in a.txt", "rebase it onto `main`", "tools/move/restack.py")), told[0]
    assert env.posts == ([] if shadow else [f"{' '.join(r['to'])} vqueue: {r['text']}" for r in told])
    assert loop.tick() == [] and env.withdrawn == []
    env.o.gh.put(1, again := env.o.change({"a.txt": "again\n"}, base=base))
    (told2,) = loop.tick()
    assert (told2["action"], told2["number"], told2["heads"], told2["done"]) == ("conflict", 1, [again], not shadow), told2
    assert len(env.posts) == (0 if shadow else 3) and env.withdrawn == []


def test_a_loop_started_again_on_the_same_log_and_origin_repeats_no_build_and_no_advance(env):
    env.put(1, {"a.txt": "1\n"})
    first = env.loop(policy, clone="a")
    (built,) = first.tick()
    second = env.loop(policy, clone="b")                       # another clone: what a tick reads is on origin and in the log
    assert second.tick() == []
    keys = env.listed(built["sha"], {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    assert _acts(first.tick()) == ["advance", "on_main"]
    assert _acts(second.tick()) == ["drop"]
    assert first.tick() == [] and second.tick() == []
    assert _acts(env.lines()) == ["build", "advance", "on_main", "drop"] and [r["tick"] for r in env.lines()] == [1, 2, 2, 3]
    assert len(env.graph2.enqueued()) == 1 and env.main() == built["sha"]


def test_shadow_logs_advance_eject_ring_and_messages_and_moves_no_main_label_or_message(env):
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy, shadow=True)
    (built,) = loop.tick()
    keys = env.listed(built["sha"], {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    (advance,) = loop.tick()
    assert (advance["action"], advance["done"]) == ("advance", False) and env.main() == main
    (drop,) = loop.tick()                                       # its `main` is the candidate it advanced
    assert drop["action"] == "drop" and drop["main"] == built["sha"] and drop["origin_main"] == main
    assert env.queued() == {}
    three = env.put(3, {"c.txt": "3\n"})
    (on,) = loop.tick()
    assert on["action"] == "build" and on["members"] == [3] and on["base"] == built["sha"]
    keys = env.listed(on["sha"], {"pytest:c": ["c.txt"]})
    env.result(keys["pytest:c"], "pytest:c", "fail")
    eject, drop = loop.tick()
    assert (eject["action"], eject["number"], eject["done"]) == ("eject", 3, False) and "pytest:c" in eject["text"]
    assert loop.tick() == []                                    # #3 stays out until marked again
    assert env.main() == main and env.withdrawn == [] and env.posts == [] and (3, three) in env.o.gh.ready
    assert all(r["shadow"] for r in env.lines()) and _acts(env.lines()) == ["build", "advance", "drop", "build", "eject", "drop"]


@pytest.mark.parametrize("how", ["held", "withdrawn", "withdrawn by its owner", "unmarked head", "conflicting head", "closed"])
def test_a_candidate_in_flight_is_dropped_for_a_member_others_took_and_else_lands_as_built(env, how):
    """A reviewer's hold or a mark another withdrew (as a red's eject does) drops the candidate, the drop naming it and by
    whom; a mark its owner withdrew, a new head or a closed PR leaves the candidate to land as built."""
    base = env.main()
    _main_to(env, {"c.txt": "main\n"})
    one = env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    by: dict[tuple[int, str], str] = {}
    env.labels.marked_by = lambda n, h: by.get((n, h))         # the author of a `ready` label `ready` read that isn't true
    loop = env.loop(vqueue.decide)
    (built,) = loop.tick()
    assert built["members"] == [1, 2], built
    if how == "held":
        env.labels.held.add(1)
    elif how.startswith("withdrawn"):
        env.o.gh.ready.discard((1, one))
        by[(1, one)] = "@Lane-1" if how.endswith("owner") else "ci"
    elif how == "unmarked head":
        env.o.gh.put(1, env.o.change({"a.txt": "1b\n"}), ready=False)
    elif how == "conflicting head":
        env.o.gh.put(1, env.o.change({"c.txt": "1\n"}, base=base))
    else:
        del env.o.gh.prs[1]
    recs = loop.tick()
    if how in ("held", "withdrawn"):
        want = {"held": "#1 isn't taken now: held by reviewer (no reason)",
                "withdrawn": "#1 isn't taken now: its ready mark was withdrawn by ci"}[how]
        assert [(r["sha"], r["why"]) for r in recs if r["action"] == "drop"] == [(built["sha"], want)], recs
        assert [r["members"] for r in recs if r["action"] == "build"] == [[2]], recs
        return
    assert not [r for r in recs if r["action"] in ("drop", "build")], recs
    keys = env.listed(built["sha"], {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    assert "advance" in _acts(loop.tick()) and env.main() == built["sha"]


@pytest.mark.parametrize("took", ["held", "held by its owner", "withdrawn by ci", "withdrawn by another", "frozen"])
@pytest.mark.parametrize("how", ["withdrawn by its owner", "unmarked head", "closed"])
def test_a_member_s_head_as_built_is_read_for_what_others_did_there_after_its_owner_moved_on(env, how, took):
    """Once its owner withdrew a member's mark, pushed a new head or closed it, the head the candidate holds is still the one
    that lands: a reviewer's hold there, a mark withdrawn there by ci or by another, or the freeze drops the candidate,
    naming that head; its owner's own hold there doesn't, and it lands as built."""
    _main_to(env, {"gone.txt": "x\n"})
    one = env.put(1, {"a.txt": "1\n", "gone.txt": None})
    env.put(2, {"b.txt": "2\n"})
    by: dict[tuple[int, str], str] = {}
    env.labels.marked_by = lambda n, h: by.get((n, h))
    loop = env.loop(vqueue.decide)
    (built,) = loop.tick()
    assert built["members"] == [1, 2], built
    if how == "withdrawn by its owner":
        env.o.gh.ready.discard((1, one))
        by[(1, one)] = "@Lane-1"
    elif how == "unmarked head":
        env.o.gh.put(1, env.o.change({"a.txt": "1b\n"}), ready=False)
    else:
        del env.o.gh.prs[1]
    assert not [r for r in loop.tick() if r["action"] in ("drop", "build")]
    if took == "held":
        env.labels.held.add(1)
    elif took == "held by its owner":
        own = Lab("@Lane-1", "hold", "2026-01-01T00:00:01Z")
        env.labels.reviews = lambda n, h, reviews=env.labels.reviews: [*reviews(n, h), *([own] if n == 1 else [])]
    elif took.startswith("withdrawn"):
        env.o.gh.ready.discard((1, one))
        by[(1, one)] = who = "ci" if took.endswith("ci") else "lane-2"
    else:
        (frozen := env.tmp / "frozen").write_text("gone.txt\n")
        loop.world.frozen_file = frozen
    recs = loop.tick()
    at = f"at {one[:12]}, the head it was built at"
    want = {"held": f"held by reviewer (no reason), {at}",
            "frozen": f"it waits for the freeze to end ({env.tmp / 'frozen'}) {at}"}.get(took)
    if took.startswith("withdrawn"):                    # an owner's withdrawal leaves its head the one built, read as ever
        want = (f"its ready mark was withdrawn by {who}" if how == "withdrawn by its owner"
                else f"its ready mark {at}, was withdrawn by {who}")
    if want is not None:
        drops = [(r["sha"], r["why"]) for r in recs if r["action"] == "drop"]
        assert drops == [(built["sha"], f"#1 isn't taken now: {want}")], recs
        assert [r["members"] for r in recs if r["action"] == "build"] == [[2]], recs
        return
    assert not [r for r in recs if r["action"] in ("drop", "build")], recs
    keys = env.listed(built["sha"], {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    assert "advance" in _acts(loop.tick()) and env.main() == built["sha"]


@pytest.mark.parametrize("owner", [L.EJECTOR, "lane-1"])
@pytest.mark.parametrize("how", ["unmarked head", "closed"])
def test_at_the_head_as_built_an_eject_drops_the_candidate_when_ci_s_and_its_owner_s_withdrawal_doesnt(env, how, owner):
    """At the head a candidate holds, once its PR got a new head or closed, its owner's withdrawn mark lands it as built, ci's
    on a PR ci owns too, as `_pulled` reads a PR at its head; an eject recorded there drops it when it's ci's, a red's, even
    of ci's own PR, and lands it when it's its owner's."""
    env.o.gh.put(1, one := env.o.change({"a.txt": "1\n"}), body=f"Lane `{owner}`.")
    env.put(2, {"b.txt": "2\n"})
    by = {(1, one): owner}
    env.labels.marked_by = lambda n, h: by.get((n, h))
    env.labels.ready_by = lambda n, h: owner if n == 1 else f"lane-{n}"
    loop = env.loop(vqueue.decide)
    (built,) = loop.tick()
    assert built["members"] == [1, 2], built
    env.o.gh.ready.discard((1, one))
    if how == "unmarked head":
        env.o.gh.put(1, env.o.change({"a.txt": "1b\n"}), ready=False, body=f"Lane `{owner}`.")
    else:
        del env.o.gh.prs[1]
    assert not [r for r in loop.tick() if r["action"] in ("drop", "build")]
    state = loop.world.q.load_state()
    state["ejected"]["1"] = {"head": one, "why": "pytest:a failed", "by": owner, "at": "2026-01-01T00:00:09Z"}
    loop.world.q.save_state(state)
    recs = loop.tick()
    if owner == L.EJECTOR:
        drops = [(r["sha"], r["why"]) for r in recs if r["action"] == "drop"]
        assert drops == [(built["sha"], f"#1 isn't taken now: ejected by {owner} at {one[:12]}, the head it was built at: "
                                        "pytest:a failed")], recs
        assert [r["members"] for r in recs if r["action"] == "build"] == [[2]], recs
        return
    assert not [r for r in recs if r["action"] in ("drop", "build")], recs
    keys = env.listed(built["sha"], {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    assert "advance" in _acts(loop.tick()) and env.main() == built["sha"]


def test_a_candidate_holding_a_pr_the_shadow_ejected_is_dropped_saying_so(env):
    """A shadow's eject leaves the mark, so the PR stays out until marked again: a candidate still holding it (its drop cut
    off) is dropped for that eject, not for a withdrawn mark."""
    three = env.put(3, {"c.txt": "3\n"})
    loop = env.loop(vqueue.decide, shadow=True)
    (built,) = loop.tick()
    with env.log.open("a") as f:
        f.write(json.dumps({**built, "action": "eject", "sha": None, "number": 3, "heads": [three]}) + "\n")
    (drop,) = loop.tick()
    assert (drop["action"], drop["sha"], drop["why"]) == (
        "drop", built["sha"], "#3 isn't taken now: the shadow ejected it at this head, and it isn't marked ready since"), drop


def test_a_dry_run_prints_the_snapshot_and_the_actions_and_writes_nothing(env):
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    assert env.loop(policy, dry_run=True).tick() == []
    said = "\n".join(env.out)
    assert f"main {main[:12]}" in said and "ready 2" in said and "would build: Build(members=(1, 2)" in said, said
    assert not env.log.exists() and env.queued() == {} and env.graph2.enqueued() == [] and env.main() == main
    assert env.posts == [] and env.withdrawn == []


def test_a_task_blocked_behind_a_failed_one_fails_the_candidate_with_its_inputs_and_a_listing_failure_is_list_failed(env):
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(lambda s: [] if s.candidates else [Build((1,), s.main, 5)])
    (built,) = loop.tick()
    sha = built["sha"]
    w = loop.world
    assert w.verdict(sha).verdict.state == "pending"
    keys = env.listed(sha, {"lean-build:x": ["x.lean"], "pytest:a": ["a.txt"]}, reads={"pytest:a": "lean-build:x"})
    env.result(keys["lean-build:x"], "lean-build:x", "fail")
    v = w.verdict(sha).verdict
    assert v.state == FAIL and [(f.task, f.key) for f in v.failed] == [("lean-build:x", keys["lean-build:x"])]
    env.graph2.list_graph(f"ready:{sha}", {"schema": GRAPH_V2, "commit": "0" * 40, "tasks": []})
    assert w.verdict(sha).verdict.state == LIST_FAILED


def test_the_policy_itself_ejects_the_member_a_failure_meets_rebuilds_the_rest_and_lands_them(env):
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    two = env.put(2, {"b.txt": "2\n"})
    loop = env.loop(vqueue.decide)
    (built,) = loop.tick()
    assert (built["action"], built["members"], built["base"], built["priority"]) == ("build", [1, 2], main, L.DEPTH), built
    assert loop.tick() == []                                    # not listed yet: pending
    keys = env.listed(built["sha"], {"pytest:a": ["a.txt"], "pytest:b": ["b.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    env.result(keys["pytest:b"], "pytest:b", "fail")
    eject, drop, again = loop.tick()
    assert (eject["action"], eject["number"], eject["heads"], env.withdrawn) == ("eject", 2, [two], [(2, two)])
    assert (drop["action"], drop["outcome"]["state"], drop["outcome"]["suspects"]) == ("drop", FAIL, [2])
    assert (again["action"], again["members"], again["base"]) == ("build", [1], main)
    keys = env.listed(again["sha"], {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    advance, on_main, landed = loop.tick()
    assert (advance["action"], on_main["action"], on_main["done"]) == ("advance", "on_main", True)
    assert (landed["action"], landed["outcome"]["state"]) == ("drop", PASS)
    assert env.main() == again["sha"] and env.queued() == {} and loop.tick() == []


@pytest.mark.parametrize("shadow", [False, True])
def test_a_failure_at_its_bases_key_reds_the_base_once_and_the_candidate_lands_past_it_once_the_rest_passed(env, shadow):
    main = env.main()
    one = env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide, shadow=shadow)
    (built,) = loop.tick()
    red, failed = {"pytest:r": "sha512:red"}, [{"task": "pytest:r", "key": "sha512:red"}]
    env.listed(main, {"pytest:r": ["red.txt"]}, keys=red)               # the base's graph: `pytest:r` at the same key
    keys = env.listed(built["sha"], {"pytest:a": ["a.txt"], "pytest:r": ["red.txt"]}, keys=red)
    env.result("sha512:red", "pytest:r", "fail")
    assert loop.tick() == []                                            # `pytest:a` still runs: the fail hasn't ended
    env.result(keys["pytest:a"], "pytest:a", "pass")
    moved = [] if shadow else ["on_main"]                               # drive's steps after a real advance only
    told, advance, *on_main, landed = loop.tick()                       # nobody blamed: `main`'s reds only shrink
    assert _acts(on_main) == moved
    assert (told["action"], told["base"], told["failed"], told["to"], told["shadow"]) == ("red", main, failed, ["@ci"], shadow)
    assert f"base {main[:12]} is red" in told["text"] and env.posts == ([] if shadow else [f"@ci vqueue: {told['text']}"])
    assert (advance["action"], landed["action"], landed["why"]) == ("advance", "drop", "on main")
    assert landed["outcome"] == {"members": [1], "heads": [one], "state": PASS, "suspects": [], "base": main, "inherited": [
        {"task": "pytest:r", "key": "sha512:red", "inputs": sorted(["red.txt", *SPEC]), "base_key": "sha512:red"}]}
    env.put(2, {"b.txt": "2\n"})
    (two,) = loop.tick()
    assert (two["action"], two["members"], two["base"]) == ("build", [2], built["sha"])
    keys = env.listed(two["sha"], {"pytest:b": ["b.txt"], "pytest:r": ["red.txt"]}, keys=red)
    env.result(keys["pytest:b"], "pytest:b", "pass")
    assert _acts(loop.tick()) == ["advance", *moved, "drop"]            # the same red, told once
    fix = env.put(3, {"red.txt": "fixed\n"})
    (three,) = loop.tick()
    assert (three["members"], three["heads"], three["base"]) == ([3], [fix], two["sha"])
    keys = env.listed(three["sha"], {"pytest:r": ["red.txt"]})
    env.result(keys["pytest:r"], "pytest:r", "pass")
    advance, *on_main, landed = loop.tick()
    assert _acts(on_main) == moved
    assert (advance["action"], landed["outcome"]) == ("advance", {"members": [3], "heads": [fix], "state": PASS,
                                                                  "suspects": []})
    assert env.main() == (main if shadow else three["sha"]) and env.withdrawn == [] and len(env.posts) == (0 if shadow else 1)


def test_a_shadow_under_the_policy_itself_builds_on_the_main_it_advanced_to_until_origin_s_moves_elsewhere(env):
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide, shadow=True)
    (built,) = loop.tick()
    keys = env.listed(built["sha"], {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    advance, drop = loop.tick()
    assert (advance["action"], advance["shadow"], advance["done"], drop["action"]) == ("advance", True, False, "drop")
    assert env.main() == main and env.queued() == {}
    env.put(2, {"b.txt": "2\n"})
    (on,) = loop.tick()
    assert (on["action"], on["members"], on["base"], on["main"], on["origin_main"]) == ("build", [2], built["sha"],
                                                                                         built["sha"], main), on
    moved = env.o.change({"z.txt": "z\n"})                          # drive lands something else on origin's `main`
    _git(env.o.origin, "update-ref", "refs/heads/main", moved)
    dropped, again = loop.tick()
    assert (dropped["action"], dropped["sha"], dropped["main"]) == ("drop", on["sha"], moved) and "not on main" in dropped["why"]
    assert (again["action"], again["members"], again["base"]) == ("build", [1, 2], moved)
    assert env.main() == moved and env.posts == [] and env.withdrawn == []


def test_an_action_that_fails_stops_the_tick_with_its_error_logged_and_a_torn_line_stays_its_own(env):
    one = env.put(1, {"a.txt": "1\n"})
    env.log.parent.mkdir(parents=True)
    env.log.write_text('{"at": "2030-03-17T17:46:40Z", "tick": 4, "act')        # a crash mid-write
    loop = env.loop(lambda s: [Eject(1, one, "its change fails the check"), Build((1,), s.main, 0)])
    loop.world.withdraw = lambda n, h, why: World.withdraw(loop.world, n, h, why)   # the real one, under the clone's lock
    held = Q._lock(loop.world.repo)                                 # a drive tick in the clone
    try:
        (rec,) = loop.tick()
    finally:
        held.close()
    assert (rec["action"], rec["tick"], rec["number"]) == ("eject", 1, 1) and "holds" in rec["error"], rec
    assert env.queued() == {} and env.posts == []                  # the build after it waits for the next tick
    assert [json.loads(x) for x in env.log.read_text().splitlines()[1:]] == [rec]
    assert any("read as none" in x for x in env.out)


@pytest.mark.parametrize("slim", [False, True])
def test_a_build_whose_enqueue_failed_is_logged_stops_the_tick_and_is_enqueued_by_the_next(env, slim):
    """With the run beside each check too, which the build's tick, stopped, didn't start: the next starts it, and the check
    stays list-failed beside it, not pending."""
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    loop = env.loop(lambda s: policy(s) + ([] if s.candidates else [Build((2,), s.main, 0)]), slim=Slim() if slim else None)
    loop.world.slim_driver = lambda: "drive"
    real, calls = loop.world.enqueue, []

    def enqueue(sha: str, priority: int) -> None:
        calls.append(sha)
        if len(calls) == 1:
            raise Q.QueueError("graph2 is down")
        real(sha, priority)

    loop.world.enqueue = enqueue
    *below, built = loop.tick()                                 # the second build waits
    assert [r["why"] for r in below] == (["started beside `main`"] if slim else [])
    assert (built["action"], built["enqueued"], built["error"]) == ("build", False, "graph2 is down"), built
    assert env.queued() == {branch(built["sha"]): built["sha"]} and env.graph2.enqueued() == []
    *started, retry = loop.tick()
    assert [(r["action"], r["sha"], r["why"]) for r in started] == ([("slim", built["sha"], "started")] if slim else [])
    assert (retry["action"], retry["enqueued"], retry["mark"], retry["priority"]) == ("retry", True, "not enqueued", 5)
    assert [e["check"] for e in env.graph2.enqueued()] == [f"ready:{built['sha']}"] and loop.tick() == []


#: `research data export` as a script: each call's arguments appended to `<script>.calls`.  It sends every id it is named
#: but those `<script>.fail` lists, each printed as one no source holds, and those `<script>.upload` lists, each an upload's
#: error naming no id, as the export prints them; it exits 1 if any failed.  Whether another held `<script>.lock` while it ran
#: is appended to `<script>.held`
EXPORT = '''
import fcntl, json, os, sys
with open(sys.argv[0] + ".calls", "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
with open(sys.argv[0] + ".lock", "a") as lock, open(sys.argv[0] + ".held", "a") as f:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        f.write("0\\n")
    except BlockingIOError:
        f.write("1\\n")
def listed(kind):
    p = sys.argv[0] + "." + kind
    return open(p).read().split() if os.path.exists(p) else []
ids = [a[5:] for a in sys.argv[1:] if a.startswith("--id=")]
bad = [("ERROR  not local and not on any source: " + x) for x in ids if x in listed("fail")]
bad += [("ERROR  objects/sha256/" + x[4:] + ": error: AccessDenied") for x in ids if x in listed("upload")]
print("\\n".join(bad))
sys.exit(1 if bad else 0)
'''
#: fetched fixtures (their paths in no tree) and a tracked one (`small.json`, committed beside the registry)
BASE_FX, NEW_FX, TRACKED_FX, OTHER_FX = (f"art:{c * 64}" for c in "bcde")


def _registry(entries: dict[str, str], **files: str) -> dict[str, str]:
    return {"fixtures/artifacts.json": json.dumps(entries, indent=1) + "\n", **files}


def _exports(env: Env, loop: Loop) -> Path:
    """``loop``'s export as `EXPORT`, holding `<script>.lock` with no wait, returns the script."""
    (script := env.tmp / "export.py").write_text(EXPORT)
    loop.world.export, loop.world.export_lock, loop.world.export_lock_s = (sys.executable, str(script)), f"{script}.lock", 0
    return script


def _calls(script: Path) -> list[list[str]]:
    p = Path(f"{script}.calls")
    return [json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else []


def test_a_build_exports_each_fetched_fixture_its_merge_registers_that_its_base_doesn_t_once_before_it_is_pushed(env):
    """#1 registers `NEW_FX`, fetched, and `TRACKED_FX`, tracked, beside `main`'s `BASE_FX`: the build exports `NEW_FX` alone, with
    `--no-registry`, before it pushes the candidate or enqueues its check, and logs it.  Built again from the same log, it
    doesn't export it again."""
    _main_to(env, _registry({BASE_FX: "data/old.bin"}))
    one = env.put(1, _registry({BASE_FX: "data/old.bin", NEW_FX: "data/big.bin", TRACKED_FX: "small.json"}, **{"small.json": "{}\n"}))
    env.put(2, {"b.txt": "2\n"})
    loop = env.loop(lambda s: [Drop(c.sha, "rebuilt") for c in s.candidates] or [Build((1, 2), s.main, 5)])
    script, pushed = _exports(env, loop), []
    real = loop.world.push
    loop.world.push = lambda sha, ref, lease=None: pushed.append(len(_calls(script))) or real(sha, ref, lease)
    exported, built = loop.tick()
    assert _calls(script) == [["--no-registry", f"--id={NEW_FX}"]] and pushed == [1]
    assert (exported["action"], exported["ids"], exported["members"], exported["heads"], exported["done"]) \
        == ("fixtures", [NEW_FX], [1], [one], True), exported
    assert (built["action"], built["members"], built["enqueued"]) == ("build", [1, 2], True)
    assert env.graph2.enqueued()[0]["check"] == f"ready:{built['sha']}"
    assert _acts(loop.tick()) == ["drop"]
    assert _acts(loop.tick()) == ["build"] and len(_calls(script)) == 1 and env.posts == []


def test_a_build_registering_no_new_fetched_fixture_exports_nothing(env):
    """A member that changes no registry, and one that registers only a tracked fixture: no export, no `fixtures` line."""
    _main_to(env, _registry({BASE_FX: "data/old.bin"}))
    env.put(1, {"a.txt": "1\n"})
    env.put(2, _registry({BASE_FX: "data/old.bin", TRACKED_FX: "small.json"}, **{"small.json": "{}\n"}))
    loop = env.loop(policy)
    script = _exports(env, loop)
    (built,) = loop.tick()
    assert (built["action"], built["members"], built["enqueued"]) == ("build", [1, 2], True)
    assert _calls(script) == [] and "fixtures" not in _acts(env.lines())


def test_a_member_whose_fixture_s_export_failed_is_left_out_its_owner_told_once_and_exported_again_each_tick(env):
    """The export of #1's `NEW_FX` fails: the build pushes and enqueues nothing with it, logs why and tells #1's owner once,
    naming `NEW_FX`; the rest (#2) is built and checked without it.  Each later build of #1 exports again, telling nobody, and
    once the export passes #1 is built."""
    _main_to(env, _registry({BASE_FX: "data/old.bin"}))
    one = env.put(1, _registry({BASE_FX: "data/old.bin", NEW_FX: "data/big.bin"}))
    env.put(2, {"b.txt": "2\n"})
    def stack(s: Snapshot) -> list:
        taken = {m for c in s.candidates for m in c.members}
        rest = tuple(x.number for x in s.ready if x.number not in taken)
        return [Build(rest, s.candidates[-1].sha if s.candidates else s.main, 5)] if rest else []

    loop = env.loop(stack)
    script = _exports(env, loop)
    Path(f"{script}.fail").write_text(NEW_FX)
    exported, skip, built = loop.tick()
    assert (exported["action"], exported["ids"], exported["members"], exported["done"]) == ("fixtures", [NEW_FX], [1], False)
    assert exported["unsent"] == {NEW_FX: "`research data export` exited 1; the store holds none of it"}, exported
    assert (skip["action"], skip["number"], skip["heads"], skip["fixtures"], skip["done"]) == ("skip", 1, [one], [NEW_FX], True)
    (post,) = env.posts
    assert f"#1 at {one[:12]} is left out" in post and NEW_FX in post and "preflight-fixtures" in post, post
    assert (built["action"], built["members"], built["enqueued"]) == ("build", [2], True), built
    assert [e["check"] for e in env.graph2.enqueued()] == [f"ready:{built['sha']}"] and list(env.queued().values()) == [built["sha"]]
    assert NEW_FX not in loop.world._registry(built["sha"])
    again, told, none = loop.tick()                             # on #2's candidate: #1 alone, its export failing again
    assert (again["action"], again["done"], told["action"], told["text"], told["done"]) == ("fixtures", False, "skip", None, False)
    assert (none["action"], none["sha"], none["why"]) == ("build", None, "no member merges onto its base with its fixtures exported")
    assert len(env.posts) == 1 and len(env.graph2.enqueued()) == 1 and len(_calls(script)) == 2
    Path(f"{script}.fail").unlink()
    exported, last = loop.tick()
    assert (exported["done"], last["members"], last["base"], last["enqueued"]) == (True, [1], built["sha"], True), last
    assert len(_calls(script)) == 3 and len(env.posts) == 1 and len(env.graph2.enqueued()) == 2


@pytest.mark.parametrize("how", ["fail", "upload"])
def test_of_two_members_exported_at_once_only_the_one_whose_own_fixture_failed_is_left_out_and_its_owner_alone_told(env, how):
    """#1 registers `NEW_FX`, whose export fails, and #2 `OTHER_FX`, whose export passes, in one call: #2 is built and
    enqueued, #1 left out, and only #1's owner told, naming `NEW_FX` alone.  An export that names the id no source holds
    (``fail``) is read in that one call; one whose error names no id (``upload``) is done again an id at a time."""
    filler = {f"art:{c * 64}": f"data/{c}.bin" for c in "0123"}
    _main_to(env, _registry({BASE_FX: "data/old.bin", **filler}))
    one = env.put(1, _registry({BASE_FX: "data/old.bin", NEW_FX: "data/new.bin", **filler}))
    two = env.put(2, _registry({BASE_FX: "data/old.bin", **filler, OTHER_FX: "data/other.bin"}))
    loop = env.loop(policy)
    loop.world.handle = lambda n: f"lane-{n}"
    script = _exports(env, loop)
    Path(f"{script}.{how}").write_text(NEW_FX)
    exported, skip, built = loop.tick()
    assert _calls(script) == [["--no-registry", f"--id={NEW_FX}", f"--id={OTHER_FX}"]] + (
        [["--no-registry", f"--id={NEW_FX}"], ["--no-registry", f"--id={OTHER_FX}"]] if how == "upload" else [])
    assert (exported["ids"], exported["exported"], list(exported["unsent"]), exported["members"]) \
        == ([NEW_FX, OTHER_FX], [OTHER_FX], [NEW_FX], [1, 2]), exported
    assert (skip["action"], skip["number"], skip["heads"], skip["fixtures"]) == ("skip", 1, [one], [NEW_FX]), skip
    (post,) = env.posts
    assert post.startswith("@lane-1 ") and NEW_FX in post and OTHER_FX not in post, post
    assert (built["action"], built["members"], built["heads"], built["enqueued"]) == ("build", [2], [two], True), built
    assert OTHER_FX in loop.world._registry(built["sha"]) and [e["check"] for e in env.graph2.enqueued()] == [f"ready:{built['sha']}"]


def test_an_export_runs_holding_the_export_lock_and_a_busy_lock_defers_it_to_the_next_tick_telling_no_one(env):
    """Each export runs while the loop holds `export_lock`, the lock the pod's own export pass holds over the same checkout
    and store.  While another holds it nothing is exported and nothing fails: the member is left out of this build alone,
    with one `fixtures` line saying `deferred`, no skip line and no owner told; once it is free the next tick exports the
    fixture and builds the member."""
    _main_to(env, _registry({BASE_FX: "data/old.bin"}))
    env.put(1, _registry({BASE_FX: "data/old.bin", NEW_FX: "data/new.bin"}))
    loop = env.loop(policy)
    script = _exports(env, loop)
    with open(loop.world.export_lock, "a") as other:
        fcntl.flock(other, fcntl.LOCK_EX)
        deferred, none = loop.tick()
    assert _calls(script) == [] and env.posts == [] and "skip" not in _acts(env.lines())
    assert (deferred["action"], deferred["ids"], deferred["members"], deferred["deferred"], deferred["done"]) \
        == ("fixtures", [NEW_FX], [1], True, False), deferred
    assert (none["action"], none["sha"], none["why"]) == ("build", None, "no member merges onto its base with its fixtures exported")
    exported, built = loop.tick()
    assert _calls(script) == [["--no-registry", f"--id={NEW_FX}"]] and Path(f"{script}.held").read_text() == "1\n"
    assert (exported["exported"], built["members"], built["enqueued"]) == ([NEW_FX], [1], True), built
    assert env.posts == []


def test_a_busy_export_lock_is_waited_on_only_its_short_bound(env):
    """A tick waits on another export's lock at most `export_lock_s`, a few seconds, then runs no export (None): that
    export's pass holds it for its whole run, and every advance and drop of the tick waits behind the build."""
    loop = env.loop(policy)
    script = _exports(env, loop)
    slept: list[float] = []
    loop.world.sleep, loop.world.export_lock_s = slept.append, 3
    with open(loop.world.export_lock, "a") as other:
        fcntl.flock(other, fcntl.LOCK_EX)
        assert loop.world.export_fixtures([NEW_FX]) is None
    assert _calls(script) == [] and sum(slept) == 3 and L.EXPORT_LOCK_S <= 10
    assert loop.world.export_fixtures([NEW_FX]) == {} and _calls(script) == [["--no-registry", f"--id={NEW_FX}"]]


GRAPH2 = '''
import argparse, json, os, sys
ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
for c in ("verdict", "cancel", "rerun"):
    p = sub.add_parser(c)
    p.add_argument("--check")
    p.add_argument("--keys", nargs="*", default=[])
a = ap.parse_args()
said, own = sys.argv[0] + ".verdict", f"{sys.argv[0]}.verdict.{a.check}"
if a.cmd == "verdict":
    with open(sys.argv[0] + ".asked", "a") as f:
        f.write(a.check + "\\n")
    print(open(own).read() if os.path.exists(own) else open(said).read() if os.path.exists(said)
          else json.dumps({"state": "fail", "check": a.check}))
else:
    with open(f"{sys.argv[0]}.{a.cmd}", "a") as f:
        f.write(" ".join([a.check, *a.keys]) + "\\n")
    if a.cmd == "rerun":
        print(f"{a.check} requeued {len(a.keys)}")
'''


def test_graph2_s_own_verdict_is_kept_beside_the_loop_s_and_its_cancel_runs_on_a_drop(env):
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy, state=(state := env.tmp / "vqueue" / "state.json"))
    script = env.graph2_script(loop)
    (built,) = loop.tick()
    sha = built["sha"]
    keys = env.listed(sha, {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    assert _acts(loop.tick()) == ["advance", "on_main"]         # the loop's own verdict, pass, is the one used
    kept = json.loads(state.read_text())
    assert kept["candidates"][sha]["mine"] == PASS and kept["candidates"][sha]["graph2"] == FAIL and kept["disagree"] == [sha]
    assert any("graph2 says fail" in x for x in env.out)
    (drop,) = loop.tick()
    assert drop["cancel"] == "cancelled" and Path(f"{script}.cancel").read_text() == f"ready:{sha}\n"


def test_a_task_graph2_says_faulted_which_it_labels_no_result_for_fails_the_candidate_and_blocks_its_readers(env):
    """graph2 labels no result for a task's own fault (an undeclared read, a race): its verdict's failed entry with a `why`
    is read as the fail it is, not as pending."""
    one = env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide)
    script = env.graph2_script(loop, {"state": PENDING})
    (built,) = loop.tick()
    sha = built["sha"]
    keys = env.listed(sha, {"pytest:a": ["a.txt"], "pytest:b": ["b.txt"]}, reads={"pytest:b": "pytest:a"})
    assert loop.tick() == []
    a, why = keys["pytest:a"], f"run r1 exit 0; error: {CLEAN}; read 1 file(s) outside its declared inputs: x.txt"
    said = {"state": FAIL, "failed": [{"task": "pytest:a", "key": a, "why": why}]}
    Path(f"{script}.verdict").write_text(json.dumps(said))
    assert [f.task for f in loop.world.verdict(sha).verdict.failed] == ["pytest:a"]
    assert _said({**said, "labelled": "2026-01-01T00:00:00Z"}) == {}       # a label outlives the rerun of what it says
    eject, drop = loop.tick()
    assert (eject["action"], eject["number"], env.withdrawn, drop["failed"]) == (
        "eject", 1, [(1, one)], [{"task": "pytest:a", "key": a, "tests": ["undeclared:x.txt"]}])


def test_only_graph2_s_labels_count_its_results_and_the_graph_it_listed(tmp_path):
    """`research data label --by` is self-asserted, so a `result` label on a task's key or a `verity_graph` label on a
    candidate's commit by anyone but graph2 (`GRAPH2_BY`) counts for nothing: a pass written before graph2's own lands leaves
    the task pending."""
    from research.store.local import LocalStore

    store, key, sha = LocalStore(tmp_path / "store"), "sha512:" + "a" * 128, "c" * 40
    graph = {"schema": GRAPH_V2, "commit": sha, "envs": {"py": {"digest": "sha512:env"}}, "files": {"t.py": "sha512:t"},
             "tasks": [{"id": "test-file:t.py", "kind": "test-file", "env": "py", "argv": ["t"], "vars": {}, "inputs": ["t.py"],
                        "reads": [], "outputs": [], "key": key}]}
    (f := tmp_path / "graph.json").write_text(json.dumps(graph))
    art = store.put_artifact(GRAPH_V2, {}, file=f)
    passed = {"schema": "verity-result/v1", "key": key, "task": "test-file:t.py", "status": PASS, "outputs": {}}
    ex = Executor(L.Transport(store))
    store.add_label(f"commit:{sha}", "verity_graph", art, by="not-graph2")
    store.add_label(f"task:{'a' * 128}", "result", passed, by="not-graph2")
    assert ex.graph_v2(f"ready:{sha}") is None and ex.settle(graph)["test-file:t.py"][0] == PENDING
    store.add_label(f"commit:{sha}", "verity_graph", art, by=L.GRAPH2_BY)
    store.add_label(f"task:{'a' * 128}", "result", passed, by=L.GRAPH2_BY)
    assert ex.graph_v2(f"ready:{sha}") == (art, graph) and ex.settle(graph)["test-file:t.py"][0] == PASS


def test_a_task_graph2_s_verdict_says_failed_or_erred_reads_so_whatever_its_labels_say(env):
    """graph2's `verdict` is read from its database, which no label writes: each task it lists as failed or infra reads so
    (`_said`), a failed one with the tests it names."""
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy)
    (built,) = loop.tick()
    sha = built["sha"]
    keys = env.listed(sha, {"pytest:a": ["a.txt"], "pytest:b": ["b.txt"]})
    env.result(keys["pytest:a"], "pytest:a", PASS)
    env.result(keys["pytest:b"], "pytest:b", PASS)
    script = env.graph2_script(loop, {"state": INFRA, "infra": [{"task": "pytest:b", "key": keys["pytest:b"], "why": "lost"}]})
    v = loop.world.verdict(sha, built["base"]).verdict
    assert (v.state, [f.task for f in v.infra]) == (INFRA, ["pytest:b"])
    Path(f"{script}.verdict").write_text(json.dumps({"state": FAIL, "failed": [
        {"task": "pytest:a", "key": keys["pytest:a"], "tests": ["t.py::x"]}]}))
    v = loop.world.verdict(sha, built["base"]).verdict
    assert (v.state, [(f.task, f.tests) for f in v.failed]) == (FAIL, [("pytest:a", frozenset({"t.py::x"}))])


def test_a_candidate_whose_tasks_all_passed_waits_while_graph2_counts_a_task_pending_or_can_t_be_read(env):
    """The loop's verdict is never more passing than graph2's own (`World._held`): with every listed task's result a pass, a
    candidate passes only once graph2's verdict can be read and counts no task pending."""
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy)
    (built,) = loop.tick()
    sha = built["sha"]
    keys = env.listed(sha, {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", PASS)
    script = env.graph2_script(loop, {"state": PENDING, "pending": 1})
    j = loop.world.verdict(sha, built["base"])
    assert (j.verdict.state, j.note) == (PENDING, "held: graph2 says pending")
    Path(f"{script}.verdict").write_text(json.dumps({"state": INFRA, "pending": 2}))
    assert loop.world.verdict(sha, built["base"]).note == "graph2 says infra; held: graph2 counts 2 task(s) pending"
    Path(f"{script}.verdict").write_text("{")
    assert loop.world.verdict(sha, built["base"]).note == "held: graph2's verdict can't be read"
    assert loop.tick() == []
    Path(f"{script}.verdict").write_text(json.dumps({"state": PASS, "pending": 0}))
    assert _acts(loop.tick()) == ["advance", "on_main"] and env.main() == sha


def test_a_fail_whose_every_failure_is_inherited_hasn_t_ended_while_graph2_counts_a_task_pending(env):
    """`Verdict.ended`, which lets a candidate whose every failure is inherited pass, waits on graph2's pending count too."""
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy)
    (built,) = loop.tick()
    sha, base, z = built["sha"], built["base"], "sha512:z"
    env.listed(base, {"pytest:z": ["z.txt"]}, keys={"pytest:z": z})
    keys = env.listed(sha, {"pytest:a": ["a.txt"], "pytest:z": ["z.txt"]}, keys={"pytest:z": z})
    env.result(keys["pytest:a"], "pytest:a", PASS)
    env.result(z, "pytest:z", FAIL)
    failed = [{"task": "pytest:z", "key": z}]
    script = env.graph2_script(loop, {"state": FAIL, "pending": 1, "failed": failed})
    v = loop.world.verdict(sha, base).verdict
    assert (v.state, [f.base_key for f in v.failed], v.ended, vqueue._reading(v)) == (FAIL, [z], False, PENDING)
    Path(f"{script}.verdict").write_text(json.dumps({"state": FAIL, "pending": 0, "failed": failed}))
    v = loop.world.verdict(sha, base).verdict
    assert (v.ended, vqueue._reading(v)) == (True, PASS)


@pytest.mark.parametrize("agree", [FAIL, PASS])
def test_a_task_graph2_says_failed_that_the_loop_reads_passed_at_another_key_holds_its_candidate_and_rings_ci_once(env, agree):
    """`--by executor` is self-asserted, and graph2's verdict overrides a label only at graph2's own key (`_said`): a pass for
    `lean-build:x` with outputs graph2 never made, written before graph2's own, and a pass at the key those make for
    `pytest:a`, which reads it, pass `pytest:a` while graph2's verdict names it failed.  The candidate is held, pending, `CI`
    is rung once with the task, both keys and both readings, and it stays held until the two agree: on graph2's fail, read at
    the loop's key, it fails and its PR is ejected; on graph2's pass, it lands."""
    one = env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide)
    script = env.graph2_script(loop)
    (built,) = loop.tick()
    sha, w = built["sha"], loop.world
    keys = env.listed(sha, {"lean-build:x": ["x.lean"], "pytest:a": ["a.txt"]}, reads={"pytest:a": "lean-build:x"})
    env.result(keys["lean-build:x"], "lean-build:x", PASS)
    _, made, _ = w._settled(w.tasks.graph_v2(f"ready:{sha}")[1])
    env.result(made["pytest:a"], "pytest:a", PASS)
    theirs = Path(f"{script}.verdict.ready:{sha}")
    theirs.write_text(json.dumps({"state": FAIL, "pending": 0, "failed": [
        {"task": "pytest:a", "key": "sha512:graph2s-own", "tests": ["t.py::test_a"], "inputs": ["a.txt"], "reads": []}]}))
    held = L.Dispute("pytest:a", FAIL, "sha512:graph2s-own", made["pytest:a"])
    assert str(held) == f"graph2 says pytest:a fail at sha512:graph2s-own, and the loop reads it pass at {made['pytest:a']}"
    j = w.verdict(sha, built["base"])
    assert (j.verdict, j.disputed, j.note) == (Verdict(PENDING), (held,),
                                               f"graph2 says fail; held until graph2 and the loop agree: {held}")
    acts = [r for _ in range(3) for r in loop.tick()]
    (ring,) = acts
    assert (ring["action"], ring["sha"], ring["why"], ring["to"], ring["done"], ring["task"]) == (
        "ring", sha, str(held), ["@ci"], True, "pytest:a")
    assert (ring["graph2"], ring["mine"]) == ({"state": FAIL, "key": "sha512:graph2s-own"},
                                              {"state": PASS, "key": made["pytest:a"]})
    assert env.posts == [f"@ci vqueue: {ring['text']}"] and f"ready:{sha} (#1) while graph2's" in ring["text"]
    assert env.main() != sha and env.withdrawn == []
    if agree == FAIL:
        theirs.write_text(json.dumps({"state": FAIL, "pending": 0, "failed": [
            {"task": "pytest:a", "key": made["pytest:a"], "tests": ["t.py::test_a"]}]}))
        assert _acts(loop.tick()) == ["eject", "drop"] and env.withdrawn == [(1, one)] and env.main() != sha
    else:
        theirs.write_text(json.dumps({"state": PASS, "pending": 0}))
        assert _acts(loop.tick()) == ["advance", "on_main", "drop"] and env.main() == sha
    assert [p for p in env.posts if p.startswith("@ci vqueue: the verified queue holds")] == [env.posts[0]]


#: a policy putting circuit-check's tasks off the merge gate, as `main`'s does
OFF_GATE = 'schema = "policy/v0"\noff_gate = ["circuit-check"]\n'


def test_a_task_the_verdict_needs_holds_it_on_graph2_s_fail_or_infra_but_one_off_the_gate_no_gating_task_reads_doesn_t(env):
    """A candidate's verdict needs its gating tasks and what they read, transitively: a task off the gate (circuit-check's, by
    the candidate's policy) that a gating task reads holds it on graph2's fail as a gating task does, and graph2's infra error
    holds it as its fail does, wherever the loop reads the task passed.  A task off the gate that no gating task reads holds
    nothing: graph2's verdict fails on circuit-check's tasks, which the loop doesn't gate on.  A `ready_verdict` label, which a
    rerun leaves standing, holds nothing either."""
    env.put(1, {"a.txt": "1\n", "tools/verity/check/policy.toml": OFF_GATE})
    loop = env.loop(policy)
    script = env.graph2_script(loop)
    (built,) = loop.tick()
    sha, w = built["sha"], loop.world
    env.graph2.list_graph(f"ready:{sha}", _graph(sha, [
        {"id": "lean-build:x", "kind": "lean-build", "inputs": ["x.lean"], "key": "sha512:x"},
        {"id": "circuit-binding:u", "kind": "circuit-binding", "inputs": ["u.txt"],
         "reads": [{"task": "lean-build:x", "output": "out"}]},
        {"id": "pytest:a", "kind": "test-file", "inputs": ["a.txt"], "reads": [{"task": "circuit-binding:u", "output": "out"}]},
        {"id": "circuit-binding:z", "kind": "circuit-binding", "inputs": ["z.txt"], "key": "sha512:z"}]))
    env.result("sha512:x", "lean-build:x", PASS)
    env.result("sha512:z", "circuit-binding:z", PASS)
    for task in ("circuit-binding:u", "pytest:a"):
        _, made, _ = w._settled(w.tasks.graph_v2(f"ready:{sha}")[1])
        env.result(made[task], task, PASS)
    verdict = Path(f"{script}.verdict.ready:{sha}")

    def says(**entries) -> L.Judged:
        verdict.write_text(json.dumps({"state": FAIL if "failed" in entries else INFRA, "pending": 0, **entries}))
        return w.verdict(sha, built["base"])

    assert (j := says(failed=[{"task": "circuit-binding:z", "key": "sha512:z-graph2"}])).verdict == Verdict(PASS)
    assert j.disputed == ()
    j = says(failed=[{"task": "circuit-binding:u", "key": "sha512:u-graph2"}])
    assert (j.verdict, j.disputed) == (Verdict(PENDING),
                                       (L.Dispute("circuit-binding:u", FAIL, "sha512:u-graph2", made["circuit-binding:u"]),))
    j = says(infra=[{"task": "pytest:a", "key": "sha512:a-graph2", "why": "lost"}])
    assert (j.verdict, j.disputed) == (Verdict(PENDING), (L.Dispute("pytest:a", INFRA, "sha512:a-graph2", made["pytest:a"]),))
    labelled = {"state": FAIL, "labelled": "2026-01-01T00:00:00Z", "failed": [{"task": "pytest:a", "key": "sha512:a-graph2"}]}
    assert L._disputed(labelled, ["pytest:a"], {"pytest:a": (PASS, None)}, {"pytest:a": made["pytest:a"]}) == ()


def test_a_fail_at_its_base_s_key_graph2_s_verdict_on_the_base_doesn_t_list_holds_the_candidate_and_rings_ci_once(env):
    """A fail labelled `--by executor` at the base's key is self-asserted too: written beside a candidate's own failure of the
    task, it would make that failure inherited.  While graph2's verdict on `ready:<base>` hasn't ended, the failure waits on
    the base, which is enqueued and rung as one not yet listed is; once that verdict has ended without naming the fail, or
    names the task at another key, the candidate is held, pending, and `CI` is rung once with the task, the base, both keys
    and both readings.  Once graph2's verdict lists the fail at the base's key, the failure is inherited and it lands."""
    env.put(1, {"t.py": "1\n"})
    loop = env.loop(vqueue.decide)
    script = env.graph2_script(loop)
    (built,) = loop.tick()
    sha, main, w, at = built["sha"], built["base"], loop.world, "sha512:base-t"
    env.listed(main, {"pytest:t": ["t.py"]}, keys={"pytest:t": at})
    keys = env.listed(sha, {"pytest:a": ["a.txt"], "pytest:t": ["t.py"]})
    env.result(keys["pytest:a"], "pytest:a", PASS)
    env.result(keys["pytest:t"], "pytest:t", FAIL, tests=["test_t.py::a"])
    env.result(at, "pytest:t", FAIL, tests=["test_t.py::a"])          # the base's fail no graph2 verdict lists
    Path(f"{script}.verdict.ready:{sha}").write_text(json.dumps({"state": FAIL, "pending": 0, "failed": [
        {"task": "pytest:t", "key": keys["pytest:t"], "tests": ["test_t.py::a"]}]}))
    on_base = Path(f"{script}.verdict.ready:{main}")
    on_base.write_text(json.dumps({"state": PENDING, "why": "not requested"}))
    j = w.verdict(sha, main)
    assert (j.verdict, j.waits_on, j.disputed) == (Verdict(PENDING), (main, L.BaseWait(L.BASE_UNSAID, "pytest:t")), ())
    assert j.note == f"graph2 says fail; waiting for its base {main[:12]}: {L.BASE_UNSAID} (pytest:t)"
    (base,) = loop.tick()
    assert (base["action"], base["sha"], env.graph2.enqueued()[-1]["check"]) == ("base", main, f"ready:{main}")
    env.clock += L.BASE_RING_S
    again, ring = loop.tick()
    assert (again["action"], ring["action"], ring["why"], ring["to"]) == (
        "base", "ring", f"base {main[:12]}: {L.BASE_UNSAID}", ["@ci", "@infra"])
    assert f"holds up ready:{sha[:12]}: {L.BASE_UNSAID} (pytest:t), 30 min after it was first asked for" in ring["text"]
    on_base.write_text(json.dumps({"state": PASS, "pending": 0}))
    held = L.Dispute("pytest:t", "", "", at, main)
    assert str(held) == f"graph2's verdict on its base {main[:12]} names no fail of pytest:t, and the loop reads it fail there at {at}"
    j = w.verdict(sha, main)
    assert (j.verdict, j.waits_on, j.disputed) == (Verdict(PENDING), None, (held,))
    acts = [r for _ in range(3) for r in loop.tick()]
    (ring,) = acts
    assert (ring["action"], ring["sha"], ring["why"], ring["to"], ring["done"], ring["task"], ring["check"]) == (
        "ring", sha, str(held), ["@ci"], True, "pytest:t", f"ready:{main}")
    assert (ring["graph2"], ring["mine"]) == ({"state": None, "key": None}, {"state": FAIL, "key": at})
    assert env.posts[-1] == f"@ci vqueue: {ring['text']}" and f"ready:{sha} (#1) while graph2's" in ring["text"]
    on_base.write_text(json.dumps({"state": FAIL, "pending": 0, "failed": [
        {"task": "pytest:t", "key": "sha512:other", "tests": ["test_t.py::a"]}]}))
    assert w.verdict(sha, main).disputed == (L.Dispute("pytest:t", FAIL, "sha512:other", at, main),)
    on_base.write_text(json.dumps({"state": FAIL, "pending": 0, "failed": [               # another task's fail at that key
        {"task": "pytest:other", "key": at, "tests": ["test_t.py::a"]}]}))
    assert w.verdict(sha, main).disputed == (held,)
    assert env.main() == main and env.withdrawn == [] and not {"red", "eject", "rerun"} & set(_acts(env.lines()))
    on_base.write_text(json.dumps({"state": FAIL, "pending": 0, "failed": [
        {"task": "pytest:t", "key": at, "tests": ["test_t.py::a"]}]}))
    assert _acts(loop.tick()) == ["red", "advance", "on_main", "drop"] and env.main() == sha
    assert [p for p in env.posts if "while graph2's verdict" in p] == [f"@ci vqueue: {ring['text']}"]


@pytest.mark.parametrize(("tests", "inherits"), [(["test_t.py::a"], True), (["test_t.py::b"], False)])
def test_a_fail_graph2_s_verdict_on_the_base_lists_at_its_key_is_inherited_with_the_failing_tests_it_names(env, tests,
                                                                                                          inherits):
    """A fail at the base that graph2's verdict on `ready:<base>` lists at the key the loop read counts toward a failure's
    being inherited, with the failing tests that verdict names: a fail labelled at that key before graph2's own, naming a
    test more, is the one a label reads (`fold_results`) but widens nothing.  A failure among graph2's tests is inherited and
    its candidate lands, `main` told red; one of a test only that label names is the candidate's own, and its PR ejected."""
    one = env.put(1, {"t.py": "1\n"})
    loop = env.loop(vqueue.decide)
    script = env.graph2_script(loop)
    (built,) = loop.tick()
    sha, main, w, at = built["sha"], built["base"], loop.world, "sha512:base-t"
    env.listed(main, {"pytest:t": ["t.py"]}, keys={"pytest:t": at})
    keys = env.listed(sha, {"pytest:a": ["a.txt"], "pytest:t": ["t.py"]})
    env.result(keys["pytest:a"], "pytest:a", PASS)
    env.result(keys["pytest:t"], "pytest:t", FAIL, tests=tests)
    env.result(at, "pytest:t", FAIL, tests=["test_t.py::a", "test_t.py::b"])
    env.result(at, "pytest:t", FAIL, tests=["test_t.py::a"])          # graph2's own
    Path(f"{script}.verdict.ready:{main}").write_text(json.dumps({"state": FAIL, "pending": 0, "failed": [
        {"task": "pytest:t", "key": at, "tests": ["test_t.py::a"]}]}))
    Path(f"{script}.verdict.ready:{sha}").write_text(json.dumps({"state": FAIL, "pending": 0, "failed": [
        {"task": "pytest:t", "key": keys["pytest:t"], "tests": tests}]}))
    j = w.verdict(sha, main)
    (f,) = j.verdict.failed
    assert (f.base_key, f.base_tests, j.waits_on, j.disputed) == (at, frozenset({"test_t.py::a"}), None, ())
    assert vqueue._inherited(f) is inherits
    if inherits:
        assert _acts(loop.tick()) == ["red", "advance", "on_main", "drop"] and env.main() == sha
    else:
        assert _acts(loop.tick()) == ["eject", "drop"] and env.withdrawn == [(1, one)] and env.main() == main


def test_a_listing_other_than_the_one_graph2_runs_holds_its_candidate_and_rings_ci_once(env):
    """The listing the loop reads, the `verity_graph` label on a candidate's commit, is self-asserted too: one that leaves a
    gating task out hides its fail.  graph2's verdict names the listing it runs (`art`, the id of its contents): while the
    loop's isn't it, the candidate is held, pending, and `CI` is rung once with both, their sizes and the tasks graph2 names
    that the loop's lacks; while graph2 has labelled none, a pass waits; once they are one, it lands."""
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide)
    script = env.graph2_script(loop)
    (built,) = loop.tick()
    sha, w = built["sha"], loop.world
    keys = env.listed(sha, {"pytest:a": ["a.txt"]})                     # graph2 runs `pytest:z` too, and failed it
    env.result(keys["pytest:a"], "pytest:a", PASS)
    ours = w.tasks.graph_v2(f"ready:{sha}")[0]
    verdict = Path(f"{script}.verdict.ready:{sha}")
    verdict.write_text(json.dumps({"state": FAIL, "pending": 0, "tasks": 2, "off_gate": 0, "art": "art:graph2s", "failed": [
        {"task": "pytest:z", "key": "sha512:z", "tests": ["test_z.py::z"], "inputs": ["z.txt"], "reads": []}]}))
    held = L.Relisted("art:graph2s", ours, 2, 1, ("pytest:z",))
    assert str(held) == (f"graph2 runs the listing art:graph2s (2 task(s)), and the loop read {ours} (1 task(s)), which "
                         "lacks pytest:z")
    j = w.verdict(sha, built["base"])
    assert (j.verdict, j.disputed, j.note) == (Verdict(PENDING), (held,),
                                               f"graph2 says fail; held until graph2 and the loop agree: {held}")
    acts = [r for _ in range(3) for r in loop.tick()]
    (ring,) = acts
    assert (ring["action"], ring["sha"], ring["why"], ring["to"], ring["done"]) == ("ring", sha, str(held), ["@ci"], True)
    assert (ring["graph2"], ring["mine"], ring["lacks"]) == (
        {"art": "art:graph2s", "tasks": 2}, {"art": ours, "tasks": 1}, ["pytest:z"])
    assert env.posts == [f"@ci vqueue: {ring['text']}"] and f"commit:{sha} --key verity_graph" in ring["text"]
    assert env.main() != sha and env.withdrawn == []
    verdict.write_text(json.dumps({"state": PASS, "pending": 0, "tasks": 1, "off_gate": 0, "art": None}))
    j = w.verdict(sha, built["base"])
    assert (j.verdict, j.disputed, j.note) == (Verdict(PENDING), (),
                                               "graph2 says pass; held: graph2 hasn't labelled the listing it runs yet")
    assert L._relisted({"labelled": "2026-01-01T00:00:00Z", "art": "art:graph2s"}, ours, {}) is None
    verdict.write_text(json.dumps({"state": PASS, "pending": 0, "tasks": 1, "off_gate": 0, "art": ours}))
    assert _acts(loop.tick()) == ["advance", "on_main", "drop"] and env.main() == sha and len(env.posts) == 1


def test_a_fault_main_has_too_is_inherited_by_what_it_names_so_no_pr_touching_its_inputs_is_ejected(env):
    """graph2 names no failing test of a task's own fault: its tests are the files it read outside its declared inputs
    (`undeclared:<path>`), else its fault (`fault:...`, without the run and node it names), from graph2's verdicts on the
    candidate and on its base, so a fault `main` has too is inherited where the change moved the task's key."""
    env.put(1, {"t.txt": "1\n"})
    loop = env.loop(vqueue.decide)
    script = env.graph2_script(loop)
    (built,) = loop.tick()
    sha, main = built["sha"], built["base"]
    env.listed(main, {"pytest:t": ["t.txt"]}, keys={"pytest:t": "sha512:t0"})
    env.listed(sha, {"pytest:t": ["t.txt"]}, keys={"pytest:t": "sha512:t1"})

    def faulted(at: str, key: str, run: str, fault: str) -> None:
        Path(f"{script}.verdict.ready:{at}").write_text(json.dumps({"state": FAIL, "pending": 0, "failed": [
            {"task": "pytest:t", "key": key, "tests": [], "why": f"run {run} exit 1; error: {CLEAN}; {fault}"}]}))

    undeclared = "read 2 file(s) outside its declared inputs: x.txt, y.txt"
    faulted(main, "sha512:t0", "r20260101-000000-aaaa on n1", undeclared)
    faulted(sha, "sha512:t1", "r20260101-000001-bbbb on node-12", undeclared)
    (f,) = loop.world.verdict(sha, main).verdict.failed
    assert f.tests == f.base_tests == {"undeclared:x.txt", "undeclared:y.txt"} and vqueue._inherited(f)
    faulted(sha, "sha512:t1", "r20260101-000001-bbbb on n2", "read 3 file(s) outside its declared inputs: x.txt, y.txt, z.txt")
    assert vqueue._inherited(loop.world.verdict(sha, main).verdict.failed[0])   # one the change adds: main faults it too
    race = 'its runner caught a race: {"error": "race (t.txt)"}'
    faulted(main, "sha512:t0", "r20260101-000000-aaaa on n1", race)
    faulted(sha, "sha512:t1", "r20260101-000001-bbbb on node-12", race)
    (f,) = loop.world.verdict(sha, main).verdict.failed
    assert f.tests == f.base_tests == {f"fault:{race}"}
    assert L._fault_tests("run r20260101-000000-aaaa on n1: storing its artifacts: gone", "sha512:t1") == {
        "fault:run: storing its artifacts: gone"}
    acts = loop.tick()
    assert "eject" not in _acts(acts) and "advance" in _acts(acts) and env.main() == sha


#: a runner's detail of a clean run, as graph2 writes it in a faulted task's `why`
CLEAN = '{"errors": 0, "exit": 0, "failed": 0, "passed": 28, "skipped": 0}'
#: the guard's report on a run that started programs it can't trace, cut at graph2's 200 characters
UNTRACED = ('{"error": "its tests started programs the read guard can\'t trace (g++, git): name the file under its suite\'s '
            '`[tool.verity.tests] untraced`, which keys it on the suite\'s whole tree, or give the test a')


@pytest.mark.parametrize(("theirs", "mine", "inherits"), [
    (CLEAN, CLEAN, True),                                   # both runs clean: only their reads differ
    (CLEAN, '{"errors": 1, "exit": -9, "failed": 0, "passed": 0, "skipped": 0}', False),   # killed beside the reads
    ('{"errors": 1, "exit": 1, "failed": 1, "failed_tests": ["t.py::a"], "passed": 3, "skipped": 0}',
     '{"errors": 1, "exit": 1, "failed": 1, "failed_tests": ["t.py::a"], "passed": 4, "skipped": 0}', True),  # main fails it too
    ('{"errors": 1, "exit": 1, "failed": 1, "failed_tests": ["t.py::a"], "passed": 3, "skipped": 0}',
     '{"errors": 1, "exit": 1, "failed": 1, "failed_tests": ["t.py::b"], "passed": 3, "skipped": 0}', False),  # a new failure
    (UNTRACED, UNTRACED, True),                             # the same report, as far as graph2 keeps it
    (CLEAN, '{"errors": 2, "exit": 1, "failed": 0, "failed_tests": ["t.py::a", "t.py::', False),  # its failures cut
    (None, None, False),                                    # neither run's detail in the 300 characters graph2 keeps
    ('{"error": "Fatal Python error: Segmentation fault"}', '{"error": "Fatal Python error: Segmentation fault"}', True),
])
def test_what_a_run_failed_beside_its_undeclared_reads_is_inherited_only_where_main_s_run_failed_it_too(env, theirs, mine,
                                                                                                        inherits):
    """A task that read files outside its declared inputs is graph2's fault whatever its run did, so the task-level rule
    (`vqueue._inherited`) would take a crash or a new failing test beside the reads for `main`'s fault.  The runner's detail
    in graph2's `why` keeps them: a clean run adds no test, a failing one its failing tests or its error, and a detail graph2
    cut is a test of the candidate's key alone, which no base fails.  The verdict's `error`, when it gives one, is read
    before the `why`."""
    env.put(1, {"t.txt": "1\n"})
    loop = env.loop(vqueue.decide)
    script = env.graph2_script(loop)
    (built,) = loop.tick()
    sha, main = built["sha"], built["base"]
    env.listed(main, {"pytest:t": ["t.txt"]}, keys={"pytest:t": "sha512:t0"})
    env.listed(sha, {"pytest:t": ["t.txt"]}, keys={"pytest:t": "sha512:t1"})

    def faulted(at: str, key: str, detail: str | None, paths: str = "x.txt, y.txt", whole: dict | None = None) -> None:
        reads = f"read 2 file(s) outside its declared inputs: {paths}"
        why = f"run r20260101-000000-aaaa on n1 exit 1; error: {detail}; {reads}" if detail else f'b"]}}; {reads}'
        entry = {"task": "pytest:t", "key": key, "tests": [], "why": why[-300:], **({"error": whole} if whole else {})}
        Path(f"{script}.verdict.ready:{at}").write_text(json.dumps({"state": FAIL, "pending": 0, "failed": [entry]}))

    faulted(main, "sha512:t0", theirs)
    faulted(sha, "sha512:t1", mine)
    (f,) = loop.world.verdict(sha, main).verdict.failed
    assert {t for t in f.tests if t.startswith("undeclared:")} == {"undeclared:x.txt", "undeclared:y.txt"}
    assert vqueue._inherited(f) is inherits, (f.tests, f.base_tests)
    message = ("its tests started programs the read guard can't trace (g++, git): name the file under its suite's "
               "`[tool.verity.tests] untraced`, which keys it on the suite's whole tree, or give the test an untraced mark")
    detail = json.dumps({"error": message})
    files = [f"adapters/vllm/verity_vllm/pipeline/{n * 'x'}.py" for n in (30, 40, 50, 60)]
    paths = ", ".join(files[:3])
    for at, key in ((main, "sha512:t0"), (sha, "sha512:t1")):   # a `why` cut before its detail
        faulted(at, key, detail[:200], paths)
    (f,) = loop.world.verdict(sha, main).verdict.failed
    assert not vqueue._inherited(f) and "fault:the run's detail graph2 cut, at sha512:t1" in f.tests
    whole = {"detail": {"error": message}, "undeclared_reads": files, "status": "error"}   # graph2's run result, whole
    faulted(main, "sha512:t0", detail[:200], paths, whole)
    faulted(sha, "sha512:t1", detail[:200], paths, whole)
    (f,) = loop.world.verdict(sha, main).verdict.failed
    assert vqueue._inherited(f) and f.tests == f.base_tests and f"fault:{message[:L.MESSAGE_KEEP]}" in f.tests
    assert f"undeclared:{files[3]}" in f.tests                  # every read, past the three a `why` names
    faulted(main, "sha512:t0", f"{detail[:200]}", "x.txt, y.txt")   # a base faulted before graph2 gave its result whole
    (f,) = loop.world.verdict(sha, main).verdict.failed
    assert vqueue._inherited(f), (f.tests, f.base_tests)
    faulted(sha, "sha512:t1", None, paths, {**whole, "detail": {"errors": 1, "exit": -9, "failed": 0}})
    assert not vqueue._inherited(loop.world.verdict(sha, main).verdict.failed[0])   # killed: never main's fault


def test_a_pr_the_priority_file_names_ahead_of_prs_its_head_holds_lands_after_them_and_drops_no_candidate(env):
    """A stack lands its bases first, ranked or not: a PR the priority file names ahead of a PR its head holds goes after it,
    so no ranking takes that PR off the ready list, and a candidate in flight holding it is never dropped for it (two were, 38
    minutes into their check, when a ranked PR held a member's head)."""
    one = env.put(1, {"a.txt": "1\n"})
    env.o.gh.put(2, env.o.change({"b.txt": "2\n"}, base=one))
    env.put(3, {"c.txt": "3\n"})
    loop = env.loop(lambda s: [])
    w = loop.world
    loop.tick()
    assert [r.number for r in w.ready(env.main())] == [1, 2, 3]       # #1 before #2, which holds it
    (prio := env.tmp / "priority").write_text("2\n")
    w.priority_file = prio
    ready = {r.number: r for r in w.ready(env.main())}
    assert list(ready) == [1, 2, 3]                                   # #2 ranked first still goes after #1
    flying = vqueue.Candidate(sha="c" * 40, base=env.main(), members=(1,), heads=(one,))
    assert vqueue._stale(flying, set(), w.left, frozenset(w.pulled)) is None     # a candidate holding #1 stays
    prio.write_text("1\n2\n")
    assert [r.number for r in w.ready(env.main())] == [1, 2, 3] and w.left == {}


def test_a_head_landed_on_the_old_paths_checked_holds_its_holders_back_from_nothing_but_still_lands_first(env):
    """`checked`, frozen since the switch, holds heads that landed under drive's rules and reach `main` only in an aggregate
    of it.  A PR holding one doesn't wait for the PR that landed it, though that PR moved on unmarked since, but still goes
    after an open PR whose landed head it holds, ranked first or not.  Without `checked` it waits; once `main` holds
    `checked`, those heads are on `main`."""
    one, four = env.o.change({"a.txt": "1\n"}), env.o.change({"d.txt": "4\n"})
    _git(env.o.dev, "checkout", "-q", "--detach", "origin/main")
    for n, head in ((1, one), (4, four)):
        _git(env.o.dev, "merge", "-q", "--no-ff", "-m", f"Queue: #{n} landed on checked", head)
    checked = _git(env.o.dev, "rev-parse", "HEAD")
    _git(env.o.dev, "push", "-q", "origin", f"{checked}:refs/heads/checked")
    env.o.gh.put(1, env.o.change({"a.txt": "2\n"}, base=one), ready=False)      # #1 moved on from the head that landed, unmarked
    env.o.gh.put(4, four)                                                        # #4 is still at the head that landed, marked
    env.o.gh.put(2, env.o.change({"b.txt": "2\n"}, base=checked))                # #2 holds both landed heads through `checked`
    env.labels.marked_by = lambda n, h: f"lane-{n}" if (n, h) in env.o.gh.ready else None   # no mark withdrawn at #1's new head
    loop = env.loop(lambda s: [])
    w = loop.world
    loop.tick()
    assert [r.number for r in w.ready(env.main())] == [4, 2]
    (prio := env.tmp / "priority").write_text("2\n")
    w.priority_file = prio
    assert [r.number for r in w.ready(env.main())] == [4, 2]                   # ranked first, #2 still goes after #4
    _git(env.o.origin, "update-ref", "-d", "refs/heads/checked")
    w.fetch()
    assert [r.number for r in w.ready(env.main())] == [4]                      # #2 contains #1's commits, and #1 isn't marked
    _git(env.o.origin, "update-ref", "refs/heads/checked", checked)
    _git(env.o.origin, "update-ref", "refs/heads/main", checked)
    w.fetch()
    assert [r.number for r in w.ready(env.main())] == [2]


def test_a_pr_holding_a_marked_pr_that_isnt_taken_waits_for_it_so_it_carries_no_unreviewed_change(env):
    """A land verdict covers its PR's own change: a PR stacked on a marked PR that is held, or has no land verdict, isn't
    taken until that one is, else it would carry that change onto `main` under its own land."""
    one = env.put(1, {"a.txt": "1\n"})
    env.o.gh.put(2, two := env.o.change({"b.txt": "2\n"}, base=one))
    env.labels.held.add(1)
    loop = env.loop(lambda s: [])
    w = loop.world
    loop.tick()
    assert [r.number for r in w.ready(env.main())] == []
    held = "held by reviewer (no reason)"
    assert w.left == {1: (one, held), 2: (two, f"it holds #1's head {one[:12]}, which isn't taken: {held}")}, w.left
    env.labels.held.discard(1)
    assert [r.number for r in w.ready(env.main())] == [1, 2] and w.left == {}


def test_a_pr_renaming_or_deleting_under_a_frozen_path_waits_while_the_freeze_lasts_and_an_edit_there_doesnt(env):
    """A freeze (a file of pathspecs, `FROZEN`) keeps another branch mergeable: while it exists, a PR whose own change renames
    or deletes a file under one of its paths waits, an ordinary edit there doesn't, and once the file is gone both land."""
    env.put(1, {"a.txt": None})
    env.put(2, {"b.txt": "2\n"})
    env.put(3, {"c.txt": "3\n", "pkg/x.py": "x = 1\n"})
    loop = env.loop(lambda s: [])
    w = loop.world
    loop.tick()
    (frozen := env.tmp / "frozen").write_text("# until remote lands\na.txt\nb.txt\n:(glob)**/x.py\n")
    w.frozen_file = frozen
    assert [r.number for r in w.ready(env.main())] == [2, 3]
    assert w.left == {1: (env.o.gh.prs[1].head, f"it waits for the freeze to end ({frozen})")}, w.left
    assert sum("#1 at" in x and "waits for the freeze to end" in x for x in env.out) == 1, env.out
    w.ready(env.main())
    assert sum("#1 at" in x and "waits for the freeze to end" in x for x in env.out) == 1, env.out    # once a head
    head = env.o.gh.prs[1].head
    cleared: dict[tuple[int, str, str], list[Lab]] = {}
    env.labels._counted = lambda n, h, key: (cleared.get((n, h, key), []), [])
    frozen.write_text(frozen.read_text() + "cleared-by vllm remote-cleared\n")
    cleared[(1, head, "remote-cleared")] = [Lab("lane-1", "true", "2026-01-01T00:00:00Z")]
    assert [r.number for r in w.ready(env.main())] == [2, 3]                # cleared, but not by the freeze's owner
    cleared[(1, head, "remote-cleared")] = [Lab("vllm", "true", "2026-01-01T00:00:00Z")]
    assert [r.number for r in w.ready(env.main())] == [1, 2, 3]             # the owner cleared that head
    cleared.clear()
    frozen.unlink()
    assert [r.number for r in w.ready(env.main())] == [1, 2, 3]


def test_a_freeze_reads_a_pr_s_own_change_above_the_old_paths_checked_not_the_landings_it_carries(env):
    """A PR that merged `checked` carries its old-path landings, renames and deletions included, which land as built: while
    `main` doesn't hold `checked`, the freeze reads only what the PR changes above it."""
    gone = env.o.change({"a.txt": None})
    _git(env.o.dev, "checkout", "-q", "--detach", "origin/main")
    _git(env.o.dev, "merge", "-q", "--no-ff", "-m", "Queue: #9 landed on checked", gone)
    checked = _git(env.o.dev, "rev-parse", "HEAD")
    _git(env.o.dev, "push", "-q", "origin", f"{checked}:refs/heads/checked")
    env.o.gh.put(4, env.o.change({"c.txt": "4\n"}, base=checked))           # merged `checked`, edits only
    env.o.gh.put(5, env.o.change({"b.txt": None}, base=checked))            # merged `checked`, deletes b.txt itself
    loop = env.loop(lambda s: [])
    w = loop.world
    loop.tick()
    (frozen := env.tmp / "frozen").write_text("a.txt\nb.txt\n")
    w.frozen_file = frozen
    assert [r.number for r in w.ready(env.main())] == [4]


def test_each_tick_reads_the_labels_afresh_as_drive_reads_them(env):
    """Each tick reads new labels, carrying a mark by own change, and by an old head's own change mapped across `main`'s moves
    while the queue maps heads (`Queue.mapped_jobs`); each ends by starting to map the old heads the labels wanted."""
    made: list[Labels] = []
    worked: list[int] = []
    loop = env.loop(lambda s: [])
    loop.world.labels_at = lambda: made.append(Labels(env.o.gh)) or made[-1]
    loop.world.q.work_out_mapped = lambda: worked.append(loop.last) or 0
    loop.tick()
    loop.world.q.mapped_jobs = 2
    loop.tick()
    w = loop.world
    assert len(made) == 2 and w.labels is made[-1] and w.q.ready == made[-1].ready and w.q.marked_at == made[-1].marked_at
    assert made[-1].change == w.q.own_change and made[-1].since == w.q.rules().marked_after
    assert made[0].moved is None and made[-1].moved == w.q.mapped_change and worked == [loop.last - 1, loop.last]


def test_a_tick_whose_mapping_of_old_heads_fails_says_so_and_returns_what_it_logged(env):
    """The end of a tick maps old heads across `main`'s moves; a git failure there is said, and the tick's records stand."""
    out: list[str] = []
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy)
    loop.out = out.append

    def fails():
        raise Q.QueueError("git ls-tree: broken")

    loop.world.q.work_out_mapped = fails
    (built,) = loop.tick()
    assert built["action"] == "build"
    assert out[-1] == f"research queue vqueue: tick {loop.last}: mapping old heads across `main`'s moves failed: git ls-tree: broken"


def test_move_carry_jobs_says_how_many_old_heads_the_loop_maps_across_moves_at_once(monkeypatch, tmp_path):
    """`--move-carry-jobs N` is the queue's `mapped_jobs` (default `MAPPED_JOBS`); 0, or less, maps none, so no mark or
    verdict carries across a move.  The mappings are kept beside the own changes, in the log's directory."""
    seen: list[int] = []

    class Built(Exception):
        pass

    def loop(world, *args, **kw):
        seen.append(world.q.mapped_jobs)
        assert (world.q._kept_at, world.q._mapped_at) == (tmp_path / "log" / "own-changes.json",
                                                          tmp_path / "log" / "mapped-changes.json")
        raise Built

    monkeypatch.setattr(L, "Loop", loop)
    monkeypatch.setattr(Q, "remote_and_repo", lambda *a: ("origin", "owner/repo"))
    monkeypatch.setattr(Q, "credential_args", lambda *a: ())
    monkeypatch.setattr(Q, "GitHub", lambda slug: None)
    common = ["--repo", str(tmp_path / "repo"), "--store", str(tmp_path / "store"), "--lock", str(tmp_path / "lock"),
              "--log", str(tmp_path / "log" / "decisions.jsonl"), "--once"]
    for extra in ([], ["--move-carry-jobs", "0"], ["--move-carry-jobs", "-3"], ["--move-carry-jobs", "7"]):
        with pytest.raises(Built):
            L.main([*common, *extra])
    assert seen == [L.MAPPED_JOBS, 0, 0, 7]


#: a land verdict by a reviewer, not the PR's owner
LAND = Lab("reviewer", "land", "2026-01-01T00:00:00Z")


def _carry_main(env: Env, mode: str) -> str:
    """`main` moved to a commit setting drive.toml's `verdict_carry` to ``mode``, its check listed with no task."""
    main = env.o.change({DRIVE_FLAGS: f'verdict_carry = "{mode}"\n'})
    _git(env.o.origin, "update-ref", "refs/heads/main", main)
    env.graph2.list_graph(f"ready:{main}", {"schema": GRAPH_V2, "commit": main, "tasks": []})
    return main


@pytest.mark.parametrize("mode", ["on", "shadow", "off"])
def test_a_land_at_an_older_head_carries_to_its_merge_of_main_as_drive_toml_s_verdict_carry_says_logged_once(
        env, monkeypatch, mode):
    """#1's land verdict is at ``old``, and its head only merges `main` into it: the verdict carry
    (`Landing._carried_from`) takes it, logged once a head.  With "on" #1 counts as reviewed and is built, with "shadow" the
    carry is only logged, and with "off" nothing is judged; each `carry.delta` is judged once while `main` stays."""
    judged, real = [], carry.delta
    monkeypatch.setattr(carry, "delta", lambda repo, old, new, trust: judged.append((old, new)) or real(repo, old, new, trust))
    old = env.o.change({"a.txt": "1\n"})
    main = _carry_main(env, mode)
    env.o.gh.put(1, new := _merge(env.o, old, main))
    env.labels.verdicts |= {(1, old): [LAND], (1, new): []}
    loop = env.loop(policy)
    recs = loop.tick()
    verb = {"on": "carried", "shadow": "would carry"}.get(mode)
    assert [(r["number"], r["heads"], r["old"], r["carries"], r["why"]) for r in recs if r["action"] == "carry"] == (
        [(1, [new], old, True, f"{verb} #1's land from {old[:12]} to {new[:12]} (its 1 commit(s) since only merge {main[:12]} "
                               "as the queue would)")] if verb else [])
    assert [r["members"] for r in recs if r["action"] == "build"] == ([[1]] if mode == "on" else [])
    assert [r for r in loop.tick() if r["action"] == "carry"] == [] and judged == ([(old, new)] if verb else [])
    assert env.labels.verdicts[1, new] == []                            # the loop writes no verdict


@pytest.mark.parametrize("change, because", [
    ("a.txt", "authored: {new} changes a.txt"),
    ("security/Proofs/P.lean", "authored: {new} changes security/Proofs/P.lean"),
    ("held", "held by other (the merge breaks a) at {new}")])
def test_a_head_that_changes_more_than_its_merge_of_main_or_is_held_carries_no_land(env, change, because):
    """drive's safety checks carry over: a commit of the PR's own since the land (a Lean one too) or a hold at the head keeps
    the land where it is, and the loop logs that it did not carry it."""
    old = env.o.change({"a.txt": "1\n"})
    main = _carry_main(env, "on")
    new = _merge(env.o, old, main)
    if change != "held":
        new = env.o.change({change: "2\n"}, base=new)
    env.o.gh.put(1, new)
    env.labels.verdicts |= {(1, old): [LAND],
                            (1, new): [Lab("other", "hold", "2026-01-02T00:00:00Z", "the merge breaks a")] if change == "held" else []}
    (rec,) = env.loop(policy).tick()
    assert (rec["action"], rec["carries"], rec["why"]) == (
        "carry", False, f"did not carry #1's land from {old[:12]} to {new[:12]} ({because.format(new=new[:12])})")


def test_a_land_by_the_pr_s_owner_at_an_older_head_carries_nothing(env):
    """#1's owning lane isn't its ready mark's author here, and its own land at ``old`` counts no more than at its head."""
    old = env.o.change({"a.txt": "1\n"})
    main = _carry_main(env, "on")
    env.o.gh.put(1, new := _merge(env.o, old, main), body="Lane `lane-9`.")
    env.labels.verdicts |= {(1, old): [Lab("lane-9", "land", "2026-01-01T00:00:00Z")], (1, new): []}
    assert env.loop(policy).tick() == []


def test_a_land_carries_only_to_a_head_marked_ready_there_and_not_to_one_the_hold_label_keeps_out(env):
    """The carry stands in for a land verdict alone, and judges only an admitted PR, as drive judges only `next`'s marked
    members: #1's merge of `main` takes no land while it isn't marked ready there (the mark at ``old`` counts no more) or
    while the hold label keeps it out, and takes it once it is marked and the label is gone."""
    old = env.o.change({"a.txt": "1\n"})
    main = _carry_main(env, "on")
    new = _merge(env.o, old, main)
    env.o.gh.put(1, old)
    env.o.gh.put(1, new, ready=False)
    env.labels.verdicts |= {(1, old): [LAND], (1, new): []}
    loop = env.loop(policy)
    assert loop.tick() == []
    env.o.gh.put(1, new, labels=("hold",))
    assert loop.tick() == []
    env.o.gh.put(1, new)
    assert [(r["action"], r["members"], r.get("carries")) for r in loop.tick() if r["action"] in ("carry", "build")] == [
        ("carry", [1], True), ("build", [1], None)]


@pytest.mark.parametrize("two", ["reviewed", "carried"])
def test_a_merge_of_another_pr_s_head_carries_once_that_head_lands_on_its_own_verdict_or_a_carried_one(env, two):
    """#1 merges #2's head, which has a land verdict of its own or one #2's carry takes: #2 is judged first, as #1 contains it,
    so #1's merge is of a trusted head and its land carries too, and both are built, #2 first."""
    old2, old1 = env.o.change({"b.txt": "2\n"}), env.o.change({"a.txt": "1\n"})
    main = _carry_main(env, "on")
    new2 = _merge(env.o, old2, main)
    new1 = _merge(env.o, old1, new2)
    env.o.gh.put(1, new1)
    env.o.gh.put(2, new2)
    env.labels.verdicts |= {(1, old1): [LAND], (1, new1): []}
    env.labels.verdicts |= {(2, new2): [LAND]} if two == "reviewed" else {(2, old2): [LAND], (2, new2): []}
    recs = env.loop(policy).tick()
    assert [(r["number"], r["carries"], r["why"].split(" (")[1]) for r in recs if r["action"] == "carry"] == [
        (1, True, f"its 1 commit(s) since only merge {new2[:12]} as the queue would)"),
        *([(2, True, f"its 1 commit(s) since only merge {main[:12]} as the queue would)")] if two == "carried" else [])]
    assert [r["members"] for r in recs if r["action"] == "build"] == [[2, 1]]


def test_a_merge_of_the_old_path_s_checked_carries_a_land_as_drive_trusts_checked_beside_main(env):
    """`checked`, frozen since the switch, holds what landed under drive's rules and reaches `main` in an aggregate of it: a
    head that only merges it carries its land, as drive's carry trusts the tips of `checked` and `main` (`carry.trust`)."""
    old = env.o.change({"a.txt": "1\n"})
    _carry_main(env, "on")
    checked = env.o.change({"c.txt": "landed on checked\n"})
    _git(env.o.origin, "update-ref", "refs/heads/checked", checked)
    env.o.gh.put(1, new := _merge(env.o, old, checked))
    env.labels.verdicts |= {(1, old): [LAND], (1, new): []}
    recs = env.loop(policy).tick()
    assert [(r["number"], r["carries"], r["why"]) for r in recs if r["action"] == "carry"] == [
        (1, True, f"carried #1's land from {old[:12]} to {new[:12]} (its 1 commit(s) since only merge {checked[:12]} as the "
                  "queue would)")]
    assert [r["members"] for r in recs if r["action"] == "build"] == [[1]]


@pytest.mark.parametrize(("flag", "default"), [("--wide-file", Path("/data/vqueue/live/wide")),
                                               ("--pins-file", Path("/data/drive/pins"))])
def test_wide_file_and_pins_file_name_the_files_the_world_reads_each_tick_by_default_the_live_loop_s(monkeypatch, tmp_path,
                                                                                                      flag, default):
    seen: list[Path | None] = []
    attr = flag.removeprefix("--").replace("-", "_")

    class Built(Exception):
        pass

    def loop(world, *args, **kw):
        seen.append(getattr(world, attr))
        raise Built

    monkeypatch.setattr(L, "Loop", loop)
    monkeypatch.setattr(Q, "remote_and_repo", lambda *a: ("origin", "owner/repo"))
    monkeypatch.setattr(Q, "credential_args", lambda *a: ())
    monkeypatch.setattr(Q, "GitHub", lambda slug: None)
    common = ["--repo", str(tmp_path / "repo"), "--store", str(tmp_path / "store"), "--lock", str(tmp_path / "lock"),
              "--log", str(tmp_path / "log" / "decisions.jsonl"), "--once"]
    for extra in ([], [flag, str(tmp_path / "file")]):
        with pytest.raises(Built):
            L.main([*common, *extra])
    assert seen == [default, tmp_path / "file"]


@pytest.mark.parametrize("heavy, on", [("1\n", "big"), ("1\n", ""), ("*\n", "big")], ids=["big", "off", "every"])
def test_a_heavy_pr_is_wide_its_batch_taking_no_other_lean_member_while_heavy_on_names_machines(env, heavy, on):
    """`--heavy-file` (`World.heavy`) is read into `Snapshot.wide` each tick beside the wide file, so a heavy PR's batch takes
    no other Lean member; with no `--heavy-on` it is no heavy PR, and the `lean` cap alone makes the batch, as it does when
    the file's `*` makes every candidate heavy, which makes no PR wide."""
    _main_to(env, {"pkg/lean-audit.json": "{}\n", "pkg/A.lean": "-- a\n"})
    env.put(1, {"pkg/A.lean": "-- b\n"})
    env.put(2, {"security/Definitions/D.lean": "-- d\n"})
    seen: list[Snapshot] = []
    loop = env.loop(lambda s: seen.append(s) or [], lean=3)
    (env.tmp / "heavy").write_text(heavy)
    loop.world.heavy_file, loop.world.heavy_on = env.tmp / "heavy", on
    loop.tick()
    wide = on and heavy == "1\n"
    assert seen[-1].wide == ({1} if wide else frozenset())
    assert vqueue.decide(seen[-1]) == [Build((1,) if wide else (1, 2), seen[-1].main, L.DEPTH)]


def test_heavy_file_and_heavy_on_name_what_the_world_reads_by_default_none_but_the_live_loop_s_file(monkeypatch, tmp_path):
    seen: list[tuple] = []

    class Built(Exception):
        pass

    def loop(world, *args, **kw):
        seen.append((world.heavy_file, world.heavy_on))
        raise Built

    monkeypatch.setattr(L, "Loop", loop)
    monkeypatch.setattr(Q, "remote_and_repo", lambda *a: ("origin", "owner/repo"))
    monkeypatch.setattr(Q, "credential_args", lambda *a: ())
    monkeypatch.setattr(Q, "GitHub", lambda slug: None)
    common = ["--repo", str(tmp_path / "repo"), "--store", str(tmp_path / "store"), "--lock", str(tmp_path / "lock"),
              "--log", str(tmp_path / "log" / "decisions.jsonl"), "--once"]
    for extra in ([], ["--heavy-file", str(tmp_path / "file"), "--heavy-on", "big1,big2"]):
        with pytest.raises(Built):
            L.main([*common, *extra])
    assert seen == [(Path("/data/vqueue/live/heavy"), ""), (tmp_path / "file", "big1,big2")]


def _heavy(file: Path, on: str) -> tuple[frozenset[int], bool]:
    return L.World.heavy(SimpleNamespace(heavy_file=file, heavy_on=on))


@pytest.mark.parametrize("text, want", [
    ("*\n", (frozenset(), True)), ("12\n*\n7\n", (frozenset({7, 12}), True)), ("12\n7\n", (frozenset({7, 12}), False)),
    ("", (frozenset(), False)),
], ids=["every", "both", "prs", "empty"])
def test_the_heavy_file_names_prs_or_every_candidate_with_a_star_and_the_wide_file_takes_no_star(tmp_path, text, want):
    (f := tmp_path / "heavy").write_text(text)
    assert _heavy(f, "big") == want
    assert _heavy(f, "") == (frozenset(), False)
    if "*" in text:
        with pytest.raises(Q.QueueError, match=r"--wide-file .* holds '\*', which is not a PR number: write PR numbers "
                                               r"separated by spaces or commas$"):
            Q.priority_list(f, "--wide-file")


def test_a_heavy_file_holding_anything_but_pr_numbers_and_a_star_is_refused_naming_the_star(tmp_path):
    (f := tmp_path / "heavy").write_text("12\nall\n")
    with pytest.raises(Q.QueueError, match=r"holds 'all', which is not a PR number: write PR numbers separated by spaces or "
                                           r"commas, or \* for every PR$"):
        _heavy(f, "big")


def test_only_the_loop_s_own_builds_are_candidates_and_another_queue_branch_is_ignored_logged_once(env):
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy, state=(state := env.tmp / "vqueue" / "state.json"))
    (built,) = loop.tick()
    sha = built["sha"]
    forged = env.o.change({"f.txt": "forged\n"})                       # anyone who can push to origin
    _git(env.o.origin, "update-ref", f"refs/heads/queue/{forged[:12]}", forged)
    _git(env.o.origin, "update-ref", "refs/heads/queue/mine", sha)
    ignored = loop.tick()
    assert sorted((r["action"], r["ref"], r["sha"]) for r in ignored) == sorted([("ignore", forged[:12], forged),
                                                                                 ("ignore", "mine", sha)])
    assert loop.tick() == [] and [e["check"] for e in env.graph2.enqueued()] == [f"ready:{sha}"]    # once, and never enqueued
    assert json.loads(state.read_text())["ignored"] == {forged[:12]: forged, "mine": sha}
    assert list(json.loads(state.read_text())["candidates"]) == [sha]
    env.clock += 49 * 3600                                          # its build is older than the log's window
    keys = env.listed(sha, {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    assert [a for a in _acts(loop.tick()) if a != "ignore"] == ["advance", "on_main"] and env.main() == sha  # the whole log's


def test_a_failed_task_s_inputs_hold_its_env_s_spec_files_and_those_of_each_task_it_reads(env):
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(lambda s: [] if s.candidates else [Build((1,), s.main, 5)])
    (built,) = loop.tick()
    sha, w = built["sha"], loop.world
    keys = env.listed(sha, {"lean-build:x": ["x.lean"], "pytest:a": ["a.txt"]}, reads={"pytest:a": "lean-build:x"})
    env.result(keys["lean-build:x"], "lean-build:x", "pass")
    _, made, _ = w._settled(w.tasks.graph_v2(f"ready:{sha}")[1])
    env.result(made["pytest:a"], "pytest:a", "fail")
    (f,) = w.verdict(sha).verdict.failed
    assert (f.task, f.key, f.inputs) == ("pytest:a", made["pytest:a"], frozenset({"a.txt", "x.lean", *SPEC}))


def test_the_snapshot_reads_the_clock_lean_and_pending_hours_and_a_check_pending_past_them_is_retried_once_and_rung(env):
    env.put(1, {"a.txt": "1\n"})
    seen: list[Snapshot] = []
    loop = env.loop(lambda s: seen.append(s) or vqueue.decide(s), lean=3, pending_s=2 * 3600)
    (built,) = loop.tick()
    assert (seen[-1].lean, seen[-1].now, seen[-1].pending_s) == (3, T0, 7200)
    env.clock += 2 * 3600
    assert loop.tick() == []
    assert (seen[-1].now, seen[-1].candidates[0].enqueued, seen[-1].candidates[0].tries) == (T0 + 7200, T0, 1)
    env.clock += 1                                                  # past `pending_s`: enqueued again, once
    (retry,) = loop.tick()
    assert (retry["action"], retry["enqueued"], retry["counts"]) == ("retry", True, {})
    assert loop.tick() == [] and len(env.graph2.enqueued()) == 2
    env.clock = T0 + 4 * 3600 + 1                                   # past twice that: rung, once
    (ring,) = loop.tick()
    assert ring["action"] == "ring" and "still pending 4 h after its first enqueue" in ring["why"], ring
    assert loop.tick() == [] and len(env.posts) == 1 and env.posts[0].startswith("@infra vqueue:")
    assert seen[-1].candidates[0].enqueued == T0 and seen[-1].candidates[0].tries == 2


def test_a_member_counts_toward_the_lean_cap_only_when_its_lean_change_compiles(env):
    """`--lean` caps a candidate's members whose Lean change compiles: a Lean source's of a package with a lock, the spec's,
    or a proof's beside a lock's records.  A change to a lock's records alone (`lean-audit.json`) compiles nothing, and
    counts no more than a change with no Lean."""
    _main_to(env, {"pkg/lean-audit.json": "{}\n", "pkg/A.lean": "-- a\n"})
    env.put(1, {"pkg/lean-audit.json": '{"records": 1}\n'})
    env.put(2, {"pkg/A.lean": "-- b\n"})
    env.put(3, {"security/Definitions/D.lean": "-- d\n"})
    env.put(4, {"pkg/lean-audit.json": '{"records": 2}\n', "security/Proofs/P.lean": "-- p\n"})
    env.put(5, {"a.txt": "5\n"})
    seen: list[Snapshot] = []
    env.loop(lambda s: seen.append(s) or []).tick()
    assert {r.number: r.lean for r in seen[-1].ready} == {1: False, 2: True, 3: True, 4: True, 5: False}


def test_the_wide_file_is_read_each_tick_as_the_priority_file_is_and_a_line_that_isnt_a_pr_number_fails_the_tick(env):
    """`--wide-file` (`World.wide`) is read into `Snapshot.wide` each tick with the priority file's parsing
    (`queue.priority_list`): one PR number a line, none while the file is missing or empty, so the `lean` cap alone makes the
    Lean batch, and a line that isn't a PR number is refused, naming the file and the line, so the tick fails until it is
    fixed.  A wide PR fills its batch's cap; the PR with no Lean change goes in a batch of its own."""
    _main_to(env, {"pkg/lean-audit.json": "{}\n", "pkg/A.lean": "-- a\n"})
    env.put(1, {"pkg/A.lean": "-- b\n"})
    env.put(2, {"security/Definitions/D.lean": "-- d\n"})
    env.put(3, {"a.txt": "3\n"})
    seen: list[Snapshot] = []
    loop = env.loop(lambda s: seen.append(s) or [], lean=3)
    loop.world.wide_file = wide = env.tmp / "wide"
    loop.tick()
    wide.write_text("")
    loop.tick()
    assert [s.wide for s in seen] == [frozenset(), frozenset()]
    assert [(r.number, r.lean) for r in seen[-1].ready] == [(1, True), (2, True), (3, False)]
    assert vqueue.decide(seen[-1]) == [Build((1, 2), seen[-1].main, L.DEPTH)]
    assert vqueue.decide(replace(seen[-1], ready=seen[-1].ready[2:])) == [Build((3,), seen[-1].main, L.DEPTH)]
    wide.write_text("1\n\n1\n")
    loop.tick()
    assert seen[-1].wide == {1} and vqueue.decide(seen[-1]) == [Build((1,), seen[-1].main, L.DEPTH)]
    wide.write_text("1\n#2\n")
    with pytest.raises(Q.QueueError) as e:
        loop.tick()
    assert str(e.value) == (f"--wide-file {wide} holds '#2', which is not a PR number: write PR numbers separated by spaces "
                            "or commas") and len(seen) == 3


def test_the_pins_file_is_read_each_tick_with_its_comments_and_a_word_that_isnt_a_pr_number_fails_the_tick(env):
    """`--pins-file` (`World.pinned`) is read into `Snapshot.pinned` each tick as the wide file is, a `#` starting a comment
    and blank lines skipped: none while it is missing or empty, and a word that isn't a PR number is refused, naming the
    file, so the tick fails until it is fixed."""
    env.put(1, {"a.txt": "1\n"})
    seen: list[Snapshot] = []
    loop = env.loop(lambda s: seen.append(s) or [])
    loop.world.pins_file = pins = env.tmp / "pins"
    loop.tick()
    pins.write_text("")
    loop.tick()
    pins.write_text("# tonight's Lean PRs\n\n12  # its goal's\n#7\n1\n")
    loop.tick()
    assert [s.pinned for s in seen] == [frozenset(), frozenset(), frozenset({1, 12})]
    pins.write_text("1\n12a\n")
    with pytest.raises(Q.QueueError) as e:
        loop.tick()
    assert str(e.value) == (f"--pins-file {pins} holds '12a', which is not a PR number: write PR numbers separated by spaces "
                            "or commas") and len(seen) == 3


@pytest.mark.parametrize("priority", [None, "", "99\n", "1\n"], ids=["none", "empty", "stale", "lean first"])
def test_at_lean_0_a_lean_pr_opens_no_batch_whatever_its_rank_until_the_pins_file_names_it_and_under_a_cap_it_opens_one(
        env, priority):
    """Rank is the priority file, then mark time: with none, an empty one, a stale one naming only PRs gone, or one listing
    the Lean PR first, the Lean PR marked longest ago ranks first.  At `--lean 0` it still goes in no batch, and the PR with
    no Lean change is built; once the pins file names it, it goes in a batch of its own, leaving that PR for the next.
    Under a `--lean` cap above 0 it opens that batch with no pin."""
    _main_to(env, {"pkg/lean-audit.json": "{}\n", "pkg/A.lean": "-- a\n"})
    env.put(1, {"pkg/A.lean": "-- b\n"})
    env.put(3, {"a.txt": "3\n"})
    seen: list[Snapshot] = []
    loop = env.loop(lambda s: seen.append(s) or [], lean=0)
    if priority is not None:
        (prio := env.tmp / "priority").write_text(priority)
        loop.world.priority_file = prio
    loop.tick()
    assert [(r.number, r.lean) for r in seen[-1].ready] == [(1, True), (3, False)] and seen[-1].lean == 0
    assert vqueue.decide(seen[-1]) == [Build((3,), seen[-1].main, L.DEPTH)]
    assert vqueue.decide(replace(seen[-1], lean=1)) == [Build((1,), seen[-1].main, L.DEPTH)]
    (pins := env.tmp / "pins").write_text("1\n")
    loop.world.pins_file = pins
    loop.tick()
    assert vqueue.decide(seen[-1]) == [Build((1,), seen[-1].main, L.DEPTH)]


def test_with_no_records_run_a_lean_pr_waits_for_a_candidate_with_no_lean_change_and_a_tick_building_nothing_logs_why_once(env):
    """End to end with no records run, where a Lean batch takes a place on main's chain, at most every second build is one:
    the pinned #1 goes alone on `main` (`--batch` 1); #2, pinned too, waits on #1's candidate, which holds a Lean change,
    and with no PR with no Lean change to take, the tick builds nothing and logs why (`wait`), once while that why holds;
    #3, with none, goes on #1's candidate, and #2 on #3's."""
    _main_to(env, {"pkg/lean-audit.json": "{}\n", "pkg/A.lean": "-- a\n", "pkg/B.lean": "-- a\n"})
    env.put(1, {"pkg/A.lean": "-- b\n"})
    env.put(2, {"pkg/B.lean": "-- b\n"})
    (pins := env.tmp / "pins").write_text("1\n2\n")
    loop = env.loop(vqueue.decide, batch=1)
    loop.world.pins_file = pins
    for _ in range(3):
        loop.tick()
    env.put(3, {"a.txt": "3\n"})
    loop.tick()
    loop.tick()
    lines = [r for r in env.lines() if r["action"] in ("build", "wait")]
    assert [(r["action"], r["members"]) for r in lines] == [("build", [1]), ("wait", []), ("build", [3]), ("build", [2])]
    assert lines[1]["why"] == (f"nothing built: with a Lean change, #2 waits while the newest candidate, "
                               f"{lines[0]['sha'][:12]}, holds #1's Lean change, and no PR with no Lean change is ready to take")
    assert [r["base"] for r in lines[2:]] == [lines[0]["sha"], lines[2]["sha"]]


def test_a_failure_s_tests_are_read_beside_its_base_s_and_one_whose_tests_the_base_fails_too_is_inherited(env):
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide)
    (built,) = loop.tick()
    sha, w = built["sha"], loop.world
    env.listed(main, {"pytest:t": ["t.py"]}, keys={"pytest:t": "sha512:base-t"})
    keys = env.listed(sha, {"pytest:a": ["a.txt"], "pytest:t": ["t.py"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    env.result(keys["pytest:t"], "pytest:t", "fail", tests=["test_t.py::a"])
    j = w.verdict(sha, main)                                        # the base's has no result yet: it waits for one
    assert (j.verdict, j.note) == (Verdict(PENDING), f"waiting for its base {main[:12]}: pytest:t there haven't ended")
    env.result("sha512:base-t", "pytest:t", "fail", tests=["test_t.py::a", "test_t.py::b"])
    (f,) = w.verdict(sha, main).verdict.failed
    assert f.base_tests == frozenset({"test_t.py::a", "test_t.py::b"})
    assert _acts(loop.tick()) == ["red", "advance", "on_main", "drop"] and env.main() == sha    # its tests fail there too
    env.put(2, {"b.txt": "2\n"})
    (two,) = loop.tick()
    keys = env.listed(two["sha"], {"pytest:t": ["t.py"]})
    env.result(keys["pytest:t"], "pytest:t", "fail", tests=["test_t.py::a", "test_t.py::c"])
    (f,) = w.verdict(two["sha"], sha).verdict.failed
    assert (f.tests, f.base_tests) == (frozenset({"test_t.py::a", "test_t.py::c"}), frozenset({"test_t.py::a"}))
    assert _acts(loop.tick()) == ["rerun"]                          # one test more fails: its own failure


def test_the_failing_tests_graph2_s_verdict_names_count_beside_its_result_s_for_a_candidate_and_for_its_base(env):
    """A `lean-suites` result's detail names no failing test (`exit`, `failed`, `suites`): graph2's verdict names them, a
    test id of that task's, so a failure whose tests the base fails too, by graph2's verdict on the base, is inherited."""
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide)
    script = env.graph2_script(loop, {"state": PENDING})
    (built,) = loop.tick()
    sha = built["sha"]
    t, node, more = "lean-suites:verity-flock:6/6", "tests/test_rec_full_draw.py::test_draw", "tests/test_x.py::test_y"
    env.listed(main, {t: ["f.py"]}, keys={t: "sha512:base-f"})
    keys = env.listed(sha, {"pytest:a": ["a.txt"], t: ["f.py"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    for at, key, tests in ((main, "sha512:base-f", [node, more]), (sha, keys[t], [node])):
        env.graph2.finish_result({"key": key, "task": t, "status": FAIL, "outputs": {"out": "sha512:o"},
                                  "detail": {"exit": 1, "failed": 1, "suites": ["verity-flock"]}})
        Path(f"{script}.verdict.ready:{at}").write_text(json.dumps({"state": FAIL, "failed": [
            {"task": t, "key": key, "tests": tests, "inputs": ["f.py"], "reads": []},
            {"task": "pytest:q", "key": "sha512:q", "tests": [], "inputs": [], "reads": [], "why": "a race"}]}))
    (f,) = loop.world.verdict(sha, main).verdict.failed
    assert (f.task, f.tests, f.base_key, f.base_tests) == (t, frozenset({node}), "sha512:base-f", frozenset({node, more}))
    assert L._named({"labelled": "2026-01-01T00:00:00Z", "failed": [{"task": t, "key": keys[t], "tests": [node]}]}) == {}
    assert _acts(loop.tick()) == ["red", "advance", "on_main", "drop"] and env.main() == sha    # not rerun: inherited


#: a lean-suites result's row of a suite that failed on one test and nothing more
ROW = {"name": "verity-flock", "suite": "verity/protocols/flock/prover", "failed_tests": ["tests/test_d.py::test_draw"], "beyond_tests": False}
#: the test a fail for more than the tests it names adds, of the candidate's key alone
BEYOND = "fault:suite {} failed beyond the tests it names, at sha512:t1"


@pytest.mark.parametrize("kind, detail, fault", [
    ("test-file", {"exit": 1}, None),
    ("test-file", {"exit": 2}, "fault:its run failed beyond the tests it names (exit 2), at sha512:t1"),
    ("lean-suites", {"failed": [ROW]}, None),
    ("lean-suites", {"failed": [{**ROW, "beyond_tests": True}]}, BEYOND.format("verity-flock")),
    ("lean-suites", {"failed": [{k: v for k, v in ROW.items() if k != "beyond_tests"}]}, BEYOND.format("verity-flock")),
    ("lean-suites", {"failed": [{**ROW, "timed_out": True}]}, BEYOND.format("verity-flock")),
    ("lean-suites", {"failed": [ROW, {**ROW, "name": "verity", "failed_tests": []}]}, BEYOND.format("verity")),
    ("lean-suites", {"failed": [ROW, 7]}, None),
    ("lean-suites", {"failed": None}, None),
])
def test_a_fail_for_more_than_the_tests_it_names_is_never_inherited_by_them(env, kind, detail, fault):
    """A crash beside the tests its base fails too would pass for the base's failure: a test file pytest ended with exit 2
    (collecting failed, or a `pytest.exit`), and a lean-suites suite that names no test, failed for more than them (its list
    cut) or timed out, fails a test of its own key too, which no base fails, so its PR is suspected.  Failed on its tests
    alone, it is inherited, and so is one whose detail has no row to read."""
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide)
    script = env.graph2_script(loop, {"state": PENDING})
    (built,) = loop.tick()
    sha, t, node = built["sha"], f"{kind}:t", ROW["failed_tests"][0]
    base = {"exit": 1, **({"failed": [ROW]} if kind == "lean-suites" else {"failed_tests": [node]})}
    for at, key, d in ((main, "sha512:base-t", base), (sha, "sha512:t1", {**base, **detail})):
        env.graph2.list_graph(f"ready:{at}", _graph(at, [{"id": "pytest:a", "kind": "test-file", "inputs": ["a.txt"],
                                                          "key": "sha512:a"},
                                                         {"id": t, "kind": kind, "inputs": ["t.py"], "key": key}]))
        env.graph2.finish_result({"key": key, "task": t, "status": FAIL, "outputs": {"out": "sha512:o"}, "detail": d})
        Path(f"{script}.verdict.ready:{at}").write_text(json.dumps({"state": FAIL, "failed": [
            {"task": t, "key": key, "tests": [node], "inputs": ["t.py"], "reads": []}]}))
    env.result("sha512:a", "pytest:a", PASS)
    (f,) = loop.world.verdict(sha, main).verdict.failed
    assert (f.task, f.tests, f.base_key, f.base_tests) == (t, {node, *([fault] if fault else [])}, "sha512:base-t", {node})
    assert vqueue._inherited(f) is (fault is None)
    assert _acts(loop.tick()) == (["red", "advance", "on_main", "drop"] if fault is None else ["rerun"])
    assert L._tests({"status": PASS, "detail": d}, "sha512:t1", kind) == L._tests({"status": FAIL, "detail": d}) \
        == frozenset(d.get("failed_tests") or ())


def test_graph2_s_verdict_on_a_base_is_read_once_a_tick_and_afresh_the_next(env):
    """Its `verdict` is a subprocess, and a base's is read for each candidate on it that failed: the base's own judging and
    the candidate above it share one read a tick."""
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    loop = env.loop(vqueue.decide, batch=1)
    script = env.graph2_script(loop, {"state": PENDING})
    (one,) = loop.tick()
    (two,) = loop.tick()
    keys = env.listed(two["sha"], {"pytest:b": ["b.txt"]})
    env.listed(one["sha"], {"pytest:a": ["a.txt"], "pytest:b": ["b.txt"]}, keys={"pytest:b": "sha512:b1"})
    env.result(keys["pytest:b"], "pytest:b", "fail")                # the candidate above fails: its base's verdict is read
    asked = Path(f"{script}.asked")
    for _ in range(2):
        asked.unlink()
        assert loop.tick() == [] and sorted(asked.read_text().splitlines()) == sorted(f"ready:{c['sha']}" for c in (one, two))


@pytest.mark.parametrize("again", ["pass", "fail", "no rerun"])
def test_a_lone_member_s_failure_its_files_dont_meet_is_rerun_once_afresh_and_the_rerun_decides(env, again):
    one = env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide)
    script = env.graph2_script(loop) if again != "no rerun" else None
    (built,) = loop.tick()
    sha = built["sha"]
    keys = env.listed(sha, {"pytest:a": ["a.txt"], "pytest:z": ["z.txt"]})
    z = keys["pytest:z"]
    env.result(keys["pytest:a"], "pytest:a", "pass")
    env.result(z, "pytest:z", "fail")
    (rerun,) = loop.tick()
    if again == "no rerun":                                         # graph2 without `rerun`: the failure stands
        assert (rerun["action"], rerun["absent"], rerun["done"], rerun["keys"]) == ("rerun", True, False, [z])
    else:
        assert (rerun["action"], rerun["keys"], rerun["counts"], rerun["done"]) == ("rerun", [z], {z: 1}, True), rerun
        assert Path(f"{script}.rerun").read_text() == f"ready:{sha} {z}\n"
        assert loop.tick() == []                                    # its fail set aside: pending until the rerun's result
        env.result(z, "pytest:z", again)
    if again == "pass":
        assert _acts(loop.tick()) == ["advance", "on_main", "drop"] and env.main() == sha and env.withdrawn == []
    else:
        eject, drop = loop.tick()
        assert (eject["action"], eject["number"], env.withdrawn, drop["outcome"]["suspects"]) == ("eject", 1, [(1, one)], [1])
    assert loop.tick() == [] or again == "pass"
    assert script is None or Path(f"{script}.rerun").read_text().count("\n") == 1


def test_a_rerun_sets_its_key_s_results_aside_for_the_candidate_above_too_as_graph2_drops_its_cached_result(env):
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide)
    script = env.graph2_script(loop, {"state": PENDING})
    (one,) = loop.tick()
    env.put(2, {"b.txt": "2\n"})
    (two,) = loop.tick()
    z = "sha512:z"                                                  # z reads nothing either member changes: one key in both
    k1 = env.listed(one["sha"], {"pytest:a": ["a.txt"], "pytest:z": ["z.txt"]}, keys={"pytest:z": z})
    k2 = env.listed(two["sha"], {"pytest:b": ["b.txt"], "pytest:z": ["z.txt"]}, keys={"pytest:z": z})
    env.result(k1["pytest:a"], "pytest:a", "pass")
    env.result(k2["pytest:b"], "pytest:b", "pass")
    env.result(z, "pytest:z", "fail")
    (rerun,) = loop.tick()
    assert (two["base"], rerun["action"], rerun["sha"], rerun["keys"]) == (one["sha"], "rerun", one["sha"], [z])
    env.result(z, "pytest:z", "pass")
    assert loop.tick() == []                                        # graph2 still counts it pending
    Path(f"{script}.verdict").write_text(json.dumps({"state": PASS, "pending": 0}))
    assert _acts(loop.tick()) == ["advance", "on_main", "drop"] * 2 and env.main() == two["sha"] and env.posts == []  # no red on z


@pytest.mark.parametrize("shadow", [False, True])
def test_a_test_graph2_says_flaked_rings_its_file_s_owner_once_a_key_and_the_shadow_only_logs_it(env, shadow):
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(lambda s: [] if s.candidates else [Build((1,), s.main, 5)], shadow=shadow)
    (built,) = loop.tick()
    sha = built["sha"]
    z = env.listed(sha, {"pytest:z": ["z.txt"]}, suites={"pytest:z": "tools/zz"})["pytest:z"]
    env.result(z, "pytest:z", "fail", tests=["tests/test_z.py::test_once"])
    env.result(z, "pytest:z", "pass")                               # what a rerun replaced it with
    env.graph2_script(loop, {"state": "pass", "check": f"ready:{sha}",
                             "flaky": [{"task": "pytest:z", "key": z, "inputs": ["z.txt"], "reads": []}]})
    loop.world.driver()._reg = {"handles": [{"handle": "zz", "owns": ["tools/zz/tests/test_z.py"]}]}
    (flaky,) = loop.tick()
    assert (flaky["action"], flaky["key"], flaky["tests"], flaky["to"], flaky["done"]) == (
        "flaky", z, ["tests/test_z.py::test_once"], ["@zz", "@ci"], not shadow), flaky
    assert "flakes is deleted, not rerun" in flaky["text"]
    assert env.posts == ([] if shadow else [f"@zz @ci vqueue: {flaky['text']}"]) and loop.tick() == []     # once a key


def test_an_infra_error_s_tree_is_graph2_s_where_vqueue_failed_has_the_field(env, monkeypatch):
    import research.vqueue_loop as L

    @dataclass(frozen=True)
    class Treed(vqueue.Failed):
        tree: bool = False

    assert getattr(L._failure("t", "k", frozenset(), tree=True), "tree", True) is True     # dropped while it has none
    monkeypatch.setattr(L, "Failed", Treed)
    monkeypatch.setattr(L, "FAILED_FIELDS", frozenset(f.name for f in fields(Treed)))
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(lambda s: [] if s.candidates else [Build((1,), s.main, 5)])
    (built,) = loop.tick()
    sha = built["sha"]
    keys = env.listed(sha, {"pytest:a": ["a.txt"], "pytest:b": ["b.txt"]})
    for task in keys:
        env.result(keys[task], task, "error", infra=True)
    env.graph2_script(loop, {"state": "infra", "infra": [{"task": "pytest:a", "key": keys["pytest:a"], "nodes": ["n1", "n2"],
                                                          "tree": True},
                                                         {"task": "pytest:b", "key": keys["pytest:b"], "nodes": ["n1"],
                                                          "tree": False}]})
    v = loop.world.verdict(sha).verdict
    assert v.state == INFRA and [(f.task, f.tree) for f in v.infra] == [("pytest:a", True), ("pytest:b", False)]


def test_a_red_is_told_once_a_base_and_task_across_a_restart(env):
    env.put(1, {"a.txt": "1\n"})

    def decide(s: Snapshot) -> list:
        if not s.candidates:
            return [Build((1,), s.main, 5)]
        c = s.candidates[0]
        return [Red(c.sha, c.base, "it fails it too", c.verdict.failed)] if c.verdict.state == FAIL else []

    loop = env.loop(decide)
    (built,) = loop.tick()
    env.listed(built["sha"], {"pytest:r": ["red.txt"]}, keys={"pytest:r": "sha512:red"})
    env.result("sha512:red", "pytest:r", "fail")
    (told,) = loop.tick()
    assert (told["action"], told["done"], len(env.posts)) == ("red", True, 1)
    (again,) = env.loop(decide, clone="again").tick()                # started again, in another clone: the log is its memory
    assert (again["action"], again["text"], again["done"], len(env.posts)) == ("red", None, False, 1)


def test_a_message_is_logged_before_its_post_so_one_cut_off_before_its_record_isnt_told_again_but_one_that_failed_is(env):
    env.put(1, {"a.txt": "1\n"})

    def decide(s: Snapshot) -> list:
        if not s.candidates:
            return [Build((1,), s.main, 5)]
        c = s.candidates[0]
        return [Red(c.sha, c.base, "it fails it too", c.verdict.failed)] if c.verdict.state == FAIL else []

    def post(answer):
        def it(text: str, to=()) -> bool:
            env.posts.append(text)
            if answer is None:
                raise RuntimeError("the loop's process died")
            return answer
        return it

    loop = env.loop(decide)
    (built,) = loop.tick()
    env.listed(built["sha"], {"pytest:r": ["red.txt"]}, keys={"pytest:r": "sha512:red"})
    env.result("sha512:red", "pytest:r", "fail")
    loop.world.post = post(False)
    (failed,) = loop.tick()
    assert (failed["done"], failed["posted"], _acts(env.lines())[-2:]) == (False, False, ["posting", "red"])
    loop.world.post = post(None)
    with pytest.raises(RuntimeError):
        loop.tick()                                                     # told again, and cut off before its record
    assert _acts(env.lines())[-1] == "posting" and len(env.posts) == 2
    loop.world.post = post(True)
    (again,) = loop.tick()
    assert (again["action"], again["text"], again["done"], len(env.posts)) == ("red", None, False, 2)


def test_a_build_pushed_with_no_line_in_the_log_is_adopted_by_the_next_build_of_the_same_members(env):
    """A tick cut off between a build's push and its line leaves a `queue/<sha12>` branch no line names, which is no candidate;
    the next build of the same members at the same heads on the same base, the same tree, takes that branch as its own."""
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy)
    w = loop.world

    def died(sha: str, priority: int) -> None:
        raise RuntimeError("the loop's process died")

    w.enqueue = died
    with pytest.raises(RuntimeError):
        loop.tick()
    (pushed,) = env.queued().values()
    then = {**os.environ, **{f"GIT_{who}_{what}": v for who in ("AUTHOR", "COMMITTER")
                             for what, v in (("NAME", "q"), ("EMAIL", "q@q"), ("DATE", "2001-01-01T00:00:00Z"))}}
    old = subprocess.run(["git", "commit-tree", f"{pushed}^{{tree}}", "-p", f"{pushed}^1", "-p", f"{pushed}^2", "-F", "-"],
                         cwd=env.o.origin, input=_git(env.o.origin, "log", "-1", "--format=%B", pushed), env=then,
                         capture_output=True, text=True, check=True).stdout.strip()     # built a second before, as it was
    _git(env.o.origin, "update-ref", f"refs/heads/queue/{old[:12]}", old)
    _git(env.o.origin, "update-ref", "-d", f"refs/heads/queue/{pushed[:12]}")
    del w.enqueue
    ignored, built = loop.tick()
    assert (ignored["action"], ignored["sha"]) == ("ignore", old)
    assert (built["action"], built["sha"], built["adopted"], built["members"]) == ("build", old, True, [1])
    assert env.queued() == {branch(old): old} and env.graph2.enqueued()[-1]["check"] == f"ready:{old}"
    assert loop.tick() == []


def _red_on_a_moved_main(env: Env, shadow: bool = False) -> tuple[Loop, str, str, str, dict[str, str]]:
    """`main` moved past every check, one PR built on it whose `pytest:a` passed and whose `pytest:r` failed at the key `main`'s
    would have: (the loop, `main`, the candidate, the PR's head, that key by task)."""
    _git(env.o.origin, "update-ref", "refs/heads/main", env.o.change({"m.txt": "m\n"}))
    main = env.main()
    one = env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide, shadow=shadow)
    (built,) = loop.tick()
    red = {"pytest:r": "sha512:red"}
    keys = env.listed(built["sha"], {"pytest:a": ["a.txt"], "pytest:r": ["red.txt"]}, keys=red)
    env.result(keys["pytest:a"], "pytest:a", "pass")
    env.result("sha512:red", "pytest:r", "fail")
    return loop, main, built["sha"], one, red


def test_a_failure_on_a_base_graph2_hasnt_listed_waits_and_the_base_is_enqueued_again_and_rung_within_the_hour_once(env):
    """Whose a failure is, the base's keys tell: on a base whose check isn't listed the candidate is pending and nobody is
    blamed, the loop enqueues `ready:<base>`, again each `BASE_RETRY_S` while it stays unlisted, and rings `CI` and
    `landing.INFRA` once, `BASE_RING_S` after it first asked for it; listed, a failure at the base's key is inherited and the candidate
    lands."""
    loop, main, sha, one, red = _red_on_a_moved_main(env)
    w = loop.world
    (base,) = loop.tick()
    assert (base["action"], base["sha"], base["candidate"], base["enqueued"], base["done"]) == ("base", main, sha, True, True)
    assert env.graph2.enqueued()[-1] == {"check": f"ready:{main}", "priority": L.DEPTH, "graph": None, "schema": GRAPH_V2}
    j = w.verdict(sha, main)
    assert (j.verdict, j.waits_on, j.note) == (Verdict(PENDING), (main, L.BaseWait(L.UNLISTED)),
                                               f"waiting for its base {main[:12]}: its check isn't listed yet")
    env.clock += L.BASE_RETRY_S - 1
    assert loop.tick() == [] and len(env.graph2.enqueued()) == 2
    env.clock += 1                                                      # `BASE_RETRY_S` after it: enqueued again
    assert _acts(loop.tick()) == ["base"] and len(env.graph2.enqueued()) == 3 and env.posts == []
    env.clock = T0 + L.BASE_RING_S                                      # still unlisted: rung
    again, ring = loop.tick()
    assert (again["action"], ring["action"], ring["sha"], ring["why"], ring["to"], ring["done"]) == (
        "base", "ring", main, f"base {main[:12]}: {L.UNLISTED}", ["@ci", "@infra"], True)
    assert f"ready:{main} holds up ready:{sha[:12]}: {L.UNLISTED}, 30 min after it was first asked for" in ring["text"]
    assert env.posts == [f"@ci @infra vqueue: {ring['text']}"]
    env.clock += L.BASE_RETRY_S
    assert _acts(loop.tick()) == ["base"] and len(env.posts) == 1      # rung once a base
    assert env.withdrawn == [] and not {"drop", "eject", "rerun"} & set(_acts(env.lines()))
    env.listed(main, {"pytest:r": ["red.txt"]}, keys=red)
    told, advance, on_main, landed = loop.tick()
    assert _acts([told, advance, on_main, landed]) == ["red", "advance", "on_main", "drop"] and env.main() == sha
    assert landed["outcome"]["inherited"][0]["base_key"] == "sha512:red" and landed["outcome"]["heads"] == [one]


@pytest.mark.parametrize("shadow", [False, True])
@pytest.mark.parametrize("why", [L.BASE_LIST_FAILED, L.BASE_UNREADABLE])
def test_a_failure_on_a_base_whose_listing_failed_or_can_t_be_read_waits_and_the_base_is_enqueued_again_and_rung_once(
        env, why, shadow):
    """A base whose listing failed, or whose graph can't be read, has no keys to say whose a failure is: the candidate is
    pending and nobody is blamed (were each failure its own, a red `main` would eject each PR built on it), the loop enqueues
    `ready:<base>` again each `BASE_RETRY_S`, and rings `CI` and `landing.INFRA` at once, naming it, once a base and why (with
    --shadow, logged only).  Listed again, a failure at the base's key is inherited and the candidate lands."""
    loop, main, sha, one, red = _red_on_a_moved_main(env, shadow)
    w = loop.world
    if why == L.BASE_LIST_FAILED:
        w.graph2_verdict = lambda s: {"state": LIST_FAILED, "why": "no graph"} if s == main else None
    else:
        env.graph2.list_graph(f"ready:{main}", {"schema": GRAPH_V2, "commit": "0" * 40, "tasks": []})
    base, ring = loop.tick()
    assert (base["action"], base["sha"], base["candidate"], base["enqueued"]) == ("base", main, sha, True)
    assert (ring["action"], ring["sha"], ring["why"], ring["to"], ring["done"], ring["shadow"]) == (
        "ring", main, f"base {main[:12]}: {why}", ["@ci", "@infra"], not shadow, shadow)
    assert f"the verified queue's base ready:{main} holds up ready:{sha[:12]}: {why} (" in ring["text"]
    assert env.posts == ([] if shadow else [f"@ci @infra vqueue: {ring['text']}"])
    j = w.verdict(sha, main)
    assert (j.verdict, j.waits_on[0], j.waits_on[1].why) == (Verdict(PENDING), main, why)
    assert j.note == f"waiting for its base {main[:12]}: {j.waits_on[1]}" and j.waits_on[1].detail
    env.clock += L.BASE_RETRY_S - 1
    assert loop.tick() == []
    env.clock += 1                                                      # `BASE_RETRY_S` after it: enqueued again, not rung
    assert _acts(loop.tick()) == ["base"] and [e["check"] for e in env.graph2.enqueued()][1:] == [f"ready:{main}"] * 2
    assert env.withdrawn == [] and not {"drop", "eject", "rerun"} & set(_acts(env.lines()))
    assert env.posts == ([] if shadow else [f"@ci @infra vqueue: {ring['text']}"])
    if why == L.BASE_LIST_FAILED:
        del w.graph2_verdict
    env.listed(main, {"pytest:r": ["red.txt"]}, keys=red)              # listed again
    told, advance, *on_main, landed = loop.tick()
    assert _acts([told, advance, *on_main, landed]) == ["red", "advance", *([] if shadow else ["on_main"]), "drop"]
    assert landed["outcome"]["inherited"][0]["base_key"] == "sha512:red" and landed["outcome"]["heads"] == [one]


def test_a_failure_on_a_listed_base_is_judged_on_its_keys_and_the_base_is_neither_enqueued_nor_rung(env):
    """A base graph2 listed tells whose a failure is as before: one its keys don't have is the candidate's own, and its PR is
    ejected, with no line of the base's and no ring."""
    main = env.main()                                                   # listed with no task: a failure is its candidate's own
    one = env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy)
    (built,) = loop.tick()
    sha, w = built["sha"], loop.world
    keys = env.listed(sha, {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "fail")
    j = w.verdict(sha, main)
    assert (j.verdict.state, j.waits_on, j.note) == (FAIL, None, "")
    eject, drop = loop.tick()
    assert _acts([eject, drop]) == ["eject", "drop"] and env.withdrawn == [(1, one)]
    assert not {"base", "ring"} & set(_acts(env.lines())) and not any(p.startswith("@ci @infra") for p in env.posts)


def test_a_base_s_check_was_first_asked_for_by_its_build_or_as_a_base_and_last_enqueued_by_either():
    past = L.Past(lines=[{"action": "build", "sha": "b", "at": L.stamp(T0), "enqueued": True},
                         {"action": "base", "sha": "b", "at": L.stamp(T0 + 60), "enqueued": False},
                         {"action": "base", "sha": "x", "at": L.stamp(T0 + 120), "enqueued": True},
                         {"action": "retry", "sha": "b", "at": L.stamp(T0 + 180), "enqueued": True},
                         {"action": "base", "sha": "b", "at": L.stamp(T0 + 240), "enqueued": False}])
    assert (past.base_enqueues("b"), past.base_enqueues("x"), past.base_enqueues("y")) == (
        (T0, T0 + 180), (T0 + 120, T0 + 120), (None, None))
    assert L.Past(lines=[{"action": "base", "sha": "b", "at": L.stamp(T0), "enqueued": False}]).base_enqueues("b") == (T0, None)


def test_a_failure_waits_while_its_task_s_key_at_the_base_waits_on_what_it_reads_there_unless_that_ended_in_infra(env):
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(lambda s: [] if s.candidates else [Build((1,), s.main, 5)])
    (built,) = loop.tick()
    sha, main, w = built["sha"], built["base"], loop.world
    keys = env.listed(sha, {"pytest:r": ["red.txt"]})
    env.result(keys["pytest:r"], "pytest:r", "fail")
    env.listed(main, {"lean-build:x": ["x.lean"], "pytest:r": ["red.txt"]}, reads={"pytest:r": "lean-build:x"},
               keys={"lean-build:x": "sha512:lx"})
    j = w.verdict(sha, main)
    assert (j.verdict, j.waits_on, j.note) == (Verdict(PENDING), None, f"waiting for its base {main[:12]}: pytest:r there "
                                                                        "still wait on what they read")
    env.result("sha512:lx", "lean-build:x", "error", infra=True)       # what it reads there may never pass: no key
    v = w.verdict(sha, main).verdict
    assert v.state == FAIL and [(f.task, f.base_key) for f in v.failed] == [("pytest:r", None)]
    env.result("sha512:lx", "lean-build:x", "pass")
    j = w.verdict(sha, main)                                         # its key there is made: it waits for its result
    assert (j.verdict, j.note) == (Verdict(PENDING), f"waiting for its base {main[:12]}: pytest:r there haven't ended")
    _, base_keys, _ = w._settled(w.tasks.graph_v2(f"ready:{main}")[1])
    env.result(base_keys["pytest:r"], "pytest:r", "pass")
    (f,) = w.verdict(sha, main).verdict.failed
    assert f.base_key == base_keys["pytest:r"] != f.key and not f.base_tests and not vqueue._inherited(f)


def test_a_failure_waits_for_its_task_s_result_at_the_base_whatever_else_there_ended_in_infra(env):
    """A failed task whose task of the same id at the base, keyed apart by the change, has no result yet leaves its candidate
    pending, so a fault `main` has too is read beside the base's before any member is blamed: an infra error elsewhere in
    the base's check leaves the wait as it is, one in the task there ends it."""
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide)
    (built,) = loop.tick()
    sha, main, w = built["sha"], built["base"], loop.world
    env.listed(main, {"pytest:t": ["t.py"], "pytest:u": ["u.py"]}, keys={"pytest:t": "sha512:base-t", "pytest:u": "sha512:base-u"})
    keys = env.listed(sha, {"pytest:a": ["a.txt"], "pytest:t": ["t.py"], "pytest:u": ["u.py"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    env.result(keys["pytest:u"], "pytest:u", "pass")
    env.result(keys["pytest:t"], "pytest:t", "fail", tests=["test_t.py::a"])
    env.result("sha512:base-u", "pytest:u", "error", infra=True)
    j = w.verdict(sha, main)
    assert (j.verdict, j.note) == (Verdict(PENDING), f"waiting for its base {main[:12]}: pytest:t there haven't ended")
    assert not {"drop", "eject", "rerun"} & set(_acts(loop.tick()))
    env.result("sha512:base-t", "pytest:t", "error", infra=True)       # main's may never end: the wait ends
    v = w.verdict(sha, main).verdict
    assert v.state == FAIL and [(f.task, f.base_tests) for f in v.failed] == [("pytest:t", None)]
    env.result("sha512:base-t", "pytest:t", "fail", tests=["test_t.py::a"])
    (f,) = w.verdict(sha, main).verdict.failed
    assert f.base_tests == {"test_t.py::a"} and vqueue._inherited(f)


def test_a_graph_lacking_a_job_main_waits_for_is_list_failed_marked_with_it_so_it_is_retried_and_rung(env):
    """drive's `executor = "on"` path promotes nothing whose graph lacks a job `main` waits for (`landing.gating`): a candidate
    whose every task passed is list-failed then, its mark naming the jobs, and the policy retries and rings it."""
    env.gates = ["pytest", "lean-audit"]
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy)
    (built,) = loop.tick()
    keys = env.listed(sha := built["sha"], {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    j = loop.world.verdict(sha)
    assert (j.verdict.state, j.mark, j.lacks, j.note) == (LIST_FAILED, "lacks lean-audit", ("lean-audit",),
                                                          "no task of lean-audit, which `main` waits for")
    acts = [r for _ in range(3) for r in loop.tick()]
    assert _acts(acts) == ["retry", "ring"] and acts[0]["mark"] == "lacks lean-audit" and env.main() != sha   # rung once
    env.gates = ["pytest"]
    assert loop.world.verdict(sha).verdict.state == PASS


def _graph(sha: str, tasks: list[dict]) -> dict:
    """A graph/v2 of ``sha`` holding ``tasks`` (each its `id`, `kind` and `inputs`, maybe its `key`, `reads` and `outputs`), in
    the env `Env.listed` lists its tasks in."""
    files = [*SPEC, *sorted({f for t in tasks for f in t["inputs"]})]
    return {"schema": GRAPH_V2, "commit": sha, "files": {f: f"sha512:{f}" for f in files},
            "envs": {"py": {"digest": "sha512:env", "spec": list(SPEC)}},
            "tasks": [{"env": "py", "argv": ["t", t["id"]], "vars": {}, "reads": [],
                       "outputs": [{"name": "out", "path": f"{t['id']}.out"}], **t} for t in tasks]}


def test_a_kind_of_task_its_base_lists_and_it_lists_none_of_fails_its_listing_but_a_fail_of_its_own_still_fails(env):
    """A family whose lister is gone lists no task and no error (`verity_graph.graph.listing`), and its job may still be
    another family's (lean-lock's and lean-module's are lean-audit): a candidate whose graph lists no task of a kind on the
    merge gate its base's graph lists is list-failed, marked `lacks kind K`, so the policy retries and rings it, a failure
    it inherits included.  One of its own fails it, so its PR is ejected all the same."""
    env.gates = ["pytest", "lean-audit"]
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy)
    (built,) = loop.tick()
    sha, main = built["sha"], built["base"]
    lean = [{"id": "lean-module:M", "kind": "lean-module", "inputs": ["M.lean"], "key": "sha512:m"},
            {"id": "lean-lock:L", "kind": "lean-lock", "inputs": ["L.json"], "key": "sha512:l"}]

    def listed(at: str, key: str, *more: dict) -> None:
        env.graph2.list_graph(f"ready:{at}", _graph(at, [{"id": "pytest:a", "kind": "test-file", "inputs": ["a.txt"],
                                                          "key": key}, *more]))

    listed(main, "sha512:a0", *lean)
    listed(sha, "sha512:a1", lean[0])                   # as with families/lean_lock.py deleted
    env.result("sha512:a1", "pytest:a", PASS)
    env.result("sha512:m", "lean-module:M", PASS)
    j = loop.world.verdict(sha, main)
    assert (j.verdict.state, j.mark, j.lacks, j.note) == (LIST_FAILED, "lacks kind lean-lock", ("kind lean-lock",),
                                                          "no task of kind lean-lock, which its base lists")
    acts = [r for _ in range(3) for r in loop.tick()]
    assert _acts(acts) == ["retry", "ring"] and acts[0]["mark"] == "lacks kind lean-lock" and env.main() != sha
    env.result("sha512:a0", "pytest:a", FAIL)
    listed(sha, "sha512:a0", lean[0])                   # `main`'s fail, inherited
    assert loop.world.verdict(sha, main).verdict.state == LIST_FAILED
    listed(sha, "sha512:a2", lean[0])
    env.result("sha512:a2", "pytest:a", FAIL)
    v = loop.world.verdict(sha, main).verdict
    assert (v.state, [f.task for f in v.failed]) == (FAIL, ["pytest:a"])


def test_the_after_lean_suites_test_files_reading_lean_build_s_outputs_stand_for_lean_suites_whose_kind_may_go(env):
    """Listed as test-file tasks that read lean-build's outputs (its `build` among them), the test files of the suites check
    runs after lean-build stand for lean-suites (`_job`): the graph lacks no job `main` waits for, and the lean-suites tasks
    its base lists may go.  A candidate whose base's check isn't listed waits for it."""
    env.gates = ["pytest", "lean-build", "lean-suites"]
    lean, exe = "verity/protocols/flock/verifier", {"name": "flock-verify", "path": "verity/protocols/flock/verifier/.lake/build/bin/v"}
    build, outs = f"lean-build:{lean}", [exe, {"name": "build", "path": f"{lean}/.lake/build.tar.gz"}]
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy)
    (built,) = loop.tick()
    sha, main, w = built["sha"], built["base"], loop.world
    a = {"id": "pytest:a", "kind": "test-file", "inputs": ["a.txt"], "key": "sha512:a"}
    env.graph2.list_graph(f"ready:{main}", _graph(main, [
        a, {"id": build, "kind": "lean-build", "inputs": [f"{lean}/Main.lean"], "key": "sha512:b0", "outputs": [exe]},
        {"id": "lean-suites:verity-flock", "kind": "lean-suites", "inputs": ["f/test_f.py"],
         "reads": [{"task": build, "output": "flock-verify"}]}]))
    env.graph2.list_graph(f"ready:{sha}", _graph(sha, [
        a, {"id": build, "kind": "lean-build", "inputs": [f"{lean}/Main.lean"], "key": "sha512:b1", "outputs": outs},
        {"id": "test-file:f/test_f.py", "kind": "test-file", "inputs": ["f/test_f.py"],
         "reads": [{"task": build, "output": o["name"]} for o in outs]}]))
    env.result("sha512:a", "pytest:a", PASS)
    env.graph2.finish_result({"key": "sha512:b1", "task": build, "status": PASS,
                              "outputs": {"flock-verify": "sha512:v", "build": "sha512:t"}})
    _, made, _ = w._settled(w.tasks.graph_v2(f"ready:{sha}")[1])
    env.result(made["test-file:f/test_f.py"], "test-file:f/test_f.py", PASS)
    j = w.verdict(sha, main)
    assert (j.verdict.state, j.lacks, j.mark, j.note) == (PASS, (), None, "")
    j = w.verdict(sha, "f" * 40)
    assert (j.verdict.state, j.waits_on, j.note) == (PENDING, ("f" * 40, L.BaseWait(L.UNLISTED)),
                                                     f"waiting for its base {'f' * 12}: {L.UNLISTED}")


def test_graph2_s_lean_agreement_task_gates_and_a_ratchet_gate_task_doesn_t(env):
    """graph2's lean-agreement task is on the merge gate, as it is in the run beside the check; ratchet-gate is off it: it
    judges each merge against its first parent, which a task keyed by its inputs can't."""
    one = env.put(1, {"v/check.py": "1\n"})
    loop = env.loop(policy)
    (built,) = loop.tick()
    sha = built["sha"]

    def listed(agree: str) -> None:
        task = {"env": "py", "argv": ["t"], "vars": {}, "reads": [], "outputs": []}
        env.graph2.list_graph(f"ready:{sha}", {
            "schema": GRAPH_V2, "commit": sha, "envs": {"py": {"digest": "sha512:env"}},
            "files": {"v/check.py": "sha512:v", "t/test_a.py": "sha512:t"},
            "tasks": [{"id": "test-file:t/test_a.py", "kind": "test-file", "inputs": ["t/test_a.py"], "key": "sha512:a", **task},
                      {"id": "lean-agreement:v", "kind": "lean-agreement", "inputs": ["v/check.py"], "key": agree, **task},
                      {"id": "ratchet-gate:v", "kind": "ratchet-gate", "inputs": ["v/check.py"], "key": "sha512:r", **task}]})

    listed("sha512:agree")
    env.result("sha512:a", "test-file:t/test_a.py", PASS)
    env.result("sha512:r", "ratchet-gate:v", FAIL)
    env.result("sha512:agree", "lean-agreement:v", PASS)
    assert loop.world.verdict(sha, built["base"]).verdict.state == PASS
    listed("sha512:agree2")
    env.result("sha512:agree2", "lean-agreement:v", FAIL)
    assert [f.task for f in loop.world.verdict(sha, built["base"]).verdict.failed] == ["lean-agreement:v"]
    assert _acts(loop.tick()) == ["eject", "drop"] and env.withdrawn == [(1, one)] and env.main() != sha


ALLOWED = '''schema = "policy/v0"
[[allow]]
id = "suite zz"
owner = "ci"
expires = "2030-06-01T00:00:00Z"
reason = "red until its fix lands"
'''


@pytest.mark.parametrize("own", [False, True])
def test_a_failure_an_allowance_of_the_candidate_s_base_covers_is_passed_as_drive_reads_it_until_it_expires(env, own):
    """drive's `executor = "on"` path takes a failed task whose every red record an unexpired allowance covers as passed
    (`Landing._task_units`, `Landing._task_allowed`), and so does the verdict, by policy.toml at the candidate's base: an
    allowance the candidate adds itself (``own``) never covers its own failure."""
    if not own:
        _main_to(env, {"tools/verity/check/policy.toml": ALLOWED})
    env.put(1, {"a.txt": "1\n", **({"tools/verity/check/policy.toml": ALLOWED} if own else {})})
    loop = env.loop(vqueue.decide)
    (built,) = loop.tick()
    sha, w = built["sha"], loop.world
    keys = env.listed(sha, {"pytest:a": ["a.txt"], "pytest:z": ["z.txt"]}, suites={"pytest:z": "zz"})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    env.result(keys["pytest:z"], "pytest:z", "fail", tests=["test_z.py::test_one"])
    j = w.verdict(sha, built["base"])
    if own:
        assert j.verdict.state == FAIL and [f.task for f in j.verdict.failed] == ["pytest:z"] and j.note != "allowed: pytest:z"
        return
    assert (j.verdict, j.note) == (Verdict(PASS), "allowed: pytest:z")
    env.clock = T0 + 365 * 86400                                       # past its expiry: red again
    assert [f.task for f in w.verdict(sha, built["base"]).verdict.failed] == ["pytest:z"]
    env.clock = T0
    assert _acts(loop.tick()) == ["advance", "on_main", "drop"] and env.main() == sha


def test_an_advance_is_logged_once_pushed_and_drive_s_steps_after_it_are_a_step_of_their_own_run_again_until_done(env):
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy)
    (built,) = loop.tick()
    keys = env.listed(sha := built["sha"], {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    calls = []

    def on_main(sha: str, base: str, members, told: list[str]) -> None:
        calls.append((sha, base, [m.pr for m in members], list(told)))
        told.append("stacked:9<1")
        if len(calls) == 1:
            raise Q.QueueError("GitHub answered 502")

    loop.world.on_main = on_main
    advance, failed = loop.tick()
    assert (advance["action"], advance["done"], advance["on_main"], env.main()) == ("advance", True, True, sha)
    assert (failed["action"], failed["done"], failed["told"]) == ("on_main", False, ["stacked:9<1"]) and "502" in failed["error"]
    assert _acts(env.lines())[-2:] == ["advance", "on_main"]
    again, drop = loop.tick()                                           # run again before the tick's actions
    assert (again["action"], again["done"], drop["action"], drop["why"]) == ("on_main", True, "drop", "on main")
    assert calls == [(sha, built["base"], [1], []), (sha, built["base"], [1], ["stacked:9<1"])]   # who was told, kept
    assert loop.tick() == [] and len(calls) == 2


def test_a_tick_advancing_through_stacked_passed_candidates_pushes_each_with_its_lease_and_runs_its_steps_once(env):
    """Each candidate passed on the one below lands in the tick its base does, in order: a push of `main` leased on its base,
    then drive's steps on its members from that base, then its drop.  A push refused part-way (here a stale lease) stops the
    tick with the candidates below it landed whole and nothing of it: the next tick lands it, its steps run once."""
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide)
    env.graph2_script(loop, {"state": PASS, "pending": 0})
    (one,) = loop.tick()
    env.put(2, {"b.txt": "2\n"})
    (two,) = loop.tick()
    env.put(3, {"c.txt": "3\n"})
    (three,) = loop.tick()
    assert (two["base"], three["base"]) == (one["sha"], two["sha"])
    for c, task, path in ((one, "pytest:a", "a.txt"), (two, "pytest:b", "b.txt"), (three, "pytest:c", "c.txt")):
        env.result(env.listed(c["sha"], {task: [path]})[task], task, "pass")
    calls, pushes, advance = [], [], loop.world.advance

    def on_main(sha: str, base: str, members, told: list[str]) -> None:
        calls.append((sha, base, [m.pr for m in members]))

    def stale(sha: str, base: str) -> None:
        pushes.append(sha)
        advance(sha, "f" * 40 if pushes.count(three["sha"]) == 1 else base)  # three's first push leases a stale `main`

    loop.world.on_main, loop.world.advance = on_main, stale
    recs = loop.tick()
    assert _acts(recs) == ["advance", "on_main", "drop", "advance", "on_main", "drop", "advance"], recs
    assert "stale info" in recs[-1]["error"] and env.main() == two["sha"] and env.queued() == {branch(three["sha"]): three["sha"]}
    assert [(r["sha"], r["base"], r.get("done")) for r in recs if r["action"] == "advance"] == [
        (one["sha"], main, True), (two["sha"], one["sha"], True), (three["sha"], None, None)]
    assert calls == [(one["sha"], main, [1]), (two["sha"], one["sha"], [2])]
    assert _acts(loop.tick()) == ["advance", "on_main", "drop"] and env.main() == three["sha"]
    assert calls[2:] == [(three["sha"], two["sha"], [3])] and pushes == [one["sha"], two["sha"], three["sha"], three["sha"]]
    assert loop.tick() == [] and len(calls) == 3 and env.queued() == {}


def test_an_outcome_s_inherited_failures_keep_their_tests_and_their_base_s_through_the_log():
    for f in (vqueue.Failed("pytest:r", "sha512:k", frozenset({"r.txt"}), "sha512:b", frozenset({"t.py::x"}),
                            frozenset({"t.py::x", "t.py::y"})),
              vqueue.Failed("pytest:r", "sha512:k", frozenset({"r.txt"}), None, frozenset({"t.py::x"}), frozenset()),
              vqueue.Failed("pytest:r", "sha512:k", frozenset({"r.txt"}))):
        assert L._as_failed(json.loads(json.dumps(L._failed_doc(f)))) == f


def test_no_slim_runs_only_with_shadow_or_dry_run(capsys):
    """Without the run beside each check, `main` would move past lean-agreement and the preflights unchecked."""
    assert L.main(["--no-slim", "--once"]) == 1
    assert "only with --shadow or --dry-run" in capsys.readouterr().err


class Slim:
    """`research.vqueue_slim`'s functions, faked: each start registers a run (one on `main` itself, ``main-N``, kept in
    ``below``; a candidate's, ``run-N``, in ``started``), judged as ``verdicts`` says (pending unless it names it), and a
    verdict other than pending is kept in its record, as `judge` keeps it: with the steps ``steps`` says it failed and passed,
    as `judge` keeps them, each failed step a `Failed` whose inputs are ``inputs``.  Each record judged is kept in ``judged``.
    With ``launching``, a candidate's start is a detached launch, pending until ``registered`` names the run ids it printed
    for its sha (none: it registered no run)."""

    def __init__(self):
        self.started: list[tuple] = []
        self.below: list[tuple] = []
        self.verdicts: dict[str, str] = {}
        self.steps: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        self.inputs = frozenset({"tests/**"})
        self.judged: list = []
        self.launching, self.registered = False, {}

    def start(self, driver, sha: str, base: str) -> dict:
        if sha == base:
            self.below.append((driver, sha))
            return {"ids": [f"main-{len(self.below)}"], "tip": sha, "base": base}
        self.started.append((driver, sha, base))
        if self.launching:
            return {"ids": [], "launch": {"dir": f"train-{len(self.started)}", "pid": 1}, "tip": sha, "base": base}
        return {"ids": [f"run-{len(self.started)}"], "tip": sha, "base": base}

    def judge(self, driver, rec: dict | None) -> Verdict:
        self.judged.append(rec)
        if rec is None:
            return Verdict(INFRA, (), (vqueue.Failed("the run beside its check registered no run", ":no run", frozenset()),))
        if "launch" in rec:
            if rec["tip"] not in self.registered:
                return Verdict(PENDING)
            del rec["launch"]
            if not (ids := self.registered[rec["tip"]]):
                rec["verdict"] = INFRA
                return Verdict(INFRA, (), (vqueue.Failed("registered no run", f"{rec['tip']}:no run", frozenset()),))
            rec["ids"] = ids
        kept, run = rec.get("verdict"), rec["ids"][0]
        state = (kept["state"] if isinstance(kept, dict) else kept) or self.verdicts.get(run, PENDING)
        failed, passed = self.steps.get(run, ((), ()))
        fails = tuple(vqueue.Failed(f"step {s}", f"{rec['tip']}:{s}", self.inputs, f"{rec['base']}:{s}", frozenset({f"step:{s}"}))
                      for s in failed) if state == FAIL else ()
        if state != PENDING:
            rec["verdict"] = state if run not in self.steps else {"state": state, "failed": [{"task": f.task} for f in fails],
                                                                  "passed": list(passed)}
        down = (vqueue.Failed(f"the run beside its check ({rec['ids'][0]}) was lost", f"{rec['tip']}:{rec['ids'][0]}",
                              frozenset()),)
        return Verdict(state, fails, down if state == INFRA else (), ended=bool(fails))

    def end(self, run: str, *failed: str) -> None:
        """``run`` ended: failing the steps ``failed`` names, passing every other preflight."""
        self.verdicts[run] = FAIL if failed else PASS
        self.steps[run] = (failed, tuple(s for s in PREFLIGHT if s not in failed))

    @staticmethod
    def combine(graph: Verdict, slim: Verdict) -> Verdict:
        state = next(s for s in (FAIL, INFRA, PENDING, LIST_FAILED, PASS) if s in (graph.state, slim.state))
        return Verdict(state, graph.failed + slim.failed, graph.infra + slim.infra)

    based = staticmethod(vqueue_slim.based)


def test_the_run_beside_a_check_starts_with_its_build_its_record_is_kept_and_both_must_pass(env):
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    slim = Slim()
    loop = env.loop(vqueue.decide, slim=slim)
    loop.world.slim_driver = lambda: "drive"
    below, built, started = loop.tick()
    sha = built["sha"]
    assert (below["action"], below["sha"], below["why"], below["record"]) == (
        "slim", main, "started beside `main`", {"ids": ["main-1"], "tip": main, "base": main})
    assert (started["action"], started["sha"], started["why"], started["record"]) == (
        "slim", sha, "started", {"ids": ["run-1"], "tip": sha, "base": main})
    keys = env.listed(sha, {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    assert loop.tick() == [] and slim.started == [("drive", sha, main)]      # graph2's passed, the run beside still runs
    slim.verdicts["run-1"] = INFRA
    judged, retry, again = loop.tick()
    assert (judged["action"], judged["record"]["verdict"], retry["action"]) == ("slim", INFRA, "retry")
    assert (again["action"], again["why"], again["record"]["ids"]) == ("slim", "started again", ["run-2"])
    assert loop.tick() == []
    slim.verdicts["run-2"] = PASS
    kept, advance, on_main, drop = loop.tick()
    assert (kept["record"]["verdict"], advance["action"], on_main["action"], drop["action"]) == (
        PASS, "advance", "on_main", "drop") and env.main() == sha
    assert len(slim.started) == 2 and loop.tick() == []             # `main`'s run beside is the candidate's
    started_again = env.loop(vqueue.decide, clone="again", slim=slim)         # a restart reads its records from the log
    started_again.world.slim_driver = lambda: "drive"
    assert started_again.tick() == [] and len(slim.started) == 2 and slim.below == [("drive", main)]


def test_the_run_beside_starts_with_each_candidate_s_build_up_to_depth_once_across_a_restart_and_a_dry_run_starts_none(env):
    main = env.main()
    for n in range(1, 5):
        env.put(n, {f"{n}.txt": f"{n}\n"})
    slim = Slim()
    dry = env.loop(vqueue.decide, slim=slim, batch=1, depth=3, dry_run=True)    # its world has no Landing: a start raises
    assert dry.tick() == [] and slim.started == [] and any(x.startswith("would build") for x in env.out)
    loop = env.loop(vqueue.decide, slim=slim, batch=1, depth=3)
    loop.world.slim_driver = lambda: "drive"
    built = []
    for i in range(3):                                              # each with its check pending, the one below's too
        *below, build, started = loop.tick()
        assert (build["action"], started["action"], started["sha"], started["why"]) == ("build", "slim", build["sha"],
                                                                                         "started")
        assert [(b["sha"], b["why"]) for b in below] == ([(main, "started beside `main`")] if i == 0 else [])
        built.append(build)
    assert [b["base"] for b in built] == [main, built[0]["sha"], built[1]["sha"]]
    assert loop.tick() == [] and slim.started == [("drive", b["sha"], b["base"]) for b in built]     # three at depth 3
    again = env.loop(vqueue.decide, clone="again", slim=slim, batch=1, depth=3)   # a restart reads their records from the log
    again.world.slim_driver = lambda: "drive"
    assert again.tick() == [] and len(slim.started) == 3 and slim.below == [("drive", main)]


def test_a_candidate_on_main_s_chain_the_log_holds_no_start_of_starts_its_run_beside_once_and_one_off_it_none(env):
    """A build whose tick stopped before the start, or one by a loop without slim, leaves a candidate with no run beside its
    check: one on `main`'s chain starts it once, and one off it, which the policy drops, none."""
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    slim = Slim()
    plain, loop = env.loop(vqueue.decide, batch=1), env.loop(vqueue.decide, slim=slim, batch=1)
    loop.world.slim_driver = lambda: "drive"
    (old,) = plain.tick()
    _git(env.o.origin, "update-ref", "refs/heads/main", env.o.change({"m.txt": "m\n"}))     # `main` moved past its base
    main = env.main()
    below, drop, one, started = loop.tick()
    assert (drop["action"], drop["sha"], one["action"], one["base"], started["sha"]) == ("drop", old["sha"], "build", main,
                                                                                           one["sha"])
    assert slim.started == [("drive", one["sha"], main)] and env.cancelled == [] and below["sha"] == main
    (two,) = plain.tick()                                           # on `one`, with no run beside its check
    assert two["base"] == one["sha"]
    judged = len(slim.judged)
    dry = env.loop(vqueue.decide, slim=slim, batch=1, dry_run=True)        # its world has no Landing: a start or judge raises
    assert dry.tick() == [] and len(slim.started) == 1 and len(slim.judged) == judged
    assert "beside: pending" in "\n".join(env.out)
    (started,) = loop.tick()
    assert (started["action"], started["sha"], started["why"]) == ("slim", two["sha"], "started")
    assert slim.started[-1] == ("drive", two["sha"], one["sha"]) and loop.tick() == [] and len(slim.started) == 2


def test_a_failed_candidate_s_run_beside_and_those_of_the_candidates_above_it_are_cancelled_with_their_drops(env):
    one_head = env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    slim = Slim()
    loop = env.loop(vqueue.decide, slim=slim, batch=1)
    loop.world.slim_driver = lambda: "drive"
    _, one, _ = loop.tick()
    two, _ = loop.tick()
    keys = env.listed(one["sha"], {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "fail")                # #1's own: ejected, and the candidate above dropped
    eject, drop1, drop2, *rebuilt = loop.tick()
    assert (eject["action"], eject["number"], env.withdrawn) == ("eject", 1, [(1, one_head)])
    assert [(d["action"], d["sha"], d["beside_cancelled"]) for d in (drop1, drop2)] == [("drop", one["sha"], ["run-1"]),
                                                                                        ("drop", two["sha"], ["run-2"])]
    assert [ids for ids, _ in env.cancelled] == [["run-1"], ["run-2"]]
    assert _acts(rebuilt) == ["build", "slim"] and rebuilt[0]["members"] == [2] and rebuilt[1]["record"]["ids"] == ["run-3"]


@pytest.mark.parametrize("registered", [["run-9"], []])
def test_a_run_beside_still_launching_when_its_candidate_is_dropped_is_cancelled_once_it_printed_its_run_ids(env, registered):
    """train.sh takes many minutes to register a run, so a candidate dropped soon after its build has a run beside it with no
    ids to cancel: the drop logs it, and each tick judges it until its launch ends; one that registered no run is settled."""
    env.put(1, {"a.txt": "1\n"})
    slim = Slim()
    slim.launching = True
    loop = env.loop(vqueue.decide, slim=slim)
    loop.world.slim_driver = lambda: "drive"
    _, built, started = loop.tick()
    sha = built["sha"]
    env.labels.held.add(1)                                          # held by a reviewer: dropped
    (drop,) = loop.tick()
    assert (drop["action"], drop["beside_launching"], env.cancelled) == ("drop", started["record"], [])
    assert "beside_cancelled" not in drop and loop.tick() == []     # still launching
    slim.registered[sha] = registered
    (settled,) = loop.tick()
    assert (settled["action"], settled["sha"], settled["dropped"], settled["record"]["ids"]) == ("slim", sha, True, registered)
    assert "launch" not in settled["record"]
    held = f"vqueue: candidate {sha[:12]} was dropped: #1 isn't taken now: held by reviewer (no reason)"
    assert env.cancelled == ([(registered, held)] if registered else [])
    assert settled.get("beside_cancelled") == (registered or None) and loop.tick() == []
    again = env.loop(vqueue.decide, clone="again", slim=slim)
    again.world.slim_driver = lambda: "drive"
    assert again.tick() == [] and len(slim.started) == 1                                          # settled, once


def test_a_dropped_candidate_s_run_beside_its_check_is_cancelled_while_it_runs_and_one_that_ended_is_left(env):
    env.put(1, {"a.txt": "1\n"})
    slim = Slim()
    loop = env.loop(vqueue.decide, slim=slim)
    loop.world.slim_driver = lambda: "drive"
    _, built, started = loop.tick()
    env.labels.held.add(1)                                          # held by a reviewer: dropped
    (drop,) = loop.tick()
    assert (started["action"], drop["action"], drop["beside_cancelled"]) == ("slim", "drop", ["run-1"])
    assert env.cancelled == [(["run-1"], f"vqueue: candidate {built['sha'][:12]} was dropped: #1 isn't taken now: held by "
                                         "reviewer (no reason)")]
    env.put(2, {"b.txt": "2\n"})
    two, _ = loop.tick()
    keys = env.listed(two["sha"], {"pytest:b": ["b.txt"]})
    env.result(keys["pytest:b"], "pytest:b", "pass")
    assert loop.tick() == []
    slim.verdicts["run-2"] = FAIL
    env.labels.held.add(2)
    judged, drop = loop.tick()
    assert (judged["record"]["verdict"], drop["action"], len(env.cancelled)) == (FAIL, "drop", 1) and "beside_cancelled" not in drop


def test_a_candidate_older_than_the_log_s_window_keeps_its_tries_first_enqueue_results_set_aside_and_run_beside(env):
    env.put(1, {"a.txt": "1\n"})
    slim, seen = Slim(), []
    loop = env.loop(lambda s: seen.append(s) or vqueue.decide(s), slim=slim, pending_s=200 * 3600)
    loop.world.slim_driver = lambda: "drive"
    _, built, started = loop.tick()
    keys = env.listed(sha := built["sha"], {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "error", infra=True)
    (retry,) = loop.tick()                                          # the run beside still runs: not started again
    assert (retry["action"], retry["counts"], retry["priority"]) == ("retry", {keys["pytest:a"]: 1}, built["priority"])
    env.clock += 49 * 3600                                          # its build, the run beside it and its retry: past the window
    assert loop.tick() == [] and len(slim.started) == 1 and slim.judged[-1] == started["record"]
    c = seen[-1].candidates[0]
    assert (c.sha, c.verdict.state, c.tries, c.enqueued) == (sha, PENDING, 2, T0)
    env.result(keys["pytest:a"], "pytest:a", "pass")
    assert loop.tick() == [] and len(slim.started) == 1
    c = seen[-1].candidates[0]
    assert (c.verdict.state, c.tries, c.enqueued) == (PENDING, 2, T0)


def test_a_rerun_graph2_refuses_enqueues_the_check_again_and_the_next_tick_reruns_it(env):
    """graph2 prunes a check an hour after its verdict once nothing of it runs, and refuses a rerun of it until an enqueue
    lists it again: the refused rerun enqueues it, logged not done, and the next tick's rerun goes ahead."""
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(lambda s: [Build((1,), s.main, 5)] if not s.candidates else
                    [Rerun(c.sha, ("sha512:k",)) for c in s.candidates if not loop.world.reran])
    loop.world.reran = False
    (built,) = loop.tick()
    env.listed(built["sha"], {"pytest:a": ["a.txt"]}, keys={"pytest:a": "sha512:k"})
    env.graph2_script(loop)
    real, calls = loop.world.rerun, []

    def refused_once(sha: str, keys) -> str | None:
        calls.append(sha)
        if len(calls) == 1:
            raise Q.QueueError(f"graph2 rerun ready:{sha[:12]} exited 1: the check was pruned")
        loop.world.reran = True
        return real(sha, keys)

    loop.world.rerun = refused_once
    (refused,) = loop.tick()
    assert (refused["action"], refused["done"]) == ("rerun", False) and "enqueued it again" in refused["why"], refused
    assert [e["check"] for e in env.graph2.enqueued()] == [f"ready:{built['sha']}"] * 2
    (rerun,) = loop.tick()
    assert (rerun["action"], rerun["done"], len(calls)) == ("rerun", True, 2)


def _rings_owner_of_one(s: vqueue.Snapshot) -> list:
    """Build #1 and #2 alone on `main`, then ring #1's owner while its check reads infra."""
    if not s.candidates:
        return [Build((1,), s.main, 5), Build((2,), s.main, 5)]
    return [Ring(c.sha, "its check keeps erring", 1) for c in s.candidates if c.members == (1,) and c.verdict.state == INFRA]


def _two_erring(env, second: str) -> tuple[Loop, dict, dict]:
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    loop = env.loop(_rings_owner_of_one)
    one, two = loop.tick()
    k1 = env.listed(one["sha"], {"pytest:a": ["a.txt"]})
    k2 = env.listed(two["sha"], {"pytest:b": ["b.txt"]})
    env.result(k1["pytest:a"], "pytest:a", "error", infra=True)
    if second == INFRA:
        env.result(k2["pytest:b"], "pytest:b", "error", infra=True)
    else:
        env.result(k2["pytest:b"], "pytest:b", second)
    return loop, one, two


def test_in_an_outage_a_ring_to_a_pr_s_owner_is_held_and_logged_while_another_check_reads_infra_too(env):
    loop, one, two = _two_erring(env, INFRA)
    (held,) = loop.tick()
    assert (held["action"], held["sha"], held["posted"]) == ("ring", one["sha"], False) and two["sha"][:12] in held["held"]
    assert env.posts == [] and loop.tick() == []                    # once: the outage told nobody


def test_a_lone_pr_whose_check_keeps_erring_rings_its_owner_unless_main_s_run_beside_reads_infra_too(env):
    loop, one, _ = _two_erring(env, PASS)
    t = loop.snapshot(L.Past.read(loop.log_path, 0), 9)
    c = next(c for c in t.snap.candidates if c.members == (1,))
    assert loop._outage(c, t) is None
    assert "ended in infra" in loop._outage(c, replace(t, below={"verdict": {"state": INFRA}}))
    (rung,) = loop.tick()
    assert (rung["action"], rung["sha"], rung["posted"]) == ("ring", one["sha"], True) and len(env.posts) == 1
    assert "keeps erring" in env.posts[0]


def test_a_check_rung_for_staying_pending_is_rung_once_still_when_its_ring_is_older_than_the_log_s_window(env):
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide, pending_s=3600)
    loop.tick()
    env.clock += 3601
    (retry,) = loop.tick()
    env.clock += 3600
    (ring,) = loop.tick()
    assert (retry["action"], ring["action"], len(env.posts)) == ("retry", "ring", 1)
    env.clock += 49 * 3600
    assert loop.tick() == [] and len(env.posts) == 1


def test_a_fail_whose_every_failure_is_inherited_lands_once_the_run_beside_it_passed_under_slim_s_own_combine(env):
    from research import vqueue_slim

    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    slim = Slim()
    slim.combine = vqueue_slim.combine                              # it keeps no `ended`: the loop carries it
    loop = env.loop(vqueue.decide, slim=slim)
    loop.world.slim_driver = lambda: "drive"
    _, built, _ = loop.tick()
    red = {"pytest:r": "sha512:red"}
    env.listed(main, {"pytest:r": ["red.txt"]}, keys=red)
    keys = env.listed(built["sha"], {"pytest:a": ["a.txt"], "pytest:r": ["red.txt"]}, keys=red)
    env.result("sha512:red", "pytest:r", "fail")
    assert loop.tick() == []                                        # `pytest:a` still runs: the fail hasn't ended
    env.result(keys["pytest:a"], "pytest:a", "pass")
    assert loop.tick() == []                                        # the run beside still runs: the fail hasn't ended
    slim.verdicts["run-1"] = PASS
    assert _acts(loop.tick()) == ["slim", "red", "advance", "on_main", "drop"] and env.main() == built["sha"]


def _ticks(env: Env, slim: Slim, loop: Loop, n: int, *failed: str) -> list[dict]:
    """``n`` ticks of ``loop``, what they logged: each candidate one builds passes graph2's check, and the run beside its check
    ends failing ``failed``."""
    logged = []
    for _ in range(n):
        logged += (got := loop.tick())
        for b in got:
            if b["action"] == "build":
                keys = env.listed(b["sha"], {"pytest:x": [f"{b['members'][0]}.txt"]})
                env.result(keys["pytest:x"], "pytest:x", "pass")
                slim.end(f"run-{len(slim.started)}", *failed)
    return logged


def test_a_red_main_on_a_step_read_whole_is_inherited_from_the_run_beside_main_and_every_lone_member_lands_with_one_red(env):
    """preflight-lints reads the whole tree, so no candidate is at `main`'s key of it: with `main` red on it, each lone member's
    run beside its check fails it, and again on its rerun, which ejected every member.  The run beside `main` itself fails it
    too: each candidate waits for that run, then takes the failure as inherited (`vqueue_slim.based`) and lands, one red rung
    for all, the run beside each candidate `main` advanced to `main`'s next."""
    main = env.main()
    for n in (1, 2, 3):
        env.put(n, {f"{n}.txt": f"{n}\n"})
    slim = Slim()
    loop = env.loop(vqueue.decide, slim=slim, batch=1, depth=1)
    loop.world.slim_driver = lambda: "drive"
    logged = _ticks(env, slim, loop, 3, "preflight-lints")
    assert _acts(logged) == ["slim", "build", "slim", "slim"] and logged[0]["why"] == "started beside `main`"
    assert logged[3]["record"]["verdict"]["state"] == FAIL and env.main() == main     # waiting for the run beside `main`
    slim.end("main-1", "preflight-lints")
    logged = _ticks(env, slim, loop, 8, "preflight-lints")
    assert _acts(logged).count("advance") == 3 and env.main() == [r for r in logged if r["action"] == "advance"][-1]["sha"]
    assert [r["action"] for r in logged if r["action"] in ("red", "eject", "rerun", "retry")] == ["red"]
    assert len(env.posts) == 1 and env.withdrawn == [] and slim.below == [("drive", main)]
    assert sorted(m for r in logged if r["action"] == "advance" for m in r["members"]) == [1, 2, 3]


def test_a_candidate_s_own_failure_of_a_step_read_whole_that_the_run_beside_main_passed_is_its_own(env):
    """With `main` green on preflight-lints, a lone member whose run beside fails it fails by its own change: no red, no wait,
    and it is ejected, its failure meeting its files."""
    env.put(1, {"1.txt": "1\n"})
    slim = Slim()
    slim.inputs = frozenset({"*.txt"})
    loop = env.loop(vqueue.decide, slim=slim, batch=1, depth=1)
    loop.world.slim_driver = lambda: "drive"
    slim.end("main-1")
    logged = _ticks(env, slim, loop, 2, "preflight-lints")
    assert _acts(logged) == ["slim", "build", "slim", "slim", "slim", "eject", "drop"]
    assert logged[3]["why"] == "judged pass beside `main`" and env.withdrawn == [(1, logged[1]["heads"][0])]
    assert env.posts and not [r for r in logged if r["action"] in ("red", "advance")]


def test_a_check_that_failed_to_list_never_advances_beside_a_run_whose_every_failure_is_inherited(env):
    """A candidate whose graph2 check failed to list ran none of its gating tasks: beside a run whose only failure the run
    beside `main` fails too, its fail never ends, so it waits as beside a run still going, and `main` doesn't move."""
    main = env.main()
    env.put(1, {"1.txt": "1\n"})
    slim = Slim()
    loop = env.loop(vqueue.decide, slim=slim, batch=1, depth=1)
    loop.world.slim_driver = lambda: "drive"
    logged = loop.tick()
    (built,) = [r for r in logged if r["action"] == "build"]
    env.graph2.list_graph(f"ready:{built['sha']}", {"schema": GRAPH_V2, "commit": "0" * 40, "tasks": []})
    slim.end("main-1", "preflight-lints")
    slim.end(f"run-{len(slim.started)}", "preflight-lints")
    for _ in range(4):
        logged += loop.tick()
    assert env.main() == main and "advance" not in _acts(logged), _acts(logged)


def test_a_fail_waits_for_the_run_beside_main_while_it_runs_but_no_longer_than_pending_s(env):
    """A whole-tree step's failure is unknown while the run beside `main` runs: the candidate waits, rather than being rerun or
    ejected, until that run says, and no longer than `pending_s` after it started; then its failure is its own."""
    env.put(1, {"1.txt": "1\n"})
    slim = Slim()
    slim.inputs = frozenset({"*.txt"})
    loop = env.loop(vqueue.decide, slim=slim, batch=1, depth=1, pending_s=3600)
    loop.world.slim_driver = lambda: "drive"
    logged = _ticks(env, slim, loop, 2, "preflight-lints")
    assert _acts(logged) == ["slim", "build", "slim", "slim"]
    env.clock += 3600
    assert loop.tick() == [] and env.withdrawn == []
    env.clock += 1
    assert _acts(loop.tick()) == ["eject", "drop"] and len(env.withdrawn) == 1 and slim.below == [("drive", env.main())]


@pytest.mark.parametrize("why", ["graph2 fails it too", "the run beside `main` ended in infra"])
def test_a_fail_waits_for_no_run_beside_main_when_it_fails_by_its_own_or_that_run_can_t_say(env, why):
    """A failure of its own besides (graph2's) blames the member at once; and a run beside `main` that ended in infra, twice,
    says nothing of the step: no wait."""
    env.put(1, {"1.txt": "1\n"})
    slim = Slim()
    slim.inputs = frozenset({"*.txt"})
    loop = env.loop(vqueue.decide, slim=slim, batch=1, depth=1)
    loop.world.slim_driver = lambda: "drive"
    if why == "graph2 fails it too":
        logged = loop.tick()
        keys = env.listed(logged[1]["sha"], {"pytest:x": ["1.txt"]})
        env.result(keys["pytest:x"], "pytest:x", "fail")
        slim.end("run-1", "preflight-lints")
    else:
        slim.verdicts["main-1"] = slim.verdicts["main-2"] = INFRA
        logged = _ticks(env, slim, loop, 2, "preflight-lints")
    logged += loop.tick()
    assert _acts(logged)[-2:] == ["eject", "drop"] and len(env.withdrawn) == 1 and len(slim.below) == 1 + (why != "graph2 fails it too")


def test_the_run_beside_main_starts_once_a_main_again_once_after_infra_and_with_dry_run_never(env):
    main = env.main()
    slim = Slim()
    dry = env.loop(vqueue.decide, slim=slim, dry_run=True)
    assert dry.tick() == [] and slim.below == []
    loop = env.loop(vqueue.decide, slim=slim)
    loop.world.slim_driver = lambda: "drive"
    (started,) = loop.tick()
    assert (started["sha"], started["base"], started["why"], slim.below) == (main, main, "started beside `main`",
                                                                           [("drive", main)])
    assert loop.tick() == []
    slim.verdicts["main-1"] = INFRA
    judged, again = loop.tick()
    assert (judged["why"], again["why"], again["record"]["ids"]) == ("judged infra beside `main`", "started again beside `main`",
                                                                      ["main-2"])
    slim.verdicts["main-2"] = INFRA
    assert _acts(loop.tick()) == ["slim"] and loop.tick() == [] and len(slim.below) == 2
    _git(env.o.origin, "update-ref", "refs/heads/main", env.o.change({"m.txt": "m\n"}))     # `main` moved by drive
    (moved,) = loop.tick()
    assert (moved["sha"], moved["why"], slim.below[-1]) == (env.main(), "started beside `main`", ("drive", env.main()))
    assert env.loop(vqueue.decide, clone="again", slim=slim, dry_run=True).tick() == [] and len(slim.below) == 3


def test_the_shadow_advances_its_own_main_once_a_candidate_and_the_comparer_reads_one_advance_each(env):
    from research import vqueue_compare as vc

    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide, shadow=True)
    (one,) = loop.tick()
    keys = env.listed(one["sha"], {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    env.put(2, {"b.txt": "2\n"})
    advance, drop, two = loop.tick()
    assert (advance["action"], drop["action"], two["action"], two["base"]) == ("advance", "drop", "build", one["sha"])
    keys = env.listed(two["sha"], {"pytest:b": ["b.txt"]})
    env.result(keys["pytest:b"], "pytest:b", "pass")
    assert _acts(loop.tick()) == ["advance", "drop"]
    for _ in range(3):
        assert loop.tick() == []
    env.clock += 49 * 3600                                          # the log's window holds no advance now
    assert loop.tick() == [] and _git(loop.world.repo, "rev-parse", "refs/vqueue/shadow") == two["sha"]
    assert env.main() == main and [r["sha"] for r in env.lines() if r["action"] == "advance"] == [one["sha"], two["sha"]]
    entries, bad = vc.parse_log(env.log.read_text())
    s = vc.shadow(entries)
    assert not bad and [e.sha for e in s.advances] == [one["sha"], two["sha"]]
    assert all(s.cands[c["sha"]].passed for c in (one, two))


class Records:
    """`research.vqueue_records`'s functions, faked: each start registers a records run on its candidate (``regen-N``), judged
    as ``verdicts`` says (pending unless it names it), a verdict other than pending kept in its record, with the run that
    passed, as `judge` keeps it; a commit commits ``files`` on top of the candidate in the clone ``repo`` as a lock commit
    (`vqueue_records.lock_message`), at ``dated`` if set (else now), kept in ``made``, with ``told`` (what is told of the
    bookkeeping its run repaired), makes none when ``files`` is empty (its run regenerated nothing), or says why not
    (``held``).  With ``launching``, a start is a detached launch, pending until ``registered`` names the run ids it printed
    for its sha."""

    def __init__(self, repo: Path):
        self.repo, self.files = repo, {"pkg/lean-audit.json": '{"records": 2}\n'}
        self.started: list[tuple] = []
        self.committed: list[tuple] = []
        self.made: list[str] = []
        self.verdicts: dict[str, RecordsRun] = {}
        self.held: str | None = None
        self.dated: str | None = None
        self.told: tuple[str, tuple[str, ...]] | None = None
        self.launching, self.registered = False, {}
        #: each start's keywords
        self.kw: list[dict] = []

    def start(self, driver, sha: str, **kw) -> dict:
        self.started.append((driver, sha))
        self.kw.append(kw)
        launch = {"ids": [], "launch": {"dir": f"records-{len(self.started)}", "pid": 1}} if self.launching else {
            "ids": [f"regen-{len(self.started)}"]}
        return {**launch, "tip": sha}

    def judge(self, driver, rec: dict | None) -> RecordsRun:
        if rec is None:
            return RecordsRun(INFRA, "it registered no run")
        if isinstance(kept := rec.get("verdict"), dict):
            return RecordsRun(kept["state"], kept["why"])
        if "launch" in rec:
            if rec["tip"] not in self.registered:
                return RecordsRun()
            del rec["launch"]
            rec["ids"] = self.registered[rec["tip"]]
        if not rec["ids"]:
            rec["verdict"] = {"state": INFRA, "why": "it registered no run"}
            return RecordsRun(INFRA, "it registered no run")
        run = rec["ids"][0]
        if (r := self.verdicts.get(run, RecordsRun())).state != PENDING:
            rec["verdict"] = {"state": r.state, "why": r.why, **({"run": run} if r.state == PASS else {})}
        return r

    @staticmethod
    def same_inputs(repo: Path, a: str, b: str) -> bool:
        return vqueue_records.same_inputs(repo, a, b)

    def commit(self, driver, sha: str, rid: str, prs, onto: str | None = None) -> tuple[str, list[str], tuple | None] | str | None:
        self.committed.append((driver, sha, rid, tuple(prs), *([onto] if onto else [])))
        assert onto is None or self.same_inputs(self.repo, sha, onto), "records carried onto other Lean inputs"
        if self.held is not None:
            return self.held
        if not self.files:
            return None
        index = {"GIT_INDEX_FILE": str(self.repo.parent / f"records-index-{rid}")}
        Q._git(self.repo, "read-tree", sha, env=index)
        for path, text in sorted(self.files.items()):
            (blob := self.repo.parent / "records-blob").write_text(text)
            Q._git(self.repo, "update-index", "--add", "--cacheinfo",
                   f"100644,{Q._out(self.repo, 'hash-object', '-w', '--', str(blob))},{path}", env=index)
        tree, files = Q._git(self.repo, "write-tree", env=index).stdout.strip(), sorted(self.files)
        message = lock_message(" ".join(f"#{n}" for n in prs), rid, f"queue/{sha[:12]}", sha, [], files)
        when = Q.IDENTITY if self.dated is None else {**Q.IDENTITY, **dict.fromkeys(Q.DATES, self.dated)}
        made = Q._git(self.repo, "commit-tree", tree, "-p", sha, "-m", message, env=when).stdout.strip()
        if onto:
            made = vqueue_records._carry(self.repo, made, onto, files, lock_message(
                " ".join(f"#{n}" for n in prs), rid, f"queue/{sha[:12]}", sha, [], files, onto=onto), when)
        self.made.append(made)
        return self.made[-1], files, self.told


#: a Lean change (`queue.lean_change`): Lean in a package with a lock, and the lock
LEAN = {"pkg/A.lean": "theorem a : True := trivial\n", "pkg/lean-audit.json": '{"records": 1}\n'}


def _records_loop(env: Env, records: Records | None = None, files: dict[str, str] | None = None, **kw,
                  ) -> tuple[Loop, Records, str, str, list[dict]]:
    """The policy's loop with ``records`` (a `Records` on its clone), and candidate ``sha`` of #1 (``head``, a Lean change:
    ``files``, else `LEAN`), built by the loop's first tick, which logged ``first``."""
    head = env.put(1, files or LEAN)
    records = records or Records(env.tmp / kw.get("clone", "clone"))
    loop = env.loop(vqueue.decide, records=records, **kw)
    loop.world.slim_driver = lambda: "drive"
    first = loop.tick()
    return loop, records, next(r["sha"] for r in first if r["action"] == "build"), head, first


def _checks(env: Env) -> list[str]:
    """The checks enqueued on graph2, oldest first."""
    return [e["check"] for e in env.graph2.enqueued()]


def test_a_lean_candidate_s_check_starts_only_after_its_records_run_and_on_its_records_commit_when_it_regenerated_some(env):
    """Its build starts its records run on its tree, and not its check: nothing is enqueued on it, and the next batch is built
    on its base beside it, not on it.  Once the run passed having regenerated records, it goes onto main's chain's top: its
    members are merged again there, whose Lean inputs are its own, and their records commit, made on the staged tree, is
    carried onto that merge and checked in its place; the next tick drops the staged batch for it, and `main` moves to the
    batch beside it, then to it, its members walked beneath it as drive reads them."""
    main = env.main()
    loop, records, sha, one, first = _records_loop(env)
    built, started = first
    assert (built["action"], built["records_first"], built["enqueued"], built["members"]) == ("build", True, False, [1])
    assert (started["action"], started["sha"], started["why"], started["record"]) == (
        "regenerate", sha, "started", {"ids": ["regen-1"], "tip": sha})
    assert records.started == [("drive", sha)] and _checks(env) == []
    env.put(2, {"b.txt": "2\n"})
    (two,) = loop.tick()
    assert (two["action"], two["members"], two["base"], two["enqueued"]) == ("build", [2], main, True)
    assert _checks(env) == [f"ready:{two['sha']}"] and env.posts == []
    records.verdicts["regen-1"] = RecordsRun(PASS)
    judged, fixed = loop.tick()
    assert (judged["action"], judged["why"], judged["record"]["verdict"]) == (
        "regenerate", "judged pass", {"state": PASS, "why": "", "run": "regen-1"})
    new = fixed["sha"]
    assert (fixed["action"], fixed["base"], fixed["members"], fixed["heads"], fixed["repairs"], fixed["run"], fixed["files"],
            fixed["enqueued"]) == ("build", two["sha"], [1], [one], sha, "regen-1", ["pkg/lean-audit.json"], True), fixed
    rebuilt = _git(env.o.origin, "rev-parse", f"{new}^")
    assert records.committed == [("drive", sha, "regen-1", (1,), rebuilt)]
    assert _git(env.o.origin, "rev-parse", f"{rebuilt}^") == two["sha"]
    assert _git(env.o.origin, "log", "-1", "--format=%s", new) == "Lock records regenerated for #1"
    assert f"Carried from {sha[:12]} onto {rebuilt[:12]}" in _git(env.o.origin, "log", "-1", "--format=%b", new)
    assert _checks(env) == [f"ready:{two['sha']}", f"ready:{new}"]
    (drop,) = loop.tick()
    assert (drop["action"], drop["sha"], drop["why"], drop["outcome"]) == (
        "drop", sha, f"its records were regenerated in {new[:12]}, which is checked in its place", None)
    assert branch(sha) not in env.queued()
    for c, tasks in ((two["sha"], {"pytest:a": ["a.txt"]}),
                     (new, {"lean-lock:pkg": ["pkg/A.lean", "pkg/lean-audit.json"], "pytest:a": ["a.txt"]})):
        for task, key in env.listed(c, tasks).items():
            env.result(key, task, "pass")
    loop.tick()
    loop.tick()
    assert [r["sha"] for r in env.lines() if r["action"] == "advance"] == [two["sha"], new]
    assert env.main() == new and _git(env.o.origin, "show", "main:pkg/lean-audit.json") == '{"records": 2}'
    assert _git(env.o.origin, "show", "main:b.txt") == "2"
    assert [m.pr for m in loop.world.members(new, two["sha"], [1], [one])] == [1]
    assert not any("isn't its members' merges as built" in x for x in env.out), env.out
    assert env.withdrawn == [] and env.posts == [] and len(records.started) == 1


def test_a_pr_changing_a_lean_input_waits_while_a_batch_is_staged_and_is_built_on_it_once_it_is_on_main_s_chain(env):
    """`Ready.lean_input` is a PR's own change meeting the Lean inputs (`vqueue_slim.lean_inputs`) of `main`'s tree with the
    ready PRs' changes in it: check's `common.py`, or a script of a Lake package, here the one the staged Lean PR adds.  Neither
    is built beside the staged batch, which would change the Lean inputs of the top it goes onto and start its records run
    again there; a PR touching none is.  The tick that puts the batch on main's chain, its records carried onto that PR's
    candidate, builds the two on it."""
    main = env.main()
    loop, records, sha, one, _ = _records_loop(env)
    env.put(2, {"pkg/notes.py": "2\n"})
    env.put(3, {"tools/verity/check/common.py": "3\n"})
    env.put(4, {"b.txt": "4\n"})
    (four,) = loop.tick()
    assert (four["members"], four["base"], four["enqueued"]) == ([4], main, True), four
    assert {r.number: (r.lean, r.lean_input) for r in loop.world.ready(main)} == {
        1: (True, True), 2: (False, True), 3: (False, True), 4: (False, False)}
    assert loop.tick() == []
    records.verdicts["regen-1"] = RecordsRun(PASS)
    judged, fixed, built = loop.tick()
    assert (fixed["action"], fixed["base"], fixed["members"], fixed["repairs"], fixed["enqueued"]) == (
        "build", four["sha"], [1], sha, True), fixed
    assert (built["action"], built["members"], built["base"], built["enqueued"]) == ("build", [2, 3], fixed["sha"], True), built
    (drop,) = loop.tick()
    assert (drop["action"], drop["sha"]) == ("drop", sha) and len(records.started) == 1


def test_the_tick_that_stages_a_lean_batch_builds_the_prs_with_no_lean_change_beside_it_on_the_same_base(env):
    """Staging takes no build from the PRs with no Lean change: the tick that stages the Lean batch, its records run started
    and no check of it enqueued, builds theirs on the same base, its check enqueued at once."""
    main = env.main()
    env.put(2, {"b.txt": "2\n"})
    loop, records, sha, _, first = _records_loop(env)
    staged, started, beside = first
    assert (staged["action"], staged["members"], staged["base"], staged["records_first"], staged["enqueued"]) == (
        "build", [1], main, True, False), staged
    assert (started["action"], started["sha"]) == ("regenerate", sha) and records.started == [("drive", sha)]
    assert (beside["action"], beside["members"], beside["base"], beside["enqueued"]) == ("build", [2], main, True), beside
    assert _checks(env) == [f"ready:{beside['sha']}"] and loop.tick() == []


@pytest.mark.parametrize("regenerated", [True, False])
def test_the_tick_that_puts_a_staged_batch_on_main_s_chain_builds_on_what_it_put_there(env, regenerated):
    """`Build.after`: once the staged batch's records run passed, the tick that puts it on main's chain, as its records
    commit or, its run having regenerated nothing, as built, stages the next Lean batch on that candidate, its records run
    started, and builds the PRs with no Lean change on it, their check enqueued: neither waits for the next tick."""
    loop, records, sha, _, _ = _records_loop(env)
    if not regenerated:
        records.files = {}
    env.put(2, {"other/B.lean": "theorem b : True := trivial\n", "other/lean-audit.json": '{"records": 1}\n'})
    env.put(3, {"c.txt": "3\n"})
    (three,) = loop.tick()
    assert (three["members"], three["enqueued"]) == ([3], True), three
    env.put(4, {"d.txt": "4\n"})
    records.verdicts["regen-1"] = RecordsRun(PASS)
    judged, fixed, staged, started, built = loop.tick()
    on = fixed["sha"]
    assert (judged["why"], fixed["action"], fixed["members"], fixed["base"], fixed["enqueued"], fixed.get("repairs")) == (
        "judged pass", "build", [1], three["sha"], True, sha), fixed
    assert (staged["action"], staged["members"], staged["base"], staged["records_first"], staged["enqueued"]) == (
        "build", [2], on, True, False), staged
    assert (started["action"], started["sha"]) == ("regenerate", staged["sha"])
    assert (built["action"], built["members"], built["base"], built["enqueued"]) == ("build", [4], on, True), built
    assert records.started == [("drive", sha), ("drive", staged["sha"])]
    assert _checks(env) == [f"ready:{c}" for c in (three["sha"], on, built["sha"])]
    (drop,) = loop.tick()
    assert (drop["action"], drop["sha"]) == ("drop", sha)


def test_at_lean_0_a_pin_puts_no_lean_batch_ahead_of_a_pr_changing_a_lean_input_which_is_built_before_the_next_is_staged(env):
    """End to end at `--lean 0`, pinned Lean batches back to back: #2 is pinned while pinned #1 is staged, and #3, which
    changes check's `common.py`, a Lean input, is marked beside it and kept out of every build.  The tick that puts #1 on main's
    chain stages no batch of #2: #3 goes first, built on #1's candidate, and the next tick stages #2 on #3's, its records
    run started then."""
    head = env.put(1, LEAN)
    env.put(2, {"other/B.lean": "theorem b : True := trivial\n", "other/lean-audit.json": '{"records": 1}\n'})
    (pins := env.tmp / "pins").write_text("1\n")
    records = Records(env.tmp / "clone")
    loop = env.loop(vqueue.decide, records=records, lean=0)
    loop.world.slim_driver, loop.world.pins_file = (lambda: "drive"), pins
    staged, started = loop.tick()
    sha = staged["sha"]
    assert (staged["members"], staged["heads"], staged["records_first"], started["action"]) == ([1], [head], True, "regenerate")
    pins.write_text("1\n2\n")
    env.put(3, {"tools/verity/check/common.py": "3\n"})
    assert loop.tick() == [] and records.started == [("drive", sha)]
    records.verdicts["regen-1"] = RecordsRun(PASS)
    judged, fixed, built = loop.tick()
    assert (fixed["action"], fixed["members"], fixed["repairs"], fixed["enqueued"]) == ("build", [1], sha, True), fixed
    assert (built["action"], built["members"], built["base"], built["enqueued"]) == ("build", [3], fixed["sha"], True), built
    assert records.started == [("drive", sha)]
    drop, next_, began = loop.tick()
    assert (drop["action"], drop["sha"]) == ("drop", sha)
    assert (next_["action"], next_["members"], next_["base"], next_["records_first"], next_["enqueued"]) == (
        "build", [2], built["sha"], True, False), next_
    assert (began["action"], began["sha"]) == ("regenerate", next_["sha"])
    assert records.started == [("drive", sha), ("drive", next_["sha"])]


def test_a_tick_whose_commit_stages_its_batch_again_or_is_held_builds_nothing_after_it(env):
    """A build after a Commit goes on what the Commit put on main's chain.  Staged again on a top whose Lean inputs aren't
    the batch's, or its records commit held, it put nothing there: the tick builds nothing after it, logs nothing of that
    build, and the next tick builds beside the batch still staged."""
    loop, records, sha, _, _ = _records_loop(env)
    moved = _main_to(env, {"pkg/notes.txt": "2\n"})
    env.put(2, {"b.txt": "2\n"})
    records.verdicts["regen-1"] = RecordsRun(PASS)
    judged, again, started = loop.tick()
    assert (again["action"], again["base"], again["members"], again["repairs"], again["records_first"]) == (
        "build", moved, [1], sha, True), again
    assert (started["action"], started["sha"]) == ("regenerate", again["sha"])
    drop, beside = loop.tick()
    assert (drop["action"], drop["sha"]) == ("drop", sha)
    assert (beside["action"], beside["members"], beside["base"], beside["enqueued"]) == ("build", [2], moved, True), beside
    records.verdicts["regen-2"], records.held = RecordsRun(PASS), "its records commit can't be made yet"
    env.put(3, {"c.txt": "3\n"})
    judged, held, ring = loop.tick()
    assert (judged["why"], held["action"], held["sha"], held["done"], ring["action"]) == (
        "judged pass", "commit", again["sha"], False, "ring"), held
    records.held = None
    lines = loop.tick()
    fixed, built = (r for r in lines if r["action"] == "build")
    assert (fixed["members"], fixed["base"], fixed["repairs"], built["members"], built["base"]) == (
        [1], beside["sha"], again["sha"], [3], fixed["sha"]), lines


@pytest.mark.parametrize("names", [False, True])
def test_a_staged_batch_whose_records_run_passed_after_a_move_landed_is_merged_again_on_the_chain_s_top_unless_it_names_it(
        env, names):
    """A Lean batch staged before a move lands, its records run passing after: its member's head lacks the move.  Where its
    own change names nothing the move's map retires, the loop takes it (`queue.lacked_moves`) and the tick puts the batch
    onto main's chain's top, its member merged again there, as for any move of `main`.  Where it names a retired path, the
    batch is dropped naming the move, the names and the fix (`vqueue._untaken`), never "isn't marked" (its head and mark are
    as staged), and its owner is told once (`Loop._lacking`)."""
    loop, records, sha, one, _ = _records_loop(env, files={**LEAN, **({"e.txt": "reads c.txt\n"} if names else {})})
    _, move = _landed_move(env.o, '[paths]\n"c.txt" = "moved/c.txt"\n', {"c.txt": None, "moved/c.txt": "c\n"})
    env.listed(move, {})
    records.verdicts["regen-1"] = RecordsRun(PASS)
    lines = loop.tick()
    if not names:
        (fixed,) = (r for r in lines if r["action"] == "build")
        assert "drop" not in _acts(lines) and "lacks" not in _acts(lines), lines
        assert (fixed["members"], fixed["heads"], fixed["base"], fixed["repairs"]) == ([1], [one], move, sha), fixed
        assert _git(env.o.origin, "merge-base", "--is-ancestor", move, fixed["sha"]) == "" and env.posts == []
        return
    (told,) = (r for r in lines if r["action"] == "lacks")
    (drop,) = (r for r in lines if r["action"] == "drop")
    why = (f"it lacks the move {move[:12]} `main` landed (tools/move/x_map.toml), whose retired names its own change names "
           "(`c.txt`): merge `main` into it and rename them, or cross it with `tools/move/restack.py`, and mark the new head "
           "ready")
    assert (drop["sha"], drop["why"]) == (sha, f"#1 isn't taken at the head it was staged at ({why}), so it isn't merged "
                                               f"again onto {move[:12]}, main's chain's top now: its other members are built "
                                               "again"), drop
    assert (told["number"], told["heads"], told["base"], told["why"]) == (1, [one], move, why), told
    assert "build" not in _acts(lines) and len(env.posts) == 1, lines
    assert "lacks" not in _acts(loop.tick()) and len(env.posts) == 1


def test_a_staged_batch_whose_chain_top_changed_its_lean_inputs_is_staged_again_there_its_records_run_first(env):
    """`main` moved by another to a change of a Lean input (a file of the staged batch's Lean package) leaves its records
    regenerated on a tree whose Lean inputs aren't the chain top's: none are carried, and its members merged onto that top are
    staged again in its place, a records run of their own started, and the staged batch is dropped for it."""
    main = env.main()
    loop, records, sha, one, _ = _records_loop(env)
    moved = _main_to(env, {"pkg/notes.txt": "2\n"})
    records.verdicts["regen-1"] = RecordsRun(PASS)
    judged, again, started = loop.tick()
    assert (judged["why"], again["action"], again["base"], again["members"], again["repairs"], again["records_first"],
            again["enqueued"]) == ("judged pass", "build", moved, [1], sha, True, False), again
    assert again["why"] == (f"{sha[:12]}'s records were regenerated on {main[:12]}, and its Lean inputs aren't that tree's: "
                            "staged again, its records run first")
    assert (started["action"], started["sha"]) == ("regenerate", again["sha"]) and records.committed == []
    assert records.started == [("drive", sha), ("drive", again["sha"])] and _checks(env) == []
    (drop,) = loop.tick()
    assert (drop["action"], drop["sha"], drop["why"]) == (
        "drop", sha, f"it is staged again in {again['sha'][:12]}, on {moved[:12]}, whose Lean inputs aren't those its "
                     "records were regenerated on: its records run first again")
    records.verdicts["regen-2"] = RecordsRun(PASS)
    judged, fixed = loop.tick()
    assert (fixed["base"], fixed["repairs"], fixed["members"], records.committed[-1]) == (
        moved, again["sha"], [1], ("drive", again["sha"], "regen-2", (1,))), fixed


def test_a_lean_candidate_whose_records_run_regenerated_nothing_is_checked_as_built_and_lands(env):
    records = Records(env.tmp / "clone")
    records.files = {}
    loop, _, sha, one, _ = _records_loop(env, records)
    records.verdicts["regen-1"] = RecordsRun(PASS)
    judged, built = loop.tick()
    assert (judged["why"], built["action"], built["sha"], built["enqueued"], built["why"], built["run"]) == (
        "judged pass", "build", sha, True, "its records run regen-1 regenerated nothing: checked as built", "regen-1")
    assert _checks(env) == [f"ready:{sha}"] and env.queued() == {branch(sha): sha} and records.made == []
    keys = env.listed(sha, {"lean-lock:pkg": ["pkg/A.lean", "pkg/lean-audit.json"]})
    env.result(keys["lean-lock:pkg"], "lean-lock:pkg", "pass")
    advance, on_main, landed = loop.tick()
    assert (env.main(), landed["outcome"]["state"], len(records.started), env.withdrawn) == (sha, PASS, 1, [])


def test_a_candidate_with_no_lean_member_starts_its_check_at_once_with_no_records_run(env):
    env.put(1, {"a.txt": "1\n"})
    records = Records(env.tmp / "clone")
    loop = env.loop(vqueue.decide, records=records)
    loop.world.slim_driver = lambda: "drive"
    (built,) = loop.tick()
    assert (built["action"], built["enqueued"], "records_first" in built) == ("build", True, False)
    assert _checks(env) == [f"ready:{built['sha']}"] and records.started == []


def test_a_lean_batch_is_pushed_only_once_drive_s_machines_can_run_its_records_run(env):
    """drive.toml naming no machines stops the tick before the push: no candidate is left whose records run can't start."""
    env.put(1, LEAN)
    loop = env.loop(vqueue.decide, records=Records(env.tmp / "clone"))

    def no_machines():
        raise Q.QueueError("drive.toml names no machines")

    loop.world.slim_driver = no_machines
    (failed,) = loop.tick()
    assert (failed["action"], env.queued()) == ("build", {}) and "names no machines" in failed["error"], failed


def test_the_run_beside_a_lean_candidate_s_check_starts_with_its_check_not_with_its_records_run(env):
    slim = Slim()
    loop, records, sha, _, first = _records_loop(env, slim=slim)
    assert _acts(first) == ["slim", "build", "regenerate"] and first[0]["why"] == "started beside `main`"
    assert slim.started == [] and loop.tick() == [] and slim.started == []
    records.verdicts["regen-1"] = RecordsRun(PASS)
    judged, fixed, beside = loop.tick()
    assert (beside["action"], beside["sha"], beside["why"]) == ("slim", fixed["sha"], "started")
    assert [s for _, s, _ in slim.started] == [fixed["sha"]] and _checks(env) == [f"ready:{fixed['sha']}"]


def test_a_records_run_that_failed_drops_its_candidate_and_ejects_its_lean_member(env):
    """A records run that failed refuses its records: no records commit and no check; the candidate is dropped, its Lean
    member ejected with the reason and out of the next batch, which takes the PR with no Lean change that waited behind it
    and is checked at once, and nobody but the ejected member's owner is told.  A push brings the Lean member back."""
    loop, records, sha, one, (built, _) = _records_loop(env)
    assert built["members"] == [1] and built["records_first"]
    two = env.put(2, {"b.txt": "2\n"})
    why = "it failed: regen-1 failed rc=1"
    records.verdicts["regen-1"] = RecordsRun(FAIL, why)
    judged, eject, drop, rebuilt = loop.tick()
    assert (judged["why"], eject["action"], eject["number"], env.withdrawn) == ("judged fail", "eject", 1, [(1, one)])
    assert eject["why"] == f"its records run refused its records: {why}" and "Failed:" not in eject["text"], eject
    assert (drop["sha"], drop["why"], drop["outcome"]["state"], drop["outcome"]["suspects"]) == (sha, eject["why"], FAIL, [1])
    assert (rebuilt["action"], rebuilt["members"], rebuilt["heads"], rebuilt["enqueued"]) == ("build", [2], [two], True)
    assert records.committed == [] and env.queued() == {branch(rebuilt["sha"]): rebuilt["sha"]}
    assert _checks(env) == [f"ready:{rebuilt['sha']}"] and len(env.posts) == 1   # the ejection's, to #1's owner
    assert loop.tick() == [] and len(records.started) == 1
    again = env.put(1, {**LEAN, "pkg/B.lean": "theorem b : True := trivial\n"})
    built, started = loop.tick()
    assert (built["members"], built["heads"], built["records_first"], started["why"]) == ([1], [again], True, "started")


@pytest.mark.parametrize("heavy, on, want", [
    ("1\n", "big", {"on": "big"}), ("*\n", "big1,big2", {"on": "big1,big2"}), (None, "big", {}), ("2\n", "big", {}),
    ("1\n", "", {}), ("*\n", "", {}),
], ids=["heavy", "every", "no-file", "another-pr", "no-machines", "every-no-machines"])
def test_a_heavy_pr_s_records_run_and_its_start_again_after_infra_take_heavy_on_and_any_other_s_the_queue_s(env, heavy, on,
                                                                                                         want):
    """A candidate holding a PR the heavy file names, or any candidate while it holds `*`, starts its records run on
    `--heavy-on`'s machines (`Loop._heavy_on`), and so does its start again after infra; with no heavy file, a file naming
    another PR, or no `--heavy-on`, the start is as before, given no machines."""
    env.put(1, LEAN)
    records = Records(env.tmp / "clone")
    loop = env.loop(vqueue.decide, records=records)
    loop.world.slim_driver = lambda: "drive"
    if heavy is not None:
        (env.tmp / "heavy").write_text(heavy)
    loop.world.heavy_file, loop.world.heavy_on = env.tmp / "heavy", on
    built, started = loop.tick()
    assert (built["records_first"], started["action"], records.kw) == (True, "regenerate", [want])
    records.verdicts["regen-1"] = RecordsRun(INFRA, "it never ran or was stopped (regen-1)")
    judged, again = loop.tick()
    assert (judged["why"], again["why"], records.kw) == (f"judged {INFRA}", "started again", [want, want])


def test_a_records_run_that_ends_in_infra_is_started_once_more_and_then_refuses_ringing_infra(env):
    def refuse(driver, sha):
        raise Q.QueueError("train.sh printed no run dir")

    records = Records(env.tmp / "clone")
    records.start = refuse
    loop, _, sha, one, first = _records_loop(env, records)
    assert (first[1]["action"], first[1]["record"], first[1]["start_error"]) == (
        "regenerate", None, "QueueError: train.sh printed no run dir")
    del records.start
    (again,) = loop.tick()                                          # one that registered no run is infra: started again
    assert (again["why"], again["record"]["ids"], env.withdrawn, _checks(env)) == ("started again", ["regen-1"], [], [])
    records.verdicts["regen-1"] = RecordsRun(INFRA, "it never ran or was stopped (regen-1)")
    judged, ring, eject, drop = loop.tick()
    assert (judged["why"], env.withdrawn, drop["outcome"]["state"]) == (f"judged {INFRA}", [(1, one)], FAIL)
    assert eject["why"] == "its records run ended in infra twice: it never ran or was stopped (regen-1)", eject["why"]
    assert env.posts[0].startswith("@infra vqueue: ") and ring["why"] == f"{sha[:12]}'s records run ended in infra twice: " \
                                                                       "it never ran or was stopped (regen-1)"
    assert ring["text"] == (f"the verified queue's candidate queue/{sha[:12]} (#1), whose check waits for its records run: "
                            f"{ring['why']}"), ring["text"]
    assert loop.tick() == [] and len(records.started) == 1 and env.queued() == {} and _checks(env) == []


def test_a_records_commit_whose_records_are_stale_still_fails_as_any_candidate_and_starts_no_second_records_run(env):
    """A lean-lock failure of a records commit's check ejects as any failure does, with no word of its own on regenerating
    records, which the records run before a check does: no records run follows a check."""
    loop, records, sha, one, _ = _records_loop(env)
    records.verdicts["regen-1"] = RecordsRun(PASS)
    _, fixed = loop.tick()
    (drop,) = loop.tick()                                           # #1's first candidate, dropped for its records commit
    keys = env.listed(new := fixed["sha"], {"lean-lock:pkg": ["pkg/A.lean", "pkg/lean-audit.json"]})
    env.result(keys["lean-lock:pkg"], "lean-lock:pkg", "fail")
    eject, drop = loop.tick()
    assert eject["why"] == "lean-lock:pkg failed, and its changed files meet their inputs, no other member's", eject["why"]
    assert eject["text"].endswith("Fix it and mark it ready again") and "regenerate" not in eject["text"], eject["text"]
    assert (env.withdrawn, drop["sha"], drop["outcome"]["state"], len(records.started)) == ([(1, one)], new, FAIL, 1)


def test_a_records_commit_the_gate_wont_vouch_for_is_held_rung_to_ci_once_and_made_once_it_can_be(env):
    loop, records, sha, _, _ = _records_loop(env)
    records.verdicts["regen-1"] = RecordsRun(PASS)
    records.held = "the merge gate refuses its lock commit: its records aren't byte for byte the ones regen-1 regenerated"
    judged, held, ring = loop.tick()
    assert (held["action"], held["done"], held["why"], held["run"]) == ("commit", False, records.held, "regen-1")
    assert (ring["action"], ring["why"]) == ("ring", f"{sha[:12]}'s records commit is held: {records.held}; it is dropped, its "
                                                     "members built again, once it is held 4 h after it was first")
    assert env.posts == [f"@ci vqueue: the verified queue's candidate queue/{sha[:12]} (#1), whose check waits for its records "
                         f"run: {ring['why']}"] and env.queued() == {branch(sha): sha} and _checks(env) == []
    (again,) = loop.tick()
    assert again["action"] == "commit" and len(env.posts) == 1 and env.withdrawn == []
    records.held = None
    (fixed,) = loop.tick()
    assert (fixed["action"], fixed["repairs"]) == ("build", sha) and len(records.committed) == 3


def test_a_records_commit_held_past_pending_s_drops_its_candidate_and_builds_its_members_again(env):
    """Bounded: `pending_s` after it was first held, the candidate is dropped, its records run settled, and its members built
    again, a new records run first; `CI` was rung once, at the first hold."""
    loop, records, sha, one, _ = _records_loop(env, pending_s=3600)
    records.verdicts["regen-1"] = RecordsRun(PASS)
    records.held = "its regenerated lock records can't be read: Unreadable: regen-1"
    judged, held, ring = loop.tick()
    env.clock += 3600
    assert _acts(loop.tick()) == ["commit"] and len(env.posts) == 1
    env.clock += 1
    drop, built, started = loop.tick()
    assert (drop["action"], drop["sha"], drop["why"], drop["outcome"]) == (
        "drop", sha, "its records commit is still held 1 h after it was first: its members are built again", None)
    assert (built["members"], built["heads"], built["records_first"], started["record"]["ids"]) == ([1], [one], True, ["regen-2"])
    assert len(env.posts) == 1 and env.withdrawn == [] and _checks(env) == [] and env.cancelled == []


def test_a_candidate_built_again_with_the_same_sha_reads_only_its_own_records_run_and_hold():
    """Dropped and built again in the same second on the same base, a candidate has the same sha: its records run and hold
    are those since its newest records-first build, and a settle of the dropped one's still-launching run isn't its record."""
    old, new = {"ids": ["r1"], "tip": "c"}, {"ids": ["r2"], "tip": "c"}
    lines = [{"action": "build", "sha": "c", "records_first": True},
             {"action": "regenerate", "sha": "c", "why": "started", "record": old},
             {"action": "commit", "sha": "c", "done": False, "at": L.stamp(T0)},
             {"action": "drop", "sha": "c"},
             {"action": "build", "sha": "c", "records_first": True},
             {"action": "regenerate", "sha": "c", "why": "started", "record": new}]
    past = L.Past(lines=lines)
    assert (past.regenerated("c"), past.held("c")) == ((1, new), None)
    settled = {"action": "regenerate", "sha": "c", "why": "its candidate was dropped while it launched", "record": old,
               "dropped": True}
    assert L.Past(lines=[*lines, settled]).regenerated("c") == (1, new)
    assert L.Past(lines=lines[:4]).held("c") == T0 and L.Past(lines=lines[:4]).regenerated("c") == (1, old)


def test_the_bookkeeping_a_records_run_repaired_is_told_as_drive_tells_it_once_a_records_commit(env):
    """Drive tells `lean` and each owner a warning names of the bookkeeping its lock run repaired (`drive.bookkeeping_told`), so
    the loop does of a records run's: once, even when a tick cut off before its build line leaves the commit to adopt."""
    loop, records, sha, _, _ = _records_loop(env)
    records.verdicts["regen-1"], records.dated = RecordsRun(PASS), "2026-01-01T00:00:00+0000"
    records.told = ("the lock run regen-1 repaired bookkeeping that #1 brought: H (@proofs)", ("lean", "proofs"))
    judged, told, fixed = loop.tick()
    assert (told["action"], told["sha"], told["to"], told["posted"]) == ("bookkeeping", fixed["sha"], ["@lean", "@proofs"], True)
    assert env.posts == [f"@lean @proofs vqueue: {records.told[0]}"] and fixed["action"] == "build"
    kept = [x for x in env.log.read_text().splitlines() if json.loads(x).get("action") != "build" or json.loads(x)["sha"] == sha]
    env.log.write_text("".join(f"{x}\n" for x in kept))              # the tick cut off before its build line
    records.dated = None
    ignored, again = loop.tick()
    assert (ignored["action"], again["action"], again["sha"], again["adopted"]) == ("ignore", "build", fixed["sha"], True)
    assert len(env.posts) == 1


def test_a_records_commit_pushed_with_no_line_in_the_log_is_adopted_and_a_restart_starts_no_second_records_run(env):
    loop, records, sha, _, _ = _records_loop(env)
    records.verdicts["regen-1"], records.dated = RecordsRun(PASS), "2026-01-01T00:00:00+0000"
    made, *_ = records.commit("drive", sha, "regen-1", (1,))
    _git(env.tmp / "clone", "push", "-q", "origin", f"{made}:refs/heads/queue/{made[:12]}")  # a tick cut off before its line
    again = env.loop(vqueue.decide, clone="again", records=records)
    again.world.slim_driver, records.repo, records.dated = lambda: "drive", again.world.repo, None   # in its loop's clone
    judged, ignored, fixed = again.tick()
    assert (judged["action"], ignored["action"], ignored["sha"]) == ("regenerate", "ignore", made)
    assert (fixed["action"], fixed["sha"], fixed["adopted"], fixed["repairs"]) == ("build", made, True, sha)
    assert records.made[0] == made != records.made[1]                # made again, at another time: the one pushed is kept
    assert env.queued() == {branch(sha): sha, branch(made): made} and len(records.started) == 1


def test_a_records_commit_carried_onto_another_base_and_pushed_with_no_line_in_the_log_is_adopted(env):
    """A tick cut off after it pushed the carried records commit leaves it with no line: the next tick merges the members
    onto the chain's top again and adopts it (`World.recommitted` of that merge built before), with no second records run."""
    loop, records, sha, _, _ = _records_loop(env)
    env.put(2, {"b.txt": "2\n"})
    (two,) = loop.tick()
    records.verdicts["regen-1"], records.dated = RecordsRun(PASS), "2026-01-01T00:00:00+0000"

    class Cut(Exception):
        pass

    def cut(t, recs, line):
        raise Cut(line["sha"])

    loop._enqueue = cut
    with pytest.raises(Cut) as pushed:
        loop.tick()
    made = str(pushed.value)
    assert env.queued()[branch(made)] == made and not any(r.get("sha") == made for r in env.lines())
    again = env.loop(vqueue.decide, clone="again", records=records)
    again.world.slim_driver, records.repo, records.dated = lambda: "drive", again.world.repo, None
    *_, fixed = again.tick()
    assert (fixed["action"], fixed["sha"], fixed["adopted"], fixed["repairs"], fixed["base"]) == (
        "build", made, True, sha, two["sha"]), fixed
    assert records.made[-1] != made and len(records.started) == 1


def test_a_staged_batch_none_of_whose_members_merges_onto_the_chain_s_top_is_dropped_and_built_again(env):
    loop, records, sha, _, _ = _records_loop(env)
    env.put(2, {"b.txt": "2\n"})
    (two,) = loop.tick()
    records.verdicts["regen-1"] = RecordsRun(PASS)
    loop.world.build = lambda prs, base, main=None: (base, [], [])       # #1 merges onto main's chain's top no more
    judged, drop = loop.tick()
    assert (drop["action"], drop["sha"], drop["why"]) == (
        "drop", sha, f"no member of it merges onto {two['sha'][:12]}, main's chain's top: its members are built again")
    assert branch(sha) not in env.queued() and records.committed == []


def test_a_dropped_candidate_s_records_run_is_cancelled_while_it_runs_and_one_that_ended_is_left(env):
    by: dict[tuple[int, str], str] = {}
    env.labels.marked_by = lambda n, h: by.get((n, h), f"lane-{n}")
    loop, records, sha, one, _ = _records_loop(env)
    env.o.gh.ready.discard((1, one))
    by[(1, one)] = "ci"                                              # its ready mark withdrawn by another: dropped
    (drop,) = loop.tick()
    assert (drop["action"], drop["sha"], drop["records_cancelled"]) == ("drop", sha, ["regen-1"])
    assert env.cancelled == [(["regen-1"], f"vqueue: candidate {sha[:12]} was dropped: #1 isn't taken now: its ready mark was withdrawn by ci")]
    assert loop.tick() == [] and len(env.cancelled) == 1
    two = env.put(2, {"pkg/B.lean": "theorem b : True := trivial\n", "pkg/lean-audit.json": '{"records": 3}\n'})
    built, started = loop.tick()
    assert (built["members"], started["record"]["ids"]) == ([2], ["regen-2"])
    records.verdicts["regen-2"] = RecordsRun(FAIL, "it failed: regen-2 failed rc=1")
    env.o.gh.ready.discard((2, two))                                 # by its owner: it stays in flight
    judged, drop = loop.tick()
    assert (judged["why"], drop["action"], len(env.cancelled)) == ("judged fail", "drop", 1) and "records_cancelled" not in drop
    assert env.withdrawn == []                                       # its refusal ejects no member its owner withdrew


def test_a_records_run_still_launching_when_its_candidate_is_dropped_is_cancelled_once_it_printed_its_run_ids(env):
    records = Records(env.tmp / "clone")
    records.launching = True
    by: dict[tuple[int, str], str] = {}
    env.labels.marked_by = lambda n, h: by.get((n, h), f"lane-{n}")
    loop, _, sha, one, (_, started) = _records_loop(env, records)
    env.o.gh.ready.discard((1, one))
    by[(1, one)] = "ci"                                              # its ready mark withdrawn by another: dropped
    (drop,) = loop.tick()
    assert (drop["action"], drop["records_launching"], env.cancelled) == ("drop", started["record"], [])
    assert "records_cancelled" not in drop and loop.tick() == []     # still launching
    records.registered[sha] = ["regen-9"]
    (settled,) = loop.tick()
    assert (settled["action"], settled["sha"], settled["dropped"], settled["records_cancelled"]) == (
        "regenerate", sha, True, ["regen-9"])
    assert env.cancelled == [(["regen-9"], f"vqueue: candidate {sha[:12]} was dropped: #1 isn't taken now: its ready mark was withdrawn by ci")]
    assert loop.tick() == [] and len(records.started) == 1


def test_without_vqueue_records_a_lean_candidate_s_check_starts_at_its_build(env):
    env.put(1, LEAN)
    (built,) = env.loop(vqueue.decide).tick()
    assert (built["enqueued"], "records_first" in built, _checks(env)) == (True, False, [f"ready:{built['sha']}"])


def test_a_candidate_waiting_for_its_records_run_read_by_a_loop_without_vqueue_records_has_its_check_enqueued(env):
    """--no-records on a log a loop with records wrote: the candidate is one whose check was never enqueued (list-failed),
    retried at once; its check enqueued, it waits no longer, so a loop with records again reads its verdict and lands it."""
    loop, records, sha, _, _ = _records_loop(env)
    loop.records = None
    (retry,) = loop.tick()
    assert (retry["action"], retry["sha"], retry["enqueued"], retry["mark"]) == ("retry", sha, True, "not enqueued"), retry
    assert _checks(env) == [f"ready:{sha}"] and len(records.started) == 1
    loop.records = records
    keys = env.listed(sha, {"lean-lock:pkg": ["pkg/A.lean", "pkg/lean-audit.json"]})
    env.result(keys["lean-lock:pkg"], "lean-lock:pkg", "pass")
    acts = loop.tick()
    assert "advance" in _acts(acts) and env.main() == sha and len(records.started) == 1, acts


def test_a_dry_run_starts_no_records_run_judges_none_and_says_what_it_would(env):
    env.put(1, LEAN)
    records = Records(env.tmp / "clone")
    dry = env.loop(vqueue.decide, clone="dry", records=records, dry_run=True)
    assert dry.tick() == [] and records.started == [] and env.lines() == []
    assert any("would build: Build(members=(1,), " in x and "records=True, after=None" in x for x in env.out), env.out
    loop = env.loop(vqueue.decide, records=records)
    loop.world.slim_driver = lambda: "drive"
    loop.tick()
    records.judge = lambda *a: pytest.fail("a dry run judges no records run")
    env.out.clear()
    assert dry.tick() == [] and len(records.started) == 1
    said = "\n".join(env.out)
    assert "(records run: pending) (its check waits for its records run)" in said and "would " not in said, said


@pytest.mark.parametrize("changes", [False, True])
def test_end_to_end_on_a_lock_run_its_records_commit_lands_naming_each_statement_change_it_makes(env, changes):
    """The real `vqueue_records` on a real Landing, faked only where it reaches a machine (the launch, the runs' states, their
    evidence and run dirs): a lock run runs on the Lean candidate's tree before its check and regenerates `pkg`'s records, and
    its records commit, a lock commit the merge gate vouches for, is checked and lands in the candidate's place.  When the regenerated records change guarantee T's statement, they land all the same, nobody rung or ejected, and
    the records commit names the change with T's owner in the candidate's lock, as the spec DM shows it once it lands."""
    from research.landing import LEAN_STEP, Landing
    from test_landing import _live_run, _steps_run

    main = env.main()
    lock = {"guarantees": {"T": {"owner": "@proofs"}}, "reads": {}}
    one = env.put(1, {"pkg/A.lean": "theorem a : True := trivial\n", "pkg/lean-audit.json": json.dumps(lock)})
    loop = env.loop(vqueue.decide, records=vqueue_records)
    launched, states, evidence = [], {}, []

    def launch(repo, commit, machine, agreement):
        launched.append((commit, machine))
        states[rid := f"r20261109-0000{len(launched):02d}-abcd"] = ("running", None, None)
        return [rid], ""

    d = Landing(loop.world.q, on="m1", launch=launch, states=lambda repo, ids: {i: states[i] for i in ids},
               post=lambda text, to=(): True, now=lambda: env.clock, cancel=lambda repo, ids, reason: None,
               evidence=lambda repo, ids: evidence)
    loop.world.q.store, loop.world.slim_driver = env.o.store, lambda: d
    built, started = loop.tick()
    sha, (rid,) = built["sha"], started["record"]["ids"]
    assert (built["base"], built["records_first"], started["record"]["tip"], launched) == (main, True, sha, [(sha, "m1")])
    assert loop.tick() == [] and _checks(env) == []
    regenerated = json.dumps({**lock, "reads": {"Pkg.M": "2"}}) + "\n"
    states[rid] = ("done", 0, None)
    evidence.append(_steps_run(env.tmp, rid, {LEAN_STEP: "passed"}))
    run, d.fetch, _ = _live_run(env.tmp, rid, sha, {"pkg": regenerated})
    audit = json.loads((run / M.AUDIT).read_text())
    review = ["guarantee T: changed", "  before:", "    theorem T : True"] if changes else ["definition Pkg.d (Pkg.M): new"]
    audit["packages"] = [{"package": "pkg", "regenerated": audit["regenerated"]["pkg"], "settled": True, "failures": [],
                          "review": review, "statement_changes": changes}]
    (run / M.AUDIT).write_text(json.dumps(audit))
    judged, fixed = loop.tick()
    new = fixed["sha"]
    assert (judged["why"], fixed["repairs"], fixed["enqueued"], _checks(env)) == ("judged pass", sha, True, [f"ready:{new}"])
    body = _git(env.o.origin, "log", "-1", "--format=%B", new)
    named = ["pkg: guarantee T: changed (@proofs)"] if changes else []
    assert body == lock_message("#1", rid, f"queue/{sha[:12]}", sha, [], ["pkg/lean-audit.json"], named).strip(), body
    assert ("\n  pkg: guarantee T: changed (@proofs)\n" in body) == changes and "before:" not in body, body
    assert M.regenerated_parent(loop.world.repo, new, env.o.store, tool="check", kinds=frozenset(), command=(), deferred=rid,
                                fetch=d.fetch) == rid
    (drop,) = loop.tick()
    keys = env.listed(new, {"lean-lock:pkg": ["pkg/A.lean", "pkg/lean-audit.json"]})
    env.result(keys["lean-lock:pkg"], "lean-lock:pkg", "pass")
    advance, on_main, landed = loop.tick()
    assert env.main() == new and _git(env.o.origin, "show", "main:pkg/lean-audit.json") == regenerated.strip()
    assert (landed["outcome"]["state"], landed["outcome"]["heads"], on_main["done"]) == (PASS, [one], True), landed
    assert _git(env.o.origin, "rev-parse", "main^") == sha and env.withdrawn == [] and env.posts == []


@pytest.mark.parametrize("flag, records", [((), vqueue_records), (("--no-records",), None)])
def test_the_command_runs_each_records_run_with_vqueue_records_and_none_with_no_records(monkeypatch, tmp_path, flag, records):
    import research.store.local as local

    made = []

    class Made:
        def __init__(self, world, log, **kw):
            made.append(kw)

        def tick(self):
            return []

    labels = type("Labels", (), dict.fromkeys(("ready", "before_cutoff", "marked_by", "marked_at", "prefetch")))()
    for mod, name, fake in ((L, "Loop", Made), (L, "StoreLabels", lambda store, log=None: labels),
                            (L, "_Clone", lambda *a, **k: type("Clone", (), {})()), (L, "World", lambda *a, **k: None),
                            (L, "Executor", lambda *a, **k: None), (L, "Transport", lambda *a, **k: None),
                            (Q, "remote_and_repo", lambda *a: ("origin", "o/r")), (Q, "GitHub", lambda slug: None),
                            (Q, "credential_args", lambda *a: ()), (local, "LocalStore", lambda root: None)):
        monkeypatch.setattr(mod, name, fake)
    assert L.main(["--dry-run", "--log", str(tmp_path / "log"), *flag]) == 0
    assert made[0]["records"] is records and made[0]["dry_run"]


def test_research_queue_vqueue_is_its_own_command_and_runs_the_run_beside_each_check_with_vqueue_slim(capsys):
    from research import vqueue_loop, vqueue_slim

    with pytest.raises(SystemExit) as e:
        Q.main(["vqueue", "--help"])
    said = " ".join(capsys.readouterr().out.split())
    assert e.value.code == 0 and all(o in said for o in ("--shadow", "--lean", "--pending-hours", "--no-slim", "--no-records"))
    assert vqueue_loop._slim() is vqueue_slim


def test_the_loop_s_batch_depth_and_lean_default_to_16_3_and_3_on_the_command_and_in_the_loop(capsys, tmp_path):
    with pytest.raises(SystemExit):
        Q.main(["vqueue", "--help"])
    said = " ".join(capsys.readouterr().out.split())
    assert all(f"{o} (default {n})" in said for o, n in (("holds", 16), ("its check", 3), ("take no cap (--pins-file)", 3))), said
    loop = Loop(None, tmp_path / "log")
    assert (loop.batch, loop.depth, loop.lean) == (L.BATCH, L.DEPTH, L.LEAN) == (16, 3, 3)


def test_merged_root_takes_the_codemod_s_worktrees_off_tmpdir_and_sweeps_one_a_killed_tick_left_there(tmp_path, monkeypatch):
    """`ported`'s codemod runs in a worktree a merge (`queue.merged_worktree`, under `queue.MERGED_ROOT`, else TMPDIR): on the
    control pod TMPDIR is a network volume, where each add and remove takes minutes.  `--merged-root` takes them to a local
    disk, as `research queue --preflight-root` takes drive's, and a tree a killed tick left there (its process gone) is
    swept."""
    import tempfile

    from test_queue import _move_main
    from test_queue_ported import _tooling

    tmp, fast, ran, vq = tmp_path / "tmpdir", tmp_path / "fast", tmp_path / "ran", tmp_path / "vq"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    monkeypatch.setattr(Q, "MERGED_ROOT", None)
    o = Origin(tmp_path)
    main = _move_main(o, {**_tooling(tmp_path / "argv", f"import os\nPath({str(ran)!r}).write_text(os.getcwd())\n"),
                          "Lib/A.lean": "import Init\n"})
    _git(tmp_path, "clone", "-q", str(o.origin), str(clone := tmp_path / "clone"))
    (left := fast / f"{Q.MERGED_PREFIX}999999999-x" / "tree").mkdir(parents=True)     # no such pid: Linux's cap is 2**22
    ported = []

    def tick(self):
        ported.append(Q.ported(self.world.repo, _git(self.world.repo, "rev-parse", f"{main}^{{tree}}")))
        return []

    monkeypatch.setattr(L.Loop, "tick", tick)
    assert L.main(["--repo", str(clone), "--remote", f"https://github.com/{Q.REPO}.git", "--store", str(tmp_path / "store"),
                   "--lock", str(vq / "lock"), "--log", str(vq / "decisions.jsonl"), "--state", str(vq / "state.json"),
                   "--merged-root", str(fast), "--once"]) == 0
    assert [p[1:] for p in ported] == [(["Lib/A.lean"], [], "")] and Path(ran.read_text()).is_relative_to(fast)
    assert not left.exists() and not list(fast.iterdir()) and not list(tmp.iterdir())


def test_the_loops_landings_post_through_it_on_its_clock_and_the_one_beside_each_check_takes_drive_tomls_on_and_warn_steps(
        env):
    """The loop's `Landing`s, the one its after-steps run on (`World.driver`) and the one beside each check and each records
    run (`World.slim_driver`), post through the loop and read its clock; the second launches on drive.toml's `on` at `main`
    and keeps its `warn_steps`, which no check that gates is judged on."""
    _main_to(env, {Q.DRIVE_FLAGS: 'on = ["m1", "m2"]\nwarn_steps = ["ratchet-gate"]\n'})
    loop = env.loop(policy)
    assert loop.tick() == []                                    # it fetched `main`
    w = loop.world
    d, s = w.driver(), w.slim_driver()
    assert (d.on, s.on, s.warn_steps) == ("", "m1,m2", ("ratchet-gate",))
    assert d.post is s.post is w.post and d.now is s.now is w.now


# -- the check that gates (`--gate check`)


#: the steps each of the three shards of a check that gates holds beside the preflights, in these tests: pytest's suites are
#: split over the last two
PLAN = (("flock-circuit-build", "lean-build", "lean-unit-cut", "lean-audit", "lean-suites"),
        ("pytest", "flock-rust-tests"), ("pytest", "circuit-check", "ratchet-gate"))
#: an auto plan's six shards of the same steps, pytest's suites split over three of them
PLAN6 = (("flock-circuit-build", "lean-build"), ("lean-unit-cut", "lean-audit", "lean-suites"),
         ("pytest", "flock-rust-tests"), ("pytest",), ("pytest", "circuit-check"), ("ratchet-gate",))
OLD, NEW = "tests/test_x.py::test_old", "tests/test_x.py::test_new"


def _row(*tests: str, beyond: bool = False) -> dict:
    """Suite pkg's row of a check's suites.json: failed on ``tests``, and with ``beyond`` on more than its tests."""
    return {"name": "pkg", "suite": "pkg", "status": "failed", "failed_tests": list(tests), "beyond_tests": beyond}


class Runs:
    """Launches, run states and evidence, faked under the real `research.vqueue_slim` on ``loop``'s own Landing, its machines
    `--on`'s: each launch is run `c-N`, running until `end` writes its run dir as a check's shard leaves it."""

    def __init__(self, env: Env, loop: Loop, monkeypatch, on: str = "m1,m2,m3"):
        monkeypatch.setattr(M, "required_steps", lambda *a, **k: {})
        monkeypatch.setattr(vqueue_slim, "left", lambda d, rid: self.state.get(rid, ("running",))[0] == "done")
        self.tmp, self.launches, self.state, self.dirs, self.cancelled = env.tmp / "runs", [], {}, [], []
        loop.world.on = on
        d = loop.world.slim_driver()
        d.launch_postland, d.states = self.launch, lambda repo, ids: {i: self.state[i] for i in ids}
        d.evidence = lambda repo, ids: [x for x in self.dirs if x.name in ids]
        d.cancel = lambda repo, ids, reason: self.cancelled.extend(ids)

    def launch(self, repo, commit, machine, steps, **kw):
        self.launches.append((commit, machine, steps, kw))
        self.state[rid := f"c-{len(self.launches)}"] = ("running", None, None)
        return [rid], ""

    def end(self, sha: str, shard: int = 0, step: str = "", log: str = "", rows: tuple[dict, ...] = (),
            also: dict[str, str | tuple[str, int]] | None = None,
            plan: tuple[tuple[str, ...], ...] = PLAN) -> None:
        """The shards of the newest check of ``sha`` ended, each running a step list of ``plan`` (`PLAN`'s three, unless
        given): each step passed but ``step`` in shard ``shard``, which failed with ``log`` and, of a step that runs suites,
        the suites.json ``rows``, and each step of ``also`` in the shard holding it, which failed with its log (and its exit, 1
        unless given)."""
        from research.landing import LIVE

        for k, n in enumerate([n for n, (c, *_) in enumerate(self.launches, 1) if c == sha][-len(plan):], 1):
            (d := self.tmp / f"c-{n}").mkdir(parents=True, exist_ok=True)
            bad = {**{s: x if isinstance(x, tuple) else (x, 1) for s, x in (also or {}).items() if s in plan[k - 1]},
                   **({step: (log, 1)} if k == shard else {})}
            recs = [{"step": x, "status": "failed", "rc": bad[x][1], "log": f"{x}.log"} if x in bad else
                    {"step": x, "status": "passed"} for x in (*PREFLIGHT, *plan[k - 1])]
            (d / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
            (d / LIVE).write_text(json.dumps({"schema": "check-live/v1", "state": "failed" if bad else "passed",
                                              "steps": [{"step": r["step"], "state": r["status"]} for r in recs]}))
            for s, (x, _) in bad.items():
                (d / f"{s}.log").write_text(x)
            if k == shard:
                (d / "suites.json").write_text(json.dumps(list(rows)))
            self.state[f"c-{n}"] = ("done", 1 if bad else 0, None)
            self.dirs += [d] if d not in self.dirs else []

    def fail(self, sha: str, shard: int, rows: tuple[dict, ...] = (), step: str = "", log: str = "", at: float = T0,
             rc: int | None = None, plan: tuple[tuple[str, ...], ...] = PLAN) -> str:
        """Shard ``shard`` of the newest check of ``sha`` has failed so far, its run ``rc`` None still running: pytest's
        suites ``rows`` (its suites.failed.json, each suite's log beside it last written at ``at``), or ``step``, which ended
        failed at ``at`` with ``log``; its state says so, as `landing.run_states` reads it.  The run's id."""
        from research.landing import LIVE, failing

        n = [n for n, (c, *_) in enumerate(self.launches, 1) if c == sha][-len(plan):][shard - 1]
        (d := self.tmp / f"c-{n}").mkdir(parents=True, exist_ok=True)
        recs = [*({"step": x, "status": "passed"} for x in PREFLIGHT),
                *([{"step": step, "status": "failed", "rc": 1, "log": f"{step}.log", "ended_utc": L.stamp(at)}] if step else [])]
        (d / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        if step:
            (d / f"{step}.log").write_text(log)
        (d / "suites.failed.json").write_text(json.dumps([{**{k: v for k, v in r.items() if k != "status"},
                                                           "log": f"{r['name']}.log"} for r in rows]))
        (d / "suites").mkdir(exist_ok=True)
        for r in rows:
            (written := d / "suites" / f"{r['name']}.log").write_text("FAILED\n")
            os.utime(written, (at, at))
        (d / LIVE).write_text(json.dumps({"schema": "check-live/v1", "state": "running", "steps": [
            {"step": r["step"], "state": r["status"]} for r in recs]}))
        self.state[f"c-{n}"] = ("running", None, None, failing(d)) if rc is None else ("done", rc, None)
        self.dirs += [d] if d not in self.dirs else []
        return f"c-{n}"


def _check(env: Env, monkeypatch, decide=vqueue.decide, **kw) -> tuple[Loop, Runs]:
    loop = env.loop(decide, slim=vqueue_slim, gate="check", **kw)
    return loop, Runs(env, loop, monkeypatch)


@pytest.mark.parametrize("graph2", ["pending", "fail", "pass"])
def test_under_gate_check_a_candidate_whose_sharded_check_passed_advances_whatever_graph2_says_and_the_two_are_compared(
        env, monkeypatch, graph2):
    """The check that gates is the after-landing run of the candidate, every step as three shards, a machine each; graph2's
    check is still enqueued, and its verdict, read in shadow, neither holds nor ejects: the candidate advances once every
    shard passed, and once both ended one line says whether they agree."""
    from research.landing import AFTER_LANDING

    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    loop, runs = _check(env, monkeypatch)
    below, built, started = loop.tick()
    sha = built["sha"]
    assert [(c, m, kw["shard"]) for c, m, _, kw in runs.launches] == [
        (c, f"m{k}", f"{k}/3") for c in (main, sha) for k in (1, 2, 3)]
    assert all(steps == ",".join(AFTER_LANDING) and kw["no_regenerate"] and not kw["agreement"]
               for _, _, steps, kw in runs.launches) and [kw["base"] for *_, kw in runs.launches[3:]] == [main] * 3
    assert started["record"]["gate"] == "check" and started["record"]["groups"] == [["c-4"], ["c-5"], ["c-6"]]
    assert env.graph2.enqueued() == [{"check": f"ready:{sha}", "priority": built["priority"], "graph": None, "schema": GRAPH_V2}]
    runs.end(main)
    if graph2 != "pending":
        keys = env.listed(sha, {"pytest:a": ["a.txt"]})
        env.result(keys["pytest:a"], "pytest:a", graph2)
    # its own shards still run, whatever graph2 says: a verdict of graph2's that came first is logged, for how long it took
    assert _acts(loop.tick()) == ["slim", *(["graph2"] if graph2 != "pending" else [])] and env.main() == main
    assert not env.withdrawn and not env.posts
    runs.end(sha)
    logged = loop.tick()
    assert _acts(logged) == ["slim", *(["compare"] if graph2 != "pending" else []), "advance", "on_main", "drop"]
    assert env.main() == sha and not env.withdrawn
    said = {"fail": ["disagree: graph2 fail (pytest:a), the check pass"], "pass": ["agree: graph2 and the check pass"]}.get(
        graph2, [])
    assert [r["why"] for r in env.lines() if r["action"] == "compare"] == said
    assert [x for x in env.out if "graph2 in shadow" in x] == [
        f"research queue vqueue: tick 3: {sha[:12]} graph2 in shadow: {why}" for why in said]
    assert loop.tick() == [] and len(runs.launches) == 6             # `main`'s check is the candidate's


def test_under_gate_check_a_step_one_shard_fails_fails_the_candidate_though_another_shard_and_graph2_passed_it(env, monkeypatch):
    """pytest's suites are split over two shards: one's pass doesn't answer for the other's failure.  The candidate fails
    on the step, its `Failed` naming the suite's failing test and the test's file, and its member is ejected, while
    graph2, in shadow, passed."""
    main = env.main()
    head = env.put(1, {"pkg/tests/test_x.py": "def test_new():\n    assert False\n"})
    loop, runs = _check(env, monkeypatch, policy)
    _, built, _ = loop.tick()
    sha = built["sha"]
    keys = env.listed(sha, {"pytest:a": ["pkg/tests/test_x.py"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    runs.end(main)
    runs.end(sha, 3, "pytest", rows=(_row(NEW),))
    logged = loop.tick()
    assert _acts(logged) == ["slim", "slim", "compare", "eject", "drop"] and env.main() == main
    assert logged[1]["record"]["verdict"]["failed"] == [
        {"task": "step pytest", "key": vqueue_slim.key(loop.world.repo, sha, "pytest"), "inputs": ["pkg/tests/test_x.py"],
         "base_key": vqueue_slim.key(loop.world.repo, main, "pytest"), "tests": [f"pkg::{NEW}"],
         "reads": sorted(vqueue_slim.suite_reads(loop.world.repo, sha, "pkg")[0]), "named": ["pkg/tests/test_x.py"],
         "kind": "own"}]
    assert "pytest" not in logged[1]["record"]["verdict"]["passed"]
    assert logged[2]["why"] == "disagree: graph2 pass, the check fail (step pytest)" and env.withdrawn == [(1, head)]


def test_under_gate_check_a_pytest_failure_is_inherited_by_its_tests_and_a_new_failing_test_in_the_step_is_the_member_s(
        env, monkeypatch):
    """`main` is red on one test of pytest.  A candidate failing that test alone inherits the failure and lands, one red
    rung; the next fails it and a new test in the same step, which the check of its base (the candidate `main` advanced
    to) passed, so its member is ejected: a red step hides no new failure."""
    main = env.main()
    env.put(1, {"1.txt": "1\n"})
    head = env.put(2, {"pkg/tests/test_x.py": "def test_new():\n    assert False\n"})
    loop, runs = _check(env, monkeypatch, batch=1, depth=1)
    _, built, _ = loop.tick()
    first = built["sha"]
    runs.end(main, 3, "pytest", rows=(_row(OLD),))
    runs.end(first, 3, "pytest", rows=(_row(OLD),))
    logged = loop.tick()
    assert _acts(logged) == ["slim", "slim", "red", "advance", "on_main", "drop", "build", "slim"] and env.main() == first
    second = logged[6]["sha"]
    runs.end(second, 3, "pytest", rows=(_row(OLD, NEW),))
    logged = loop.tick()
    assert _acts(logged) == ["slim", "eject", "drop"] and env.main() == first
    assert [f["tests"] for f in logged[1]["failed"]] == [[f"pkg::{NEW}", f"pkg::{OLD}"]] and env.withdrawn == [(2, head)]
    assert len(env.posts) == 2                                       # the red, and the ejected member's owner


@pytest.mark.parametrize("how", ["no test", "more than its tests", "a step with no test"])
def test_under_gate_check_a_failure_that_names_no_test_is_never_inherited(env, monkeypatch, how):
    """`main`'s check and the candidate's fail alike on what names no test: a suite that failed with none, one that failed
    for more than the test it names (a timeout, a read outside its inputs), a step whose log names none.  Nothing says the
    two failures are one, so the candidate's is its own: run again once, as a lone member's is, then ejected."""
    main = env.main()
    env.put(1, {"1.txt": "1\n"})
    loop, runs = _check(env, monkeypatch, batch=1, depth=1)
    failed = {"no test": (3, "pytest", "Killed\n", (_row(beyond=True),)),
              "more than its tests": (3, "pytest", "", (_row(OLD, beyond=True),)),
              "a step with no test": (1, "lean-build", "error: build failed\n", ())}[how]
    _, built, _ = loop.tick()
    sha = built["sha"]
    runs.end(main, *failed)
    runs.end(sha, *failed)
    logged = loop.tick()
    assert _acts(logged) == ["slim", "slim", "rerun", "slim"] and logged[3]["why"] == "started again: its failure is rerun"
    (mine,), (theirs,) = logged[1]["record"]["verdict"]["failed"], logged[0]["record"]["verdict"]["failed"]
    assert mine["task"] == theirs["task"] and mine["key"] != theirs["key"]
    assert set(mine["tests"]) - set(theirs["tests"]) == {
        ("" if how == "a step with no test" else "pkg::") + vqueue_slim.unnamed(mine["key"])}
    runs.end(sha, *failed)
    assert _acts(loop.tick()) == ["slim", "eject", "drop"] and env.main() == main and len(env.withdrawn) == 1


def test_under_gate_check_a_suite_failed_while_the_shards_run_ejects_its_lone_suspect_at_once_and_stops_them(env, monkeypatch):
    """pytest's suite pkg fails in one shard while every shard still runs: the candidate fails now, its failure read as at its
    end.  The member whose files meet the failing test's is ejected, told which test failed; the candidate's shards are
    cancelled, its drop's line says when the suite's failure was written and when it was dropped, and the other member is
    built again on `main` in the same tick."""
    main = env.main()
    head = env.put(1, {"pkg/tests/test_x.py": "def test_new():\n    assert False\n"})
    env.put(2, {"a.txt": "1\n"})
    loop, runs = _check(env, monkeypatch)
    _, built, _ = loop.tick()
    sha = built["sha"]
    runs.end(main)
    runs.fail(sha, 2, rows=(_row(NEW),), at=T0 + 60)
    env.clock = T0 + 300
    logged = loop.tick()
    assert _acts(logged) == ["slim", "slim", "eject", "drop", "build", "slim"] and env.main() == main
    judged, eject, drop, rebuilt = logged[1:5]
    assert "verdict" not in judged["record"] and judged["record"]["early"]["failed"] == [
        {"task": "step pytest", "key": vqueue_slim.key(loop.world.repo, sha, "pytest"), "inputs": ["pkg/tests/test_x.py"],
         "base_key": vqueue_slim.key(loop.world.repo, main, "pytest"), "tests": [f"pkg::{NEW}"],
         "reads": sorted(vqueue_slim.suite_reads(loop.world.repo, sha, "pkg")[0]), "named": ["pkg/tests/test_x.py"],
         "kind": "own"}]
    assert env.withdrawn == [(1, head)] and f"failing pkg::{NEW}" in eject["text"]
    assert drop["sha"] == sha and drop["outcome"]["state"] == FAIL and drop["beside_cancelled"] == ["c-4", "c-5", "c-6"]
    assert drop["early_red"] == {"failed_at": L.stamp(T0 + 60), "dropped_at": L.stamp(T0 + 300)}
    assert rebuilt["members"] == [2] and rebuilt["base"] == main


@pytest.mark.parametrize("killed", [True, False])
def test_under_gate_check_a_step_killed_while_the_shards_run_is_no_early_red(env, monkeypatch, killed):
    """lean-audit fails in one shard while the others run.  Its log saying a process of its own was killed (`KILLED_SAYS`,
    the kernel replay's exit 137) is what the reading at the check's end takes as infra, so the early reading waits for the
    end; a failure that says no such thing is an early red, and the lone member's is run again."""
    from research.landing import KILLED_SAYS

    main = env.main()
    env.put(1, {"1.txt": "1\n"})
    loop, runs = _check(env, monkeypatch, batch=1, depth=1)
    _, built, _ = loop.tick()
    sha = built["sha"]
    runs.end(main)
    log = f"replay of Security: {KILLED_SAYS} (exit 137)\n" if killed else "error: security/Foo.lean:3:0: type mismatch\n"
    runs.fail(sha, 1, step="lean-audit", log=log, at=T0 + 60)
    env.clock = T0 + 300
    logged = loop.tick()
    early = [x for x in logged if x.get("sha") == sha and "early" in (x.get("record") or {})]
    if killed:
        assert not {"rerun", "eject", "drop"} & {*_acts(logged)} and not early and not env.cancelled
    else:
        assert early and {"rerun", "eject", "drop"} & {*_acts(logged)}


def test_under_gate_check_an_early_red_is_compared_with_nothing_and_its_drop_cancels_graph2_s_check(env, monkeypatch):
    """graph2's check, in shadow, ended before the check that gates, which then fails a suite while its other shards run: that
    check hasn't ended, so graph2's verdict is logged as the one that came first and no line compares it with the fail so
    far, and the drop cancels graph2's check, as every drop of a check that hasn't ended in a pass or a fail does."""
    main = env.main()
    env.put(1, {"pkg/tests/test_x.py": "def test_new():\n    assert False\n"})
    loop, runs = _check(env, monkeypatch)
    script = env.graph2_script(loop, {"state": FAIL, "pending": 0})
    _, built, _ = loop.tick()
    sha = built["sha"]
    runs.end(main)
    keys = env.listed(sha, {"pytest:a": ["pkg/tests/test_x.py"]})
    env.result(keys["pytest:a"], "pytest:a", "fail")
    runs.fail(sha, 2, rows=(_row(NEW),), at=T0 + 60)
    env.clock = T0 + 300
    logged = loop.tick()
    assert _acts(logged) == ["slim", "slim", "graph2", "eject", "drop"]
    first, drop = logged[2], logged[-1]
    assert first["why"] == "graph2, in shadow, ended before the check: fail (pytest:a)" and first["graph2"] == FAIL
    assert "early_red" in drop and (drop["cancel"], drop["graph2"]) == ("cancelled", FAIL) and "follow" not in drop
    assert Path(f"{script}.cancel").read_text() == f"ready:{sha}\n"
    env.clock += 600
    loop.tick()
    assert not [r for r in env.lines() if r["action"] == "compare"]


@pytest.mark.parametrize("main_said", [True, False])
def test_under_gate_check_a_red_above_a_running_candidate_drops_the_stack_above_it_once_its_base_or_main_has_said(
        env, monkeypatch, main_said):
    """Four candidates stacked, one PR each: the third one's check fails a suite while those below it still run.  The
    failing test meets only its own member's files, so `main`'s run stands for its base's (`vqueue.Failed.main_tests`): once
    that passed, it is decided at once; while `main`'s runs too, it waits, and in the tick its base's check passes, while the
    lowest's still runs, it is decided.  Its member, meeting the failing test, is ejected, the one above it is dropped as a
    base failure, each one's shards cancelled, and that one's PR is built again on its base in the same tick."""
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "1\n"})
    head = env.put(3, {"pkg/tests/test_x.py": "def test_new():\n    assert False\n"})
    env.put(4, {"c.txt": "1\n"})
    seen: list[Snapshot] = []
    loop, runs = _check(env, monkeypatch, lambda s: seen.append(s) or vqueue.decide(s), batch=1, depth=4)
    loop.tick()
    two, three, four = (loop.tick()[0]["sha"] for _ in range(3))
    if main_said:
        runs.end(main)
    runs.fail(three, 1, rows=(_row(NEW),))
    if not main_said:
        assert not {"eject", "drop"} & {*_acts(loop.tick())} and not env.withdrawn and not env.cancelled
        runs.end(two)
    logged = loop.tick()
    failed = {c.sha: c for c in seen[-1].candidates}[three].verdict.failed
    assert {f.main_tests for f in failed} == {frozenset() if main_said else None}       # not its base's word, once that said
    assert _acts(logged)[-5:] == ["eject", "drop", "drop", "build", "slim"] and env.withdrawn == [(3, head)]
    failed, above, rebuilt = logged[-4:-1]
    assert (failed["sha"], failed["outcome"]["state"], failed["beside_cancelled"]) == (three, FAIL, ["c-10", "c-11", "c-12"])
    assert (above["sha"], above["why"], above["beside_cancelled"]) == (four, "its base failed", ["c-13", "c-14", "c-15"])
    assert (rebuilt["base"], rebuilt["members"], rebuilt["replaces"]) == (two, [4], four)
    assert "early_red" in failed and "early_red" not in above


#: `PLAN`'s shards of a check that takes lean-audit from its base's: every step but lean-audit
FREE = tuple(tuple(s for s in p if s != "lean-audit") for p in PLAN)
#: lean-audit's log of a package that failed its build: the error its culprit's Lean file names
AUDIT = ("AUDIT /w/lean-audit-scratch-1-x/Security/Proofs: FAIL; built WITHOUT a sandbox\n  build: exit 1\n"
         "    error: ../Proofs/A.lean:100:2: Tactic `rfl` failed\n")


def _steps(runs: Runs, sha: str) -> list[str]:
    """The steps each launch of a shard of ``sha``'s checks was sent, in launch order."""
    return [steps for c, _, steps, _ in runs.launches if c == sha]


def test_under_gate_check_a_candidate_changing_no_lean_input_takes_lean_audit_from_its_base_s_check(env, monkeypatch):
    """Three candidates stacked, one PR each.  The lowest, on `main`, runs every step.  The middle one's member changes
    check's `common.py`, which every audit's key holds, so it runs lean-audit too, though its base's check does.  The top
    one's changes what no audit's key and no records key holds, check's `slot.py`, a test of the Lean check and a README of
    the Lean tree: its check runs every step but lean-audit, its record saying whose verdict on it it takes
    (``lean_audit_from``)."""
    from research.landing import AFTER_LANDING

    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    common = env.put(2, {"tools/verity/check/common.py": "# the auditor imports this\n"})
    env.put(3, {"tools/verity/check/slot.py": "# slots\n", "tools/verity/lean/tests/test_cache.py": "# a test\n",
                "security/README.md": "# the Lean tree\n"})
    loop, runs = _check(env, monkeypatch, batch=1, depth=3)
    _, built, _ = loop.tick()
    one, (two, three) = built["sha"], (loop.tick()[0]["sha"] for _ in range(2))
    every, free = ",".join(AFTER_LANDING), ",".join(s for s in AFTER_LANDING if s != "lean-audit")
    assert [_steps(runs, c) for c in (main, one, two, three)] == [[every] * 3] * 3 + [[free] * 3]
    started = {r["sha"]: r for r in env.lines() if r["action"] == "slim" and r["why"] == "started"}
    assert [started[c]["record"].get("lean_audit_from") for c in (one, two, three)] == [None, None, two]
    assert loop.world._changed(main, common) == {"tools/verity/check/common.py"}


def test_under_gate_check_a_candidate_changing_only_the_task_graph_s_contract_audits_itself(env, monkeypatch):
    """The task graph's contract is in every audit's key (`lean_audit.CONTRACT`): the candidate on a Lean one in flight
    whose member changes only it runs lean-audit itself, its record naming no check whose verdict it takes."""
    from research.landing import AFTER_LANDING

    env.put(1, {"Security/Proofs/A.lean": "-- a\n"})
    env.put(2, {"tools/verity/graph/verity_graph/contract.py": "CONTRACT = 2\n"})
    loop, runs = _check(env, monkeypatch, batch=1, depth=2)
    _, built, _ = loop.tick()
    lean, top = built["sha"], loop.tick()[0]["sha"]
    assert [_steps(runs, c) for c in (lean, top)] == [[",".join(AFTER_LANDING)] * 3] * 2
    started = {r["sha"]: r for r in env.lines() if r["action"] == "slim" and r["why"] == "started"}
    assert started[top]["base"] == lean and started[top]["record"].get("lean_audit_from") is None


@pytest.mark.parametrize("path, takes", [("verity/protocols/pouw/verifier/lean/check/models.json", False),
                                         ("verity/protocols/pouw/verifier/lean/models.json", True)])
def test_under_gate_check_a_candidate_changing_only_a_lean_package_s_file_audits_itself_and_one_beside_it_does_not(
        env, monkeypatch, path, takes):
    """An audit reads every file of its package (`lean_audit.inputs`), one a module reads with `include_str` among them, as
    pouw's `check/models.json`: the candidate on a Lean one in flight, which adds that package, whose member changes only
    such a file runs lean-audit itself, its record naming no check whose verdict it takes; one whose member changes only a
    file beside the package's directory takes its base's."""
    from research.landing import AFTER_LANDING

    pkg = "verity/protocols/pouw/verifier/lean/check"
    env.put(1, {f"{pkg}/lakefile.toml": 'name = "PouwCheck"\n', f"{pkg}/PouwCheck/Credit.lean": "-- a\n"})
    env.put(2, {path: "{}\n"})
    loop, runs = _check(env, monkeypatch, batch=1, depth=2)
    _, built, _ = loop.tick()
    lean, top = built["sha"], loop.tick()[0]["sha"]
    every, free = ",".join(AFTER_LANDING), ",".join(s for s in AFTER_LANDING if s != "lean-audit")
    assert [_steps(runs, c) for c in (lean, top)] == [[every] * 3, [free if takes else every] * 3]
    started = {r["sha"]: r for r in env.lines() if r["action"] == "slim" and r["why"] == "started"}
    assert started[top]["base"] == lean and started[top]["record"].get("lean_audit_from") == (lean if takes else None)


@pytest.mark.parametrize("path, takes", [("verity/protocols/pouw/verifier/lean/check/models.json", False),
                                         ("verity/protocols/pouw/verifier/lean/check/README.md", True)])
def test_under_gate_check_a_candidate_deleting_only_a_file_an_audit_reads_audits_itself(env, monkeypatch, path, takes):
    """The candidate's tree no longer lists a file its member deletes, and the audit's declaration says of a path what it
    says of a file there: the member's change is judged in that tree's files with the paths it changes, so the candidate on
    a Lean one in flight whose member only deletes a file of a Lean package that an audit reads runs lean-audit itself; one
    whose member only deletes the package's README takes its base's."""
    from research.landing import AFTER_LANDING

    pkg = "verity/protocols/pouw/verifier/lean/check"
    _main_to(env, {f"{pkg}/lakefile.toml": 'name = "PouwCheck"\n', f"{pkg}/models.json": "{}\n", f"{pkg}/README.md": "# pouw\n"})
    env.put(1, {f"{pkg}/PouwCheck/Credit.lean": "-- a\n"})
    env.put(2, {path: None})
    loop, runs = _check(env, monkeypatch, batch=1, depth=2)
    _, built, _ = loop.tick()
    lean, top = built["sha"], loop.tick()[0]["sha"]
    every, free = ",".join(AFTER_LANDING), ",".join(s for s in AFTER_LANDING if s != "lean-audit")
    assert [_steps(runs, c) for c in (lean, top)] == [[every] * 3, [free if takes else every] * 3]
    started = {r["sha"]: r for r in env.lines() if r["action"] == "slim" and r["why"] == "started"}
    assert started[top]["base"] == lean and started[top]["record"].get("lean_audit_from") == (lean if takes else None)
    assert path not in loop.world._tree(top)


@pytest.mark.parametrize("first", ["its base's check", "its own"])
def test_under_gate_check_a_candidate_that_took_lean_audit_from_its_base_passes_only_on_its_base_s_verdict_on_it(
        env, monkeypatch, first):
    """The top candidate's check runs every step but lean-audit.  Its own check passed while its base's runs: pending, though
    every step it ran passed.  Its base's check passed: the base advances, the top one still pending in that tick, and then
    the top one, judged on the base's verdict on lean-audit once its base is `main`, as recorded, whichever check ended
    first; its record keeps lean-audit as passed, so the run beside `main`, which it is now, answers for it."""
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "1\n"})
    seen: list[Snapshot] = []
    loop, runs = _check(env, monkeypatch, lambda s: seen.append(s) or vqueue.decide(s), batch=1, depth=2)
    _, built, _ = loop.tick()
    lower, upper = built["sha"], loop.tick()[0]["sha"]
    runs.end(main)
    if first == "its own":
        runs.end(upper, plan=FREE)
        assert loop.tick() and env.main() == main
        assert {c.sha: c.verdict.state for c in seen[-1].candidates} == {lower: PENDING, upper: PENDING}
    runs.end(lower)
    assert "advance" in _acts(loop.tick()) and env.main() == lower
    assert {c.sha: c.verdict.state for c in seen[-1].candidates}[upper] == PENDING
    if first != "its own":
        runs.end(upper, plan=FREE)
    logged = loop.tick()
    assert "advance" in _acts(logged) and env.main() == upper
    rec = [r for r in env.lines() if r["action"] == "slim" and r["sha"] == upper][-1]["record"]
    assert rec["lean_audit_from"] == lower and "lean-audit" in rec["verdict"]["passed"]
    assert loop.tick() == [] and len(runs.launches) == 9                # `main`'s check is the top candidate's


@pytest.mark.parametrize("red", ["lean-audit", "pytest"])
def test_under_gate_check_a_candidate_that_took_lean_audit_from_its_base_drops_with_it(env, monkeypatch, red):
    """The top candidate's own check passed; its base's failed, on lean-audit, which the top one took from it, or on
    pytest.  Either way the top one reads lean-audit as pending while its base is in flight, nothing taken into its record,
    and drops as its base failed, with no outcome, so nobody of its own is blamed or capped; its member, built again on
    `main`, runs lean-audit itself."""
    from research.landing import AFTER_LANDING

    main = env.main()
    head = env.put(1, {"Security/Proofs/A.lean": "-- a\n"} if red == "lean-audit" else
                   {"pkg/tests/test_x.py": "def test_new():\n    assert False\n"})
    env.put(2, {"b.txt": "1\n"})
    seen: list[Snapshot] = []
    loop, runs = _check(env, monkeypatch, lambda s: seen.append(s) or vqueue.decide(s), batch=1, depth=2)
    _, built, _ = loop.tick()
    lower, upper = built["sha"], loop.tick()[0]["sha"]
    runs.end(main)
    runs.end(upper, plan=FREE)
    if red == "lean-audit":
        runs.end(lower, 1, "lean-audit", log=AUDIT)
    else:
        runs.end(lower, 3, "pytest", rows=(_row(NEW),))
    logged = loop.tick()
    assert {c.sha: c.verdict for c in seen[-1].candidates}[upper] == Verdict(PENDING)
    assert not any(vqueue_slim.answers(r["record"]) for r in env.lines() if r["action"] == "slim" and r["sha"] == upper)
    dropped = {r["sha"]: r for r in logged if r["action"] == "drop"}
    assert (dropped[upper]["why"], dropped[upper]["outcome"], dropped[upper]["failed"]) == ("its base failed", None, [])
    assert lower in dropped and env.withdrawn == [(1, head)]
    rebuilt = next(r for r in logged if r["action"] == "build")
    assert (rebuilt["base"], rebuilt["members"]) == (main, [2])
    assert _steps(runs, upper) == [",".join(s for s in AFTER_LANDING if s != "lean-audit")] * 3
    assert _steps(runs, rebuilt["sha"]) == [",".join(AFTER_LANDING)] * 3


def test_under_gate_check_lean_audit_is_taken_from_the_candidate_that_ran_it_through_candidates_that_took_it(
        env, monkeypatch):
    """Four candidates stacked, one PR each.  The lowest one's member changes a Lean file, so its check runs lean-audit; the
    second's and the third's change no Lean input.  The second takes lean-audit from the lowest, and so does the third,
    built on the second, which ran none: its record and its launch's line name the lowest, whose check ran it, not its
    base.  The fourth's member changes a Lean file, so it runs lean-audit itself, though its base took it."""
    from research.landing import AFTER_LANDING

    env.put(1, {"Security/Proofs/A.lean": "-- a\n"})
    env.put(2, {"b.txt": "1\n"})
    env.put(3, {"c.txt": "1\n"})
    env.put(4, {"Security/Proofs/B.lean": "-- b\n"})
    loop, runs = _check(env, monkeypatch, batch=1, depth=4)
    _, built, _ = loop.tick()
    lean, (a, b, top) = built["sha"], (loop.tick()[0]["sha"] for _ in range(3))
    every, free = ",".join(AFTER_LANDING), ",".join(s for s in AFTER_LANDING if s != "lean-audit")
    assert [_steps(runs, c) for c in (lean, a, b, top)] == [[every] * 3, [free] * 3, [free] * 3, [every] * 3]
    started = {r["sha"]: r for r in env.lines() if r["action"] == "slim" and r["why"] == "started"}
    assert [started[c]["record"].get("lean_audit_from") for c in (lean, a, b, top)] == [None, lean, lean, None]
    assert any(f"as candidate {b[:12]}'s check, lean-audit {lean[:12]}'s: " in x for x in env.out), env.out


@pytest.mark.parametrize("says", ["passed", "failed"])
def test_under_gate_check_a_candidate_that_took_lean_audit_through_its_base_follows_the_check_that_ran_it(
        env, monkeypatch, says):
    """Three candidates stacked, one PR each: the lowest one's member changes a Lean file, the other two's none, so the top
    one takes lean-audit from the lowest one's check, as its base did.  While that check runs, the top one is pending,
    though every step its own check ran passed, and so is its base.  Where that check passes lean-audit, the lowest lands,
    and the two above, pending while it was in flight, land together in the next tick, the top one's record keeping
    lean-audit as passed.  Where it fails it, the lowest one is ejected, and the two
    above, still pending, nothing taken into their records, drop as their base failed, with no outcome, so nobody of theirs
    is blamed or capped."""
    main = env.main()
    head = env.put(1, {"Security/Proofs/A.lean": "-- a\n"})
    env.put(2, {"b.txt": "1\n"})
    env.put(3, {"c.txt": "1\n"})
    seen: list[Snapshot] = []
    loop, runs = _check(env, monkeypatch, lambda s: seen.append(s) or vqueue.decide(s), batch=1, depth=3)
    _, built, _ = loop.tick()
    lean, (a, b) = built["sha"], (loop.tick()[0]["sha"] for _ in range(2))
    started = {r["sha"]: r for r in env.lines() if r["action"] == "slim" and r["why"] == "started"}
    assert [started[c]["record"].get("lean_audit_from") for c in (a, b)] == [lean, lean]
    runs.end(main)
    runs.end(a, plan=FREE)
    runs.end(b, plan=FREE)
    assert "advance" not in _acts(loop.tick()) and env.main() == main
    assert {c.sha: c.verdict.state for c in seen[-1].candidates} == {lean: PENDING, a: PENDING, b: PENDING}
    if says == "passed":
        runs.end(lean)
        assert _acts(loop.tick()).count("advance") == 1 and env.main() == lean
        assert _acts(loop.tick()).count("advance") == 2 and env.main() == b
        rec = [r for r in env.lines() if r["action"] == "slim" and r["sha"] == b][-1]["record"]
        assert rec["lean_audit_from"] == lean and "lean-audit" in rec["verdict"]["passed"]
        return
    runs.end(lean, 1, "lean-audit", log=AUDIT)
    logged = loop.tick()
    assert {c.sha: c.verdict.state for c in seen[-1].candidates} == {lean: FAIL, a: PENDING, b: PENDING}
    assert not any(vqueue_slim.answers(r["record"]) for r in env.lines() if r["action"] == "slim" and r["sha"] in (a, b))
    dropped = {r["sha"]: r for r in logged if r["action"] == "drop"}
    assert [(dropped[x]["why"], dropped[x]["outcome"], dropped[x]["failed"]) for x in (a, b)] == [
        ("its base failed", None, [])] * 2
    assert "eject" in _acts(logged) and lean in dropped and env.withdrawn == [(1, head)]


def test_under_gate_check_a_candidate_takes_lean_audit_from_a_landed_one_by_the_run_beside_main_and_else_audits(
        env, monkeypatch):
    """The lowest candidate's member changes a Lean file, and the one on it, changing no Lean input, takes lean-audit from
    it; the lowest lands while that one's check runs.  The candidate built next, on that one, changing no Lean input
    either, takes lean-audit from the landed one too, whose verdict on it the run beside `main` keeps.  Once the log keeps
    no record of that run, as a loop that lost it, the run beside `main` starts again, and a candidate built then audits:
    nothing beside `main` keeps a verdict on lean-audit to take.  The two that took it are pending while that run runs,
    though their own checks passed, and land on its verdict in one tick, with the one that audited after them."""
    from research.landing import AFTER_LANDING

    main = env.main()
    env.put(1, {"Security/Proofs/A.lean": "-- a\n"})
    env.put(2, {"b.txt": "1\n"})
    loop, runs = _check(env, monkeypatch, batch=1, depth=3)
    _, built, _ = loop.tick()
    lean, a = built["sha"], loop.tick()[0]["sha"]
    runs.end(main)
    runs.end(lean)
    assert "advance" in _acts(loop.tick()) and env.main() == lean
    env.put(3, {"c.txt": "1\n"})
    b = next(r for r in loop.tick() if r["action"] == "build")["sha"]
    env.log.write_text("".join(json.dumps(r) + "\n" for r in env.lines() if not (r["action"] == "slim" and r["sha"] == lean)))
    env.put(4, {"d.txt": "1\n"})
    c = next(r for r in loop.tick() if r["action"] == "build")["sha"]
    every, free = ",".join(AFTER_LANDING), ",".join(s for s in AFTER_LANDING if s != "lean-audit")
    assert [_steps(runs, x) for x in (lean, a, b, c)] == [[every] * 6, [free] * 3, [free] * 3, [every] * 3]
    started = {r["sha"]: r for r in env.lines() if r["action"] == "slim" and r["why"] == "started"}
    assert [started[x]["record"].get("lean_audit_from") for x in (a, b, c)] == [lean, lean, None]
    runs.end(a, plan=FREE)
    runs.end(b, plan=FREE)
    runs.end(c)
    assert "advance" not in _acts(loop.tick()) and env.main() == lean
    runs.end(lean)
    assert _acts(loop.tick()).count("advance") == 3 and env.main() == c


def test_under_gate_check_a_candidate_that_took_lean_audit_is_pending_while_the_check_that_ran_it_is_rerun(env, monkeypatch):
    """The lowest candidate's member changes a Lean file, and the batch of two on it, changing no Lean input, takes
    lean-audit from it.  The lowest one's check fails lean-audit on an error in a file its member doesn't change, as a flaky
    audit might, so it runs again alone.  The batch's own check passed, and it reads lean-audit as pending throughout,
    nothing taken into its record, so that failure is never its own: the rerun passes, the lowest one lands, and the batch
    lands after it, nobody dropped, split or ejected."""
    main = env.main()
    env.put(1, {"Security/Proofs/B.lean": "-- b\n"})
    seen: list[Snapshot] = []
    loop, runs = _check(env, monkeypatch, lambda s: seen.append(s) or vqueue.decide(s), batch=2, depth=2)
    _, built, _ = loop.tick()
    lean = built["sha"]
    env.put(2, {"b.txt": "1\n"})
    env.put(3, {"c.txt": "1\n"})
    built = next(r for r in loop.tick() if r["action"] == "build")
    top = built["sha"]
    assert built["members"] == [2, 3] and built["base"] == lean
    runs.end(main)
    runs.end(top, plan=FREE)
    runs.end(lean, 1, "lean-audit", log=AUDIT)
    logged = loop.tick()
    assert any(r["sha"] == lean and r.get("why") == "started again: its failure is rerun" for r in logged), logged
    assert {c.sha: c.verdict.state for c in seen[-1].candidates} == {lean: FAIL, top: PENDING}
    runs.end(lean)
    logged += loop.tick()
    assert env.main() == lean and {c.sha: c.verdict.state for c in seen[-1].candidates}[top] == PENDING
    assert not any(vqueue_slim.answers(r["record"]) for r in env.lines() if r["action"] == "slim" and r["sha"] == top)
    logged += loop.tick()
    rec = [r for r in env.lines() if r["action"] == "slim" and r["sha"] == top][-1]["record"]
    assert env.main() == top and "lean-audit" in rec["verdict"]["passed"]
    assert [(r["action"], r["why"]) for r in logged if r["action"] in ("drop", "eject", "build")] == [("drop", "on main")] * 2
    assert env.withdrawn == []


def test_under_gate_check_on_main_red_on_lean_audit_a_candidate_that_took_it_keeps_its_failure_once_its_source_landed(
        env, monkeypatch):
    """`main`'s check fails lean-audit, and the lowest candidate's, whose member changes a Lean file, fails it the same way,
    inherited: it lands.  The one on it, changing no Lean input, took lean-audit from it: pending while the lowest is in
    flight, nothing taken into its record, then failed on the run beside `main`'s failure of it once the lowest has landed,
    inherited too, so it lands, its record keeping that failure.  The candidate on it, whose member changes a Lean file,
    fails lean-audit the same way: read against that record, the run beside `main` once it lands, its failure is inherited,
    not its own, and it lands in the same tick with no rerun or ejection."""
    main = env.main()
    env.put(1, {"Security/Proofs/A.lean": "-- a\n"})
    env.put(2, {"b.txt": "1\n"})
    seen: list[Snapshot] = []
    loop, runs = _check(env, monkeypatch, lambda s: seen.append(s) or vqueue.decide(s), batch=1, depth=3)
    _, built, _ = loop.tick()
    lean, a = built["sha"], loop.tick()[0]["sha"]
    env.put(3, {"Security/Proofs/C.lean": "-- c\n"})
    c = next(r for r in loop.tick() if r["action"] == "build")["sha"]
    for x in (main, lean, c):
        runs.end(x, 1, "lean-audit", log=AUDIT)
    runs.end(a, plan=FREE)
    logged = loop.tick()
    assert env.main() == lean and {x.sha: x.verdict.state for x in seen[-1].candidates}[a] == PENDING
    assert not any(vqueue_slim.answers(r["record"]) for r in env.lines() if r["action"] == "slim" and r["sha"] == a)
    logged += loop.tick()
    recs = {x: [r for r in env.lines() if r["action"] == "slim" and r["sha"] == x][-1]["record"] for x in (lean, a)}
    assert env.main() == c and recs[a]["lean_audit_from"] == lean
    assert recs[a]["verdict"]["failed"] == recs[lean]["verdict"]["failed"] and vqueue_slim.answers(recs[a])
    assert not {"rerun", "eject"} & set(_acts(logged)) and env.withdrawn == []


def test_under_gate_check_a_red_while_the_shards_run_with_several_suspects_is_bisected(env, monkeypatch):
    """Two suites fail in two shards while the check runs, each meeting one member's files: both are suspects, nobody is
    ejected, and the next batch takes half of them."""
    main = env.main()
    env.put(1, {"pkg/tests/test_x.py": "def test_new():\n    assert False\n"})
    env.put(2, {"other/tests/test_y.py": "def test_y():\n    assert False\n"})
    loop, runs = _check(env, monkeypatch)
    _, built, _ = loop.tick()
    runs.end(main)
    runs.fail(built["sha"], 2, rows=(_row(NEW),))
    runs.fail(built["sha"], 3, rows=({"name": "other", "suite": "other", "status": "failed",
                                      "failed_tests": ["tests/test_y.py::test_y"], "beyond_tests": False},))
    logged = loop.tick()
    assert _acts(logged) == ["slim", "slim", "drop", "build", "slim"] and not env.withdrawn
    assert logged[2]["outcome"]["suspects"] == [1, 2] and logged[3]["members"] == [1]


def test_under_gate_check_a_step_off_the_gate_or_a_killed_shard_fails_nothing_while_the_check_runs(env, monkeypatch):
    """circuit-check is off the merge gate and lean-audit a warn step: one shard ended failing both, which fails nothing
    while the others run; nor does a shard killed by the machine, which is infra once every shard ended, as before, nor a
    suite a stop cut short (failed beyond its tests alone) in a shard that runs.  A gating suite's failure in a shard that
    runs fails the candidate then, naming none of them."""
    loop, runs, main, sha = _off(env, monkeypatch, {"pkg/tests/test_x.py": "def test_new():\n    assert False\n"},
                                 decide=vqueue.decide)
    runs.end(main)
    runs.fail(sha, 3, step="circuit-check", log=_circuit(HEAD), rc=1)
    runs.fail(sha, 1, step="lean-audit", log="AUDIT security/Foo: FAIL\n", rc=137)
    runs.fail(sha, 2, rows=({"name": "pkg", "suite": "pkg", "status": "failed", "failed_tests": [], "beyond_tests": True},))
    assert [(r["action"], r["sha"]) for r in loop.tick()] == [("slim", main)] and not env.withdrawn and not env.cancelled
    runs.fail(sha, 2, rows=(_row(NEW),))
    logged = loop.tick()
    assert _acts(logged) == ["slim", "eject", "drop"] and [f["task"] for f in logged[2]["failed"]] == ["step pytest"]


@pytest.mark.parametrize("heavy, on", [("1\n", "big"), ("*\n", "big"), ("1\n", "")], ids=["big", "every", "off"])
def test_under_gate_check_a_heavy_candidate_s_shard_auditing_security_proofs_and_its_start_again_take_heavy_on(
        env, monkeypatch, heavy, on):
    """A candidate holding a heavy PR launches the shard its own ci.toml's plan gives security/Proofs' audit (here the second)
    on `--heavy-on`'s machines alone, its other shards and `main`'s check on their usual ones, and its check started again
    after infra (a shard killed) does the same; while the heavy file holds `*`, a candidate holding no PR it lists does so
    too, and `main`'s check as well; with no `--heavy-on`, every shard launches as before."""
    from test_landing import CI_TOML

    plan = [["pytest"], ["lean-audit:security/Proofs", "lean-audit:security/Properties"], ["lean-audit:*"]]
    main = _main_to(env, {"tools/verity/check/ci.toml": CI_TOML + f"[[plan]]\nshards = {json.dumps(plan)}\n"})
    env.put(1, {"a.txt": "1\n"})
    loop, runs = _check(env, monkeypatch)
    (env.tmp / "heavy").write_text(heavy)
    loop.world.heavy_file, loop.world.heavy_on = env.tmp / "heavy", on
    _, built, started = loop.tick()
    sha, usual = built["sha"], ["m1", "m2", "m3"]
    want = ["m1", "big", "m3"] if on else usual
    below = want if heavy == "*\n" else usual
    assert [(c, m) for c, m, _, _ in runs.launches] == [(main, m) for m in below] + [(sha, m) for m in want]
    assert started["record"].get("heavy_on") == (on or None)
    runs.end(main)
    runs.end(sha, 2, "pytest", "error: no\n")
    runs.state["c-5"] = ("done", 137, None)
    logged = loop.tick()
    assert [r["why"] for r in logged if r["action"] == "slim" and r["sha"] == sha] == ["judged infra", "started again"]
    assert [(c, m) for c, m, _, _ in runs.launches[6:]] == [(sha, m) for m in want]


def test_under_gate_check_a_run_started_while_graph2_gated_is_started_again_as_the_check(env, monkeypatch):
    """A candidate whose run beside its check judged only what graph2 doesn't check never advances on it once the check
    gates: the check starts in its place, beside `main` too."""
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    before = env.loop(vqueue.decide, slim=vqueue_slim)
    runs = Runs(env, before, monkeypatch)
    _, built, _ = before.tick()
    assert [steps for _, _, steps, _ in runs.launches] == ["ratchet-gate"] * 2 and "shard" not in runs.launches[0][3]
    loop, runs = _check(env, monkeypatch)
    below, started = loop.tick()
    assert (below["why"], started["why"]) == ("started again beside `main`", "started")
    assert [(c, kw["shard"]) for c, _, _, kw in runs.launches] == [(c, f"{k}/3") for c in (main, built["sha"]) for k in (1, 2, 3)]
    assert loop.tick() == [] and len(runs.launches) == 6


@pytest.mark.parametrize("regenerated", [True, False])
def test_under_gate_check_a_lean_candidate_s_sharded_check_starts_only_once_its_records_run_passed_on_its_records_commit(
        env, monkeypatch, regenerated):
    """The records run comes first under the check that gates too: while it runs, only `main`'s shards run and nothing is
    enqueued; once it passed, the three shards start, with graph2's check in shadow, on its records commit, or on the
    candidate as built when it regenerated nothing, and never on a candidate whose records were regenerated."""
    main = env.main()
    head = env.put(1, LEAN)
    records = Records(env.tmp / "clone")
    if not regenerated:
        records.files = {}
    loop, runs = _check(env, monkeypatch, records=records)
    below, built, regen = loop.tick()
    sha = built["sha"]
    assert (below["sha"], built["records_first"], built["enqueued"], regen["action"], regen["sha"]) == (
        main, True, False, "regenerate", sha)
    assert [(c, kw["shard"]) for c, _, _, kw in runs.launches] == [(main, f"{k}/3") for k in (1, 2, 3)] and _checks(env) == []
    assert loop.tick() == [] and len(runs.launches) == 3 and _checks(env) == []
    records.verdicts["regen-1"] = RecordsRun(PASS)
    judged, enqueued, started = loop.tick()
    target = enqueued["sha"]
    assert (judged["why"], enqueued["action"], enqueued["enqueued"], started["action"], started["sha"]) == (
        "judged pass", "build", True, "slim", target)
    assert (target != sha, enqueued.get("repairs")) == ((True, sha) if regenerated else (False, None))
    assert [(c, kw["shard"]) for c, _, _, kw in runs.launches[3:]] == [(target, f"{k}/3") for k in (1, 2, 3)]
    assert started["record"]["gate"] == "check" and _checks(env) == [f"ready:{target}"]
    runs.end(main)
    runs.end(target)
    for _ in range(3):
        loop.tick()
    assert env.main() == target and env.withdrawn == [] and len(runs.launches) == 6
    assert _git(env.o.origin, "rev-parse", "main^" if regenerated else "main") == sha
    assert [x for x in env.lines() if x["action"] == "advance"][0]["heads"] == [head]


def test_under_gate_check_a_batch_built_in_the_tick_that_commits_a_lean_one_takes_lean_audit_from_its_check_as_started(
        env, monkeypatch):
    """The tick that puts a staged Lean batch on main's chain builds the PRs with no Lean change on it (`Build.after`).  Its
    check started in that tick, so it is judged by none yet: the check of the batch built on it reads its record as
    started, and, its member changing no Lean input, takes its verdict on lean-audit rather than running the step."""
    from research.landing import AFTER_LANDING

    env.put(1, LEAN)
    loop, runs = _check(env, monkeypatch, records=(records := Records(env.tmp / "clone")))
    _, staged, _ = loop.tick()
    env.put(2, {"b.txt": "2\n"})
    records.verdicts["regen-1"] = RecordsRun(PASS)
    judged, fixed, began, built, started = loop.tick()
    on, top = fixed["sha"], built["sha"]
    assert (fixed["action"], fixed["repairs"], fixed["enqueued"], began["action"], began["sha"]) == (
        "build", staged["sha"], True, "slim", on), fixed
    assert (built["action"], built["members"], built["base"], built["enqueued"]) == ("build", [2], on, True), built
    assert (started["action"], started["sha"], started["record"].get("lean_audit_from")) == ("slim", top, on), started
    every, free = ",".join(AFTER_LANDING), ",".join(x for x in AFTER_LANDING if x != "lean-audit")
    assert [_steps(runs, c) for c in (on, top)] == [[every] * 3, [free] * 3]


#: circuit-check's targets in these tests: the head check's, which its sidecar's pins file pins; one no pins file pins, whose
#: statics hold brackets as real targets' do (a regex would read them as a class); one an allowance covers; one whose process died
HEAD, ROPE, DEAD, BIG = "NetworkHeadCheck_v2{L=6,T=32,COUNTS=0}", "RopeHead_v1{D=128,POS=[0,4096]}", "Callee_v1", "Big_v1{N=4}"
#: `main`'s tree for what a check fails off the merge gate: its policy takes circuit-check and ratchet-gate off the gate and
#: allows `DEAD`'s failure, drive.toml makes two steps warn steps, ci.toml gives each job but lean-audit an owner, the registry
#: holds the PRs' lanes, heads, who owns the head check's sidecar, whose pins file pins `HEAD`, and the owner of a fixture
#: naming `HEAD` that is no pins file
OFF_TREE = {"tools/verity/check/policy.toml": 'schema = "policy/v0"\noff_gate = ["circuit-check", "ratchet-gate"]\n'
                                              '[[allow]]\nid = "circuit-check lowering/dead Callee_v1"\nowner = "circuits"\n'
                                              'expires = "2030-06-01T00:00:00Z"\nreason = "red until its fix lands"\n',
            "tools/verity/check/drive.toml": 'warn_steps = ["flock-rust-tests", "lean-audit"]\n',
            "tools/verity/check/ci.toml": "".join(f'[[job]]\nname = "{j}"\nowner = "{o}"\n' for j, o in (
                ("pytest", "ci"), ("circuit-check", "circuits"), ("ratchet-gate", "vllm"), ("flock-rust-tests", "proofs"))),
            Q.REGISTRY.as_posix(): json.dumps({"handles": [
                *({"handle": h} for h in ("ci", "circuits", "vllm", "lane-1", "lane-2")),
                {"handle": "heads", "owns": ["verity/protocols/certifier/"]},
                {"handle": "artifacts", "owns": ["fixtures/"]}]}),
            "verity/protocols/certifier/verifier/head_check.circuit-check.json": json.dumps({HEAD: {"digest": "sha512:h"}}) + "\n",
            "fixtures/head.json": json.dumps({HEAD: "no pin"}) + "\n"}
#: a change meeting circuit-check's inputs (`queue.forces_cold`), and one that doesn't
CIRCUITS, OTHER = {"verity/circuits/rope.py": "# rope\n"}, {"a.txt": "1\n"}


def _circuit(*failed: str) -> str:
    """circuit-check's log failing each of ``failed`` (`CHECK/CODE TARGET`), as `checks.Findings.fail` writes it."""
    return ("".join(f"FAIL {x}{':' if x.startswith('circuit-check/') else ''} it failed\n" for x in failed)
            + "circuit-check: done\n")


def _ratchet(*failed: int) -> str:
    """ratchet-gate's log failing the merge of each PR of ``failed``, and passing #3's, as `ratchet_gate.render` writes it."""
    return ("ratchet gate: 3 change(s)\n" + "".join(f"  #{n} ({n:012x}): failed: grows p01.json\n" for n in failed)
            + "  #3 (000000000003): passed: grows nothing\n")


def _off(env: Env, monkeypatch, *changes: dict[str, str], decide=None, **kw) -> tuple[Loop, Runs, str, str]:
    """`main` at `OFF_TREE`, a PR of each of ``changes`` (#1, #2, ...), and a loop whose check gates, which built a candidate of
    them and holds it, deciding nothing more (unless ``decide``): (the loop, its runs, `main`, the candidate)."""
    _git(env.o.origin, "update-ref", "refs/heads/main", env.o.change(OFF_TREE))
    main = env.main()
    env.graph2.list_graph(f"ready:{main}", {"schema": GRAPH_V2, "commit": main, "tasks": []})
    for n, files in enumerate(changes, 1):
        env.put(n, files)
    loop = env.loop(decide or (lambda s: [] if s.candidates else policy(s)), slim=vqueue_slim, gate="check", **kw)
    loop.world.fetch()                                              # as a tick does before its slim driver reads drive.toml
    runs = Runs(env, loop, monkeypatch)
    _, built, _ = loop.tick()
    return loop, runs, main, built["sha"]


def _off_said(sha: str, members: str, step: str, base: str, *lines: str) -> str:
    """The ring of what candidate ``sha`` failed of ``step`` off the gate that ``base`` doesn't: each of ``lines``."""
    return (f"the verified queue's candidate ready:{sha} ({members}) failed {step}, off the merge gate, where its base "
            f"{base[:12]} doesn't: {'; '.join(f'`{x}`' for x in lines)}. Nothing is held or ejected for it: fix it forward")


def test_under_gate_check_a_red_ratchet_gate_rings_the_lane_of_each_member_whose_merge_it_fails_and_its_owner_once(
        env, monkeypatch):
    """ratchet-gate is off the merge gate (policy.toml's `off_gate` at the base): the candidate passes on its check all the
    same, and once that verdict is in, the lane of each member whose merge the gate fails (#1's and #2's, not #3's) and the
    step's owner in ci.toml at the candidate (vllm) are rung, with the gate's line for each, once a candidate and step.  A
    merge the gate names that is no member's (one below the base) rings nobody."""
    loop, runs, main, sha = _off(env, monkeypatch, OTHER, {"b.txt": "2\n"}, {"c.txt": "3\n"})
    runs.end(main)
    runs.end(sha, also={"ratchet-gate": _ratchet(1, 2, 9)})
    logged = loop.tick()
    assert _acts(logged) == ["slim", "slim", "ring"] and logged[1]["record"]["verdict"]["state"] == PASS
    ring = logged[2]
    assert (ring["sha"], ring["step"], ring["failures"], ring["to"], ring["done"], ring["posted"]) == (
        sha, "ratchet-gate", ["#1", "#2"], ["@lane-1", "@lane-2", "@vllm"], True, True)
    assert ring["why"] == "off the gate: ratchet-gate failed" and ring["base"] == main and ring["members"] == [1, 2, 3]
    assert env.posts == ["@lane-1 @lane-2 @vllm vqueue: " + _off_said(
        sha, "#1 #2 #3", "ratchet-gate", main, "#1 (000000000001): grows p01.json", "#2 (000000000002): grows p01.json")]
    assert loop.tick() == [] and len(env.posts) == 1
    assert [r["posting"] for r in env.lines() if r["action"] == "posting"] == ["ring"]


def test_under_gate_check_with_the_graph2_shadow_off_no_graph2_verdict_is_read_and_the_check_alone_judges(env, monkeypatch):
    """--no-graph2-shadow: the check gates as before, and no tick reads graph2's verdict of a candidate or of a followed
    drop (`World.verdict`, `World._verdict_of`), each of which lists every task of a check and pulls each pending task's
    results; no compare line comes of it."""
    loop, runs, main, sha = _off(env, monkeypatch, OTHER, graph2_shadow=False)

    def read(*_a, **_k):
        raise AssertionError("graph2's verdict was read in shadow")

    monkeypatch.setattr(loop.world, "verdict", read)
    monkeypatch.setattr(loop.world, "_verdict_of", read)
    runs.end(main)
    runs.end(sha)
    logged = loop.tick()
    assert _acts(logged) == ["slim", "slim"] and logged[1]["record"]["verdict"]["state"] == PASS
    assert loop.tick() == [] and not [r for r in env.lines() if r["action"] in ("compare", "graph2")]


@pytest.mark.parametrize("check", [PASS, FAIL])
def test_under_gate_check_with_the_graph2_shadow_off_a_candidate_dropped_after_its_check_ended_has_its_graph2_check_cancelled(
        env, monkeypatch, check):
    """--no-graph2-shadow follows nothing (`_followed`), so a drop follows nothing either: a candidate dropped once its check
    passed (`main` advanced to it) or failed (its member ejected) has its graph2 check cancelled at the drop, which keeps no
    `follow`, and no later tick follows it, however long graph2's check would have run."""
    main = env.main()
    env.put(1, {"pkg/tests/test_x.py": "def test_new():\n    assert False\n"} if check == FAIL else {"a.txt": "1\n"})
    loop, runs = _check(env, monkeypatch, policy if check == FAIL else vqueue.decide, graph2_shadow=False)
    script = env.graph2_script(loop, {"state": PENDING, "pending": 1})
    cancelled: list[str] = []
    cancel = loop.world.cancel
    monkeypatch.setattr(loop.world, "cancel", lambda sha, *a, **k: cancelled.append(sha) or cancel(sha, *a, **k))
    _, built, _ = loop.tick()
    sha = built["sha"]
    runs.end(main)
    runs.end(sha, *((3, "pytest") if check == FAIL else ()), **({"rows": (_row(NEW),)} if check == FAIL else {}))
    logged = loop.tick()
    assert _acts(logged)[-2:] == (["eject", "drop"] if check == FAIL else ["on_main", "drop"]) and env.main() == (
        main if check == FAIL else sha)
    drop = logged[-1]
    assert (drop["sha"], drop["cancel"], "follow" in drop, cancelled) == (sha, "cancelled", False, [sha])
    assert Path(f"{script}.cancel").read_text() == f"ready:{sha}\n"
    for at in (T0 + 600, T0 + L.FOLLOW_S):
        env.clock = at
        assert loop.tick() == [] and cancelled == [sha]
    assert not L.Past.read(env.log, 0).followed() and not [r for r in env.lines() if r["action"] == "compare"]


@pytest.mark.parametrize("base", ["passed", "failed"])
def test_under_gate_check_a_ratchet_gate_naming_no_merge_it_fails_rings_its_owner_unless_its_base_failed_it_so(
        env, monkeypatch, base):
    """A ratchet gate that fails naming no merge (it crashed) is any step's red: rung to the step's owner, unless the base's
    check failed it so too."""
    loop, runs, main, sha = _off(env, monkeypatch, OTHER)
    crashed = "Traceback (most recent call last):\nValueError: no\n"
    runs.end(main, also={"ratchet-gate": crashed} if base == "failed" else None)
    runs.end(sha, also={"ratchet-gate": crashed})
    rings = [(r["to"], r["failures"]) for r in loop.tick() if r["action"] == "ring"]
    assert rings == ([(["@vllm"], ["ratchet-gate"])] if base == "passed" else [])


@pytest.mark.parametrize("who", ["pins", "own pins", "unowned pins", "members", "stranger", "job"])
def test_under_gate_check_a_circuit_check_target_its_base_passed_rings_its_pins_owner_else_the_members_meeting_its_inputs(
        env, monkeypatch, who):
    """circuit-check is off the merge gate.  Of what the candidate's check fails of it, a target the base's check fails too
    (`main`'s own red), one whose process died and a failure an allowance at the candidate covers ring nobody.  A target the
    base passed rings the owner of each sidecar's pins file at the candidate pinning it (heads, the candidate's own sidecar
    too); one no pins file pins, or only one the registry gives no owner, rings the lane of each member whose change meets
    circuit-check's inputs (#2's, not #1's), and the step's owner in ci.toml (circuits) where one such member has no lane in
    the registry (#3) or none meets them."""
    second = {"own pins": {"verity/protocols/certifier/verifier/rope.circuit-check.json": json.dumps({ROPE: {}}) + "\n"},
              "unowned pins": {"verity/circuits/rope.circuit-check.json": json.dumps({ROPE: {}}) + "\n"},
              "job": {"b.txt": "2\n"}}.get(who, CIRCUITS)
    prs = [OTHER, second, *([{"verity/circuits/x.py": "# x\n"}] * (who == "stranger"))]
    loop, runs, main, sha = _off(env, monkeypatch, *prs)
    target = HEAD if who == "pins" else ROPE
    runs.end(main, also={"circuit-check": _circuit(f"lowering/pin-mismatch {HEAD if who != 'pins' else ROPE}")})
    runs.end(sha, also={"circuit-check": _circuit(f"lowering/pin-mismatch {HEAD}", f"lowering/pin-mismatch {ROPE}",
                                                  f"lowering/dead {DEAD}", f"circuit-check/killed {BIG}")})
    logged = loop.tick()
    assert _acts(logged) == ["slim", "slim", "ring"] and logged[1]["record"]["verdict"]["state"] == PASS
    to = {"pins": ["@heads"], "own pins": ["@heads"], "unowned pins": ["@lane-2"], "members": ["@lane-2"],
          "stranger": ["@lane-2", "@circuits"], "job": ["@circuits"]}[who]
    assert (logged[2]["step"], logged[2]["failures"], logged[2]["to"]) == (
        "circuit-check", [f"lowering/pin-mismatch {target}"], to)
    members = " ".join(f"#{n}" for n in range(1, len(prs) + 1))
    assert env.posts == [f"{' '.join(to)} vqueue: "
                         + _off_said(sha, members, "circuit-check", main, f"lowering/pin-mismatch {target} it failed")]
    assert loop.tick() == [] and len(env.posts) == 1


@pytest.mark.parametrize("base", ["passed", "failed", "killed", "passed, the candidate's killed"])
def test_under_gate_check_what_a_candidate_fails_off_the_gate_waits_for_its_bases_check_and_what_that_fails_rings_nobody(
        env, monkeypatch, base):
    """What the candidate's check fails off the gate is rung only once the base's check ended, and only what that check
    passed: a red the base shares (`main`'s own: a target it fails, in whatever way) rings nobody, and nor does any where
    either check didn't run the step to its end (its process killed), since what that check fails is unknown."""
    loop, runs, main, sha = _off(env, monkeypatch, CIRCUITS)
    failed = _circuit(f"lowering/pin-mismatch {HEAD}")
    runs.end(sha, also={"circuit-check": (failed, 137) if base.endswith("killed") and base != "killed" else failed})
    assert _acts(loop.tick()) == ["slim"] and env.posts == []             # the base's check still runs
    runs.end(main, also={"failed": {"circuit-check": _circuit(f"lowering/dead {HEAD}")},
                         "killed": {"circuit-check": (failed, 137)}}.get(base, {}))
    logged = loop.tick()
    assert _acts(logged) == ["slim", *(["ring"] if base == "passed" else [])]
    assert len(env.posts) == (base == "passed") and loop.tick() == []


def test_under_gate_check_a_red_off_the_gate_waits_for_the_run_beside_main_s_check_started_again_after_infra(env, monkeypatch):
    """The run beside `main`'s check that ended in infra errors runs again, and what a candidate fails off the gate is compared
    with that run once it ends, not with the one that ended so."""
    loop, runs, main, sha = _off(env, monkeypatch, CIRCUITS)
    runs.end(main, 3, "pytest", "error: no\n")
    runs.state["c-3"] = ("done", 137, None)
    runs.end(sha, also={"circuit-check": _circuit(f"lowering/pin-mismatch {HEAD}")})
    logged = loop.tick()
    assert [r["why"] for r in logged if r["sha"] == main] == ["judged infra beside `main`", "started again beside `main`"]
    assert "ring" not in _acts(logged)
    runs.end(main)
    assert [r["to"] for r in loop.tick() if r["action"] == "ring"] == [["@heads"]]


@pytest.mark.parametrize("verdict", [FAIL, INFRA])
def test_under_gate_check_a_red_off_the_gate_rings_beside_a_failed_verdict_and_not_until_an_infra_one_is_run_again(
        env, monkeypatch, verdict):
    """A candidate whose check failed on the gate is rung about what it failed off the gate too; one whose check ended in
    infra errors (its shard killed) rings nothing until its check runs again."""
    loop, runs, main, sha = _off(env, monkeypatch, OTHER)
    runs.end(main)
    runs.end(sha, 3, "pytest", "error: no\n", also={"ratchet-gate": _ratchet(1)})
    if verdict == INFRA:
        runs.state["c-6"] = ("done", 137, None)
    logged = loop.tick()
    assert logged[1]["record"]["verdict"]["state"] == verdict
    assert [r["to"] for r in logged if r["action"] == "ring"] == ([["@lane-1", "@vllm"]] if verdict == FAIL else [])


def test_under_gate_check_a_candidate_on_another_is_rung_about_what_that_one_s_check_passed_once_it_is_logged(env, monkeypatch):
    """A candidate built on another is judged off the gate against that candidate's check, as its log holds it: the red both
    fail rings only about the lower one, against `main`'s check, which passed."""
    def stack(s: Snapshot) -> list:
        if len(s.candidates) > 1:
            return []
        return [Build((len(s.candidates) + 1,), s.candidates[0].sha if s.candidates else s.main, 5)]

    loop, runs, main, low = _off(env, monkeypatch, OTHER, {"b.txt": "2\n"}, decide=stack, batch=1)
    (high,) = [r["sha"] for r in loop.tick() if r["action"] == "build"]
    failed = _circuit(f"lowering/pin-mismatch {HEAD}")
    runs.end(main)
    runs.end(low, also={"circuit-check": failed})
    runs.end(high, also={"circuit-check": failed})
    assert [(r["sha"], r["base"], r["to"]) for r in loop.tick() if r["action"] == "ring"] == [(low, main, ["@heads"])]
    assert [r for r in loop.tick() if r["action"] == "ring"] == [] and len(env.posts) == 1


def test_under_gate_check_a_warn_step_s_red_rings_its_owner_in_ci_toml_else_ci(env, monkeypatch):
    """drive.toml's warn steps are off the gate as policy.toml's are: the red of one whose log names nothing of it rings the
    step's owner in ci.toml (flock-rust-tests', proofs), else `CI` (lean-audit, which ci.toml here gives none)."""
    loop, runs, main, sha = _off(env, monkeypatch, OTHER)
    runs.end(main)
    runs.end(sha, also={"flock-rust-tests": "error: test failed\n", "lean-audit": "AUDIT: FAIL\n"})
    rings = {r["step"]: (r["to"], r["failures"]) for r in loop.tick() if r["action"] == "ring"}
    assert rings == {"flock-rust-tests": (["@proofs"], ["flock-rust-tests"]), "lean-audit": (["@ci"], ["lean-audit"])}


@pytest.mark.parametrize("cap", [2, 3])
def test_under_gate_check_a_ring_off_the_gate_quotes_its_first_failures_in_the_log_s_order_each_cut_short(env, monkeypatch, cap):
    """A ring quotes the first `OFF_SAID` failures in the log's order, each cut at 200 characters, and counts the rest."""
    monkeypatch.setattr(L, "OFF_SAID", cap)
    loop, runs, main, sha = _off(env, monkeypatch, OTHER)
    runs.end(main)
    said = [f"lowering/pin-mismatch {ROPE} " + "y" * 300, f"lowering/pin-mismatch {HEAD} it differs", "lowering/dead Other_v1"]
    runs.end(sha, also={"circuit-check": "".join(f"FAIL {x}\n" for x in said)})
    (ring,) = [r for r in loop.tick() if r["action"] == "ring"]
    assert ring["failures"] == [f"lowering/pin-mismatch {ROPE}", f"lowering/pin-mismatch {HEAD}", "lowering/dead Other_v1"]
    assert ring["to"] == ["@heads", "@circuits"]
    assert ring["text"] == _off_said(sha, "#1", "circuit-check", main, said[0][:200], *said[1:cap]).replace(
        ". Nothing", f"{'; and 1 more' if cap == 2 else ''}. Nothing")


def test_under_gate_check_the_steps_off_the_gate_are_its_bases_and_the_allowances_and_owners_the_candidate_s_own(
        env, monkeypatch):
    """What a candidate fails off the gate is read by the policy its verdict and its landing are: the steps off the gate are its
    base's (one it takes off the gate itself fails it on the gate, and rings nothing off it), the allowances and the jobs'
    owners its own, as the check after it lands reads them (an allowance it adds covers its failure, one expired covers
    nothing, and the ratchet gate's new owner is rung)."""
    allow = '[[allow]]\nid = "circuit-check lowering/dead {}"\nowner = "circuits"\nexpires = "{}"\nreason = "red for now"\n'
    pol = (OFF_TREE["tools/verity/check/policy.toml"].replace('"ratchet-gate"]', '"ratchet-gate", "pytest"]')
           + allow.format("New_v1", "2030-06-01T00:00:00Z") + allow.format("Old_v1", "2020-06-01T00:00:00Z"))
    jobs = OFF_TREE["tools/verity/check/ci.toml"].replace('"vllm"', '"ratchets"')
    loop, runs, main, sha = _off(env, monkeypatch, {"tools/verity/check/policy.toml": pol, "tools/verity/check/ci.toml": jobs})
    runs.end(main)
    runs.end(sha, 3, "pytest", "error: no\n", also={"circuit-check": _circuit("lowering/dead New_v1", "lowering/dead Old_v1"),
                                                     "ratchet-gate": _ratchet(1)})
    logged = loop.tick()
    assert logged[1]["record"]["verdict"]["state"] == FAIL
    rings = {r["step"]: r for r in logged if r["action"] == "ring"}
    assert set(rings) == {"circuit-check", "ratchet-gate"} and rings["circuit-check"]["failures"] == ["lowering/dead Old_v1"]
    assert rings["ratchet-gate"]["to"] == ["@lane-1", "@ratchets"]


def test_under_gate_check_a_ring_off_the_gate_in_shadow_is_only_logged_once(env, monkeypatch):
    loop, runs, main, sha = _off(env, monkeypatch, OTHER, shadow=True)
    runs.end(main)
    runs.end(sha, also={"ratchet-gate": _ratchet(1)})
    (ring,) = [r for r in loop.tick() if r["action"] == "ring"]
    assert (ring["done"], ring["shadow"], ring["to"], env.posts) == (False, True, ["@lane-1", "@vllm"], [])
    assert [r for r in loop.tick() if r["action"] == "ring"] == [] and env.posts == []


def test_under_gate_check_a_ring_off_the_gate_whose_post_failed_is_rung_again_by_the_next_tick(env, monkeypatch):
    loop, runs, main, sha = _off(env, monkeypatch, OTHER)
    runs.end(main)
    runs.end(sha, also={"ratchet-gate": _ratchet(1)})
    answer = [False]
    loop.world.post = lambda text, to=(): env.posts.append(text) or answer[0]
    (failed,) = [r for r in loop.tick() if r["action"] == "ring"]
    assert (failed["done"], failed["posted"], failed["error"]) == (
        False, False, f"ringing @lane-1, @vllm about {sha[:12]}'s ratchet-gate failed: the next tick rings again")
    answer[0] = True
    (again,) = loop.tick()
    assert (again["action"], again["done"], again["posted"], again["why"], len(env.posts)) == (
        "ring", True, True, "off the gate: ratchet-gate failed", 2)
    assert loop.tick() == [] and len(env.posts) == 2


def test_under_gate_check_auto_shards_launch_once_each_checks_plan_is_made_and_a_candidate_dropped_while_it_is_made_launches_none(
        env, monkeypatch):
    """`--shards auto:6`: each check, `main`'s and the candidate's, has the auto plan of its tree made first, and the first tick
    that finds it made launches its 6 shards `--shard auto:K/6` with its file, a machine each; the candidate advances once
    every shard passed.  One dropped while its plan is made launches none, the plan given up (`vqueue_slim.abandon`)."""
    from research.landing import AFTER_LANDING, AUTO_PLAN
    from test_landing import _plan_doc
    from test_vqueue_slim import Plans

    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide, slim=vqueue_slim, gate="check", shards="auto:6")
    runs = Runs(env, loop, monkeypatch, on=",".join(f"m{k}" for k in range(1, 7)))
    plans = Plans(loop.world.slim_driver(), env.tmp / "plans")
    _, built, started = loop.tick()
    sha = built["sha"]
    assert runs.launches == [] and plans.asked == [(c, ",".join(AFTER_LANDING), None, 6) for c in (main, sha)]
    assert started["record"]["plan"]["n"] == 6 and loop.tick() == [] and runs.launches == [], "pending while they are made"
    plans.answers.update({str(plans.tmp / f"plan-{k}"): (_plan_doc(6), None) for k in (1, 2)})
    assert _acts(loop.tick()) == ["slim", "slim"]
    assert [(c, m, kw["shard"], kw["shard_plan"]) for c, m, _, kw in runs.launches] == [
        (c, f"m{k}", f"auto:{k}/6", str(plans.tmp / f"plan-{p}" / AUTO_PLAN)) for p, c in ((1, main), (2, sha))
        for k in range(1, 7)]
    assert not any((plans.tmp / f"plan-{k}").exists() for k in (1, 2)) and loop.tick() == []
    runs.end(main, plan=PLAN6)
    runs.end(sha, plan=PLAN6)
    assert _acts(loop.tick()) == ["slim", "slim", "advance", "on_main", "drop"] and env.main() == sha
    env.put(2, {"b.txt": "1\n"})
    built, started = loop.tick()
    assert (built["action"], started["record"]["plan"]["out"]) == ("build", str(plans.tmp / "plan-3"))
    env.labels.held.add(2)                                          # held by a reviewer: dropped
    (drop,) = loop.tick()
    assert drop["action"] == "drop" and "beside_launching" not in drop and "beside_cancelled" not in drop
    assert not (plans.tmp / "plan-3").exists() and not (plans.tmp / "job-3").exists()
    assert loop.tick() == [] and len(runs.launches) == 12 and len(plans.asked) == 3


def _landed(env: Env, monkeypatch, decide=vqueue.decide) -> tuple[Loop, Path, str, dict, list[str]]:
    """Under `--gate check`, a candidate of #1 whose check passed 600 s after its launch and `main` advanced to while graph2,
    which says its check runs, hadn't ended: the loop, graph2's script, the candidate, its drop's line, and each sha
    `World.verdict` reads from here on."""
    env.put(1, {"a.txt": "1\n"})
    loop, runs = _check(env, monkeypatch, decide)
    script = env.graph2_script(loop, {"state": PENDING, "pending": 1})
    _, built, _ = loop.tick()
    env.clock += 600
    runs.end(env.main())
    runs.end(built["sha"])
    logged = loop.tick()
    assert _acts(logged) == ["slim", "slim", "advance", "on_main", "drop"] and env.main() == built["sha"]
    read: list[str] = []
    verdict = loop.world.verdict
    monkeypatch.setattr(loop.world, "verdict", lambda sha, *a, **k: read.append(sha) or verdict(sha, *a, **k))
    return loop, script, built["sha"], logged[-1], read


@pytest.mark.parametrize("graph2", ["pass", "fail"])
def test_under_gate_check_the_graph2_check_of_a_candidate_main_advanced_to_runs_on_and_gets_its_one_compare_line_when_it_ends(
        env, monkeypatch, graph2):
    """graph2 takes longer than the check that gates, so most candidates advance first.  Its check of one isn't cancelled at
    the drop, which logs what the check said; the loop reads it whole only once graph2's own verdict stops saying it runs,
    and then logs the one line: graph2's verdict beside the check's at the advance, with how long each took."""
    loop, script, sha, drop, read = _landed(env, monkeypatch)
    assert (drop["cancel"], drop["graph2"]) == ("left running, in shadow", PENDING) and not Path(f"{script}.cancel").exists()
    assert drop["follow"] == {"check": PASS, "said": PASS, "check_s": 600, "enqueued": L.stamp(T0)}
    env.clock += 600
    assert loop.tick() == [] and read == []                         # graph2 says it runs: no graph is read for it
    keys = env.listed(sha, {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", graph2)
    Path(f"{script}.verdict").write_text(json.dumps({"state": graph2, "pending": 0}))
    env.clock = T0 + 7200
    (line,) = loop.tick()
    assert read == [sha] and (line["action"], line["sha"], line["members"]) == ("compare", sha, [1])
    assert line["why"] == {"pass": "agree: graph2 and the check pass",
                           "fail": "disagree: graph2 fail (pytest:a), the check pass"}[graph2]
    assert (line["graph2"], line["check"], line["graph2_s"], line["check_s"]) == (graph2, PASS, 7200, 600)
    assert env.out[-1] == f"research queue vqueue: tick 4: {sha[:12]} graph2 in shadow: {line['why']}"
    env.result(keys["pytest:a"], "pytest:a", "fail" if graph2 == "pass" else "pass")   # whatever graph2 says later
    env.clock += 600
    assert loop.tick() == [] and read == [sha]                      # one line a candidate, and it is read no more
    assert not Path(f"{script}.cancel").exists() and not env.withdrawn


@pytest.mark.parametrize("how", ["never ends", "can't be read"])
def test_under_gate_check_a_followed_graph2_check_that_hasn_t_ended_a_day_after_its_enqueue_is_cancelled_and_unfinished(
        env, monkeypatch, how):
    """The loop gives up `FOLLOW_S` after the enqueue: it reads the check once more, cancels it and logs the candidate's one
    line all the same, `unfinished`, with how long graph2 ran and how many tasks ended.  A read that raises stops no tick."""
    loop, script, sha, _, read = _landed(env, monkeypatch)
    keys = env.listed(sha, {"pytest:a": ["a.txt"], "pytest:b": ["b.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    if how == "can't be read":
        Path(f"{script}.verdict").write_text(json.dumps({"state": PASS, "pending": 0}))

        def broken(sha: str, *a, **k):
            raise RuntimeError("the store is away")

        monkeypatch.setattr(loop.world, "verdict", broken)
    env.clock = T0 + L.FOLLOW_S - 1
    assert loop.tick() == [] and read == [] and not Path(f"{script}.cancel").exists()
    env.clock = T0 + L.FOLLOW_S
    (line,) = loop.tick()
    did = {"never ends": "1 of its 2 tasks on the merge gate ended",
           "can't be read": "its read failed (RuntimeError: the store is away)"}[how]
    assert line["why"] == f"unfinished: graph2 pending 86400 s after its enqueue ({did}), the check pass"
    assert (line["action"], line["graph2"], line["check"], line["graph2_s"], line["check_s"], line["cancel"]) == (
        "compare", "unfinished", PASS, 86400, 600, "cancelled")
    assert line.get("tasks") == ([1, 2] if how == "never ends" else None) and read == ([sha] if how == "never ends" else [])
    assert Path(f"{script}.cancel").read_text() == f"ready:{sha}\n"
    env.result(keys["pytest:b"], "pytest:b", "pass")                 # it ends after all: nothing more
    Path(f"{script}.verdict").write_text(json.dumps({"state": PASS, "pending": 0}))
    env.clock += 3600
    assert loop.tick() == [] and [r["graph2"] for r in env.lines() if r["action"] == "compare"] == ["unfinished"]


@pytest.mark.parametrize("graph2", ["pass", "fail"])
def test_under_gate_check_the_graph2_check_of_a_candidate_whose_check_failed_runs_on_past_its_drop_and_is_compared_when_it_ends(
        env, monkeypatch, graph2):
    """A candidate whose check failed is followed as one `main` advanced to is: its drop, beside its member's eject, leaves
    graph2's check running and logs what the check said, and once graph2's ends its one line compares the two."""
    main = env.main()
    env.put(1, {"pkg/tests/test_x.py": "def test_new():\n    assert False\n"})
    loop, runs = _check(env, monkeypatch, policy)
    script = env.graph2_script(loop, {"state": PENDING, "pending": 1})
    _, built, _ = loop.tick()
    sha = built["sha"]
    runs.end(main)
    runs.end(sha, 3, "pytest", rows=(_row(NEW),))
    logged = loop.tick()
    assert _acts(logged) == ["slim", "slim", "eject", "drop"] and env.main() == main
    drop = logged[-1]
    assert (drop["cancel"], drop["graph2"]) == ("left running, in shadow", PENDING) and not Path(f"{script}.cancel").exists()
    assert (drop["follow"]["check"], drop["follow"]["said"]) == (FAIL, "fail (step pytest)")
    env.clock = T0 + 300
    assert loop.tick() == []                                        # graph2 says it runs
    keys = env.listed(sha, {"pytest:a": ["pkg/tests/test_x.py"]})
    env.result(keys["pytest:a"], "pytest:a", graph2)
    Path(f"{script}.verdict").write_text(json.dumps({"state": graph2, "pending": 0}))
    env.clock = T0 + 300 + L.G2_READ_S                              # graph2's word that it runs stands until then
    (line,) = loop.tick()
    assert (line["action"], line["sha"], line["members"], line["graph2"], line["check"]) == ("compare", sha, [1], graph2, FAIL)
    assert line["why"] == {"pass": "disagree: graph2 pass, the check fail (step pytest)",
                           "fail": "agree: graph2 and the check fail"}[graph2]
    env.clock = T0 + L.FOLLOW_S
    assert loop.tick() == [] and not Path(f"{script}.cancel").exists()


def test_under_gate_check_a_candidate_dropped_while_its_check_runs_has_nothing_to_compare_and_its_graph2_check_is_cancelled(
        env, monkeypatch):
    """Its check hasn't ended, so its drop has no reading to compare graph2's with: graph2's check is cancelled at the drop,
    which follows nothing."""
    pulled: list[bool] = []

    def decide(s: Snapshot) -> list:
        return [Drop(c.sha, "a member was pulled") for c in s.candidates] if pulled else policy(s)

    env.put(1, {"a.txt": "1\n"})
    loop, runs = _check(env, monkeypatch, decide)
    script = env.graph2_script(loop, {"state": PENDING, "pending": 1})
    _, built, _ = loop.tick()
    pulled.append(True)
    (drop,) = loop.tick()
    assert (drop["sha"], drop["cancel"], "follow" in drop) == (built["sha"], "cancelled", False)
    assert Path(f"{script}.cancel").read_text() == f"ready:{built['sha']}\n"


def test_under_gate_check_at_most_two_dropped_candidates_graph2_checks_are_followed_and_the_oldest_past_them_is_given_up(
        env, monkeypatch):
    """Three candidates land one after another while their graph2 checks run, each left running at its drop.  The tick after
    the third's, `KEPT_MAX` (2) newer ones run beside the first's, so it is read once more, cancelled and closed `unfinished`,
    its line naming the two; the other two run on and get their own compare lines once they end, and the first no other."""
    loop, runs = _check(env, monkeypatch)
    script = env.graph2_script(loop, {"state": PENDING, "pending": 1})
    shas = []
    for k in (1, 2, 3):
        env.put(k, {f"{k}.txt": f"{k}\n"})
        shas.append(sha := next(r["sha"] for r in loop.tick() if r["action"] == "build"))
        if k == 1:
            runs.end(env.main())                                    # later, `main`'s check is the one that landed last
        runs.end(sha)
        drop = loop.tick()[-1]
        assert (drop["action"], drop["cancel"], env.main()) == ("drop", "left running, in shadow", sha)
        assert not Path(f"{script}.cancel").exists()
    first, second, third = shas
    assert L.KEPT_MAX == 2
    (line,) = loop.tick()
    assert (line["action"], line["sha"], line["graph2"], line["cancel"], line["followed"]) == (
        "compare", first, "unfinished", "cancelled", [third, second])
    assert line["why"].startswith("unfinished: graph2 pending ") and line["why"].endswith(
        "the check pass; at most 2 are followed, and newer ones are")
    assert Path(f"{script}.cancel").read_text() == f"ready:{first}\n"
    assert loop.tick() == []                                        # the two newer ones run on
    for k, sha in enumerate(shas, 1):
        keys = env.listed(sha, {"pytest:a": [f"{k}.txt"]})
        env.result(keys["pytest:a"], "pytest:a", PASS)
        Path(f"{script}.verdict.ready:{sha}").write_text(json.dumps({"state": PASS, "pending": 0}))
    assert loop.tick() == []                                        # graph2's word that they run stands `G2_READ_S`
    env.clock += L.G2_READ_S
    compared = loop.tick()
    assert [(r["action"], r["sha"], r["why"]) for r in compared] == [
        ("compare", sha, "agree: graph2 and the check pass") for sha in (second, third)]
    assert loop.tick() == [] and Path(f"{script}.cancel").read_text() == f"ready:{first}\n"
    assert [r["sha"] for r in env.lines() if r["action"] == "compare"] == [first, second, third]


@pytest.mark.parametrize("first", ["graph2", "the check"])
def test_under_gate_check_a_compare_line_says_how_long_graph2_s_verdict_took_from_its_enqueue_and_the_check_s_from_its_launch(
        env, monkeypatch, first):
    """Each duration ends when its own verdict did, not when the line is written: the one that ended first is read from the
    line that logged it then (graph2's `graph2` line, the run's judged record)."""
    def decide(s: Snapshot) -> list:
        return [] if s.candidates else [Build(tuple(x.number for x in s.ready), s.main, 5)]

    env.put(1, {"a.txt": "1\n"})
    loop, runs = _check(env, monkeypatch, decide)
    _, built, _ = loop.tick()
    sha = built["sha"]
    keys = env.listed(sha, {"pytest:a": ["a.txt"]})

    def ends(what: str) -> list[dict]:
        if what == "graph2":
            env.result(keys["pytest:a"], "pytest:a", "pass")
        else:
            runs.end(env.main())
            runs.end(sha)
        return loop.tick()

    env.clock = T0 + 300
    logged = ends(first)
    if first == "graph2":
        assert _acts(logged) == ["graph2"] and (logged[0]["graph2"], logged[0]["graph2_s"]) == (PASS, 300)
        assert logged[0]["why"] == "graph2, in shadow, ended before the check: pass" and loop.tick() == []
    else:
        assert _acts(logged) == ["slim", "slim"]
    env.clock = T0 + 900
    line = ends("the check" if first == "graph2" else "graph2")[-1]
    assert (line["action"], line["why"]) == ("compare", "agree: graph2 and the check pass")
    assert (line["graph2_s"], line["check_s"]) == ((300, 900) if first == "graph2" else (900, 300))
    env.clock += 300
    assert loop.tick() == []


def test_the_gate_is_graph2_unless_the_command_says_check_which_needs_the_run_beside_each_check(capsys, tmp_path):
    with pytest.raises(SystemExit):
        Q.main(["vqueue", "--help"])
    said = " ".join(capsys.readouterr().out.split())
    assert all(o in said for o in ("--gate {graph2,check}", "--shards", "--on", "(default 3)", "or auto:N")), said
    loop = Loop(None, tmp_path / "log")
    assert (loop.gate, loop.shards, L.GATES) == ("graph2", 3, ("graph2", "check")) and vqueue_slim.CHECK == L.CHECK_GATE
    assert L.main(["--gate", "check", "--no-slim", "--shadow", "--once"]) == 1
    assert "neither --no-slim nor --shards below 1" in capsys.readouterr().err
    with pytest.raises(ValueError, match="a gate is one of graph2, check"):
        Loop(None, tmp_path / "log", gate="check")
    for bad in ("0", "-1", "auto:0", "auto:x"):
        assert L.main(["--gate", "check", "--shards", bad, "--shadow", "--once"]) == 1
        assert "and --shards is N (ci.toml's plan) or auto:N (an auto plan)" in capsys.readouterr().err
    assert (L._shards("6"), L._shards("auto:6")) == (6, "auto:6")
    assert Loop(None, tmp_path / "log", slim=vqueue_slim, gate="check", shards="auto:6").shards == "auto:6"
    with pytest.raises(ValueError, match="'check' with auto:0 shard"):
        Loop(None, tmp_path / "log", slim=vqueue_slim, gate="check", shards="auto:0")


def _under(tasks: dict[str, dict], i: str, seen: set[str]) -> list[str]:
    """The walk the verdict made per gate: the tasks ``i`` reads, transitively, ``i`` first, as a recursive walk lists them."""
    if i in seen:
        return []
    seen.add(i)
    return [i, *(j for r in tasks[i]["reads"] for j in _under(tasks, r["task"], seen))]


def _reach(tasks: dict[str, dict], envs: dict, i: str, memo: dict[str, frozenset[str]]) -> frozenset[str]:
    """The verdict's inputs of a task as unions of frozensets: its own, its env's spec files and those of each task it reads."""
    if i not in memo:
        memo[i] = frozenset()
        t = tasks[i]
        memo[i] = frozenset(t["inputs"]).union((envs.get(t["env"]) or {}).get("spec") or (),
                                               *(_reach(tasks, envs, r["task"], memo) for r in t["reads"]))
    return memo[i]


@pytest.mark.parametrize("seed", range(200))
def test_the_closure_of_what_tasks_read_is_the_walks_per_task(seed):
    """`_walk` and `_Closure` work out once per task what the verdict's walk per gate and union per task did (a Lean graph's
    5 million reads over chains 241 long took either past eight minutes): on random graphs whose tasks read each other in
    either order, a task twice, and inputs and env spec files shared, the same tasks needed in the same order, the same
    failed tasks behind each in the same order, the same reach of the erred, and the same inputs."""
    import random

    rng = random.Random(seed)
    n = rng.randrange(1, 40)
    rank = rng.sample(range(n), n)
    envs = {"py": {"spec": ["uv.lock"]}, "lean": {"spec": ["lean-toolchain", "lake-manifest.json"]}, "bare": {}}
    tasks = {}
    for k in range(n):
        below = [j for j in range(n) if rank[j] < rank[k]]
        reads = [f"t{j}" for j in rng.sample(below, rng.randrange(min(len(below), 5) + 1))]
        if reads and rng.random() < 0.2:
            reads.append(reads[0])
        tasks[f"t{k}"] = {"id": f"t{k}", "env": rng.choice(sorted(envs)), "reads": [{"task": r, "output": "o"} for r in reads],
                          "inputs": [f"src/{rng.randrange(n)}.py" for _ in range(rng.randrange(3))]}
    ids = list(tasks)
    gates = rng.sample(ids, rng.randrange(1, n + 1))
    failed = frozenset(rng.sample(ids, rng.randrange(n // 3 + 1)))
    erred = frozenset(rng.sample(ids, rng.randrange(n // 4 + 1)))
    closure = L._Closure(tasks, envs)
    assert L._walk(tasks, gates) == list(dict.fromkeys(j for g in gates for j in _under(tasks, g, set())))
    memo: dict[str, frozenset[str]] = {}
    for i in rng.sample(ids, n):
        walked = _under(tasks, i, set())
        assert closure.fails(i, failed) == tuple(j for j in walked if j in failed)
        assert closure.meets(i, erred) == bool(set(walked) & erred)
        assert closure.inputs(i) == _reach(tasks, envs, i, memo)


def test_a_cancel_leaves_running_the_runs_the_keep_file_names(tmp_path, monkeypatch):
    """A first shard's Lean audit writes its node's module index only when it ends, so a drop's cancel leaves the runs
    `KEEP_RUNS` names running, and cancels the rest; without the file it cancels them all, and with one it can't read it
    cancels none, refusing with the fix."""
    cancelled = []
    monkeypatch.setattr(L, "cancel_runs", lambda repo, ids, why: cancelled.append(list(ids)) or {})
    keep = tmp_path / "keep-runs"
    keep.write_text("# first shards\nr-2  # cpu-2's audit\n\n")
    w = World(type("Q", (), {"workdir": tmp_path})(), None, None, keep_file=keep)
    assert w.cancel_runs(["r-1", "r-2", "r-3"], "dropped") == {"r-2": f"left running: {keep} names it"}
    assert cancelled == [["r-1", "r-3"]]
    assert w.cancel_runs(["r-2"], "dropped") == {"r-2": f"left running: {keep} names it"} and cancelled == [["r-1", "r-3"]]
    keep.unlink()
    assert w.cancel_runs(["r-2"], "dropped") == {} and cancelled[-1] == ["r-2"]
    assert L.kept_runs(None) == frozenset()
    keep.mkdir()
    with pytest.raises(L.QueueError, match="can't be read.*or remove it"):
        w.cancel_runs(["r-4"], "dropped")
    assert cancelled[-1] == ["r-2"]


def test_under_gate_check_graph2_s_reading_of_a_candidate_stands_g2_read_s_between_ticks_and_is_read_again_once_its_check_ended(
        env, monkeypatch):
    """graph2 only shadows where the check gates, and each of its verdicts is megabytes and seconds: a candidate's reading is
    kept beside the log, so the next `--once` tick, a loop of its own, takes it too, and is read again `G2_READ_S` after it,
    or at once in the tick that finds the candidate's check ended, whose compare line reads graph2 as it is then."""
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    loop, runs = _check(env, monkeypatch)
    _, built, _ = loop.tick()
    sha = built["sha"]
    runs.end(main)
    read: list[str] = []
    verdict = loop.world.verdict
    monkeypatch.setattr(loop.world, "verdict", lambda s, *a, **k: read.append(s) or verdict(s, *a, **k))
    assert _acts(loop.tick()) == ["slim"] and read == [sha]
    assert set(json.loads((env.log.parent / L.G2_READS).read_text())) == {sha}
    timed: list[str] = []
    env.clock += L.G2_READ_S - 1
    nxt = Loop(loop.world, env.log, slim=vqueue_slim, gate="check", now=lambda: env.clock, out=env.out.append,
               timing=timed.append)
    assert nxt.tick() == [] and read == [sha] and timed[-1].endswith(", 1 cached")
    env.clock += 1
    assert nxt.tick() == [] and read == [sha, sha] and timed[-1].endswith(", 0 cached")
    keys = env.listed(sha, {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", PASS)
    assert nxt.tick() == [] and read == [sha, sha]                  # graph2 passed it, read `G2_READ_S` from now
    runs.end(sha)
    logged = nxt.tick()
    assert _acts(logged) == ["slim", "compare", "advance", "on_main", "drop"] and read == [sha] * 3
    assert logged[1]["why"] == "agree: graph2 and the check pass" and logged[-1]["graph2"] == PASS


def test_under_gate_check_a_kept_reading_is_read_again_at_the_drop_and_forgotten_at_an_enqueue(env, monkeypatch):
    """A drop logs graph2's reading and follows by it, so one kept from an earlier tick is read again there; an enqueue (a
    rerun and a cancel too) of graph2's check forgets the reading of it."""
    pulled: list[bool] = []

    def decide(s: Snapshot) -> list:
        return [Drop(c.sha, "a member was pulled") for c in s.candidates] if pulled else policy(s)

    env.put(1, {"a.txt": "1\n"})
    loop, runs = _check(env, monkeypatch, decide)
    _, built, _ = loop.tick()
    sha = built["sha"]
    read: list[str] = []
    verdict = loop.world.verdict
    monkeypatch.setattr(loop.world, "verdict", lambda s, *a, **k: read.append(s) or verdict(s, *a, **k))
    loop.tick()
    loop.tick()
    assert read == [sha] and sha in loop.reads.docs
    loop.world.enqueue(sha, 5)
    assert sha not in loop.reads.docs
    loop.tick()
    assert read == [sha, sha]
    pulled.append(True)
    (drop,) = loop.tick()
    assert drop["action"] == "drop" and read == [sha] * 3           # kept in the snapshot, read again for the drop


def test_under_gate_graph2_no_reading_is_kept_and_a_verdict_is_trimmed_to_what_the_loop_reads(env):
    """Where graph2 gates, its verdict decides, so it is read each tick; and either way only the fields the loop reads of it
    are kept, the megabytes of the rest dropped at the read."""
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy)
    assert loop.reads is None and loop.world.reads is None
    script = env.graph2_script(loop, {"state": PENDING, "pending": 1})
    (built,) = loop.tick()
    assert loop.tick() == [] and loop.tick() == []
    assert Path(f"{script}.asked").read_text().splitlines().count(f"ready:{built['sha']}") == 2
    Path(f"{script}.verdict").write_text(json.dumps({
        "state": PENDING, "pending": 1, "graph": {"tasks": ["x"] * 1000}, "tasks": "many",
        "failed": [{"task": "t", "key": "k", "why": "w", "log": "x" * 100}, "junk"],
        "flaky": [{"task": "t", "key": "k", "inputs": ["a"], "tests": ["x"]}]}))
    assert loop.world.graph2_verdict(built["sha"]) == {
        "state": PENDING, "pending": 1, "failed": [{"task": "t", "key": "k", "why": "w"}],
        "flaky": [{"task": "t", "key": "k", "inputs": ["a"]}]}
    assert not (env.log.parent / L.G2_READS).exists()


def test_a_tick_prints_one_line_of_its_steps_seconds_and_peak_resident_size(env):
    """Each tick, however it ends, gives `timing` one line: its seconds and peak resident size, then each step's seconds and
    the peak after it, then graph2's verdicts read (how many, their megabytes, their seconds) and the readings kept."""
    env.put(1, {"a.txt": "1\n"})
    timed: list[str] = []
    now = iter(float(k) for k in range(1000))
    loop = env.loop(policy, clock=lambda: next(now), peak=lambda: 2.5, timing=timed.append)
    (built,) = loop.tick()
    assert built["action"] == "build"
    (line,) = timed
    assert line.startswith("research queue vqueue: tick 1: ") and ", peak 2.5 GB; s/GB: log 1.0/2.5 fetch 1.0/2.5 " in line
    steps = line.split("; s/GB: ")[1].split("; ")[0].split()
    assert steps[::2] == ["log", "fetch", "ready", "below", "judge", "chain", "decide", "notices", "followed", "on_main",
                          "build", "keep", "mapped"]
    assert all(x.endswith("/2.5") for x in steps[1::2]) and line.endswith("; graph2 0 read(s) 0.0 MB 0.0 s, 0 cached")

    def broken():
        raise L.QueueError("fetch failed")

    loop.world.fetch = broken
    with pytest.raises(L.QueueError):
        loop.tick()
    assert len(timed) == 2 and timed[1].startswith("research queue vqueue: tick 2: ") and "s/GB: log " in timed[1]


def test_with_after_the_steps_after_main_moved_start_detached_and_the_tick_that_finds_them_ended_logs_them(env, monkeypatch):
    """Outside --shadow, the advance's tick starts the steps after `main` moved as a job of their own and returns: nothing
    in it runs them.  Each later tick looks at the job and logs nothing while it runs; the one that finds it ended logs
    them done with what they told, or, when they failed, prints why and starts them again."""
    env.put(1, {"a.txt": "1\n"})
    loop = env.loop(policy, after=["vqueue", "--log", "x"])
    started: list[tuple[dict, list[str], Path]] = []

    def start_after(job, argv, root):
        started.append((job, argv, root))
        (d := root / f"job-{len(started)}").mkdir(parents=True)
        return {"dir": str(d), "pid": os.getpid(), "out": str(d), "at": env.clock}

    monkeypatch.setattr(loop.world, "start_after", start_after)
    monkeypatch.setattr(loop.world, "on_main", lambda *a: pytest.fail("the tick ran the steps after `main` moved"))
    (built,) = loop.tick()
    sha = built["sha"]
    keys = env.listed(sha, {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    advance, on_main = loop.tick()
    assert (advance["action"], on_main["action"], on_main["done"], env.main()) == ("advance", "on_main", False, sha)
    ((job, argv, root),) = started
    assert (job["sha"], job["base"], job["members"], argv, root) == (sha, built["base"], [1], ["vqueue", "--log", "x"],
                                                                     env.log.parent / L.AFTER)
    assert [(m["pr"], m["head"]) for m in job["merged"]] == [(1, built["heads"][0])] and job["told"] == []
    assert on_main["job"]["dir"] == str(root / "job-1")
    (drop,) = loop.tick()                                           # the job runs: nothing of it is logged
    assert drop["action"] == "drop" and len(started) == 1
    d = root / "job-1"
    (d / "result.json").write_text(json.dumps({"done": False, "told": ["stacked:2<1"], "error": "RuntimeError: push"}))
    (d / "log").write_text("pushing\n")
    (d / "rc").write_text("0\n")
    (failed,) = loop.tick()
    assert (failed["action"], failed["done"], failed["error"], failed["told"]) == ("on_main", False, "RuntimeError: push",
                                                                                ["stacked:2<1"])
    assert f"research queue vqueue: tick 4: the steps after `main` moved to {sha[:12]} failed, the next tick runs them " \
           "again: RuntimeError: push" in env.out
    assert not d.exists() and len(started) == 1
    (again,) = loop.tick()
    assert (again["action"], again["done"], len(started), started[1][0]["told"]) == ("on_main", False, 2, ["stacked:2<1"])
    assert loop.tick() == []
    d = root / "job-2"
    (d / "result.json").write_text(json.dumps({"done": True, "told": ["stacked:2<1"]}))
    (d / "log").write_text("")
    (d / "rc").write_text("0\n")
    (done,) = loop.tick()
    assert (done["action"], done["done"], done["told"], done["ended"]) == ("on_main", True, ["stacked:2<1"], str(d))
    assert loop.tick() == [] and len(started) == 2


def test_an_after_landing_job_runs_in_a_session_of_its_own_and_the_tick_reads_how_it_ended(tmp_path):
    """`World.start_after` returns while its job still runs, which `after_ended` says (None); once it exits, it reads what
    the job wrote, and the job's output is in `AFTER_LOG` beside its directory."""
    from waits import Gate, wait_exit

    gate = Gate(tmp_path / "gate")
    w = World(type("Q", (), {"workdir": tmp_path})(), None, None, now=lambda: T0)
    code = ("import json, pathlib, sys; open(sys.argv[1], 'rb').read(1); print('after'); "
            "pathlib.Path(sys.argv[3], 'result.json').write_text(json.dumps({'done': True, 'told': ['x']}))")
    job = w.start_after({"sha": "ab" * 20}, [sys.executable, "-c", code, str(gate.path)], tmp_path / "live" / L.AFTER)
    try:
        assert Path(job["dir"]).name.startswith("abababababab-") and json.loads(
            (Path(job["dir"]) / "job.json").read_text()) == {"sha": "ab" * 20}
        assert w.after_ended(job) is None
    finally:
        gate.release()
    wait_exit(job["pid"])
    gate.close()
    assert w.after_ended(job) == {"done": True, "told": ["x"]} and not Path(job["dir"]).exists()
    assert (tmp_path / "live" / L.AFTER_LOG).read_text() == "after\n"
    gone = {"dir": str(tmp_path / "gone"), "pid": job["pid"], "out": str(tmp_path / "gone"), "at": T0}
    (tmp_path / "gone").mkdir()
    assert w.after_ended(gone).startswith("its process is gone") and not (tmp_path / "gone").exists()


def test_after_landing_runs_the_steps_at_main_and_writes_what_they_did(tmp_path):
    """`--after-landing DIR` runs the steps of the advance `job.json` names at `main` as origin has it, and writes
    `result.json`: done with the owners told, or the error."""
    calls: list = []

    class W:
        def at_main(self):
            calls.append("at_main")

        def on_main(self, sha, base, members, told):
            calls.append((sha, base, members))
            told.append("stacked:2<1")
            if sha == "bad":
                raise RuntimeError("push refused")

    (d := tmp_path / "j").mkdir()
    (d / "job.json").write_text(json.dumps({"sha": "s" * 40, "base": "b", "merged": [
        {"pr": 1, "head": "h", "owner": "", "commit": "c"}], "told": ["stacked:3<1"]}))
    assert L._after_landing(W(), d) == 0
    assert calls == ["at_main", ("s" * 40, "b", [Q.Member(1, "h", "", "c")])]
    assert json.loads((d / "result.json").read_text()) == {"done": True, "told": ["stacked:3<1", "stacked:2<1"]}
    (d / "job.json").write_text(json.dumps({"sha": "bad", "base": "b", "merged": []}))
    assert L._after_landing(W(), d) == 0
    assert json.loads((d / "result.json").read_text()) == {"done": False, "told": ["stacked:2<1"],
                                                           "error": "RuntimeError: push refused"}


def test_under_gate_check_a_tick_reads_one_candidate_s_graph2_reading_afresh_and_the_others_wait_their_turn(env, monkeypatch):
    """A fresh reading lists the check's whole graph and pulls each pending task's results, so a tick reads `G2_FRESH` (one)
    afresh: the lowest candidate whose reading is out first, each other keeping the one kept of it, or pending, till its turn."""
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    loop, runs = _check(env, monkeypatch, batch=1, depth=2)
    built = [r["sha"] for _ in range(2) for r in loop.tick() if r["action"] == "build"]
    assert len(built) == 2
    low, high = built
    read: list[str] = []
    verdict = loop.world.verdict
    monkeypatch.setattr(loop.world, "verdict", lambda s, *a, **k: read.append(s) or verdict(s, *a, **k))
    loop.reads.docs.clear()
    loop.tick()
    assert read == [low]
    loop.tick()
    assert read == [low, high]
    loop.tick()
    assert read == [low, high]
    env.clock += L.G2_READ_S
    loop.tick()
    loop.tick()
    assert read == [low, high, low, high]


def _wide(sha: str, n: int = 300) -> dict:
    """A graph/v2 listing of ``sha`` shaped as a real check's is, at a small scale: a few keyed tasks, and many that each read
    every one listed before them (a real one has thousands of tasks and millions of reads, gigabytes once parsed)."""
    def task(k: int) -> dict:
        return {"id": f"t{k}", "kind": "test-file", "env": "py", "argv": ["t"], "vars": {}, "inputs": ["a.txt"],
                "reads": [{"task": f"t{j}", "output": "out"} for j in range(k)] if k >= 3 else [],
                "outputs": [{"name": "out", "path": f"t{k}.out"}], **({} if k >= 3 else {"key": f"sha512:{sha[:8]}-{k}"})}

    return {"schema": GRAPH_V2, "commit": sha, "files": {f: f"sha512:{f}" for f in (*SPEC, "a.txt")},
            "envs": {"py": {"digest": "sha512:env", "spec": list(SPEC)}}, "tasks": [task(k) for k in range(n)]}


def test_under_gate_check_a_shadow_read_holds_one_candidate_s_graphs_at_a_time(env, monkeypatch):
    """A candidate's shadow read lists its check's graph and its base's, each gigabytes for a real check, so a tick holds
    those of one read at a time (`World.one_read`), however many it reads: three candidates read afresh in one tick peak
    under two and a half graphs' memory, where graphs kept for the whole snapshot would hold three and more."""
    import tracemalloc

    for k in (1, 2, 3):
        env.put(k, {f"{k}.txt": f"{k}\n"})
    loop, runs = _check(env, monkeypatch, batch=1, depth=3, g2_fresh=3)
    built = [r["sha"] for _ in range(3) for r in loop.tick() if r["action"] == "build"]
    assert len(built) == 3
    for sha in built:
        env.graph2.list_graph(f"ready:{sha}", _wide(sha))
    tracemalloc.start()
    try:
        got = loop.world.tasks.graph_v2(f"ready:{built[0]}")
        one = tracemalloc.get_traced_memory()[0]
        del got
        loop.reads.docs.clear()
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        loop.tick()
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    assert loop.reads.hits == 0 and len(loop.reads.used) == 3
    assert peak < 2.5 * one, f"a tick of three shadow reads peaked at {peak / one:.1f} graphs"
