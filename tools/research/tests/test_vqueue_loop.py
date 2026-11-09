"""`research queue vqueue` (`research.vqueue_loop`): the verified queue's loop on a throwaway origin and a clone of it, graph2's
listings and results as `FileTransport` files, a fake GitHub and fake labels, and a small policy in place of `vqueue.decide`.
Every test reads the clock it injects."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path

import pytest

import research.merge as M
import research.queue as Q
import research.vqueue_loop as L
from research import vqueue, vqueue_slim
from research.drive import PREFLIGHT
from research.executor import GRAPH_V2, Executor, FileTransport
from research.vqueue import (FAIL, INFRA, LIST_FAILED, PASS, PENDING, Advance, Build, Drop, Eject, Outcome, Red, Rerun, Retry,
                             Ring, Snapshot, Verdict)
from research.vqueue_loop import Loop, World, _Clone, _said, branch
from test_queue import World as Origin, _git

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
    """A PR's ready mark is the fake GitHub's (`ready`), and each marked head has a land verdict by `reviewer` unless ``held``."""

    def __init__(self, gh):
        self.gh, self.held = gh, set()

    def ready(self, n: int, head: str) -> bool:
        return (n, head) in self.gh.ready

    def reviews(self, n: int, head: str) -> list[Lab]:
        return [Lab("reviewer", "hold" if n in self.held else "land", "2026-01-01T00:00:00Z")]

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


def test_an_ejection_for_a_failed_lean_lock_task_tells_its_owner_how_to_regenerate_the_package_s_records(env):
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
    assert "lock records of Security don't match" in post and "check.py --build --update PKG" in post, post
    env.put(3, {"c.txt": "3\n"})
    (again,) = loop.tick()
    keys = env.listed(again["sha"], {"pytest:b": ["b.txt"], "pytest:c": ["c.txt"]})
    env.result(keys["pytest:b"], "pytest:b", "fail")
    env.result(keys["pytest:c"], "pytest:c", "pass")
    loop.tick()
    assert "lock records" not in env.posts[-1] and two[:12] in env.posts[-1], env.posts[-1]


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
    script = env.graph2_script(loop, {"state": said})
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


def test_a_member_that_doesnt_merge_onto_the_build_is_left_out_logged_and_its_owner_told_once(env):
    env.put(1, {"a.txt": "1\n"})
    two = env.put(2, {"a.txt": "2\n"})
    loop = env.loop(lambda s: [Drop(c.sha, "rebuilt") for c in s.candidates] or [Build((1, 2), s.main, 5)])
    skip, built = loop.tick()
    assert (skip["action"], skip["number"], skip["heads"], skip["done"]) == ("skip", 2, [two], True)
    assert "a.txt" in skip["why"] and built["members"] == [1], (skip, built)
    (post,) = env.posts
    assert "#2 at" in post and "doesn't merge" in post
    assert _acts(loop.tick()) == ["drop"]
    again, built = loop.tick()                                      # the same head onto the same base: told once
    assert (again["action"], again["text"], again["done"], built["action"]) == ("skip", None, False, "build")
    assert len(env.posts) == 1


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
    script = env.graph2_script(loop, {"state": FAIL, "pending": 1})
    v = loop.world.verdict(sha, base).verdict
    assert (v.state, [f.base_key for f in v.failed], v.ended, vqueue._reading(v)) == (FAIL, [z], False, PENDING)
    Path(f"{script}.verdict").write_text(json.dumps({"state": FAIL, "pending": 0}))
    v = loop.world.verdict(sha, base).verdict
    assert (v.ended, vqueue._reading(v)) == (True, PASS)


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
    files = [f"integrations/vllm/verity_vllm/pipeline/{n * 'x'}.py" for n in (30, 40, 50, 60)]
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


def test_each_tick_reads_the_labels_afresh_as_drive_reads_them(env):
    made: list[Labels] = []
    loop = env.loop(lambda s: [])
    loop.world.labels_at = lambda: made.append(Labels(env.o.gh)) or made[-1]
    loop.tick()
    loop.tick()
    w = loop.world
    assert len(made) == 2 and w.labels is made[-1] and w.q.ready == made[-1].ready and w.q.marked_at == made[-1].marked_at
    assert made[-1].change == w.q.own_change and made[-1].since == w.q.rules().marked_after


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
    assert _acts(loop.tick()) == ["advance", "on_main", "drop"] and env.main() == one["sha"]
    assert _acts(loop.tick()) == ["advance", "on_main", "drop"] and env.main() == two["sha"] and env.posts == []  # no red on z


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
    blamed, the loop enqueues `ready:<base>`, again each `BASE_RETRY_S` while it stays unlisted, and rings `@ci` and `@infra`
    once, `BASE_RING_S` after it first asked for it; listed, a failure at the base's key is inherited and the candidate
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
    `ready:<base>` again each `BASE_RETRY_S`, and rings `@ci` and `@infra` at once, naming it, once a base and why (with
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
    """drive's `executor = "on"` path promotes nothing whose graph lacks a job `main` waits for (`drive.gating`): a candidate
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
    lean, exe = "verity/core/service/flock", {"name": "flock-verify", "path": "verity/core/service/flock/.lake/build/bin/v"}
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


def test_a_failure_an_allowance_of_the_candidate_s_policy_covers_is_passed_as_drive_reads_it_until_it_expires(env):
    """drive's `executor = "on"` path takes a failed task whose every red record an unexpired allowance of the checked commit's
    policy covers as passed (`Driver._task_units`, `Driver._task_allowed`), and so does the verdict."""
    env.put(1, {"a.txt": "1\n", "tools/verity/check/policy.toml": ALLOWED})
    loop = env.loop(vqueue.decide)
    (built,) = loop.tick()
    sha, w = built["sha"], loop.world
    keys = env.listed(sha, {"pytest:a": ["a.txt"], "pytest:z": ["z.txt"]}, suites={"pytest:z": "zz"})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    env.result(keys["pytest:z"], "pytest:z", "fail", tests=["test_z.py::test_one"])
    j = w.verdict(sha, built["base"])
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
    dry = env.loop(vqueue.decide, slim=slim, batch=1, depth=3, dry_run=True)    # its world has no Driver: a start raises
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
    dry = env.loop(vqueue.decide, slim=slim, batch=1, dry_run=True)        # its world has no Driver: a start or judge raises
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
    env.o.gh.ready.discard((1, built["heads"][0]))                  # its ready mark withdrawn: dropped
    (drop,) = loop.tick()
    assert (drop["action"], drop["beside_launching"], env.cancelled) == ("drop", started["record"], [])
    assert "beside_cancelled" not in drop and loop.tick() == []     # still launching
    slim.registered[sha] = registered
    (settled,) = loop.tick()
    assert (settled["action"], settled["sha"], settled["dropped"], settled["record"]["ids"]) == ("slim", sha, True, registered)
    assert "launch" not in settled["record"]
    assert env.cancelled == ([(registered, f"vqueue: candidate {sha[:12]} was dropped: #1's ready mark was withdrawn")]
                             if registered else [])
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
    env.o.gh.ready.discard((1, built["heads"][0]))                  # its ready mark withdrawn: dropped
    (drop,) = loop.tick()
    assert (started["action"], drop["action"], drop["beside_cancelled"]) == ("slim", "drop", ["run-1"])
    assert env.cancelled == [(["run-1"], f"vqueue: candidate {built['sha'][:12]} was dropped: #1's ready mark was withdrawn")]
    env.put(2, {"b.txt": "2\n"})
    two, _ = loop.tick()
    keys = env.listed(two["sha"], {"pytest:b": ["b.txt"]})
    env.result(keys["pytest:b"], "pytest:b", "pass")
    assert loop.tick() == []
    slim.verdicts["run-2"] = FAIL
    env.o.gh.ready.discard((2, two["heads"][0]))
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


def test_research_queue_vqueue_is_its_own_command_and_runs_the_run_beside_each_check_with_vqueue_slim(capsys):
    from research import vqueue_loop, vqueue_slim

    with pytest.raises(SystemExit) as e:
        Q.main(["vqueue", "--help"])
    said = " ".join(capsys.readouterr().out.split())
    assert e.value.code == 0 and all(o in said for o in ("--shadow", "--lean", "--pending-hours", "--no-slim"))
    assert vqueue_loop._slim() is vqueue_slim


def test_the_loop_s_batch_depth_and_lean_default_to_16_3_and_3_on_the_command_and_in_the_loop(capsys, tmp_path):
    with pytest.raises(SystemExit):
        Q.main(["vqueue", "--help"])
    said = " ".join(capsys.readouterr().out.split())
    assert all(f"{o} (default {n})" in said for o, n in (("holds", 16), ("its check", 3), ("or a spec", 3))), said
    loop = Loop(None, tmp_path / "log")
    assert (loop.batch, loop.depth, loop.lean) == (L.BATCH, L.DEPTH, L.LEAN) == (16, 3, 3)


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
