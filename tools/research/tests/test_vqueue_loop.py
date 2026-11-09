"""`research queue vqueue` (`research.vqueue_loop`): the verified queue's loop on a throwaway origin and a clone of it, graph2's
listings and results as `FileTransport` files, a fake GitHub and fake labels, and a small policy in place of `vqueue.decide`.
Every test reads the clock it injects."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import pytest

import research.merge as M
import research.queue as Q
import research.vqueue_loop as L
from research import vqueue
from research.executor import GRAPH_V2, Executor, FileTransport
from research.vqueue import (FAIL, INFRA, LIST_FAILED, PASS, PENDING, Advance, Build, Drop, Eject, Outcome, Red, Retry, Ring,
                             Snapshot, Verdict)
from research.vqueue_loop import Loop, World, _Clone, _faulted, branch
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
        """A graph2 with `verdict`, `cancel` and `rerun` for ``loop``: its verdict ``verdict`` (default a fail), and each cancel
        and rerun appended to `<script>.cancel` and `<script>.rerun`."""
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
    assert (built["action"], built["members"], built["base"], built["priority"]) == ("build", [1, 2], main, 2), built
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
    """With the run beside each check too, which doesn't start for it, so it is list-failed and not pending."""
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
    (built,) = loop.tick()                                      # the second build waits
    assert (built["action"], built["enqueued"], built["error"]) == ("build", False, "graph2 is down"), built
    assert env.queued() == {branch(built["sha"]): built["sha"]} and env.graph2.enqueued() == []
    (retry,) = loop.tick()
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
said = sys.argv[0] + ".verdict"
if a.cmd == "verdict":
    print(open(said).read() if os.path.exists(said) else json.dumps({"state": "fail", "check": a.check}))
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
    a, why = keys["pytest:a"], "read 1 file(s) outside its declared inputs: x.txt"
    said = {"state": FAIL, "failed": [{"task": "pytest:a", "key": a, "why": why}]}
    Path(f"{script}.verdict").write_text(json.dumps(said))
    assert [f.task for f in loop.world.verdict(sha).verdict.failed] == ["pytest:a"]
    assert _faulted({**said, "labelled": "2026-01-01T00:00:00Z"}) == {}       # a label outlives the rerun of what it says
    eject, drop = loop.tick()
    assert (eject["action"], eject["number"], env.withdrawn, drop["failed"]) == ("eject", 1, [(1, one)],
                                                                                 [{"task": "pytest:a", "key": a}])


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
    (f,) = w.verdict(sha, main).verdict.failed                      # the base's has no result yet
    assert (f.tests, f.base_tests, f.base_key) == (frozenset({"test_t.py::a"}), None, "sha512:base-t")
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
    env.graph2_script(loop, {"state": PENDING})
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


def test_a_failure_on_a_base_graph2_hasnt_listed_waits_for_its_listing_which_the_loop_enqueues_once(env):
    """Whose a failure is, the base's keys tell: on a base whose check isn't listed (`main` moved past every check), the loop
    enqueues `ready:<base>` once, logged, and the candidate is pending; a base whose listing failed has no keys, logged, so the
    failure is the candidate's own; listed, a failure at the base's key is inherited and the candidate lands."""
    _git(env.o.origin, "update-ref", "refs/heads/main", env.o.change({"m.txt": "m\n"}))
    main = env.main()
    one = env.put(1, {"a.txt": "1\n"})
    loop = env.loop(vqueue.decide)
    (built,) = loop.tick()
    sha, w, red = built["sha"], loop.world, {"pytest:r": "sha512:red"}
    keys = env.listed(sha, {"pytest:a": ["a.txt"], "pytest:r": ["red.txt"]}, keys=red)
    env.result(keys["pytest:a"], "pytest:a", "pass")
    env.result("sha512:red", "pytest:r", "fail")
    (base,) = loop.tick()
    assert (base["action"], base["sha"], base["candidate"], base["enqueued"], base["done"]) == ("base", main, sha, True, True)
    assert env.graph2.enqueued()[-1] == {"check": f"ready:{main}", "priority": 2, "graph": None, "schema": GRAPH_V2}
    j = w.verdict(sha, main)
    assert (j.verdict, j.unlisted, j.note) == (Verdict(PENDING), main, "waiting for the base's listing: its check isn't listed yet")
    assert loop.tick() == [] and len(env.graph2.enqueued()) == 2 and env.withdrawn == []       # once a base
    w.graph2_verdict = lambda s: {"state": LIST_FAILED, "why": "no graph"} if s == main else None
    v = w.verdict(sha, main).verdict
    assert v.state == FAIL and [(f.task, f.base_key) for f in v.failed] == [("pytest:r", None)]
    assert any(f"ready:{main[:12]}'s listing failed (no graph)" in x for x in env.out), env.out
    del w.graph2_verdict
    env.listed(main, {"pytest:r": ["red.txt"]}, keys=red)
    told, advance, on_main, landed = loop.tick()
    assert _acts([told, advance, on_main, landed]) == ["red", "advance", "on_main", "drop"] and env.main() == sha
    assert landed["outcome"]["inherited"][0]["base_key"] == "sha512:red" and landed["outcome"]["heads"] == [one]


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
    assert (j.verdict, j.unlisted, j.note) == (Verdict(PENDING), None, "waiting for the base's listing: pytest:r there still "
                                                                        "wait on what they read")
    env.result("sha512:lx", "lean-build:x", "error", infra=True)       # what it reads there may never pass: no key
    v = w.verdict(sha, main).verdict
    assert v.state == FAIL and [(f.task, f.base_key) for f in v.failed] == [("pytest:r", None)]
    env.result("sha512:lx", "lean-build:x", "pass")
    (f,) = w.verdict(sha, main).verdict.failed
    assert f.base_key is not None and f.base_key != f.key


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
    """`research.vqueue_slim`'s three functions, faked: each start registers a run, judged as ``verdicts`` says (pending
    unless it names it), and a verdict other than pending is kept in its record, as `judge` keeps it.  Each record judged is
    kept in ``judged``."""

    def __init__(self):
        self.started: list[tuple] = []
        self.verdicts: dict[str, str] = {}
        self.judged: list = []

    def start(self, driver, sha: str, base: str) -> dict:
        self.started.append((driver, sha, base))
        return {"ids": [f"run-{len(self.started)}"], "tip": sha, "base": base}

    def judge(self, driver, rec: dict | None) -> Verdict:
        self.judged.append(rec)
        if rec is None:
            return Verdict(INFRA, (), (vqueue.Failed("the run beside its check registered no run", ":no run", frozenset()),))
        state = rec.get("verdict") or self.verdicts.get(rec["ids"][0], PENDING)
        if state != PENDING:
            rec["verdict"] = state
        down = (vqueue.Failed(f"the run beside its check ({rec['ids'][0]}) was lost", f"{rec['tip']}:{rec['ids'][0]}",
                              frozenset()),)
        return Verdict(state, (), down if state == INFRA else ())

    @staticmethod
    def combine(graph: Verdict, slim: Verdict) -> Verdict:
        state = next(s for s in (FAIL, INFRA, PENDING, LIST_FAILED, PASS) if s in (graph.state, slim.state))
        return Verdict(state, graph.failed + slim.failed, graph.infra + slim.infra)


def test_the_run_beside_a_check_starts_once_its_check_passed_its_record_is_kept_and_both_must_pass(env):
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    slim = Slim()
    loop = env.loop(vqueue.decide, slim=slim)
    loop.world.slim_driver = lambda: "drive"
    (built,) = loop.tick()
    sha = built["sha"]
    keys = env.listed(sha, {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    (started,) = loop.tick()
    assert (started["action"], started["sha"], started["why"], started["record"]) == (
        "slim", sha, "started", {"ids": ["run-1"], "tip": sha, "base": main})
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
    assert len(slim.started) == 2 and loop.tick() == []
    started_again = env.loop(vqueue.decide, clone="again", slim=slim)         # a restart reads its records from the log
    started_again.world.slim_driver = lambda: "drive"
    assert started_again.tick() == [] and len(slim.started) == 2


def test_the_run_beside_starts_for_no_candidate_above_another_or_whose_check_runs_and_a_dry_run_neither_starts_nor_judges(env):
    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    slim = Slim()
    loop = env.loop(vqueue.decide, slim=slim, batch=1)
    loop.world.slim_driver = lambda: "drive"
    (one,) = loop.tick()
    (two,) = loop.tick()
    assert (one["members"], two["members"], two["base"]) == ([1], [2], one["sha"])
    keys = env.listed(two["sha"], {"pytest:b": ["b.txt"]})
    env.result(keys["pytest:b"], "pytest:b", "pass")                # above a candidate
    keys = env.listed(one["sha"], {"pytest:a": ["a.txt"], "pytest:z": ["z.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")                # its check still runs
    assert loop.tick() == [] and slim.started == [] and slim.judged == []
    env.result(keys["pytest:z"], "pytest:z", "pass")
    dry = env.loop(vqueue.decide, slim=slim, batch=1, dry_run=True)        # its world has no Driver: a start or judge raises
    assert dry.tick() == [] and slim.started == [] and slim.judged == [] and "beside: pending" in "\n".join(env.out)
    (started,) = loop.tick()
    assert (started["action"], started["sha"], slim.started) == ("slim", one["sha"], [("drive", one["sha"], main)])
    slim.verdicts["run-1"] = PASS
    assert _acts(loop.tick()) == ["slim", "advance", "on_main", "drop"]
    (above,) = loop.tick()                                          # the lowest now, its check passed
    assert (above["action"], above["sha"], slim.started[-1]) == ("slim", two["sha"], ("drive", two["sha"], one["sha"]))


def test_a_dropped_candidate_s_run_beside_its_check_is_cancelled_while_it_runs_and_one_that_ended_is_left(env):
    env.put(1, {"a.txt": "1\n"})
    slim = Slim()
    loop = env.loop(vqueue.decide, slim=slim)
    loop.world.slim_driver = lambda: "drive"
    (built,) = loop.tick()
    keys = env.listed(built["sha"], {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "pass")
    (started,) = loop.tick()
    env.o.gh.ready.discard((1, built["heads"][0]))                  # its ready mark withdrawn: dropped
    (drop,) = loop.tick()
    assert (started["action"], drop["action"], drop["beside_cancelled"]) == ("slim", "drop", ["run-1"])
    assert env.cancelled == [(["run-1"], f"vqueue: candidate {built['sha'][:12]} was dropped: #1's ready mark was withdrawn")]
    env.put(2, {"b.txt": "2\n"})
    (two,) = loop.tick()
    keys = env.listed(two["sha"], {"pytest:b": ["b.txt"]})
    env.result(keys["pytest:b"], "pytest:b", "pass")
    assert _acts(loop.tick()) == ["slim"]
    slim.verdicts["run-2"] = FAIL
    env.o.gh.ready.discard((2, two["heads"][0]))
    judged, drop = loop.tick()
    assert (judged["record"]["verdict"], drop["action"], len(env.cancelled)) == (FAIL, "drop", 1) and "beside_cancelled" not in drop


def test_a_candidate_older_than_the_log_s_window_keeps_its_tries_first_enqueue_results_set_aside_and_run_beside(env):
    env.put(1, {"a.txt": "1\n"})
    slim, seen = Slim(), []
    loop = env.loop(lambda s: seen.append(s) or vqueue.decide(s), slim=slim, pending_s=200 * 3600)
    loop.world.slim_driver = lambda: "drive"
    (built,) = loop.tick()
    keys = env.listed(sha := built["sha"], {"pytest:a": ["a.txt"]})
    env.result(keys["pytest:a"], "pytest:a", "error", infra=True)
    (retry,) = loop.tick()
    assert (retry["action"], retry["counts"], retry["priority"]) == ("retry", {keys["pytest:a"]: 1}, built["priority"])
    env.clock += 49 * 3600                                          # its build and its retry: past the window
    assert loop.tick() == [] and slim.started == []
    c = seen[-1].candidates[0]
    assert (c.sha, c.verdict.state, c.tries, c.enqueued) == (sha, PENDING, 2, T0)
    env.result(keys["pytest:a"], "pytest:a", "pass")
    assert _acts(loop.tick()) == ["slim"]
    env.clock += 49 * 3600                                          # its run beside too
    assert loop.tick() == [] and len(slim.started) == 1
    c = seen[-1].candidates[0]
    assert (c.verdict.state, c.tries, c.enqueued) == (PENDING, 2, T0)


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
    (built,) = loop.tick()
    red = {"pytest:r": "sha512:red"}
    env.listed(main, {"pytest:r": ["red.txt"]}, keys=red)
    keys = env.listed(built["sha"], {"pytest:a": ["a.txt"], "pytest:r": ["red.txt"]}, keys=red)
    env.result("sha512:red", "pytest:r", "fail")
    assert loop.tick() == [] and slim.started == []                 # `pytest:a` still runs: the fail hasn't ended
    env.result(keys["pytest:a"], "pytest:a", "pass")
    assert _acts(loop.tick()) == ["slim"]                           # it reads as a pass: the run beside starts
    assert loop.tick() == []                                        # and still runs: the fail hasn't ended
    slim.verdicts["run-1"] = PASS
    assert _acts(loop.tick()) == ["slim", "red", "advance", "on_main", "drop"] and env.main() == built["sha"]


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
    said = capsys.readouterr().out
    assert e.value.code == 0 and all(o in said for o in ("--shadow", "--lean", "--pending-hours", "--no-slim"))
    assert vqueue_loop._slim() is vqueue_slim
