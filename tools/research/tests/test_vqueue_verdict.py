"""`research queue vqueue --gate verdict` (`research.vqueue_loop.VERDICT_GATE`): a plain candidate lands on its members' land
verdicts and the quick tier on its own merge, `main`'s whole check follows it, and a red of that check reverts the one PR it
names or, naming none, sends plain candidates through the whole check until it passes.  The loop of `test_vqueue_loop` (its
`Env` and its faked check, `Runs`), with the quick tier faked on the same Landing (`Quick`)."""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

import research.vqueue_loop as L
from research import vqueue, vqueue_slim
from research.digest import REVERT_SUBJECT
from research.merge import REVERTS
from research.queue import DRIVE_FLAGS
from test_queue import AUTHORED, NAMED, OWN, _git, _who
from test_vqueue_loop import NEW, Env, Runs, _acts, _main_to, _move, _row

#: drive.toml on `main` in these tests: the low-risk lane on, keeping Lean and the verifiers off it, and those two off the
#: quick tier where the land verdicts gate
DRIVE = 'lowrisk_lane = true\nlowrisk_excludes = ["*.lean", "verity/core/"]\nverdict_excludes = ["*.lean", "verity/core/"]\n'
#: a change failing suite pkg's new test
FAILING = {"pkg/tests/test_x.py": "def test_new():\n    assert False\n"}


@pytest.fixture
def env(tmp_path, monkeypatch) -> Env:
    e = Env(tmp_path, monkeypatch)
    _main_to(e, {DRIVE_FLAGS: DRIVE})
    return e


class Quick:
    """The quick tier faked under the real `vqueue_slim.start_quick` and its judge on ``runs``' Landing: each launch is run `q-N`,
    running until `end` writes its result.json as `quick_tier.py` writes it."""

    def __init__(self, loop: L.Loop, runs: Runs):
        self.runs, self.launches = runs, []
        d = loop.world.slim_driver()
        d.launch_quick = self.launch
        d.fetch = lambda rid, names: (runs.tmp / rid, {}) if (runs.tmp / rid / "result.json").is_file() else None

    def launch(self, repo, commit: str, base: str, machine: str):
        self.launches.append((commit, base, machine))
        self.runs.state[rid := f"q-{len(self.launches)}"] = ("running", None, None)
        return [rid], ""

    def end(self, sha: str, rows: tuple[dict, ...] = ()) -> None:
        n, (_, base, _) = [(n, x) for n, x in enumerate(self.launches, 1) if x[0] == sha][-1]
        (d := self.runs.tmp / f"q-{n}").mkdir(parents=True, exist_ok=True)
        failed = any(r["status"] == "failed" for r in rows)
        facts = {"kind": "quick-tier/v1", "head": sha, "base": base, "dirty": [], "lean": [], "changed": True,
                 "rc": 1 if failed else 0, "rows": list(rows), "logs": "out/logs", "error": None}
        (d / "result.json").write_text(json.dumps({"schema": "research/result/v0.1", "kind": "quick-tier/v1", "run_id": f"q-{n}",
                                                   "validation": {"status": "failed" if failed else "passed"}, "tier": facts}))
        self.runs.state[f"q-{n}"] = ("done", 1 if failed else 0, None)


def _verdict(env: Env, monkeypatch, decide=vqueue.decide, gate: str | None = None, **kw) -> tuple[L.Loop, Runs, Quick]:
    """The loop under ``gate`` (`--gate verdict` unless given) with its check and quick tier faked."""
    loop = env.loop(decide, slim=vqueue_slim, gate=gate or L.VERDICT_GATE, **kw)
    runs = Runs(env, loop, monkeypatch)
    return loop, runs, Quick(loop, runs)


def _shards_of(runs: Runs, sha: str) -> list[dict]:
    return [kw for c, _, _, kw in runs.launches if c == sha]


def _landed(env: Env, monkeypatch, members: dict[int, dict[str, str]], **kw) -> tuple[L.Loop, Runs, Quick, str, str]:
    """``members`` built as one plain candidate that lands on the quick tier, `main`'s check below it passed: the loop, its
    fakes, the `main` it landed on and the candidate, now `main`, whose own check has started."""
    below = env.main()
    for n, files in members.items():
        env.put(n, files)
    loop, runs, quick = _verdict(env, monkeypatch, **kw)
    _, built, _ = loop.tick()
    sha = built["sha"]
    runs.end(below)
    quick.end(sha)
    assert "advance" in _acts(loop.tick()) and env.main() == sha
    assert _acts(loop.tick()) == ["slim"] and len(_shards_of(runs, sha)) == 3
    return loop, runs, quick, below, sha


def test_under_gate_verdict_a_plain_candidate_lands_on_its_land_verdicts_and_the_quick_tier_and_main_s_whole_check_follows(
        env, monkeypatch):
    """A candidate of two PRs, each with its own land verdict at its head and a low-risk change, runs the quick tier on its own
    merge, a `research run` on one of --on's machines, in place of the whole check: once it passed, the candidate lands, while
    `main`'s whole check still runs.  The next tick starts `main`'s whole check on the landed tip, every step of
    `AFTER_LANDING` as the --shards shards, against itself, logged as `main`'s, one record for the tip."""
    from research.landing import AFTER_LANDING

    main = env.main()
    env.put(1, {"a.txt": "1\n"})
    env.put(2, {"b.txt": "2\n"})
    loop, runs, quick = _verdict(env, monkeypatch)
    below, built, started = loop.tick()
    sha = built["sha"]
    assert below["record"]["gate"] == "check" and below["sha"] == main
    assert started["record"]["gate"] == vqueue_slim.QUICK and started["record"]["ids"] == ["q-1"]
    assert started["why"] == "started: the quick tier, its members' land verdicts gating it"
    assert quick.launches == [(sha, main, f"m{int(sha[:8], 16) % 3 + 1}")] and _shards_of(runs, sha) == []
    assert _acts(loop.tick()) == [] and env.main() == main                      # it runs
    quick.end(sha)
    logged = loop.tick()
    assert _acts(logged) == ["slim", "advance", "on_main", "drop"] and env.main() == sha
    assert logged[0]["record"]["verdict"] == {"state": "pass", "passed": ["pytest"]}
    after = loop.tick()
    assert _acts(after) == ["slim"] and (after[0]["sha"], after[0]["base"], after[0]["members"]) == (sha, sha, [])
    assert after[0]["why"].startswith("started beside `main` after a landing on the quick tier")
    assert after[0]["record"]["gate"] == "check" and after[0]["record"]["groups"] == [["c-4"], ["c-5"], ["c-6"]]
    assert (after[0]["record"]["tip"], after[0]["record"]["base"]) == (sha, sha)
    assert [(c, kw["shard"]) for c, _, steps, kw in runs.launches[3:]] == [(sha, f"{k}/3") for k in (1, 2, 3)]
    assert all(steps == ",".join(AFTER_LANDING) for c, _, steps, _ in runs.launches)
    assert loop.tick() == [] and len(runs.launches) == 6                       # one record per landed tip
    runs.end(sha)
    assert [r["why"] for r in loop.tick()] == ["judged pass beside `main`"]
    assert not [r for r in env.lines() if r["action"] in ("revert", "fallback")] and not env.posts


@pytest.mark.parametrize("how", ["a Lean change", "a verdict_excludes path", "no land verdict of its own"])
def test_under_gate_verdict_a_candidate_that_is_not_plain_still_takes_the_whole_check(env, monkeypatch, how):
    """A PR changing Lean or a path drive.toml's `verdict_excludes` names, or one taken on its ready mark alone with no land
    verdict at its head, makes its candidate take the whole check before `main`, as under `--gate check`: no quick tier."""
    files = {"a Lean change": {"Lib/A.lean": "def a := 1\n"}, "a verdict_excludes path": {"verity/core/v.py": "x = 1\n"},
             "no land verdict of its own": {"a.txt": "1\n"}}[how]
    head = env.put(1, files)
    if how == "no land verdict of its own":
        env.labels.verdicts[1, head] = []
    loop, runs, quick = _verdict(env, monkeypatch)
    _, built, started = loop.tick()
    sha = built["sha"]
    assert built["members"] == [1] and started["record"]["gate"] == "check" and quick.launches == []
    assert [kw["shard"] for kw in _shards_of(runs, sha)] == ["1/3", "2/3", "3/3"]


@pytest.mark.parametrize("what, files, gate", [
    ("the root's tests", {"tests/test_tree.py": "x = 1\n"}, vqueue_slim.QUICK),
    ("the circuit language", {"verity/circuits/program.py": "x = 1\n"}, vqueue_slim.QUICK),
    ("a Lean source", {"Lib/A.lean": "def a := 1\n"}, "check"),
    ("a lock", {"pkg/lean-audit.json": "{}\n"}, "check"),
    ("a move", {"c.txt": None, "moved/c.txt": "c\n", "a.txt": "moved/c\n"}, "check"),
    ("the loop", {"tools/research/src/research/vqueue_loop.py": "x = 1\n"}, "check"),
], ids=["the root's tests", "the circuit language", "a Lean source", "a lock", "a move", "the loop"])
def test_under_gate_verdict_what_is_plain_is_what_the_tree_s_verdict_excludes_leaves_whatever_lowrisk_excludes_names(
        env, monkeypatch, what, files, gate):
    """`main`'s drive.toml holds this tree's own `verdict_excludes`, and a `lowrisk_excludes` naming the root's tests and the
    circuit language, as the low-risk lane's does.  A PR with a land verdict of its own that changes either is plain all the
    same, and its candidate takes the quick tier: the gate reads `verdict_excludes` alone.  One changing a Lean source, a
    lock or the loop's own code, which `verdict_excludes` names, takes the whole check, as does a move, whatever it touches."""
    real = tomllib.loads((Path(__file__).resolve().parents[3] / DRIVE_FLAGS).read_text())["verdict_excludes"]
    _main_to(env, {DRIVE_FLAGS: 'lowrisk_lane = true\nlowrisk_excludes = ["*.lean", "tests/", "verity/circuits/"]\n'
                                f"verdict_excludes = {json.dumps(real)}\n"})
    if what == "a move":
        _move(env, 1, files)
    else:
        env.put(1, files)
    loop, runs, quick = _verdict(env, monkeypatch)
    logged = loop.tick()
    built = next(r for r in logged if r["action"] == "build")
    started = next(r for r in logged if r["action"] == "slim" and r["sha"] == built["sha"])
    assert built["members"] == [1] and started["record"]["gate"] == gate
    assert (len(quick.launches), len(_shards_of(runs, built["sha"]))) == ((1, 0) if gate == vqueue_slim.QUICK else (0, 3))


def test_under_gate_verdict_a_plain_candidate_whose_quick_tier_fails_is_a_red_of_its_own_and_ejects_its_member(env, monkeypatch):
    """The quick tier fails suite pkg's new test: the candidate fails on step pytest as a check's failure of it reads, so its
    member, whose change meets the failing test, is ejected and the other built again, as under `--gate check`."""
    main = env.main()
    head = env.put(1, FAILING)
    env.put(2, {"a.txt": "1\n"})
    loop, runs, quick = _verdict(env, monkeypatch)
    _, built, _ = loop.tick()
    sha = built["sha"]
    runs.end(main)
    quick.end(sha, (_row(NEW),))
    logged = loop.tick()
    assert _acts(logged)[:4] == ["slim", "slim", "eject", "drop"] and env.main() == main and env.withdrawn == [(1, head)]
    (failed,) = logged[1]["record"]["verdict"]["failed"]
    assert (failed["task"], failed["tests"], failed["named"], failed["kind"]) == (
        "step pytest", [f"pkg::{NEW}"], ["pkg/tests/test_x.py"], "own")


def test_under_gate_verdict_a_quick_tier_that_ended_in_infra_is_started_again_on_another_machine(env, monkeypatch):
    """A plain candidate's quick tier ends with no report of the tier: infra, no red.  The loop starts it again, as it starts
    a check again, on another of the queue's machines than the one it ended so on, and the candidate lands on that one."""
    below = env.main()
    env.put(1, {"a.txt": "1\n"})
    loop, runs, quick = _verdict(env, monkeypatch)
    _, built, started = loop.tick()
    sha = built["sha"]
    runs.end(below)
    (d := runs.tmp / "q-1").mkdir(parents=True)
    (d / "result.json").write_text("{}")
    runs.state["q-1"] = ("done", 1, None)
    logged = loop.tick()
    assert _acts(logged)[-2:] == ["retry", "slim"] and env.main() == below
    again = logged[-1]["record"]
    assert (again["gate"], again["ids"]) == (vqueue_slim.QUICK, ["q-2"]) and logged[-1]["why"].startswith("started again")
    (first, second) = (m for _, _, m in quick.launches)
    assert first == started["record"]["on"] and second == again["on"] and second != first
    quick.end(sha)
    assert "advance" in _acts(loop.tick()) and env.main() == sha


def test_under_gate_verdict_a_plain_candidate_falls_back_to_the_whole_check_once_a_member_has_no_land_verdict(env, monkeypatch):
    """A member's land verdict is gone from its head while its candidate's quick tier runs: the candidate isn't plain, so its
    whole check starts in the quick tier's place."""
    env.put(1, {"a.txt": "1\n"})
    head = env.put(2, {"b.txt": "2\n"})
    loop, runs, quick = _verdict(env, monkeypatch, batch=2)
    _, built, started = loop.tick()
    sha = built["sha"]
    assert started["record"]["gate"] == vqueue_slim.QUICK
    env.labels.verdicts[2, head] = []
    logged = [r for r in loop.tick() if r.get("sha") == sha and r["action"] == "slim"]
    assert logged and logged[0]["record"]["gate"] == "check" and len(_shards_of(runs, sha)) == 3


@pytest.mark.parametrize("rules, who", [(None, OWN), (AUTHORED, NAMED)], ids=["as its own", "as the rules' author"])
def test_under_gate_verdict_a_red_of_main_s_check_naming_one_pr_reverts_it_and_holds_the_ready_prs_meeting_its_failure(
        env, monkeypatch, rules, who):
    """Two PRs land on the quick tier; `main`'s whole check on their tip then fails suite pkg's new test, whose reads only #2's
    change meets.  The loop reverts #2's merge on `main` (`git revert -m 1`, git's message, which drive's readers read),
    pushed as it lands a candidate, with a `revert` line saying why; rings ci and #2's owner; and holds the ready PR whose
    change meets what the failure read until `main`'s check passes again, building the other.  Once the revert's check
    passes, the held PR is built.  The revert is the queue's own commit, or the author's the rules at `main` name
    (`Rules.author`), as its merges are."""
    if rules:
        _main_to(env, {"ci/queue.toml": rules})
    seen: list = []
    loop, runs, quick, below, sha = _landed(env, monkeypatch, {1: {"a.txt": "1\n"}, 2: FAILING},
                                            decide=lambda s: seen.append(s) or vqueue.decide(s))
    merges = dict(line.split(" ", 1) for line in _git(env.o.origin, "log", "--first-parent", "--format=%H %s", f"{below}..{sha}")
                  .splitlines())
    two = next(m for m, s in merges.items() if s.startswith("Queue: #2"))
    runs.end(sha, 2, "pytest", rows=(_row(NEW),))
    env.put(3, {"pkg/tests/test_z.py": "def test_z():\n    pass\n"})
    env.put(4, {"c.txt": "1\n"})
    logged = loop.tick()
    assert _acts(logged) == ["slim", "revert"]
    rev = logged[1]
    assert (rev["red"], rev["passed"], rev["number"], rev["merge"], rev["landed"]) == (sha, below, 2, two, [1, 2])
    assert rev["why"] == "#2's change meets what step pytest read, and no other's landed since" and rev["done"]
    assert env.main() == rev["sha"] and _git(env.o.origin, "rev-parse", f"{rev['sha']}^@").split() == [sha]
    assert _who(env.o.origin, rev["sha"]) == _who(env.o.origin, two) == who
    message = _git(env.o.origin, "log", "-1", "--format=%B", rev["sha"])
    assert REVERT_SUBJECT.match(message) and REVERT_SUBJECT.match(message)[1] == "2" and REVERTS.findall(message) == [two]
    assert f"changes made to {_git(env.o.origin, 'rev-parse', f'{two}^1')}." in message
    files = _git(env.o.origin, "ls-tree", "-r", "--name-only", rev["sha"]).split()
    assert "pkg/tests/test_x.py" not in files and "a.txt" in files
    assert len(env.posts) == 1 and env.posts[0].endswith(rev["text"]) and rev["to"][0] == "@ci"
    assert "reverted as" in rev["text"] and "#3" in rev["text"] and "#4" not in rev["text"]
    logged = loop.tick()                                                      # the revert's own check starts, #4 is built
    built = [r for r in logged if r["action"] == "build"]
    assert [r["members"] for r in built] == [[4]] and "revert" not in _acts(logged)
    held = {n: why for n, _, why in seen[-1].left}
    assert held[3].startswith(f"held: `main`'s check at {sha[:12]} failed step pytest") and 4 not in held
    runs.end(rev["sha"])
    loop.tick()
    assert 3 not in {n for n, _, _ in seen[-1].left} and 3 in {x.number for x in seen[-1].ready}
    assert len([r for r in env.lines() if r["action"] == "revert"]) == 1 and len(env.posts) == 1


def test_under_gate_verdict_a_ready_pr_held_for_main_s_red_has_a_leave_line_as_any_marked_pr_left_out_has(env, monkeypatch):
    """A ready PR whose change meets what `main`'s red check read is held out of every candidate, and the loop logs it as it
    logs any marked PR its candidates leave out: one `leave` line at its head, with a cause that stays the same while the
    red holds and the reason its owner reads, so the time it waits is counted under that cause."""
    loop, runs, quick, below, sha = _landed(env, monkeypatch, {1: {"a.txt": "1\n"}, 2: FAILING}, left_out=vqueue.left_out)
    runs.end(sha, 2, "pytest", rows=(_row(NEW),))
    h3 = env.put(3, {"pkg/tests/test_z.py": "def test_z():\n    pass\n"})
    env.put(4, {"c.txt": "1\n"})
    loop.tick()
    loop.tick()
    (leave,) = [r for r in env.lines() if r["action"] == "leave"]
    assert (leave["number"], leave["heads"], leave["cause"]) == (3, [h3], L.MAIN_FAILED)
    assert leave["why"].startswith(f"held: `main`'s check at {sha[:12]} failed step pytest")


@pytest.mark.parametrize("how", ["two PRs meet what it read", "nothing bounds what it read"])
def test_under_gate_verdict_a_red_of_main_s_check_naming_no_one_pr_reverts_none_and_sends_plain_candidates_through_the_check(
        env, monkeypatch, how):
    """`main`'s whole check on two PRs landed on the quick tier fails where no one PR can be named: both changes meet what suite
    pkg read, or a step failed whose reads nothing bounds.  Nothing is reverted, a `fallback` line says why, ci is rung with
    the failed steps once, no PR is held, and a plain candidate now takes the whole check before `main`, until `main`'s check
    passes again."""
    loop, runs, quick, below, sha = _landed(env, monkeypatch, {1: {"pkg/tests/test_y.py": "def test_y():\n    pass\n"}, 2: FAILING})
    if how == "two PRs meet what it read":
        runs.end(sha, 2, "pytest", rows=(_row(NEW),))
    else:
        runs.end(sha, 1, "lean-build", log="error: build failed\n")
    env.put(3, {"pkg/tests/test_z.py": "def test_z():\n    pass\n"})
    logged = loop.tick()
    assert _acts(logged)[:2] == ["slim", "fallback"] and env.main() == sha
    fallback = logged[1]
    assert (fallback["red"], fallback["passed"], fallback["members"]) == (sha, below, [1, 2]) and fallback["done"]
    assert fallback["why"] == ("#1, #2 each meet what they read" if how == "two PRs meet what it read" else
                               "nothing bounds what step lean-build read")
    assert len(env.posts) == 1 and env.posts[0].endswith(fallback["text"]) and fallback["to"] == ["@ci"]
    assert "No PR is reverted" in fallback["text"] and ("step pytest" in fallback["text"] or "step lean-build" in fallback["text"])
    built = next(r for r in logged if r["action"] == "build")
    started = next(r for r in logged if r["action"] == "slim" and r["sha"] == built["sha"])
    assert built["members"] == [3] and started["record"]["gate"] == "check" and len(quick.launches) == 1
    assert not [r for r in env.lines() if r["action"] == "revert"]
    assert _acts(loop.tick()) == [] and len(env.posts) == 1                     # rung once
    runs.end(built["sha"])
    assert "advance" in _acts(loop.tick()) and env.main() == built["sha"]
    env.put(4, {"c.txt": "1\n"})
    logged = loop.tick()
    started = next(r for r in logged if r["action"] == "slim" and r["members"] == [4])
    assert started["record"]["gate"] == vqueue_slim.QUICK                      # `main`'s check passed: plain again


def _two_tips(env: Env, monkeypatch, first: dict[str, str], **kw) -> tuple[L.Loop, Runs, Quick, str, str, str, dict]:
    """#1 (``first``) and then #2 landed on the quick tier, one tick apart, so that two of `main`'s own checks run at once:
    the loop, its fakes, the `main` they landed on (its check passed), the two tips, and the line that started the newer
    tip's check."""
    loop, runs, quick, below, one = _landed(env, monkeypatch, {1: first}, **kw)
    env.put(2, {"a.txt": "1\n"})
    built = next(r for r in loop.tick() if r["action"] == "build")
    quick.end(two := built["sha"])
    assert "advance" in _acts(loop.tick()) and env.main() == two
    [started] = loop.tick()
    return loop, runs, quick, below, one, two, started


@pytest.mark.parametrize("earlier", ["still runs", "passed"])
def test_under_gate_verdict_main_s_check_leaves_suites_to_an_earlier_tip_s_and_says_nothing_until_that_one_passed_them(
        env, monkeypatch, earlier):
    """Two landings on the quick tier a tick apart: the newer tip's own check starts while the earlier tip's still runs, and
    leaves to it the suites keyed there as here (``suites_from``, as the earlier left its to the passed check below it).
    Its own shards pass.  While the earlier check runs it is read as a check still running: its record keeps its verdict but
    no ``suites_passed``, `main` isn't red, and a candidate stacked on a candidate on it is lent no suite, which only a
    `main` whose check passed both suite steps allows.  Once the earlier check passed, judged below the tip, the record
    says so and the stacked candidate's check leaves its suites to its base's."""
    loop, runs, quick, below, one, two, started = _two_tips(env, monkeypatch, {"b.txt": "1\n"}, batch=1, depth=2)
    assert [r["record"].get("suites_from") for r in env.lines() if r["action"] == "slim" and r["sha"] == one][-1] == below
    assert started["why"].startswith("started beside `main` after a landing on the quick tier")
    assert (started["sha"], started["record"]["suites_from"], started["record"]["gate"]) == (two, one, "check")
    assert len(_shards_of(runs, two)) == 3
    runs.end(two)
    if earlier == "passed":
        runs.end(one)
    logged = loop.tick()
    mine = [r for r in logged if r["action"] == "slim" and r["sha"] == two][-1]["record"]
    assert mine["verdict"]["state"] == "pass" and mine.get("suites_passed", False) is (earlier == "passed")
    assert [r["why"] for r in logged if r["sha"] == one] == (
        ["judged pass beside `main`'s line, below its tip"] if earlier == "passed" else [])
    assert loop.red is None and loop.tick() == []
    env.put(3, {"verity/core/v.py": "x = 1\n"})
    env.put(4, {"verity/core/w.py": "x = 1\n"})
    for _ in range(2):
        loop.tick()
    on, top = [r for r in env.lines() if r["action"] == "slim" and r.get("members") in ([3], [4])]
    assert (on["base"], top["base"], on["record"]["gate"], top["record"]["gate"]) == (two, on["sha"], "check", "check")
    assert "suites_from" not in on["record"]
    assert top["record"].get("suites_from") == (on["sha"] if earlier == "passed" else None)
    assert not [r for r in env.lines() if r["action"] in ("revert", "fallback")] and len(_shards_of(runs, two)) == 3


@pytest.mark.parametrize("newer", ["still runs", "passed its own steps"])
def test_under_gate_verdict_a_red_of_the_earlier_tip_s_check_is_its_own_and_the_newer_tip_s_check_runs_its_suites_itself(
        env, monkeypatch, newer):
    """The earlier tip's check fails suite pkg's new test, which #1 added, after the newer tip's check left its suites to
    it.  The red is the earlier check's, of a suite it ran: read at the earlier tip over the landings up to it, it names #1,
    whose change its failure meets, and #2, which landed above it and changed no input of a suite the newer check left out,
    is no suspect.  The newer tip's check never reads as passed: it is started again, its runs cancelled where they still
    ran, with no lender, and runs every suite itself; the revert's own check leaves its suites to that one."""
    loop, runs, quick, below, one, two, started = _two_tips(env, monkeypatch, FAILING)
    assert started["record"]["suites_from"] == one and started["record"]["ids"] == ["c-7", "c-8", "c-9"]
    if newer == "passed its own steps":
        runs.end(two)
        assert [r["why"] for r in loop.tick()] == ["judged pass beside `main`"] and loop.red is None
    runs.end(one, 2, "pytest", rows=(_row(NEW),))
    logged = loop.tick()
    assert _acts(logged) == ["slim", "slim", "revert"] and [r["sha"] for r in logged[:2]] == [one, two]
    assert logged[0]["why"] == "judged fail beside `main`'s line, below its tip"
    again, rev = logged[1:]
    assert again["why"] == (f"started again beside `main`: {one[:12]}'s check, which it left suites to, didn't pass them: it "
                            "runs them itself")
    assert "suites_from" not in again["record"] and again["record"]["ids"] == ["c-10", "c-11", "c-12"]
    assert again.get("beside_cancelled") == (None if newer == "passed its own steps" else ["c-7", "c-8", "c-9"])
    assert [ids for ids, _ in env.cancelled] == ([] if newer == "passed its own steps" else [["c-7", "c-8", "c-9"]])
    assert (rev["red"], rev["passed"], rev["number"], rev["landed"]) == (one, below, 1, [1])
    assert rev["why"] == "#1's change meets what step pytest read, and no other's landed since" and env.main() == rev["sha"]
    [after] = [r for r in loop.tick() if r["action"] == "slim" and r["sha"] == rev["sha"]]
    assert after["record"]["suites_from"] == two                              # the one that runs every suite lends them


def test_under_gate_verdict_a_red_of_an_earlier_tip_s_check_is_not_hidden_by_the_pass_of_one_above_that_left_suites_to_it(
        env, monkeypatch):
    """Three landings.  The second tip's check left its suites to the first's and passed its own steps; the third's left
    its to the second's.  Then the first tip's check fails suite pkg's new test.  The second's pass, below the tip now,
    says nothing of the suites it left out, so `main` is not read green there: the red is read at the first tip and names
    #1, alone landed up to it.  The tip's check, whose chain of lenders ends in that fail, is started again with none."""
    loop, runs, quick, below, one, two, _ = _two_tips(env, monkeypatch, FAILING)
    runs.end(two)
    assert [r["why"] for r in loop.tick()] == ["judged pass beside `main`"]
    env.put(3, {"c.txt": "1\n"})
    built = next(r for r in loop.tick() if r["action"] == "build")
    quick.end(three := built["sha"])
    assert "advance" in _acts(loop.tick()) and env.main() == three
    started = loop.tick()[0]
    assert (started["sha"], started["record"]["suites_from"]) == (three, two) and loop.red is None
    runs.end(one, 2, "pytest", rows=(_row(NEW),))
    logged = loop.tick()
    assert [(r["action"], r["sha"]) for r in logged[:2]] == [("slim", one), ("slim", three)]
    assert logged[1]["why"].startswith(f"started again beside `main`: {two[:12]}'s check, which it left suites to, didn't")
    assert "suites_from" not in logged[1]["record"]
    [rev] = [r for r in logged if r["action"] == "revert"]
    assert (rev["red"], rev["passed"], rev["number"], rev["landed"]) == (one, below, 1, [1])
    assert not any(r["record"].get("suites_passed") for r in env.lines() if r["action"] == "slim" and r["sha"] == two)


def test_under_gate_verdict_an_earlier_tip_s_check_that_ended_in_infra_leaves_the_newer_tip_s_to_run_its_suites_itself(
        env, monkeypatch):
    """The earlier tip's check is lost (a shard killed by a signal: infra) after the newer tip's check left its suites to it.
    Below the tip it is not started again, so nothing will have run those suites: the newer tip's check is started again
    with no lender and runs them itself.  No red: an infra names no PR below the tip, and the check that passes is one that
    ran every suite."""
    loop, runs, quick, below, one, two, started = _two_tips(env, monkeypatch, {"b.txt": "1\n"})
    runs.end(two)
    loop.tick()
    runs.end(one, 2, "pytest", "error: no\n")
    runs.state["c-5"] = ("done", 137, None)
    logged = loop.tick()
    assert [(r["sha"], r["why"].split(":")[0]) for r in logged if r["action"] == "slim"] == [
        (one, "judged infra beside `main`'s line, below its tip"), (two, "started again beside `main`")]
    assert "suites_from" not in logged[1]["record"] and len(_shards_of(runs, two)) == 6 and env.cancelled == []
    assert loop.red is None and loop.tick() == []
    runs.end(two)
    [judged] = loop.tick()
    assert judged["why"] == "judged pass beside `main`" and "suites_passed" not in judged["record"] and loop.red is None
    assert not [r for r in env.lines() if r["action"] in ("revert", "fallback")]


@pytest.mark.parametrize("then", ["`main`'s check passes them", "`main`'s check fails one"])
def test_under_gate_verdict_a_candidate_that_left_suites_to_a_base_that_landed_on_the_quick_tier_waits_for_main_s_check(
        env, monkeypatch, then):
    """#1 has no land verdict yet, so its candidate takes the whole check, and the candidate of #2 on it, never plain,
    leaves its suites to that check.  The check is lost (infra); #1's land verdict has arrived, so its candidate is started
    again on the quick tier and lands on it: its whole check judged no suite.  #2's candidate, its own steps passed, doesn't
    land with it, nor after: it waits for `main`'s own check of that tip, which runs the suites it left out or leaves them
    in turn to a passed check below.  Once that passed both suite steps it lands; where that fails one, its check is started
    again with no lender and runs them itself."""
    main = env.main()
    head = env.put(1, FAILING if then.endswith("fails one") else {"a.txt": "1\n"})
    env.labels.verdicts[1, head] = []
    env.put(2, {"verity/core/v.py": "x = 1\n", "tools/verity/check/common.py": "# the auditor imports this\n"})
    loop, runs, quick = _verdict(env, monkeypatch, batch=1, depth=2)
    loop.tick()
    runs.end(main)
    loop.tick()
    started = {r["sha"]: r for r in env.lines() if r["action"] == "slim" and r["why"] == "started" and r["sha"] != main}
    (one, base), (top, lent) = started.items()
    assert (base["record"]["gate"], lent["record"]["gate"], lent["record"]["suites_from"], lent["base"]) == (
        "check", "check", one, one)
    assert "lean_audit_from" not in lent["record"]                  # it audits itself: it waits on no other verdict
    runs.end(one, 2, "pytest", "error: no\n")
    runs.state["c-5"] = ("done", 137, None)
    del env.labels.verdicts[1, head]
    again = [r for r in loop.tick() if r["action"] == "slim" and r["sha"] == one][-1]
    assert again["why"].startswith("started again") and again["record"]["gate"] == vqueue_slim.QUICK
    runs.end(top)
    quick.end(one)
    assert _acts(loop.tick()).count("advance") == 1 and env.main() == one       # only its base lands
    logged = loop.tick()
    own = next(r for r in logged if r["action"] == "slim" and r["sha"] == one)
    assert (own["why"], own["record"]["gate"], own["record"]["suites_from"]) == ("started again beside `main`", "check", main)
    assert env.main() == one and _acts(loop.tick()) == ["slim"] and loop.tick() == [] and env.main() == one     # it waits
    assert not [r for r in env.lines() if r["sha"] == top and r["why"].startswith("started again")]
    if then.endswith("passes them"):
        runs.end(one)
        assert "advance" in _acts(loop.tick()) and env.main() == top
        assert "suites_from" in [r for r in env.lines() if r["action"] == "slim" and r["sha"] == top][-1]["record"]
        return
    runs.end(one, 2, "pytest", rows=(_row(NEW),))
    logged = loop.tick()
    [redo] = [r for r in logged if r["action"] == "slim" and r["sha"] == top]
    assert redo["why"] == (f"started again: {one[:12]}'s check, which it left suites to, landed without passing them: it runs "
                           "them itself")
    assert "suites_from" not in redo["record"] and "beside_cancelled" not in redo
    assert top not in _git(env.o.origin, "rev-list", env.main()).split()


def test_under_gate_check_a_red_of_main_s_check_reverts_nothing_holds_nothing_and_quick_tiers_nothing(env, monkeypatch):
    """The flag off: under `--gate check` the same PRs take the whole check, land on it, and a red of `main`'s own check after is
    what it is today: no revert, no fallback, no hold, no quick tier, `main` unmoved, and the PR meeting the failure built."""
    below = env.main()
    env.put(1, {"a.txt": "1\n"})
    env.put(2, FAILING)
    loop, runs, quick = _verdict(env, monkeypatch, gate=L.CHECK_GATE)
    _, built, started = loop.tick()
    sha = built["sha"]
    assert started["record"]["gate"] == "check" and quick.launches == []
    runs.end(below)
    runs.end(sha)
    assert "advance" in _acts(loop.tick()) and env.main() == sha
    _git(env.o.origin, "update-ref", "refs/heads/main", main := env.o.change({"z.txt": "z\n"}))
    env.listed(main, {})
    assert _acts(loop.tick()) == ["slim"]
    runs.end(main, 2, "pytest", rows=(_row(NEW),))
    env.put(3, {"pkg/tests/test_z.py": "def test_z():\n    pass\n"})
    logged = loop.tick()
    assert _acts(logged) == ["slim", "build", "slim"] and env.main() == main
    assert logged[1]["members"] == [3] and logged[2]["record"]["gate"] == "check" and quick.launches == []
    assert not {"revert", "fallback"} & {r["action"] for r in env.lines()} and not env.posts
    assert not [r for r in env.lines() if re.search("quick tier|held: `main`", r.get("why") or "")]
