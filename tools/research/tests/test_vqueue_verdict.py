"""`research queue vqueue --gate verdict` (`research.vqueue_loop.VERDICT_GATE`): a plain candidate lands on its members' land
verdicts and the quick tier on its own merge, `main`'s whole check follows it, and a red of that check reverts the one PR it
names or, naming none, sends plain candidates through the whole check until it passes.  The loop of `test_vqueue_loop` (its
`Env` and its faked check, `Runs`), with the quick tier faked on the same Landing (`Quick`)."""
from __future__ import annotations

import json
import re

import pytest

import research.vqueue_loop as L
from research import vqueue, vqueue_slim
from research.digest import REVERT_SUBJECT
from research.merge import REVERTS
from research.queue import DRIVE_FLAGS
from test_queue import _git
from test_vqueue_loop import NEW, Env, Runs, _acts, _main_to, _row

#: drive.toml on `main` in these tests: the low-risk lane on, keeping Lean and the verifiers off it
DRIVE = 'lowrisk_lane = true\nlowrisk_excludes = ["*.lean", "verity/core/"]\n'
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


@pytest.mark.parametrize("how", ["a Lean change", "a lowrisk_excludes path", "no land verdict of its own"])
def test_under_gate_verdict_a_candidate_that_is_not_plain_still_takes_the_whole_check(env, monkeypatch, how):
    """A PR changing Lean or a path drive.toml's `lowrisk_excludes` names, or one taken on its ready mark alone with no land
    verdict at its head, makes its candidate take the whole check before `main`, as under `--gate check`: no quick tier."""
    files = {"a Lean change": {"Lib/A.lean": "def a := 1\n"}, "a lowrisk_excludes path": {"verity/core/v.py": "x = 1\n"},
             "no land verdict of its own": {"a.txt": "1\n"}}[how]
    head = env.put(1, files)
    if how == "no land verdict of its own":
        env.labels.verdicts[1, head] = []
    loop, runs, quick = _verdict(env, monkeypatch)
    _, built, started = loop.tick()
    sha = built["sha"]
    assert built["members"] == [1] and started["record"]["gate"] == "check" and quick.launches == []
    assert [kw["shard"] for kw in _shards_of(runs, sha)] == ["1/3", "2/3", "3/3"]


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


def test_under_gate_verdict_a_red_of_main_s_check_naming_one_pr_reverts_it_and_holds_the_ready_prs_meeting_its_failure(
        env, monkeypatch):
    """Two PRs land on the quick tier; `main`'s whole check on their tip then fails suite pkg's new test, whose reads only #2's
    change meets.  The loop reverts #2's merge on `main` (`git revert -m 1`, git's message, which drive's readers read),
    pushed as it lands a candidate, with a `revert` line saying why; rings ci and #2's owner; and holds the ready PR whose
    change meets what the failure read until `main`'s check passes again, building the other.  Once the revert's check
    passes, the held PR is built."""
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
