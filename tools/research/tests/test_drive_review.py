"""The review landing (`land = "review"`, here with `land_into = "main"`: PRs land on review, and the full check runs on
`main` after; `test_drive_checked.py` has `land_into = "checked"`, where `main` moves only to a checked commit).  A tick
lands `next`'s longest prefix whose every member has a land verdict at its head by a reviewer who isn't its owner, a Lean
change as any other, with no Lean check of its own; the after-landing run checks `main`, and its failure reverts what it
implicates once `main` passed that step before, bisecting when several PRs are implicated."""
from __future__ import annotations

import contextlib
import calendar
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

import research.queue as Q
from research import merge
from research.drive import AFTER_LANDING, AFTER_LANDING_OPTIONAL
from research.queue import PR, QueueError, StoreLabels, label_target
from research.store import vocab
from test_queue import World, _git
from test_queue_drive import Harness, _failed_run, _killed_run, _steps_run, _tip

EVERY = {s: "passed" for s in AFTER_LANDING if s not in AFTER_LANDING_OPTIONAL}
#: a ci.toml with a job for each after-landing step, circuit-check off the merge gate as in the tree
CI_TOML = 'schema = "ci/v0"\n' + "".join(f'[[job]]\nname = "{s}"\nowner = "ci"\ngroup = 1\n'
                                         + ('gate = false\n' if s == "circuit-check" else "") for s in AFTER_LANDING)


def test_the_after_landing_run_runs_circuit_check_which_is_off_the_merge_gate():
    """circuit-check is off the merge gate (ci.toml's `gate = false`): no recorded check runs it, so the after-landing run must,
    and must pass it to verify `main` (it is no step the run may skip by name)."""
    from research.drive import CHECK_STEPS

    assert AFTER_LANDING == tuple(s for s in CHECK_STEPS if not s.startswith("preflight-"))
    assert "circuit-check" in AFTER_LANDING and "circuit-check" not in AFTER_LANDING_OPTIONAL and "circuit-check" in EVERY


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_main_waits_for_the_steps_on_the_gate_of_the_commits_own_ci_toml(world, monkeypatch):
    """`_promote` waits at a commit for every after-landing step but those a run skips by name and the jobs that commit's own
    ci.toml takes off the merge gate (circuit-check); for every such step when that ci.toml can't be read."""
    h = Review(world, monkeypatch)
    every = [s for s in AFTER_LANDING if s not in AFTER_LANDING_OPTIONAL]
    assert h.d._gating(_main(world)) == every
    _move_main(world, {"tools/verity/check/ci.toml": CI_TOML})
    h.q.fetch()
    assert h.d._gating(_main(world)) == [s for s in every if s != "circuit-check"]
    _move_main(world, {"tools/verity/check/ci.toml": "not toml ["})
    h.q.fetch()
    assert h.d._gating(_main(world)) == every


def _registry(world: World, *handles: str) -> None:
    reg = {"handles": [{"handle": h, "holder": f"the {h} lane"} for h in handles]}
    main = world.change({".agents/skills/using-slack/registry.json": json.dumps(reg)})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")


class Review(Harness):
    """A drive landing on reviews, whose verdicts are `reviewed` labels in the world's store.  Unless ``baseline`` is off, ci has
    recorded `main` green (`research queue baseline`), so a tick lands without first running the after-landing run on `main`."""

    def __init__(self, world, monkeypatch, baseline: bool = True, **kw):
        super().__init__(world, monkeypatch, **kw)
        self.d.reviews = lambda pr, head: world.store.labels(label_target(pr, head), key=vocab.REVIEWED)
        self.d.ready_by = lambda pr, head: next((lab.by for lab in sorted(world.store.labels(label_target(pr, head), key=vocab.READY),
                                                                          key=lambda lab: lab.ts, reverse=True)), None)
        if baseline:
            assert Q.baseline(self.q, _git(world.origin, "rev-parse", "main"), "ci") == 0

    def review(self, pr: int, verdict: str = "land", by: str = "reviewer", ref: str | None = None) -> None:
        self.world.store.add_label(label_target(pr, self.world.gh.prs[pr].head), vocab.REVIEWED, verdict, by, ref=ref)

    def postland(self) -> dict:
        return self.d.q.load_state()["drive"]["postland"]


def _subject(world: World, ref: str) -> str:
    return _git(world.origin, "log", "-1", "--format=%s", ref)


def _main(world: World) -> str:
    return _git(world.origin, "rev-parse", "main")


def _move_main(world: World, files: dict[str, str]) -> None:
    """A commit onto `main` that isn't a landing of drive's (the lander's `research merge`), so the next sync rebuilds `next`."""
    _git(world.dev, "push", "-q", "origin", f"{world.change(files)}:refs/heads/main")


def test_a_reviewed_prefix_lands_and_awaits_the_after_landing_run_of_every_step(world, monkeypatch):
    h = Review(world, monkeypatch)
    h.d.after_landing_no_cache = True
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.review(1)
    h.review(2)
    assert h.tick().startswith("landed #2"), world.logs
    main = _main(world)
    assert _subject(world, "main").startswith("Queue: #2") and _subject(world, "main^1").startswith("Queue: #1")
    assert _subject(world, "next").startswith("Queue: #3") and _git(world.origin, "rev-parse", "next^1") == main
    assert h.launched == []
    verified = _git(world.origin, "rev-parse", "main^1^1")
    assert h.posted == [(main, ",".join(AFTER_LANDING), {"agreement": False, "no_cache": True, "circuit_cache": True,
                                                            "no_regenerate": True, "base": verified})]
    assert h.after and h.after[0][1] == {main: "", _git(world.origin, "rev-parse", "main^1"): ""}
    pl = h.postland()
    assert pl["pending"] == [main] and pl["run"]["skip"] == ["lean-agreement", "lean-changed"]
    assert pl["run"]["as"] == {"steps": [s for s in AFTER_LANDING if s not in AFTER_LANDING_OPTIONAL], "no_cache": True}
    h.run_state["r-post-0"] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, "r-post-0", EVERY)]
    h.tick()
    pl = h.postland()
    assert pl["verified"] == main and pl["verified_as"]["no_cache"] is True
    assert "lean-agreement" in pl["verified_as"]["steps"]         # ci's baseline passed it, and nothing it checks changed since


def test_a_sharded_after_landing_run_verifies_main_only_once_every_shard_passed(world, monkeypatch):
    """`after_landing_shards = N`: the after-landing run is N runs at once, shard K of ci.toml's plan with N shards each, so it
    takes about its longest shard rather than every step on one slot (2.5 h). `main` is verified once each shard passed; a shard
    that never started launches the run again, whole."""
    h = Review(world, monkeypatch)
    h.d.after_landing_shards = 3
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    main = _main(world)
    assert [(c, kw.get("shard")) for c, _, kw in h.posted] == [(main, "1/3"), (main, "2/3"), (main, "3/3")]
    h.run_state.update({"r-post-0": ("done", 0, None), "r-post-1": ("running", None, None), "r-post-2": ("done", 0, None)})
    h.evidence = [_steps_run(world.tmp, "r-post-0", {s: "passed" for s in AFTER_LANDING})]
    h.tick()
    pl = h.postland()
    assert pl["run"]["groups"] == [["r-post-0"], ["r-post-1"], ["r-post-2"]] and pl["verified"] != main, "one shard runs on"
    h.run_state["r-post-1"] = ("done", 75, None)
    h.tick()
    assert h.postland()["verified"] != main and len(h.posted) == 6, "a shard that never started: the run again, whole"
    h.run_state.update({f"r-post-{i}": ("done", 0, None) for i in (3, 4, 5)})
    h.tick()
    pl = h.postland()
    assert pl["verified"] == main and pl["run"] is None, world.logs


def test_an_after_landing_run_sends_a_base_below_its_tip_so_ratchet_gate_judges_the_landings_above_it(world, monkeypatch):
    """Against origin/main, its own tip, the after-landing run's ratchet-gate judged no change and could never fail.  Drive
    sends the last verified `main` as its base (`train.sh --base`), below the tip with every landing since above it; the
    baseline run, of the verified `main` itself, sends the tip's first parent."""
    _move_main(world, {"x.txt": "x\n"})
    h = Review(world, monkeypatch, baseline=False)
    verified = _main(world)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.review(1)
    h.review(2)
    assert h.tick().startswith("nothing lands before an after-landing run of `main`"), world.logs
    ((tip, _, kw),) = h.posted
    assert tip == verified and kw["base"] == _git(world.origin, "rev-parse", f"{verified}^1")
    h.run_state["r-post-0"] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, "r-post-0", {s: "passed" for s in AFTER_LANDING})]
    assert h.tick().startswith("landed #2"), world.logs
    tip, _, kw = h.posted[-1]
    assert tip == _main(world) != verified and kw["base"] == verified == h.postland()["verified"]
    assert _git(world.origin, "merge-base", kw["base"], tip) == verified                   # below the tip
    assert [m.pr for m in h.d.landed_since(kw["base"], tip)] == [1, 2]                    # the landings it judges


def test_a_lean_change_and_a_lock_change_land_on_their_verdicts_with_no_lean_check_of_their_own(world, monkeypatch):
    """A member that changes Lean or a lock lands on its verdict as any other: drive launches no Lean check of it and waits for
    none, and the after-landing run checks what landed."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/Pkg/A.lean": "def a := 1\n"}))
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{}"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3"), world.logs
    assert h.launched == [] and [c for c, _, _ in h.posted] == [_main(world)], h.posted


def test_a_pr_whose_preflight_onto_next_refuses_it_is_told_and_stays_out_while_the_members_above_it_land(world, monkeypatch):
    _registry(world, "lane-2")
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
        h.review(n)
    h.q.preflight = lambda pr, base, commit, members: "a.py: SyntaxError" if pr.number == 2 else None
    assert h.tick().startswith("landed #3"), world.logs
    assert [_subject(world, r)[:9] for r in ("main", "main^1")] == ["Queue: #3", "Queue: #1"]
    (post,) = [p for p in h.posts if "out of `next`" in p]
    assert f"#2 at {world.gh.prs[2].head[:8]} is out of `next`: a.py: SyntaxError" in post and "@lane-2" in post, post


def test_an_unreviewed_member_new_to_next_or_a_held_one_stops_the_prefix_below_it(world, monkeypatch):
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.review(1)
    h.review(3)
    assert h.tick().startswith("landed #1"), world.logs        # the lift moves only what `next` held before the sync
    assert _subject(world, "main").startswith("Queue: #1")
    h.review(2)
    h.review(2, "hold", by="other", ref="reads a stale fixture")
    out = h.tick()
    assert out.startswith("nothing reviewed lands: #2") and "held by other (reads a stale fixture)" in out, out
    assert _subject(world, "main").startswith("Queue: #1")
    h.review(2, by="other")                                     # each reviewer's newest verdict counts
    assert h.tick().startswith("landed #3")
    assert _main(world) == _tip(world)
    assert h.launched == []


def test_no_verdict_counts_by_the_owner_its_handle_or_its_ready_marks_author_whatever_the_case(world, monkeypatch):
    world.with_store_remote()
    _registry(world, "lane-1")
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.store.add_label(label_target(1, world.gh.prs[1].head), vocab.READY, "true", "marker")
    for by in ("Lane-1", "@LANE-1", "Marker"):                  # #1's `Lane lane-1`, its handle, its ready mark's author
        h.review(1, by=by)
        out = h.tick()
        assert "no land verdict but its owner's own" in out and _subject(world, "main") == "change ['.agents/skills/using-slack/registry.json']", out
    with pytest.raises(QueueError, match="#1 is MARKER's own PR"):
        Q.review(h.q, world.store, StoreLabels(world.store, log=lambda s: None), 1, "land", None, "MARKER")
    h.review(1, by="@lane-7")
    assert h.tick().startswith("landed #1")


def test_an_unreviewed_member_below_reviewed_ones_no_longer_stops_them_landing(world, monkeypatch):
    """A member with no land verdict held every reviewed member above it, deep in `next`.  The members with a land verdict
    go below those waiting for one, and land that tick; the one waiting holds only itself."""
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2, 3]
    h.review(2)
    h.review(3)
    assert h.tick().startswith("landed #3"), world.logs
    assert _subject(world, "main").startswith("Queue: #3") and _subject(world, "main^1").startswith("Queue: #2")
    assert [m.pr for m in h.q.chain()] == [1]
    assert any("#2 #3 (reviewed) go to the bottom" in x for x in world.logs), world.logs
    h.review(1)
    assert h.tick().startswith("landed #1") and _main(world) == _tip(world)


def test_a_held_member_still_stops_the_members_above_it_a_hold_is_a_verdict(world, monkeypatch):
    """Nothing moves past a held member: what is below it lands, past an unreviewed one, and what is above it waits."""
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt"), (4, "d.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    h.review(2)
    h.review(3, "hold", by="other", ref="reads a stale fixture")
    h.review(4)
    assert h.tick().startswith("landed #2"), world.logs
    assert [m.pr for m in h.q.chain()] == [1, 3, 4]
    out = h.tick()
    assert out.startswith("nothing reviewed lands: #1"), out
    assert [m.pr for m in h.q.chain()] == [1, 3, 4] and _subject(world, "main").startswith("Queue: #2")


def test_reviewed_first_keeps_a_stacked_pr_after_the_pr_it_contains(world, monkeypatch):
    """A reviewed PR stacked on an unreviewed one stays above it; one stacked on a reviewed one goes down after it."""
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}, base=world.gh.prs[1].head))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    world.gh.put(4, world.change({"d.txt": "4\n"}, base=world.gh.prs[3].head))
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2, 3, 4]
    for n in (2, 3, 4):
        h.review(n)
    assert h.tick().startswith("landed #4"), world.logs
    assert _subject(world, "main").startswith("Queue: #4") and _subject(world, "main^1").startswith("Queue: #3")
    assert [m.pr for m in h.q.chain()] == [1, 2]
    h.review(1)
    assert h.tick().startswith("landed #2") and _main(world) == _tip(world)


def test_a_member_whose_labels_cant_be_read_stops_the_lift_not_the_tick(world, monkeypatch):
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    h.review(2)
    h.review(3)
    reviews = h.d.reviews

    def stale(pr, head):
        if pr == 3:
            raise QueueError(f"STALE: the labels of pr:3@{head} weren't pulled from the store's remote")
        return reviews(pr, head)

    h.d.reviews = stale
    assert h.tick().startswith("landed #2"), world.logs
    assert [m.pr for m in h.q.chain()] == [1, 3]
    assert any("the lift stops below #3: STALE" in x for x in world.logs), world.logs


def test_reviewed_first_off_keeps_an_unreviewed_member_where_it_is(world, monkeypatch):
    h = Review(world, monkeypatch, reviewed_first=False)
    for n, f in ((1, "a.txt"), (2, "b.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    h.review(2)
    assert h.tick().startswith("nothing reviewed lands: #1"), world.logs
    assert [m.pr for m in h.q.chain()] == [1, 2]


def test_a_pr_re_marked_at_its_ejected_head_lands(world, monkeypatch):
    """The queue's record of an ejection outlives a re-mark at the same head (`Queue._marked_again` re-admits it): read as
    withdrawn, it held the PR, and every PR above it, off `main` for good."""
    h = Review(world, monkeypatch)
    world.gh.put(1, head := world.change({"a.txt": "1\n"}))
    h.review(1)
    h.q.sync()
    h.q.eject(1, "x", "ci")
    world.gh.ready.discard((1, head))
    assert not h.tick().startswith("landed")
    world.gh.ready.add((1, head))                                           # its owner marks it again at that head
    h.q.marked_at = lambda n, hd: "2099-01-01T00:00:00Z"
    assert h.tick().startswith("landed #1"), world.logs


def test_nothing_lands_before_the_after_landing_run_of_main_itself_and_a_red_main_lands_reverting_nothing_on_its_red_step(
        world, monkeypatch, tmp_path):
    """A first after-landing run fails on anything `main` already had: reverting the landings it follows would revert innocent
    PRs.  So the review landing's first run is of `main` itself, and nothing lands until it ends.  Failed, `main` is verified
    with the steps it passed alone: landing goes on (a red `main` cleared only by a false baseline wedged it), and a later
    failure of a step red then reverts nothing."""
    _registry(world, "lane-2")
    h = Review(world, monkeypatch, baseline=False)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    h.review(1)
    h.review(2)
    main = _main(world)
    out = h.tick()
    assert out.startswith("nothing lands before an after-landing run of `main`"), out
    below = _git(world.origin, "rev-parse", f"{main}^1")
    assert h.posted == [(main, ",".join(AFTER_LANDING), {"agreement": True, "no_cache": False, "circuit_cache": False,
                                                            "no_regenerate": True, "base": below})]
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_steps_run(world.tmp, "r-post-0", {**{s: "passed" for s in AFTER_LANDING}, "lean-suites": "failed"})]
    assert h.tick().startswith("landed #2"), world.logs
    (post,) = [p for p in h.posts if "baseline after-landing run" in p]
    assert "was red before any review landing" in post and "until a run of `main` passes lean-suites" in post, post
    assert h.postland()["verified"] == main and "lean-suites" not in h.postland()["verified_as"]["steps"]
    tip = _main(world)
    h.run_state["r-post-1"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    assert _main(world) == tip
    (post,) = [p for p in h.posts if "Reverted nothing" in p]
    assert "suite pkg is red" in post and "It never passed in an after-landing run since" in post, post
    assert "@lane-2" not in post, post


def test_ci_records_a_baseline_only_on_the_control_pod_and_it_clears_an_open_bisection(world, monkeypatch):
    """Off the control pod `research queue baseline` wrote a state file there that drive never reads, and said nothing."""
    from research.merge import DRIVE_ENV

    h = Review(world, monkeypatch, baseline=False)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    main = _main(world)
    h.q._clone()
    state = h.q.load_state()
    state["drive"] = {"launched": {}, "told": [], "postland": {"verified": main, "pending": [], "run": None, "bisect": {
        "tip": main, "base": main, "steps": ["pytest"], "why": "x", "key": "step pytest", "no_cache": False, "cands": [], "good": -1, "bad": 0,
        "run": {"ids": ["r-gone"], "i": 0, "at": 0}}}}
    h.q.save_state(state)
    argv = ["baseline", main, "--by", "ci", "--workdir", str(h.q.workdir), "--remote", str(world.origin), "--repo", "o/r",
            "--store", str(world.tmp / "store"), "--rules", str(world.dev / "ci" / "queue.toml")]
    monkeypatch.delenv(DRIVE_ENV, raising=False)
    assert Q.main(argv) == 1 and "bisect" in h.postland()
    with pytest.raises(QueueError, match="only ci"):
        Q.baseline(h.q, main, "lane-3")
    monkeypatch.setenv(DRIVE_ENV, "1")
    with pytest.raises(QueueError, match="not after-landing steps"):
        Q.baseline(h.q, main, "ci", "nonsense")
    assert Q.main([*argv, "--steps", "lean-audit"]) == 0 and h.postland()["verified_as"]["steps"] == ["lean-audit"]
    assert Q.main(argv) == 0 and "bisect" not in h.postland()
    assert h.tick().startswith("landed #1")


def _owning(world: World, owns: dict[str, list[str]]) -> None:
    """`main`'s registry, each handle owning its paths (`research msg owners`)."""
    reg = {"handles": [{"handle": h, "holder": f"the {h} lane", "owns": paths} for h, paths in owns.items()]}
    _git(world.dev, "push", "-q", "origin", f"{world.change({'.agents/skills/using-slack/registry.json': json.dumps(reg)})}"
                                          ":refs/heads/main")


def _red_landing(world, monkeypatch, tmp_path) -> tuple[Review, str]:
    """#1, #2 and #3 land; #2 breaks suite pkg (owned by @pkg-owner), whose after-landing run `r-post-0` fails."""
    _owning(world, {"lane-2": [], "pkg-owner": ["pkg/"]})
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3")
    tip = _main(world)
    assert [c for c, *_ in h.posted] == [tip]
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    return h, tip


def test_an_after_landing_failure_rings_its_culprit_and_the_suites_owner_and_reverts_it_an_hour_later(world, monkeypatch,
                                                                                                       tmp_path):
    """Without a culprit, PRs keep landing on a `main` whose after-landing runs all fail, each failure going to @ci alone.
    Drive names the one landing whose change touches the failing test, rings its owner and the suite's with the run and the first
    error line, and reverts it once `FIX_WINDOW_S` passed with no fix marked."""
    from research.drive import FIX_WINDOW_S

    h, tip = _red_landing(world, monkeypatch, tmp_path)
    assert _main(world) == tip
    (ring,) = [p for p in h.posts if "is red on" in p]
    assert ring.startswith("@ci @pkg-owner @lane-2 queue: suite pkg is red on `main`") and "(run r-post-0)" in ring, ring
    assert "failed tests/test_x.py::test_y" in ring and "NameError: name '_once' is not defined" in ring, ring
    assert "Culprit: #2 (its change touches pkg/tests/test_x.py)" in ring and "unless a fix is marked" in ring, ring
    assert h.postland()["red"]["suite pkg"]["culprit"][0] == 2
    h.clock += FIX_WINDOW_S - 1
    h.tick()
    assert _main(world) == tip and not [p for p in h.posts if "reverted" in p]
    h.clock += 2
    h.tick()
    main = _main(world)
    assert _git(world.origin, "rev-parse", f"{main}^") == tip                            # one revert on top of the landing
    assert "pkg/tests/test_x.py" not in _git(world.origin, "ls-tree", "-r", "--name-only", main)
    (post,) = [p for p in h.posts if "reverted" in p]
    assert post.startswith("@ci @lane-2 queue: #2 reverted on `main`") and "no fix was marked within 60 min" in post, post
    assert h.posted[-1][0] == main and h.launched == []                                  # the next run checks the new tip
    h.run_state[f"r-post-{len(h.posted) - 1}"] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, f"r-post-{len(h.posted) - 1}", {s: "passed" for s in h.posted[-1][1].split(",")})]
    (h.evidence[0] / "suites-lean.json").write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "passed"}]))
    h.tick()
    assert h.postland()["verified"] == main and not h.postland()["red"] and h.postland()["green"]["suite pkg"] == main


def test_a_red_ratchet_gate_names_the_merge_it_fails_as_the_culprit(world, monkeypatch, tmp_path):
    """The after-landing run's ratchet-gate step runs the vLLM integration's ratchet gate on each landing's merge, and its log
    names no file of the change, only the PR whose merge grows an allowlist: drive takes that PR as the culprit.  The log
    here is the gate's own, run on the merges drive landed."""
    import sys

    p01 = "integrations/vllm/tests/lint/allowlists/p01_core.json"

    def allowlist(*files: str) -> str:
        return json.dumps({"rule": "p01", "entries": [{"file": f, "kind": "k", "symbol": "s", "detail": "d"} for f in files]})

    _git(world.dev, "push", "-q", "origin", f"{world.change({p01: allowlist('a.py', 'b.py')})}:refs/heads/main")
    h = Review(world, monkeypatch)
    base = _main(world)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"c.py": "C = 1\n", p01: allowlist("a.py", "b.py", "c.py")}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3")
    tip = _main(world)
    merge2 = _git(world.origin, "rev-parse", "main^1")
    assert _subject(world, merge2).startswith("Queue: #2")
    _git(world.dev, "fetch", "-q", "origin")
    gate = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[3] / "integrations/vllm/tools/ratchet_gate.py"),
                           base, tip], cwd=world.dev, capture_output=True, text=True, env={**os.environ, "RATCHET_GATE_CACHE": ""})
    assert gate.returncode == 1 and f"  #2 ({merge2[:12]}): failed: grows {p01} 2 -> 3 entries" in gate.stdout, gate
    run = _steps_run(tmp_path, "r-post-0", {s: "passed" for s in EVERY if s != "ratchet-gate"})
    with open(run / "steps.jsonl", "a") as f:
        f.write(json.dumps({"step": "ratchet-gate", "status": "failed", "rc": 1, "log": "ratchet-gate.log"}) + "\n")
    (run / "ratchet-gate.log").write_text(gate.stdout)
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [run]
    h.tick()
    (ring,) = [p for p in h.posts if "is red on" in p]
    assert f"step ratchet-gate is red on `main` {tip[:12]}" in ring and f"`#2 ({merge2[:12]}): failed: grows {p01}" in ring, ring
    assert "Culprit: #2 (the ratchet gate fails its merge), of the 3 landing(s)" in ring, ring
    assert h.postland()["red"]["step ratchet-gate"]["culprit"][:1] == [2]


@pytest.mark.parametrize("fix", ["names-the-culprit", "fixes-suite", "touches-the-failure", "new-head"])
def test_a_marked_fix_holds_the_culprits_revert_while_it_stays_marked(world, monkeypatch, tmp_path, fix):
    """A fix is a ready PR whose body names the culprit or says `fixes suite NAME`, or whose change touches what the failure
    named, or a new head of the culprit; each body case changes an unrelated file, so only its body can count."""
    from research.drive import FIX_WINDOW_S

    h, tip = _red_landing(world, monkeypatch, tmp_path)
    if fix == "new-head":
        world.gh.put(n := 2, head := world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}))
    elif fix == "touches-the-failure":
        world.gh.put(n := 9, head := world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}), body="Lane `lane-9`.")
    else:
        world.gh.put(n := 9, head := world.change({"elsewhere.txt": "9\n"}),
                     body="Lane `lane-9`. " + ("FIXES suite pkg: test_y." if fix == "fixes-suite" else "Fixes #2's test."))
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    assert _main(world) == tip and any(f"a fix is marked, #{n} at {head[:8]}" in x for x in world.logs), world.logs
    world.gh.ready.discard((n, head))                                   # its owner withdrew the fix: the clock ran out
    h.tick()
    assert _git(world.origin, "rev-parse", "main^") == tip and any("#2 reverted on `main`" in p for p in h.posts), h.posts


def test_a_landed_fix_holds_the_culprits_revert_until_a_run_of_a_main_with_it_is_red_again(world, monkeypatch, tmp_path):
    """A fix counted only while it is an open PR would have drive ask for the culprit's revert after its fix landed.  A fix that landed after the red tip
    holds the revert, by its merge's title; once a run of a `main` with it is red again, it no longer does."""
    from research.drive import FIX_WINDOW_S

    h, tip = _red_landing(world, monkeypatch, tmp_path)
    world.gh.put(9, world.change({"elsewhere.txt": "9\n"}))
    world.gh.prs[9] = Q.PR(9, world.gh.prs[9].head, "fixes suite pkg", "Lane `lane-9`.")
    h.review(9)
    assert h.tick().startswith("landed #9"), world.logs
    fixed = _main(world)
    del world.gh.prs[9]                                                  # landed: no open PR holds the revert any more
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    assert _main(world) == fixed and any("a fix is marked, #9, landed at" in x for x in world.logs), world.logs
    h.run_state[f"r-post-{len(h.posted) - 1}"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path / "again", "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    h.tick()
    assert any("#2 reverted on `main`" in p for p in h.posts), (h.posts, world.logs)


def test_an_unrelated_pr_that_names_the_red_suite_holds_no_revert(world, monkeypatch, tmp_path):
    """Counting any ready PR whose body names the suite would let a suite called `research` be held
    for good by whatever PR mentions research.  Neither the name nor `suite research` without `fixes` holds the revert."""
    from research.drive import FIX_WINDOW_S

    _owning(world, {"lane-2": [], "ci": ["tools/research/"]})
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"tools/research/tests/test_x.py": "def test_y():\n    assert False\n"}))
    for n in (1, 2):
        h.review(n)
    assert h.tick().startswith("landed #2")
    tip = _main(world)
    run = tmp_path / "run"
    run.mkdir()
    (run / "steps.jsonl").write_text(json.dumps({"step": "pytest", "status": "failed"}) + "\n")
    (run / "suites.json").write_text(json.dumps([{"name": "research", "suite": "tools/research", "status": "failed",
                                                  "failed_tests": ["tests/test_x.py::test_y"]}]))
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [run]
    h.tick()
    (ring,) = [p for p in h.posts if "is red on" in p]
    assert "suite research is red" in ring and "Culprit: #2" in ring and "says `fixes suite research`" in ring, ring
    world.gh.put(9, world.change({"docs/notes.txt": "9\n"}),
                 body="Lane `lane-9`. research: the research store's docs, as suite research reads them.")
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    assert _git(world.origin, "rev-parse", "main^") == tip and any("#2 reverted on `main`" in p for p in h.posts), h.posts
    assert not any("a fix is marked, #" in x for x in world.logs), world.logs


def _one_red(world, monkeypatch, tmp_path) -> tuple[Review, str]:
    """#1 breaks suite pkg and lands; its after-landing run `r-post-0` fails, ringing #1 as its culprit."""
    _registry(world, "lane-1")
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    tip = _main(world)
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    assert h.postland()["red"]["suite pkg"]["culprit"][0] == 1
    return h, tip


def _reverts(world: World, since: str) -> int:
    return len(_git(world.origin, "log", "--format=%H", "--grep=^This reverts commit", f"{since}..main").split())


def test_a_culprits_revert_that_conflicts_leaves_main_and_tells_its_owner_to_revert_by_hand(world, monkeypatch, tmp_path):
    from research.drive import FIX_WINDOW_S

    h, tip = _one_red(world, monkeypatch, tmp_path)
    _move_main(world, {"pkg/tests/test_x.py": "def test_y():\n    assert False  # edited on main\n"})
    moved = _main(world)
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    assert _main(world) == moved
    (post,) = [p for p in h.posts if "must be reverted by hand" in p]
    assert post.startswith("@ci @lane-1 queue: #1 must be reverted by hand") and "the revert didn't apply" in post, post


def test_a_culprit_reverted_by_hand_is_not_reverted_again_nor_told_as_a_hand_revert(world, monkeypatch, tmp_path):
    from research.drive import FIX_WINDOW_S

    h, tip = _one_red(world, monkeypatch, tmp_path)
    merge_commit = _git(world.origin, "log", "--format=%H", "--grep=^PR: #1$", "main").split()[0]
    _git(world.dev, "fetch", "-q", "origin")
    _git(world.dev, "checkout", "-q", "--detach", "origin/main")
    _git(world.dev, "revert", "--no-edit", "-m", "1", merge_commit)
    _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    fixed = _main(world)
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    assert _main(world) == fixed and _reverts(world, tip) == 1
    assert h.postland()["red"]["suite pkg"]["reverted"] == "undone"
    assert not any("by hand" in p or "reverted on `main`" in p for p in h.posts), h.posts


def test_a_revert_whose_push_loses_a_race_is_retried_on_the_next_tick(world, monkeypatch, tmp_path):
    from research.drive import FIX_WINDOW_S

    h, tip = _one_red(world, monkeypatch, tmp_path)
    real, raced = h.q._remote, []

    def remote(repo, *args, **kw):
        if "main:refs/heads/main" in args and not raced:
            _move_main(world, {"c.txt": "landed in between\n"})
            raced.append(_main(world))
        return real(repo, *args, **kw)

    monkeypatch.setattr(h.q, "_remote", remote)
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    assert raced and _main(world) == raced[0] and not any("reverted" in p or "by hand" in p for p in h.posts), h.posts
    h.tick()
    main = _main(world)
    assert _git(world.origin, "rev-parse", f"{main}^") == raced[0] and _reverts(world, tip) == 1
    assert _git(world.origin, "show", f"{main}:c.txt") == "landed in between"
    assert any(p.startswith("@ci @lane-1 queue: #1 reverted on `main`") for p in h.posts), h.posts


def test_a_drive_killed_after_a_revert_pushed_does_not_revert_again_on_restart(world, monkeypatch, tmp_path):
    from research.drive import FIX_WINDOW_S

    h, tip = _one_red(world, monkeypatch, tmp_path)

    def killed(text, to=()):
        raise KeyboardInterrupt("the drive is killed")

    h.clock += FIX_WINDOW_S + 1
    h.d.post = killed
    with pytest.raises(KeyboardInterrupt):
        h.tick()
    assert _reverts(world, tip) == 1
    again = Review(world, monkeypatch, baseline=False)
    again.clock, again.run_state, again.evidence, again.posted = h.clock, h.run_state, h.evidence, h.posted
    again.tick()
    assert _reverts(world, tip) == 1 and not any("by hand" in p for p in again.posts), again.posts


def test_a_drive_restarted_while_its_after_landing_run_runs_reads_it_back_and_verifies_its_tip(world, monkeypatch):
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    tip = _main(world)
    again = Review(world, monkeypatch, baseline=False)                  # a new process: only state.json carries over
    again.run_state, again.posted = h.run_state, h.posted
    again.run_state["r-post-0"] = ("done", 0, None)
    again.evidence = [_steps_run(world.tmp, "r-post-0", {s: "passed" for s in AFTER_LANDING})]
    again.tick()
    assert again.postland()["verified"] == tip and again.postland()["pending"] == [] and len(h.posted) == 1


def test_an_after_landing_run_the_machine_killed_runs_again_once_and_killed_again_is_told(world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    tip = _main(world)
    h.run_state["r-post-0"] = ("done", 137, None)
    h.evidence = [_killed_run(tmp_path, "r-post-0")]
    h.tick()
    assert [c for c, *_ in h.posted] == [tip, tip] and h.postland()["run"]["resend"]
    h.run_state["r-post-1"] = ("done", 137, None)
    h.evidence = [_killed_run(tmp_path / "again", "r-post-1")]
    h.tick()
    assert [p for p in h.posts if "killed again on its resend" in p], h.posts
    assert _main(world) == tip and h.postland()["failed"] == tip


def test_a_red_after_landing_run_runs_on_and_is_cancelled_only_with_its_circuit_check_tail_left(world, monkeypatch, tmp_path):
    """Cancelling an after-landing run at its first failed suite leaves the rest of a ~4 h run unknown, and the cancel kills
    lean-agreement mid-step, which reads as failed.  A red run rings at once and runs on until every step but
    circuit-check and flock-rust-tests has ended, so every red suite is named; a step a cancel stopped reads `not run`."""
    from research.drive import TAIL

    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    run = _failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2", steps="")
    (run / "suites-lean.failed.json").write_text((run / "suites-lean.json").read_text())
    (run / "suites-lean.json").unlink()
    steps = {"lean-build": "passed", "lean-agreement": "stopped"}
    (run / "steps.jsonl").write_text("".join(json.dumps({"step": k, "status": v}) + "\n" for k, v in steps.items()))
    h.run_state["r-post-0"] = ("running", None, None, "suite pkg (tests/test_x.py::test_y)")
    h.evidence = [run]
    h.tick()
    assert h.cancelled == [] and h.postland()["run"]["ids"] == ["r-post-0"]
    (ring,) = [p for p in h.posts if "is red on" in p]
    assert "suite pkg is red" in ring and "Culprit: #1" in ring and "Not run (a cancel stopped them): lean-agreement" in ring
    h.tick()
    assert h.cancelled == [] and len([p for p in h.posts if "is red on" in p]) == 1             # rung once
    steps |= {s: "passed" for s in AFTER_LANDING if s not in TAIL and s not in steps}
    steps["lean-suites"] = "failed"
    (run / "steps.jsonl").write_text("".join(json.dumps({"step": k, "status": v}) + "\n" for k, v in steps.items()))
    h.tick()
    assert h.cancelled == ["r-post-0"] and h.postland()["run"] is None and h.postland()["green"]["step lean-audit"] == _main(world)
    assert "step lean-agreement" not in h.postland()["red"] and "step lean-suites" not in h.postland()["red"]


def test_a_read_guard_failure_rings_the_pr_that_added_the_reading_test(world, monkeypatch, tmp_path):
    """A suite the read guard failed names no failing test: its record's `outside` (path -> the test module that read it) is what
    implicates the PR that added the reading test."""
    _owning(world, {"lane-1": [], "lane-2": [], "pkg-owner": ["pkg/"]})
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/tests/test_reads.py": "def test_r():\n    open('data/x.json')\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    for n in (1, 2):
        h.review(n)
    assert h.tick().startswith("landed #2")
    run = tmp_path / "run"
    run.mkdir()
    (run / "steps.jsonl").write_text(json.dumps({"step": "pytest", "status": "failed"}) + "\n")
    (run / "suites.json").write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "failed",
                                                  "outside": {"data/x.json": "pkg/tests/test_reads.py"}}]))
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [run]
    h.tick()
    (ring,) = [p for p in h.posts if "is red on" in p]
    assert ring.startswith("@ci @pkg-owner @lane-1 queue: suite pkg is red"), ring
    assert "read outside its declared inputs: data/x.json" in ring and "Culprit: #1" in ring, ring


def test_main_far_behind_its_last_verified_commit_rings_ci_and_top_every_hour(world, monkeypatch, tmp_path):
    """A `main` many hours without a verified commit must reach someone.  Once its oldest unverified landing is
    `BEHIND_S` old, drive rings @ci and @top, again every `BEHIND_EVERY_S` while it stays behind."""
    from research.drive import BEHIND_EVERY_S, BEHIND_S

    h, tip = _red_landing(world, monkeypatch, tmp_path)
    world.gh.put(9, world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}), body="Lane `lane-9`. Fixes pkg.")
    h.clock += BEHIND_S - 10
    h.tick()
    assert not [p for p in h.posts if "behind its tip" in p] and _main(world) == tip
    h.clock += 20
    h.tick()
    (alarm,) = [p for p in h.posts if "behind its tip" in p]
    assert alarm.startswith("@ci @top queue: `main`'s last verified commit") and "red: suite pkg (culprit #2)" in alarm, alarm
    h.clock += BEHIND_EVERY_S - 30
    h.tick()
    assert len([p for p in h.posts if "behind its tip" in p]) == 1
    h.clock += 60
    h.tick()
    assert len([p for p in h.posts if "behind its tip" in p]) == 2


def test_a_healthy_after_landing_run_under_way_is_not_an_alarm(world, monkeypatch):
    """A cold after-landing run takes ~4 h, so `main` is routinely more than `BEHIND_S` behind while its run is green so far:
    drive rings only once that run has gone `POSTLAND_RUN_S`."""
    from research.drive import BEHIND_S, POSTLAND_RUN_S

    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    h.clock += BEHIND_S + 60
    h.tick()
    assert not [p for p in h.posts if "behind its tip" in p] and h.postland()["run"]["ids"] == ["r-post-0"]
    h.clock += POSTLAND_RUN_S
    h.tick()
    (alarm,) = [p for p in h.posts if "behind its tip" in p]
    assert "its run r-post-0" in alarm and "red: nothing named red" in alarm, alarm


def test_an_after_landing_failure_several_prs_touch_is_bisected_to_the_one_that_broke_it(world, monkeypatch, tmp_path):
    """Reverting every landing a failure's files reach took innocent PRs with the culprit: each step of the bisection checks one
    landing's commit and reads the failed suite alone, and only the first landing that fails it is rung, then reverted.
    Landing goes on meanwhile: the bisection checks fixed commits."""
    from research.drive import FIX_WINDOW_S

    _registry(world, "lane-1", "lane-2")
    h = Review(world, monkeypatch)
    world.gh.put(1, h1 := world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}, base=h1))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3")
    tip = _main(world)
    one, two = _git(world.origin, "rev-parse", "main^1^1"), _git(world.origin, "rev-parse", "main^1")
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    world.gh.put(4, world.change({"c.txt": "4\n"}))
    h.review(4)
    out = h.tick()
    assert out.startswith("landed #4"), out
    assert h.posted[1] == (one, "lean-suites", {"agreement": False, "no_cache": False, "circuit_cache": False,
                                                   "no_regenerate": True, "base": _git(world.origin, "rev-parse", f"{one}^1")})
    assert h.posted[2][0] == _main(world)                                        # #4's after-landing run
    (ring,) = [p for p in h.posts if "is red on" in p]
    assert "#1 #2 each touch it" in ring and "drive bisects them" in ring, ring
    h.run_state["r-post-1"] = ("done", 1, None)                                  # another suite failed there: pkg passed
    h.evidence = [_steps_run(world.tmp, "r-post-1", {"lean-suites": "failed"})]
    (h.evidence[0] / "suites-lean.json").write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "passed"},
                                                                 {"name": "other", "suite": "other", "status": "failed"}]))
    h.tick()
    assert h.posted[3][0] == two and _main(world) != tip
    h.run_state["r-post-3"] = ("done", 1, None)
    h.evidence = [_failed_run(world.tmp / "r3", "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    assert "bisect" not in h.postland() and not [p for p in h.posts if "reverted" in p]
    (ring,) = [p for p in h.posts if "Culprit: #2, by bisection" in p]
    assert ring.startswith("@ci @lane-2 queue: suite pkg is red") and "passes on #1's landing" in ring, ring
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    main = _main(world)
    revert = _git(world.origin, "rev-list", "--first-parent", "--reverse", f"{tip}..main").split()[-1]
    assert _subject(world, revert).startswith('Revert "Queue: #2')
    assert _git(world.origin, "show", f"{main}:pkg/tests/test_x.py") == "def test_y():\n    pass"
    (post,) = [p for p in h.posts if "reverted" in p]
    assert post.startswith("@ci @lane-2 queue: #2 reverted on `main`"), post


def _bisecting(world, monkeypatch, tmp_path, revert: bool = False) -> tuple[Review, str]:
    """#1 and #2 both change the failing test's file and land with #3; the after-landing run fails there, and drive bisects them
    (its first run, `r-post-1`, of #1's landing).  With ``revert``, a hand revert lands on `main` first."""
    _registry(world, "lane-1", "lane-2")
    h = Review(world, monkeypatch)
    if revert:
        _git(world.dev, "fetch", "-q", "origin")
        _git(world.dev, "checkout", "-q", "--detach", "origin/main")
        _git(world.dev, "commit", "-q", "--allow-empty", "-m", f"Revert \"x\"\n\nThis reverts commit {_main(world)}.")
        _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    world.gh.put(1, h1 := world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}, base=h1))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    for n in (1, 2, 3):
        h.review(n)
    assert h.tick().startswith("landed #3"), world.logs
    tip = _main(world)
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    return h, tip


def test_a_bisection_run_lost_past_the_bound_runs_once_more_then_ends_reverting_nothing(world, monkeypatch, tmp_path):
    """A run whose state can't be read counted as running for ever: while a bisection is open nothing lands, and ci's baseline
    was refused, so `next` stopped until someone edited drive's state by hand."""
    from research.drive import LOST_AFTER_S

    h, tip = _bisecting(world, monkeypatch, tmp_path)
    assert len(h.posted) == 2
    h.run_state["r-post-1"] = ("unknown", None, None)
    h.tick()
    assert len(h.posted) == 2
    h.clock += LOST_AFTER_S + 1
    h.tick()
    assert len(h.posted) == 3 and h.posted[2][0] == h.posted[1][0]       # the same landing once more
    h.run_state["r-post-2"] = ("unknown", None, None)
    h.clock += LOST_AFTER_S + 1
    out = h.tick()
    assert "bisect" not in h.postland() and _main(world) == tip, out
    (post,) = [p for p in h.posts if "ended twice" in p]
    assert post.startswith("@ci queue:") and "(lost)" in post and "Reverted nothing" in post, post


def test_a_bisection_run_fails_a_landing_only_on_a_failed_step_it_bisects(world, monkeypatch, tmp_path):
    """Any run that wasn't clean marked its landing bad, so an exit 0 whose records couldn't be read, or a failure in another
    step (setup, an older tree's ci.toml), reverted an innocent PR."""
    h, tip = _bisecting(world, monkeypatch, tmp_path)
    h.run_state["r-post-1"] = ("done", 0, None)
    h.evidence = []                                                         # exit 0, its steps.jsonl unread
    h.tick()
    assert len(h.posted) == 3 and h.posted[2][0] == h.posted[1][0] and h.postland()["bisect"]["bad"] == 2
    h.run_state["r-post-2"] = ("done", 2, None)
    h.evidence = [_steps_run(world.tmp, "r-post-2", {"setup": "failed"})]
    h.tick()
    assert "bisect" not in h.postland() and _main(world) == tip and not [p for p in h.posts if "reverted" in p]
    (post,) = [p for p in h.posts if "ended twice" in p]
    assert "neither passing nor failing pkg" in post, post


def test_a_bisection_with_a_revert_among_its_landings_reverts_nothing(world, monkeypatch, tmp_path):
    """A revert inside the range means a landing's commit isn't what `main` ran: bisecting it could blame the wrong PR."""
    h, tip = _bisecting(world, monkeypatch, tmp_path, revert=True)
    assert "bisect" not in h.postland() and len(h.posted) == 1 and _main(world) == tip
    (post,) = [p for p in h.posts if "a revert lies in" in p]
    assert "Reverted nothing" in post, post


def test_a_landing_that_does_not_compile_lands_what_is_below_the_pr_that_broke_it(world, monkeypatch):
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/x.py": "def f(:\n"}))
    for n in (1, 2):
        h.review(n)
    assert h.tick().startswith("landed #1"), world.logs
    assert _subject(world, "main").startswith("Queue: #1")
    (post,) = [p for p in h.posts if "doesn't compile" in p]
    assert "#2: pkg/x.py: SyntaxError" in post, post


def test_review_lists_the_members_lacking_a_land_verdict_and_records_one_refusing_the_owner(world, monkeypatch, capsys):
    world.with_store_remote()
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{}"}))
    h.d.review = False
    h.q.sync()
    labels = StoreLabels(world.store, log=lambda s: None)
    assert Q.review(h.q, world.store, labels, 1, "land", "note:ci/x", "lane-9") == 0
    with pytest.raises(QueueError, match="#2 is lane-2's own PR"):
        Q.review(h.q, world.store, labels, 2, "land", None, "lane-2")
    capsys.readouterr()
    assert Q.review(h.q, world.store, StoreLabels(world.store, log=lambda s: None), None, None, None, None, as_json=True) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert (row["pr"], row["owner"], row["files"], row["why"]) == (2, "lane-2", ["pkg/lean-audit.json"], "no land verdict")
    (lab,) = world.store.labels(label_target(1, world.gh.prs[1].head), key=vocab.REVIEWED)
    assert (lab.value, lab.by, lab.ref) == ("land", "lane-9", "note:ci/x")


def test_review_without_sha_refuses_a_head_pushed_since_the_sync_and_takes_a_short_sha(world, monkeypatch):
    world.with_store_remote()
    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.d.review = False
    h.q.sync()
    old = world.gh.prs[1].head
    world.gh.put(1, new := world.change({"a.txt": "2\n"}))
    labels = StoreLabels(world.store, log=lambda s: None)
    with pytest.raises(QueueError, match=f"GitHub has its head at {new[:12]}.*pass --sha"):
        Q.review(h.q, world.store, labels, 1, "land", None, "lane-9")
    assert Q.review(h.q, world.store, labels, 1, "land", None, "lane-9", sha=new[:10]) == 0
    assert Q.review(h.q, world.store, labels, 1, "hold", "old", "lane-9", sha=old[:10]) == 0
    assert [lab.value for lab in world.store.labels(label_target(1, new), key=vocab.REVIEWED)] == ["land"]
    assert [lab.value for lab in world.store.labels(label_target(1, old), key=vocab.REVIEWED)] == ["hold"]
    with pytest.raises(QueueError, match="full 40-character SHA"):
        Q.review(h.q, world.store, labels, 1, "land", None, "lane-9", sha="0badc0de")
    with pytest.raises(QueueError, match="neither #1's head"):                     # a mistyped full SHA
        Q.review(h.q, world.store, labels, 1, "land", None, "lane-9", sha=new[:12] + "0" * 28)
    assert Q.review(h.q, world.store, labels, 1, "land", None, "lane-9", sha=new) == 0


def test_the_owner_of_an_unmarked_pr_a_marked_one_waits_for_is_told_once_per_pair_of_heads(world, monkeypatch):
    """A marked PR can wait on another's unmarked commits, and that one's owner is told, the marked one's beside it.  They hear once per
    (head of the waiting PR, head of the unmarked one), and nothing once it's admitted or when the waiting PR isn't marked."""
    _registry(world, "lane-5", "lane-7", "lane-8", "lane-9")
    h = Review(world, monkeypatch)
    old = world.change({"x.txt": "x = 1\n"})
    world.gh.put(5, old, ready=False)
    h.tick()                                                                    # drive sees #5 at `old`
    world.gh.put(5, world.change({"x.txt": "x = 2\n"}, base=old), ready=False)  # moved on from `old`, unmarked
    world.gh.put(7, world.change({"k.txt": "kernels\n"}, base=old))             # branched from #5 at `old`
    world.gh.put(8, world.change({"y.txt": "y\n"}, base=old), ready=False)      # waits too, but isn't marked: no one hears
    world.gh.put(9, world.change({"z.txt": "z\n"}, base=world.gh.prs[5].head))  # on #5's current head
    world.store.add_label(label_target(7, world.gh.prs[7].head), vocab.READY, "true", "marker7")

    def told() -> list[str]:
        return [p for p in h.posts if "waits for your mark" in p]

    h.tick()
    five = world.gh.prs[5].head
    assert told() == [
        f"@ci @lane-5 @lane-7 queue: #7 (marked by marker7) waits for your mark of #5, which it contains (at {old[:12]}): mark #5 ready at "
        f"{five[:8]}, or withdraw it (`research queue eject 5 --reason ...`) and #7 carries it",
        f"@ci @lane-5 @lane-9 queue: #9 (marked by lane-9) waits for your mark of #5, which it contains (at {five[:12]}): mark #5 ready at "
        f"{five[:8]}, or withdraw it (`research queue eject 5 --reason ...`) and #9 carries it"], told()
    h.tick()
    assert len(told()) == 2                                     # once per pair of heads
    world.gh.put(7, world.change({"k.txt": "kernels 2\n"}, base=world.gh.prs[7].head))
    h.tick()
    assert len(told()) == 3 and told()[-1].startswith("@ci @lane-5 @lane-7 queue: #7 (marked by lane-7) waits for your mark of #5")
    world.gh.put(5, world.change({"x.txt": "x = 3\n"}, base=five), ready=False)
    h.tick()
    # #9 holds #5's older head now; it only shares `old` with #8, which waits for nothing of #9's or #7's (`Queue.entries`)
    assert [p.split(": mark")[0] for p in told()[3:]] == [
        f"@ci @lane-5 @lane-7 queue: #7 (marked by lane-7) waits for your mark of #5, which it contains (at {old[:12]})",
        f"@ci @lane-5 @lane-9 queue: #9 (marked by lane-9) waits for your mark of #5, which it contains (at {five[:12]})"], told()
    world.gh.ready.add((5, world.gh.prs[5].head))
    h.tick()
    assert len(told()) == 5 and h.q.status()[7].state != "waiting"


def test_a_marked_pr_left_out_for_an_ejected_pr_it_holds_an_older_head_of_is_told_once_with_the_fix(world, monkeypatch):
    """#7 branched from #6, which moved on and was ejected at its new head: #7 holds #6's commits at an older head, so the sync
    leaves it off `next`.  Both owners hear once per pair of heads, with the fix, and nothing at the same heads after."""
    _registry(world, "lane-6", "lane-7")
    h = Review(world, monkeypatch)
    old = world.change({"x.txt": "x = 1\n"})
    world.gh.put(6, old, ready=False)
    h.tick()                                                                    # drive sees #6 at `old`
    world.gh.put(6, new := world.change({"x.txt": "x = 2\n"}, base=old))
    world.gh.put(7, seven := world.change({"k.txt": "kernels\n"}, base=old))
    h.q.eject(6, "breaks circuits", "ci")
    world.gh.ready.discard((6, new))

    def told() -> list[str]:
        return [p for p in h.posts if "is out of `next`" in p]

    h.tick()
    assert told() == [
        f"@ci @lane-7 @lane-6 queue: #7 at {seven[:8]} is out of `next`: it contains #6's commits (at an older head), and #6 at "
        f"{new[:8]} isn't admitted (ejected at {new[:8]}: breaks circuits (a new push or ready mark re-admits it)). Fix: mark #6 "
        f"ready at {new[:8]}, drop #6's commits from #7, or ask @top to close #6"], told()
    assert h.q.status()[7].state == "waiting"
    h.tick()
    assert len(told()) == 1


def _detached(h: Review, tmp_path) -> list[str]:
    """``h``'s post-landing runs launched as train.sh is: detached, the launch returning at once with no run ids
    (`drive.detach`'s record), each launch's commit in the list returned.  `h.printed[launch["dir"]]` is what its train.sh has
    printed (`drive.launched`): unset while it runs, its run ids, or [] once it exited without them."""
    starts, h.printed = [], {}

    def start(repo, commit, machine, *a, **kw):
        starts.append(commit)
        return {"dir": str(tmp_path / f"launch-{len(starts)}"), "pid": 0}

    h.d.launch_postland = start
    h.d.launched = lambda launch: h.printed.get(launch["dir"], (None, "train: abc on m1; log x"))
    return starts


def test_the_after_landing_run_launches_detached_and_one_that_registered_no_run_is_told_and_launched_again(
        world, monkeypatch, tmp_path):
    h = Review(world, monkeypatch, baseline=False)
    starts = _detached(h, tmp_path)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    main = _main(world)
    assert h.tick().startswith("nothing lands before an after-landing run")
    run = h.postland()["run"]
    assert starts == [main] and run["ids"] == [] and run["baseline"] is True
    out = h.tick()
    assert out.startswith("nothing lands before the post-landing run (launching) of `main`") and starts == [main], out
    h.printed[run["launch"]["dir"]] = ([], "train: research run exited 1 before launching a run")
    h.tick()
    (post,) = [p for p in h.posts if "registered no run" in p]
    assert f"train.sh {main[:12]} m1 --deferred" in post and "exited 1 before launching a run" in post, post
    assert starts == [main, main]
    run = h.postland()["run"]
    h.printed[run["launch"]["dir"]] = (["r-post"], "RUN r-post")
    h.run_state["r-post"] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, "r-post", {s: "passed" for s in AFTER_LANDING})]
    assert h.tick().startswith("landed #1"), world.logs
    assert h.postland()["verified"] == main


def test_detach_returns_before_train_sh_prints_its_run_ids_and_launched_reads_them_or_an_exit_without_them(tmp_path, monkeypatch):
    """`detach`'s job, with a stand-in for train.sh that prints its `RUN` line only once the test lets it."""
    from research.drive import LAUNCHES, detach, launched

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    gate, rid = tmp_path / "gate", "r20261026-093000-abcd"
    os.mkfifo(gate)
    job = detach(tmp_path, ["sh", "-c", f'echo "train: abc on m1"; cat "$1" > /dev/null; echo "RUN {rid}"', "sh", str(gate)],
                 dict(os.environ))
    assert job["dir"].startswith(str(tmp_path / LAUNCHES)) and launched(job)[0] is None
    with open(gate, "w"):
        pass
    with contextlib.suppress(ChildProcessError):                         # its end, not a clock
        os.waitpid(job["pid"], 0)
    assert launched(job) == ([rid], f"train: abc on m1\nRUN {rid}")
    job = detach(tmp_path, ["sh", "-c", "echo 'train: no run launched' >&2; exit 1"], dict(os.environ))
    with contextlib.suppress(ChildProcessError):
        os.waitpid(job["pid"], 0)
    assert launched(job) == ([], "train: no run launched") and (Path(job["dir"]) / "rc").read_text() == "1\n"
    gone = subprocess.Popen(["true"])
    gone.wait()
    (Path(job["dir"]) / "rc").unlink()
    assert launched({**job, "pid": gone.pid})[0] == []


def test_a_detached_train_launch_runs_its_own_copy_of_train_sh(tmp_path, monkeypatch):
    """Bash reads a script as it runs, and a later tick resets and checks out the queue's clone, so
    a train.sh that `main` changed meanwhile would change under a running launch.  It runs a copy in its launch directory, from
    the clone, and a rewrite of the clone's train.sh doesn't reach it."""
    from research.drive import LAUNCHES, launched, train_launch

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    repo, gate, rid = tmp_path / "clone", tmp_path / "gate", "r20261026-093000-abcd"
    script = repo / "tools" / "verity" / "check" / "train.sh"
    script.parent.mkdir(parents=True)
    script.write_text(f'echo "train: $0 in $PWD"\ncat {gate} > /dev/null\necho "RUN {rid}"\n')
    os.mkfifo(gate)
    job = train_launch(repo, "c0ffee", "m1", False, deferred="lean-audit", detached=True)
    copy = Path(job["dir"]) / "train.sh"
    assert job["dir"].startswith(str(tmp_path / LAUNCHES)) and copy.read_text() == script.read_text()
    with open(script, "w") as f:                                           # in place, as a checkout of a changed train.sh
        f.write('echo "RUN r20261026-093000-ffff"\n' * 8)
    with open(gate, "w"):
        pass
    with contextlib.suppress(ChildProcessError):                         # its end, not a clock
        os.waitpid(job["pid"], 0)
    ids, tail = launched(job)
    assert ids == [rid] and f"train: {copy} in {repo}" in tail, tail


def test_an_eject_off_drive_moves_no_branch_and_drives_next_tick_rebuilds_next_without_it(world, monkeypatch, capsys):
    """`research queue eject N` run off drive records the ejection and withdraws the mark, and moves no branch: only drive's App
    moves `next`, and its next tick rebuilds it without the PR."""
    import research.drive as D
    from research.cli import main as research_main

    h = Review(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(9, head := world.change({"c.txt": "9\n"}))
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 9], world.logs
    one = h.q.chain()[0].commit
    monkeypatch.setattr(Q, "GitHub", lambda repo: world.gh)
    monkeypatch.setattr(D, "merge_thread_post", lambda *a, **k: True)
    monkeypatch.delenv("RESEARCH_DRIVE")
    refs = lambda: _git(world.origin, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads/")
    before = refs()
    assert research_main(["queue", "eject", "9", "--reason", "x", "--by", "compute-accounting", "--workdir", str(h.q.workdir),
                          "--remote", str(world.origin), "--repo", "o/verity", "--store", str(world.tmp / "store")]) == 0
    assert refs() == before and "drive's next tick rebuilds `next` without it" in capsys.readouterr().out
    state = h.q.load_state()
    assert state["ejected"]["9"]["head"] == head and state["ejected"]["9"]["by"] == "compute-accounting"
    (mark,) = world.store.labels(label_target(9, head), key=vocab.READY)
    assert mark.value is False and mark.by == "compute-accounting"
    with pytest.raises(QueueError, match="moved only by drive's App"):
        h.q.sync()                                                          # nor any other caller
    assert refs() == before
    monkeypatch.setenv("RESEARCH_DRIVE", "1")
    h.tick()
    assert [m.commit for m in h.q.chain()] == [one], world.logs


def test_research_queue_drive_refuses_off_drive_and_pushes_nothing(world, monkeypatch, capsys):
    """A tick lands onto `main` and rebuilds `next`, which only drive's App may move."""
    import research.merge as M
    from research.cli import main as research_main

    calls = []

    def gh(repo, *args):
        calls.append(args)
        assert args[:2] == ("pr", "list"), args
        return "[]"

    monkeypatch.setattr(M, "_gh_cli", gh)                                   # the branch sweep's, which runs before the sync
    monkeypatch.setattr(Q, "GitHub", lambda repo: world.gh)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    monkeypatch.delenv("RESEARCH_DRIVE")
    refs = lambda: _git(world.origin, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads/")
    before = refs()
    assert research_main(["queue", "drive", "--on", "m1", "--workdir", str(world.tmp / "q" / "verity"), "--remote",
                          str(world.origin), "--repo", "o/verity", "--store", str(world.tmp / "store"), "--rules",
                          str(world.dev / "ci" / "queue.toml")]) == 1
    assert "/data/infra/drive.sh once" in capsys.readouterr().err and refs() == before and calls == []


def test_drives_own_pushes_of_main_refuse_off_drive_and_in_shadow_too(world, monkeypatch):
    """`_land` and `_revert` push `main` whoever calls them; off drive they refuse before anything, `--shadow` too,
    which never reaches them."""
    h = Review(world, monkeypatch, baseline=False)                         # nothing lands before `main`'s own run
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert not h.tick().startswith("landed")
    refs = lambda: _git(world.origin, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads/")
    before = refs()
    monkeypatch.delenv("RESEARCH_DRIVE")
    with pytest.raises(QueueError, match="moved only by drive's App"):
        h.d.land_reviewed(h.q.load_state()["drive"], dry_run=False)
    _move_main(world, {"z.txt": "z\n"})
    h.q.fetch()
    before, moved = refs(), _main(world)
    with pytest.raises(QueueError, match="moved only by drive's App"):
        h.d._revert([moved])
    h.q.shadow = True
    with pytest.raises(QueueError, match="moved only by drive's App"):
        h.d._revert([moved])
    assert refs() == before
    h.q.shadow = False
    monkeypatch.setenv("RESEARCH_DRIVE", "1")
    assert h.d._revert([moved]) == "pushed" and _git(world.origin, "rev-parse", "main^1") == moved

