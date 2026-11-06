"""No Lean check is stacked above a failed one: once a Lean check of `next` fails, `_stack_lean` launches none whose prefix
holds every Lean change of the failed prefix still on `next` (`Driver._live_failures`), though a sync that rebuilds `next`
above a running check's prefix takes the failed check's commit off `next`; the walk checks each Lean change's own commit,
and stacking resumes once a member of the failed prefix is held, ejected or pushed."""
from __future__ import annotations

import pytest

from research.drive import LEAN_STEP
from test_drive_review import Review
from test_queue import World
from test_queue_drive import _steps_run


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _failed_above_running(world: World, monkeypatch) -> tuple[Review, str]:
    """`next` = #1, #2 (Lean changes, reviewed), a Lean check of #1's commit running and the one stacked at #2's failed; then a
    `--first` #5 goes above the running check's prefix, below #2, rebuilding #2's commit, and a Lean change #6 comes on top:
    (the drive, the failed check's run id)."""
    h = Review(world, monkeypatch)
    for n in (1, 2):
        world.gh.put(n, world.change({f"pkg/L{n}.lean": f"-- {n}\n"}))
        h.review(n)
    assert not h.tick().startswith("landed"), world.logs
    chain = h.q.chain()
    assert h.launched == [(chain[0].commit, "m1", False), (chain[1].commit, "m1", False)], world.logs
    low, high = h.lean_ids()
    h.run_state[high] = ("done", 1, None)
    h.evidence = [_steps_run(world.tmp, high, {LEAN_STEP: "failed"})]
    h.tick()
    assert h.d.q.load_state()["drive"]["lean"][chain[1].commit]["state"] == "failed"
    world.gh.put(5, world.change({"e.txt": "5\n"}))
    h.review(5)
    h.q.first = (5,)
    world.gh.put(6, world.change({"pkg/L6.lean": "-- 6\n"}))
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 5, 2, 6] and h.q.chain()[0].commit == chain[0].commit, world.logs
    assert chain[1].commit not in h.d.q.load_state()["drive"]["lean"], "its commit left `next`: the record is dropped"
    return h, high


def test_no_lean_check_is_stacked_above_a_failed_one_after_a_rebuild_takes_its_commit_off_next(world, monkeypatch):
    h, high = _failed_above_running(world, monkeypatch)
    for _ in range(3):
        n = len(world.logs)
        h.tick()
        assert len(h.launched) == 2 and h.cancelled == [], world.logs
        (why,) = [x for x in world.logs[n:] if "stacks no Lean check" in x]
        assert f"the Lean check {high} of #1 #2 failed" in why and "#2" in why, why


@pytest.mark.parametrize("how", ["hold", "eject", "push"])
def test_stacking_resumes_once_a_member_of_the_failed_prefix_is_held_ejected_or_pushed(world, monkeypatch, how):
    h, high = _failed_above_running(world, monkeypatch)
    h.tick()
    assert len(h.launched) == 2, world.logs
    head = world.gh.prs[2].head
    if how == "hold":
        h.review(2, "hold", by="other", ref="breaks the build")
    elif how == "eject":
        h.q.eject(2, "breaks the build", "ci")
        world.gh.ready.discard((2, head))
    else:
        world.gh.put(2, world.change({"pkg/L2.lean": "-- 2, fixed\n"}))
    h.tick()
    h.tick()
    chain = h.q.chain()
    assert [m.pr for m in chain][:3] == [1, 5, 6], world.logs
    assert any(f"the failed Lean check {high}" in x and "no longer holds" in x for x in world.logs), world.logs
    assert h.launched[2:] == [(chain[2].commit, "m1", False)] and h.cancelled == [], world.logs
    assert h.d.q.load_state()["drive"]["lean"][chain[2].commit]["stacked"] is True


def test_the_walk_checks_the_lean_changes_own_commits_and_their_landing_drops_the_failure(world, monkeypatch):
    """The lower check's pass lands #1 and #5; #2's own commit is checked, not a cut through #6 above it, and stacking waits
    for it; its pass lands #2, and the failure, none of its Lean changes left on `next`, is dropped."""
    h, high = _failed_above_running(world, monkeypatch)
    low = h.lean_ids()[0]
    h.run_state[low] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, low, {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #5"), world.logs
    chain = h.q.chain()
    assert [m.pr for m in chain] == [2, 6] and h.launched[2:] == [(chain[0].commit, "m1", False)], world.logs
    two = h.lean_ids()[2]
    h.run_state[two] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, two, {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #2"), world.logs
    assert any(f"the failed Lean check {high}" in x and "none of its Lean changes is left" in x for x in world.logs)
    assert not h.d.q.load_state()["drive"]["lean_failed"], world.logs


def test_a_failure_whose_lean_changes_a_passed_check_holds_is_dropped(world, monkeypatch):
    """A failed check whose Lean changes on `next` a Lean check that passed holds at their heads (it failed by a fault no check
    saw) holds nothing."""
    h, high = _failed_above_running(world, monkeypatch)
    drive, chain = h.d.q.load_state()["drive"], h.q.chain()
    assert [f for f, _ in h.d._live_failures(drive, chain, set())] == list(drive["lean_failed"].values())
    drive["lean"][chain[2].commit] = {"ids": ["r-pass"], "run": "r-pass", "state": "passed",
                                      "members": [[m.pr, m.head] for m in chain[:3]]}
    assert h.d._live_failures(drive, chain, set()) == [] and drive["lean_failed"] == {}
    assert f"the failed Lean check {high} of #1 #2 no longer holds" in world.logs[-1]
    assert "the Lean check r-pass that holds its Lean changes on `next` passed" in world.logs[-1], world.logs[-1]
