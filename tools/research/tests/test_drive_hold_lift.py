"""One hold stops only itself: each sync moves a member of `next` held by a verdict at its head, with every PR that contains
it, to the top of `next` (`Driver._park`, `Queue.park`), so the members that were above it land with those below it.  It stays
at the top until a land verdict at that head, which lands it by the normal order.  A Lean check below it keeps running."""
from __future__ import annotations

import json

import pytest

from research.drive import LEAN_STEP
from test_drive_review import Review, _main, _subject
from test_queue import World
from test_queue_drive import _steps_run, _tip


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _landed(world: World, n: int) -> list[int]:
    """The PRs of `main`'s ``n`` newest landings, newest first."""
    return [int(_subject(world, f"main{'^1' * i}").split("#")[1].split()[0]) for i in range(n)]


@pytest.mark.parametrize("reviewed_first", [True, False])
def test_a_held_member_goes_to_the_top_and_the_members_above_and_below_it_land(world, monkeypatch, reviewed_first):
    """With the lift (`reviewed_first`) or without it: the held member no longer holds the reviewed members above it."""
    h = Review(world, monkeypatch, reviewed_first=reviewed_first)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt"), (4, "d.txt"), (5, "e.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    assert h.tick().startswith("nothing reviewed lands: #1"), world.logs
    assert [m.pr for m in h.q.chain()] == [1, 2, 3, 4, 5]
    for n in (1, 2, 4, 5):
        h.review(n)
    h.review(3, "hold", by="other", ref="reads a stale fixture")
    assert h.tick().startswith("landed #5"), world.logs
    assert _landed(world, 4) == [5, 4, 2, 1]
    assert [m.pr for m in h.q.chain()] == [3]
    assert any("#3 (held, or containing a held PR) go to the top" in x for x in world.logs), world.logs
    out = h.tick()
    assert out.startswith("nothing reviewed lands: #3") and "held by other" in out, out


def test_a_member_that_contains_the_held_one_goes_to_the_top_with_it_and_both_land_on_its_land_verdict(world, monkeypatch):
    h = Review(world, monkeypatch, reviewed_first=False)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    world.gh.put(4, world.change({"d.txt": "4\n"}, base=world.gh.prs[3].head))     # stacked on #3
    world.gh.put(5, world.change({"e.txt": "5\n"}))
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2, 3, 4, 5]
    for n in (1, 2, 4, 5):
        h.review(n)
    h.review(3, "hold", by="other", ref="reads a stale fixture")
    assert h.tick().startswith("landed #5"), world.logs
    assert _landed(world, 3) == [5, 2, 1]
    assert [m.pr for m in h.q.chain()] == [3, 4]
    h.review(3, by="other")                                     # the reviewer's newest verdict: land
    assert h.tick().startswith("landed #4"), world.logs
    assert _landed(world, 2) == [4, 3] and _main(world) == _tip(world)


def test_a_held_member_stays_above_what_comes_new_and_lands_on_a_land_verdict(world, monkeypatch):
    """A PR admitted after the move goes below the held member, so the hold doesn't stop it; a land verdict at the held
    member's head lands it by the normal order at the next tick."""
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    h.review(1, "hold", by="other", ref="reads a stale fixture")
    h.review(2)
    h.review(3)
    assert h.tick().startswith("landed #3"), world.logs
    assert [m.pr for m in h.q.chain()] == [1]
    world.gh.put(6, world.change({"f.txt": "6\n"}))
    h.review(6)
    assert h.tick().startswith("landed #6"), world.logs
    assert [m.pr for m in h.q.chain()] == [1]
    world.gh.put(7, world.change({"g.txt": "7\n"}))
    h.tick()
    assert [m.pr for m in h.q.chain()] == [7, 1]
    h.review(1, by="other")                                 # the normal order: the lift takes it below the unreviewed #7
    assert h.tick().startswith("landed #1"), world.logs
    assert [m.pr for m in h.q.chain()] == [7]


def _lean_below(world: World, monkeypatch) -> tuple[Review, str]:
    """`next` = #1 (a lock change), #2, #3, #4, #5, with a Lean check of #2's commit running; (the drive, that commit)."""
    h = Review(world, monkeypatch, nonlean_first=False, reviewed_first=False)
    world.gh.put(1, world.change({"pkg/lean-audit.json": json.dumps({"guarantees": {"T": "2"}, "reads": {},
                                                                     "dependencies": {"M": "1"}})}))
    world.gh.put(2, world.change({"a.txt": "2\n"}))
    h.review(1)
    h.review(2)
    assert not h.tick().startswith("landed"), world.logs
    check = _tip(world)
    assert h.launched == [(check, "m1", False)]
    for n, f in ((3, "b.txt"), (4, "c.txt"), (5, "d.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2, 3, 4, 5] and h.q.chain()[1].commit == check
    return h, check


def test_a_running_lean_check_below_the_held_member_runs_on_and_lands_the_members_that_were_above_it(world, monkeypatch):
    h, check = _lean_below(world, monkeypatch)
    h.review(3, "hold", by="other", ref="reads a stale fixture")
    h.review(4)
    h.review(5)
    assert not h.tick().startswith("landed"), world.logs
    chain = h.q.chain()
    assert [m.pr for m in chain] == [1, 2, 4, 5, 3] and chain[1].commit == check
    assert h.cancelled == [] and h.d.q.load_state()["drive"]["lean"][check]["state"] is None
    (rid,) = h.lean_ids()
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {"lean-build": "passed", LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #5"), world.logs
    assert _landed(world, 4) == [5, 4, 2, 1] and [m.pr for m in h.q.chain()] == [3]
    assert h.cancelled == [] and len(h.launched) == 1


def test_a_held_member_a_running_lean_check_holds_goes_to_the_top_and_only_that_check_is_cancelled(world, monkeypatch):
    """The held member moves though a running check covers it: that check is of a prefix that can't land, and the next tick
    cancels it as a check of a commit no longer on `next`."""
    h, check = _lean_below(world, monkeypatch)
    first = h.lean_ids()
    h.review(2, "hold", by="other", ref="reads a stale fixture")
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 3, 4, 5, 2]
    h.tick()
    assert h.cancelled == first and check not in h.d.q.load_state()["drive"]["lean"], world.logs


def test_a_lean_check_stacked_above_a_running_one_stops_below_a_held_member(world, monkeypatch):
    """The held member sits at `next`'s top, and each PR a sync adds goes below it: a check stacked through it would leave
    `next` at the next admission, and checks a prefix that can't land."""
    h = Review(world, monkeypatch, nonlean_first=False)
    world.gh.put(1, world.change({"pkg/lean-audit.json": "{}"}))
    h.review(1)
    h.tick()
    one = _tip(world)
    assert h.launched == [(one, "m1", False)]
    world.gh.put(2, world.change({"pkg/T.lean": "theorem T : True := trivial\n"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    h.review(3, "hold", by="other", ref="reads a stale fixture")
    h.tick()
    h.tick()                                                # #2 and #3 settled
    chain = h.q.chain()
    assert [m.pr for m in chain] == [1, 2, 3]
    assert h.launched == [(one, "m1", False), (chain[1].commit, "m1", False)] and h.cancelled == [], world.logs


@pytest.mark.parametrize("how", ["label", "eject"])
def test_a_pr_with_a_hold_label_or_ejected_leaves_next_and_the_members_above_it_land(world, monkeypatch, how):
    h = Review(world, monkeypatch)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    for n in (1, 2, 3):
        h.review(n)
    head = world.gh.prs[2].head
    if how == "label":
        world.gh.put(2, head, labels=("hold",))
    else:
        h.q.eject(2, "x", "ci")
        world.gh.ready.discard((2, head))                   # the store's `ready = false`, as drive's `ready` reads it
    assert h.tick().startswith("landed #3"), world.logs
    assert _landed(world, 2) == [3, 1] and h.q.chain() == []
