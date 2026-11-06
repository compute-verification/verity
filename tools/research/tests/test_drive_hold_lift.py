"""One hold stops only itself: each sync moves a member of `next` held by a verdict at its head, with every PR that contains
it, to the top of `next` (`Driver._park`, `Queue.park`), so the members that were above it land with those below it.  It stays
at the top until a land verdict at that head, which lands it by the normal order.  And one member waiting for its verdict
stops only itself: the lift (`Driver._lift`) moves the reviewed members below it, and one that gains its verdict goes after
those already moved."""
from __future__ import annotations

import pytest

from test_drive_review import Review, _main, _subject
from test_queue import World
from test_queue_drive import _tip


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _landed(world: World, n: int) -> list[int]:
    """The PRs of `main`'s ``n`` newest landings, newest first."""
    return [int(_subject(world, f"main{'^1' * i}").split("#")[1].split()[0]) for i in range(n)]


@pytest.mark.parametrize("verdict_first", [True, False])
def test_a_held_member_goes_to_the_top_and_the_members_above_and_below_it_land(world, monkeypatch, verdict_first):
    """With the lift (`verdict_first`) or without it: the held member no longer holds the reviewed members above it."""
    h = Review(world, monkeypatch, verdict_first=verdict_first)
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
    h = Review(world, monkeypatch)
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


def _waiting(world: World, monkeypatch) -> Review:
    """A drive whose landing waits (`main` has no baseline yet: `Driver._unbaselined`), so what the lift moves stays on `next`,
    which holds #1 to #5, none reviewed."""
    h = Review(world, monkeypatch, baseline=False)
    for n in range(1, 6):
        world.gh.put(n, world.change({f"{n}.txt": f"{n}\n"}))
    assert not h.tick().startswith("landed"), world.logs
    assert [m.pr for m in h.q.chain()] == [1, 2, 3, 4, 5]
    return h


def test_a_member_that_gains_its_verdict_goes_after_those_already_moved(world, monkeypatch):
    """The members already moved keep their commits: nothing below the one that joins them is merged again."""
    h = _waiting(world, monkeypatch)
    h.review(4)
    assert not h.tick().startswith("landed"), world.logs
    chain = h.q.chain()
    assert [m.pr for m in chain] == [4, 1, 2, 3, 5]
    four = chain[0].commit
    h.review(2)
    h.tick()
    chain = h.q.chain()
    assert [m.pr for m in chain] == [4, 2, 1, 3, 5] and chain[0].commit == four


def test_first_goes_below_the_reviewed_members_the_lift_moves(world, monkeypatch):
    h = _waiting(world, monkeypatch)
    h.review(4)
    h.review(5)
    h.q.first = (5,)
    h.tick()
    assert [m.pr for m in h.q.chain()] == [5, 4, 1, 2, 3]
