"""Pearl-C's pinned host arena (`pearl_c.host._Arena`), CPU only, over a stand-in for pinned memory: a take within half a
ring lands in the first ring with room, and its give-back frees that ring's region; a take past half a ring gets memory of
its own and makes no ring; a take no ring has room for waits for a give, and every give wakes the takes that wait.  The
copies that land in the arena on a card are the e2e's.
"""
from __future__ import annotations

import pytest

from verity.protocols.accounting.pouw.prover.serving.pearl_c.host import _Arena


@pytest.fixture
def unpinned(monkeypatch):
    """`torch.empty` blind to `pin_memory`: a host with no accelerator has no pinned memory to give."""
    torch = pytest.importorskip("torch")
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, pin_memory=False, **kw: empty(*a, **kw))


class Waits:
    """An arena's condition where no other thread runs: a take that waits gets the next give-back on hand, as a thread that
    gave meanwhile would have left it."""

    def __init__(self, gives):
        self.gives, self.waited, self.notified = list(gives), 0, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def wait(self):
        self.waited += 1
        self.gives.pop(0)()

    def notify_all(self):
        self.notified += 1


def _at(ring, host):
    """Where in the ring's bytes the host tensor lies: (its first byte, the byte past its last)."""
    first = host.data_ptr() - ring.buf.data_ptr()
    return [first, first + host.numel()]


def test_a_take_lands_in_the_first_ring_with_room_and_its_give_back_frees_that_rings_region(unpinned):
    """`_Arena.take`: sizes are rounded up to 512 bytes; the rings are made at the first take; a take goes to the first
    ring that has room.  Each give-back frees its own region in its own ring, whatever was taken after it: a region
    given back before an earlier one of its ring stays live until that one is given back too."""
    arena = _Arena(4096, 2)
    assert arena.rings == []
    (give_a, a), (give_b, b), (_, c), (give_d, d) = (arena.take(n) for n in (2000, 1500, 100, 2048))
    first, second = arena.rings
    assert [_at(first, t) for t in (a, b, c)] == [[0, 2048], [2048, 3584], [3584, 4096]] and _at(second, d) == [0, 2048]
    give_d()
    assert not second.live and [r[:2] for r in first.live] == [[0, 2048], [2048, 3584], [3584, 4096]]
    give_b()
    assert [r[:2] for r in first.live] == [[0, 2048], [2048, 3584], [3584, 4096]]         # b is free only once a is
    give_a()
    assert [r[:2] for r in first.live] == [[3584, 4096]] and not second.live


def test_a_take_past_half_a_ring_gets_memory_of_its_own_and_makes_no_ring(unpinned):
    """`_Arena.take`: a copy past half a ring is no ring's: its bytes are as many as asked, and its give-back gives
    nothing back."""
    arena = _Arena(4096, 2)
    give, host = arena.take(2049)
    assert host.numel() == 2049 and give() is None and arena.rings == []


def test_a_take_no_ring_has_room_for_waits_for_a_give_and_every_give_wakes_the_takes_that_wait(unpinned, monkeypatch):
    """`_Arena.take` and `_give`: with its one ring full a take waits on the arena's condition, and lands once a give has
    made room (here the region past the ring's end is skipped, so it lands at the ring's start).  Each give notifies."""
    arena = _Arena(4096, 1)
    (give_a, _), (give_b, _) = arena.take(2048), arena.take(2048)
    waits = Waits([give_a])
    monkeypatch.setattr(arena, "cv", waits)
    give_c, c = arena.take(1024)
    (ring,) = arena.rings
    assert (waits.waited, waits.notified, waits.gives) == (1, 1, []) and _at(ring, c) == [0, 1024]
    give_b()
    assert [r[:2] for r in ring.live] == [[0, 1024]]
    give_c()
    assert not ring.live and waits.notified == 3
