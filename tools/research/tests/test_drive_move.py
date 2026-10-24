"""Drive's launches that wait in slot lines: `max_running` counts the checks that started and only `UNSTARTED_COUNTED` that wait,
and a waiting train moves to a machine of `on` that could start it now (24 Oct 05:28Z: ci cancelled 23 waiting twins by hand so
drive would relaunch their trains onto vy-nebius-1 and vy-nebius-3, added to `--on` while the trains waited 1-4 h elsewhere)."""
from __future__ import annotations

import pytest

from research import drive
from research.drive import launch_vars
from test_queue import World, _git
from test_queue_drive import Harness


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _launched(world, monkeypatch, n: int, max_running: int) -> Harness:
    """A harness whose first tick launched the trains of ``n`` one-PR trains, each launch one twin on m1."""
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 1, max_running
    for k in range(1, n + 1):
        world.gh.put(k, world.change({f"f{k}.txt": f"{k}\n"}))
    h.tick()
    return h


def _ids(h: Harness, commit: str) -> list[str]:
    return h.q.load_state()["drive"]["launched"][commit]["ids"]


def _depth(h: Harness, commit: str) -> int:
    return int(_git(h.d.repo, "rev-list", "--count", "--first-parent", commit))


def test_a_machine_added_to_on_takes_the_lowest_waiting_train_in_one_tick(world, monkeypatch):
    h = _launched(world, monkeypatch, 3, max_running=4)
    chain = h.q.chain()
    assert [c for c, _, _ in h.launched] == [m.commit for m in chain]
    for rid in h.run_state:
        h.run_state[rid] = ("waiting", None, None, None, "m1")       # every line on m1 full
    h.d.on, free = "m1,m2", {"m1": 0, "m2": 1}                       # m2 joins `on` with one train slot free
    h.d.slots = lambda repo, on: dict(free) if on == "m1,m2" else pytest.fail(on)
    ranks, plain = [], h.d.launch
    h.d.launch = lambda *a: ranks.append(launch_vars().get("CHECK_SLOT_RANK")) or plain(*a)
    lowest = chain[0].commit
    old = _ids(h, lowest)
    out = h.tick()
    assert h.cancelled == old, "only the lowest train's waiting twin is cancelled: m2 has one slot"
    assert h.launched[3:] == [(lowest, "m1,m2", False)] and ranks == [str(_depth(h, lowest))], "relaunched with its rank"
    rec = h.q.load_state()["drive"]["launched"][lowest]
    assert rec["ids"] != old and rec["moved"] == 1
    assert f"{lowest[:12]}" in out and "running" in out


def test_a_check_that_started_is_never_cancelled_even_if_its_twin_took_the_slot_just_now(world, monkeypatch):
    h = _launched(world, monkeypatch, 3, max_running=4)
    chain = h.q.chain()
    started, racing, waiting = (_ids(h, m.commit)[0] for m in chain)
    h.run_state[started] = ("running", None, None, None, "m1")       # holds a slot on m1
    h.run_state[racing] = ("waiting", None, None, None, "m1")
    h.run_state[waiting] = ("waiting", None, None, None, "m1")
    reads: dict[str, int] = {}

    def states(repo, ids):
        out = {}
        for rid in ids:
            reads[rid] = reads.get(rid, 0) + 1
            out[rid] = ("running", None, None, None, "m1") if rid == racing and reads[rid] > 1 else h.run_state[rid]
        return out

    h.d.states, h.d.on = states, "m1,m2"
    h.d.slots = lambda repo, on: {"m1": 0, "m2": 5}
    h.tick()
    assert started not in h.cancelled and racing not in h.cancelled, "a twin holding a slot, or that took one since, runs on"
    assert h.cancelled == [waiting] and [c for c, _, _ in h.launched[3:]] == [chain[2].commit]


def test_max_running_counts_no_unstarted_check_past_the_bound(world, monkeypatch):
    """Three trains wait for m2's slots, which just freed: their twins there take three, and the fourth slot gets a new train in
    the same tick, which counting every waiting launch (3 of `max_running` 3) held back."""
    h = _launched(world, monkeypatch, 5, max_running=3)
    chain = h.q.chain()
    first = [c for c, _, _ in h.launched]
    assert first == [chain[0].commit, chain[1].commit, chain[4].commit], "the lowest two and the top"
    for rid in h.run_state:
        h.run_state[rid] = ("waiting", None, None, None, "m2")
    h.d.slots = lambda repo, on: {"m1": 0, "m2": 4}
    assert drive.UNSTARTED_COUNTED == 2
    out = h.tick()
    assert h.cancelled == [], "each waits where a slot is free for it"
    assert [c for c, _, _ in h.launched[3:]] == [chain[2].commit], out
    assert f"{chain[3].commit[:12]} waits: 3 checks of `next` running" in out


def test_past_the_bound_a_train_launches_only_onto_a_free_slot(world, monkeypatch):
    h = _launched(world, monkeypatch, 3, max_running=8)
    assert len(h.launched) == 3
    for rid in h.run_state:
        h.run_state[rid] = ("waiting", None, None, None, "m1")
    h.d.slots = lambda repo, on: {"m1": 0}
    for k in (4, 5):                                                  # two more trains, and no slot free anywhere
        world.gh.put(k, world.change({f"f{k}.txt": f"{k}\n"}))
    out = h.tick()
    chain = h.q.chain()
    assert h.cancelled == [] and f"{chain[3].commit[:12]} waits: 3 checks of `next` wait for a check slot, and no machine " \
                                  "of m1 has one free" in out, out
    assert [c for c, _, _ in h.launched[3:]] == [chain[4].commit], "the top still takes its place in the lines, right after the lowest"
    h.d.slots = lambda repo, on: {"m1": 1}                           # one frees where they all wait: the lowest twin takes it
    out = h.tick()
    assert h.cancelled == [] and len(h.launched) == 4, out
