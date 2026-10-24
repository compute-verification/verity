"""Where drive's checks stand in the slot lines (`slot.py --rank`) and what it lands without: trains first, the lowest train
first of them, and the post-landing run after every train (top, 24 Oct 04:44Z and 04:53Z)."""
from __future__ import annotations

import pytest

from research.drive import Driver, launch_vars
from test_drive_postland import Postland
from test_queue import World


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_the_post_landing_run_waits_unranked_behind_every_train(world, monkeypatch, tmp_path):
    """Rank 0 put the post-landing run ahead of every train in each slot line; unranked, it takes a slot only where no train
    waits, still ahead of the lanes' checks."""
    h = Postland(world, monkeypatch, tmp_path)
    ranks, post = [], h.d.launch_postland
    h.d.launch_postland = lambda *a: ranks.append(launch_vars().get("CHECK_SLOT_RANK")) or post(*a)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.land_deferred()
    assert len(h.posted) == 1 and ranks == [None]


def test_the_lean_checks_steps_are_never_deferred_but_its_suites_may_be(world, monkeypatch, tmp_path):
    """Daniel, 11 Oct: a part whose cold run takes over ~10 minutes runs after its train lands, and the Lean check stays before.
    lean-suites, the pytest suites on the Lean build, took 28-65 min before landing (24 Oct), after lean-audit had passed."""
    h = Postland(world, monkeypatch, tmp_path)
    d = Driver(h.q, on="m1", launch=h.launch, defer=frozenset({"lean-audit", "lean-build", "lean-suites", "pytest"}))
    assert d.defer == {"lean-suites", "pytest"}
