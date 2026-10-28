"""The review landing onto `checked` (drive.toml's `land_into`: `main` holds only tested commits).  A
landing fast-forwards `checked`, never `main`; `main` is fast-forwarded only to a commit of `checked` whose after-landing run
passed every step; a red run moves nothing on `main`, and its culprit is reverted on `checked`."""
from __future__ import annotations

import json

import pytest

from research.drive import FIX_WINDOW_S
from research.queue import DriveFlags, QueueError
from test_drive_review import Review, _main, _move_main, _owning, _subject
from test_queue import World, _git
from test_queue_drive import _failed_run, _steps_run


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _checked(world: World) -> str:
    return _git(world.origin, "rev-parse", "refs/heads/checked")


def _pass(h: Review, suites: tuple[str, ...] = ()) -> None:
    """The newest after-landing run passes every step it runs (and ``suites``)."""
    rid = f"r-post-{len(h.posted) - 1}"
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(h.world.tmp, rid, {s: "passed" for s in h.posted[-1][1].split(",")})]
    (h.evidence[0] / "suites-lean.json").write_text(json.dumps([{"name": s, "suite": s, "status": "passed"} for s in suites]))


def test_a_review_landing_moves_checked_and_main_follows_only_once_its_after_landing_run_passed_every_step(world, monkeypatch):
    h = Review(world, monkeypatch, land_into="checked")
    main = _main(world)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.review(1)
    h.review(2)
    assert h.tick().startswith("landed #2"), world.logs
    tip = _checked(world)
    assert _subject(world, "checked").startswith("Queue: #2") and _subject(world, "checked^1").startswith("Queue: #1")
    assert _main(world) == main and h.after == []               # what follows a landing on `main` waits for `main`
    assert _git(world.origin, "rev-parse", "next^1") == tip     # `next` is built on `checked`
    assert [c for c, *_ in h.posted] == [tip]                   # the after-landing run checks `checked`'s tip
    h.tick()                                                    # #1 and #2 are still open on GitHub: not admitted again
    assert _main(world) == main and world.on_next(h.q) == [3]
    st = h.q.status()
    assert st[1].state == "queued" and st[1].why.startswith("landed on `checked`: it reaches `main` when"), st[1]
    assert st[3].state == "queued" and st[3].position == 1
    _pass(h)
    h.tick()
    assert h.postland()["verified"] == tip and _main(world) == tip
    assert h.after == [(main, {_git(world.origin, "rev-parse", "checked^1"): "r-post-0", tip: "r-post-0"})]
    assert any(f"`main` fast-forwarded to {tip[:12]}" in x for x in world.logs), world.logs


def test_a_red_after_landing_run_moves_nothing_on_main_and_its_culprit_is_reverted_on_checked(world, monkeypatch, tmp_path):
    _owning(world, {"lane-2": [], "pkg-owner": ["pkg/"]})
    h = Review(world, monkeypatch, land_into="checked")
    main = _main(world)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}))
    for n in (1, 2):
        h.review(n)
    assert h.tick().startswith("landed #2")
    tip = _checked(world)
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    (ring,) = [p for p in h.posts if "is red on" in p]
    assert "suite pkg is red on `checked`" in ring and "Culprit: #2" in ring, ring
    assert _main(world) == main
    h.clock += FIX_WINDOW_S + 1
    h.tick()
    reverted = _checked(world)
    assert _git(world.origin, "rev-parse", f"{reverted}^") == tip and _main(world) == main
    assert "pkg/tests/test_x.py" not in _git(world.origin, "ls-tree", "-r", "--name-only", reverted)
    assert any(p.startswith("@ci @lane-2 queue: #2 reverted on `checked`") for p in h.posts), h.posts
    assert h.posted[-1][0] == reverted
    _pass(h, ("pkg",))
    h.tick()
    assert _main(world) == reverted and not h.postland()["red"]
    assert world.on_next(h.q) == []                             # #2, reverted, is still open: its head is on `checked`


def test_main_follows_no_baseline_that_passed_fewer_steps(world, monkeypatch):
    """ci's baseline of only some steps lets landing go on, as a red baseline run does, but isn't a full check: `main` stays."""
    from research import queue as Q

    h = Review(world, monkeypatch, land_into="checked")
    main = _main(world)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    tip = _checked(world)
    state = h.q.load_state()
    state["drive"]["postland"]["run"] = None                    # its run lost; ci records `checked` by hand instead
    h.q.save_state(state)
    assert Q.baseline(h.q, tip, "ci", "lean-audit") == 0
    h.tick()
    assert _main(world) == main
    assert Q.baseline(h.q, tip, "ci") == 0
    h.tick()
    assert _main(world) == tip


def test_a_main_pushed_outside_drive_is_merged_into_checked_and_main_follows_once_that_is_checked(world, monkeypatch):
    """The lander's pure move lands on `main` on its own check while `checked` holds landings `main` lacks: drive merges it
    into `checked`, whose next after-landing run checks the two together before `main` moves on."""
    h = Review(world, monkeypatch, land_into="checked")
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    landed = _checked(world)
    _move_main(world, {"c.txt": "moved\n"})
    moved = _main(world)
    h.tick()
    merged = _checked(world)
    assert _git(world.origin, "rev-parse", f"{merged}^1", f"{merged}^2").split() == [landed, moved]
    _pass(h)                                                    # the run of `landed`: `main` isn't below it, so it stays
    h.tick()
    assert _main(world) == moved and h.posted[-1][0] == merged
    _pass(h)
    h.tick()
    assert _main(world) == merged


def test_a_checked_below_a_moved_main_fast_forwards_to_it(world, monkeypatch):
    h = Review(world, monkeypatch, land_into="checked")
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1")
    _pass(h)
    h.tick()
    assert _main(world) == _checked(world)
    _move_main(world, {"c.txt": "moved\n"})
    h.tick()
    assert _checked(world) == _main(world)


def test_land_into_takes_checked_or_main_and_only_drive_pushes_checked(world, monkeypatch):
    assert DriveFlags.parse('land_into = "checked"\n').land_into == "checked" and DriveFlags().land_into == "main"
    with pytest.raises(QueueError, match="land_into"):
        DriveFlags.parse('land_into = "staging"\n')
    monkeypatch.delenv("RESEARCH_DRIVE", raising=False)
    with pytest.raises(QueueError, match="`checked` is moved only by drive's App"):
        world.queue(land_into="checked")._drive_only("checked", "once", False)
