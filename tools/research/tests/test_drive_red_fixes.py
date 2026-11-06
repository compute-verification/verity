"""The low-risk lane's fixes of reds on `checked` (`Driver._fix_wait`): the after-landing run waits up to `FIX_WAIT_S` after a
fix's mark for the lane to land it, as a run without it would go red again."""
from __future__ import annotations

import time

import pytest

from research.drive import FIX_WAIT_S
from test_drive_checked import _checked
from test_drive_lowrisk import _lane
from test_drive_review import Review
from test_queue import World, _git


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


#: a fix of red `suite pkg`, by its body (`Driver._fixes`)
FIX_BODY = "Lane `lane-{n}`. Fixes suite `pkg`."


def _red(h: Review, key: str = "suite pkg") -> None:
    """``key`` red on `checked` with no culprit named, so nothing reverts it (`Driver._revert_reds`)."""
    state = h.q.load_state()
    pl = state["drive"]["postland"]
    pl["red"] = {key: {"tip": pl["verified"], "ids": ["r-red"], "first": "x", "culprit": None}}
    h.q.save_state(state)


def test_the_after_landing_run_waits_for_a_fresh_low_risk_fix_of_a_red_and_launches_fix_wait_s_after_its_mark(world, monkeypatch):
    """#4, a fix of red `suite pkg` marked just now, is on `next` above #2 and #3, low-risk too, and the lane lands one member a
    tick: the after-landing run of #2's landing waits for #4, and launches `FIX_WAIT_S` after #4's mark without it, at the top
    of that tick, before its landing of #3."""
    h = _lane(world, monkeypatch, per_tick=1)
    h.d.lowrisk_lane = False
    world.gh.put(2, world.change({"x.txt": "2\n"}))
    world.gh.put(3, world.change({"y.txt": "3\n"}))
    world.gh.put(4, four := world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}), body=FIX_BODY.format(n=4))
    h.tick()
    _red(h)
    marked = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(h.clock))
    h.q.marked_at = lambda n, head: marked if (n, head) == (4, four) else None
    h.d.lowrisk_lane = True
    assert h.tick().endswith("landed #2 on the low-risk lane"), world.logs
    tip = _checked(world)
    assert h.posted == [], world.logs
    assert f"research queue drive: the after-landing run of `checked` {tip[:12]} waits: #4 at {four[:8]}, a low-risk fix of " \
           f"suite pkg marked 0 min ago, isn't landed yet: it launches once that lands, or {FIX_WAIT_S // 60} min after its " \
           "mark" in world.logs, world.logs
    h.clock += FIX_WAIT_S
    assert h.tick().endswith("landed #3 on the low-risk lane"), world.logs
    assert [c for c, *_ in h.posted] == [tip] and _git(world.origin, "rev-parse", "checked^1") == tip, world.logs
    assert "pkg/tests/test_x.py" not in _git(world.origin, "ls-tree", "-r", "--name-only", tip)
