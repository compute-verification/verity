"""The low-risk lane's fixes of reds on `checked` (`Driver._land_fixes`, `_fix_wait`): while a Lean check of `next` can't move
across a low-risk landing, a member that fixes a red still lands, onto `checked` without rebuilding `next`, and the Lean check's
verdict moves with its prefix at the next sync (`_rekey_lean`); the after-landing run waits up to `FIX_WAIT_S` after a fix's
mark for it to land."""
from __future__ import annotations

import time

import pytest

from research.drive import FIX_WAIT_S, LEAN_STEP
from test_drive_checked import _checked
from test_drive_lowrisk import EXCLUDES, TAG, _landed, _lane
from test_drive_review import Review, _subject
from test_queue import World, _git
from test_queue_drive import _steps_run, _tip


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


def _fixes_only(world: World) -> list[str]:
    return [x for x in world.logs if "each a fix of a red on `checked`, without rebuilding `next`" in x]


def test_a_fix_of_a_red_lands_while_a_lean_check_cant_move_and_the_next_after_landing_run_holds_it(world, monkeypatch):
    """A hand Lean check of #2 runs, so no Lean check moves across a low-risk landing (`_keep_lean`) and the lane waits; #3,
    a fix of red `suite pkg` above every check's prefix, lands anyway, onto `checked` with `next` left as it is, and the
    after-landing run checks it.  #2, low-risk but no fix, waits.  `next` stays on its base while the checks run; once they
    end it is rebuilt without #3, and drive's passed check of #1 moves with #1 across #3's file (`_rekey_lean`)."""
    h = _lane(world, monkeypatch, nonlean_first=False)
    h.d.lowrisk_lane = False
    world.gh.put(1, world.change({"pkg/T.lean": "theorem T : True := trivial\n"}))
    world.gh.put(2, world.change({"x.txt": "2\n"}))
    world.gh.put(3, world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}), body=FIX_BODY.format(n=3))
    h.review(1)
    h.tick()
    one, two, _ = (m.commit for m in h.q.chain())
    assert [c for c, *_ in h.launched] == [one], world.logs
    h.hand[two] = ["r-hand"]
    h.run_state["r-hand"] = ("running", None, None)
    _red(h)
    h.d.lowrisk_lane = True
    checked, nxt = _checked(world), _tip(world)
    assert h.tick().endswith("landed #3 on the low-risk lane"), world.logs
    tip = _checked(world)
    assert _subject(world, "checked") == "Queue: #3 change 3" and _git(world.origin, "rev-parse", "checked^1") == checked
    assert _tip(world) == nxt and _landed(world) == [f"research queue drive: landed #3 on `checked` at {tip[:12]} {TAG}"]
    assert any("low-risk lane waits" in x for x in world.logs) and len(_fixes_only(world)) == 1, world.logs
    assert h.posted[-1][0] == tip and "pkg/tests/test_x.py" in _git(world.origin, "ls-tree", "-r", "--name-only", tip)
    h.tick()
    assert _tip(world) == nxt and _checked(world) == tip and h.cancelled == [], world.logs
    (rid,) = [r for r in h.lean_ids() if r != "r-hand"]
    h.run_state.update({rid: ("done", 0, None), "r-hand": ("done", 0, None)})
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #1"), world.logs
    assert len(h.launched) == 1 and h.cancelled == []
    assert any(f"the Lean check {rid} of {one[:12]} holds " in x and x.endswith(
        "trees that differ in no file `lowrisk_excludes` names (pkg/tests/test_x.py)") for x in world.logs), world.logs
    assert "Queue: #1 change 1" in _git(world.origin, "log", "--format=%s", "checked")


@pytest.mark.parametrize("excludes", ["drive.toml's", "without *.lean"])
def test_a_fix_whose_merge_changes_lean_doesnt_land_while_a_lean_check_cant_move(world, monkeypatch, excludes):
    """drive.toml's `lowrisk_excludes` keep a change to Lean off the lane; with `*.lean` missing from them, the lane's rule for
    fixes refuses it by name: a Lean check of `next` reads it."""
    h = _lane(world, monkeypatch)
    if excludes != "drive.toml's":
        h.d.lowrisk_excludes = tuple(x for x in EXCLUDES if x != "*.lean")
    h.d.lowrisk_lane = False
    world.gh.put(1, world.change({"verity/core/v.py": "1\n"}))
    world.gh.put(2, world.change({"pkg/U.lean": "theorem U : True := trivial\n"}), body=FIX_BODY.format(n=2))
    h.tick()
    h.hand[h.q.chain()[0].commit] = ["r-hand"]
    h.run_state["r-hand"] = ("running", None, None)
    _red(h)
    h.d.lowrisk_lane = True
    checked = _checked(world)
    h.tick()
    assert _checked(world) == checked and not _landed(world) and not _fixes_only(world), world.logs
    refused = [x for x in world.logs if "low-risk lane, for fixes of reds only, skipped" in x]
    if excludes == "drive.toml's":
        assert refused == [] and not any("low-risk lane waits" in x for x in world.logs), world.logs
    else:
        assert refused == ["research queue drive: low-risk lane, for fixes of reds only, skipped #2 (a fix of suite pkg, "
                           "but its change touches what a Lean check reads: pkg/U.lean)"], world.logs


def test_the_after_landing_run_waits_for_a_fresh_low_risk_fix_of_a_red_and_launches_fix_wait_s_after_its_mark(world, monkeypatch):
    """#1 lands on review; #2, a fix of red `suite pkg` marked just now, lies in a running Lean check's prefix, so the lane
    doesn't land it yet: the after-landing run of #1 waits for it, and launches `FIX_WAIT_S` after #2's mark without it."""
    h = _lane(world, monkeypatch)
    h.d.lowrisk_lane = False
    world.gh.put(1, world.change({"x.txt": "1\n"}))
    world.gh.put(2, two := world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}), body=FIX_BODY.format(n=2))
    h.tick()
    h.hand[h.q.chain()[1].commit] = ["r-hand"]
    h.run_state["r-hand"] = ("running", None, None)
    _red(h)
    marked = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(h.clock))
    h.q.marked_at = lambda n, head: marked if (n, head) == (2, two) else None
    h.review(1)
    h.d.lowrisk_lane = True
    assert h.tick().startswith("landed #1"), world.logs
    tip = _checked(world)
    assert h.posted == [] and not _landed(world), world.logs
    assert f"research queue drive: the after-landing run of `checked` {tip[:12]} waits: #2 at {two[:8]}, a low-risk fix of " \
           f"suite pkg marked 0 min ago, isn't landed yet: it launches once that lands, or {FIX_WAIT_S // 60} min after its " \
           "mark" in world.logs, world.logs
    h.clock += FIX_WAIT_S
    h.tick()
    assert [c for c, *_ in h.posted] == [tip] and _checked(world) == tip, world.logs
