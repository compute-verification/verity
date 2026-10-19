"""Every sync keeps `next` on its base while drive's hold applies (CI-100), and a former train tip's check is a hedge drive
cancels once a train above it has passed (CI-85).  On `test_queue`'s throwaway origin, with drive's launches and run states
injected (`test_queue_drive.Harness`)."""
from __future__ import annotations

import pytest

from research.queue import built_on
from test_queue import World, _git
from test_queue_drive import Harness, _check, _steps_run, _tip


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _base(world: World) -> str:
    """The base `next` was built on, on the origin."""
    q = world.queue()
    q._clone()
    q.fetch()
    return built_on(q.workdir, "refs/remotes/origin/next")[0]


def _launched(h: Harness) -> dict:
    return h.q.load_state()["drive"]["launched"]


def test_eject_while_drive_holds_next_leaves_its_base_and_rebuilds_once_the_check_ends(world, monkeypatch):
    """19 Oct 08:16Z: `research queue eject 2727` ran `Queue.sync`, which had no hold: it rebuilt all 118 PRs onto a `main` that
    had moved by files `next` doesn't change, and pushed `next`, which would have cancelled every running train."""
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.tick()
    old, base = _tip(world), _base(world)
    _git(world.dev, "push", "-q", "origin", f"{world.change({'z.txt': 'lander\n'})}:refs/heads/main")
    assert "stays on" in h.tick() and _tip(world) == old

    q = world.queue()                                   # `research queue eject`'s own queue: drive's launches from the state file
    q.run_states = lambda repo, ids: {i: h.run_state[i] for i in ids}
    q.eject(2, "a test", "ci")
    members, _ = q.sync()
    assert _tip(world) == old and _base(world) == base and [m.pr for m in members] == [1, 2]
    assert any("stays on" in x for x in world.logs)

    (rid,) = h.run_state
    h.run_state[rid] = ("done", 1, None)
    q.sync()
    assert _base(world) == _git(world.origin, "rev-parse", "main") and world.on_next(q) == [1]


def test_eject_rebuilds_next_when_main_moved_over_a_file_the_running_train_changes(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.tick()
    _git(world.dev, "push", "-q", "origin", f"{world.change({'b.txt': 'lander\n'})}:refs/heads/main")
    q = world.queue()
    q.run_states = lambda repo, ids: {i: h.run_state[i] for i in ids}
    q.eject(2, "a test", "ci")
    q.sync()
    assert _base(world) == _git(world.origin, "rev-parse", "main") and world.on_next(q) == [1]


def _three_trains(world: World, monkeypatch, **kw) -> tuple[Harness, list[str]]:
    """`next` = #1..#4 in trains of 2: [#1 #2] and [#1..#4] checked, and [#1 #2 #3], the tip before #4 joined, still running
    its check: (the harness, the commits of the members)."""
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 2, 3
    for k, v in kw.items():
        setattr(h.d, k, v)
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    world.gh.put(4, world.change({"d.txt": "4\n"}))
    h.tick()
    commits = [m.commit for m in h.q.chain()]
    assert [c for c, *_ in h.launched] == [commits[1], commits[2], commits[3]]
    return h, commits


def test_a_former_tips_check_is_cancelled_and_forgotten_once_the_train_above_it_passed(world, monkeypatch):
    """CI-85: after a re-cut of `next`, a commit that was a train's tip keeps its check running.  `land` would land it should
    the train above fail, but once that train has passed, the check only holds a slot.  The landing of the train that passed
    is kept from this tick (a push that lost a race, say) so the rule shows on its own."""
    h, c = _three_trains(world, monkeypatch)
    middle = _launched(h)[c[2]]["ids"]
    _check(world, monkeypatch, c[3])
    h.d.land = lambda drive, dry_run: None
    h.tick()
    assert h.cancelled == middle
    assert c[2] not in _launched(h) and {c[1], c[3]} <= set(_launched(h))


def test_a_former_tips_check_is_cancelled_once_the_train_above_passed_all_but_its_touched_suites(world, monkeypatch, tmp_path):
    """With pytest deferred, a train whose check passed every other step waits on its touched suites before it lands: the
    former tip below it is cancelled then, not once it lands."""
    from research.drive import CHECK_STEPS

    touched = iter(range(100))
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running, h.d.defer = 2, 3, frozenset({"pytest"})

    def launch_touched(repo, commit, machine, base):
        rid = f"r-touched-{next(touched)}"
        h.run_state[rid] = ("running", None, None)
        return [rid], ""

    h.d.launch_touched = launch_touched
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    world.gh.put(4, world.change({"d.txt": "4\n"}))
    h.tick()
    c = [m.commit for m in h.q.chain()]
    launched = _launched(h)
    top, middle = launched[c[3]], launched[c[2]]
    steps = [_steps_run(tmp_path, top["ids"][0], {x: "passed" for x in CHECK_STEPS if x != "pytest"})]
    h.d.evidence = lambda repo, ids: steps if ids == top["ids"] else []
    h.tick()
    assert _git(world.origin, "rev-parse", "main") != c[3]                    # its touched suites still run
    assert sorted(h.cancelled) == sorted([*middle["ids"], *middle["touched"]])
    assert c[2] not in _launched(h) and c[3] in _launched(h)


def test_a_former_tips_check_runs_on_outside_the_count_while_the_train_above_runs(world, monkeypatch):
    """Until the train above passes, the former tip is a hedge: it keeps running, and it takes no train's slot."""
    h, c = _three_trains(world, monkeypatch)
    h.tick()
    assert h.cancelled == [] and c[2] in _launched(h)
    world.gh.put(5, world.change({"e.txt": "5\n"}))
    world.gh.put(6, world.change({"f.txt": "6\n"}))
    h.tick()
    assert h.launched[-1][0] == _tip(world)                         # 3 running, 2 counted: the new tip's train starts
    assert h.cancelled == [] and c[2] in _launched(h)


def test_a_former_tip_bisection_checks_is_never_cancelled(world, monkeypatch):
    h, c = _three_trains(world, monkeypatch)
    state = h.q.load_state()
    state["drive"]["bisect"] = {c[3]: {"check": c[2], "among": [3, 4], "ids": state["drive"]["launched"][c[2]]["ids"],
                                       "at": h.clock, "fails": []}}
    h.q.save_state(state)
    _check(world, monkeypatch, c[3])
    h.d.land = lambda drive, dry_run: None
    h.tick()
    assert h.cancelled == [] and c[2] in _launched(h)
