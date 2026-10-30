"""The review landing onto `checked` (drive.toml's `land_into`: `main` holds only tested commits).  A
landing fast-forwards `checked`, never `main`; `main` is fast-forwarded only to a commit of `checked` whose after-landing run
passed every step on the merge gate; a red run moves nothing on `main`, and its culprit is reverted on `checked`.  A job off
the gate (ci.toml's `gate = false`: circuit-check) runs in that run and reverts its culprit, but holds no promotion."""
from __future__ import annotations

import json

import pytest

from research.drive import AFTER_LANDING, AFTER_LANDING_OPTIONAL, FIX_WINDOW_S, TAIL
from research.queue import DriveFlags, QueueError
from test_drive_review import CI_TOML, Review, _main, _move_main, _owning, _subject
from test_queue import World, _git
from test_queue_drive import _failed_run, _steps_run

#: the steps `main` waits for under `CI_TOML`
GATED = [s for s in AFTER_LANDING if s not in AFTER_LANDING_OPTIONAL and s != "circuit-check"]


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


def _gated(world: World, monkeypatch) -> Review:
    """A review landing onto `checked` in a tree whose ci.toml takes circuit-check off the merge gate."""
    _move_main(world, {"tools/verity/check/ci.toml": CI_TOML})
    return Review(world, monkeypatch, land_into="checked")


def _landed(h: Review, files: dict[str, str]) -> str:
    h.world.gh.put(1, h.world.change(files))
    h.review(1)
    assert h.tick().startswith("landed #1"), h.world.logs
    return _checked(h.world)


def test_main_follows_once_every_step_on_the_gate_passed_while_circuit_check_still_runs(world, monkeypatch, tmp_path):
    h = _gated(world, monkeypatch)
    main = _main(world)
    tip = _landed(h, {"a.txt": "1\n"})
    h.tick()
    assert _main(world) == main                                 # nothing passed yet
    h.evidence = [_steps_run(tmp_path, "r-post-0", {s: "passed" for s in GATED[:-1]})]
    h.tick()
    assert _main(world) == main                                 # one step on the gate hasn't passed
    h.evidence = [_steps_run(tmp_path / "all", "r-post-0", {s: "passed" for s in GATED})]
    h.tick()
    pl = h.postland()
    assert _main(world) == tip and pl["gated"] == tip and pl["run"]["ids"] == ["r-post-0"] and h.cancelled == []
    assert pl["verified"] != tip                                # the run, circuit-check in it, goes on
    assert h.after == [(main, {tip: "r-post-0"})]


def _circuit_red(h: Review, tmp_path) -> None:
    """The after-landing run of `checked`'s tip passes every step on the gate and fails circuit-check."""
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_steps_run(tmp_path, "r-post-0", {**{s: "passed" for s in GATED}, "circuit-check": "failed"})]


def test_a_red_circuit_check_reverts_its_culprit_on_checked_at_once_and_holds_no_promotion(world, monkeypatch, tmp_path):
    """No `FIX_WINDOW_S` for circuit-check, and no marked fix holds its revert: it runs only after landing, so each landing on its
    red goes unchecked."""
    h = _gated(world, monkeypatch)
    tip = _landed(h, {"pkg/x.circuit-check.py": "X = 1\n"})
    world.gh.put(9, world.change({"c.txt": "9\n"}), body="Lane `lane-9`. fixes step circuit-check")
    _circuit_red(h, tmp_path)
    h.tick()
    (ring,) = [p for p in h.posts if "Culprit: #1" in p]
    assert "step circuit-check is red on `checked`" in ring and "Drive reverts #1 now" in ring, ring
    reverted = _checked(world)
    assert _git(world.origin, "rev-parse", f"{reverted}^") == tip and h.posted[-1][0] == reverted
    assert any("a red circuit-check is reverted at once" in p for p in h.posts), h.posts
    assert _main(world) == tip
    assert "pkg/x.circuit-check.py" not in _git(world.origin, "ls-tree", "-r", "--name-only", reverted)
    h.evidence = []
    h.tick()
    assert _main(world) == tip                                  # the revert's own run hasn't passed yet
    _pass(h)
    h.tick()
    assert _main(world) == reverted and not h.postland()["red"]


def test_while_circuit_check_is_red_a_pr_touching_its_inputs_waits_and_an_unrelated_one_lands(world, monkeypatch, tmp_path):
    """The red here has no culprit (#1 touches none of circuit-check's inputs), so nothing is reverted and it stands until a run
    passes circuit-check; meanwhile only the PRs whose change touches those inputs (`queue.forces_cold`) wait."""
    h = _gated(world, monkeypatch)
    _landed(h, {"a.txt": "1\n"})
    _circuit_red(h, tmp_path)
    h.tick()
    assert h.postland()["red"]["step circuit-check"]["culprit"] is None
    world.gh.put(2, world.change({"verity/ml/circuits/library.py": "X = 2\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    world.gh.put(4, world.change({"verity/ml/circuits/README.md": "4\n", "verity/ml/circuits/tests/test_x.py": "4\n"}))
    for n in (2, 3, 4):
        h.review(n)
    assert h.tick().startswith("landed #4"), world.logs
    assert [m.pr for m in h.q.chain()] == [2]
    assert _subject(world, "checked").startswith("Queue: #4") and _subject(world, "checked^1").startswith("Queue: #3")
    out = h.tick()
    assert out.startswith("nothing reviewed lands: #2") and "touches circuit-check's inputs" in out, out
    _pass(h)
    assert h.tick().startswith("landed #2"), world.logs
    assert "step circuit-check" not in h.postland()["red"]


def test_a_pr_that_says_it_fixes_the_red_circuit_check_lands_through_the_hold(world, monkeypatch, tmp_path):
    h = _gated(world, monkeypatch)
    _landed(h, {"a.txt": "1\n"})
    _circuit_red(h, tmp_path)
    h.tick()
    world.gh.put(2, world.change({"verity/ml/circuits/library.py": "X = 2\n"}), body="Lane `lane-2`. Fixes step `circuit-check`")
    h.review(2)
    assert h.tick().startswith("landed #2"), world.logs


def test_a_red_circuit_check_cancels_no_step_on_the_gate_still_running(world, monkeypatch, tmp_path):
    """A red run is cancelled once only its tail (circuit-check, flock-rust-tests) is left; when only circuit-check, off the
    gate, is red, flock-rust-tests runs to its end first, so its pass can still move `main`."""
    h = _gated(world, monkeypatch)
    tip = _landed(h, {"a.txt": "1\n"})
    steps = {**{s: "passed" for s in AFTER_LANDING if s not in TAIL}, "circuit-check": "failed"}
    h.run_state["r-post-0"] = ("running", None, None, "step circuit-check")
    h.evidence = [_steps_run(tmp_path, "r-post-0", steps)]
    h.tick()
    assert h.cancelled == [] and h.postland()["run"]["ids"] == ["r-post-0"] and _main(world) != tip
    h.evidence = [_steps_run(tmp_path / "2", "r-post-0", {**steps, "flock-rust-tests": "passed"})]
    h.tick()
    assert h.cancelled == ["r-post-0"] and h.postland()["run"] is None and _main(world) == tip


@pytest.mark.parametrize("red", ["pytest", "lean-audit"])
def test_main_stays_while_a_step_on_the_gate_is_red_whatever_circuit_check_did(world, monkeypatch, tmp_path, red):
    h = _gated(world, monkeypatch)
    main = _main(world)
    _landed(h, {"a.txt": "1\n"})
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_steps_run(tmp_path, "r-post-0", {**{s: "passed" for s in AFTER_LANDING}, red: "failed"})]
    h.tick()
    h.tick()
    assert _main(world) == main and "gated" not in h.postland()
    assert any(f"step {red} is red on `checked`" in p for p in h.posts), h.posts


def test_main_follows_a_baseline_of_every_step_on_the_gate(world, monkeypatch):
    """ci's `--steps` baseline counts once it names every step on the merge gate: circuit-check, off it, needn't be among them."""
    from research import queue as Q

    h = _gated(world, monkeypatch)
    main = _main(world)
    tip = _landed(h, {"a.txt": "1\n"})
    state = h.q.load_state()
    state["drive"]["postland"]["run"] = None
    h.q.save_state(state)
    assert Q.baseline(h.q, tip, "ci", ",".join(GATED[:-1])) == 0
    h.tick()
    assert _main(world) == main
    assert Q.baseline(h.q, tip, "ci", ",".join(GATED)) == 0
    h.tick()
    assert _main(world) == tip and any("(ci baseline)" in x for x in world.logs), world.logs


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
