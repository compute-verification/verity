"""The review landing onto `checked` (drive.toml's `land_into`: `main` holds only tested commits).  A
landing fast-forwards `checked`, never `main`; `main` is fast-forwarded only to a commit of `checked` whose after-landing run
passed every step on the merge gate; a red run moves nothing on `main`, and its culprit, unless fixed forward in time, is
reverted on `checked`.  A job off the gate (drive's policy, policy.toml's `off_gate`: circuit-check) runs in that run and names
its culprit as any red does, but holds no promotion."""
from __future__ import annotations

import json

import pytest

import research.drive as D
from research.drive import AFTER_LANDING, AFTER_LANDING_OPTIONAL, TAIL
from research.queue import DriveFlags, QueueError
from test_drive_review import GATED_TREE, Review, _main, _move_main, _owning, _rings, _subject
from test_queue import World, _git
from test_queue_drive import _failed_run, _steps_run

#: the steps `main` waits for under `GATED_TREE`
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


def test_the_status_command_reads_the_verified_queues_admission_onto_main_not_nexts_places_or_checkeds_landings(world, monkeypatch):
    """`next` and `checked` stay frozen behind the verified queue, which builds onto `main` from the marked PRs: `research queue
    status` gives no place on `next`, and a head is landed once it is on `main`, not on `checked`."""
    h = Review(world, monkeypatch, land_into="checked")
    for n, f in ((1, "a.txt"), (2, "b.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    h.tick()
    drive, loop = h.q.status(), h.q.status(loop=True)
    assert [(drive[n].state, drive[n].position) for n in (1, 2)] == [("queued", None), ("queued", 1)], drive
    assert [(loop[n].state, loop[n].position) for n in (1, 2)] == [("ready", None), ("ready", None)], loop
    assert loop[1].why == loop[2].why and loop[2].why.startswith("admitted: the verified queue builds it into a batch onto `main`")
    _pass(h)
    h.tick()
    assert _main(world) == _checked(world), world.logs
    loop = h.q.status(loop=True)
    assert (loop[1].state, loop[1].why, loop[2].state) == ("queued", "landed: its head is on `main`", "ready"), loop


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
    (ring,) = _rings(h)
    assert "suite pkg is red on `checked`" in ring and "Culprit: #2" in ring, ring
    assert _checked(world) == tip and _main(world) == main
    h.clock += D.FIX_FORWARD_S
    h.tick()
    reverted = _checked(world)
    assert _git(world.origin, "rev-parse", f"{reverted}^") == tip and _main(world) == main
    assert "pkg/tests/test_x.py" not in _git(world.origin, "ls-tree", "-r", "--name-only", reverted)
    assert any(p.startswith("@lane-2 @pkg-owner queue: #2 reverted on `checked`") for p in h.posts), h.posts
    assert h.posted[-1][0] == reverted
    _pass(h, ("pkg",))
    h.tick()
    assert _main(world) == reverted and not h.postland()["red"]
    assert world.on_next(h.q) == []                             # #2, reverted, is still open: its head is on `checked`


def test_a_failure_at_or_below_a_tip_that_passed_it_since_is_no_red(world, monkeypatch, tmp_path):
    """A late rerun of an old shard can fail a suite that a newer run passed above it (its green), or that `main` was verified
    past: that failure is no red, so it holds no landing and reverts no culprit."""
    h = Review(world, monkeypatch, land_into="checked")
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    old = _checked(world)
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.review(2)
    assert h.tick().startswith("landed #2"), world.logs
    new = _checked(world)
    late = {"tip": old, "ids": ["r-late"]}
    for pl in ({"green": {"suite pkg": new}, "verified": old}, {"verified": new, "verified_as": {"steps": ["pytest"]}}):
        h.d._name({}, pl, late, "suite pkg", tmp_path, {"suite": "pkg", "step": "pytest"}, [])
        assert pl["red"] == {}, pl
    assert sum(f"suite pkg failed at `checked` {old[:12]} (run r-late), but it passed at" in x for x in world.logs) == 2


def test_a_pr_containing_a_head_landed_on_checked_never_waits_for_it_and_one_on_an_unlanded_pr_still_does(world, monkeypatch):
    """A PR that landed on `checked` stays open until `main` follows it, and may be held or lose its mark meanwhile: a PR built
    on its head doesn't wait for it.  A PR built on an unlanded PR that isn't admitted still waits."""
    h = Review(world, monkeypatch, land_into="checked")
    world.gh.put(5, five := world.change({"x.txt": "5\n"}))
    h.review(5)
    assert h.tick().startswith("landed #5"), world.logs
    world.gh.put(5, five, labels=("hold",))
    world.gh.put(7, world.change({"k.txt": "7\n"}, base=five))
    world.gh.put(6, six := world.change({"y.txt": "6\n"}, base=five), ready=False)
    world.gh.put(8, world.change({"z.txt": "8\n"}, base=six))
    h.tick()
    st = h.q.status()
    assert st[7].state != "waiting" and st[7].after == (), st[7]
    assert st[8].state == "waiting" and st[8].why == "contains #6, which isn't admitted" and st[8].held == {6: six}


def _cli(h: Review, *args: str) -> list[str]:
    """`research queue ARGS` against ``h``'s drive: its clone, origin and store."""
    w = h.world
    return [*args, "--workdir", str(h.q.workdir), "--remote", str(w.origin), "--repo", "o/r", "--store", str(w.tmp / "store"),
            "--rules", str(w.dev / "ci" / "queue.toml")]


def test_ci_baseline_moves_main_at_once_beside_a_tick_holding_its_lock_and_writes_nothing_of_drives(world, monkeypatch, capsys):
    """ci's `research queue baseline` fast-forwards `main` itself, at once: a tick holds the clone's lock for its whole run (an
    hour and more), and a baseline that waited for it, then left `main` to the tick after, kept `main` that far behind a green
    `checked`.  It reads drive's clone as the tick left it and writes no ref of it and nothing of drive's state, takes only a
    commit on `checked`'s first-parent chain whose steps cover every step on the merge gate, and is refused but by ci and off
    drive."""
    from research import queue as Q
    from research.merge import DRIVE_ENV

    _move_main(world, {**GATED_TREE, "tools/verity/check/drive.toml": 'land_into = "checked"\n'})
    h = Review(world, monkeypatch, land_into="checked")
    main = _main(world)
    tip = _landed(h, {"a.txt": "1\n"})
    argv = _cli(h, "baseline", tip, "--by", "ci")
    state, refs = h.q.state_path.read_bytes(), _git(h.q.workdir, "for-each-ref")
    lock = Q._lock(h.q.workdir)                                 # a tick runs
    try:
        assert Q.main(_cli(h, "status")) == 1 and "another `research queue` holds" in capsys.readouterr().err
        monkeypatch.delenv(DRIVE_ENV)
        assert Q.main(argv) == 1 and "this isn't drive" in capsys.readouterr().err
        monkeypatch.setenv(DRIVE_ENV, "1")
        assert Q.main(_cli(h, "baseline", tip, "--by", "lane-3")) == 1 and "only with --by ci" in capsys.readouterr().err
        assert Q.main([*argv, "--steps", ",".join(GATED[:-1])]) == 1
        assert f"--steps leaves out {GATED[-1]}, on the merge gate" in capsys.readouterr().err
        assert Q.main(_cli(h, "baseline", f"{tip}^2", "--by", "ci")) == 1      # #1's head, merged into `checked`
        assert "is not on `checked`'s first-parent chain" in capsys.readouterr().err
        assert _main(world) == main
        assert Q.main([*argv, "--steps", ",".join(GATED)]) == 0                # circuit-check, off the gate, needn't be among them
        assert _main(world) == tip and f"`main` fast-forwarded from {main[:12]} to {tip[:12]}" in capsys.readouterr().out
        assert Q.main(argv) == 0 and "already: nothing to move" in capsys.readouterr().out
    finally:
        lock.close()
    assert h.q.state_path.read_bytes() == state and _git(h.q.workdir, "for-each-ref") == refs


def _gated(world: World, monkeypatch) -> Review:
    """A review landing onto `checked` in a tree whose drive policy takes circuit-check off the merge gate."""
    _move_main(world, GATED_TREE)
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


def test_a_red_circuit_check_names_its_culprit_holds_no_promotion_and_its_revert_waits_for_a_marked_fix(world, monkeypatch,
                                                                                                         tmp_path):
    """circuit-check runs only after landing, so each landing on its red goes unchecked: its ring says no PR touching its inputs
    lands until it passes, and `main` moves all the same.  Its culprit is fixed forward or reverted as every red's is: a fix
    marked (here one saying `fixes step circuit-check`) holds the revert past `FIX_FORWARD_S`, and once it is ejected the next
    tick reverts it on `checked`."""
    h = _gated(world, monkeypatch)
    tip = _landed(h, {"pkg/x.circuit-check.py": "X = 1\n"})
    world.gh.put(9, world.change({"c.txt": "9\n"}), body="Lane `lane-9`. fixes step circuit-check")
    _circuit_red(h, tmp_path)
    h.tick()
    (ring,) = [p for p in h.posts if "Culprit: #1" in p]
    assert "step circuit-check is red on `checked`" in ring and ring.endswith(
        "or drive reverts #1 then. No PR touching circuit-check's inputs lands until it passes on `checked`."), ring
    assert _checked(world) == _main(world) == tip
    h.clock += D.FIX_FORWARD_S
    h.tick()
    assert _checked(world) == tip and any("culprit #1 isn't reverted: #9 at " in x for x in world.logs), world.logs
    world.gh.ready.discard((9, world.gh.prs[9].head))
    h.tick()
    reverted = _checked(world)
    assert _git(world.origin, "rev-parse", f"{reverted}^") == tip and h.posted[-1][0] == reverted
    assert any("#1 reverted on `checked`: step circuit-check is red" in p for p in h.posts), h.posts
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
    world.gh.put(2, world.change({"verity/definitions/circuits/library.py": "X = 2\n"}))
    world.gh.put(3, world.change({"b.txt": "3\n"}))
    world.gh.put(4, world.change({"verity/circuits/README.md": "4\n", "verity/circuits/tests/test_x.py": "4\n"}))
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
    world.gh.put(2, world.change({"verity/definitions/circuits/library.py": "X = 2\n"}), body="Lane `lane-2`. Fixes step `circuit-check`")
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


def test_a_sharded_run_that_passes_moves_main_and_the_verified_record_together_and_nothing_rings_or_relaunches(world, monkeypatch):
    """The 3-shard after-landing run of `checked`'s tip passed and `main` moved to it on its gate, but no shard ran ratchet-gate
    (in no shard of ci.toml's plans): drive took the run as one that passed without it, so `checked`'s last verified commit and
    its red record stayed, the behind-ring rang, and no run launched for that tip.  With ratchet-gate in the last shard
    (`check.py shard`), the pass verifies the tip and ends the red record in the tick `main` moves, and the next ticks neither
    ring nor launch again."""
    from research.drive import BEHIND_EVERY_S, BEHIND_S

    _move_main(world, {**GATED_TREE, "tools/verity/check/policy.toml": 'schema = "policy/v0"\noff_gate = ["circuit-check", "ratchet-gate"]\n'})
    h = Review(world, monkeypatch, land_into="checked")
    h.d.after_landing_shards, h.d.after_landing_no_cache = 3, True
    h.d.records_artifact = lambda rid: f"art:{'a' * 64}@{'b' * 64}" if rid == "r-post-0" else None
    before = h.postland()["verified"]
    state = h.q.load_state()
    state["drive"]["postland"]["red"] = {"step lean-suites": {"tip": before, "ids": ["r-old"], "first": "x", "culprit": None}}
    h.q.save_state(state)
    tip = _landed(h, {"a.txt": "1\n"})
    shards = (("lean-audit", "pytest"), ("lean-build", "lean-unit-cut", "lean-audit", "lean-suites"),
              ("flock-circuit-build", "circuit-check", "flock-rust-tests", "ratchet-gate"))
    assert {s for x in shards for s in x} == set(AFTER_LANDING) - set(AFTER_LANDING_OPTIONAL)
    assert [kw for _, _, kw in h.posted] == [{"cc_records": True}], "circuit-check's records step first"
    h.run_state["r-post-0"] = ("done", 0, None)
    h.tick()
    assert [kw["shard"] for _, _, kw in h.posted[1:]] == ["1/3", "2/3", "3/3"]
    h.run_state.update({f"r-post-{k + 1}": ("done", 0, None) for k in range(3)})
    h.evidence = [_steps_run(world.tmp, f"r-post-{k + 1}", {s: "passed" for s in x}) for k, x in enumerate(shards)]
    h.tick()
    pl = h.postland()
    assert _main(world) == tip == pl["verified"] and pl["run"] is None and not pl["red"] and not pl["pending"], world.logs
    assert "failed" not in pl and pl["verified_run"] == "r-post-1 r-post-2 r-post-3"
    for _ in range(2):
        h.clock += BEHIND_S + BEHIND_EVERY_S
        h.tick()
    assert len(h.posted) == 4 and not any("last verified commit" in p for p in h.posts), h.posts
    assert not any("passed without passing" in p for p in h.posts), h.posts


def _shard(tmp_path, rid: str, steps: dict[str, str], live: dict[str, str] | None = None, preflight: bool = False,
           state: str = "running"):
    """A shard's run dir: ``steps`` ended in its steps.jsonl, and its live record (`LIVE`, the check's ``state``) naming the
    preflight's steps, those and ``live``'s, still to end (pending, running); with ``preflight``, the preflight's alone, its
    groups not yet planned."""
    from research.drive import LIVE, PREFLIGHT

    d = _steps_run(tmp_path, rid, steps)
    planned = {s: "running" if preflight else "passed" for s in PREFLIGHT} | ({} if preflight else {**steps, **(live or {})})
    (d / LIVE).write_text(json.dumps({"schema": "check-live/v1", "state": state,
                                      "steps": [{"step": s, "state": v} for s, v in planned.items()]}))
    return d


def _part(suite: str, k: int, status: str = "passed") -> dict:
    """``suite``'s row of its part K of 3 by test id, as suites.py reports it."""
    return {"name": suite, "suite": suite, "status": status, "failed_tests": ["tests/test_x.py::test_y"] * (status == "failed"),
            "part": {"ks": [k], "m": 3, "digest": "d", "by": "id"}}


def test_a_step_more_than_one_shard_holds_passed_only_once_every_shard_that_holds_it_passed_it(tmp_path):
    """A sharded after-landing run's shards hold parts of one step (lean-audit's packages, lean-suites' suites, lean-build beside
    them): it passed once every shard that holds it passed it (or skipped it, or failed it only on what an allowance covers),
    none having it still to run by its live record, so another shard's pass doesn't pass a part that runs on, failed or was
    stopped; a twin's pass passes it for its shard, and twins' other words rank whatever their order (`RANK`); a step an ended
    check never started was stopped; and nothing passed while a shard's plan isn't known.  A suite in parts passed once every
    part of one partition did."""
    from research.drive import PREFLIGHT, shard_ended, shard_passed, shard_steps, suite_passed

    one = _shard(tmp_path, "r1", {"lean-build": "passed", "lean-suites": "passed", "pytest": "passed"})
    two = _shard(tmp_path, "r2", {"lean-build": "passed"}, {"lean-suites": "running", "lean-agreement": "pending"})
    view = shard_steps([[one], [two]], [True, False])
    assert view[1] == {**dict.fromkeys(PREFLIGHT, "passed"), "lean-build": "passed", "lean-suites": None, "lean-agreement": None}
    assert shard_passed(view) == shard_ended(view) == {*PREFLIGHT, "lean-build", "pytest"}

    failed, stopped, skipped = (shard_steps([[one], [_shard(tmp_path / s, "r2", {"lean-suites": s})]], [True, False])
                                for s in ("failed", "stopped", "skipped"))
    assert "lean-suites" not in shard_passed(failed) and "lean-suites" in shard_passed(failed, {"lean-suites"})
    assert "lean-suites" not in shard_passed(stopped, {"lean-suites"}) and "lean-suites" in shard_ended(stopped)
    assert "lean-suites" in shard_passed(skipped)
    twins = [_shard(tmp_path / "t", "r2", {"lean-suites": "stopped"}), _shard(tmp_path / "t", "r2b", {"lean-suites": "passed"})]
    assert "lean-suites" in shard_passed(shard_steps([[one], twins], [True, False])), "a twin's pass passes its shard"
    twins[1] = _shard(tmp_path / "u", "r2b", {}, {"lean-suites": "running"})
    assert shard_steps([[one], twins], [True, False])[1]["lean-suites"] is None, "one twin still runs it"
    for a, b in (("stopped", "failed"), ("failed", "stopped"), ("skipped", "stopped"), ("stopped", "skipped")):
        pair = [_shard(tmp_path / f"{a}-{b}", "r2", {"lean-suites": a}), _shard(tmp_path / f"{a}-{b}", "r2b", {"lean-suites": b})]
        assert shard_steps([[one], pair], [True, False])[1]["lean-suites"] == ("failed" if "failed" in (a, b) else "skipped")
    closed = shard_steps([[one], [_shard(tmp_path / "c", "r2", {"lean-build": "failed"}, {"lean-suites": "pending"}, state="failed")]],
                         [True, False])
    assert closed[1]["lean-suites"] == "stopped", "its group stopped on lean-build's failure: it never starts"
    assert "lean-suites" in shard_ended(closed) and "lean-suites" not in shard_passed(closed, {"lean-suites"})

    early, bare = _shard(tmp_path / "e", "r2", {}, preflight=True), _steps_run(tmp_path / "b", "r2", {"lean-build": "passed"})
    assert shard_steps([[one], [early]], [True, False]) is None and shard_steps([[one], [bare]], [True, False]) is None
    assert shard_steps([[one], [bare]], [True, True])[1] == {"lean-build": "passed"}, "a twin exited 0: its steps.jsonl"
    assert shard_steps([[bare]], [False]) == [{"lean-build": "passed"}], "an unsharded run: its steps.jsonl"

    assert suite_passed([{"name": "s", "status": "passed"}]) and suite_passed([_part("s", 3), _part("s", 1), _part("s", 2)])
    assert not suite_passed([_part("s", 1), _part("s", 3)]) and not suite_passed([_part("s", 1), _part("s", 2, "failed"), _part("s", 3)])
    assert not suite_passed([_part("s", 1), {**_part("s", 2), "part": {"ks": [2], "m": 3, "digest": "e"}}, _part("s", 3)])


def test_a_sharded_run_moves_main_only_once_every_shard_that_holds_a_step_passed_it(world, monkeypatch):
    """ci.toml's 3-shard plan runs lean-suites in each shard (the composition suite's three parts, lean-build again beside two
    of them): while it still runs in one shard, by that shard's live record, the others' passes move nothing on `main`; once it
    fails there the run is red, and neither the suite nor the step is green at its tip, so the red's culprit is sought among the
    landings since each last passed whole."""
    _move_main(world, GATED_TREE)
    h = Review(world, monkeypatch, land_into="checked")
    h.d.after_landing_shards = 3
    h.d.records_artifact = lambda rid: f"art:{'a' * 64}@{'b' * 64}" if rid == "r-post-0" else None
    h.d.evidence = lambda repo, ids: [d for d in h.evidence if d.name in ids]
    main = _main(world)
    tip = _landed(h, {"a.txt": "1\n"})
    h.run_state["r-post-0"] = ("done", 0, None)
    h.tick()
    assert [kw["shard"] for _, _, kw in h.posted[1:]] == ["1/3", "2/3", "3/3"]
    comp = "verity-core-composition"
    shards = ({s: "passed" for s in ("lean-audit", "pytest", "lean-build", "lean-suites")},
              {s: "passed" for s in ("lean-build", "lean-unit-cut", "lean-audit", "lean-agreement", "lean-changed")},
              {s: "passed" for s in ("flock-circuit-build", "circuit-check", "flock-rust-tests", "lean-build", "lean-suites",
                                     "ratchet-gate")})
    assert {s for x in shards for s in x} == set(AFTER_LANDING)
    h.run_state.update({"r-post-1": ("done", 0, None), "r-post-2": ("running", None, None), "r-post-3": ("done", 0, None)})
    h.evidence = [_shard(world.tmp, "r-post-1", shards[0]), _shard(world.tmp, "r-post-2", shards[1], {"lean-suites": "running"}),
                  _shard(world.tmp, "r-post-3", shards[2])]
    for d, k in ((h.evidence[0], 3), (h.evidence[2], 2)):
        (d / "suites-lean.json").write_text(json.dumps([_part(comp, k)]))
    h.tick()
    assert _main(world) == main and "gated" not in h.postland(), "lean-suites runs on in shard 2"

    h.run_state["r-post-2"] = ("done", 1, None)
    h.evidence[1] = _shard(world.tmp / "ended", "r-post-2", {**shards[1], "lean-suites": "failed"})
    (h.evidence[1] / "suites-lean.json").write_text(json.dumps([_part(comp, 1, "failed")]))
    h.tick()
    pl = h.postland()
    assert _main(world) == main and "gated" not in pl and pl["run"] is None and pl["failed"] == tip, world.logs
    green = pl.get("green") or {}
    assert green.get("step lean-build") == tip and green.get("step lean-suites") != tip and green.get(f"suite {comp}") != tip
    assert pl["red"][f"suite {comp}"]["base"] != tip, pl["red"]


def test_main_follows_a_baseline_of_every_step_on_the_gate(world, monkeypatch):
    """ci's `--steps` baseline counts once it names every step on the merge gate: circuit-check, off it, needn't be among them."""
    from research import queue as Q

    h = _gated(world, monkeypatch)
    main = _main(world)
    tip = _landed(h, {"a.txt": "1\n"})
    state = h.q.load_state()
    state["drive"]["postland"]["run"] = None
    h.q.save_state(state)
    with pytest.raises(QueueError, match=f"--steps leaves out {GATED[-1]}, on the merge gate"):
        Q.baseline(h.q, tip, "ci", ",".join(GATED[:-1]))
    assert _main(world) == main
    assert Q.baseline(h.q, tip, "ci", ",".join(GATED)) == 0 and _main(world) == tip
    h.tick()
    assert h.postland()["verified"] == tip, world.logs


def test_a_tick_takes_a_main_moved_beside_it_as_verified_and_follows_what_it_brought_once(world, monkeypatch):
    """A `main` moved beside drive (`research queue baseline` or `promote`) is `verified` from the next tick on, as drive's own
    promotion leaves it: the red records at or below it of steps on the merge gate end (a `suite` one too, the gate holding
    pytest and lean-suites), one off the gate stays, keeping as its newest green the old `verified` that passed it, and what
    `main` brought follows as a landing on `main` does, once."""
    from research import queue as Q

    h = _gated(world, monkeypatch)
    main = _main(world)
    tip = _landed(h, {"a.txt": "1\n"})
    state = h.q.load_state()
    red = lambda key: {"tip": tip, "base": main, "step": key.split(" ")[1], "culprit": None, "first": "x"}  # noqa: E731
    state["drive"]["postland"]["red"] = {k: red(k) for k in ("step lean-audit", "suite pkg::test_y", "step circuit-check")}
    h.q.save_state(state)
    assert Q.baseline(h.q, tip, "ci") == 0 and h.after == []
    h.tick()
    pl = h.postland()
    assert pl["verified"] == tip and pl["verified_as"]["steps"] == GATED and pl["pending"] == [] and pl["main"] == tip
    assert list(pl["red"]) == ["step circuit-check"] and pl["green"]["step circuit-check"] == main
    assert pl["green"]["step lean-audit"] == pl["green"]["suite pkg::test_y"] == tip
    assert h.after == [(main, {tip: ""})] and pl["run"]["ids"] == ["r-post-0"]
    h.tick()
    assert h.after == [(main, {tip: ""})] and h.postland()["verified"] == tip


def test_a_tick_whose_promotion_finds_main_moved_past_it_meanwhile_leaves_main_there(world, monkeypatch):
    """ci's baseline moves `main` while a tick runs, past the commit that tick moves it to: the tick's push is refused, which is
    no failure of the tick and moves `main` nowhere back; the next tick takes `main` and follows what it brought once."""
    from research import queue as Q

    h = Review(world, monkeypatch, land_into="checked")
    main = _main(world)
    one = _landed(h, {"a.txt": "1\n"})
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.review(2)
    assert h.tick().startswith("landed #2"), world.logs
    two = _checked(world)
    _pass(h)                                                    # the run of `one`: this tick moves `main` to it
    revert_reds = h.d._revert_reds

    def baseline_meanwhile(drive: dict) -> None:
        revert_reds(drive)
        assert Q.baseline(h.q, two, "ci") == 0

    monkeypatch.setattr(h.d, "_revert_reds", baseline_meanwhile)
    h.tick()
    assert _main(world) == two and h.postland()["verified"] == one and h.after == []
    assert any(f"`main` is at {two[:12]} already, which holds {one[:12]}" in x for x in world.logs), world.logs
    assert not any("was refused" in x for x in world.logs), world.logs
    monkeypatch.setattr(h.d, "_revert_reds", revert_reds)
    h.evidence = []                                             # the run of `two` has passed nothing yet
    for _ in range(2):
        h.tick()
        assert h.postland()["verified"] == two and h.after == [(main, {one: "", two: ""})]


def test_promote_moves_main_once_every_shard_ended_passing_the_gate_and_never_before(world, monkeypatch, capsys):
    """`research queue promote` moves `main` to the after-landing run's tip as soon as the run ends, not at drive's next tick
    (up to its 2 min sleep and an hour's tick later): nothing while a shard runs or once a step on the gate failed, `main` there
    once every shard passed them.  It reads drive's state and writes none; drive's next tick follows what `main` brought."""
    from research import queue as Q

    h = _gated(world, monkeypatch)
    h.d.after_landing_shards = 3
    h.d.records_artifact = lambda rid: f"art:{'a' * 64}@{'b' * 64}" if rid == "r-post-0" else None
    main = _main(world)
    tip = _landed(h, {"a.txt": "1\n"})
    h.run_state["r-post-0"] = ("done", 0, None)                 # circuit-check's records step
    h.tick()
    h.tick()
    shards = ["r-post-1", "r-post-2", "r-post-3"]
    assert h.postland()["run"]["ids"] == shards and _main(world) == main

    def promote(out: str) -> None:
        assert Q.promote(h.q, states=h.d.states, evidence=h.d.evidence) == 0
        assert out in capsys.readouterr().out

    def ended(k: int, passed: list[str], failed: str | None = None):
        h.run_state[shards[k]] = ("done", 1 if failed else 0, None)
        return _steps_run(world.tmp / f"{k}-{failed}", shards[k], {**dict.fromkeys(passed, "passed"),
                                                                    **({failed: "failed"} if failed else {})})

    state = h.q.state_path.read_bytes()
    rest = [s for s in h.posted[-1][1].split(",") if s not in GATED[:7]]          # the last shard's: ratchet-gate, circuit-check
    h.evidence = [ended(0, GATED[:4]), ended(1, GATED[4:7])]
    promote("r-post-3 still runs")
    h.evidence.append(ended(2, [s for s in rest if s != GATED[-1]], GATED[-1]))
    promote("it didn't pass every step on the merge gate")
    assert _main(world) == main
    h.evidence[2] = ended(2, rest)
    promote(f"`main` fast-forwarded from {main[:12]} to {tip[:12]}, whose after-landing run r-post-1 r-post-2 r-post-3 passed")
    assert _main(world) == tip and h.q.state_path.read_bytes() == state
    promote("already: nothing to move")
    h.tick()
    assert h.postland()["verified"] == tip and h.after == [(main, {tip: "r-post-1 r-post-2 r-post-3"})]


def test_a_main_pushed_outside_drive_is_merged_into_checked_and_main_follows_once_that_is_checked(world, monkeypatch):
    """The lander's `research merge --push` lands on `main` on its own check while `checked` holds landings `main` lacks: drive
    merges it into `checked`, whose next after-landing run checks the two together before `main` moves on."""
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


def _push_outside(world: World, files: dict[str, str], name: str) -> str:
    """A commit onto `checked` by ``name``, pushed to it by someone other than drive; its sha."""
    import os
    import subprocess

    c = world.change(files, base="origin/checked")
    env = {**os.environ, **{f"GIT_{who}_{k}": v for who in ("AUTHOR", "COMMITTER")
                            for k, v in (("NAME", name), ("EMAIL", f"{name}@example.invalid"))}}
    sha = subprocess.run(["git", "commit-tree", f"{c}^{{tree}}", "-p", f"{c}^", "-m", "an outside commit"], cwd=world.dev, env=env,
                         check=True, capture_output=True, text=True).stdout.strip()
    _git(world.dev, "push", "-q", "origin", f"{sha}:refs/heads/checked")
    return sha


def test_a_checked_drive_didnt_write_holds_landing_next_and_main_and_rings_ci_and_top_once_until_it_is_reset(world, monkeypatch):
    h = Review(world, monkeypatch, land_into="checked")
    main = _main(world)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    landed = _checked(world)
    assert h.q.load_state()["drive"]["checked_tips"][-1] == landed
    _pass(h)                                                    # #1's after-landing run passed: `main` would follow it
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.review(2)
    queued = _git(world.origin, "rev-parse", "next")
    outside = _push_outside(world, {"x.txt": "x\n"}, "mallory")
    launched = len(h.posted)
    for _ in range(2):
        why = h.tick()
        assert why.startswith(f"`checked` is at {outside[:12]} ('an outside commit', by mallory <mallory@example.invalid>), "
                              "which drive didn't write: drive lands nothing, syncs no `next` and moves no `main`"), why
        assert f"back at {landed[:12]}, the last tip drive wrote" in why and f"accept-checked {outside} --by ci" in why
        assert _checked(world) == outside and _main(world) == main and _git(world.origin, "rev-parse", "next") == queued
        assert len(h.posted) == launched
    (ring,) = [p for p in h.posts if outside[:12] in p]
    assert ring.startswith("@ci @top queue: `checked` is at"), ring
    _git(world.dev, "push", "-q", "--force", "origin", f"{landed}:refs/heads/checked")
    assert h.tick().startswith("landed #2"), world.logs
    assert _main(world) == landed and _git(world.origin, "rev-parse", "checked^1") == landed


def test_ci_accepting_a_checked_drive_didnt_write_lets_drive_go_on_from_it_and_no_one_else_can(world, monkeypatch):
    from research import queue as Q

    h = Review(world, monkeypatch, land_into="checked")
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    landed = _checked(world)
    outside = _push_outside(world, {"x.txt": "x\n"}, "mallory")
    assert h.tick().startswith(f"`checked` is at {outside[:12]}")
    with pytest.raises(QueueError, match="accepted only with --by ci"):
        Q.accept_checked(h.q, outside, "lane-1")
    with pytest.raises(QueueError, match=f"is not `checked`'s tip, {outside[:12]}"):
        Q.accept_checked(h.q, landed, "ci")
    assert Q.accept_checked(h.q, outside, "ci") == 0
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.review(2)
    assert h.tick().startswith("landed #2"), world.logs
    assert _git(world.origin, "rev-parse", "checked^1") == outside


def test_drives_own_landings_and_merges_of_main_are_tips_it_takes_and_a_refused_push_records_none(world, monkeypatch):
    """Each tip drive pushes to `checked` is recorded before its push (the revert's: the red tests above tick on after it); a
    push `checked` refuses leaves no record, and a state from before drive kept them takes the tip it finds."""
    h = Review(world, monkeypatch, land_into="checked")
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    landed = _checked(world)
    _move_main(world, {"c.txt": "moved\n"})
    h.tick()
    merged = _checked(world)
    assert merged != landed and h.q.load_state()["drive"]["checked_tips"][-2:] == [landed, merged]
    drive = h.q.load_state()["drive"]
    stale = _git(h.q.workdir, "commit-tree", f"{landed}^{{tree}}", "-p", f"{landed}^", "-m", "not a fast-forward")
    assert h.d._push_into(drive, stale).returncode and stale not in drive["checked_tips"] and _checked(world) == merged
    state = h.q.load_state()
    del state["drive"]["checked_tips"]
    h.q.save_state(state)
    h.tick()
    assert h.q.load_state()["drive"]["checked_tips"] == [merged]
    assert not any("didn't write" in x for x in world.logs), world.logs


def test_land_into_takes_checked_or_main_and_only_drive_pushes_checked(world, monkeypatch):
    assert DriveFlags.parse('land_into = "checked"\n').land_into == "checked" and DriveFlags().land_into is None
    # a config that doesn't set it leaves the queue to read drive.toml's at `main`, as with no config
    assert DriveFlags.parse('on = ["m"]\n').land_into is None
    with pytest.raises(QueueError, match="land_into"):
        DriveFlags.parse('land_into = "staging"\n')
    monkeypatch.delenv("RESEARCH_DRIVE", raising=False)
    with pytest.raises(QueueError, match="`checked` is moved only by drive's App"):
        world.queue(land_into="checked")._drive_only("checked", "once", False)
