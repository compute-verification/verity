"""Drive's alarms: whatever is past its bound rings its owner once, and again only every `ALARM_EVERY_S` while it stays past
it.  A step of drive's after-landing run past `SLOW_STEP_FACTOR` times its median rings its ci.toml job's owner; a member
of `next` marked ready `STUCK_S` ago and not landed rings the owner of what holds it; a red on `checked` rings the
culprit's lane, or with no culprit the owner of what failed, never ci and top beside it."""
from __future__ import annotations

import json
import time

import pytest

from research.drive import ALARM_EVERY_S, AFTER_LANDING, LEAN_STEP, PREFLIGHT, STUCK_S, running_steps
from test_drive_review import Review, _move_main, _owning
from test_queue import World, _git

#: a ci.toml giving each after-landing step a group and an owner: pytest alone in group 1, the Rust build and circuit-check
#: in group 2, the Lean steps in group 3 (lean-suites after lean-build), ratchet-gate in group 4
JOBS = {"pytest": ("suites", 1), "flock-circuit-build": ("proofs", 2), "circuit-check": ("circuits", 2),
        "flock-rust-tests": ("proofs", 2), "lean-changed": ("lean", 3), "lean-build": ("proofs", 3),
        "lean-unit-cut": ("proofs", 3), "lean-agreement": ("proofs", 3), "lean-audit": ("lean", 3),
        "lean-suites": ("ci", 5), "ratchet-gate": ("vllm", 4)}
CI_TOML = 'schema = "ci/v0"\n' + "".join(
    f'[[job]]\nname = "{s}"\nowner = "{JOBS[s][0]}"\ngroup = {JOBS[s][1]}\n'
    + ('gate = false\n' if s in ("circuit-check", "ratchet-gate") else "")
    + ('after = "lean-build"\n' if s == "lean-suites" else "") for s in AFTER_LANDING)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _utc(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _run_dir(root, rid: str, start: float, steps: list[dict]):
    """A check's run dir: its root span opened at ``start``, and ``steps`` (each with its `ended` time) in its steps.jsonl."""
    d = root / rid
    d.mkdir(parents=True, exist_ok=True)
    (d / "spans.jsonl").write_text(json.dumps({"trace_id": rid, "span_id": "root", "parent_span_id": None, "name": "check",
                                               "start_utc": _utc(start), "end_utc": None, "status": None}) + "\n")
    (d / "steps.jsonl").write_text("".join(json.dumps({**{k: v for k, v in s.items() if k != "ended"},
                                                       "ended_utc": _utc(s["ended"])}) + "\n" for s in steps))
    return d


def _preflights(start: float) -> list[dict]:
    """The preflights' records, 20 s each from ``start``."""
    return [{"step": s, "status": "passed", "seconds": 20.0, "ended": (t := start + 20 * (i + 1)), "ended_utc": _utc(t)}
            for i, s in enumerate(PREFLIGHT)]


def _rings(h, says: str) -> list[str]:
    return [p for p in h.posts if says in p]


def test_running_steps_follow_the_preflights_then_each_group_in_order():
    jobs = [{"name": n, "owner": o, "group": g, **({"after": "lean-build"} if n == "lean-suites" else {})}
            for n, (o, g) in JOBS.items()]
    steps = list(AFTER_LANDING)
    pre = _preflights(0.0)
    assert running_steps(pre[:1], steps, jobs, 0.0) == {PREFLIGHT[1]: 20.0}
    assert running_steps([], steps, jobs, 0.0) == {PREFLIGHT[0]: 0.0}
    got = running_steps([*pre, {"step": "pytest", "status": "passed", "ended_utc": _utc(500)},
                         {"step": "flock-circuit-build", "status": "failed", "ended_utc": _utc(300)},
                         {"step": "lean-changed", "status": "passed", "ended_utc": _utc(100)}], steps, jobs, 0.0)
    # group 1 is done, group 2 stopped at its failure, lean-suites waits for lean-build, which runs
    assert got == {"lean-build": 100.0, "ratchet-gate": 60.0}
    assert running_steps([pre[0], {**pre[1], "status": "failed"}], steps, jobs, 0.0) == {}


def _landed(world: World, monkeypatch, **kw) -> Review:
    """#1 landed on `checked` under `CI_TOML`, its after-landing run `r-post-0` running."""
    _move_main(world, {"tools/verity/check/ci.toml": CI_TOML})
    h = Review(world, monkeypatch, land_into="checked", **kw)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    assert h.postland()["run"]["ids"] == ["r-post-0"]
    return h


def test_a_step_past_one_and_a_half_times_its_median_rings_its_owner_once_an_interval(world, monkeypatch, tmp_path):
    h = _landed(world, monkeypatch)
    state = h.q.load_state()
    state["drive"]["step_s"] = {"postland pytest": [["r1", 600.0], ["r2", 600.0], ["r3", 700.0]]}
    h.q.save_state(state)
    start = h.clock
    h.evidence = [_run_dir(tmp_path, "r-post-0", start, _preflights(start))]
    h.clock = start + 60 + 890                                 # pytest has run 890 s, under 1.5 times its median of 600 s
    h.tick()
    assert not _rings(h, "has run"), h.posts
    h.clock += 20
    h.tick()
    (ring,) = _rings(h, "has run")
    assert ring.startswith("@suites queue: step pytest of the after-landing run of `checked`") and "(run r-post-0)" in ring, ring
    assert "past 1.5 times its median of 10 min over drive's last 3 uncached passes" in ring, ring
    h.clock += 60
    h.tick()
    assert len(_rings(h, "has run")) == 1                      # within the interval: no second ring
    h.clock += ALARM_EVERY_S
    h.tick()
    assert len(_rings(h, "has run")) == 2                      # still past it an interval later
    assert not [p for p in _rings(h, "has run") if not p.startswith("@suites ")], h.posts


def test_an_ended_runs_uncached_passes_are_its_steps_durations(world, monkeypatch, tmp_path):
    h = _landed(world, monkeypatch)
    start = h.clock
    h.evidence = [_run_dir(tmp_path, "r-post-0", start, [
        *_preflights(start), {"step": "pytest", "status": "passed", "seconds": 640.0, "ended": start + 700},
        {"step": "lean-build", "status": "passed", "seconds": 5.0, "cache": {"hit": "k"}, "ended": start + 70},
        {"step": "circuit-check", "status": "failed", "seconds": 90.0, "ended": start + 150}])]
    h.run_state["r-post-0"] = ("done", 0, None)
    h.tick()
    times = h.q.load_state()["drive"]["step_s"]
    assert times["postland pytest"] == [["r-post-0", 640.0]] and "postland lean-build" not in times, times
    assert "postland circuit-check" not in times and times[f"postland {PREFLIGHT[0]}"] == [["r-post-0", 20.0]], times


def test_a_ready_pr_unlanded_two_hours_for_its_land_verdict_rings_the_reviewer_once_an_interval(world, monkeypatch):
    marks: dict[int, str] = {}
    h = Review(world, monkeypatch, land_into="checked", marked_at=lambda n, head: marks.get(n))
    world.gh.put(1, head := world.change({"a.txt": "1\n"}))
    marks[1] = _utc(h.clock - STUCK_S + 60)
    h.tick()
    assert not _rings(h, "not landed"), h.posts
    h.clock += 60
    h.tick()
    (ring,) = _rings(h, "not landed")
    assert ring.startswith("@ci queue: marked ready 2 h ago or more and not landed on `checked`: #1 (2.0 h), held by #1 at "
                           f"{head[:8]}: no land verdict"), ring
    h.clock += 60
    h.tick()
    assert len(_rings(h, "not landed")) == 1
    h.clock += ALARM_EVERY_S
    h.tick()
    assert len(_rings(h, "not landed")) == 2


def test_a_held_member_below_rings_its_own_owner_for_the_ready_prs_above_it(world, monkeypatch):
    _owning(world, {"lane-1": [], "lane-2": [], "lane-3": []})
    marks: dict[int, str] = {}
    h = Review(world, monkeypatch, land_into="checked", reviewed_first=False, marked_at=lambda n, head: marks.get(n))
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    marks.update({1: _utc(h.clock - 3 * 3600), 2: _utc(h.clock - 3 * 3600), 3: _utc(h.clock - 600)})
    h.review(1, "hold", ref="needs a test")
    h.review(2)
    h.review(3)
    h.tick()
    (ring,) = _rings(h, "not landed")
    assert ring.startswith("@lane-1 queue: ") and "#1 (3.0 h), #2 (3.0 h), held by #1 at " in ring, ring
    assert "held by reviewer (needs a test)" in ring and "#3" not in ring, ring       # #3 was marked 10 min ago
    h.tick()
    assert len(_rings(h, "not landed")) == 1


def test_a_reviewed_pr_waiting_on_its_running_lean_check_rings_the_lean_audits_owner(world, monkeypatch):
    _move_main(world, {"tools/verity/check/ci.toml": CI_TOML})
    marks: dict[int, str] = {}
    h = Review(world, monkeypatch, land_into="checked", marked_at=lambda n, head: marks.get(n))
    lock = json.dumps({"guarantees": {"T": "2"}, "reads": {}, "dependencies": {"M": "1"}})
    world.gh.put(1, world.change({"pkg/lean-audit.json": lock}))
    marks[1] = _utc(h.clock - STUCK_S)
    h.review(1)
    h.tick()
    (rid,) = h.lean_ids()
    (ring,) = _rings(h, "not landed")
    assert ring.startswith(f"@{JOBS[LEAN_STEP][0]} queue: ") and f"held by the Lean check {rid} of " in ring, ring


def test_a_red_step_on_checked_with_no_culprit_rings_its_ci_toml_owner_alone_once(world, monkeypatch, tmp_path):
    h = _landed(world, monkeypatch)
    start = h.clock
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_run_dir(tmp_path, "r-post-0", start, [
        *_preflights(start), {"step": "flock-rust-tests", "status": "failed", "rc": 101, "ended": start + 200}])]
    h.tick()
    (ring,) = _rings(h, "is red on")
    assert ring.startswith("@proofs queue: step flock-rust-tests is red on `checked`"), ring
    assert "@ci" not in ring and "@top" not in ring, ring
    h.tick()
    assert len(_rings(h, "is red on")) == 1
    assert _git(world.origin, "rev-parse", "main") != _git(world.origin, "rev-parse", "checked")
