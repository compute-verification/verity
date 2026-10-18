"""Red team of `research queue drive`'s bisection (#2542): the ways it could eject an innocent PR, eject nobody when it should,
loop, or waste a check slot, each driven through whole ticks on `test_queue`'s throwaway origin."""
from __future__ import annotations

import itertools
import json
import subprocess

import pytest

from research.drive import CHECK_STEPS, NOT_STARTED, SIGTERM_RC
from test_merge_gate import CHECKED
from test_queue import World, _git
from test_queue_drive import Harness, _check, _failed_run, _tip

FAILURE = ("tests/test_x.py::test_y", "tests/test_x.py:2")      # suite pkg's test, which no PR changes
_seq = itertools.count()


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _record(world: World, monkeypatch, commit: str, ok: bool) -> None:
    _check(world, monkeypatch, commit, CHECKED if ok else ("false",))


def _has(world: World, commit: str, path: str) -> bool:
    return subprocess.run(["git", "-C", str(world.origin), "cat-file", "-e", f"{commit}:{path}"], capture_output=True).returncode == 0


def _main(world: World) -> str:
    return _git(world.origin, "rev-parse", "main")


def _run(h: Harness, world: World, monkeypatch, outcome, until, ticks: int = 24, seen: int = 0) -> None:
    """Tick until ``until()``, recording each launch from ``seen`` on as ``outcome(commit)`` says: True passes, False fails."""
    for _ in range(ticks):
        for c, *_ in h.launched[seen:]:
            _record(world, monkeypatch, c, outcome(c))
        seen = len(h.launched)
        h.tick()
        if until():
            return
    raise AssertionError(f"not settled in {ticks} ticks: {h.posts} {world.logs[-6:]}")


def _attempt_commit(h: Harness, rid: str) -> str | None:
    rows = h.d.q.store.index.sql("SELECT source_commit FROM attempts WHERE id = ?", (rid,))
    return rows[0][0] if rows else None


def _suites_run(tmp_path, *names: str):
    """A check's run dir whose suites ``names`` failed, each a test under `pkg` that no PR changes."""
    d = tmp_path / f"suites-{next(_seq)}"
    (d / "suites").mkdir(parents=True)
    (d / "steps.jsonl").write_text('{"step": "lean-suites", "status": "failed"}\n')
    (d / "suites-lean.json").write_text(json.dumps([{"name": n, "suite": "pkg", "status": "failed", "failed_tests": [FAILURE[0]],
                                                     "log": f"/w/runs/r/suites/{n}.log"} for n in names]))
    for n in names:
        (d / "suites" / f"{n}.log").write_text(f"E       AssertionError\n\n{FAILURE[1]}: AssertionError\n")
    return d


def _ejected(h: Harness) -> dict:
    return h.d.q.load_state().get("ejected", {})


# -- 1. `main` itself red

@pytest.mark.parametrize("touch", ["none", "bottom"])
def test_a_failure_main_shares_ejects_nobody_and_tells_ci_main_is_red(world, monkeypatch, tmp_path, touch):
    """18 Oct 17:00Z: a revert left benchmarks/stacked_share broken on `main`, and every train failed that suite whatever it held.
    Bisection narrows any failure the base shares to the lowest member, which is innocent.  With no lower prefix passed and no
    passing check of `main` itself, drive checks `main` before it ejects on bisection alone: `main` failing too is told as red."""
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 4, 2
    for n in (1, 2, 3, 4):
        world.gh.put(n, world.change({f"pkg/f{n}.py" if touch == "bottom" and n == 1 else f"src{n}/f.py": f"{n}\n"}))
    h.tick()
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    _run(h, world, monkeypatch, lambda c: False, until=lambda: bool(h.posts))
    assert len(h.posts) == 1 and "Ejected" not in h.posts[0] and "is red" in h.posts[0], h.posts
    assert not _ejected(h) and [m.pr for m in h.q.chain()] == [1, 2, 3, 4]
    assert _main(world) in [c for c, *_ in h.launched]                           # `main` was checked itself


def test_a_failure_main_does_not_share_still_ejects_the_bottom_pr(world, monkeypatch, tmp_path):
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 4, 2
    for n in (1, 2, 3, 4):
        world.gh.put(n, world.change({f"pkg/f{n}.py": f"{n}\n"}))
    h.tick()
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    _run(h, world, monkeypatch, lambda c: not _has(world, c, "pkg/f1.py"), until=lambda: bool(h.posts))
    assert len(h.posts) == 1 and "Ejected #1: it is the only PR above `main`" in h.posts[0], h.posts


def test_a_bisect_check_that_fails_as_its_tip_did_is_not_resent(world, monkeypatch, tmp_path):
    """A failure no change reaches is resent once at the tip; each bisect check of it is already a second run, and resending
    every failing midpoint too doubled the checks bisection takes."""
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 4, 2
    for n in (1, 2, 3, 4):
        world.gh.put(n, world.change({f"src{n}/f.py": f"{n}\n"}))
    h.tick()
    members, tip, main = h.q.chain(), _tip(world), _main(world)
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    _run(h, world, monkeypatch, lambda c: not _has(world, c, "src1/f.py"), until=lambda: bool(h.posts))
    assert "Ejected #1" in h.posts[0], h.posts
    assert [c for c, *_ in h.launched][:5] == [tip, tip, members[1].commit, members[0].commit, main]   # then the rebuilt train


# -- 2. a flake

def test_a_flaky_failure_that_passes_on_the_midpoint_recheck_is_checked_again_before_anyone_is_ejected(world, monkeypatch, tmp_path):
    """The tip failed once and its midpoint passed and landed: bisection alone names the PR above it, on one run's word."""
    h = Harness(world, monkeypatch)
    h.d.train_size = 2
    for n in (1, 2):
        world.gh.put(n, world.change({f"pkg/f{n}.py": f"{n}\n"}))
    h.tick()
    tip, flaked = _tip(world), []
    h.evidence = [_failed_run(tmp_path, *FAILURE)]

    def outcome(c):
        if c == tip and not flaked:
            flaked.append(c)
            return False
        return True

    _run(h, world, monkeypatch, outcome, until=lambda: bool(h.posts) or _main(world) == tip)
    assert not h.posts and _main(world) == tip and not _ejected(h), h.posts


def test_a_midpoint_failing_on_another_suite_than_its_tip_is_checked_again_before_it_ejects(world, monkeypatch, tmp_path):
    """The tip's failure is #3's (suite A), and its midpoint flaked on suite B: the midpoint's own failure is one run's word."""
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 3, 2
    for n in (1, 2, 3):
        world.gh.put(n, world.change({f"pkg/f{n}.py": f"{n}\n"}))
    h.tick()
    members, tip = h.q.chain(), _tip(world)
    flaked: list[str] = []

    def outcome(c):
        if c == members[1].commit and not flaked:
            flaked.append(c)
            return False
        return not _has(world, c, "pkg/f3.py")

    a, b = _suites_run(tmp_path, "A"), _suites_run(tmp_path, "B")
    h.d.evidence = lambda repo, ids: [b if _attempt_commit(h, ids[0]) == members[1].commit and len(flaked) == 1 else a]
    _run(h, world, monkeypatch, outcome, until=lambda: bool(h.posts))
    assert len(h.posts) == 1 and "Ejected #3" in h.posts[0], h.posts
    assert set(_ejected(h)) == {"3"}


# -- 3. a new head or an ejection while a bisect check runs

def test_a_pr_whose_new_head_came_while_the_check_that_blames_it_ran_is_not_ejected_at_that_head(world, monkeypatch, tmp_path):
    """Top, 17 Oct: a running check's members keep their place at the head it checks.  When that check then blames one, the
    ejection must not land on the PR's new head, the owner's push (often the fix): `eject` records the PR's current head."""
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 3, 2
    heads = {n: world.change({f"pkg/f{n}.py": f"{n}\n"}) for n in (1, 2, 3)}
    for n, sha in heads.items():
        world.gh.put(n, sha)
    h.tick()
    members, tip = h.q.chain(), _tip(world)
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    _record(world, monkeypatch, tip, False)
    h.tick()
    assert h.launched[-1][0] == members[0].commit                                # bisect: #1 alone
    _record(world, monkeypatch, members[0].commit, True)
    h.tick()
    assert _main(world) == members[0].commit and h.launched[-1][0] == members[1].commit
    fixed = world.change({"pkg/f2.py": "2 fixed\n"}, base=heads[2])
    world.gh.put(2, fixed)
    h.tick()
    assert [(m.pr, m.head) for m in h.q.chain()][0] == (2, heads[2])            # frozen: the bisect check covers it
    _record(world, monkeypatch, members[1].commit, False)
    h.tick()
    assert _ejected(h).get("2", {}).get("head") != fixed, h.posts
    h.tick()
    assert [(m.pr, m.head) for m in h.q.chain()][0] == (2, fixed)


@pytest.mark.parametrize("pr", [1, 4])
def test_an_ejection_by_hand_while_a_bisect_check_runs_cancels_that_check_only_when_it_leaves_next(world, monkeypatch, tmp_path, pr):
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 4, 2
    for n in (1, 2, 3, 4):
        world.gh.put(n, world.change({f"pkg/f{n}.py": f"{n}\n"}))
    h.tick()
    members, tip = h.q.chain(), _tip(world)
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    _record(world, monkeypatch, tip, False)
    h.tick()
    rec = h.d.q.load_state()["drive"]["bisect"][tip]
    assert rec["check"] == members[1].commit
    h.q.eject(pr, "a hand ejection", "lane")
    h.tick()
    assert (rec["ids"][0] in h.cancelled) == (pr == 1) and not h.posts
    assert not h.d.q.load_state()["drive"]["bisect"]                             # the failed tip left `next`
    assert [m.pr for m in h.q.chain()] == [n for n in (1, 2, 3, 4) if n != pr]


# -- 4. two failures in one train

def test_two_failures_in_one_train_at_two_members_eject_those_two_only(world, monkeypatch, tmp_path):
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 4, 2
    for n in (1, 2, 3, 4):
        world.gh.put(n, world.change({f"pkg/f{n}.py": f"{n}\n"}))
    h.tick()
    a, b, ab = _suites_run(tmp_path, "A"), _suites_run(tmp_path, "B"), _suites_run(tmp_path, "A", "B")

    def evidence(repo, ids):
        c = _attempt_commit(h, ids[0])
        return [{(True, False): a, (False, True): b, (True, True): ab}[(_has(world, c, "pkg/f2.py"), _has(world, c, "pkg/f4.py"))]]

    h.d.evidence = evidence
    _run(h, world, monkeypatch, lambda c: not (_has(world, c, "pkg/f2.py") or _has(world, c, "pkg/f4.py")),
         until=lambda: not h.q.chain() and _has(world, _main(world), "pkg/f3.py"), ticks=40)
    assert set(_ejected(h)) == {"2", "4"} and _has(world, _main(world), "pkg/f1.py"), h.posts


# -- 5. a bisect check that never ran

@pytest.mark.parametrize("state, rc", [("failed", NOT_STARTED), ("done", SIGTERM_RC)], ids=["twins-never-started", "sigterm"])
def test_a_bisect_check_whose_twins_never_started_or_that_was_cancelled_is_launched_again(world, monkeypatch, tmp_path, state, rc):
    h = Harness(world, monkeypatch)
    h.d.train_size = 2
    for n in (1, 2):
        world.gh.put(n, world.change({f"pkg/f{n}.py": f"{n}\n"}))
    h.tick()
    members, tip = h.q.chain(), _tip(world)
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    _record(world, monkeypatch, tip, False)
    h.tick()
    (rid,) = h.d.q.load_state()["drive"]["bisect"][tip]["ids"]
    h.run_state[rid] = (state, rc, None)
    h.tick()
    assert [c for c, *_ in h.launched] == [tip, members[0].commit, members[0].commit] and not h.posts
    assert h.d.q.load_state()["drive"]["bisect"][tip]["ids"] != [rid]


# -- 6. max_running full

def test_a_bisect_waiting_hours_on_a_full_max_running_reads_the_failure_once_a_tick_and_launches_when_a_slot_frees(world, monkeypatch,
                                                                                                                   tmp_path):
    """Every evidence read is a `research status` per run over ssh, twice with its logs: waiting, a tick read the failure twice,
    once before the sync and once for the trains."""
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 2, 2
    for n in (1, 2, 3, 4):
        world.gh.put(n, world.change({f"pkg/f{n}.py": f"{n}\n"}))
    h.tick()
    members = h.q.chain()
    low, tip = members[1].commit, members[3].commit
    assert [c for c, *_ in h.launched] == [low, tip]
    h.d.max_running = 1                                                          # the upper train holds the one slot
    _record(world, monkeypatch, low, False)
    reads: list[list[str]] = []
    run = _failed_run(tmp_path, *FAILURE)
    h.d.evidence = lambda repo, ids: reads.append(ids) or [run]
    for _ in range(4):
        h.clock += 3600
        reads.clear()
        h.tick()
        assert len(h.launched) == 2 and not h.posts and len(reads) == 1, reads
    _record(world, monkeypatch, tip, False)
    h.tick()
    assert h.launched[-1][0] == members[0].commit and not h.posts


# -- 7. `--defer`

def test_a_midpoint_that_passed_but_deferred_clears_nobody_of_a_failure_in_a_deferred_step(world, monkeypatch, tmp_path):
    """With circuit-check deferred, the tip failed circuit-check (#1's), and its midpoint #1 passed every other step and landed
    with circuit-check still running: bisection took `main` for good and ejected #2, and #1's landing was reverted after."""
    h = Harness(world, monkeypatch)
    h.d.defer, h.d.train_size = frozenset({"circuit-check", "flock-rust-tests"}), 2
    for n in (1, 2):
        world.gh.put(n, world.change({f"verity/ml/circuits/c{n}.py": f"{n}\n"}))
    h.tick()
    members, tip = h.q.chain(), _tip(world)
    (tip_rid,) = h.d.q.load_state()["drive"]["launched"][tip]["ids"]
    h.run_state[tip_rid] = ("running", None, None, "step circuit-check")
    cc = tmp_path / "cc"
    cc.mkdir()
    (cc / "steps.jsonl").write_text('{"step": "circuit-check", "status": "failed", "rc": 1, "log": "circuit-check.log"}\n')
    (cc / "circuit-check.log").write_text("circuit-check: c1: FAIL\n")
    ev: dict[str, object] = {tip_rid: cc}
    h.d.evidence = lambda repo, ids: [ev[i] for i in ids if i in ev]
    h.tick()
    (mid_rid,) = h.d.q.load_state()["drive"]["bisect"][tip]["ids"]
    assert h.launched[-1][0] == members[0].commit
    others = tmp_path / mid_rid                                                  # `_passed_but_deferred` names the run by its dir
    others.mkdir()
    (others / "steps.jsonl").write_text("".join(json.dumps({"step": x, "status": "passed"}) + "\n" for x in CHECK_STEPS
                                                if x not in h.d.defer))
    ev[mid_rid] = others
    for _ in range(2):
        h.tick()
    assert _main(world) == members[0].commit                                     # #1 landed, its circuit-check running on
    h.run_state[mid_rid] = ("done", 1, None)
    ev[mid_rid] = cc
    h.tick()
    assert any("reverted on `main`" in p for p in h.posts), h.posts
    assert not any("Ejected" in p for p in h.posts) and not _ejected(h), h.posts


def test_a_resend_with_pytest_deferred_runs_the_suites_its_change_touches_too(world, monkeypatch, tmp_path):
    """CI-57: with pytest deferred, a train lands once its touched suites pass; a resend launched its check alone, so a resent
    train landed with no suite run at all."""
    h = Harness(world, monkeypatch)
    h.d.defer, h.d.train_size = frozenset({"pytest"}), 2
    touched: list[str] = []

    def launch_touched(repo, commit, machine, base):
        touched.append(commit)
        h.run_state[rid := f"r-touched-{len(touched)}"] = ("running", None, None)
        return [rid], ""

    h.d.launch_touched = launch_touched
    for n in (1, 2):
        world.gh.put(n, world.change({f"src{n}/f.py": f"{n}\n"}))
    h.tick()
    tip = _tip(world)
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    _record(world, monkeypatch, tip, False)
    h.tick()
    assert [c for c, *_ in h.launched] == [tip, tip] and touched == [tip, tip]   # resent, with its touched suites


def test_a_resend_of_a_check_still_running_after_it_failed_cancels_that_run(world, monkeypatch, tmp_path):
    """A `--keep-going` check that has already failed is a failed verdict while it runs: resent, it went on holding its slot
    to its end, outside `max_running`'s count."""
    h = Harness(world, monkeypatch)
    h.d.train_size = 2
    for n in (1, 2):
        world.gh.put(n, world.change({f"src{n}/f.py": f"{n}\n"}))
    h.tick()
    tip, (rid,) = _tip(world), h.run_state
    h.run_state[rid] = ("running", None, None, "suite pkg (tests/test_x.py::test_y)")
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    h.tick()
    assert [c for c, *_ in h.launched] == [tip, tip] and h.cancelled == [rid]


# -- 8. `--first`

def test_first_while_a_bisect_check_runs_keeps_that_check_and_bisection_still_ejects_only_the_cause(world, monkeypatch, tmp_path):
    """`--first` below the failed train re-stacks it above the running bisect check.  Once that check lands, the failed train's
    commit is no longer a train's index, and no check of a commit above it may start: it holds the same failure."""
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 4, 2
    for n in (1, 2, 3, 4):
        world.gh.put(n, world.change({f"pkg/f{n}.py": f"{n}\n"}))
    h.tick()
    members, tip = h.q.chain(), _tip(world)
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    _record(world, monkeypatch, tip, False)
    h.tick()
    (mid_rid,) = h.d.q.load_state()["drive"]["bisect"][tip]["ids"]
    world.gh.put(9, world.change({"other/fix.txt": "fix\n"}))
    h.q.first = (9,)
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2, 9, 3, 4] and h.q.chain()[1].commit == members[1].commit
    assert mid_rid not in h.cancelled
    above = _tip(world)
    _run(h, world, monkeypatch, lambda c: not _has(world, c, "pkg/f3.py"), until=lambda: bool(h.posts), seen=1)
    assert len(h.posts) == 1 and "Ejected #3" in h.posts[0] and set(_ejected(h)) == {"3"}, h.posts
    assert above not in [c for c, *_ in h.launched]                              # #3's failed train holds it


def test_no_train_above_a_failed_member_is_checked_while_its_failure_is_settled(world, monkeypatch, tmp_path):
    """A landing below a failed train shifts the trains' indices: once [#1 #2] failed and #1 landed, [#2 #3] is a train's
    index, and a check of it holds #2's failure while #2's own is still being settled."""
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 2, 2
    for n in (1, 2, 3, 4):
        world.gh.put(n, world.change({f"pkg/f{n}.py": f"{n}\n"}))
    h.tick()
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    members = h.q.chain()
    _run(h, world, monkeypatch, lambda c: not _has(world, c, "pkg/f2.py"), until=lambda: bool(h.posts))
    assert "Ejected #2" in h.posts[0], h.posts
    holding = {c for c, *_ in h.launched if _has(world, c, "pkg/f2.py")}
    assert holding == {members[1].commit, members[3].commit}, [c[:12] for c in holding]
