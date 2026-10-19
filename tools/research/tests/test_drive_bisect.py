"""`research queue drive` finds a failed train's cause by bisection over the checked prefixes of `next`: it is among the members
above the highest lower prefix whose check passed (or the base).  One member there is ejected; of several, a member whose change
touches a file the failure names is ejected at once, and otherwise drive checks the prefix halfway between.  Overlap with what
failed's declared inputs ejects no PR (18 Oct: #2478 and #2486 beside #2484; 01:55Z: #2348 and nine others)."""
from __future__ import annotations

import pytest

from test_merge_gate import CHECKED
from test_queue import World, _git
from test_queue_drive import Harness, _check, _failed_run, _lean_run, _tip

FAILURE = ("tests/test_x.py::test_y", "tests/test_x.py:2")      # suite pkg's test, which no PR below changes


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _bisect(h: Harness) -> dict:
    return h.d.q.load_state()["drive"].get("bisect", {})


@pytest.mark.parametrize("lands", [True, False])
def test_the_one_member_above_the_highest_passed_prefix_is_ejected(world, monkeypatch, tmp_path, lands):
    """18 Oct 01:55Z: 2671bd's test_dc failure fell back on the suite's inputs and ejected 13 PRs, #2348 among them from the
    train below it, whose check was 11 steps in and was cancelled.  The failure names neither change here: drive waits on the
    lower check, and once it passed (landed or not), the one PR above it is the cause, once its one failure is seen again."""
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 1, 2
    if not lands:
        monkeypatch.setattr(h.d, "land", lambda drive, dry_run: None)
    world.gh.put(1, world.change({"pkg/data.json": "1\n"}))
    world.gh.put(2, world.change({"pkg/other.py": "2\n"}))
    h.tick()
    low, tip = h.q.chain()[0].commit, _tip(world)
    assert [c for c, *_ in h.launched] == [low, tip]
    _, tip_run = sorted(h.run_state)
    _check(world, monkeypatch, tip, ("false",))
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    h.tick()
    assert not h.posts and len(h.launched) == 2 and h.cancelled == []              # the running lower check halves them
    assert any("halves them: waiting on it" in x for x in world.logs)
    _check(world, monkeypatch, low)
    h.tick()
    assert not h.posts and h.launched[-1][0] == tip and tip_run not in h.cancelled  # one run's word: the tip runs again
    (again,) = h.d.q.load_state()["drive"]["launched"][tip]["ids"]
    _check(world, monkeypatch, tip, ("false",))
    h.tick()
    assert len(h.posts) == 1 and "Ejected #2: it is the only PR above " in h.posts[0], h.posts
    assert (f"`main` {low[:12]}" if lands else f"{low[:12]}, whose check passed") in h.posts[0]
    assert (_git(world.origin, "rev-parse", "main") == low) == lands
    h.tick()
    assert [m.pr for m in h.q.chain()] == ([] if lands else [1])
    assert again in h.cancelled and tip_run not in h.cancelled                     # the failed tip's own, off next


def _suite_inputs(tmp_path):
    return _failed_run(tmp_path, *FAILURE)


def _lean_package(tmp_path):
    return _lean_run(tmp_path, "AUDIT /w/lean-audit-scratch-1-x/pkgA: FAIL\n  dependencies: the .olean files of the modules the "
                               "package imports from toolchain differ from the record\nAUDIT /w/lean-audit-scratch-1-x/pkgB: PASS\n")


@pytest.mark.parametrize("one, two, evidence, cause", [
    # 16 Oct: #2164's train failed in benchmarks/pouw's suite, whose inputs its own pyproject.toml changed
    ({"a.txt": "1\n"}, {"pkg/pyproject.toml": "[tool.verity.tests]\ninputs = ['data/x.json']\n"}, _suite_inputs, 2),
    # 18 Oct 03:30Z: lean-audit failed on a package's stale record and named no file
    ({"pkgA/A.lean": "theorem a : True := trivial\n"}, {"pkgB/B.lean": "theorem b : True := trivial\n"}, _lean_package, 1),
], ids=["suite-inputs", "lean-package"])
def test_a_failure_only_input_overlap_reaches_launches_a_midpoint_check_instead_of_ejecting(world, monkeypatch, tmp_path, one,
                                                                                           two, evidence, cause):
    h = Harness(world, monkeypatch)
    h.d.train_size = 2
    world.gh.put(1, world.change(one))
    world.gh.put(2, world.change(two))
    h.tick()
    members, tip = h.q.chain(), _tip(world)
    _check(world, monkeypatch, tip, ("false",))
    h.evidence = [evidence(tmp_path)]
    h.tick()
    assert not h.posts and [c for c, *_ in h.launched] == [tip, members[0].commit], h.posts
    assert _bisect(h)[tip]["check"] == members[0].commit and _bisect(h)[tip]["among"] == [1, 2]
    h.tick()
    assert len(h.launched) == 2 and not h.posts                                    # one bisect check at a time
    _check(world, monkeypatch, members[0].commit, CHECKED if cause == 2 else ("false",))
    h.tick()
    if cause == 2:                                                                 # #1 passed and landed: the tip runs again
        assert not h.posts and h.launched[-1][0] == tip
        _check(world, monkeypatch, tip, ("false",))
        h.tick()
    else:                                                                          # #1 alone failed: `main` itself is checked
        main = _git(world.origin, "rev-parse", "main")
        assert not h.posts and h.launched[-1][0] == main
        _check(world, monkeypatch, main)
        h.tick()
    n = 3
    assert len(h.posts) == 1 and f"Ejected #{cause}: it is the only PR above `main`" in h.posts[0], h.posts
    assert [c for c, *_ in h.launched[n:]] == [m.commit for m in h.q.chain()]        # then only the rebuilt train's check


@pytest.mark.parametrize("cause", [1, 2, 3, 4])
def test_bisection_finds_the_cause_of_a_four_member_train_in_two_checks_and_one_more_to_confirm_it(world, monkeypatch, tmp_path,
                                                                                                    cause):
    """Each check passes when its prefix is below the cause; every PR changes a file under the failed suite, none the failing
    test, so nothing but the checks' verdicts says which PR it is.  When no bisect check failed, the tip's one failure is seen
    again before #4 is ejected, and #1 is ejected once `main` itself passed."""
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 4, 2
    for n in (1, 2, 3, 4):
        world.gh.put(n, world.change({f"pkg/f{n}.py": f"{n}\n"}))
    base = _git(world.origin, "rev-parse", "main")
    h.tick()
    members = h.q.chain()
    prefix = {m.commit: k + 1 for k, m in enumerate(members)}               # the PRs a commit's check covers
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    seen = 0
    for _ in range(6):
        for c, *_ in h.launched[seen:]:
            _check(world, monkeypatch, c, CHECKED if prefix.get(c, 0) < cause else ("false",))
        seen = len(h.launched)
        h.tick()
        if h.posts:
            break
    assert len(h.posts) == 1 and f"Ejected #{cause}: it is the only PR above `main`" in h.posts[0], h.posts
    assert seen == (3 if cause in (2, 3) else 4)        # the train's check, two bisect checks, and the tip's again or `main`'s
    assert _git(world.origin, "rev-parse", "main") == (members[cause - 2].commit if cause > 1 else base)
    h.tick()
    assert [m.pr for m in h.q.chain()] == [n for n in (1, 2, 3, 4) if n > cause]


def test_a_direct_file_hit_still_ejects_at_once(world, monkeypatch, tmp_path):
    h = Harness(world, monkeypatch)
    h.d.train_size = 4
    for n in (1, 2, 4):
        world.gh.put(n, world.change({f"pkg/f{n}.py": f"{n}\n"}))
    world.gh.put(3, world.change({"pkg/tests/test_x.py": "def test_y():\n    _once()\n"}))
    h.tick()
    _check(world, monkeypatch, _tip(world), ("false",))
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    h.tick()
    assert len(h.posts) == 1 and "Ejected #3: its change touches pkg/tests/test_x.py" in h.posts[0], h.posts
    assert [c for c, *_ in h.launched[1:]] == [h.q.chain()[-1].commit] and not _bisect(h)      # the rebuilt train's check only


def test_a_failure_nothing_reaches_is_resent_once_and_bisected_when_it_fails_again(world, monkeypatch, tmp_path):
    from research.drive import PUBLISH_GRACE_S

    h = Harness(world, monkeypatch)
    h.d.train_size = 2
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.tick()
    members, tip = h.q.chain(), _tip(world)
    _check(world, monkeypatch, tip, ("false",))
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    h.tick()
    assert [c for c, *_ in h.launched] == [tip, tip] and not h.posts                # resent: maybe a machine's fault
    rid = h.d.q.load_state()["drive"]["launched"][tip]["ids"][0]
    h.run_state[rid] = ("done", 1, None)
    h.clock += PUBLISH_GRACE_S + 1
    h.tick()
    assert [c for c, *_ in h.launched] == [tip, tip, members[0].commit] and not h.posts


def test_a_bisect_check_counts_against_max_running(world, monkeypatch, tmp_path):
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 2, 2
    world.gh.put(1, world.change({"pkg/f1.py": "1\n"}))
    world.gh.put(2, world.change({"pkg/f2.py": "2\n"}))
    h.tick()
    members, tip = h.q.chain(), _tip(world)
    world.gh.put(3, world.change({"verity/ml/circuits/x.py": "3\n"}))              # cold: pre-warms in a check slot
    h.tick()
    (warm,) = [r["ids"] for r in h.d.q.load_state()["drive"]["prewarm"].values()]
    h.d.max_running = 1
    _check(world, monkeypatch, tip, ("false",))
    h.evidence = [_failed_run(tmp_path, *FAILURE)]
    h.tick()
    assert len(h.launched) == 2 and any("waits: 1 checks running" in x for x in world.logs)
    h.run_state[warm[0]] = ("done", 1, None)
    h.tick()
    assert h.launched[-1][0] == members[0].commit and not h.posts
