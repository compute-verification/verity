"""Red team of #2544, drive's post-landing run (CI-30): landings during a run, a `main` broken before `verified`, reverts that
conflict or were made by hand, a pass that skipped a deferred step, the old `drive["deferred"]` records, `--regenerate-locks`
landings, the gate, and a drive restarted mid-run."""
from __future__ import annotations

import pytest

from research import merge
from research.cli import main as research_main
from research.drive import CHECK_STEPS
from test_drive_postland import DEFER, Postland, _registry, _tree
from test_merge_gate import CHECKED
from test_queue import World, _git
from test_queue_drive import Harness, _check, _failed_run, _live_run, _steps_run, _tip

FAILING = {"pkg/tests/test_x.py": "def test_y():\n    assert False\n"}


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _passed(h: Postland, rid: str) -> None:
    """``rid``, a post-landing run, passed every deferred step."""
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(h.tmp, rid, {s: "passed" for s in DEFER})]


def _fails_in_pkg(h: Postland, rid: str, name: str = "run") -> None:
    """``rid`` failed `pkg`'s tests/test_x.py::test_y."""
    h.run_state[rid] = ("done", 1, None)
    h.evidence = [_failed_run(h.tmp / name, "tests/test_x.py::test_y", "tests/test_x.py:2")]


def _push_main(world: World, files: dict[str, str]) -> str:
    """A commit outside the queue pushed onto `main` (a lander's landing, or a hand fix)."""
    c = world.change(files)
    _git(world.dev, "push", "-q", "origin", f"{c}:refs/heads/main")
    return c


def _hand_revert(world: World, merge_commit: str) -> str:
    _git(world.dev, "fetch", "-q", "origin")
    _git(world.dev, "checkout", "-q", "--detach", "origin/main")
    _git(world.dev, "revert", "--no-edit", "-m", "1", merge_commit)
    c = _git(world.dev, "rev-parse", "HEAD")
    _git(world.dev, "push", "-q", "origin", f"{c}:refs/heads/main")
    return c


def _merge_of(world: World, pr: int, ref: str = "main") -> str:
    return _git(world.origin, "log", "--format=%H", f"--grep=^PR: #{pr}$", ref).split()[0]


def _reverts(world: World, since: str) -> int:
    return len(_git(world.origin, "log", "--format=%H", "--grep=^This reverts commit", f"{since}..main").split())


# -- 1. a landing while the post-landing run runs


def test_1_a_landing_during_a_run_waits_for_the_next_run_and_a_failure_reverts_only_the_runs_culprit(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    world.gh.put(1, world.change(FAILING))
    _, one = h.land_deferred()
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    _, two = h.land_deferred()                                           # lands while r-post-0 checks `one`
    assert h.postland()["pending"] == [one, two] and len(h.posted) == 1
    _fails_in_pkg(h, "r-post-0")
    h.tick()
    main = _git(world.origin, "rev-parse", "main")
    assert _git(world.origin, "rev-parse", f"{main}^") == two                        # #1 reverted on top of #2, #2 stays
    assert _git(world.origin, "show", f"{main}:b.txt") == "2"
    assert h.posted[-1][0] == main and h.postland()["pending"] == [one, two]           # `two` is still unverified
    _passed(h, "r-post-1")
    h.tick()
    assert h.postland()["verified"] == main and h.postland()["pending"] == []


# -- 2. `main` already broken before `verified`


def test_2_a_failure_no_landing_touches_reverts_nothing_and_tells_ci(world, monkeypatch, tmp_path):
    """18 Oct: a revert left benchmarks/stacked_share broken on `main`.  A post-landing run then fails on it, and no PR landed
    since `verified` touches it or its inputs: reverting them all leaves `main` as red, minus innocent work."""
    _registry(world, "lane-1")
    _push_main(world, FAILING)                                           # broken before the queue's landing
    h = Postland(world, monkeypatch, tmp_path)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    _, tip = h.land_deferred()
    _fails_in_pkg(h, "r-post-0")
    h.tick()
    assert _git(world.origin, "rev-parse", "main") == tip, "an innocent landing was reverted"
    (post,) = h.posts
    assert post.startswith("@ci queue:") and "Reverted nothing" in post, post
    h.tick()
    assert len(h.posted) == 1                                            # that tip isn't run again until `main` moves


def test_4b_a_hand_landing_that_breaks_main_reverts_no_queue_landing(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    _, one = h.land_deferred()
    _passed(h, "r-post-0")
    h.tick()
    _push_main(world, FAILING)                                           # a lander's landing, between runs
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    _, two = h.land_deferred()
    _fails_in_pkg(h, "r-post-1")
    h.tick()
    assert _git(world.origin, "rev-parse", "main") == two, "#2, innocent, was reverted"
    assert any("Reverted nothing" in p for p in h.posts), h.posts


# -- 3. the culprit's revert conflicts


def test_3_a_culprits_revert_that_conflicts_leaves_main_and_tells_ci_and_its_owner_to_revert_by_hand(world, monkeypatch, tmp_path):
    _registry(world, "lane-1")
    h = Postland(world, monkeypatch, tmp_path)
    world.gh.put(1, world.change(FAILING))
    _, one = h.land_deferred()
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    assert False  # 2\n"}))
    _, two = h.land_deferred()
    _fails_in_pkg(h, "r-post-0")
    h.tick()
    assert _git(world.origin, "rev-parse", "main") == two
    (post,) = h.posts
    assert post.startswith("@ci @lane-1 queue: #1 must be reverted by hand"), post
    assert h.postland()["failed"] == one and h.posted[-1][0] == two                  # `two` is unverified: it runs
    _fails_in_pkg(h, "r-post-1", "run2")                                             # both touch it: both go, newest first
    h.tick()
    assert _tree(world, "main") == _tree(world, f"{_merge_of(world, 1)}^1")


# -- 4. a hand revert or hand landing between runs


def test_4a_a_culprit_reverted_by_hand_while_its_run_ran_is_not_reverted_again_nor_told_as_a_hand_revert(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    world.gh.put(1, world.change(FAILING))
    _, one = h.land_deferred()
    fix = _hand_revert(world, _merge_of(world, 1))
    _fails_in_pkg(h, "r-post-0")
    h.tick()
    assert _git(world.origin, "rev-parse", "main") == fix and _reverts(world, one) == 1
    assert not any("by hand" in p for p in h.posts), h.posts
    assert h.posted[-1][0] == fix                                                    # the fixed `main` gets its run


def test_4c_a_hand_landing_between_runs_alone_launches_no_run_and_the_next_landing_covers_it(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.land_deferred()
    _passed(h, "r-post-0")
    h.tick()
    _push_main(world, {"c.txt": "lander\n"})
    h.tick()
    assert len(h.posted) == 1                                            # its full check passed: nothing to run
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    _, two = h.land_deferred()
    assert h.posted[-1][0] == two


# -- 5. a pass that lacks a deferred step


def test_5_a_post_landing_run_that_passed_without_running_a_deferred_step_verifies_nothing(world, monkeypatch, tmp_path):
    """`--deferred` drops a step `deferred_parts` maps to no part, and a step whose command is `skipped:` passes the run: rc 0
    is not the deferred steps' verdict."""
    h = Postland(world, monkeypatch, tmp_path)
    base = _git(world.origin, "rev-parse", "main")
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    _, tip = h.land_deferred()
    h.run_state["r-post-0"] = ("done", 0, None)
    h.evidence = [_steps_run(tmp_path, "r-post-0", {"flock-circuit-build": "passed", "circuit-check": "skipped"})]
    h.tick()
    pl = h.postland()
    assert pl["verified"] == base and pl["pending"] == [tip], "a run that skipped circuit-check verified `main`"
    (post,) = h.posts
    assert "circuit-check" in post and "flock-rust-tests" in post and "verified nothing" in post, post
    h.tick()
    assert len(h.posted) == 1 and _git(world.origin, "rev-parse", "main") == tip     # reverts nothing, loops on nothing


# -- 6. the old `drive["deferred"]` records beside `postland`


def test_6_a_landing_the_old_rule_still_follows_is_not_taken_as_verified_by_the_post_landing_run(world, monkeypatch, tmp_path):
    """The live drive switches over with a `deferred` record of #1 open.  #2 then lands under `postland`, whose `verified` must
    not be #1's tip: a failure of #1's then reverts #1 (the PR it implicates), not #2."""
    h = Postland(world, monkeypatch, tmp_path)
    base = _git(world.origin, "rev-parse", "main")
    world.gh.put(1, world.change(FAILING))
    h.tick()
    one = _tip(world)
    _check(world, monkeypatch, one)
    h.d.defer = frozenset()                                              # the old drive: a full check, a `deferred` record
    assert h.tick().startswith("landed #1")
    state = h.q.load_state()
    state["drive"]["deferred"] = {one: {"ids": ["r-old"], "base": base, "prs": "#1"}}
    h.q.save_state(state)
    h.run_state["r-old"] = ("running", None, None)
    h.d.defer = DEFER
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    _, two = h.land_deferred()
    assert h.postland()["verified"] == base, h.postland()
    _fails_in_pkg(h, "r-post-0")
    h.tick()
    main = _git(world.origin, "rev-parse", "main")
    assert _git(world.origin, "show", f"{main}:b.txt") == "2" and _reverts(world, two) == 1
    assert "pkg/tests/test_x.py" not in _git(world.origin, "ls-tree", "-r", "--name-only", main)
    h.run_state["r-old"] = ("done", 1, None)                             # the old rule's check of #1 fails too
    h.evidence = [_steps_run(tmp_path, "r-old", {"circuit-check": "failed"})]
    h.tick()
    h.tick()
    assert _reverts(world, two) == 1 and not h.q.load_state()["drive"]["deferred"]   # #1 isn't reverted twice
    assert not any("by hand" in p for p in h.posts), h.posts


# -- 7. `--regenerate-locks` landings


def _land_with_locks(h: Postland, monkeypatch, tmp_path, pkg: str = "pkg") -> str:
    """Check `next`'s tip and land it deferred, its check having regenerated ``pkg``'s lock: `main`, the lock commit on top."""
    h.d.regenerate_locks = True
    before = set(h.run_state)
    h.tick()
    (rid,) = [r for r in h.run_state if r not in before and not r.startswith("r-post-")]
    tip = _tip(h.world)
    _, h.d.fetch, _ = _live_run(tmp_path, rid, tip, {pkg: '{"records": 2}\n'})
    h.evidence = [_steps_run(tmp_path, rid, {x: "passed" for x in CHECK_STEPS if x not in DEFER})]
    assert h.tick().startswith("landed"), h.world.logs
    main = _git(h.world.origin, "rev-parse", "main")
    assert _git(h.world.origin, "rev-parse", f"{main}^") == tip
    assert _git(h.world.origin, "log", "-1", "--format=%s", main).startswith("Lock records regenerated for #")
    return main


def test_7a_a_culprit_landed_with_regenerated_locks_is_reverted_with_its_lock_commit(world, monkeypatch, tmp_path):
    """The `Lock records regenerated for #N` commit on top rewrote the lean-audit.json #1 changed: `git revert -m 1` of #1's merge
    alone conflicts on it, and `main` stays red."""
    h = Postland(world, monkeypatch, tmp_path)
    base = _git(world.origin, "rev-parse", "main")
    world.gh.put(1, world.change({**FAILING, "pkg/A.lean": "theorem a : True := trivial\n", "pkg/lean-audit.json": '{"records": 1}\n'}))
    _land_with_locks(h, monkeypatch, tmp_path)
    _fails_in_pkg(h, "r-post-0")
    h.tick()
    assert _tree(world, "main") == _tree(world, base), h.posts
    assert any(p.startswith("@ci queue: #1 reverted on `main`") for p in h.posts), h.posts


def test_7b_reverting_every_landing_reverts_their_lock_commits_too(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    h.d.train_size = 2
    base = _git(world.origin, "rev-parse", "main")
    world.gh.put(1, world.change({"pkg/helpers.py": "X = 2\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}))
    _land_with_locks(h, monkeypatch, tmp_path)
    h.run_state["r-post-0"] = ("done", 1, None)                                      # both PRs are implicated
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", 'File "/workspace/research/src/abc/pkg/helpers.py", line 1')]
    h.tick()
    assert _tree(world, "main") == _tree(world, base), "a lock commit of a reverted landing stayed on `main`"


def test_7c_a_culprit_that_touches_no_regenerated_package_leaves_its_trains_lock_commit(world, monkeypatch, tmp_path):
    """The lock commit holds #1's records, which reverting #2 doesn't change."""
    h = Postland(world, monkeypatch, tmp_path)
    h.d.train_size = 2
    world.gh.put(1, world.change({"lean/A.lean": "theorem a : True := trivial\n"}))
    world.gh.put(2, world.change(FAILING))
    locked = _land_with_locks(h, monkeypatch, tmp_path, "lean")
    _fails_in_pkg(h, "r-post-0")
    h.tick()
    main = _git(world.origin, "rev-parse", "main")
    assert _git(world.origin, "rev-parse", f"{main}^") == locked and _reverts(world, locked) == 1
    assert _git(world.origin, "show", f"{main}:lean/lean-audit.json") == '{"records": 2}'
    assert "pkg/tests/test_x.py" not in _git(world.origin, "ls-tree", "-r", "--name-only", main)


# -- 8. the post-landing run is never a check


def test_8_a_post_landing_attempt_of_mains_tip_never_passes_the_check_gate(world, monkeypatch, tmp_path):
    """`check.py --record --deferred` records no `--tool check`, a `check-deferred/v1` result and a `--deferred` argv: each alone
    keeps `research merge` (and drive's `verdict`, which reads the same `passing`) from taking it, even forged as a `check/v1`."""
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    tip = _tip(world)
    monkeypatch.setenv("RESEARCH_STORE", str(world.tmp / "store"))
    tree = world.tmp / "deferred-run"
    _git(world.dev, "fetch", "-q", "origin")
    _git(world.dev, "worktree", "add", "-q", "--detach", str(tree), tip)
    monkeypatch.chdir(tree)
    for tool in ([], ["--tool", "mytools.decl_check:T"]):               # as recorded, and as if --tool check were added
        research_main(["run", "--runs-dir", str(world.tmp / "runs"), "--project", "t", "--no-sampler", *tool, "--cwd", "source",
                       "--", *CHECKED, "--keep-going", "--deferred", "circuit-check,flock-rust-tests"])
    kinds, command = merge.produced(h.q.workdir, "check"), merge.declared_command(h.q.workdir, "check")
    assert merge.passing(world.store, tip, "check", kinds=kinds, command=command, repo=h.q.workdir) == []
    assert merge.invocation([*CHECKED, "--deferred", "circuit-check"], command)[0] == "refused"
    state = h.q.load_state()
    assert h.d.verdict(tip, _git(world.origin, "rev-parse", "main"), state["drive"]).state != "passed"


# -- 9. a drive restarted mid-run


def test_9a_a_drive_restarted_while_its_post_landing_run_runs_reads_it_back_and_verifies_its_tip(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    _, tip = h.land_deferred()
    again = Postland(world, monkeypatch, tmp_path)                      # a new process: only state.json carries over
    again.run_state, again.posted = h.run_state, h.posted
    _passed(again, "r-post-0")
    again.tick()
    assert again.postland()["verified"] == tip and again.postland()["pending"] == [] and len(h.posted) == 1


def test_9b_a_drive_killed_after_a_deferred_landing_pushed_still_runs_that_landing_after_restart(world, monkeypatch, tmp_path):
    """The landing is pushed, then the tick dies before it saves state.json (a restart, a VM suspended, the sweep raising):
    the landing must still await its post-landing run, or the next landing's `verified` takes it in unchecked."""
    h = Postland(world, monkeypatch, tmp_path)
    base = _git(world.origin, "rev-parse", "main")
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    (rid,) = h.run_state
    h.evidence = [_steps_run(tmp_path, rid, {x: "passed" for x in CHECK_STEPS if x not in DEFER})]

    def killed(*a, **k):
        raise KeyboardInterrupt("the drive is killed")

    monkeypatch.setattr(merge, "_sweep_after", killed)
    with pytest.raises(KeyboardInterrupt):
        h.tick()
    tip = _git(world.origin, "rev-parse", "main")
    assert tip != base
    monkeypatch.setattr(merge, "_sweep_after", lambda *a: None)
    again = Postland(world, monkeypatch, tmp_path)
    again.run_state, again.evidence = h.run_state, h.evidence
    again.tick()
    pl = again.postland()
    assert pl and pl["verified"] == base and tip in pl["pending"], pl
    assert again.posted and again.posted[-1][0] == tip


def test_9c_a_drive_killed_after_a_revert_pushed_does_not_revert_again_on_restart(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    world.gh.put(1, world.change(FAILING))
    _, one = h.land_deferred()
    _fails_in_pkg(h, "r-post-0")

    def killed(text, to=()):
        raise KeyboardInterrupt("the drive is killed")

    h.d.post = killed
    with pytest.raises(KeyboardInterrupt):
        h.tick()
    assert _reverts(world, one) == 1
    again = Postland(world, monkeypatch, tmp_path)
    again.run_state, again.evidence, again.posted = h.run_state, h.evidence, h.posted
    again.tick()
    assert _reverts(world, one) == 1 and not any("by hand" in p for p in again.posts), again.posts


# -- races: a push the remote refuses


def test_r_a_revert_whose_push_loses_a_race_with_a_landing_is_retried_not_told_as_a_hand_revert(world, monkeypatch, tmp_path):
    h = Postland(world, monkeypatch, tmp_path)
    world.gh.put(1, world.change(FAILING))
    _, one = h.land_deferred()
    _fails_in_pkg(h, "r-post-0")
    real = h.q._remote
    raced = []

    def remote(repo, *args, **kw):
        if args[:1] == ("push",) and not raced:
            raced.append(_push_main(world, {"c.txt": "landed in between\n"}))
        return real(repo, *args, **kw)

    monkeypatch.setattr(h.q, "_remote", remote)
    h.tick()
    assert not any("by hand" in p for p in h.posts), h.posts
    h.tick()
    main = _git(world.origin, "rev-parse", "main")
    assert _git(world.origin, "rev-parse", f"{main}^") == raced[0] and _reverts(world, one) == 1
    assert _git(world.origin, "show", f"{main}:c.txt") == "landed in between"
