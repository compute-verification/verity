"""`research queue drive`: one tick lands what passed, syncs `next`, and checks its tip, with no agent in between; what it refuses
goes to the lander once.  On `test_queue`'s throwaway origin, with the launch, the runs' states and the merge-thread post injected."""
from __future__ import annotations

import os
import pytest

from research import merge
from research.drive import PUBLISH_GRACE_S, Driver
from research.cli import main as research_main
from test_merge_gate import CHECKED
from test_queue import World, _git


class Harness:
    def __init__(self, world: World, monkeypatch, **kw):
        self.world, self.launched, self.posts, self.after, self.cancelled = world, [], [], [], []
        self.run_state: dict[str, tuple] = {}
        self.clock = 1_000_000.0
        monkeypatch.setattr(merge, "caught_up", lambda repo, remote, tool, since="", log=print, runs=None:
                            self.after.append((since, runs)))
        monkeypatch.setattr(merge, "_sweep_after", lambda *a: None)
        self.q = world.queue(**kw)
        self.d = Driver(self.q, on="m1", launch=self.launch, states=lambda repo, ids: {i: self.run_state[i] for i in ids},
                        post=lambda text, to=(): self.posts.append(text) or True, now=lambda: self.clock,
                        cancel=lambda repo, ids, reason: self.cancelled.extend(ids), evidence=lambda repo, ids: self.evidence)
        self.evidence: list = []

    def launch(self, repo, commit, machine, agreement):
        rid = f"r20261012-0000{len(self.launched):02d}-abcd"
        self.launched.append((commit, machine, agreement))
        self.run_state[rid] = ("running", None, None)
        return [rid], ""

    def tick(self, **kw) -> str:
        return self.d.tick(**kw)


def _check(world: World, monkeypatch, commit: str, command=CHECKED) -> None:
    """A recorded `check` attempt of `commit` that runs `command` (CHECKED passes)."""
    monkeypatch.setenv("RESEARCH_STORE", str(world.tmp / "store"))
    tree = world.tmp / f"check-{commit[:12]}"
    _git(world.dev, "fetch", "-q", "origin")
    _git(world.dev, "worktree", "add", "-q", "--detach", str(tree), commit)
    monkeypatch.chdir(tree)
    research_main(["run", "--runs-dir", str(world.tmp / "runs"), "--project", "t", "--no-sampler", "--tool", "mytools.decl_check:T",
                   "--cwd", "source", "--", *command])


def _tip(world: World) -> str:
    return _git(world.origin, "rev-parse", "refs/heads/next")


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_an_unchecked_tip_is_checked_once_with_agreement_only_for_a_guarantee_change(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    assert h.tick().startswith("launched")
    assert h.launched == [(_tip(world), "m1", False)]
    assert "running" in h.tick() and len(h.launched) == 1          # its launch is still running: no second check
    world.gh.put(2, world.change({"pkg/lean-audit.json": "{}\n"}))
    h.tick()
    assert h.launched[-1] == (_tip(world), "m1", True)             # a new tip, and it changes a lean-audit.json


def test_launches_wait_while_max_running_checks_of_next_run(world, monkeypatch):
    h = Harness(world, monkeypatch)
    h.d.max_running = 1
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    assert "waits" in h.tick() and len(h.launched) == 1


def test_a_passing_tip_fast_forwards_main_and_the_landing_post_names_its_check(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.tick()
    tip, before = _tip(world), _git(world.origin, "rev-parse", "main")
    _check(world, monkeypatch, tip)
    assert h.tick().startswith("landed #2")
    assert _git(world.origin, "rev-parse", "main") == tip           # exactly the checked commit
    (since, runs), = h.after
    assert since == before and set(runs.values()) == {next(iter(runs.values()))} and tip in runs
    lines = merge.landing_lines(h.q.workdir, before, tip, runs=runs)
    assert [ln.split(" ", 4)[-1] for ln in lines] == [f"#1; checked by {runs[tip]}", f"#2; checked by {runs[tip]}"]


def test_the_steps_after_a_landing_run_with_the_queues_credentials_and_git_never_prompts(world, monkeypatch):
    """12 Oct 08:49Z: `caught_up`'s `ls-remote` in the queue's clone, which has no credential helper, waited on a password."""
    h = Harness(world, monkeypatch)
    seen = []
    monkeypatch.setattr(merge, "caught_up", lambda *a, **kw: seen.append({k: os.environ.get(k) for k in
                                                                           ("GIT_TERMINAL_PROMPT", "GIT_CONFIG_COUNT")}))
    monkeypatch.delenv("GIT_TERMINAL_PROMPT", raising=False)
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    h.q.auth = ("-c", "credential.helper=!h get")
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    _check(world, monkeypatch, _tip(world))
    assert h.tick().startswith("landed #1")
    assert seen == [{"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_COUNT": "1"}]
    assert "GIT_TERMINAL_PROMPT" not in os.environ and "GIT_CONFIG_COUNT" not in os.environ


def test_a_tip_checked_before_main_moved_lands_as_a_merge_commit_when_they_share_no_file(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    tip = _tip(world)
    _check(world, monkeypatch, tip)
    other = world.change({"c.txt": "outside the queue\n"})
    _git(world.dev, "push", "-q", "origin", f"{other}:refs/heads/main")
    assert h.tick().startswith("landed #1")
    main = _git(world.origin, "rev-parse", "main")
    assert _git(world.origin, "rev-list", "--parents", "-n", "1", main).split()[1:] == [other, tip]
    body = _git(world.origin, "log", "-1", "--format=%B", main)
    assert body.startswith(f"Merge {tip[:12]} (") and "\nChecked: r" in body and f"Skipped-check: main " in body


def test_a_tip_checked_before_main_moved_over_one_of_its_files_is_rebuilt_and_checked_again(world, monkeypatch):
    h = Harness(world, monkeypatch)
    lines = [f"{i}\n" for i in range(8)]
    _git(world.dev, "push", "-q", "origin", f"{world.change({'m.txt': ''.join(lines)})}:refs/heads/main")
    world.gh.put(1, world.change({"m.txt": "".join(["first\n", *lines[1:]])}))
    h.tick()
    old = _tip(world)
    _check(world, monkeypatch, old)
    other = world.change({"m.txt": "".join([*lines[:-1], "last\n"])})
    _git(world.dev, "push", "-q", "origin", f"{other}:refs/heads/main")
    assert h.tick().startswith("launched")
    assert _git(world.origin, "rev-parse", "main") == other and h.launched[-1][0] == _tip(world) != old


def test_a_main_that_changed_check_tooling_is_never_landed_on_unchecked(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    old = _tip(world)
    _check(world, monkeypatch, old)
    other = world.change({"tools/verity/check/check.py": "# a new step\n"})
    _git(world.dev, "push", "-q", "origin", f"{other}:refs/heads/main")
    assert h.tick().startswith("launched")
    assert _git(world.origin, "rev-parse", "main") == other and h.launched[-1][0] == _tip(world) != old


def test_a_failed_check_and_a_launch_that_registers_nothing_are_told_to_the_lander_once(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    _check(world, monkeypatch, _tip(world), ("false",))
    h.tick()
    h.tick()
    assert len(h.posts) == 1 and "failed its check" in h.posts[0] and h.posts[0].startswith("@ci ")
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.d.launch = lambda *a: ([], "train: no parent store key")
    h.tick()
    h.tick()
    assert len(h.posts) == 2 and "registered no run: train: no parent store key" in h.posts[1]


def test_a_failure_is_told_even_when_the_same_ticks_rebuild_takes_its_commit_off_next(world, monkeypatch):
    """12 Oct 17:23Z: check 3d11 failed, a re-mark of a `--first` PR rebuilt `next` in the tick that would have seen it, and the
    failure was never told: the train's PRs went straight back into the next check."""
    h = Harness(world, monkeypatch, first=(2,))
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    _check(world, monkeypatch, _tip(world), ("false",))
    world.gh.put(2, world.change({"b.txt": "2\n"}))                 # `--first`: below #1, so #1's commit leaves `next`
    h.tick()
    assert len(h.posts) == 1 and "failed its check" in h.posts[0] and "(#1)" in h.posts[0]


def test_a_dry_run_tells_nobody_and_a_real_tick_after_it_still_does(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    _check(world, monkeypatch, _tip(world), ("false",))
    h.tick(dry_run=True)
    assert h.posts == [] and any("would tell" in x and "[dry run]" in x for x in world.logs)
    h.tick()
    assert len(h.posts) == 1 and "failed its check" in h.posts[0]


def test_a_check_that_has_already_failed_is_told_while_it_keeps_going_and_its_rebuild_cancels_it_not(world, monkeypatch):
    """12 Oct: cc39's repository suite failed at 20:25Z and drive told it at 20:55Z, when `--keep-going` let the check end.  What
    a running check has failed so far is a failure now; once the PR is ejected, the check runs on for the full picture."""
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.d.train_size = 2
    h.tick()
    (rid,) = h.run_state
    h.run_state[rid] = ("running", None, None, "suite repository (tests/test_repository.py::test_x)")
    h.tick()
    assert len(h.posts) == 1 and "failed its check" in h.posts[0] and "has already failed: suite repository" in h.posts[0]
    h.tick()
    assert len(h.posts) == 1 and h.cancelled == []
    h.q.eject(2, "breaks the repository suite", "ci")
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1] and h.cancelled == []
    assert any("keeps going for the full picture" in x for x in world.logs)


def _failed_run(tmp_path, test: str, trace: str, steps: str = '{"step": "lean-suites", "status": "failed"}\n'):
    """A check's run dir whose suite `pkg` failed ``test``, its log naming ``trace``."""
    import json

    d = tmp_path / "run"
    (d / "suites").mkdir(parents=True)
    (d / "steps.jsonl").write_text(steps)
    (d / "suites-lean.json").write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "failed", "failed_tests": [test],
                                                     "log": "/workspace/research/runs/r/suites/pkg.log"}]))
    (d / "suites" / "pkg.log").write_text(f"E       NameError: name '_once' is not defined\n\n{trace}: NameError\n")
    return d


def test_a_failure_only_one_trains_change_reaches_ejects_that_pr_and_tells_ci_and_its_owner(world, monkeypatch, tmp_path):
    """13 Oct: f476d3d63's check failed at 01:57Z on #1847's new call of a helper `main` had moved, and the notice went to the
    other Project's lander, who acts on none of it: the train sat until another lane read the log.  Only #1847 changed the
    failing test's file, so drive ejects it."""
    h = Harness(world, monkeypatch)
    h.d.train_size = 2
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    _once()\n"}))
    h.tick()
    _check(world, monkeypatch, _tip(world), ("false",))
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    assert len(h.posts) == 1 and h.posts[0].startswith("@ci ") and "Ejected #2" in h.posts[0], h.posts
    assert "pkg/tests/test_x.py" in h.posts[0] and "@old-circuits-and-proofs" not in h.posts[0]
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1] and len(h.posts) == 1


def test_a_failure_two_changes_reach_rings_ci_and_ejects_nothing(world, monkeypatch, tmp_path):
    h = Harness(world, monkeypatch)
    h.d.train_size = 2
    world.gh.put(1, world.change({"pkg/helpers.py": "X = 2\n"}))
    world.gh.put(2, world.change({"pkg/tests/test_x.py": "def test_y():\n    pass\n"}))
    h.tick()
    _check(world, monkeypatch, _tip(world), ("false",))
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", 'File "/workspace/research/src/abc/pkg/helpers.py", line 1')]
    h.tick()
    assert len(h.posts) == 1 and "Not ejected: 2 PRs (#1 #2) touch pkg/helpers.py, pkg/tests/test_x.py" in h.posts[0], h.posts
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2]


LINT_LOG = """..........F........                                                      [100%]
=================================== FAILURES ===================================
___________________________ test_markdown_size_caps ____________________________

>       assert not offenders, "maintained documents are small:\\n" + "\\n".join(offenders)
E       AssertionError: maintained documents are small:
E         tools/research/src/research/store/README.md: 48 KB > 48 KB, 17 bytes over; move detail to the notes store

tests/test_repository.py:180: AssertionError
=========================== short test summary info ============================
FAILED tests/test_repository.py::test_markdown_size_caps - AssertionError: ma...
1 failed, 18 passed in 17.49s
"""


def _lint_run(tmp_path, log: str = LINT_LOG):
    d = tmp_path / "run"
    d.mkdir(parents=True)
    (d / "steps.jsonl").write_text('{"step": "preflight-lock", "status": "passed"}\n'
                                   '{"step": "preflight-lints", "status": "failed", "rc": 1, "log": "preflight-lints.log"}\n')
    (d / "preflight-lints.log").write_text(log)
    return d


def test_a_lint_failure_only_one_trains_change_reaches_ejects_that_pr(world, monkeypatch, tmp_path):
    """13 Oct 07:35Z: 898eae8a1f74 failed `preflight-lints` because the store README one PR edited was 17 bytes over its cap,
    and drive rang ci: the offender is named in the lint's assertion, and drive read only the suites' tracebacks."""
    h = Harness(world, monkeypatch)
    h.d.train_size = 2
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"tools/research/src/research/store/README.md": "big\n"}))
    h.tick()
    _check(world, monkeypatch, _tip(world), ("false",))
    h.evidence = [_lint_run(tmp_path)]
    h.tick()
    assert len(h.posts) == 1 and "Ejected #2" in h.posts[0] and "store/README.md" in h.posts[0], h.posts
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1]


def test_a_failed_pytest_step_points_at_its_tests_and_the_paths_its_failures_name(tmp_path):
    from research.drive import implicated

    assert implicated(_lint_run(tmp_path)) == {"tests/test_repository.py", "tools/research/src/research/store/README.md"}
    assert implicated(_lint_run(tmp_path / "b", "error: lock is stale\n")) is None


def test_a_failure_outside_the_suites_or_without_a_failing_test_points_at_no_file(tmp_path):
    from research.drive import implicated, touches

    d = _failed_run(tmp_path, "tests/test_x.py::test_y", 'File "/x/site/pkg/conftest.py", line 3')
    assert implicated(d) >= {"pkg/tests/test_x.py", "/x/site/pkg/conftest.py"}
    assert touches({"conftest.py", "pkg/conftest.py"}, {"/x/site/pkg/conftest.py"}) == {"pkg/conftest.py"}
    (d / "steps.jsonl").write_text('{"step": "circuit-check", "status": "failed"}\n')
    assert implicated(d) is None


def test_a_guard_failed_suite_with_no_failing_test_points_at_its_outside_reads_and_their_readers(tmp_path):
    """Train 21b3d26fe1a6 (14 Oct 21:08Z): verity-lean-audit failed with 146/146 tests passing — the read guard alone — and
    drive said "the failure names no file", so attribution took an agent's archaeology. The suite's record names the files
    read outside the inputs and the test module that read each (`outside`, path -> module; a child's reads in
    `child_outside`), and those implicate the PR that added the reading test or changed the file read."""
    import json

    from research.drive import implicated, touches

    d = tmp_path / "run"
    (d / "suites").mkdir(parents=True)
    (d / "steps.jsonl").write_text('{"step": "lean-suites", "status": "failed"}\n')
    record = {"name": "verity-lean-audit", "suite": "tools/verity/lean", "status": "failed", "failed_tests": [],
              "outside": {"tools/verity/check/lean_slot.py": "tools/verity/lean/tests/test_lean_changed.py",
                          "tools/verity/check/slot.py": "*"},
              "child_outside": {"tools/verity/lean/tests/test_cache.py": ["elsewhere/data.txt"]}, "log": None}
    (d / "suites-lean.json").write_text(json.dumps([record]))
    got = implicated(d)
    assert got == {"tools/verity/check/lean_slot.py", "tools/verity/check/slot.py",
                   "tools/verity/lean/tests/test_lean_changed.py", "tools/verity/lean/tests/test_cache.py", "elsewhere/data.txt"}
    assert touches({"tools/verity/lean/tests/test_lean_changed.py", "a.txt"}, got) == {"tools/verity/lean/tests/test_lean_changed.py"}
    (d / "suites-lean.json").write_text(json.dumps([{**record, "outside": {}, "child_outside": {}}]))
    assert implicated(d) is None


def test_what_a_running_check_has_failed_is_read_from_its_steps_and_its_suites_failures(tmp_path):
    import json

    from research.drive import failing

    assert failing(tmp_path) is None
    (tmp_path / "steps.jsonl").write_text('{"step": "pytest", "status": "passed"}\n{"step": "circuit-check", "status": "failed"}\n')
    (tmp_path / "suites-lean.failed.json").write_text(json.dumps([{"name": "repository", "failed_tests": ["a", "b", "c"]}]))
    assert failing(tmp_path) == "step circuit-check; suite repository (a, b ...)"


def test_status_fetches_only_paths_under_the_run_dir_beside_its_records(tmp_path):
    from research import remote

    for bad in ("../x", "/etc/passwd", "a//b", ".fetched", "suites/../x", ""):
        with pytest.raises(remote.Refused, match="paths of files under the run dir"):
            remote.fetch(tmp_path, also=(bad,))


def test_each_launched_runs_state_is_read_once_a_tick(world, monkeypatch):
    """A tick asks of a running launch for `held`, before the sync, for the running count and for its train; each read is a
    `research status` over ssh (12 Oct: 4 of them were 25 s of a 52 s tick)."""
    h = Harness(world, monkeypatch)
    reads: list[str] = []
    states = h.d.states
    h.d.states = lambda repo, ids: reads.extend(ids) or states(repo, ids)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    assert "running" in h.tick()
    (rid,) = h.run_state
    assert reads == [rid]
    h.run_state[rid] = ("failed", 75, None)
    h.tick()                                        # the next tick reads it afresh, sees no twin started, and relaunches
    assert reads == [rid, rid] and len(h.launched) == 2


def test_a_launch_that_ended_without_an_attempt_is_lost_after_the_grace_and_one_whose_twins_never_started_is_relaunched(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    (rid,) = h.run_state
    h.run_state[rid] = ("failed", 75, None)
    h.tick()
    assert len(h.launched) == 2                                     # no twin took a slot: launched again
    rid = sorted(h.run_state)[-1]
    h.run_state[rid] = ("done", 0, None)
    assert "running" in h.tick() and not h.posts                    # its attempt may still be on its way to the store
    h.clock += PUBLISH_GRACE_S + 1
    h.tick()
    assert len(h.posts) == 1 and "no attempt reached the store" in h.posts[0]


def test_next_is_checked_in_stacked_trains_and_a_failed_train_holds_up_only_the_ones_above_it(world, monkeypatch):
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 2, 3
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt"), (4, "d.txt"), (5, "e.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    members = h.q.chain()
    assert [c for c, _, _ in h.launched] == [members[1].commit, members[3].commit, members[4].commit]
    _check(world, monkeypatch, members[1].commit)
    _check(world, monkeypatch, members[3].commit, ("false",))
    h.tick()
    assert _git(world.origin, "rev-parse", "main") == members[1].commit       # the train below the failure lands
    assert len(h.posts) == 1 and f"{members[3].commit[:12]} on `next` (#3 #4) failed" in h.posts[0]


def test_a_pr_that_makes_circuit_check_run_cold_goes_above_the_warm_ones_and_they_get_a_check_of_their_own(world, monkeypatch):
    """15 Oct: Lean-only PRs from proofs sat for hours in trains with a circuits change, whose check ran circuit-check cold."""
    h = Harness(world, monkeypatch)
    h.d.train_size = 6
    for n, f in ((1, "a.txt"), (2, "verity/ml/circuits/library.py"), (3, "Security/X.lean")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    members = h.q.chain()
    assert [m.pr for m in members] == [1, 3, 2]
    assert [c for c, _, _ in h.launched] == [members[1].commit, members[2].commit]


def test_forces_cold_reads_the_paths_that_move_most_circuit_check_keys():
    from research.queue import forces_cold

    assert forces_cold(["verity/ml/circuits/library.py"]) and forces_cold(["uv.lock"]) and forces_cold(["x/y.circuit-check.py"])
    assert forces_cold(["tools/verity/circuit_check/src/circuit_check/cache.py"])
    assert not forces_cold(["Security/Proofs/X.lean", "verity/ml/circuits/README.md", "verity/ml/circuits/tests/test_x.py"])
    assert not forces_cold([])


def test_shadow_launches_lands_and_tells_nothing(world, monkeypatch):
    h = Harness(world, monkeypatch, shadow=True)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    before = _git(world.origin, "rev-parse", "main")
    assert h.tick().startswith("would check")
    _check(world, monkeypatch, _git(world.origin, "rev-parse", "refs/heads/queue-shadow"), ("false",))
    h.tick()
    assert not h.launched and not h.posts and _git(world.origin, "rev-parse", "main") == before
    assert any("would tell @ci" in s for s in world.logs)


def test_train_sh_runs_with_the_queues_credentials_and_git_never_prompts(tmp_path, monkeypatch):
    """12 Oct, drive's first live tick: train.sh's `git fetch` in the queue's clone, which has no credential helper of its own,
    waited on a password at the tty, and drive with it."""
    from research.drive import train_launch

    (tmp_path / "tools" / "verity" / "check").mkdir(parents=True)
    (tmp_path / "tools" / "verity" / "check" / "train.sh").write_text(
        'echo "prompt=$GIT_TERMINAL_PROMPT n=$GIT_CONFIG_COUNT $GIT_CONFIG_KEY_1=$GIT_CONFIG_VALUE_1 $GIT_CONFIG_KEY_2=$GIT_CONFIG_VALUE_2"\n'
        'read -r line && echo "stdin: $line"\necho "RUN r20261012-060000-aaaa"\n')
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "safe.directory")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "*")
    ids, tail = train_launch(tmp_path, "c" * 40, "m1", False, auth=("-c", "credential.helper=", "-c", "credential.helper=!h get"))
    assert ids == ["r20261012-060000-aaaa"]
    assert "prompt=0 n=3 credential.helper= credential.helper=!h get" in tail and "stdin:" not in tail


def test_a_disjoint_move_of_main_keeps_next_while_its_check_runs_and_a_shared_one_rebuilds_it(world, monkeypatch):
    """12 Oct: the lander held a passed PR rather than land it beside drive's running check, since any move of `main` rebuilt
    `next` and wasted the check."""
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    old = _tip(world)
    other = world.change({"z.txt": "lander's\n"})
    _git(world.dev, "push", "-q", "origin", f"{other}:refs/heads/main")
    assert "stays on" in h.tick() and _tip(world) == old and len(h.launched) == 1
    _check(world, monkeypatch, old)
    assert h.tick().startswith("landed #1")
    assert _git(world.origin, "rev-parse", "main^1", "main^2").split() == [other, old]


def test_a_disjoint_landing_of_a_commit_a_member_is_stacked_on_keeps_next(world, monkeypatch):
    """12 Oct 10:53Z: #1847, on `next` and stacked on #1793, made #1793's head a second common ancestor of `main` and `next` once
    the e2e train's merge commit landed; `merge-base` picked it, the chain read empty, and a disjoint check was cancelled."""
    h = Harness(world, monkeypatch)
    h.d.max_running = 1                                             # drive's live setting: #2's train waits
    main0 = _git(world.origin, "rev-parse", "main")
    _git(world.dev, "push", "-q", "origin", f"{world.change({'z.txt': 'z\n'})}:refs/heads/main")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2030-01-01T00:00:00Z")          # newer than `main`'s: `merge-base` prefers it
    below = world.change({"x.txt": "landed by the lander\n"}, base=main0)     # on a `main` that has moved on since
    monkeypatch.delenv("GIT_COMMITTER_DATE")
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    world.gh.put(2, world.change({"y.txt": "stacked\n"}, base=below))
    h.tick()                                                        # #2 joins `next` above #1, whose check runs
    old = _tip(world)
    _git(world.dev, "fetch", "-q", "origin")
    _git(world.dev, "checkout", "-q", "--detach", "origin/main")
    _git(world.dev, "merge", "-q", "--no-ff", "-m", "Merge the e2e train", below)
    _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    assert "stays on" in h.tick() and _tip(world) == old and h.cancelled == []


def test_a_move_of_main_sharing_a_file_with_the_running_train_rebuilds_next(world, monkeypatch):
    h = Harness(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    old = _tip(world)
    other = world.change({"a.txt": "lander's\n"})
    _git(world.dev, "push", "-q", "origin", f"{other}:refs/heads/main")
    h.tick()
    assert _tip(world) != old
    assert h.cancelled == ["r20261012-000000-abcd"]               # the old tip's check, still waiting, is cancelled


def test_first_goes_above_the_train_a_running_check_covers_and_cancels_nothing(world, monkeypatch):
    """12 Oct 16:08Z: restarting drive with `--first 1869` restacked `next` under the running f6ee and cancelled it, 35 minutes
    of check thrown away.  A running check's members keep their place; `--first` starts the train after them."""
    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 2, 1
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    running = h.q.chain()[1].commit
    assert [c for c, _, _ in h.launched] == [running]
    world.gh.put(9, world.change({"fix.txt": "fix\n"}))
    h.q.first = (9,)
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2, 9, 3] and h.q.chain()[1].commit == running
    assert h.cancelled == []


def test_a_train_whose_check_ended_before_its_attempt_reached_the_store_keeps_its_place(world, monkeypatch):
    """12 Oct 19:47Z: 5897 passed, and the tick that ran in the seconds before its attempt reached the store took its train for
    ended, restacked `next` under `--first 1826` and dropped the pass."""
    import time as _time

    h = Harness(world, monkeypatch)
    h.d.train_size, h.d.max_running = 2, 1
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    h.tick()
    running = h.q.chain()[1].commit
    (rid,) = h.run_state
    h.run_state[rid] = ("done", 0, _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(h.clock)))
    world.gh.put(9, world.change({"fix.txt": "fix\n"}))
    h.q.first = (9,)
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2, 9, 3] and h.q.chain()[1].commit == running


def test_a_conflict_is_told_to_ci_and_the_lane_that_owns_the_pr(world, monkeypatch):
    """12 Oct: drive's ejection posts named only the lander's subteam, so an owner heard of a conflict through a relay.  The
    owner's handle comes from main's registry: the lane the body names, the agent id it carries, the ready mark's author, or
    the agent whose id ends in the branch's suffix."""
    import dataclasses
    import json

    agent = "bc-b8aaadaa-aefd-503e-9863-5eabb3a28c79"
    reg = {"handles": [{"handle": "circuits", "holder": f"the circuits coordinator ({agent})"},
                       {"handle": "infra", "holder": "infra (bc-17cc41f1-6227-5fab-bc64-3fa2f224558b)"}]}
    main = world.change({".agents/skills/using-slack/registry.json": json.dumps(reg)})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    h = Harness(world, monkeypatch)
    sent = []
    h.d.post = lambda text, to: sent.append((text, to)) or True
    world.gh.put(1, world.change({"a.txt": "one\n"}))
    world.gh.put(2, world.change({"a.txt": "two\n"}), body="Lane `catalog`.")
    world.gh.prs[2] = dataclasses.replace(world.gh.prs[2], branch="cursor/catalog-8c79")
    h.tick()
    assert [to for _, to in sent] == [("@ci", "@circuits")]
    assert sent[0][0].startswith("@ci @circuits queue: #2 at ")
    q = h.q
    for pr, want in ((dataclasses.replace(world.gh.prs[2], body=f"[agent]({agent})", branch=""), "circuits"),
                     (dataclasses.replace(world.gh.prs[2], body="Lane `infra`.", branch=""), "infra"),
                     (dataclasses.replace(world.gh.prs[2], branch="cursor/x-0000"), None)):
        assert q.handle(pr) == want


def test_a_notice_reaches_the_merge_thread_without_naming_its_sender_and_a_refused_post_is_retried(world, monkeypatch):
    """13 Oct: `research msg send` refuses a `--to` naming the sender, and drive posted as ci to @ci, so from 05:07Z no notice
    was posted. The sender is also selectable (`queue eject --by lean`), so its handle, not always @ci, must be dropped."""
    from research import drive

    argv = []
    monkeypatch.setattr(drive.subprocess, "run", lambda cmd, **kw: argv.append(cmd) or
                        drive.subprocess.CompletedProcess(cmd, 0))
    assert drive.merge_thread_post("x", ("@ci", "@memory-accounting"))
    assert argv[-1][argv[-1].index("--as") + 1] == "ci"
    assert argv[-1][argv[-1].index("--to"):argv[-1].index("--quiet")] == ["--to", "@memory-accounting"]
    assert drive.merge_thread_post("x", ("@ci",)) and "--to" not in argv[-1]
    assert drive.merge_thread_post("x", ("@ci", "@lean"), sender="lean")
    assert argv[-1][argv[-1].index("--as") + 1] == "lean"
    assert argv[-1][argv[-1].index("--to"):argv[-1].index("--quiet")] == ["--to", "@ci"]

    h = Harness(world, monkeypatch)
    logged = []
    h.q.log = logged.append
    ok = [False]
    h.d.post = lambda text, to: ok[0]
    state = {"told": []}
    h.d.refuse(state, "k", "#1 conflicts", "circuits")
    assert state["told"] == [] and "could not post" in logged[-1]
    ok[0] = True
    h.d.refuse(state, "k", "#1 conflicts", "circuits")
    assert state["told"] == ["k"] and logged[-1].startswith("research queue drive: told @ci @circuits")


@pytest.mark.parametrize("to, sender", [(("@ci",), "ci"), (("@ci", "@circuits"), "ci"), (("@ci", "@lean"), "lean")])
def test_every_notice_drive_sends_passes_msg_sends_own_refusals(monkeypatch, to, sender):
    """13 Oct 17:41Z: train b22a's failure, for @ci alone, started `@ci queue:`, and `msg send` refused it as a post that wakes
    only its author.  A stand-in for `msg send` had hidden that refusal, so here each post goes through `msg send` itself."""
    from research import drive, slack

    monkeypatch.setattr(slack, "acting_as", lambda ns, reg, name: slack.name_entry(ns, reg, name))

    def run(cmd, **kw):
        assert cmd[1:5] == ["-m", "research", "msg", "send"]
        return drive.subprocess.CompletedProcess(cmd, slack.main([*cmd[4:], "--dry-run"]))

    monkeypatch.setattr(drive.subprocess, "run", run)
    assert drive.merge_thread_post(f"{' '.join(to)} queue: b22a0c54d29e on `next` (#1 #2) failed its check", to, sender=sender)
