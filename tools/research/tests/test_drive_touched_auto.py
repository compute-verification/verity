"""A train's touched-suites run goes to `research run --on auto` (vy-nebius-2's agent-test queue), outside the check slots: on 18 Oct,
in a priority check slot, those runs waited 21-93 min for a 560 GB slot and ran 27-48 min, competing with the trains for six slots.
And `queue.touched_suites` is the one way both the preflight and drive run a merged tree's touched suites (CI-73).  The launch's
subprocess is faked; the worktrees are real."""
from __future__ import annotations

import itertools
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from research import queue as Q
from research.drive import touched_launch
from test_queue import _git, _move_main
from test_queue_drive import Harness, World, _tip

RID, RID2 = "r20261018-192500-ab12", "r20261018-193000-cd34"
#: what `research run --on auto` prints of a run it queued and launched (cli.py's `_run_queued` and `_run_remote`)
LAUNCHED = (f"research run --on auto: queued onto vy-nebius-2 as agent-test, from /tmp/x\n"
            f"research: the queue places run {RID} (agent-test, cpu phase) on vy-nebius-2 (share verity)\n"
            f"research: run {RID} launched on vy-nebius-2 (vy-nebius-2) pid=4242 -> /workspace/research/runs/{RID}\n")


class FakeLaunch:
    """`research run` as the launch sees it: records each argv, its cwd's state at launch and its environment, and prints ``said``."""

    def __init__(self, said: str = LAUNCHED, rc: int = 0):
        self.said, self.rc, self.calls = said, rc, []

    def __call__(self, argv, cwd, env):
        self.calls.append({"argv": argv, "cwd": Path(cwd), "env": env, "head": _git(Path(cwd), "rev-parse", "HEAD"),
                           "dirty": _git(Path(cwd), "status", "--porcelain")})
        return subprocess.CompletedProcess(argv, self.rc, "", self.said)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    return World(tmp_path)


def _no_worktree_left(world: World, repo: Path) -> bool:
    return not any((world.tmp / "tmp").iterdir()) and len(_git(repo, "worktree", "list").splitlines()) == 1


def _job(argv: list[str]) -> list[str]:
    """The `sh -c` job's words: `exec python3 .../quick_tier.py ARGS`."""
    assert argv[-3:-1] == ["sh", "-c"], argv
    return shlex.split(argv[-1])


def test_the_run_is_research_run_on_auto_from_a_worktree_of_exactly_the_commit_which_is_removed_after(world):
    base = _git(world.dev, "rev-parse", "origin/main")
    commit = world.change({"pkg/tests/test_x.py": "def test_x():\n    pass\n"})
    launch = FakeLaunch()
    t = Q.touched_suites(world.dev, commit, base, Q.RemoteSuites(launch=launch, env={"RESEARCH_LANE": "ci"}))
    assert t.ids == [RID] and RID in t.tail and not t.notes
    (call,) = launch.calls
    argv, wt = call["argv"], call["cwd"]
    assert argv[:4] == [sys.executable, "-m", "research", "run"]
    head = argv[: argv.index("--")]
    assert head[4:] == ["--on", "auto", "--source", str(wt), "--cwd", "source", "--cpus", "16", "--mem-gb", "160",
                        "--max-min", "120", "--send", str(Path(Q.quick_tier.__file__)), "--require-result",
                        "--declared-output", "out/*"]
    assert call["head"] == commit and call["dirty"] == ""                     # research run refuses a dirty tree
    assert call["env"] == {"RESEARCH_LANE": "ci"}
    job = _job(argv)
    assert job[:3] == ["exec", "python3", "$RESEARCH_RUN_DIR/inputs/quick_tier.py"]
    assert job[3:8] == ["--head", commit, "--base", base, "--"]
    assert job[8:] == ["uv", "run", "--locked", "--all-packages", "--extra", "torch-cpu", "python", "tools/verity/check/suites.py",
                       "--jobs", "16"]                       # the job's quota, not the share's affinity suites.py would read
    assert not wt.exists() and _no_worktree_left(world, world.dev)


def test_local_is_passed_only_to_a_suites_py_that_takes_it_and_base_is_resolved_to_a_full_sha(world):
    """`--local` (#2477) keeps suites.py on the node from sending itself to a node again; an older suites.py refuses the flag."""
    _move_main(world, {"tools/verity/check/suites.py": 'ap.add_argument("--changed")\n'})
    old = world.change({"a.txt": "old\n"})
    _move_main(world, {"tools/verity/check/suites.py": 'ap.add_argument("--changed")\nap.add_argument("--local")\n'})
    new = world.change({"a.txt": "new\n"})
    launch = FakeLaunch()
    for commit in (old, new):
        Q.touched_suites(world.dev, commit, f"{commit}~1", Q.RemoteSuites(launch=launch))
    (o, n) = (_job(c["argv"]) for c in launch.calls)
    assert o[-3:] == ["tools/verity/check/suites.py", "--jobs", "16"] and n[-4:] == ["tools/verity/check/suites.py", "--jobs", "16", "--local"]
    assert n[n.index("--base") + 1] == _git(world.dev, "rev-parse", f"{new}~1")
    assert _no_worktree_left(world, world.dev)


@pytest.mark.parametrize("rc, said", [(3, "research run --on auto: refused: the queue refuses the job: share expired\n"),
                                      (0, "research: something else\n")])
def test_a_launch_that_prints_no_run_id_registers_nothing_and_says_why(world, rc, said):
    commit = world.change({"a.txt": "1\n"})
    t = Q.touched_suites(world.dev, commit, f"{commit}~1", Q.RemoteSuites(launch=FakeLaunch(said, rc)))
    assert t.ids == [] and said.strip() in t.tail
    assert t.notes == [f"`research run --on auto` launched no run (exit {rc}): {said.strip()}"]
    assert _no_worktree_left(world, world.dev)


def test_the_worktree_is_removed_when_the_launch_raises(world):
    commit = world.change({"a.txt": "1\n"})

    def boom(argv, cwd, env):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        Q.touched_suites(world.dev, commit, f"{commit}~1", Q.RemoteSuites(launch=boom))
    assert _no_worktree_left(world, world.dev)


def test_a_commit_the_repo_lacks_launches_nothing(world):
    """A worktree that can't be made is a note, not a crash of drive's tick."""
    launch = FakeLaunch()
    t = Q.touched_suites(world.dev, "0" * 40, "1" * 40, Q.RemoteSuites(launch=launch))
    assert t.ids == [] and launch.calls == [] and "no worktree of 000000000000" in t.notes[0] and t.notes[0] in t.tail
    assert _no_worktree_left(world, world.dev)


def _default_touched(h: Harness) -> None:
    """``h``'s driver as `research queue drive` builds it, with its own `launch_touched` (Harness stubs that out)."""
    h.d.__init__(h.q, on="m1", launch=h.launch, states=lambda repo, ids: {i: h.run_state[i] for i in ids},
                 post=lambda text, to=(): h.posts.append(text) or True, now=lambda: h.clock,
                 cancel=lambda repo, ids, reason: h.cancelled.extend(ids), evidence=lambda repo, ids: h.evidence)


def test_drive_launches_a_trains_touched_suites_with_research_run_on_auto_and_lands_once_that_run_passes(world, monkeypatch, tmp_path):
    """Drive's own wiring, with subprocess faked for `research run` alone: the train's commit, against `main`, from a worktree of
    the queue's clone that is gone after the tick; the run id it printed is what drive waits on."""
    from research.drive import CHECK_STEPS
    from test_queue_drive import _steps_run

    real, seen = subprocess.run, []

    def run(argv, *a, **kw):
        if list(argv[1:4]) == ["-m", "research", "run"]:
            wt = Path(kw["cwd"])
            seen.append({"argv": list(argv), "head": _git(wt, "rev-parse", "HEAD"), "env": kw["env"], "stdin": kw.get("stdin")})
            return subprocess.CompletedProcess(argv, 0, "", LAUNCHED)
        return real(argv, *a, **kw)

    monkeypatch.setattr(subprocess, "run", run)
    h = Harness(world, monkeypatch)
    _default_touched(h)
    h.d.defer = frozenset({"pytest"})
    h.run_state[RID] = ("running", None, None)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    tip, main = _tip(world), _git(world.origin, "rev-parse", "main")
    (call,) = seen
    argv = call["argv"]
    assert argv[argv.index("--on") + 1] == "auto" and argv[argv.index("--cwd") + 1] == "source"
    assert call["head"] == tip and _job(argv)[3:7] == ["--head", tip, "--base", main]
    assert call["env"]["RESEARCH_LANE"] == "ci" and call["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert call["stdin"] is subprocess.DEVNULL
    assert h.q.load_state()["drive"]["launched"][tip]["touched"] == [RID]
    assert not Path(argv[argv.index("--source") + 1]).exists() and _no_worktree_left(world, h.q.workdir)
    (rid,) = [r for r in h.run_state if r != RID]
    h.evidence = [_steps_run(tmp_path, rid, {x: "passed" for x in CHECK_STEPS if x != "pytest"})]
    h.tick()
    assert _git(world.origin, "rev-parse", "main") != tip and len(seen) == 1      # its touched suites still run, launched once
    h.run_state[RID] = ("done", 0, None)
    h.evidence = [_run_dir(tmp_path, "passed", [PASSED])]
    h.tick()
    assert _git(world.origin, "rev-parse", "main") == tip


PASSED = {"name": "pkg", "suite": "pkg", "status": "passed", "failed_tests": [], "log": None}


def _run_dir(tmp_path, name: str, report: list | None = None) -> Path:
    """A run dir as `evidence` fetches it: the check's steps, every one but the deferred pytest passed, and, when given, a
    touched-suites run's `out/suites.json`."""
    from research.drive import CHECK_STEPS
    from test_queue_drive import _steps_run

    d = _steps_run(tmp_path, name, {x: "passed" for x in CHECK_STEPS if x != "pytest"})
    if report is not None:
        (d / "out").mkdir()
        (d / "out" / "suites.json").write_text(json.dumps(report))
    return d


def _gated(world, monkeypatch, tmp_path, *changes: dict[str, str]) -> tuple[Harness, str]:
    """A train with pytest deferred whose check, still running, has passed every other step, and whose touched-suites runs
    `launch_touched` hands out as r-touched-1, -2, ... (each recorded in ``h.touched`` with its commit and base)."""
    h = Harness(world, monkeypatch)
    h.d.defer, h.d.train_size, h.touched = frozenset({"pytest"}), 2, []
    n = itertools.count(1)

    def launch_touched(repo, commit, machine, base):
        h.run_state[rid := f"r-touched-{next(n)}"] = ("running", None, None)
        h.touched.append((rid, commit, base))
        return [rid], ""

    h.d.launch_touched = launch_touched
    for pr, change in enumerate(changes or ({"a.txt": "1\n"},), 1):
        world.gh.put(pr, world.change(change))
    h.tick()
    h.evidence = [_run_dir(tmp_path, "check")]
    return h, _tip(world)


def _landed(world, tip: str) -> bool:
    return _git(world.origin, "rev-parse", "main") == tip


def _record(h: Harness, tip: str) -> dict:
    return h.d.q.load_state()["drive"]["launched"][tip]


@pytest.mark.parametrize("killed", [("done", 143), ("failed", 137), ("failed", 129), ("failed", -9), ("cancelled", None),
                                    ("failed", None)], ids=["cap-143", "137", "129", "signal", "cancelled", "no-exit"])
def test_a_capped_or_killed_touched_run_runs_again_once_beside_the_same_check_and_the_landing_waits(world, monkeypatch, tmp_path,
                                                                                                   killed):
    """@ci, 18 Oct: a touched-suites run the 120-min cap ends exits 143, and read as a twin's SIGTERM it let the train land as
    if its suites had passed. A kill (a signal or 128+N, a cancel drive didn't issue, no exit at all) runs it again once,
    beside the same check, and the landing waits on that run."""
    h, tip = _gated(world, monkeypatch, tmp_path)
    main, check = _git(world.origin, "rev-parse", "main"), _record(h, tip)["ids"]
    h.run_state["r-touched-1"] = (*killed, None)
    h.tick(dry_run=True)
    assert len(h.touched) == 1 and not _landed(world, tip)                          # a dry run relaunches nothing
    h.tick()
    rec = _record(h, tip)
    assert not _landed(world, tip) and not h.posts and h.touched[1][1:] == (tip, main)
    assert rec["touched"] == ["r-touched-2"] and rec["touched_killed"] == ["r-touched-1"]
    assert rec["ids"] == check and len(h.launched) == 1                             # beside the same check, not a new one
    h.tick()
    assert len(h.touched) == 2 and not _landed(world, tip)                          # the relaunch runs: nothing more
    h.run_state["r-touched-2"] = ("done", 0, None)
    h.evidence = [_run_dir(tmp_path, "passed", [PASSED])]
    h.tick()
    assert _landed(world, tip) and not h.posts


def test_a_touched_run_killed_again_on_its_relaunch_holds_the_train_and_tells_ci_once(world, monkeypatch, tmp_path):
    h, tip = _gated(world, monkeypatch, tmp_path)
    h.run_state["r-touched-1"] = ("done", 143, None)
    h.tick()
    h.run_state["r-touched-2"] = ("failed", 137, None)
    h.tick()
    assert not _landed(world, tip) and len(h.touched) == 2                          # no third run
    (post,) = h.posts
    assert post.startswith("@ci queue:") and "r-touched-2" in post and "killed again on its relaunch" in post
    assert "the first: r-touched-1" in post and "Held, not landed" in post
    main = _git(world.origin, "rev-parse", "main")
    assert h.d.verdict(tip, main, h.d.q.load_state()["drive"]).state == "running"
    h.tick()
    assert not _landed(world, tip) and len(h.posts) == 1 and len(h.touched) == 2


def test_an_exit_0_whose_report_has_a_failed_row_fails_the_train_and_ejects_the_pr_that_changed_the_test(world, monkeypatch,
                                                                                                          tmp_path):
    from test_queue_drive import _touched_run

    h, tip = _gated(world, monkeypatch, tmp_path, {"a.txt": "1\n"}, {"pkg/tests/test_x.py": "def test_y():\n    assert False\n"})
    h.run_state["r-touched-1"] = ("done", 0, None)
    steps, report = h.evidence, [_touched_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.d.evidence = lambda repo, ids: report if ids == ["r-touched-1"] else steps
    h.tick()
    assert not _landed(world, tip)
    assert any("Ejected #2" in x and "touched suites failed" in x for x in h.posts), h.posts


def test_an_exit_0_with_no_report_is_held_and_told_to_ci_once_its_report_is_overdue(world, monkeypatch, tmp_path):
    """suites.py writes `--json` whenever it ends, so an exit 0 with no report is a run that didn't do its job (or a fetch that
    hasn't brought it yet): it passes nothing."""
    from research.drive import PUBLISH_GRACE_S

    h, tip = _gated(world, monkeypatch, tmp_path)
    h.run_state["r-touched-1"] = ("done", 0, None)
    h.tick()
    assert not _landed(world, tip) and not h.posts and len(h.touched) == 1
    h.clock += PUBLISH_GRACE_S
    h.tick()
    (post,) = h.posts
    assert not _landed(world, tip) and "exited 0, but no out/suites.json" in post and len(h.touched) == 1
    h.evidence = [_run_dir(tmp_path, "late", [])]                                   # no suite touched: an empty report passes
    h.tick()
    assert _landed(world, tip) and len(h.posts) == 1


def test_a_touched_run_that_never_started_runs_again_each_time_without_counting_as_its_kill(world, monkeypatch, tmp_path):
    h, tip = _gated(world, monkeypatch, tmp_path)
    for n in (1, 2):
        h.run_state[f"r-touched-{n}"] = ("done", 75, None)
        h.tick()
        assert len(h.touched) == n + 1 and not _landed(world, tip) and "touched_killed" not in _record(h, tip)
    h.run_state["r-touched-3"] = ("done", 143, None)
    h.tick()
    assert len(h.touched) == 4 and _record(h, tip)["touched_killed"] == ["r-touched-3"] and not h.posts
    h.run_state["r-touched-4"] = ("done", 0, None)
    h.evidence = [_run_dir(tmp_path, "passed", [PASSED])]
    h.tick()
    assert _landed(world, tip) and not h.posts


def test_a_touched_launch_that_registered_no_run_holds_the_train_and_launches_again(world, monkeypatch, tmp_path):
    h = Harness(world, monkeypatch)
    h.d.defer = frozenset({"pytest"})
    outs = itertools.chain([([], "share expired"), ([], "share expired"), (["r-touched-1"], "")], itertools.repeat(([], "")))
    h.d.launch_touched = lambda repo, commit, machine, base: next(outs)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    tip = _tip(world)
    h.evidence = [_run_dir(tmp_path, "check")]
    h.tick()
    (post,) = h.posts
    assert not _landed(world, tip) and "registered no run (share expired)" in post and "Held, not landed" in post
    h.run_state["r-touched-1"] = ("running", None, None)
    h.tick()
    assert _record(h, tip)["touched"] == ["r-touched-1"] and not _landed(world, tip) and len(h.posts) == 1
    h.run_state["r-touched-1"] = ("done", 0, None)
    h.evidence = [_run_dir(tmp_path, "passed", [PASSED])]
    h.tick()
    assert _landed(world, tip)


def test_a_resend_runs_the_touched_suites_again_through_research_run_on_auto_and_cancels_only_its_own(world, monkeypatch,
                                                                                                     tmp_path):
    """#2542's resend, with drive's own `launch_touched`: the resent check's touched suites are a new `research run --on auto`,
    the run it replaces is cancelled, and that cancel, drive's own, is no kill of the new one."""
    from test_queue_drive import _failed_run

    real, seen, rids = subprocess.run, [], iter([RID, RID2])
    h = Harness(world, monkeypatch)

    def run(argv, *a, **kw):
        if list(argv[1:4]) == ["-m", "research", "run"]:
            seen.append(rid := next(rids))
            h.run_state[rid] = ("running", None, None)
            return subprocess.CompletedProcess(argv, 0, "", LAUNCHED.replace(RID, rid))
        return real(argv, *a, **kw)

    monkeypatch.setattr(subprocess, "run", run)
    _default_touched(h)
    h.d.defer, h.d.train_size = frozenset({"pytest"}), 2
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.tick()
    tip = _tip(world)
    (check,) = [r for r in h.run_state if r not in seen]
    h.run_state[check] = ("running", None, None, "suite pkg (tests/test_x.py::test_y)")
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    rec = _record(h, tip)
    assert seen == [RID, RID2] and h.cancelled == [check, RID] and rec["touched"] == [RID2] and check not in rec["ids"]
    assert any("resent" in x for x in world.logs) and not h.posts
    h.run_state[RID] = ("cancelled", 143, None)
    h.run_state[RID2] = ("done", 0, None)
    h.evidence = [_run_dir(tmp_path, "passed", [PASSED])]
    h.tick()
    assert _landed(world, tip) and not h.posts and seen == [RID, RID2]


def test_touched_launch_returns_the_ids_and_why_nothing_launched(world):
    commit = world.change({"a.txt": "1\n"})
    assert touched_launch(world.dev, commit, f"{commit}~1", how=Q.RemoteSuites(launch=FakeLaunch()))[0] == [RID]
    ids, tail = touched_launch(world.dev, commit, f"{commit}~1", how=Q.RemoteSuites(launch=FakeLaunch("nope", 2)))
    assert ids == [] and "nope" in tail


def test_the_preflight_and_drive_share_touched_suites_and_a_local_run_reports_failures_as_suites_json_rows(world):
    """CI-73: the preflight's collection is `touched_suites` with `LocalSuites`; its failures come back as the rows drive reads from
    a remote run's `out/suites.json`."""
    from test_queue import PREFLIGHT_FILES

    base = _move_main(world, PREFLIGHT_FILES)
    commit = world.change({"pkg/tests/test_rows.py": "import json\nfrom pathlib import Path\n"
                           "ROW = json.loads((Path(__file__).parents[1] / 'rows.json').read_text())['r9']\n"})

    def run(argv, cwd):
        return subprocess.run(argv, cwd=cwd, capture_output=True, text=True), ""

    t = Q.touched_suites(world.dev, commit, base, Q.LocalSuites(sys.executable, run, free=lambda: 1.0))
    assert sorted(t.suites) == [".", "pkg"] and t.ids == []                    # the root suite keys on the whole tree
    (row,) = t.failed
    assert set(Q.quick_tier.ROW_KEYS) <= set(row) and row["status"] == "failed" and row["suite"] == "pkg"
    assert row["failed_tests"] == ["tests/test_rows.py"] and "KeyError: 'r9'" in row["summary"][0]
    assert any(n.startswith("collected 2 suite(s)") for n in t.notes), t.notes
    assert _no_worktree_left(world, world.dev)


def test_the_preflight_collects_through_touched_suites_in_its_own_worktree(world, monkeypatch, capsys):
    """The preflight's merged tree is the worktree `touched_suites` runs in (its synced environment), and nothing else makes one."""
    from test_queue import _preflight, _preflight_main

    _preflight_main(world)
    real, seen = Q.touched_suites, []
    monkeypatch.setattr(Q, "touched_suites", lambda *a, **kw: seen.append((a, kw)) or real(*a, **kw))
    world.gh.put(2, world.change({"pkg/tests/test_more.py": "def test_more():\n    pass\n"}), ready=False)
    assert Q.ready(world.gh, world.store, 2, "lane-2", Q.MarkOnly(), preflight=_preflight(world)) == 0, capsys.readouterr().err
    ((args, kw),) = seen
    assert isinstance(args[3], Q.LocalSuites) and args[3].suites is None and kw["wt"].name == "tree"
    assert any("collected 2 suite(s)" in s for s in world.logs)
