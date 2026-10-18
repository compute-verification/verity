"""A train's touched-suites run goes to `research run --on auto` (vy-nebius-2's agent-test queue), outside the check slots: on 18 Oct,
in a priority check slot, those runs waited 21-93 min for a 560 GB slot and ran 27-48 min, competing with the trains for six slots.
And `queue.touched_suites` is the one way both the preflight and drive run a merged tree's touched suites (CI-73).  The launch's
subprocess is faked; the worktrees are real."""
from __future__ import annotations

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

RID = "r20261018-192500-ab12"
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
    assert head[4:] == ["--on", "auto", "--source", str(wt), "--cwd", "source", "--cpus", "16", "--mem-gb", "64",
                        "--max-min", "90", "--send", str(Path(Q.quick_tier.__file__)), "--require-result",
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
    assert _git(world.origin, "rev-parse", "main") != tip                         # its touched suites still run
    h.run_state[RID] = ("done", 0, None)
    h.tick()
    assert _git(world.origin, "rev-parse", "main") == tip
    assert len(seen) == 1                                                          # launched once


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
