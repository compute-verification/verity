"""A `--cwd source` workload that builds Lean takes the shared tree's lock first (research/treelock.py): two `lake build`s of one
commit on a machine wrote one `.lake` at once and clang died of SIGBUS on the other's truncated .c files (INF-641)."""
from __future__ import annotations

import fcntl
import json
import os
import sys
from pathlib import Path

import pytest

from research import treelock
from research.cli import main
from waits import FakeClock

PY = sys.executable
#: a `lake` that says whether its tree's lock (job.json run.lean_tree_lock) is held while it runs: a second open of it can't take it
FAKE_LAKE = f"""#!{PY}
import fcntl, json, os, sys
d = os.environ["RESEARCH_RUN_DIR"]
fd = os.open(json.load(open(os.path.join(d, "job.json")))["run"]["lean_tree_lock"], os.O_RDWR)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    held = "free"
except BlockingIOError:
    held = "held"
open(os.path.join(d, "lake.txt"), "w").write(" ".join([held, os.getcwd(), *sys.argv[1:]]))
"""


@pytest.mark.parametrize("command, builds", [
    (["bash", "-lc", "uv sync -q && lake build Flock"], True),
    (["python3", "tools/verity/lean/lean_changed.py", "--base", "x"], True),
    (["python3", "tools/verity/check/slot.py", "--by", "t", "--", "nice", "-n", "19", "python3", "tools/verity/check/check.py"], True),
    (["bash", "-lc", "uv sync -q --all-packages && uv run tools/verity/check/suites.py --quick"], True),
    (["uv", "run", "--extra", "torch-cpu", "verity", "check", "lean"], True),
    (["/home/research/.elan/bin/lake", "env", "lean", "X.lean"], True),
    (["python", "bench.py", "--out", "o"], False),
    (["bash", "-c", "du -sh .lake/build"], False),
    (["echo", "lakeside", "check.pyc"], False),
    (["verity", "bench", "one_stage"], False),
])
def test_what_builds_lean(command, builds):
    assert treelock.builds_lean(command) is builds


def test_a_waiter_takes_the_lock_once_its_holder_lets_go_and_gives_up_after_its_wait(tmp_path, capsys):
    lock = tmp_path / "tree.lock"
    holder = os.open(lock, os.O_RDWR | os.O_CREAT)
    fcntl.flock(holder, fcntl.LOCK_EX)
    clock = FakeClock()
    clock.at(60, lambda: os.close(holder))
    fd = treelock.acquire(str(lock), 300, "/t", clock=clock.module.monotonic, sleep=clock.sleep)
    assert fd is not None and clock.now == 60
    err = capsys.readouterr().err
    assert err.count("waiting up to 300 s for another run's Lean build in the shared tree /t") == 1 and "--cwd clone" in err
    assert treelock.acquire(str(lock), 300, "/t", clock=clock.module.monotonic, sleep=clock.sleep) is None, "until the wait is spent"
    assert clock.now == 360
    assert treelock.main([str(lock), "0", "/t", "--", "true"]) == treelock.EX_NOT_STARTED
    err = capsys.readouterr().err
    assert "nothing ran" in err and "launch it with `research run --cwd clone`" in err
    os.close(fd)


def test_a_lean_build_in_the_shared_tree_runs_holding_its_lock_and_other_runs_take_none(tmp_path, monkeypatch):
    tree, bin_dir, runs = tmp_path / "tree", tmp_path / "bin", tmp_path / "runs"
    tree.mkdir()
    bin_dir.mkdir()
    (bin_dir / "lake").write_text(FAKE_LAKE)
    (bin_dir / "lake").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir), prepend=os.pathsep)
    monkeypatch.chdir(tree)
    assert main(["run", "--runs-dir", str(runs), "--no-sampler", "--cwd", "source", "--", "lake", "build", "Flock"]) == 0
    [run_dir] = [d for d in runs.iterdir() if d.is_dir()]
    lock = json.loads((run_dir / "job.json").read_text())["run"]["lean_tree_lock"]
    assert lock == str(treelock.lock_path(runs, tree)) and Path(lock).parent == runs
    assert (run_dir / "lake.txt").read_text() == f"held {tree.resolve()} build Flock", "the build ran in the tree, holding its lock"
    for i, (args, code) in enumerate(((["--cwd", "source"], "print('bench')"), ([], "print('lake build')"))):
        run_id = f"r20261109-000000-000{i}"
        assert main(["run", "--runs-dir", str(runs), "--id", run_id, "--no-sampler", *args, "--", PY, "-c", code]) == 0
        assert "lean_tree_lock" not in json.loads((runs / run_id / "job.json").read_text())["run"], "no Lean build, or not the tree"
