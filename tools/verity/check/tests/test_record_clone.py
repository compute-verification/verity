"""`check.py --record --on`: every run of a commit on a machine shares the shipped tree <root>/src/<sha>, so sibling checks of one
commit would race in one .lake. Each launch asks `research run` for its own clone of that tree, and placement still knows the commit a
cloned check's slot holds, though the clone's path (<root>/scratch/<run id>/tree) names none."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from research import clone

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("repo_check_record_clone", ROOT / "tools" / "verity" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)
SHA = "6b3d704c8" * 4 + "abcd"


def _launches(monkeypatch, *argvs):
    real, launched = subprocess.run, []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            launched.append(cmd[:cmd.index("--")])
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    monkeypatch.setattr(CHECK.subprocess, "run", run)
    monkeypatch.setattr(CHECK, "landable", lambda base: True)
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: None)
    monkeypatch.setattr(CHECK, "gather_verdicts", lambda d, given=None: None)
    for argv in argvs:
        assert CHECK.record(["--base", "HEAD", *argv]) == 0
    return launched


def test_every_launch_on_a_machine_asks_for_its_own_clone_and_a_local_one_runs_in_its_checkout(monkeypatch):
    """A train's check and drive's after-landing run of the same commit, on one machine: each is a `research run --cwd clone`."""
    on = _launches(monkeypatch, ["--on", "vy-x", "--priority", "--keep-going"],
                   ["--on", "vy-x", "--priority", "--keep-going", "--deferred", "lean-audit"])
    assert len(on) == 2
    for head in on:
        assert head.count("--cwd") == 1 and head[head.index("--cwd") + 1] == "clone", head
        assert head[head.index("--source") + 1] == "."
    [local] = _launches(monkeypatch, [])
    assert local[local.index("--cwd") + 1] == "source"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


def test_two_runs_of_one_commit_on_one_machine_get_two_trees_whose_lake_builds_are_their_own(tmp_path):
    """What the machine's runner does for `--cwd clone` (research.cli._make_clone): a clone of the shipped tree per run id."""
    root = tmp_path / "research"
    shipped = root / "src" / "x"
    shipped.mkdir(parents=True)
    _git(shipped, "init", "-q")
    (shipped / "lakefile.toml").write_text('name = "p"\n')
    _git(shipped, "add", ".")
    _git(shipped, "commit", "-q", "-m", "one")
    sha = _git(shipped, "rev-parse", "HEAD")
    shipped = shipped.rename(root / "src" / sha)
    (root / "runs").mkdir()
    trees = []
    for rid in ("r20261025-050553-4d0a", "r20261025-050601-77e1"):
        tree = clone.home(root / "runs", rid, machine=True) / "tree"
        assert clone.make(shipped, tree, commit=sha) == sha
        trees.append(tree)
    a, b = trees
    assert a != b and a.parent.parent == b.parent.parent == root / "scratch" and _git(b, "rev-parse", "HEAD") == sha
    (a / ".lake" / "build").mkdir(parents=True)
    (a / ".lake" / "build" / "RecSparse.olean").write_bytes(b"")
    assert not (b / ".lake").exists() and not (shipped / ".lake").exists()


def test_placement_knows_the_commit_a_cloned_checks_slot_holds():
    """slot.py records the shipped commit (RESEARCH_SOURCE_SHA) beside its cwd, which in a clone names no commit."""
    holder = {"pid": 11, "run": "r1", "tree": "/workspace/research/scratch/r1/tree", "commit": SHA}
    old = {"pid": 12, "run": "r0", "tree": f"/workspace/research/src/{'a' * 40}"}
    doc = {"check": {"slots": [{"name": "a", "holder": holder}], "waiters": [old]}}
    assert CHECK._trees(json.loads(json.dumps(doc))) == {SHA[:12], "a" * 12}
    assert f"tree {SHA[:12]}" in CHECK._who({**holder, "since": "2026-10-25T05:09Z", "minutes": 3})
