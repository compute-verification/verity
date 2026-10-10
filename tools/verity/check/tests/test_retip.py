"""`retip`: a failed check of a train's tip whose only failures are Lean record differences has its regenerated records committed
onto the prep branch as a fast-forward; any other failure, another commit, or a record that isn't the tip's refuses and pushes
nothing.  On a toy repository with a bare remote and a stand-in store."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("retip_under_test", HERE.parent / "retip.py")
R = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
spec.loader.exec_module(R)

ENV = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
OLD = {"roots": ["Pkg"], "guarantees": {"Pkg.g": {"signature": "old"}}, "reads": {}}
NEW = {"roots": ["Pkg"], "guarantees": {"Pkg.g": {"signature": "new"}}, "reads": {}}
BRANCH = "cursor/train-prep-1-on-abc-4292"


def text(doc: dict) -> bytes:
    return (json.dumps(doc, indent=1) + "\n").encode()


@pytest.fixture(autouse=True)
def git_env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(tmp_path, "init", "-q", "-b", "main", str(work))
    (work / "Pkg").mkdir()
    (work / "Pkg" / "lean-audit.json").write_bytes(text(OLD))
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", "tip")
    git(work, "remote", "add", "origin", str(remote))
    git(work, "push", "-q", "origin", f"HEAD:refs/heads/{BRANCH}")
    return work


@dataclass
class Manifest:
    meta: dict


class Store:
    """The store calls retip makes: an attempt, its result's steps, its run files' listing, and a fetch of some of them."""

    def __init__(self):
        self.attempts, self.results, self.files = {}, {}, {}

    def add(self, run, commit, steps, files, rc=1, tool="check", dirty=False):
        self.attempts[run] = {"tool": {"name": tool}, "source": {"commit": commit, "dirty": dirty},
                              "status": {"state": "failed" if rc else "done", "rc": rc},
                              "outputs": {"result": f"art:res-{run}", "run_files": f"art:files-{run}"}}
        self.results[f"art:res-{run}"] = {"steps": steps}
        self.files[f"art:files-{run}"] = {p: (b if isinstance(b, bytes) else json.dumps(b).encode()) for p, b in files.items()}

    def get_attempt(self, run):
        return self.attempts.get(run)

    def get_manifest(self, art):
        return Manifest(self.results[art])

    def listing(self, art):
        return [{"path": p} for p in self.files[art]]

    def fetch(self, art, to, *, paths):
        for p in paths:
            (Path(to) / p).parent.mkdir(parents=True, exist_ok=True)
            (Path(to) / p).write_bytes(self.files[art][p])
        return Path(to)


def failed_run(store, tip, run="r20261010-170000-aaaa", regen=NEW, steps=None, report=None):
    rep = report or {"package": "Pkg", "failures": ["guarantees: Pkg.g's signature changed: ..."],
                     "regenerated": "/workspace/research/runs/x/lean-audit/audit-Pkg/lean-audit.json"}
    store.add(run, tip, steps or [{"step": "lean-audit", "status": "failed"}, {"step": "pytest", "status": "stopped"},
                                  {"step": "preflight-lock", "status": "passed"}],
              {"lean-audit/audit.json": {"passed": False, "failed": ["Pkg"], "packages": [rep]},
               "lean-audit/audit-Pkg/lean-audit.json": text(regen), "lean-audit/audit-Pkg/review.txt": b"..."})
    return run


def tip_of(repo):
    return git(repo, "ls-remote", "origin", f"refs/heads/{BRANCH}").split()[0]


def test_record_differences_land_on_the_tip_as_one_fast_forward_commit_naming_the_files(repo):
    store, tip, said = Store(), tip_of(repo), []
    run = failed_run(store, tip)
    assert R.main([BRANCH, run, "--repo", str(repo)], store=store, log=said.append) == 0, said
    new = tip_of(repo)
    assert git(repo, "rev-parse", f"{new}^") == tip
    assert git(repo, "show", f"{new}:Pkg/lean-audit.json") + "\n" == text(NEW).decode()
    msg = git(repo, "log", "-1", "--format=%B", new)
    assert run in msg and git(repo, "log", "-1", "--format=%(trailers:key=Regenerated,valueonly)", new) == "Pkg/lean-audit.json"
    assert new[:12] in said[-1] and "train.sh" in said[-1]
    assert not git(repo, "worktree", "list").count("\n")


@pytest.mark.parametrize("case, want", [
    ("another step failed", "failed more than its Lean records: pytest"),
    ("not regenerated", "Pkg failed for more than its records"),
    ("another commit", "not the tip"),
    ("passed", "did not fail"),
    ("not this tip's record", "also changes roots"),
    ("the tip's own", "rerun the check"),
    ("unreadable", "can't be read from the store"),
])
def test_anything_but_record_differences_of_this_tip_refuses_and_pushes_nothing(repo, case, want):
    store, tip, said = Store(), tip_of(repo), []
    if case == "another step failed":
        run = failed_run(store, tip, steps=[{"step": "lean-audit", "status": "failed"}, {"step": "pytest", "status": "failed"}])
    elif case == "not regenerated":
        run = failed_run(store, tip, report={"package": "Pkg", "failures": ["axioms: Pkg.g uses sorryAx"]})
    elif case == "another commit":
        run = failed_run(store, "0" * 40)
    elif case == "passed":
        run = failed_run(store, tip)
        store.attempts[run]["status"] = {"state": "done", "rc": 0}
    elif case == "not this tip's record":
        run = failed_run(store, tip, regen={**NEW, "roots": ["Other"]})
    elif case == "unreadable":
        run = failed_run(store, tip)
        del store.files[f"art:files-{run}"]
    else:
        run = failed_run(store, tip, regen=OLD)
    assert R.main([BRANCH, run, "--repo", str(repo)], store=store, log=said.append) == 1
    assert want in said[-1], said
    assert tip_of(repo) == tip


def test_shards_add_up_and_a_dry_run_pushes_nothing(repo):
    store, tip, said = Store(), tip_of(repo), []
    a = failed_run(store, tip, run="r20261010-170000-aaaa")
    b = failed_run(store, tip, run="r20261010-170000-bbbb")
    assert R.main([BRANCH, a, b, "--repo", str(repo), "--dry-run"], store=store, log=said.append) == 0
    assert "would commit Pkg/lean-audit.json" in said[-1] and tip_of(repo) == tip
    store.files[f"art:files-{b}"]["lean-audit/audit-Pkg/lean-audit.json"] = text({**NEW, "reads": {"x": 1}})
    assert R.main([BRANCH, a, b, "--repo", str(repo)], store=store, log=said.append) == 1
    assert "different records" in said[-1] and tip_of(repo) == tip
