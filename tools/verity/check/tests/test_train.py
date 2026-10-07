"""`train.sh`'s refusal of a commit that can't land on origin/main as it is, before it checks anything out or launches."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
TRAIN = ROOT / "tools" / "verity" / "check" / "train.sh"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, name: str, text: str) -> str:
    (repo / name).write_text(text)
    _git(repo, "add", name)
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", name)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def toy(tmp_path):
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    _git(tmp_path, "clone", "-q", str(origin), str(work))
    _git(work, "checkout", "-q", "-b", "main")
    _commit(work, "a", "base\n")
    _git(work, "push", "-q", "origin", "main")
    return work


def _train(work: Path, tmp_path: Path, commit: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "AWS_ACCESS_KEY_ID": "dummy-key-id", "R2_ENDPOINT": "https://dummy.invalid", "TMPDIR": str(tmp_path)}
    env.pop("AWS_SESSION_TOKEN", None)
    env.pop("VERITY_SKIP_STORE", None)
    return subprocess.run(["bash", str(TRAIN), commit, "toy-pod"], cwd=work, env=env, capture_output=True, text=True)


def _main_moves(work: Path, tip: str) -> None:
    _git(work, "checkout", "-q", "main")
    _commit(work, "b", "main moved\n")
    _git(work, "push", "-q", "origin", "main")
    _git(work, "checkout", "-q", tip)


def test_a_tip_main_has_moved_past_is_refused(toy, tmp_path):
    _git(toy, "checkout", "-q", "-b", "tip")
    tip = _commit(toy, "c", "train\n")
    _main_moves(toy, tip)
    r = _train(toy, tmp_path, tip)
    assert r.returncode == 1, r.stderr
    assert "can't land on origin/main as it is" in r.stderr
    assert not list(tmp_path.glob("train-*")), "refused before checking anything out"


def test_a_tip_that_conflicts_with_main_is_refused(toy, tmp_path):
    _git(toy, "checkout", "-q", "-b", "tip")
    tip = _commit(toy, "b", "train's b\n")
    _main_moves(toy, tip)
    r = _train(toy, tmp_path, tip)
    assert r.returncode == 1, r.stderr
    assert "can't land on origin/main as it is" in r.stderr


def test_a_tip_already_merged_with_main_passes_the_refusal(toy, tmp_path):
    _git(toy, "checkout", "-q", "-b", "tip")
    _commit(toy, "c", "train\n")
    _main_moves(toy, "tip")
    _git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-edit", "origin/main")
    tip = _git(toy, "rev-parse", "HEAD")
    env_path = tmp_path / "bin"
    env_path.mkdir()
    (env_path / "uv").write_text("#!/bin/sh\necho 'check: refused (toy)'\n")
    (env_path / "uv").chmod(0o755)
    env = {**os.environ, "PATH": f"{env_path}:{os.environ['PATH']}", "AWS_ACCESS_KEY_ID": "dummy-key-id",
           "R2_ENDPOINT": "https://dummy.invalid", "TMPDIR": str(tmp_path)}
    r = subprocess.run(["bash", str(TRAIN), tip, "toy-pod"], cwd=toy, env=env, capture_output=True, text=True)
    assert "can't land" not in r.stderr
    assert f"train: {tip[:12]} on toy-pod" in r.stdout
