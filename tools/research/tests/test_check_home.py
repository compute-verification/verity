"""Drive's and the queue's configuration (`CHECK_HOME`: drive.toml, queue.toml, ci.toml) is read from the repository each reader
is given, at the commit it names, never from the working directory, so drive can run from another repository against verity's."""
from __future__ import annotations

import subprocess
from pathlib import Path

from research import merge, queue


def _repo(root: Path, files: dict[str, str]) -> str:
    root.mkdir(parents=True)
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "config"]):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=root,
                       check=True, capture_output=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def test_the_three_files_are_at_one_home():
    assert {queue.DRIVE_FLAGS, queue.RULES_BEFORE_MOVE, queue.CI_JOBS} == {
        f"{queue.CHECK_HOME}/{f}" for f in ("drive.toml", "queue.toml", "ci.toml")}


def test_each_reader_reads_the_repository_it_is_given_not_the_working_directory(tmp_path, monkeypatch):
    verity = tmp_path / "verity"
    sha = _repo(verity, {queue.DRIVE_FLAGS: 'land_into = "checked"\nlanding_gates = ["gate.py"]\nstale_days = 9\n',
                         queue.RULES_BEFORE_MOVE: 'hold_label = "parked"\n',
                         queue.CI_JOBS: '[[job]]\nname = "pytest"\n[[job]]\nname = "circuit-check"\ngate = false\n'})
    _repo(tmp_path / "ops", {"README.md": "the ops repository, which holds none of verity's configuration\n"})
    monkeypatch.chdir(tmp_path / "ops")
    assert queue.flags_at(verity, sha).stale_days == 9
    assert queue.gates_at(verity, sha) == ("gate.py",)
    assert queue.into_at(verity, sha) == "checked"
    assert queue.rules_at(verity, sha).hold_label == "parked"
    assert merge.declared_jobs(verity, sha) == ["pytest"]
    ops = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    assert queue.into_at(tmp_path / "ops", ops) == "main" and queue.rules_at(tmp_path / "ops", ops) is None
    assert "can't be read" in merge.declared_jobs(tmp_path / "ops", ops)
