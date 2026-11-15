"""`resolve_renames.py`: in a merge of `main` after a word move into a PR's head, a conflict that is only the move's rename
against the PR's edit is resolved by putting the edit at the new name; a conflict with another change on `main` is left,
with its hunk printed, for the owner."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[3] / "tools" / "move" / "resolve_renames.py"
ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
MAP = '[words]\ndrawn = "sampled"\n'
UNIT = "def first(drawn):\n    return drawn + 1\n\n\ndef second(drawn):\n    return drawn * 2\n"


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=check, capture_output=True,
                          text=True)


def commit(repo: Path, msg: str, files: dict[str, str]) -> None:
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)


def repo_with(tmp_path: Path, main_unit: str, pr_unit: str) -> Path:
    """main renames drawn to sampled (a word move under its move line) and then holds ``main_unit``; the PR, from before the
    move, holds ``pr_unit``. The PR's head is checked out, stopped in `git merge main`."""
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    commit(r, "old", {"app/unit.py": UNIT})
    git(r, "branch", "pr")
    commit(r, "rename: drawn to sampled\n\nMove: tools/move/sample_words_map.toml",
           {"tools/move/sample_words_map.toml": MAP, "app/unit.py": UNIT.replace("drawn", "sampled")})
    if main_unit != UNIT.replace("drawn", "sampled"):
        commit(r, "main's own edit", {"app/unit.py": main_unit})
    git(r, "checkout", "-q", "pr")
    commit(r, "the PR's edit", {"app/unit.py": pr_unit})
    assert git(r, "merge", "-q", "--no-edit", "main", check=False).returncode == 1
    return r


def binary_conflict(tmp_path: Path) -> Path:
    """Both sides change a binary file (a NUL byte in it) after the word move; the merge stops on it."""
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "app").mkdir()
    (r / "app/blob.bin").write_bytes(b"drawn\0old\n")
    commit(r, "old", {"app/unit.py": UNIT})
    git(r, "branch", "pr")
    commit(r, "rename: drawn to sampled\n\nMove: tools/move/sample_words_map.toml",
           {"tools/move/sample_words_map.toml": MAP, "app/unit.py": UNIT.replace("drawn", "sampled")})
    (r / "app/blob.bin").write_bytes(b"drawn\0main\n")
    commit(r, "main's blob", {})
    git(r, "checkout", "-q", "pr")
    (r / "app/blob.bin").write_bytes(b"drawn\0pr\n")
    commit(r, "the PR's blob", {})
    assert git(r, "merge", "-q", "--no-edit", "main", check=False).returncode == 1
    return r


def resolve(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), ".", "main"], cwd=repo, capture_output=True, text=True)


def test_a_rename_against_the_pr_s_edit_is_resolved_at_the_new_name(tmp_path):
    r = repo_with(tmp_path, UNIT.replace("drawn", "sampled"), UNIT.replace("return drawn + 1", "return drawn + 10"))
    p = resolve(r)
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.strip() == "app/unit.py: resolved"
    assert (r / "app/unit.py").read_text() == UNIT.replace("drawn", "sampled").replace("+ 1", "+ 10")
    assert git(r, "ls-files", "-u").stdout == ""


def test_a_conflict_with_another_change_on_main_is_left_for_the_owner(tmp_path):
    moved = UNIT.replace("drawn", "sampled")
    r = repo_with(tmp_path, moved.replace("return sampled + 1", "return sampled + 3"),
                  UNIT.replace("return drawn + 1", "return drawn + 10").replace("drawn * 2", "drawn * 4"))
    p = resolve(r)
    assert p.returncode == 1
    assert p.stdout.splitlines() == ["app/unit.py: 1 hunks left", "    left:     return sampled + 10"]
    text = (r / "app/unit.py").read_text()
    assert "<<<<<<< ours (the PR, renamed)" in text and "return sampled * 4" in text and "drawn" not in text
    assert git(r, "ls-files", "-u", "--", "app/unit.py").stdout != ""


def test_a_binary_file_both_sides_changed_is_left_unmerged_as_git_left_it(tmp_path):
    r = binary_conflict(tmp_path)
    left = (r / "app/blob.bin").read_bytes()
    p = resolve(r)
    assert p.returncode == 1
    assert p.stdout.splitlines() == ["app/blob.bin: the owner's: a binary file, which no map rewrites and git can't merge"]
    assert (r / "app/blob.bin").read_bytes() == left and left
    assert git(r, "ls-files", "-u", "--", "app/blob.bin").stdout != ""
