"""`verity lean try` ships the checkout FILE lies in (or the current directory's), not the one the CLI's code lives in."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import verity_cli


def _git(cwd: Path, *argv: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *argv], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def checkouts(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "main"
    (main / "tools" / "verity" / "lean").mkdir(parents=True)
    (main / "tools" / "verity" / "lean" / "lean_changed.py").write_text("")
    (main / "A.lean").write_text("-- main's\n")
    _git(main, "init", "-q")
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "base")
    _git(main, "worktree", "add", "-q", "-b", "lane", str(tmp_path / "wt"))
    (tmp_path / "wt" / "B.lean").write_text("-- the worktree's only\n")
    return main, tmp_path / "wt"


def test_ships_the_worktree_the_file_is_in(checkouts, monkeypatch):
    main, wt = checkouts
    monkeypatch.chdir(main)
    assert verity_cli._tree(str(wt / "A.lean")) == wt.resolve()


def test_ships_the_current_directorys_checkout(checkouts, monkeypatch):
    main, wt = checkouts
    monkeypatch.chdir(wt)
    assert verity_cli._tree("A.lean") == wt.resolve()
    monkeypatch.chdir(wt / "tools")
    assert verity_cli._tree("B.lean") == wt.resolve()


def test_snapshot_of_a_dirty_worktree_holds_its_edits(checkouts, tmp_path, monkeypatch):
    _, wt = checkouts
    for k in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(k, "t@t")
    shipped = verity_cli._snapshot(wt.resolve(), tmp_path)
    assert (shipped / "B.lean").read_text() == "-- the worktree's only\n"
    _git(wt, "worktree", "remove", "--force", str(shipped))


def test_refuses_a_checkout_that_isnt_verity(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    monkeypatch.chdir(other)
    with pytest.raises(SystemExit, match="not a Verity checkout"):
        verity_cli._tree("X.lean")
