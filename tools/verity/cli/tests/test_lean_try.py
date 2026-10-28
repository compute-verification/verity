"""`verity lean try FILE`, rooted at the git checkout it runs in, not where verity is installed. Without `--on`: `lake env lean
FILE` in the directory of the nearest lakefile above FILE, with lean's exit status passed through, against a stand-in `lake`
on PATH, so no Lean toolchain is needed. With `--on`: the `research run` it launches (a stand-in) ships that checkout's
working tree and names FILE from its root, and the snapshot's worktree is removed there."""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import verity_cli


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def worktrees(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """A repository with a Lake package `pkg` holding `Pkg/A.lean`, committed, and a second worktree of it, edited there."""
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t",
                        "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}.items():
        monkeypatch.setenv(name, value)
    first, second = tmp_path / "first", tmp_path / "second"
    (first / "pkg" / "Pkg").mkdir(parents=True)
    (first / "pkg" / "lakefile.toml").write_text('name = "pkg"\n')
    (first / "pkg" / "Pkg" / "A.lean").write_text("theorem a : True := trivial\n")
    git(first, "init", "-q")
    git(first, "add", "-A")
    git(first, "commit", "-q", "-m", "a")
    git(first, "worktree", "add", "-q", str(second))
    (second / "pkg" / "Pkg" / "A.lean").write_text("theorem a : True := by trivial\n")
    return first.resolve(), second.resolve()


def stand_in_lake(tmp_path: Path, monkeypatch, rc: int) -> Path:
    """A `lake` that writes its working directory and arguments to the file it returns, and exits with ``rc``."""
    called, bin_ = tmp_path / "called", tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "lake").write_text(f'#!/bin/sh\n{{ pwd -P; echo "$@"; }} > "{called}"\nexit {rc}\n')
    (bin_ / "lake").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_), prepend=os.pathsep)
    return called


def test_try_runs_lake_env_lean_in_the_nearest_package_and_passes_its_status(tmp_path, monkeypatch, capsys):
    (tmp_path / "lakefile.toml").write_text('name = "Outer"\n')
    pkg = tmp_path / "Outer" / "Inner"
    (pkg / "Inner" / "Area").mkdir(parents=True)
    (pkg / "lakefile.lean").write_text("")
    file = pkg / "Inner" / "Area" / "File.lean"
    file.write_text("theorem t : True := trivial\n")
    called = stand_in_lake(tmp_path, monkeypatch, 3)
    assert verity_cli.main(["lean", "try", str(file)]) == 3
    assert called.read_text().splitlines() == [str(pkg.resolve()), "env lean Inner/Area/File.lean"]
    assert f"verity lean try: lake env lean Inner/Area/File.lean (in {pkg.resolve()})" in capsys.readouterr().out


def test_try_refuses_a_file_in_no_package_and_remote_options_without_on(tmp_path, monkeypatch, capsys):
    called = stand_in_lake(tmp_path, monkeypatch, 0)
    loose = tmp_path / "Loose.lean"
    loose.write_text("")
    with pytest.raises(SystemExit, match="Loose.lean is not a .lean file in a Lake package"):
        verity_cli.main(["lean", "try", str(loose)])
    with pytest.raises(SystemExit):
        verity_cli.main(["lean", "try", str(loose), "--pool", "P"])
    assert "--pool and --as go with --on" in capsys.readouterr().err
    assert not called.exists()


def test_try_resolves_file_from_the_root_of_the_worktree_it_runs_in(worktrees, tmp_path, monkeypatch, capsys):
    _, second = worktrees
    called = stand_in_lake(tmp_path, monkeypatch, 0)
    monkeypatch.chdir(second / "pkg")
    assert verity_cli.main(["lean", "try", "pkg/Pkg/A.lean"]) == 0
    assert called.read_text().splitlines() == [str(second / "pkg"), "env lean Pkg/A.lean"]
    assert "verity lean try: lake env lean Pkg/A.lean (in pkg)" in capsys.readouterr().out


def test_try_on_ships_the_worktree_it_runs_in_and_names_file_from_its_root(worktrees, monkeypatch, capsys):
    first, second = worktrees
    launched, popen = {}, subprocess.Popen

    def research_run(argv, *a, **kw):
        if argv[:3] != [sys.executable, "-m", "research"]:
            return popen(argv, *a, **kw)
        source = Path(argv[argv.index("--source") + 1])
        launched.update(cwd=Path(kw["cwd"]), file=argv[-1], shipped=(source / "pkg" / "Pkg" / "A.lean").read_text())
        return types.SimpleNamespace(stderr=iter(()), wait=lambda: 0)

    monkeypatch.setattr(subprocess, "Popen", research_run)
    monkeypatch.chdir(second / "pkg")
    assert verity_cli.main(["lean", "try", "Pkg/A.lean", "--on", "node"]) == 1
    assert launched == {"cwd": second, "file": "pkg/Pkg/A.lean", "shipped": "theorem a : True := by trivial\n"}
    assert "`research run` exited 0 without starting a run" in capsys.readouterr().err
    assert len(git(second, "worktree", "list", "--porcelain").split("\n\n")) == 2
    launched.clear()
    with pytest.raises(SystemExit, match="is outside the checkout this runs in"):
        verity_cli.main(["lean", "try", str(first / "pkg" / "Pkg" / "A.lean"), "--on", "node"])
    assert not launched
