"""`RECORD_DRIVER`'s lean drivers run `origin/main`'s `tools/verity/lean/merge.py` when the repository has that ref, and the
worktree's otherwise.  Git runs a driver from the worktree of the side merged into, so an old branch that merged main ran its
own older copy, which collapsed Security's lock to one `reads` group.  The copy is extracted once per `origin/main` commit,
under the git dir, rather than once per conflicted file."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from research.clone import RECORD_DRIVER

LEAN = sorted(k for k, v in RECORD_DRIVER.items() if k.endswith(".driver") and "tools/verity/lean/merge.py" in v)

#: a stand-in for merge.py: writes, into OURS, which copy ran (its sibling module's `WHO`) and the mode it was given
STUB = ("import pathlib, sys\nsys.path.insert(0, str(pathlib.Path(__file__).parent))\nimport sibling\n"
        "pathlib.Path(sys.argv[-3]).write_text(' '.join([sibling.WHO, *sys.argv[1:-4]]) + '\\n')\n")


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert not check or r.returncode == 0, r.stderr
    return r


def repository(tmp_path: Path, monkeypatch, key: str, files: tuple[str, ...] = ("f.txt",)):
    """A repository on branch `old`, whose `files` conflict with `main`'s, the driver ``key`` set for them, and the stub as
    merge.py, its sibling saying `worktree` on `old` and `main` on `main`; and a function committing on the branch checked
    out (``who``: the sibling's `WHO` there)."""
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com", "TMPDIR": str(tmp_path / "tmp")}.items():
        monkeypatch.setenv(k, v)
    (tmp_path / "tmp").mkdir()
    repo, name = tmp_path / "repo", key.split(".")[1]
    (lean := repo / "tools" / "verity" / "lean").mkdir(parents=True)
    (lean / "merge.py").write_text(STUB)
    (repo / ".gitattributes").write_text("".join(f"{f} merge={name}\n" for f in files))

    def commit(text: str, who: str | None = None) -> None:
        if who:
            (lean / "sibling.py").write_text(f"WHO = {who!r}\n")
        for f in files:
            (repo / f).write_text(text)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", text)
    git(repo, "init", "-q", "-b", "main")
    commit("base\n", "worktree")
    git(repo, "checkout", "-q", "-b", "old")
    commit("old\n")
    git(repo, "checkout", "-q", "main")
    commit("main\n", "main")
    git(repo, "checkout", "-q", "old")
    git(repo, "config", key, RECORD_DRIVER[key])
    return repo, commit


def test_every_lean_driver_is_one_command_with_its_mode():
    assert [k.split(".")[1] for k in LEAN] == ["lakefile-lists", "lean-audit", "lean-caps", "lean-imports", "pyproject-test-lists"]
    plain = RECORD_DRIVER["merge.lean-audit.driver"]
    for k, flag in zip(LEAN, ("--lakefile ", "", "--caps ", "--lean ", "--pyproject ")):
        assert RECORD_DRIVER[k] == plain.replace("%O", f"{flag}%O")


@pytest.mark.parametrize("key", LEAN)
def test_a_lean_driver_runs_origin_main_s_copy_when_the_repository_has_it_and_the_worktree_s_otherwise(tmp_path, monkeypatch, key):
    repo, _ = repository(tmp_path, monkeypatch, key)
    flag = re.search(r'python3 "\$f" (.*?)%O', RECORD_DRIVER[key])[1]

    git(repo, "merge", "--no-edit", "main")
    assert (repo / "f.txt").read_text() == f"worktree {flag}".rstrip() + "\n"
    git(repo, "reset", "-q", "--hard", "ORIG_HEAD")

    git(repo, "update-ref", "refs/remotes/origin/main", "main")
    git(repo, "merge", "--no-edit", "main")
    assert (repo / "f.txt").read_text() == f"main {flag}".rstrip() + "\n"
    assert list((tmp_path / "tmp").iterdir()) == []
    git(repo, "reset", "-q", "--hard", "ORIG_HEAD")
    tree = git(repo, "merge-tree", "--write-tree", "old", "main").stdout.split()[0]
    assert git(repo, "show", f"{tree}:f.txt").stdout == f"main {flag}".rstrip() + "\n"

    (moved := tmp_path / "moved").mkdir()
    (moved / "merge.py").write_text(STUB)
    (moved / "sibling.py").write_text("WHO = 'moved'\n")
    git(repo, "config", key, RECORD_DRIVER[key].replace("m=tools/verity/lean/merge.py", f"m={moved}/merge.py", 1))
    git(repo, "merge", "--no-edit", "main")
    assert (repo / "f.txt").read_text() == f"moved {flag}".rstrip() + "\n"


def test_a_lean_driver_extracts_origin_main_s_copy_once_per_commit_and_keeps_the_two_newest(tmp_path, monkeypatch):
    """Two driver calls (one merge, two conflicted files) with one `origin/main` extract its copy once, into
    `.git/lean-merge/<sha>/`, and a later merge runs that copy; a moved `origin/main` extracts again, and only the two newest
    commits' copies stay.  A call that finds another's extraction lock taken runs a temporary copy of its own and leaves the
    cache to the lock's holder."""
    repo, commit = repository(tmp_path, monkeypatch, "merge.lean-audit.driver", ("f.txt", "g.txt"))
    (shim := tmp_path / "bin").mkdir()
    (shim / "tar").write_text(f'#!/bin/sh\necho "$3" >> {tmp_path / "archived"}\nexec {shutil.which("tar")} "$@"\n')
    (shim / "tar").chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")
    cache = repo / ".git" / "lean-merge"

    def archived() -> list[str]:
        """Each extraction (git puts its own directory first on a driver's PATH, so the shim is tar's): the commit whose copy
        went into the cache, or `temporary`."""
        dirs = (tmp_path / "archived").read_text().split() if (tmp_path / "archived").exists() else []
        return [Path(d).name if Path(d).resolve().parent == cache.resolve() else "temporary" for d in dirs]

    def merge(who: str) -> None:
        git(repo, "merge", "--no-edit", "main")
        assert [(repo / f).read_text() for f in ("f.txt", "g.txt")] == [f"{who}\n"] * 2
        git(repo, "reset", "-q", "--hard", "ORIG_HEAD")

    def move(who: str) -> str:
        git(repo, "checkout", "-q", "main")
        commit(f"{who}\n", who)
        git(repo, "update-ref", "refs/remotes/origin/main", "main")
        git(repo, "checkout", "-q", "old")
        return git(repo, "rev-parse", "main").stdout.strip()

    git(repo, "update-ref", "refs/remotes/origin/main", "main")
    first = git(repo, "rev-parse", "main").stdout.strip()
    merge("main")
    assert archived() == [first] and sorted(p.name for p in cache.iterdir()) == [first]
    (cache / first / "tools" / "verity" / "lean" / "sibling.py").write_text("WHO = 'cached'\n")
    merge("cached")
    assert archived() == [first], "the second merge runs the copy the first extracted"

    second = move("main2")
    merge("main2")
    os.utime(cache / first, (1, 1))
    os.utime(cache / second, (2, 2))
    third = move("main3")
    merge("main3")
    assert archived() == [first, second, third]
    assert sorted(p.name for p in cache.iterdir()) == sorted([second, third]), "the two newest stay, and no lock"

    fourth = move("main4")
    (cache / f"{fourth}.lock").mkdir()
    merge("main4")
    assert not (cache / fourth).exists() and (cache / f"{fourth}.lock").is_dir() and list((tmp_path / "tmp").iterdir()) == []
    assert archived() == [first, second, third, "temporary", "temporary"], "a copy of its own for each call while it's taken"
    (cache / f"{fourth}.lock").rmdir()
    merge("main4")
    assert archived() == [first, second, third, "temporary", "temporary", fourth]
    assert (cache / fourth / ".ok").is_file() and not (cache / f"{fourth}.lock").exists()
