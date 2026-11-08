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


def shim(tmp_path: Path, monkeypatch, then: str = "") -> Path:
    """`tar` as a driver finds it (git puts its own directory first on a driver's PATH, so the shim is tar's): it logs the
    directory each extraction goes into, then runs ``then`` and the real tar; returns the log."""
    (bin_ := tmp_path / "bin").mkdir()
    log = tmp_path / "archived"
    (bin_ / "tar").write_text(f'#!/bin/sh\necho "$3" >> {log}\n{then}\nexec {shutil.which("tar")} "$@"\n')
    (bin_ / "tar").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    return log


def archived(log: Path, cache: Path) -> list[str]:
    """Each extraction in ``log``: `cache` for one into a directory beside the cache's keys, `temporary` for one elsewhere."""
    dirs = log.read_text().split() if log.exists() else []
    return ["cache" if Path(d).resolve().parent == cache.resolve() else "temporary" for d in dirs]


def kept(cache: Path) -> list[str]:
    return sorted(p.name for p in cache.iterdir())


def test_a_lean_driver_extracts_origin_main_s_copy_once_per_commit_and_keeps_the_two_newest(tmp_path, monkeypatch):
    """Two driver calls (one merge, two conflicted files) with one `origin/main` extract its copy once, at
    `.git/lean-merge/<full sha>/`, and a later merge runs that copy; a moved `origin/main` extracts again, and only the two
    newest commits' copies stay, with nothing else beside them."""
    repo, commit = repository(tmp_path, monkeypatch, "merge.lean-audit.driver", ("f.txt", "g.txt"))
    log, cache = shim(tmp_path, monkeypatch), repo / ".git" / "lean-merge"

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
    assert len(first) == 40
    merge("main")
    assert archived(log, cache) == ["cache"] and kept(cache) == [first]
    (cache / first / "tools" / "verity" / "lean" / "sibling.py").write_text("WHO = 'cached'\n")
    merge("cached")
    assert archived(log, cache) == ["cache"], "the second merge runs the copy the first extracted"

    second = move("main2")
    merge("main2")
    os.utime(cache / first, (1, 1))
    os.utime(cache / second, (2, 2))
    third = move("main3")
    merge("main3")
    assert archived(log, cache) == ["cache"] * 3 and kept(cache) == sorted([second, third]), "the two newest stay"
    assert list((tmp_path / "tmp").iterdir()) == []


def test_a_killed_extraction_leaves_nothing_at_its_key_and_the_next_call_extracts_it_whole(tmp_path, monkeypatch):
    """The driver killed while it extracts `origin/main`'s copy (here once tar has written it, before the rename) stops the
    merge and leaves nothing at `.git/lean-merge/<sha>/`, only its temporary directory and archive beside it, so no later
    call can run a part of a copy; the next merge extracts the copy whole and merges by it."""
    repo, _ = repository(tmp_path, monkeypatch, "merge.lean-audit.driver")
    once = tmp_path / "kill"
    once.touch()
    log = shim(tmp_path, monkeypatch, f'if [ -e {once} ]; then rm {once}; {shutil.which("tar")} "$@"; kill -KILL $PPID; exit 1; fi')
    cache = repo / ".git" / "lean-merge"
    git(repo, "update-ref", "refs/remotes/origin/main", "main")
    sha = git(repo, "rev-parse", "main").stdout.strip()

    killed = git(repo, "merge", "--no-edit", "main", check=False)
    assert killed.returncode and "died of signal 9" in killed.stderr and git(repo, "status", "--short").stdout == ""
    assert not (cache / sha).exists() and all(p.startswith(".tmp-") for p in kept(cache)), kept(cache)
    git(repo, "merge", "--no-edit", "main")
    assert (repo / "f.txt").read_text() == "main\n" and (cache / sha / "tools" / "verity" / "lean" / "merge.py").is_file()
    assert archived(log, cache) == ["cache", "cache"]


def test_two_merges_racing_for_one_key_both_merge_by_a_whole_copy_and_one_stays(tmp_path, monkeypatch):
    """Two merges in two worktrees of one repository, each finding no copy of one `origin/main` commit: the second runs
    whole while the first extracts (the shim runs it then, so the order is fixed), its rename puts its copy at the key, and
    the first's rename then finds the key taken, so the first removes its own and merges by the one there."""
    repo, _ = repository(tmp_path, monkeypatch, "merge.lean-audit.driver")
    git(repo, "branch", "old2", "old")
    git(repo, "worktree", "add", "-q", str(other := tmp_path / "other"), "old2")
    once = tmp_path / "race"
    once.touch()
    unset = "env -u GIT_DIR -u GIT_WORK_TREE -u GIT_INDEX_FILE -u GIT_PREFIX -u GIT_REFLOG_ACTION"
    log = shim(tmp_path, monkeypatch, f'if [ -e {once} ]; then rm {once}; {unset} git -C {other} merge -q --no-edit main '
                                      f'> {tmp_path / "second.log"} 2>&1 || echo failed >> {tmp_path / "second.log"}; fi')
    cache = repo / ".git" / "lean-merge"
    git(repo, "update-ref", "refs/remotes/origin/main", "main")
    sha = git(repo, "rev-parse", "main").stdout.strip()

    git(repo, "merge", "--no-edit", "main")
    assert "failed" not in (tmp_path / "second.log").read_text()
    assert (repo / "f.txt").read_text() == (other / "f.txt").read_text() == "main\n"
    assert archived(log, cache) == ["cache", "cache"] and kept(cache) == [sha], "both extracted, one copy stays, nothing else"
    assert {"merge.py", "sibling.py"} <= {p.name for p in (cache / sha / "tools" / "verity" / "lean").iterdir()}


def test_a_copy_of_origin_main_that_won_t_extract_leaves_the_file_to_git_s_conflict_not_to_the_worktree_s_copy(tmp_path,
                                                                                                           monkeypatch):
    """Where `origin/main` resolves, the worktree's copy never runs: its copy failing to extract, at the key and in a temporary
    directory, leaves git's conflict and says why, the file as the side merged into has it."""
    repo, _ = repository(tmp_path, monkeypatch, "merge.lean-audit.driver")
    log = shim(tmp_path, monkeypatch, "exit 1")
    cache = repo / ".git" / "lean-merge"
    git(repo, "update-ref", "refs/remotes/origin/main", "main")

    refused = git(repo, "merge", "--no-edit", "main", check=False)
    assert refused.returncode and "CONFLICT (content): Merge conflict in f.txt" in refused.stdout
    assert "resolves, but its tools/verity/lean could not be extracted" in refused.stderr, refused.stderr
    assert (repo / "f.txt").read_text() == "old\n", "not merged by the worktree's copy"
    assert archived(log, cache) == ["cache", "temporary"] and kept(cache) == [] and list((tmp_path / "tmp").iterdir()) == []


