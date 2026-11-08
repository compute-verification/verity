"""`RECORD_DRIVER`'s lean drivers run `origin/main`'s `tools/verity/lean/merge.py` when the repository has that ref, and the
worktree's otherwise.  Git runs a driver from the worktree of the side merged into, so an old branch that merged main ran its
own older copy, which collapsed Security's lock to one `reads` group."""
from __future__ import annotations

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


def test_every_lean_driver_is_one_command_with_its_mode():
    assert [k.split(".")[1] for k in LEAN] == ["lakefile-lists", "lean-audit", "lean-caps", "lean-imports", "pyproject-test-lists"]
    plain = RECORD_DRIVER["merge.lean-audit.driver"]
    for k, flag in zip(LEAN, ("--lakefile ", "", "--caps ", "--lean ", "--pyproject ")):
        assert RECORD_DRIVER[k] == plain.replace("%O", f"{flag}%O")


@pytest.mark.parametrize("key", LEAN)
def test_a_lean_driver_runs_origin_main_s_copy_when_the_repository_has_it_and_the_worktree_s_otherwise(tmp_path, monkeypatch, key):
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com", "TMPDIR": str(tmp_path / "tmp")}.items():
        monkeypatch.setenv(k, v)
    (tmp_path / "tmp").mkdir()
    repo, name = tmp_path / "repo", key.split(".")[1]
    (lean := repo / "tools" / "verity" / "lean").mkdir(parents=True)
    (lean / "merge.py").write_text(STUB)
    (repo / ".gitattributes").write_text(f"f.txt merge={name}\n")

    def commit(text: str, who: str | None = None) -> None:
        if who:
            (lean / "sibling.py").write_text(f"WHO = {who!r}\n")
        (repo / "f.txt").write_text(text)
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
    flag = RECORD_DRIVER[key].split('python3 "$f" ', 1)[1].removesuffix("%O %A %B %P; r=$?; rm -rf \"$d\"; exit $r")

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
