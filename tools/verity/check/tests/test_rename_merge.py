"""`rename_merge.py`: a PR merged onto a tip that moved files it edits, resolved at the new paths or refused."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "tools" / "verity" / "check" / "rename_merge.py"
spec = importlib.util.spec_from_file_location("rename_merge_under_test", SCRIPT)
RM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RM)

ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
LINES = "".join(f"line {i}\n" for i in range(1, 11))


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


def write(repo: Path, files: dict[str, str | None]) -> None:
    for path, text in files.items():
        p = repo / path
        if text is None:
            p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)


def commit(repo: Path, msg: str, files: dict[str, str | None] = {}, moves: dict[str, str] = {}) -> str:
    write(repo, files)
    for old, new in moves.items():
        (repo / new).parent.mkdir(parents=True, exist_ok=True)
        git(repo, "mv", old, new)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def edit(text: str, n: int, new: str) -> str:
    lines = text.splitlines(keepends=True)
    lines[n - 1] = new + "\n"
    return "".join(lines)


@pytest.fixture
def repo(tmp_path):
    """A merge base with `old/a.py`, `old/b.py` and `keep.txt`, a branch `pr` off it, and `main` checked out."""
    git(tmp_path, "init", "-q", "-b", "main")
    commit(tmp_path, "base", {"old/a.py": LINES, "old/b.py": "b\n", "keep.txt": LINES})
    git(tmp_path, "branch", "pr")
    return tmp_path


def on(repo: Path, branch: str, *steps) -> str:
    git(repo, "checkout", "-q", branch)
    for msg, files, moves in steps:
        sha = commit(repo, msg, files, moves)
    git(repo, "checkout", "-q", "main")
    return sha


def blob(repo: Path, rev: str, path: str) -> str:
    return git(repo, "show", f"{rev}:{path}")


def files(repo: Path, rev: str) -> set[str]:
    return set(git(repo, "ls-tree", "-r", "--name-only", rev).splitlines())


MOVE = ("move old/ to new/", {}, {"old/a.py": "new/a.py", "old/b.py": "new/b.py"})


def test_a_merge_with_nothing_moved_is_the_ordinary_merge(repo):
    commit(repo, "tip edits keep", {"keep.txt": edit(LINES, 1, "tip")})
    pr = on(repo, "pr", ("pr edits a", {"old/a.py": edit(LINES, 9, "pr")}, {}))
    r = RM.rename_merge("main", "pr", "Merge #1 (pr)", cwd=str(repo))
    expected = git(repo, "merge-tree", "--write-tree", "main", "pr")
    assert r["resolutions"] == [] and r["tree"] == expected
    assert git(repo, "log", "-1", "--format=%P%n%B", r["commit"]).splitlines() == [f"{git(repo, 'rev-parse', 'main')} {pr}",
                                                                                   "Merge #1 (pr)"]


def test_an_edit_to_a_file_the_tip_moved_and_changed_lands_at_the_new_path(repo):
    tip = commit(repo, *MOVE)
    tip = commit(repo, "tip edits a after the move", {"new/a.py": edit(LINES, 1, "tip")})
    pr = on(repo, "pr", ("pr edits a", {"old/a.py": edit(LINES, 9, "pr")}, {}))
    r = RM.rename_merge("main", "pr", "Merge #7 (pr)", cwd=str(repo))
    assert r["resolutions"] == [{"old": "old/a.py", "new": "new/a.py", "change": "edited"}]
    assert blob(repo, r["commit"], "new/a.py") + "\n" == edit(edit(LINES, 1, "tip"), 9, "pr")
    assert files(repo, r["commit"]) == {"new/a.py", "new/b.py", "keep.txt"}
    assert git(repo, "log", "-1", "--format=%P", r["commit"]) == f"{tip} {pr}"
    assert git(repo, "log", "-1", "--format=%B", r["commit"]) == "Merge #7 (pr)\n\nRenamed-by-ci: old/a.py -> new/a.py"
    assert RM.summary(r).splitlines()[1:] == ["Renamed-by-ci: old/a.py -> new/a.py (edited)"]


def test_a_file_the_pr_adds_in_a_moved_directory_is_relocated(repo):
    commit(repo, *MOVE)
    on(repo, "pr", ("pr adds c", {"old/c.py": "c\n", "old/sub/d.py": "d\n"}, {}))
    r = RM.rename_merge("main", "pr", "Merge #2 (pr)", cwd=str(repo))
    assert [(x["old"], x["new"], x["change"]) for x in r["resolutions"]] == [
        ("old/c.py", "new/c.py", "added"), ("old/sub/d.py", "new/sub/d.py", "added")]
    assert files(repo, r["commit"]) == {"new/a.py", "new/b.py", "new/c.py", "new/sub/d.py", "keep.txt"}


def test_both_sides_editing_the_same_lines_after_the_move_is_refused(repo):
    commit(repo, *MOVE)
    commit(repo, "tip edits a", {"new/a.py": edit(LINES, 5, "tip")})
    on(repo, "pr", ("pr edits a", {"old/a.py": edit(LINES, 5, "pr")}, {}))
    refs = git(repo, "show-ref")
    with pytest.raises(RM.Refused) as e:
        RM.rename_merge("main", "pr", "Merge #3 (pr)", cwd=str(repo))
    assert [p for p, _ in e.value.refusals] == ["old/a.py"] and "new/a.py" in e.value.refusals[0][1]
    assert git(repo, "show-ref") == refs


def test_an_edit_to_a_file_the_tip_deleted_is_refused(repo):
    commit(repo, "tip deletes keep", {"keep.txt": None})
    on(repo, "pr", ("pr edits keep", {"keep.txt": edit(LINES, 2, "pr")}, {}))
    with pytest.raises(RM.Refused) as e:
        RM.rename_merge("main", "pr", "Merge #4 (pr)", cwd=str(repo))
    assert [p for p, _ in e.value.refusals] == ["keep.txt"]


def test_a_file_added_under_a_directory_split_two_ways_is_refused(repo):
    commit(repo, "split old/", {}, {"old/a.py": "x/a.py", "old/b.py": "y/b.py"})
    on(repo, "pr", ("pr adds c", {"old/c.py": "c\n"}, {}))
    with pytest.raises(RM.Refused) as e:
        RM.rename_merge("main", "pr", "Merge #5 (pr)", cwd=str(repo))
    assert e.value.refusals == [("old/c.py", "added under old/, which the tip moved to more than one directory")]


def test_a_directory_the_tip_only_partly_moved_is_no_rename(repo):
    commit(repo, "move a only", {}, {"old/a.py": "new/a.py"})
    on(repo, "pr", ("pr adds c", {"old/c.py": "c\n"}, {}))
    r = RM.rename_merge("main", "pr", "Merge #6 (pr)", cwd=str(repo))
    assert r["resolutions"] == [] and "old/c.py" in files(repo, r["commit"])


def test_a_deletion_of_a_file_moved_unchanged_follows_it(repo):
    commit(repo, *MOVE)
    on(repo, "pr", ("pr deletes b", {"old/b.py": None}, {}))
    r = RM.rename_merge("main", "pr", "Merge #8 (pr)", cwd=str(repo))
    assert r["resolutions"] == [{"old": "old/b.py", "new": "new/b.py", "change": "deleted"}]
    assert files(repo, r["commit"]) == {"new/a.py", "keep.txt"}


def test_an_explicit_map_covers_a_move_too_changed_to_detect(repo):
    rewritten = LINES
    for n in range(1, 7):
        rewritten = edit(rewritten, n, f"tip {n}")
    commit(repo, "move keep and rewrite most of it", {"keep.txt": None, "docs/keep.md": rewritten})
    on(repo, "pr", ("pr edits keep", {"keep.txt": edit(LINES, 9, "pr")}, {}))
    with pytest.raises(RM.Refused):
        RM.rename_merge("main", "pr", "m", cwd=str(repo))
    r = RM.rename_merge("main", "pr", "m", [("keep.txt", "docs/keep.md")], cwd=str(repo))
    assert blob(repo, r["commit"], "docs/keep.md") + "\n" == edit(rewritten, 9, "pr")


def test_the_cli_prints_json_and_fast_forwards_only_the_named_worktree(repo, tmp_path_factory):
    commit(repo, *MOVE)
    on(repo, "pr", ("pr edits a", {"old/a.py": edit(LINES, 9, "pr")}, {}))
    tip = git(repo, "rev-parse", "main")
    wt = tmp_path_factory.mktemp("wt") / "tip"
    git(repo, "worktree", "add", "-q", "--detach", str(wt), tip)
    run = lambda *a: subprocess.run(["python3", str(SCRIPT), *a], cwd=repo, capture_output=True, text=True)
    p = run("main", "pr", "-m", "Merge #9 (pr)", "--json")
    assert p.returncode == 0, p.stderr
    r = json.loads(p.stdout)
    assert r["status"] == "merged" and r["resolutions"][0]["new"] == "new/a.py"
    assert git(repo, "rev-parse", "HEAD") == tip and git(repo, "status", "--porcelain") == ""
    p = run("main", "pr", "-m", "Merge #9 (pr)", "--worktree", str(wt))
    assert p.returncode == 0, p.stderr
    assert git(wt, "rev-parse", "HEAD") == p.stdout.split(": ")[1].splitlines()[0]
    assert (wt / "new/a.py").read_text() == edit(LINES, 9, "pr") and git(repo, "rev-parse", "main") == tip


def test_the_cli_refuses_with_paths_and_exit_1(repo):
    commit(repo, "tip deletes keep", {"keep.txt": None})
    on(repo, "pr", ("pr edits keep", {"keep.txt": edit(LINES, 2, "pr")}, {}))
    p = subprocess.run(["python3", str(SCRIPT), "main", "pr", "-m", "m", "--json"], cwd=repo, capture_output=True, text=True)
    assert p.returncode == 1 and json.loads(p.stdout)["refusals"][0]["path"] == "keep.txt"
