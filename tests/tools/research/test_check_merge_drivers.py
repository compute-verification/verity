"""`RECORD_DRIVER`'s drivers for `artifacts.json` and Markdown tables run `merge_map.py` and `merge_rows.py` from the worktree
git merges into, at `tools/check/` or, on a branch cut before `tools/verity/check/` moved there, at the old place.  A branch
of the old layout that merges `main` runs the drivers in its own worktree, which holds no `tools/check/`: with the new path
alone both drivers failed, and every fixture map and Markdown table either side changed came out a conflict."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from research import clone

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ("merge_map.py", "merge_rows.py")
NEW, OLD = "tools/check", "tools/verity/check"
A, B, C = (f"art:{c * 64}" for c in "abc")
HEAD = "| Directory | What it measures |\n| --- | --- |\n"
ROW = {c: f"| `{c}/` | what {c} measures |\n" for c in "abc"}


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, capture_output=True, text=True)
    assert not check or r.returncode == 0, r.stderr
    return r


def crossing(tmp_path: Path, monkeypatch, layout: str | None) -> Path:
    """A repository on branch `old`, whose scripts are at ``layout`` (None: nowhere), checked out to merge `main`, which moved
    them to `tools/check/`; each side registered one more fixture and one more table row at the same place, and the drivers
    are registered as a run clone registers them (`clone.set_drivers`)."""
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_AUTHOR_NAME": "t",
                 "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitattributes").write_text("artifacts.json merge=fixture-map\n*.md merge=markdown-rows\n")

    def commit(at: str | None, fixtures: dict[str, str], rows: str) -> None:
        for d in (NEW, OLD):
            for s in SCRIPTS:
                if d == at:
                    (repo / d).mkdir(parents=True, exist_ok=True)
                    (repo / d / s).write_text((ROOT / NEW / s).read_text())
                else:
                    (repo / d / s).unlink(missing_ok=True)
        (repo / "artifacts.json").write_text(json.dumps(fixtures, indent=1, sort_keys=True) + "\n")
        (repo / "README.md").write_text(f"# Index\n\n{HEAD}{rows}\nThe rules follow.\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", at or "none")

    git(repo, "init", "-q", "-b", "main")
    commit(layout, {A: "f/a.json"}, ROW["a"])
    git(repo, "checkout", "-q", "-b", "old")
    commit(layout, {A: "f/a.json", B: "f/b.json"}, ROW["a"] + ROW["b"])
    git(repo, "checkout", "-q", "main")
    commit(NEW, {A: "f/a.json", C: "f/c.json"}, ROW["a"] + ROW["c"])
    git(repo, "checkout", "-q", "old")
    assert {"merge.fixture-map.driver", "merge.markdown-rows.driver"} <= set(clone.set_drivers(repo))
    return repo


@pytest.mark.parametrize("layout", [NEW, OLD])
def test_a_worktree_of_either_layout_merges_the_fixture_map_and_a_table_through_the_drivers(tmp_path, monkeypatch, layout):
    repo = crossing(tmp_path, monkeypatch, layout)
    r = git(repo, "merge", "--no-edit", "main", check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((repo / "artifacts.json").read_text()) == {A: "f/a.json", B: "f/b.json", C: "f/c.json"}
    assert (repo / "README.md").read_text() == f"# Index\n\n{HEAD}{ROW['a']}{ROW['b']}{ROW['c']}\nThe rules follow.\n"


def test_a_worktree_with_neither_script_leaves_both_files_conflicted_and_says_which_script_is_missing(tmp_path, monkeypatch):
    repo = crossing(tmp_path, monkeypatch, None)
    r = git(repo, "merge", "--no-edit", "main", check=False)
    assert r.returncode == 1
    assert git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split() == ["README.md", "artifacts.json"]
    assert "CONFLICT (content): Merge conflict in artifacts.json" in r.stdout
    assert f"{OLD}/merge_map.py" in r.stderr and f"{OLD}/merge_rows.py" in r.stderr
