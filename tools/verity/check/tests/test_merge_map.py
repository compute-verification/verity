"""`tools/verity/check/merge_map.py`: the fixture replica map merged key by key, as a function and as git's merge driver."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("merge_map_under_test", ROOT / "tools" / "verity" / "check" / "merge_map.py")
MM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MM)

A, B, C, D = (f"art:{c * 64}" for c in "abcd")


def test_two_prs_that_each_register_a_fixture_merge_and_a_key_set_two_ways_refuses():
    """12 Oct: #1870 ejected from `next` for a text conflict with #1847, each having added a key."""
    base = {A: "f/a.json"}
    merged, why = MM.merge(base, {A: "f/a.json", B: "f/b.json"}, {A: "f/a.json", C: "f/c.json"})
    assert why == [] and merged == {A: "f/a.json", B: "f/b.json", C: "f/c.json"}
    _, why = MM.merge(base, {**base, B: "f/b.json"}, {**base, B: "f/other.json"})
    assert why == [f"{B}: added on both sides, differently"]
    _, why = MM.merge(base, {}, {A: "f/moved.json"})
    assert why == [f"{A}: removed on one side and changed on the other"]


def test_a_refresh_on_each_side_merges_previous_by_path_and_two_ids_for_one_path_refuse():
    base = {A: "f/a.json", B: "f/b.json"}
    ours = {C: "f/a.json", B: "f/b.json", "previous": {C: [A]}}
    theirs = {A: "f/a.json", D: "f/b.json", "previous": {D: [B]}}
    merged, why = MM.merge(base, ours, theirs)
    assert why == [] and merged == {C: "f/a.json", D: "f/b.json", "previous": {C: [A], D: [B]}}
    _, why = MM.merge(base, ours, {D: "f/a.json", B: "f/b.json", "previous": {D: [A]}})
    assert why == [f"f/a.json: named by both {C} and {D}"]


def test_the_merged_map_is_written_as_refresh_fixtures_writes_it():
    from research.store.fixtures import dump_map

    doc = {"previous": {C: [A]}, C: "f/ä.json", B: "f/b.json"}
    assert MM.dump(doc) == dump_map(doc)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(repo.parent), "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=check, capture_output=True, text=True,
                          env=env)


def test_git_merges_two_branches_that_each_register_a_fixture_with_the_driver(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tools" / "verity" / "check").mkdir(parents=True)
    (repo / "tools" / "verity" / "check" / "merge_map.py").write_text((ROOT / "tools/verity/check/merge_map.py").read_text())
    (repo / ".gitattributes").write_text("fixtures/artifacts.json merge=fixture-map\n")
    m = repo / "fixtures" / "artifacts.json"
    m.parent.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "merge.fixture-map.driver", "python3 tools/verity/check/merge_map.py %O %A %B %P")
    for name, extra in (("main", {}), ("a", {B: "f/b.json"}), ("b", {C: "f/c.json"})):
        if name != "main":
            _git(repo, "checkout", "-q", "-b", name, "main")
        m.write_text(MM.dump({A: "f/a.json", **extra}))
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", name)
    _git(repo, "checkout", "-q", "a")
    _git(repo, "-c", "merge.fixture-map.driver=false", "merge", "--no-edit", "b", check=False)
    assert _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split() == ["fixtures/artifacts.json"]
    _git(repo, "merge", "--abort")
    _git(repo, "merge", "--no-edit", "b")
    assert json.loads(m.read_text()) == {A: "f/a.json", B: "f/b.json", C: "f/c.json"}
