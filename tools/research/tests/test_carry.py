"""`research.carry`: a new head carries its older head's land verdict only when everything between them is merges of trusted
commits that the queue's own merge reproduces, and regenerated locks or pins."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research import carry
from research.queue import lowrisk

ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_AUTHOR_DATE": "2026-11-05T00:00:00Z", "GIT_COMMITTER_DATE": "2026-11-05T00:00:00Z", "HOME": "/nonexistent",
       "GIT_CONFIG_NOSYSTEM": "1", "PATH": "/usr/bin:/bin"}


def git(repo: Path, *a: str) -> str:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *a], cwd=repo, env=ENV, check=True, capture_output=True,
                          text=True).stdout.strip()


def commit(repo: Path, files: dict[str, str], msg: str) -> str:
    for f, text in files.items():
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", msg)
    return git(repo, "rev-parse", "HEAD")


def merge(repo: Path, other: str, resolve: dict[str, str] | None = None) -> str:
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", "merge", "--no-ff", "-qm", f"Merge {other[:8]}", other], cwd=repo,
                       env=ENV, capture_output=True, text=True)
    if r.returncode:
        assert resolve, r.stdout + r.stderr
        return commit(repo, resolve, f"Merge {other[:8]}")
    if resolve:
        for f, text in resolve.items():
            (repo / f).write_text(text)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "--amend", "--no-edit")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q", "-b", "main")
    commit(tmp_path, {"a.py": "a\n", "b.py": "b\n", "Pkg/lean-audit.json": "{}\n", "docs/README.md": "r\n"}, "base")
    return tmp_path


def branch(repo: Path, name: str, at: str = "main") -> None:
    git(repo, "checkout", "-q", "-b", name, at)


def landed(repo: Path, *heads: str):
    return carry.trust(repo, [git(repo, "rev-parse", "main")], heads)


def test_a_merge_of_landed_main_carries_and_names_what_it_merged(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    git(repo, "checkout", "-q", "main")
    m = commit(repo, {"b.py": "b2\n"}, "landed")
    git(repo, "checkout", "-q", "pr")
    new = merge(repo, "main")
    d = carry.delta(repo, old, new, landed(repo))
    assert (d.kind, d.carries, d.parents) == ("merge-only", True, (m,))


def test_a_lock_conflict_resolved_any_way_carries_since_the_lean_check_regenerates_it(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n", "Pkg/lean-audit.json": '{"pr": 1}\n'}, "the PR")
    git(repo, "checkout", "-q", "main")
    commit(repo, {"b.py": "b2\n", "Pkg/lean-audit.json": '{"main": 1}\n'}, "landed")
    git(repo, "checkout", "-q", "pr")
    new = merge(repo, "main", {"Pkg/lean-audit.json": '{"main": 1, "pr": 1}\n', "a.py": "a2\n", "b.py": "b2\n"})
    regen = commit(repo, {"Pkg/lean-audit.json": '{"regenerated": 1}\n'}, "records")
    d = carry.delta(repo, old, regen, landed(repo))
    assert d.carries and d.generated == ("Pkg/lean-audit.json",)
    assert new != regen


def test_the_owners_own_commit_needs_a_verdict(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    new = commit(repo, {"b.py": "sneaked\n"}, "more")
    d = carry.delta(repo, old, new, landed(repo))
    assert (d.kind, d.carries) == ("authored", False) and "b.py" in d.why


def test_a_merge_that_edits_beyond_the_queues_merge_needs_a_verdict(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    git(repo, "checkout", "-q", "main")
    commit(repo, {"b.py": "b2\n"}, "landed")
    git(repo, "checkout", "-q", "pr")
    new = merge(repo, "main", {"b.py": "b2 and an edit\n"})
    assert carry.delta(repo, old, new, landed(repo)).kind == "edited-merge"


def test_a_conflict_resolved_by_hand_outside_generated_files_needs_a_verdict(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "pr\n"}, "the PR")
    git(repo, "checkout", "-q", "main")
    commit(repo, {"a.py": "main\n"}, "landed")
    git(repo, "checkout", "-q", "pr")
    new = merge(repo, "main", {"a.py": "both\n"})
    d = carry.delta(repo, old, new, landed(repo))
    assert (d.kind, d.carries) == ("hand-resolved", False) and "a.py" in d.why


def test_merging_another_prs_head_carries_only_once_that_head_is_trusted(repo: Path):
    branch(repo, "other")
    theirs = commit(repo, {"b.py": "other\n"}, "another PR")
    branch(repo, "pr", "main")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    new = merge(repo, "other")
    assert carry.delta(repo, old, new, landed(repo)).kind == "untrusted-parent"
    assert carry.delta(repo, old, new, landed(repo, theirs)).kind == "merge-only"
    branch(repo, "above", "pr")
    above = commit(repo, {"docs/README.md": "above\n"}, "a reviewed PR stacked on the new head")
    assert carry.delta(repo, old, new, landed(repo, above)).kind == "untrusted-parent"
    git(repo, "checkout", "-q", "other")
    newer = commit(repo, {"b.py": "other, re-headed\n"}, "the other PR moves on")
    assert carry.delta(repo, old, new, landed(repo, newer)).kind == "untrusted-parent"


def test_a_reviewed_pr_built_on_the_old_head_and_merged_back_carries(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    branch(repo, "dependant", "pr")
    theirs = commit(repo, {"b.py": "dependant\n"}, "a PR stacked on it")
    git(repo, "checkout", "-q", "pr")
    new = merge(repo, "dependant")
    assert carry.delta(repo, old, new, landed(repo)).kind == "untrusted-parent"
    assert carry.delta(repo, old, new, landed(repo, theirs)).kind in carry.CARRIES


def test_a_reviewed_pr_stacked_on_the_new_head_vouches_for_none_of_its_own_commits(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    new = commit(repo, {"b.py": "sneaked\n"}, "more")
    branch(repo, "above", "pr")
    above = commit(repo, {"docs/README.md": "above\n"}, "a reviewed PR stacked on the new head")
    assert carry.delta(repo, old, new, landed(repo, above)).kind == "authored"


def test_a_rebased_head_needs_a_verdict(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    git(repo, "checkout", "-q", "main")
    commit(repo, {"b.py": "b2\n"}, "landed")
    git(repo, "checkout", "-q", "pr")
    git(repo, "rebase", "-q", "main")
    new = git(repo, "rev-parse", "HEAD")
    assert carry.delta(repo, old, new, landed(repo)).kind == "rebased"


def test_a_docs_only_delta_carries_only_where_the_low_risk_lane_would_land_it(repo: Path):
    branch(repo, "pr")
    old = commit(repo, {"a.py": "a2\n"}, "the PR")
    new = commit(repo, {"docs/README.md": "trimmed\n"}, "trim")
    assert carry.delta(repo, old, new, landed(repo)).kind == "authored"
    d = carry.delta(repo, old, new, landed(repo), lowrisk=lambda fs: lowrisk(fs, ["*.py"]))
    assert (d.kind, d.carries) == ("lowrisk-delta", True)
    code = commit(repo, {"b.py": "b2\n"}, "code")
    assert carry.delta(repo, old, code, landed(repo), lowrisk=lambda fs: lowrisk(fs, ["*.py"])).kind == "authored"
