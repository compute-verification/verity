"""`research queue open`: a PR for any pushed branch, through gh, refused when the branch isn't on origin, has moved from the
owner's pinned head, or already has an open PR.  A throwaway bare origin stands in for GitHub's git side and a recording fake
stands in for gh."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from research.queue import QueueError, broker_gh, open_pr


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(tmp_path, "init", "-q", "-b", "main", str(work))
    git(work, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "base")
    git(work, "remote", "add", "origin", str(origin))
    git(work, "push", "-q", "origin", "main", "HEAD:refs/heads/lead/done")
    return work, git(work, "rev-parse", "HEAD")


class FakeGh:
    """Answers `gh pr list` with ``existing`` and `gh pr create` with a URL, recording every call."""

    def __init__(self, existing=()):
        self.existing, self.calls = list(existing), []

    def __call__(self, argv, **kw):
        self.calls.append(argv[1:])
        out = json.dumps(self.existing) if argv[2] == "list" else "https://github.com/o/r/pull/7\n"
        return subprocess.CompletedProcess(argv, 0, out, "")


def test_it_opens_the_pr_with_the_exact_title_body_and_base(checkout):
    work, sha = checkout
    gh = FakeGh()
    url = open_pr(work, "o/r", "lead/done", "T", "Lane x\nbody", sha=sha[:9], gh="gh", run=gh)
    assert url == "https://github.com/o/r/pull/7"
    assert gh.calls[-1] == ["pr", "create", "--repo", "o/r", "--head", "lead/done", "--base", "main", "--title", "T",
                            "--body", "Lane x\nbody"]


def test_draft_is_off_unless_asked(checkout):
    work, _ = checkout
    gh = FakeGh()
    open_pr(work, "o/r", "lead/done", "T", "b", draft=True, gh="gh", run=gh)
    assert gh.calls[-1][-1] == "--draft"


def test_a_branch_not_on_origin_is_refused_before_gh_runs(checkout):
    work, _ = checkout
    gh = FakeGh()
    with pytest.raises(QueueError, match="no branch 'lead/missing' on origin"):
        open_pr(work, "o/r", "lead/missing", "T", "b", gh="gh", run=gh)
    assert gh.calls == []


def test_a_moved_head_is_refused(checkout):
    work, _ = checkout
    with pytest.raises(QueueError, match="the head moved"):
        open_pr(work, "o/r", "lead/done", "T", "b", sha="deadbeef", gh="gh", run=FakeGh())


def test_an_existing_open_pr_for_the_head_is_refused_and_nothing_is_created(checkout):
    work, _ = checkout
    gh = FakeGh(existing=[{"number": 5, "url": "https://github.com/o/r/pull/5"}])
    with pytest.raises(QueueError, match="already has PR #5"):
        open_pr(work, "o/r", "lead/done", "T", "b", gh="gh", run=gh)
    assert [c[1] for c in gh.calls] == ["list"]


def test_the_broker_wrapper_is_preferred_when_the_checkout_has_one(checkout):
    work, _ = checkout
    wrapper = work / ".git" / "verity-auth" / "bin" / "gh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("#!/bin/sh\n")
    assert broker_gh(work) == str(wrapper)
