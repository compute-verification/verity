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


def test_open_with_dry_run_sends_nothing(tmp_path, monkeypatch, capsys):
    """17 Oct (CI-52): `research queue open --dry-run` opened #2276 for real."""
    import subprocess

    from research import queue as Q
    from research.cli import main as research_main

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    body = tmp_path / "body.md"
    body.write_text("Lane ci\n")
    monkeypatch.setattr(Q, "open_pr", lambda *a, **k: (_ for _ in ()).throw(AssertionError("opened a PR on a dry run")))
    assert research_main(["queue", "open", "--dry-run", "--head", "cursor/x", "--title", "T", "--body-file", str(body),
                          "--remote", "https://github.com/o/verity.git"]) == 0
    assert "would open cursor/x onto main in o/verity" in capsys.readouterr().out


AGENT = "bc-d7be0308-38ba-5176-93d1-9169ce7d6cd3"
FOOTER = (f'<div><a href="https://cursor.com/agents/{AGENT}?cursor_ref=pr_footer&cursor_cta=open_in_web"><picture>'
          '<img alt="Open in Web" width="114" height="28"></picture></a></div>')
BEGIN, END = "<!-- CURSOR_AGENT_PR_BODY_BEGIN -->", "<!-- CURSOR_AGENT_PR_BODY_END -->"
#: #4622's body as it was: the one `open` gave it, then the PR tool's marked section below it, and the footer
STACKED = f"Lane memory-accounting\n\nthin lanes and a MAC cap\n\n{BEGIN}\nLane memory-accounting\n\nthe rule\n{END}\n\n{FOOTER}\n"


class ViewGh:
    """Answers `gh pr view` with ``body``, recording every call."""

    def __init__(self, body: str):
        self.body, self.calls = body, []

    def __call__(self, argv, **kw):
        self.calls.append(argv[1:])
        out = json.dumps({"body": self.body, "url": "https://github.com/o/r/pull/4622"}) if argv[2] == "view" else ""
        return subprocess.CompletedProcess(argv, 0, out, "")


def test_edit_replaces_the_whole_body_keeping_only_the_pr_tools_markers_and_footer(tmp_path):
    """TOP-90 row 5: #4491, #4622, #4514 and #4468 kept the body they were opened with above every later one."""
    from research.queue import PR, edit_pr

    gh = ViewGh(STACKED)
    url, body = edit_pr(tmp_path, "o/r", 4622, "Lane memory-accounting\n\nthe current text\n", gh="gh", run=gh)
    assert url == "https://github.com/o/r/pull/4622"
    assert body == f"{BEGIN}\nLane memory-accounting\n\nthe current text\n{END}\n\n{FOOTER}\n"
    assert gh.calls[-1] == ["pr", "edit", "4622", "--repo", "o/r", "--body", body]
    assert "thin lanes" not in body and PR(4622, "", body=body).owner == "memory-accounting"
    again = ViewGh(body)                                                     # an edit of an edited body stacks nothing
    assert edit_pr(tmp_path, "o/r", 4622, f"{STACKED}\n", gh="gh", run=again)[1].count(FOOTER) == 1


def test_edit_of_a_body_without_the_pr_tools_markup_is_the_file_verbatim(tmp_path):
    from research.queue import edit_pr

    gh = ViewGh("Lane ci\n\nold\n")
    assert edit_pr(tmp_path, "o/r", 7, "Lane ci\n\nnew\n", gh="gh", run=gh)[1] == "Lane ci\n\nnew\n"


def test_edit_refuses_a_body_that_drops_or_changes_the_prs_lane_and_dry_run_edits_nothing(tmp_path):
    from research.queue import edit_pr

    for new, says in (("no lane here\n", f"names owner {AGENT}"), ("Lane proofs\n", "names owner proofs")):
        gh = ViewGh(STACKED)
        with pytest.raises(QueueError, match=f"{says}, and the PR's is memory-accounting: keep its `Lane memory-accounting` line"):
            edit_pr(tmp_path, "o/r", 4622, new, gh="gh", run=gh)
        assert [c[1] for c in gh.calls] == ["view"]
    gh = ViewGh(f"[Open in Web](https://cursor.com/agents/{AGENT})\n")         # owned by its agent id: a Lane line may come in
    assert edit_pr(tmp_path, "o/r", 9, "Lane ci\n", gh="gh", run=gh)[1] == "Lane ci\n"
    gh = ViewGh(STACKED)
    edit_pr(tmp_path, "o/r", 4622, "Lane memory-accounting\n", dry_run=True, gh="gh", run=gh)
    assert [c[1] for c in gh.calls] == ["view"]
