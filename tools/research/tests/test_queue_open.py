"""`research queue open`: a PR for any pushed branch, through gh, refused when the branch isn't on origin, has moved from the
owner's pinned head, or already has an open PR, and linked to its Linear issue through the Linear app.  A throwaway bare
origin stands in for GitHub's git side, a recording fake for gh, and another for Linear's API."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from research.queue import PR, QueueError, broker_gh, open_pr


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


class FakeLinear:
    """Linear's token endpoint and GraphQL API as `open` and `link` use them: the ``issues`` it has (key to id), every query
    sent with its variables (`sent`, shared with the stubbed `open_pr` through ``log``), and ``fail``, the error
    attachmentLinkGitHubPR answers with instead of linking, refusing a URL an issue has already as Linear does
    (`attached`: URL to issue keys); `app` is the real `Linear` class."""

    def __init__(self, log: list, fail: str | None = None):
        self.issues, self.fail, self.sent = {"ENG-123": "uuid-eng-123", "ENG-5": "uuid-eng-5"}, fail, log
        self.attached: dict[str, list[str]] = {}

    def __call__(self, url: str, data: bytes, headers: dict) -> dict:
        from research import linear
        if url == linear.TOKEN_URL:
            return {"access_token": "tok", "expires_in": 86400}
        q, v = json.loads(data)["query"], json.loads(data)["variables"]
        self.sent.append(("gql", q, v))
        if "issue(id: $id) { id }" in q:
            if v["id"] not in self.issues:
                return {"errors": [{"message": "Entity not found: Issue"}], "data": None}
            return {"data": {"issue": {"id": self.issues[v["id"]]}}}
        if "attachmentLinkGitHubPR(issueId: $i, url: $u)" in q:
            if self.fail or v["u"] in self.attached:
                return {"errors": [{"message": self.fail or "Duplicate attachment for duplicate url"}], "data": None}
            self.attached[v["u"]] = [k for k, i in self.issues.items() if i == v["i"]]
            return {"data": {"attachmentLinkGitHubPR": {"success": True, "attachment": {"id": "A1", "url": v["u"]}}}}
        if "attachmentsForURL(url: $u)" in q:
            return {"data": {"attachmentsForURL": {"nodes": [{"issue": {"id": self.issues[k], "identifier": k}}
                                                             for k in self.attached.get(v["u"], [])]}}}
        raise AssertionError(q)


@pytest.fixture
def opening(tmp_path, monkeypatch):
    """A git checkout to run `research queue open` in, a body file, `open_pr` stubbed to log what it would send and answer
    PR 7's URL, and the Linear app on a `FakeLinear` logging to the same list: (log, fake, body, argv without --issue)."""
    from research import linear
    from research import queue as Q

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    body = tmp_path / "body.md"
    body.write_text("Lane ci\n\nthe change\n")
    log: list = []
    fake = FakeLinear(log)
    fake.app = linear.Linear
    monkeypatch.setattr(linear, "Linear", lambda: fake.app({"LINEAR_CLIENT_ID": "dummy-client", "LINEAR_CLIENT_SECRET": "dummy-secret"},
                                                           fake, cache=tmp_path / "tok"))
    monkeypatch.setattr(Q, "open_pr", lambda top, repo, head, title, text, **k:
                        log.append(("open", repo, text)) or f"https://github.com/{repo}/pull/7")
    argv = ["queue", "open", "--head", "cursor/x", "--title", "T", "--body-file", str(body), "--remote",
            "https://github.com/o/verity.git"]
    return log, fake, body, argv


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


def test_open_with_dry_run_sends_nothing(opening, capsys):
    """`research queue open --dry-run` opens no PR for real and links nothing; it only reads the issue's id."""
    from research.cli import main as research_main

    log, _, _, argv = opening
    assert research_main([*argv, "--dry-run", "--issue", "ENG-123"]) == 0
    out = capsys.readouterr().out
    assert "would open cursor/x onto main in o/verity" in out and "ending 'Refs ENG-123'" in out
    assert "would link it to ENG-123 through the Linear app" in out
    assert [e[0] for e in log] == ["gql"] and "attachmentLinkGitHubPR" not in log[0][1]


def test_the_issue_line_is_linears_non_closing_magic_word_or_a_reason_and_nothing_else():
    from research.queue import issue_line, named_issue, with_issue

    assert issue_line("ENG-123") == "Refs ENG-123" and issue_line("A1B-7") == "Refs A1B-7"
    assert issue_line("none: a typo fix") == "Issue: none: a typo fix"
    for bad in ("eng-123", "ENG123", "ENG-", "1ENG-3", "ENG-12x", "Refs ENG-123", ""):
        with pytest.raises(QueueError, match=f"--issue {bad!r} is neither a Linear key"):
            issue_line(bad)
    with pytest.raises(QueueError, match="none:REASON needs the reason"):
        issue_line("none:  ")
    body = with_issue("Lane ci\n\nthe change\n", "ENG-123")
    assert body == "Lane ci\n\nthe change\n\nRefs ENG-123\n" and with_issue(body, "ENG-123") == body
    assert named_issue(body) == "ENG-123" and PR(1, "", body=body).owner == "ci"
    assert named_issue(with_issue("Lane ci\n", "none:docs only")) == "none: docs only"


def test_open_needs_an_issue_and_refuses_a_bad_one_before_anything_is_sent(tmp_path, monkeypatch, capsys):
    from research import queue as Q
    from research.cli import main as research_main

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    body = tmp_path / "body.md"
    body.write_text("Lane ci\n")
    monkeypatch.setattr(Q, "open_pr", lambda *a, **k: (_ for _ in ()).throw(AssertionError("opened a PR")))
    argv = ["queue", "open", "--head", "cursor/x", "--title", "T", "--body-file", str(body), "--remote",
            "https://github.com/o/verity.git"]
    with pytest.raises(SystemExit) as e:
        research_main(argv)
    assert e.value.code == 2 and "--issue KEY (the PR's Linear issue" in capsys.readouterr().err
    assert research_main([*argv, "--issue", "eng-1"]) == 1
    assert "'eng-1' is neither a Linear key" in capsys.readouterr().err
    assert research_main([*argv, "--issue", "none:"]) == 1
    assert "none:REASON needs the reason" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        research_main(["queue", "status", "--issue", "ENG-1"])
    assert "--issue is for open and link only" in capsys.readouterr().err


def test_open_resolves_the_key_opens_the_pr_with_its_refs_line_then_links_it_through_the_linear_app(opening, capsys):
    from research.cli import main as research_main

    log, _, _, argv = opening
    assert research_main([*argv, "--issue", "ENG-123"]) == 0
    url = "https://github.com/o/verity/pull/7"
    (k1, resolve, got), opened, (k3, mutation, link) = log
    assert (k1, got) == ("gql", {"id": "ENG-123"}) and "issue(id: $id) { id }" in resolve
    assert opened == ("open", "o/verity", "Lane ci\n\nthe change\n\nRefs ENG-123\n")
    assert k3 == "gql" and "attachmentLinkGitHubPR(issueId: $i, url: $u)" in mutation and link == {"i": "uuid-eng-123", "u": url}
    assert capsys.readouterr().out.splitlines() == [url, f"research queue open: {url} linked to ENG-123 through the Linear app"]


def test_a_key_linear_has_no_issue_for_opens_nothing_and_none_links_nothing(opening, capsys):
    from research.cli import main as research_main

    log, _, _, argv = opening
    assert research_main([*argv, "--issue", "ENG-9"]) == 1
    assert [e[0] for e in log] == ["gql"]
    assert "--issue ENG-9: Linear has no issue ENG-9 that app (client credentials) can read" in capsys.readouterr().err
    log.clear()
    assert research_main([*argv, "--issue", "none:docs only"]) == 0
    assert [e[:2] for e in log] == [("open", "o/verity")] and log[0][2].endswith("\n\nIssue: none: docs only\n")


@pytest.mark.parametrize("why", ["link", "credentials"])
def test_a_link_that_fails_leaves_the_pr_open_and_prints_why_and_the_command_that_links_it(opening, monkeypatch, capsys, why):
    from research import linear
    from research.cli import main as research_main

    log, fake, _, argv = opening
    if why == "link":
        fake.fail = "the workspace has no GitHub integration"
    else:
        monkeypatch.setattr(linear, "Linear", lambda: fake.app({}, fake))
    assert research_main([*argv, "--issue", "ENG-123"]) == 0
    out, err = capsys.readouterr()
    assert out.splitlines()[0] == "https://github.com/o/verity/pull/7" and ("open", "o/verity") in [e[:2] for e in log]
    assert ("the workspace has no GitHub integration" if why == "link" else "no Linear credentials") in err
    assert "is not linked to ENG-123 in Linear" in err and "link it with `research queue link 7 --issue ENG-123`" in err


def test_link_links_an_open_pr_and_fails_on_an_unknown_key_or_a_refused_link(opening, capsys):
    from research.cli import main as research_main

    log, fake, _, _ = opening
    link = ["queue", "link", "6859", "--repo", "o/verity", "--remote", "https://github.com/o/verity.git"]
    assert research_main([*link, "--issue", "ENG-123"]) == 0
    assert log[-1][2] == {"i": "uuid-eng-123", "u": "https://github.com/o/verity/pull/6859"}
    assert "pull/6859 linked to ENG-123 through the Linear app" in capsys.readouterr().out
    assert research_main([*link, "--issue", "ENG-123"]) == 0
    assert "pull/6859 was linked already to ENG-123" in capsys.readouterr().out
    assert research_main([*link, "--issue", "ENG-5"]) == 1
    assert "(pull/6859 is attached to ENG-123 already)" in capsys.readouterr().err.replace("https://github.com/o/verity/", "")
    assert research_main([*link, "--issue", "ENG-9"]) == 1
    assert "Linear has no issue ENG-9" in capsys.readouterr().err
    fake.fail = "refused"
    assert research_main([*link, "--issue", "ENG-123"]) == 1
    assert "(Linear: refused); link it with `research queue link 6859 --issue ENG-123`" in capsys.readouterr().err
    for bad in (["--issue", "none:x"], [], ["--issue", "ENG-1", "--dry-run"]):
        with pytest.raises(SystemExit):
            research_main([*link, *bad])
        assert "link needs the PR number and --issue KEY" in capsys.readouterr().err


AGENT = "bc-d7be0308-38ba-5176-93d1-9169ce7d6cd3"
FOOTER = (f'<div><a href="https://cursor.com/agents/{AGENT}?cursor_ref=pr_footer&cursor_cta=open_in_web"><picture>'
          '<img alt="Open in Web" width="114" height="28"></picture></a></div>')
BEGIN, END = "<!-- CURSOR_AGENT_PR_BODY_BEGIN -->", "<!-- CURSOR_AGENT_PR_BODY_END -->"
#: a stacked PR's body: the one `open` gave it, then the PR tool's marked section below it, and the footer
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
    """An edited PR doesn't keep the body it was opened with above every later one."""
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
