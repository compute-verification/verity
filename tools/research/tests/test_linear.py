from __future__ import annotations

import io
import json
import re
import stat
import urllib.error
import urllib.parse
from datetime import UTC, datetime

import pytest

from research import linear, slack
from test_slack import COORD, bells, fake, inbox, registry, routes_dirs, run  # noqa: F401  (Slack's fixtures)

APP = {"LINEAR_CLIENT_ID": "dummy-client", "LINEAR_CLIENT_SECRET": "dummy-secret"}
KEY = {"LINEAR_API_KEY": "dummy-key"}


MILESTONES = {"nodes": [{"id": "MS1", "name": "M1 · Mon 8 AM PDT · walking skeleton"}, {"id": "MS3", "name": "M3 · composed"},
                        {"id": "MS10", "name": "M10 · later"}]}
PROJECTS = [{"id": "PJ1", "name": "E2E toy deployment on one H200", "projectMilestones": MILESTONES},
            {"id": "PJ2", "name": "Full proofs: one soundness theorem", "projectMilestones": {"nodes": [{"id": "F2", "name": "F2 · Wed"}]}},
            {"id": "PJ3", "name": "Full stack", "projectMilestones": {"nodes": []}}]


class FakeLinear:
    """Linear's token endpoint and GraphQL API: one team per name in `teams`, labels made on first use; the parent CMP-92 on
    PJ1's M3, the active `projects`, the open issues the audit pages through (`open`), the team keys the credentials see
    (`visible`, default the teams'), the team of the issue a comment reads (`team_key`), the issues' `relations` (an
    issue's id is `I-` and its identifier), the workspace's `users`, and each relation, project or milestone mutation as
    it was sent (`wrote`)."""

    def __init__(self, teams=("infra", "comms")):
        self.teams, self.labels, self.calls, self.grants, self.refuse = list(teams), {}, [], 0, set()
        self.subscribers, self.authors, self.comments, self.moved = {}, {}, [], []
        self.state_type = "backlog"
        self.projects, self.open, self.updated, self.refuse_project = PROJECTS, [], [], False
        self.visible, self.team_key = None, "INF"
        self.parent = {"id": "P1", "project": {"id": "PJ1"}, "projectMilestone": {"id": "MS3"}}
        self.relations, self.wrote = [], []
        self.users = [{"id": "U1", "name": "Daniel Reuter"}, {"id": "U2", "name": "Verity Asks"}]

    @staticmethod
    def _issue(x: str) -> dict:
        ident = x[2:] if x.startswith("I-") else x.upper()
        return {"id": f"I-{ident}", "identifier": ident}

    @staticmethod
    def _author(user: str, i: dict, field: str) -> dict:
        if "createAsUser" in i:
            return {field: None, "botActor": {"name": "Verity Asks", "userDisplayName": i["createAsUser"]}}
        return {field: {"name": user}, "botActor": None}

    def __call__(self, url: str, data: bytes, headers: dict) -> dict:
        if url == linear.TOKEN_URL:
            self.grants += 1
            self.calls.append(("grant", urllib.parse.parse_qs(data.decode()), headers))
            return {"access_token": f"tok{self.grants}", "token_type": "Bearer", "expires_in": 30 * 86400, "scope": "read write"}
        req = json.loads(data)
        q, v = req["query"], req["variables"]
        assert q.count("{") == q.count("}") and q.count("(") == q.count(")"), f"Linear would refuse this query's syntax: {q}"
        self.calls.append((q, v, headers))
        if headers["Authorization"] in self.refuse:
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, io.BytesIO(b'{"errors": [{"message": "auth"}]}'))
        user = "the app" if headers["Authorization"].startswith("Bearer") else "Daniel"
        if m := re.search(r"\{ (issueRelationCreate|issueRelationDelete|projectUpdateCreate|projectCreate|projectMilestoneCreate|"
                          r"projectMilestoneUpdate)\(", q):
            self.wrote.append((m[1], v))
            if m[1] == "issueRelationCreate":
                self.relations.append({"id": f"R{len(self.wrote)}", "type": v["i"]["type"], "issue": self._issue(v["i"]["issueId"]),
                                       "relatedIssue": self._issue(v["i"]["relatedIssueId"])})
            if m[1] == "issueRelationDelete":
                self.relations = [r for r in self.relations if r["id"] != v["id"]]
            return {"data": {m[1]: {"success": True, "project": {"url": "https://linear.app/x/project/1"},
                                    "projectUpdate": {"url": "https://linear.app/x/update/1"}}}}
        if "inverseRelations" in q:
            a = self._issue(v["a"])
            return {"data": {"a": {**a, "relations": {"nodes": [r for r in self.relations if r["issue"] == a]},
                                   "inverseRelations": {"nodes": [r for r in self.relations if r["relatedIssue"] == a]}},
                             "b": self._issue(v["b"])}}
        if "projects(filter" in q:
            return {"data": {"projects": {"nodes": [{"name": p["name"]} for p in self.projects if p["name"].lower() == v["n"].lower()]}}}
        if "users(first" in q:
            return {"data": {"users": {"nodes": self.users}}}
        if q.startswith("{ teams(first"):
            nodes = ([{"key": k, "name": k.lower()} for k in self.visible] if self.visible is not None
                     else [{"key": t[:3].upper(), "name": t} for t in self.teams])
            return {"data": {"teams": {"nodes": nodes}}}
        if "teams(filter" in q:
            team = next((t for t in self.teams if v["n"] in (t, t[:3].upper())), None)
            return {"data": {"teams": {"nodes": [{"id": f"T-{team}", "key": team[:3].upper(), "name": team}] if team else []}}}
        if "state { type }" in q:
            return {"data": {"issue": {"state": {"type": self.state_type}, "comments": {"nodes": self.comments}}}}
        if "botActor" in q:
            return {"data": {"issue": {**self.authors["I1"], "comments": {"nodes": self.comments}}}}
        if "issueLabels" in q:
            return {"data": {"issueLabels": {"nodes": [{"id": self.labels[v["n"]]}] if v["n"] in self.labels else []}}}
        if "issueLabelCreate" in q:
            self.labels[v["i"]["name"]] = f"L{len(self.labels)}"
            return {"data": {"issueLabelCreate": {"issueLabel": {"id": self.labels[v["i"]["name"]]}}}}
        if "issueCreate" in q:
            if self.refuse_project and "projectId" in v["i"]:
                return {"errors": [{"message": "Argument Validation Error: the team is not in the project"}]}
            self.subscribers["I1"] = [user]
            self.authors["I1"] = self._author(user, v["i"], "creator")
            return {"data": {"issueCreate": {"issue": {"id": "I1", "identifier": "INF-1", "url": "https://linear.app/x/issue/INF-1"}}}}
        if "commentCreate" in q:
            self.comments.append(self._author(user, v["i"], "user"))
            self.subscribers["I1"] = sorted({*self.subscribers["I1"], user})
            return {"data": {"commentCreate": {"success": True}}}
        if "issueUnsubscribe" in q:
            self.subscribers[v["id"]].remove(user)
            return {"data": {"issueUnsubscribe": {"success": True}}}
        if "subscribers" in q:
            return {"data": {"issue": {"subscribers": {"nodes": [{"name": n} for n in self.subscribers[v["id"]]]}}}}
        if "states" in q:
            return {"data": {"issue": {"id": "I1", "team": {"states": {"nodes": [{"id": "S1", "name": "Todo", "type": "unstarted"},
                                                                    {"id": "S2", "name": "In Review", "type": "started"},
                                                                    {"id": "S3", "name": "In Progress", "type": "started"},
                                                                    {"id": "S5", "name": "Done", "type": "completed"},
                                                                    {"id": "S9", "name": "Canceled", "type": "canceled"}]}}}}}
        if "issue(id: $id) { id project" in q:
            return {"data": {"issue": self.parent}}
        if "projects(first" in q:
            return {"data": {"projects": {"nodes": self.projects}}}
        if "IssueUpdateInput" in q:
            self.updated.append((v["id"], v["i"]))
            return {"data": {"issueUpdate": {"success": True}}}
        if "pageInfo" in q:
            self.audited, start = v["f"], int(v["after"] or 0)
            page = self.open[start:start + int(re.search(r"issues\(filter: \$f, first: (\d+)", q)[1])]
            more = start + len(page) < len(self.open)
            return {"data": {"issues": {"pageInfo": {"hasNextPage": more, "endCursor": str(start + len(page))}, "nodes": page}}}
        if "issues(filter" in q:
            self.listed = v["f"]
            return {"data": {"issues": {"nodes": [
                {"identifier": "INF-2", "title": "later", "createdAt": "2026-10-14T02:00Z", "state": {"name": "In Progress"}},
                {"identifier": "INF-1", "title": "first", "createdAt": "2026-10-14T01:00Z", "state": {"name": "Backlog"}}]}}}
        if "team { key }" in q:
            return {"data": {"issue": {"id": "I1", "identifier": "INF-1", "url": "https://linear.app/x/issue/INF-1",
                                       "team": {"key": self.team_key}}}}
        if "issue(id: $id) { id }" in q:
            return {"data": {"issue": {"id": "P1" if v["id"] == "CMP-92" else "I1"}}}
        if "issueUpdate" in q:
            self.moved.append((v["id"], v["s"]))
            return {"data": {"issueUpdate": {"success": True}}}
        if "webhooks" in q:
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, io.BytesIO(b'{"errors": [{"message": "Invalid scope: `admin` required"}]}'))
        if "viewer" in q:
            return {"data": {"viewer": {"name": user}, "organization": {"urlKey": "verity-research"},
                             "teams": {"nodes": [{"name": t, "key": t[:3].upper()} for t in self.teams]}}}
        raise AssertionError(q)

    def sent(self, word: str) -> list:
        return [(q, v, h) for q, v, h in self.calls if q != "grant" and word in q]


def test_the_app_files_an_ask_as_the_sender_with_its_label_and_never_assigns_it(tmp_path):
    fake = FakeLinear()
    issue = linear.Linear(APP, fake, cache=tmp_path / "tok").file("infra", "compute-accounting", "4 GPUs on node 2\nfor 6 h")
    assert issue["identifier"] == "INF-1"
    (_, v, h), = fake.sent("issueCreate")
    assert v["i"] == {"teamId": "T-infra", "title": "4 GPUs on node 2", "description": "4 GPUs on node 2\nfor 6 h",
                      "labelIds": ["L0"], "createAsUser": "compute-accounting"}                 # no assignee, delegate or subscribers
    assert fake.labels == {"from:compute-accounting": "L0"} and h["Authorization"] == "Bearer tok1"
    assert fake.subscribers["I1"] == []


def test_the_client_credentials_token_is_cached_private_until_near_expiry_and_refetched_when_refused(tmp_path):
    fake, cache, t = FakeLinear(), tmp_path / "tok", [1000.0]
    linear.Linear(APP, fake, now=lambda: t[0], cache=cache).team("infra")
    (_, form, headers), = [c for c in fake.calls if c[0] == "grant"]
    assert form == {"grant_type": ["client_credentials"], "scope": ["read,write"]}
    assert headers["Authorization"].startswith("Basic ") and stat.S_IMODE(cache.stat().st_mode) == 0o600
    linear.Linear(APP, fake, now=lambda: t[0], cache=cache).team("infra")                    # another process: the cache
    assert fake.grants == 1
    t[0] += 30 * 86400 - linear.EARLY_S + 1                                                    # within the hour before expiry
    linear.Linear(APP, fake, now=lambda: t[0], cache=cache).team("infra")
    assert fake.grants == 2
    fake.refuse.add("Bearer tok2")                                                             # revoked: one fresh grant, one retry
    linear.Linear(APP, fake, now=lambda: t[0], cache=cache).team("infra")
    assert fake.grants == 3 and fake.calls[-1][2]["Authorization"] == "Bearer tok3"
    other = {**APP, "LINEAR_CLIENT_ID": "dummy-other-app"}                                     # another app never reads this one's
    linear.Linear(other, fake, now=lambda: t[0], cache=cache).team("infra")
    assert fake.grants == 4


def test_a_personal_key_unsubscribes_its_owner_after_filing_and_after_each_comment(tmp_path):
    fake = FakeLinear()
    lin = linear.Linear(KEY, fake, cache=tmp_path / "tok")
    issue = lin.file("infra", "comms", "x")
    assert fake.subscribers["I1"] == []
    lin.comment(issue["id"], "comms", "a reply")
    assert fake.subscribers["I1"] == [] and fake.grants == 0
    (_, v, h), = fake.sent("commentCreate")
    assert v["i"] == {"issueId": "I1", "body": "@comms: a reply"} and h["Authorization"] == "dummy-key"
    assert "createAsUser" not in fake.sent("issueCreate")[0][1]["i"]


def test_the_app_is_preferred_and_no_credentials_is_refused(tmp_path):
    assert linear.Linear({**APP, **KEY}, FakeLinear(), cache=tmp_path / "t").app
    assert not linear.Linear({**KEY, "LINEAR_CLIENT_ID": "dummy-client"}, FakeLinear(), cache=tmp_path / "t").app
    with pytest.raises(linear.LinearError, match="no Linear credentials"):
        linear.Linear({}, FakeLinear())


def test_check_reports_reach_and_admin_and_exercises_the_write_path(tmp_path):
    lines, ok = linear.Linear(KEY, FakeLinear(), cache=tmp_path / "t").check("infra")
    assert ok and lines[0] == "credentials: personal key, acting as Daniel in verity-research"
    assert lines[1] == "teams (2): comms COM, infra INF" and lines[2].startswith("admin: none (Linear: HTTP 403: Invalid scope")
    assert lines[3:] == ["filed INF-1 https://linear.app/x/issue/INF-1 as Daniel (subscribers: none)",
                         "commented as Daniel (subscribers: none)", "moved to Canceled"]


def test_check_with_the_app_names_a_team_by_key_and_shows_the_sender_as_the_author(tmp_path):
    lines, ok = linear.Linear(APP, FakeLinear(), cache=tmp_path / "t").check("INF", "compute-accounting")
    assert ok and lines[0] == "credentials: app (client credentials), acting as the app in verity-research"
    assert lines[3:5] == ["filed INF-1 https://linear.app/x/issue/INF-1 as compute-accounting (via Verity Asks) (subscribers: none)",
                          "commented as compute-accounting (via Verity Asks) (subscribers: none)"]


LINKED = "\nLinear: <https://linear.app/x/issue/INF-1|INF-1>"


@pytest.fixture
def lin(monkeypatch):
    fake_linear = FakeLinear()
    monkeypatch.setattr(linear, "_http", fake_linear)
    monkeypatch.setenv("LINEAR_API_KEY", KEY["LINEAR_API_KEY"])
    return fake_linear


def test_an_ask_files_in_its_owners_team_and_still_goes_out_when_linear_fails(fake, registry, capsys, lin):
    assert slack.ASKS_ON_LINEAR
    rc, out, _ = run(capsys, "ask", "--as", "compute-accounting", "--to", "@infra", "--text", "4 GPUs on node 2")
    assert rc == 0 and out.startswith("linear: INF-1 https://linear.app/x/issue/INF-1")
    assert fake.messages[COORD][-1]["text"].endswith("4 GPUs on node 2" + LINKED)
    assert lin.sent("teams(filter")[0][1] == {"n": "INF"}                                      # the registry's key, not the handle
    rc, _, _ = run(capsys, "ask", "--as", "compute-accounting", "--to", "@infra", "--text", "4 GPUs for 6 h\nspec: the panel\nby 8 PM")
    assert rc == 0 and fake.messages[COORD][-1]["text"].endswith("> 4 GPUs for 6 h" + LINKED)              # Slack: one line and the link
    assert lin.sent("issueCreate")[-1][1]["i"]["description"] == "4 GPUs for 6 h\nspec: the panel\nby 8 PM"  # the issue: the whole brief
    lin.teams = []
    rc, _, err = run(capsys, "ask", "--as", "compute-accounting", "--to", "@infra", "--text", "again")
    assert rc == 0 and "not filed on Linear (no Linear team named 'INF')" in err
    assert fake.messages[COORD][-1]["text"].endswith("again")
    n = len(lin.calls)
    assert run(capsys, "ask", "--as", "infra", "--to", "@console", "--text", "no team")[0] == 0
    assert len(lin.calls) == n and fake.messages[COORD][-1]["text"].endswith("no team")


def test_a_top_level_send_to_one_handle_is_an_ask_and_files_but_a_reply_or_a_second_name_does_not(fake, registry, capsys, lin):
    rc, out, _ = run(capsys, "send", "--as", "compute-accounting", "--to", "@infra", "--text", "4 GPUs on node 2")
    assert rc == 0 and slack.plain(fake.messages[COORD][0]["text"]) == "@infra 4 GPUs on node 2\nLinear: INF-1 (https://linear.app/x/issue/INF-1)"
    root = fake.messages[COORD][0]["ts"]
    n = len(lin.sent("issueCreate"))
    for argv in (["--thread", root, "--to", "@infra"], ["--to", "@infra", "--to", "@ci"], ["--to", "@infra", "--kind", "fyi"]):
        assert run(capsys, "send", "--as", "compute-accounting", *argv, "--text", "more")[0] == 0
    assert len(lin.sent("issueCreate")) == n


def test_closing_or_declining_an_ask_moves_its_issue_to_done_or_canceled(fake, registry, capsys, lin):
    done = fake.add(COORD, text="@infra 4 GPUs" + LINKED, username="compute-accounting", bot_id=slack_bot(registry))
    declined, refused = (fake.add(COORD, text="@infra again" + LINKED, username="compute-accounting", bot_id=slack_bot(registry))
                         for _ in range(2))
    plain = fake.add(COORD, text="@infra no issue", username="compute-accounting", bot_id=slack_bot(registry))
    rc, out, _ = run(capsys, "close", "--as", "infra", "--thread", done["ts"], "--note", "granted")
    assert rc == 0 and "linear: INF-1 Done" in out and lin.moved == [("I1", "S5")]
    rc, out, _ = run(capsys, "close", "--as", "infra", "--thread", declined["ts"], "--declined", "node 2 is busy")
    assert rc == 0 and "linear: INF-1 Canceled" in out and lin.moved[-1] == ("I1", "S9")
    lin.refuse.add(KEY["LINEAR_API_KEY"])
    rc, _, err = run(capsys, "close", "--as", "infra", "--thread", refused["ts"])
    assert rc == 0 and "INF-1 stays open on Linear" in err
    n = len(lin.calls)
    assert run(capsys, "close", "--as", "infra", "--thread", plain["ts"])[0] == 0 and len(lin.calls) == n


def test_a_state_is_matched_by_name_before_type_so_in_progress_is_never_the_first_started_state(tmp_path):
    fake = FakeLinear()
    lin = linear.Linear(KEY, fake, cache=tmp_path / "t")
    assert lin.move("INF-1", "in progress") == "In Progress" and fake.moved[-1] == ("I1", "S3")   # not In Review, the first started
    assert lin.move("INF-1", "completed") == "Done" and fake.moved[-1] == ("I1", "S5")            # a type still works
    with pytest.raises(linear.LinearError, match="no state 'Doing': its states are Todo, In Review, In Progress, Done, Canceled"):
        lin.move("INF-1", "Doing")


def test_the_state_comment_and_list_verbs(fake, registry, capsys, lin):
    lin.subscribers["I1"] = []                                                                 # an issue filed earlier
    rc, out, _ = run(capsys, "linear", "state", "INF-1", "In Progress")
    assert rc == 0 and out.strip() == "INF-1 In Progress" and lin.moved == [("I1", "S3")]
    rc, out, _ = run(capsys, "linear", "comment", "INF-1", "picked up", "--as", "infra")
    assert rc == 0 and out.strip() == "INF-1 commented"
    (_, v, _), = lin.sent("commentCreate")
    assert v["i"] == {"issueId": "I1", "body": "@infra: picked up"}
    rc, _, err = run(capsys, "linear", "comment", "INF-1", "no sender")
    assert rc != 0 and "--as HANDLE" in err
    rc, out, _ = run(capsys, "linear", "list", "infra")
    assert rc == 0 and out.splitlines()[:2] == ["INF-1  Backlog  first", "INF-2  In Progress  later"]   # oldest first
    assert out.splitlines()[2] == f"states: {slack.LINEAR_STATES}" and "In Progress = someone is on it now" in slack.LINEAR_STATES
    assert lin.listed == {"team": {"id": {"eq": "T-infra"}}, "state": {"type": {"nin": ["completed", "canceled"]}}}
    assert run(capsys, "linear", "list", "INF", "--state", "In Progress")[0] == 0
    assert lin.listed["state"] == {"name": {"eqIgnoreCase": "In Progress"}}
    rc, _, err = run(capsys, "linear", "state", "INF-1")
    assert rc != 0 and "give the STATE" in err


def test_a_dry_run_writes_nothing_to_linear_and_prints_what_it_would_do(fake, registry, routes_dirs, capsys, lin):
    """--dry-run suppresses Linear's writes as well as Slack's sends: suppressing Slack's alone lets a dry run comment."""
    lin.subscribers["I1"] = []
    infra = inbox(fake, routes_dirs, "infra")
    rc, out, _ = run(capsys, "linear", "state", "INF-1", "In Progress", "--dry-run")
    assert rc == 0 and out.strip() == "dry-run: would move INF-1 to In Progress" and lin.moved == []
    rc, out, _ = run(capsys, "linear", "comment", "INF-1", "picked up", "--as", "proofs", "--dry-run")
    assert rc == 0 and out.strip() == "dry-run: would comment on INF-1 as proofs: picked up"
    assert lin.sent("commentCreate") == [] and bells(fake, infra) == []
    rc, out, _ = run(capsys, "linear", "place", "INF-1", "Runs", "--dry-run")
    assert rc == 0 and out.strip() == "dry-run: would place INF-1 in Runs"

def test_a_comment_rings_the_issues_team_and_the_names_it_starts_with_but_never_its_author(fake, registry, routes_dirs, capsys, lin):
    """No one subscribes to Linear comments (each would echo its author), so without this ring a lane would poll its issues."""
    lin.subscribers["I1"] = []
    infra, circuits = inbox(fake, routes_dirs, "infra"), inbox(fake, routes_dirs, "circuits")
    rc, _, err = run(capsys, "linear", "comment", "INF-1", "@circuits the trace is up", "--as", "compute-accounting")
    assert rc == 0 and "route @infra: rang" in err and "route @circuits: rang" in err
    (bell,) = bells(fake, infra)
    assert bell["text"] == "from compute-accounting, Linear INF-1 (https://linear.app/x/issue/INF-1): @circuits the trace is up"
    assert len(bells(fake, circuits)) == 1 and not bell["text"].startswith("@")
    rc, _, err = run(capsys, "linear", "comment", "INF-1", "picked up", "--as", "infra")              # the owner on its own issue
    assert rc == 0 and "route" not in err and len(bells(fake, infra)) == 1
    rc, _, err = run(capsys, "linear", "comment", "INF-1", "see this, @circuits", "--as", "proofs")
    assert rc == 0 and "past its start" in err and len(bells(fake, circuits)) == 1                    # posted with a note; rings no one


def test_traffic_counts_an_ask_taken_up_on_linear_as_answered(fake, registry, routes_dirs, capsys, lin):
    """Asks are Linear-first: an owner who moved the issue out of Backlog answered it, whatever the Slack root shows."""
    ask = fake.add(COORD, text="@infra 4 GPUs" + LINKED, username="compute-accounting", bot_id=slack_bot(registry), ts="1600000000.000001")
    rc, out, _ = run(capsys, "traffic", "--hours", "1e6", "--now", "1700100000", "--json")
    assert rc == 0 and ask["ts"] in [s["ts"] for s in json.loads(out)["unanswered"]]       # still in Backlog
    lin.comments = [FakeLinear._author("x", {"createAsUser": "compute-accounting"}, "user")]   # the asker's own comment
    assert ask["ts"] in [s["ts"] for s in json.loads(run(capsys, "traffic", "--hours", "1e6", "--now", "1700100000", "--json")[1])["unanswered"]]
    lin.comments.append(FakeLinear._author("x", {"createAsUser": "infra"}, "user"))            # its owner's reply
    assert ask["ts"] not in [s["ts"] for s in json.loads(run(capsys, "traffic", "--hours", "1e6", "--now", "1700100000", "--json")[1])["unanswered"]]
    lin.comments = []
    lin.state_type = "started"
    rc, out, _ = run(capsys, "traffic", "--hours", "1e6", "--now", "1700100000", "--json")
    assert rc == 0 and ask["ts"] not in [s["ts"] for s in json.loads(out)["unanswered"]]
    assert lin.sent("state { type }")[-1][1] == {"id": "INF-1"}


def test_subscriptions_name_the_leads_linear_team_and_the_registry_refuses_a_shared_key(registry, capsys):
    rc, out, _ = run(capsys, "subscriptions", "--as", "comms")
    assert rc == 0 and 'subscribe_linear_issue(options={"scope": "team", "teamId": "COMM"}' in out
    assert "subscribe_linear_comment" not in out
    assert "subscribe_linear" not in run(capsys, "subscriptions", "--as", "console")[1]
    reg = json.loads(registry.read_text())
    next(h for h in reg["handles"] if h["handle"] == "ci")["linear"] = "COMM"
    registry.write_text(json.dumps(reg))
    rc, _, err = run(capsys, "subscriptions", "--as", "comms")
    assert rc != 0 and "a Linear team key no other handle has" in err


@pytest.mark.parametrize("body, want", [
    ("@infra memory-accounting, 12:15Z, an ask for one auditor-only host in vy-nebius\u2026",
     "An ask for one auditor-only host in vy-nebius\u2026"),
    ("top 06:55Z, Daniel's ruling (25 Oct, 11:43 PM PDT): the toy leaves `integration\u2026",
     "Daniel's ruling (25 Oct, 11:43 PM PDT): the toy leaves `integration\u2026"),
    ("proofs 03:20Z: the lock DM will show it", "The lock DM will show it"),
    ("top (URGENT) 06:55Z: drive is down", "Drive is down"),
    ("infra 01:58Z (sent 01:48Z): check.py hangs after lean-suites fails", "check.py hangs after lean-suites fails"),
    ("proofs, 26 Oct 03:20Z: provisionally accepted A", "Provisionally accepted A"),
    ("<!subteam^SINFRA|@infra> <@U0DANIEL> @ci,\ntraining 21:00Z:\n\na hand-launched Security check", "A hand-launched Security check"),
    ("Two tools/research gaps from tonight's resubmits (fyi, not blocking): (1) `rese\u2026",
     "Two tools/research gaps from tonight's resubmits (fyi, not blocking): (1) `rese\u2026"),
    ("A Layout JSON loader in verity.ml.commitments", "A Layout JSON loader in verity.ml.commitments"),
    ("vLLM on the H200", "vLLM on the H200"),                     # a lower-case first letter of a name stays
    ("hm96 rows for the store", "hm96 rows for the store"),
    ("@infra top 06:55Z:", "@infra top 06:55Z:"),                 # nothing left: the first line as it was
    ("@infra " + "x" * 100, "X" * 1 + "x" * 78 + "\u2026"),
])
def test_a_title_drops_a_slack_posts_ring_tags_and_signature_capitalizes_and_never_comes_out_empty(body, want):
    assert linear.title(body) == want and len(want) <= linear.TITLE_MAX


def test_an_ask_under_a_parent_lands_on_its_project_and_milestone_and_under_the_parent_alone_when_refused(tmp_path, capsys):
    fake = FakeLinear()
    lin = linear.Linear(APP, fake, cache=tmp_path / "t")
    lin.file("infra", "top", "@infra top 06:55Z: one auditor-only host\nthe brief", parent="CMP-92")
    (_, v, _), = fake.sent("issueCreate")
    assert fake.sent("{ id project")[0][1] == {"id": "CMP-92"}
    assert {k: v["i"][k] for k in ("title", "parentId", "projectId", "projectMilestoneId")} == {
        "title": "One auditor-only host", "parentId": "P1", "projectId": "PJ1", "projectMilestoneId": "MS3"}
    assert capsys.readouterr().err == ""
    fake.refuse_project = True
    lin.file("infra", "top", "the brief", parent="CMP-92", title=" The auditor's host ")
    refused, filed = (c[1]["i"] for c in fake.sent("issueCreate")[1:])
    assert refused["projectId"] == "PJ1" and filed["parentId"] == "P1" and "projectId" not in filed and "projectMilestoneId" not in filed
    assert filed["title"] == "The auditor's host"
    assert "Linear refused CMP-92's project for this issue (Linear: Argument Validation Error" in capsys.readouterr().err
    fake.parent = {"id": "P2", "project": None, "projectMilestone": None}                       # a parent on no board
    lin.file("infra", "top", "the brief", parent="CMP-93")
    i = fake.sent("issueCreate")[-1][1]["i"]
    assert i["parentId"] == "P2" and "projectId" not in i and "projectMilestoneId" not in i and len(fake.sent("issueCreate")) == 4


def test_place_takes_one_active_project_and_its_milestone_by_prefix_and_names_the_candidates_otherwise(tmp_path):
    fake = FakeLinear()
    lin = linear.Linear(KEY, fake, cache=tmp_path / "t")
    assert lin.place("TOP-1", "e2e", "m3") == {"issue": "TOP-1", "project": "E2E toy deployment on one H200", "milestone": "M3 · composed"}
    assert fake.updated == [("I1", {"projectId": "PJ1", "projectMilestoneId": "MS3"})]
    assert 'status: {type: {nin: ["completed", "canceled"]}}' in fake.sent("projects(first")[0][0]
    assert lin.place("TOP-1", "E2E", "M1")["milestone"] == "M1 · Mon 8 AM PDT · walking skeleton"     # M1, not M10
    assert lin.place("TOP-1", "full p") == {"issue": "TOP-1", "project": "Full proofs: one soundness theorem", "milestone": None}
    assert fake.updated[-1] == ("I1", {"projectId": "PJ2"})
    with pytest.raises(linear.LinearError, match="active project 'Full' matches 2 of: Full proofs: one soundness theorem; Full stack"):
        lin.place("TOP-1", "Full")
    with pytest.raises(linear.LinearError, match="milestone of E2E toy deployment on one H200 'M9' matches none of: M1 · Mon"):
        lin.place("TOP-1", "E2E", "M9")
    with pytest.raises(linear.LinearError, match="there is no milestone of Full stack to match 'M1'"):
        lin.place("TOP-1", "Full s", "M1")
    assert len(fake.updated) == 3


AUDIT_NOW = 1792440000.0                                          # 2026-10-19T20:00:00Z


def issue(ident: str, state: str = "In Progress", **kw) -> dict:
    """An open issue as the audit reads it, with nothing to find: on PJ1's M3, due later, started and updated an hour ago."""
    return {"identifier": ident, "title": "The auditor-only host", "url": f"https://linear.app/x/issue/{ident}", "dueDate": "2026-10-30",
            "startedAt": "2026-10-19T19:00:00Z", "updatedAt": "2026-10-19T19:00:00Z", "createdAt": "2026-10-18T00:00:00Z",
            "state": {"name": state}, "team": {"key": ident.split("-")[0]}, "project": {"id": "PJ1", "name": PROJECTS[0]["name"]},
            "projectMilestone": {"id": "MS3"}, "labels": {"nodes": []}, "comments": {"nodes": []}, "attachments": {"nodes": []}, **kw}


def child(updated: str, commented: str | None = None) -> dict:
    """A sub-issue as the audit reads it: when it was last updated (its start and state changes move that), and commented on."""
    return {"updatedAt": updated, "comments": {"nodes": [{"createdAt": commented}] if commented else []}}


def pr(n: int, status: str, closed: str | None = None) -> dict:
    return {"url": f"https://github.com/compute-verification/verity/pull/{n}", "sourceType": "github",
            "metadata": {"status": status, "number": n, **({"closedAt": closed} if closed else {})}}


AUDITED = [
    issue("INF-1"),
    issue("INF-2", "Todo", project=None, projectMilestone=None),                                        # orphan
    issue("INF-3", "In Review", projectMilestone=None),                                                 # milestone
    issue("INF-4", dueDate=None),                                                                       # undated
    issue("INF-5", dueDate="2026-10-18"),                                                               # overdue
    issue("INF-6", startedAt="2026-10-18T14:00:00Z", comments={"nodes": [{"createdAt": "2026-10-18T19:00:00Z"}]}),   # quiet
    issue("INF-7", startedAt="2026-10-18T14:00:00Z", comments={"nodes": [{"createdAt": "2026-10-19T18:00:00Z"}]}),   # commented
    issue("INF-8", "Todo", updatedAt="2026-10-11T19:00:00Z"),                                           # stale
    issue("INF-9", title="infra 01:58Z (sent 01:48Z): check.py hangs"),                                 # title
    issue("INF-10", title="A deliverable cut at the cap\u2026"),                                         # title
    issue("INF-11", "In Review", attachments={"nodes": [pr(1, "merged"), pr(2, "closed"),               # landed
                                                        {"url": "https://github.com/x/y/issues/3", "sourceType": "github", "metadata": {}}]}),
    issue("INF-12", "In Review", attachments={"nodes": [pr(4, "merged"), pr(5, "open")]}),
    issue("INF-13", "Todo", attachments={"nodes": [pr(6, "merged")]}),
    issue("INF-14", "Triage", project=None, projectMilestone=None, dueDate=None),
    issue("INF-15", project={"id": "PJ3", "name": "Full stack"}, projectMilestone=None),                # no milestones to set
    issue("DAN-1"),                                                                                     # lane
    issue("DAN-2", labels={"nodes": [{"name": "lane:compute-research"}, {"name": "from:top"}]}),
    issue("CMP-3", labels={"nodes": [{"name": "lane:compute-research"}]}),
    issue("INF-16", title="The served step's commit overhead on the H200: from ~72 s a step to Daniel's 20-second target"),
    issue("INF-17", title="pous: the weights-only obligation in Lean"),                                 # an area prefix
    issue("INF-18", title="<!subteam^SINFRA|@infra> the host"),                                         # title
    issue("INF-19", startedAt="2026-10-17T00:00:00Z", children={"nodes": [child("2026-10-19T19:00:00Z")]}),   # a child moved
    issue("INF-20", startedAt="2026-10-17T00:00:00Z", children={"nodes": [child("2026-10-15T00:00:00Z", "2026-10-19T18:00:00Z")]}),
    issue("INF-21", "Todo", updatedAt="2026-10-11T19:00:00Z", children={"nodes": [child("2026-10-18T19:00:00Z")]}),
    issue("INF-22", "Todo", updatedAt="2026-10-11T19:00:00Z", comments={"nodes": [{"createdAt": "2026-10-18T19:00:00Z"}]}),
    issue("INF-23", startedAt="2026-10-17T00:00:00Z", children={"nodes": [child("2026-10-17T00:00:00Z", "2026-10-17T01:00:00Z")]}),
    issue("INF-24", attachments={"nodes": [pr(7, "merged", "2026-10-19T17:00:00Z"), pr(8, "closed", "2026-10-19T18:00:00Z")]},
          comments={"nodes": [{"createdAt": "2026-10-19T18:30:00Z"}]}),                             # the piece left, named after
    issue("INF-25", attachments={"nodes": [pr(9, "merged", "2026-10-19T18:00:00Z")]},
          comments={"nodes": [{"createdAt": "2026-10-19T17:30:00Z"}]}),                             # landed: commented before
    issue("INF-26", attachments={"nodes": [pr(10, "merged", "2026-10-19T17:00:00Z"), pr(11, "closed")]},
          comments={"nodes": [{"createdAt": "2026-10-19T18:30:00Z"}]}),                             # landed: #11's close unknown
]


def test_the_audit_finds_each_kind_once_on_its_issue_and_nothing_on_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(linear, "AUDIT_PAGE", 4)                  # several pages
    fake = FakeLinear()
    fake.open = AUDITED
    found = linear.Linear(KEY, fake, cache=tmp_path / "t").audit({"INF": "infra", "CMP": "compute-accounting"}, now=AUDIT_NOW)
    assert [(f["issue"], f["kind"]) for f in found] == [
        ("DAN-1", "lane"), ("INF-2", "orphan"), ("INF-3", "milestone"), ("INF-4", "undated"), ("INF-5", "overdue"),
        ("INF-6", "quiet"), ("INF-8", "stale"), ("INF-9", "title"), ("INF-10", "title"), ("INF-11", "landed"),
        ("INF-18", "title"), ("INF-23", "quiet"), ("INF-25", "landed"), ("INF-26", "landed")]
    assert {f["issue"]: f["lane"] for f in found}["DAN-1"] == linear.NO_LANE
    by = {(f["issue"], f["kind"]): f for f in found}
    assert by[("INF-2", "orphan")] == {"issue": "INF-2", "url": "https://linear.app/x/issue/INF-2", "title": "The auditor-only host",
                                       "team": "INF", "lane": "infra", "state": "Todo", "kind": "orphan",
                                       "fix": "put it in its project and milestone (`research msg linear place INF-2 PROJECT "
                                              "--milestone M`), or move it to Backlog"}
    assert '`research msg linear place INF-3 "E2E toy deployment on one H200" --milestone M`' in by[("INF-3", "milestone")]["fix"]
    assert by[("INF-6", "quiet")]["fix"].startswith("no comment or start for 25 h: comment one line")
    assert by[("INF-8", "stale")]["fix"].startswith("not updated or commented on for 8 days: plan it")
    assert by[("INF-23", "quiet")]["fix"].startswith("no comment or start for 67 h")                   # its child's comment, 67 h ago
    assert by[("INF-9", "title")]["fix"] == ("retitle it to the deliverable, with no handle, sender, time or trailing \u2026 "
                                             "(`research msg linear edit INF-9 --title T`)")
    assert by[("INF-4", "undated")]["fix"] == ("set the date it lands, its milestone's at the latest (`research msg linear edit "
                                               "INF-4 --due YYYY-MM-DD`), or move it to Backlog")
    assert "(`research msg linear edit INF-5 --due YYYY-MM-DD`)" in by[("INF-5", "overdue")]["fix"]
    assert by[("INF-11", "landed")]["fix"].startswith("every attached PR is merged or closed (#1, #2): move it to Done")
    assert by[("INF-25", "landed")]["fix"].endswith("(`research msg linear comment INF-25 TEXT`)")
    assert by[("DAN-1", "lane")]["fix"] == "team DAN has no handle: add `lane:<owner>`"
    assert len(fake.sent("pageInfo")) == 8 and all("first: 4," in q for q, _, _ in fake.sent("pageInfo"))
    assert all("children(first: 50, orderBy: updatedAt) { nodes { updatedAt comments(first: 1, orderBy: createdAt)" in q
               for q, _, _ in fake.sent("pageInfo"))                                                  # one field, no pass per issue
    assert fake.audited == {"state": {"type": {"in": ["triage", "unstarted", "started"]}}}
    assert linear.tally(found) == {"-": {"lane": 1}, "infra": {"orphan": 1, "milestone": 1, "undated": 1, "overdue": 1, "quiet": 2,
                                                                "stale": 1, "title": 3, "landed": 3}}
    linear.Linear(KEY, fake, cache=tmp_path / "t").audit({}, ["INF", "infra"], now=AUDIT_NOW)
    assert fake.audited["team"] == {"or": [{"key": {"in": ["INF", "infra"]}}, {"name": {"in": ["INF", "infra"]}}]}


def test_audit_prints_each_lanes_findings_and_their_counts_and_exits_1_on_any(fake, registry, capsys, lin):
    lin.open = [issue("INF-2", "Todo", project=None, projectMilestone=None), issue("INF-9", title="x\u2026"), issue("DAN-1"),
                issue("CMP-3", dueDate=None, labels={"nodes": [{"name": "lane:compute-research"}]})]
    lin.visible = sorted(set(linear.owners(json.loads(registry.read_text()))))
    now = ("--now", str(AUDIT_NOW))
    rc, out, err = run(capsys, "linear", "audit", *now)
    lines = out.splitlines()
    assert rc == 1 and lines[:7] == [
        "@- (1):", "  DAN-1 lane: team DAN has no handle: add `lane:<owner>`",
        "@compute-research (1):", "  CMP-3 undated: set the date it lands, its milestone's at the latest (`research msg linear edit "
                                  "CMP-3 --due YYYY-MM-DD`), or move it to Backlog",
        "@infra (2):", "  INF-2 orphan: put it in its project and milestone (`research msg linear place INF-2 PROJECT --milestone M`), "
                       "or move it to Backlog",
        "  INF-9 title: retitle it to the deliverable, with no handle, sender, time or trailing \u2026 (`research msg linear edit "
        "INF-9 --title T`)"]
    assert err == ""                                                                              # every team read
    assert lines[7] == "4 finding(s), by lane and kind:" and lines[8].split() == ["lane", "orphan", "undated", "title", "lane", "total"]
    assert [ln.split() for ln in lines[9:]] == [["@-", "0", "0", "0", "1", "1"], ["@compute-research", "0", "1", "0", "0", "1"],
                                                ["@infra", "1", "0", "1", "0", "2"], ["total", "1", "1", "1", "1", "4"]]
    rc, out, _ = run(capsys, "linear", "audit", "--lane", "@infra", "--json", *now)
    rep = json.loads(out)
    assert rc == 1 and [f["issue"] for f in rep["findings"]] == ["INF-2", "INF-9"] and rep["counts"] == {"infra": {"orphan": 1, "title": 1}}
    rc, out, _ = run(capsys, "linear", "audit", "infra", "CMP", "--lane", "comms", *now)
    assert rc == 0 and out.strip() == "0 finding(s)"
    assert lin.audited["team"]["or"][0] == {"key": {"in": ["INF", "CMP"]}}                        # a handle as its team's key


def test_the_audit_names_the_teams_it_cant_read_skips_them_and_never_fails_on_them(fake, registry, capsys, lin):
    """The app may lack access to a team, and Linear answers a filter on such a team with no issues, so a lane would read
    its unread team as clean."""
    reg = json.loads(registry.read_text())
    for h in reg["handles"]:
        if h["handle"] in ("kernels", "training"):
            h["linear"] = {"kernels": "HASH", "training": "TRN"}[h["handle"]]
    registry.write_text(json.dumps(reg))
    lin.visible = [k for k in linear.owners(reg) if k not in ("HASH", "TRN")]
    lin.open = [issue("INF-2", "Todo", project=None, projectMilestone=None)]
    now = ("--now", str(AUDIT_NOW))
    rc, out, err = run(capsys, "linear", "audit", "--json", *now)
    rep = json.loads(out)
    assert rc == 1 and rep["unread"] == ["HASH", "TRN"] and [f["issue"] for f in rep["findings"]] == ["INF-2"]
    assert err.strip() == "linear audit: the app can't read team(s) HASH, TRN, so their issues aren't audited"
    n = len(lin.sent("pageInfo"))
    rc, out, err = run(capsys, "linear", "audit", "kernels", "TRN", *now)
    assert rc == 0 and out.strip() == "0 finding(s)" and "HASH, TRN" in err and len(lin.sent("pageInfo")) == n   # no audit of every team
    rc, out, err = run(capsys, "linear", "audit", "kernels", "infra", *now)
    assert rc == 1 and "INF-2 orphan" in out and "team(s) HASH," in err and lin.audited["team"]["or"][0] == {"key": {"in": ["INF"]}}
    assert linear.Linear(KEY, FakeLinear()).unread(["inf", "Infra", "COMMS", "HASH"]) == ["HASH"]   # a key or a name, any case


def test_place_and_an_ask_under_a_parent_from_the_cli(fake, registry, capsys, lin, monkeypatch):
    monkeypatch.setattr(slack, "_now", lambda: datetime(2026, 10, 27, 7, 10, tzinfo=UTC))
    rc, out, _ = run(capsys, "linear", "place", "TOP-1", "E2E", "toy", "--milestone", "M3")
    assert rc == 0 and out.strip() == "TOP-1 placed in E2E toy deployment on one H200 / M3 · composed"
    assert lin.updated == [("I1", {"projectId": "PJ1", "projectMilestoneId": "MS3"})]
    rc, _, err = run(capsys, "linear", "place", "TOP-1", "Full")
    assert rc == 2 and "matches 2 of" in err
    rc, out, _ = run(capsys, "ask", "--as", "top", "--to", "@infra", "--parent", "CMP-92", "--text", "top 06:55Z: one host\nthe brief")
    (_, v, _), = lin.sent("issueCreate")
    assert rc == 0 and v["i"]["parentId"] == "P1" and v["i"]["projectMilestoneId"] == "MS3" and v["i"]["title"] == "One host"
    assert out.startswith("linear: INF-1 https://linear.app/x/issue/INF-1 under CMP-92")
    assert re.fullmatch(r"<!subteam\^SINFRA\|@infra> top 06:55Z \(sent 07:10Z\): one host", fake.messages[COORD][-1]["text"].removesuffix(LINKED))  # Slack: the first line, as it was
    rc, _, _ = run(capsys, "send", "--as", "top", "--to", "@infra", "--title", "The host", "--text", "one host")
    assert rc == 0 and lin.sent("issueCreate")[-1][1]["i"]["title"] == "The host" and "parentId" not in lin.sent("issueCreate")[-1][1]["i"]
    root = fake.messages[COORD][-1]["ts"]
    for argv in (["--thread", root, "--to", "@infra"], ["--to", "@infra", "@ci"], ["--to", "@infra", "--kind", "fyi"]):
        rc, _, err = run(capsys, "send", "--as", "top", *argv, "--parent", "CMP-92", "--text", "more")
        assert rc == 2 and "--parent and --title go with an ask" in err
    assert len(lin.sent("issueCreate")) == 2


def test_create_files_an_issue_into_your_own_team_with_its_brief_title_and_due_date(fake, registry, capsys, lin, tmp_path):
    """`ask` refuses your own handle, so a lane had no command to file its own deliverable."""
    brief = tmp_path / "brief.md"
    brief.write_text("**Goal:** the figures.\n**Done when:** published.")
    argv = ["linear", "create", "--as", "infra", "--title", "Publish the figures", "--file", str(brief)]
    rc, out, _ = run(capsys, *argv, "--dry-run")
    assert rc == 0 and out.strip() == "dry-run: would file 'Publish the figures' in INF as infra" and lin.sent("issueCreate") == []
    rc, out, _ = run(capsys, *argv, "--due", "2026-10-29", "--parent", "CMP-92")
    (_, v, _), = lin.sent("issueCreate")
    assert rc == 0 and out.startswith("INF-1 https://linear.app/x/issue/INF-1")
    assert v["i"]["title"] == "Publish the figures" and v["i"]["description"].startswith("**Goal:**") and v["i"]["parentId"] == "P1"
    assert lin.updated[-1][1] == {"dueDate": "2026-10-29"}
    for missing, why in ((["--title", "T"], "--file BRIEF.md"), (["--file", str(brief)], "--title")):
        rc, _, err = run(capsys, "linear", "create", "--as", "infra", *missing)
        assert rc == 2 and why in err


def test_edit_sets_the_title_due_date_description_and_parent_and_refuses_a_bad_date_or_nothing_to_set(tmp_path):
    fake = FakeLinear()
    lin = linear.Linear(KEY, fake, cache=tmp_path / "t")
    sent = lin.edit("INF-1", " The auditor's host ", "2026-10-30", "## Done when\n- one host", "CMP-92")
    assert sent == {"title": "The auditor's host", "dueDate": "2026-10-30", "description": "## Done when\n- one host", "parentId": "P1"}
    assert fake.updated == [("I1", sent)]
    assert lin.edit("INF-1", due="none") == {"dueDate": None} and fake.updated[-1] == ("I1", {"dueDate": None})
    assert lin.edit("INF-1", parent="CMP-92", send=False) == {"parentId": "CMP-92"}                 # built, not sent
    for kw, says in (({}, "give at least one of --title, --due, --file and --parent"), ({"title": " "}, "--title: give"),
                     ({"due": "2026-10-32"}, "--due '2026-10-32': give a date as YYYY-MM-DD"), ({"due": "2026-1-5"}, "YYYY-MM-DD"),
                     ({"due": "Friday"}, "YYYY-MM-DD"), ({"description": "\n"}, "--file: the file is empty")):
        with pytest.raises(linear.LinearError, match=re.escape(says)):
            lin.edit("INF-1", **kw)
    assert len(fake.updated) == 2


def test_edit_from_the_cli_replaces_the_description_with_the_file_and_a_dry_run_sends_nothing(fake, registry, capsys, lin, tmp_path):
    brief = tmp_path / "BRIEF.md"
    brief.write_text("# The host\n\n- [ ] one auditor-only host\n")
    rc, out, _ = run(capsys, "linear", "edit", "INF-1", "--title", "The auditor's host", "--file", str(brief), "--due", "2026-10-30",
                     "--parent", "CMP-92")
    assert rc == 0 and out.strip() == "INF-1 set title The auditor's host, dueDate 2026-10-30, description 40 characters, parentId P1"
    assert lin.updated == [("I1", {"title": "The auditor's host", "dueDate": "2026-10-30",
                                   "description": "# The host\n\n- [ ] one auditor-only host\n", "parentId": "P1"})]
    rc, out, _ = run(capsys, "linear", "edit", "INF-1", "--due", "none", "--dry-run")
    assert rc == 0 and out.strip() == "dry-run: would set dueDate None" and len(lin.updated) == 1
    for argv, says in ((["INF-1"], "give at least one of --title, --due, --file and --parent"),
                       (["INF-1", "--file", str(tmp_path / "missing.md")], "--file " + str(tmp_path / "missing.md") + ": No such file"),
                       (["INF-1", "--due", "30 Oct"], "give a date as YYYY-MM-DD"), ([], "give the ISSUE")):
        rc, _, err = run(capsys, "linear", "edit", *argv)
        assert rc == 2 and says in err
    assert len(lin.updated) == 1


def test_a_lanes_worker_comments_as_lane_slash_name_through_the_app_and_rings_no_one(fake, registry, routes_dirs, capsys, lin,
                                                                                     monkeypatch, tmp_path):
    """Workers hold no handle, so `--as` refuses them; the Linear MCP connector, their other way to comment, acts as Daniel."""
    monkeypatch.setenv("LINEAR_CLIENT_ID", APP["LINEAR_CLIENT_ID"])
    monkeypatch.setenv("LINEAR_CLIENT_SECRET", APP["LINEAR_CLIENT_SECRET"])
    monkeypatch.setenv("RESEARCH_LINEAR_TOKEN_CACHE", str(tmp_path / "tok"))
    lin.subscribers["I1"] = []
    infra, circuits = inbox(fake, routes_dirs, "infra"), inbox(fake, routes_dirs, "circuits")
    rc, out, err = run(capsys, "linear", "comment", "INF-1", "the trace is up", "--worker", "LINTOOLS")
    assert rc == 0 and out.strip() == "INF-1 commented as infra/LINTOOLS" and err == ""
    (_, v, _), = lin.sent("commentCreate")
    assert v["i"] == {"issueId": "I1", "body": "the trace is up", "createAsUser": "infra/LINTOOLS"}
    rc, _, err = run(capsys, "linear", "comment", "INF-1", "@circuits the trace is up", "--worker", "lintools-2")
    assert rc == 0 and lin.sent("commentCreate")[-1][1]["i"]["createAsUser"] == "infra/lintools-2"
    assert "a worker's comment rings no one, so @circuits won't hear of it" in err
    assert bells(fake, infra) == [] and bells(fake, circuits) == []                                     # rings no one
    for argv, says in ((["comment", "INF-1", "x", "--worker", "W", "--as", "infra"], "never with --as"),
                       (["state", "INF-1", "Done", "--worker", "W"], "--worker goes with `comment` alone"),
                       (["comment", "INF-1", "x", "--worker", "two words"], "--worker 'two words': a name of letters")):
        rc, _, err = run(capsys, "linear", *argv)
        assert rc == 2 and says in err
    lin.team_key = "DAN"
    rc, _, err = run(capsys, "linear", "comment", "INF-1", "x", "--worker", "W")
    assert rc == 2 and "INF-1's team DAN has no handle in the registry" in err
    assert len(lin.sent("commentCreate")) == 2 and lin.moved == []


def stateless(fake: FakeLinear) -> bool:
    """No request sent a state or a status: an edit that sends one can reset an issue's state (Done back to Todo)."""
    return not any(k in json.dumps(v) for q, v, _ in fake.calls if q != "grant" for k in ("stateId", "statusId"))


def test_relate_sends_each_kind_its_way_round_and_refuses_one_issue_twice_or_a_relation_that_exists(tmp_path):
    fake = FakeLinear()
    lin = linear.Linear(APP, fake, cache=tmp_path / "t")
    assert lin.relate("INF-1", "blocks", "INF-2") == "INF-1 blocks INF-2"
    assert lin.relate("INF-3", "blocked-by", "INF-1") == "INF-1 blocks INF-3"                     # the other one blocks it
    assert lin.relate("INF-1", "related", "inf-4") == "INF-1 is related to INF-4"
    assert lin.relate("INF-5", "duplicate", "INF-1") == "INF-5 is a duplicate of INF-1 (Linear moves INF-5 to its Duplicate status)"
    assert fake.sent("a: issue(id: $a)")[0][1] == {"a": "INF-1", "b": "INF-2"}
    assert fake.wrote == [("issueRelationCreate", {"i": {"issueId": f"I-INF-{x}", "relatedIssueId": f"I-INF-{y}", "type": t}})
                          for x, y, t in ((1, 2, "blocks"), (1, 3, "blocks"), (1, 4, "related"), (5, 1, "duplicate"))]
    n = len(fake.calls)
    with pytest.raises(linear.LinearError, match="'blocking' is no relation: give one of blocks, blocked-by, related, duplicate"):
        lin.relate("INF-1", "blocking", "INF-2")
    assert len(fake.calls) == n                                                                    # refused before any request
    for args, says in ((("INF-2", "blocked-by", "INF-1"), "INF-1 blocks INF-2 already: there is nothing to add"),
                       (("INF-4", "related", "INF-1"), "INF-4 is related to INF-1 already"),                # either way round
                       (("INF-1", "blocks", "inf-1"), "INF-1 and inf-1 are one issue: name two different issues")):
        with pytest.raises(linear.LinearError, match=re.escape(says)):
            lin.relate(*args)
    assert len(fake.wrote) == 4
    assert lin.relate("INF-2", "related", "INF-1") == "INF-2 is related to INF-1"                 # another kind is no duplicate
    assert lin.unrelate("INF-2", "INF-1") == ["INF-2 is related to INF-1", "INF-1 blocks INF-2"]   # every one, either way
    assert fake.wrote[-2:] == [("issueRelationDelete", {"id": "R5"}), ("issueRelationDelete", {"id": "R1"})]
    with pytest.raises(linear.LinearError, match="INF-1 and INF-2 have no relation to remove"):
        lin.unrelate("INF-1", "INF-2")
    with pytest.raises(linear.LinearError, match="are one issue"):
        lin.unrelate("INF-1", "INF-1")
    assert len(fake.wrote) == 7 and stateless(fake)


def test_update_posts_its_health_and_body_on_the_one_active_project_and_refuses_otherwise(tmp_path):
    fake = FakeLinear()
    lin = linear.Linear(APP, fake, cache=tmp_path / "t")
    assert lin.update("e2e", "atrisk", "## Status\nM3 slips a day") == ("E2E toy deployment on one H200: update posted, atRisk "
                                                                        "https://linear.app/x/update/1")
    assert fake.wrote == [("projectUpdateCreate", {"i": {"projectId": "PJ1", "health": "atRisk", "body": "## Status\nM3 slips a day"}})]
    for args, says in ((("e2e", None, "x"), "--health: give one of onTrack, atRisk, offTrack"),
                       (("e2e", "green", "x"), "--health 'green': give one of onTrack, atRisk, offTrack"),
                       (("e2e", "onTrack", None), "--body-file F: give the update's text as a markdown file"),
                       (("e2e", "onTrack", " \n"), "--body-file: the file is empty"),
                       (("Full", "onTrack", "x"), "active project 'Full' matches 2 of: Full proofs: one soundness theorem; Full stack")):
        with pytest.raises(linear.LinearError, match=re.escape(says)):
            lin.update(*args)
    assert len(fake.wrote) == 1 and stateless(fake)


def test_project_create_sends_its_teams_lead_date_and_body_and_refuses_a_name_taken_in_any_case(tmp_path):
    fake = FakeLinear()
    lin = linear.Linear(APP, fake, cache=tmp_path / "t")
    out = lin.create_project(" Smaller messaging code ", ["INF", "comms"], "daniel", "2026-11-30", "# Why\nfewer lines")
    assert out == ("project Smaller messaging code created: teams INF COM, lead Daniel Reuter, targetDate 2026-11-30, content 17 "
                   "characters https://linear.app/x/project/1")
    assert fake.wrote == [("projectCreate", {"i": {"name": "Smaller messaging code", "teamIds": ["T-infra", "T-comms"],
                                                   "targetDate": "2026-11-30", "content": "# Why\nfewer lines", "leadId": "U1"}})]
    assert fake.sent("projects(filter")[0][1] == {"n": "Smaller messaging code"}
    for args, says in (((" full STACK", ["INF"]), "project 'Full stack' exists already: pick another name, or work in that one"),
                       (("New", []), "project create: give its team, --team KEY (more: repeat it)"),
                       (("New", ["NOPE"]), "no Linear team named 'NOPE'"),
                       (("New", ["INF"], "lean"), "Linear user 'lean' matches none of: Daniel Reuter; Verity Asks"),
                       (("New", ["INF"], None, "30 Nov"), "--target '30 Nov': give a date as YYYY-MM-DD"),
                       (("New", ["INF"], None, None, ""), "--description-file: the file is empty")):
        with pytest.raises(linear.LinearError, match=re.escape(says)):
            lin.create_project(*args)
    assert len(fake.wrote) == 1 and stateless(fake)


def test_milestone_create_and_edit_by_prefix_and_refuse_a_name_the_project_has_in_any_case(tmp_path):
    fake = FakeLinear()
    lin = linear.Linear(APP, fake, cache=tmp_path / "t")
    e2e = "E2E toy deployment on one H200"
    assert lin.milestone("e2e", "M4 · served", "2026-11-02", "the served step", create=True) == (
        f"{e2e} / M4 · served: created, targetDate 2026-11-02, description 15 characters")
    assert lin.milestone("full p", "F3", create=True) == "Full proofs: one soundness theorem / F3: created"
    assert lin.milestone("e2e", "m3", "none", rename="M3 · composed and served") == (
        f"{e2e} / M3 · composed: set name M3 · composed and served, targetDate None")
    assert lin.milestone("e2e", "M3", rename="m3 · COMPOSED") == f"{e2e} / M3 · composed: set name m3 · COMPOSED"   # its own name
    assert fake.wrote == [
        ("projectMilestoneCreate", {"i": {"projectId": "PJ1", "name": "M4 · served", "targetDate": "2026-11-02",
                                          "description": "the served step"}}),
        ("projectMilestoneCreate", {"i": {"projectId": "PJ2", "name": "F3"}}),
        ("projectMilestoneUpdate", {"id": "MS3", "i": {"name": "M3 · composed and served", "targetDate": None}}),
        ("projectMilestoneUpdate", {"id": "MS3", "i": {"name": "m3 · COMPOSED"}})]
    for args, kw, says in (
            (("e2e", "m3 · Composed"), {"create": True}, f"{e2e} has milestone 'M3 · composed' already: pick another name, or "
                                                          "edit that one (`research msg linear milestone edit PROJECT NAME`)"),
            (("e2e", "M3"), {"rename": "M1 · mon 8 AM PDT · walking skeleton"}, "has milestone 'M1 · Mon 8 AM PDT · walking skeleton' already"),
            (("e2e", "M3"), {}, "milestone edit: give at least one of --date, --description-file and --rename"),
            (("e2e", "M9"), {"date": "2026-11-02"}, f"milestone of {e2e} 'M9' matches none of: M1"),
            (("Full s", "M1"), {"date": "2026-11-02"}, "there is no milestone of Full stack to match 'M1'"),
            (("e2e", "M3"), {"date": "2026-02-30"}, "--date '2026-02-30': give a date as YYYY-MM-DD"),
            (("e2e", "M3"), {"rename": " "}, "--rename: give the new name"),
            (("Full", "X"), {"create": True}, "active project 'Full' matches 2 of")):
        with pytest.raises(linear.LinearError, match=re.escape(says)):
            lin.milestone(*args, **kw)
    assert len(fake.wrote) == 4 and stateless(fake)


def test_the_relation_and_project_verbs_from_the_cli_print_one_line_and_refuse_with_exit_2(fake, registry, capsys, lin, tmp_path):
    body = tmp_path / "UPDATE.md"
    body.write_text("M3 slips a day\n")
    e2e = "E2E toy deployment on one H200"
    for argv, want in (
            (["relate", "INF-1", "blocked-by", "INF-2"], "INF-2 blocks INF-1"),
            (["unrelate", "INF-1", "INF-2"], "INF-1 unrelated: removed INF-2 blocks INF-1"),
            (["update", "E2E", "toy", "--health", "offTrack", "--body-file", str(body)],
             f"{e2e}: update posted, offTrack https://linear.app/x/update/1"),
            (["project", "create", "Smaller", "Lean", "--team", "INF", "--team", "COM", "--target", "2026-11-30",
              "--description-file", str(body)],
             "project Smaller Lean created: teams INF COM, targetDate 2026-11-30, content 15 characters https://linear.app/x/project/1"),
            (["milestone", "create", "E2E", "M4 · served", "--date", "2026-11-02"], f"{e2e} / M4 · served: created, targetDate 2026-11-02"),
            (["milestone", "edit", "E2E", "M3", "--rename", "M3 · served", "--description-file", str(body)],
             f"{e2e} / M3 · composed: set name M3 · served, description 15 characters")):
        assert run(capsys, "linear", *argv) == (0, want + "\n", "")
    assert [w for w, _ in lin.wrote] == ["issueRelationCreate", "issueRelationDelete", "projectUpdateCreate", "projectCreate",
                                         "projectMilestoneCreate", "projectMilestoneUpdate"]
    assert lin.wrote[2][1]["i"]["body"] == "M3 slips a day\n" and lin.wrote[3][1]["i"]["content"] == "M3 slips a day\n"
    for argv, says in ((["relate", "INF-1", "INF-2"], "linear relate: give ISSUE blocks|blocked-by|related|duplicate OTHER"),
                       (["relate", "INF-1", "blocks", "INF-1"], "INF-1 and INF-1 are one issue"),
                       (["unrelate", "INF-1"], "linear unrelate: give the OTHER issue"),
                       (["unrelate", "INF-1", "INF-2"], "INF-1 and INF-2 have no relation to remove"),
                       (["update", "--health", "onTrack", "--body-file", str(body)], "linear update: give the PROJECT"),
                       (["update", "E2E", "--health", "onTrack", "--body-file", str(tmp_path / "missing.md")], "No such file"),
                       (["update", "E2E", "--body-file", str(body)], "--health: give one of onTrack, atRisk, offTrack"),
                       (["project", "edit", "Smaller", "Lean"], "linear project: give `create NAME`"),
                       (["project", "create", "--team", "INF"], "linear project: give the project's NAME"),
                       (["milestone", "rename", "E2E", "M3"], "linear milestone: give `create` or `edit`, then PROJECT NAME"),
                       (["milestone", "create", "E2E"], "linear milestone create: give PROJECT"),
                       (["relate", "INF-1", "blocks", "INF-2", "--dry-run"], "linear relate has no --dry-run (it would write)")):
        rc, _, err = run(capsys, "linear", *argv)
        assert rc == 2 and says in err, (argv, err)
    assert len(lin.wrote) == 6 and stateless(lin)


def slack_bot(registry) -> str:
    return json.loads(registry.read_text())["bot_id"]
