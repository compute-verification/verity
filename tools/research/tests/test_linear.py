from __future__ import annotations

import io
import json
import stat
import urllib.error
import urllib.parse

import pytest

from research import linear, slack
from test_slack import COORD, fake, registry, routes_dirs, run  # noqa: F401  (Slack's fixtures)

APP = {"LINEAR_CLIENT_ID": "dummy-client", "LINEAR_CLIENT_SECRET": "dummy-secret"}
KEY = {"LINEAR_API_KEY": "dummy-key"}


class FakeLinear:
    """Linear's token endpoint and GraphQL API: one team per name in `teams`, labels made on first use."""

    def __init__(self, teams=("infra", "comms")):
        self.teams, self.labels, self.calls, self.grants, self.refuse = list(teams), {}, [], 0, set()
        self.subscribers, self.authors, self.comments, self.moved = {}, {}, [], []

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
        self.calls.append((q, v, headers))
        if headers["Authorization"] in self.refuse:
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, io.BytesIO(b'{"errors": [{"message": "auth"}]}'))
        user = "the app" if headers["Authorization"].startswith("Bearer") else "Daniel"
        if "teams(filter" in q:
            team = next((t for t in self.teams if v["n"] in (t, t[:3].upper())), None)
            return {"data": {"teams": {"nodes": [{"id": f"T-{team}", "key": team[:3].upper(), "name": team}] if team else []}}}
        if "botActor" in q:
            return {"data": {"issue": {**self.authors["I1"], "comments": {"nodes": self.comments}}}}
        if "issueLabels" in q:
            return {"data": {"issueLabels": {"nodes": [{"id": self.labels[v["n"]]}] if v["n"] in self.labels else []}}}
        if "issueLabelCreate" in q:
            self.labels[v["i"]["name"]] = f"L{len(self.labels)}"
            return {"data": {"issueLabelCreate": {"issueLabel": {"id": self.labels[v["i"]["name"]]}}}}
        if "issueCreate" in q:
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
                                                                    {"id": "S5", "name": "Done", "type": "completed"},
                                                                    {"id": "S9", "name": "Canceled", "type": "canceled"}]}}}}}
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
    lin.teams = []
    rc, _, err = run(capsys, "ask", "--as", "compute-accounting", "--to", "@infra", "--text", "again")
    assert rc == 0 and "not filed on Linear (no Linear team named 'INF')" in err
    assert fake.messages[COORD][-1]["text"].endswith("again")
    n = len(lin.calls)
    assert run(capsys, "ask", "--as", "infra", "--to", "@old-accounting", "--text", "no team")[0] == 0
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


def test_subscriptions_name_the_leads_linear_team_and_the_registry_refuses_a_shared_key(registry, capsys):
    rc, out, _ = run(capsys, "subscriptions", "--as", "comms")
    assert rc == 0 and 'subscribe_linear_comment(options={"scope": "team", "teamId": "COMM"}' in out
    assert "subscribe_linear" not in run(capsys, "subscriptions", "--as", "old-accounting")[1]
    reg = json.loads(registry.read_text())
    next(h for h in reg["handles"] if h["handle"] == "ci")["linear"] = "COMM"
    registry.write_text(json.dumps(reg))
    rc, _, err = run(capsys, "subscriptions", "--as", "comms")
    assert rc != 0 and "a Linear team key no other handle has" in err


def slack_bot(registry) -> str:
    return json.loads(registry.read_text())["bot_id"]
