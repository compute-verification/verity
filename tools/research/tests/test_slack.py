"""`research msg` against a fake Slack Web API: a local HTTP server that keeps channels, threads, reactions and user groups the
way the real one returns them. No real token or workspace is involved (conftest clears SLACK_*)."""
from __future__ import annotations

import email.message
import http.server
import io
import itertools
import json
import os
import re
import shutil
import socketserver
import tempfile
import threading
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pytest

from research import slack

REAL_REGISTRY = Path(__file__).resolve().parents[3] / slack.REGISTRY_REL
ACTING_AS = slack.acting_as
TOKEN = "dummy-verity-bot"
BOT = "B0VERITY"
COORD, ALERTS = "CCOORD", "CALERTS"
HANDLES = ["infra", "proofs", "circuits", "compute-accounting", "memory-accounting", "network-accounting", "console",
           "ci", "lean", "comms", "top", "vllm", "benchmarks", "old-accounting", "old-circuits-and-proofs"]


class FakeSlack:
    def __init__(self):
        self.messages: dict[str, list[dict]] = {}
        self.groups: list[dict] = []
        self.reactions: list[tuple[str, str, str]] = []
        self.calls: list[tuple[str, dict]] = []
        self._ts = itertools.count(1)
        self.page = 200
        self.replies_keep_root = False

    def add(self, channel: str, **msg) -> dict:
        msg.setdefault("ts", f"1700000000.{next(self._ts):06d}")
        msg.setdefault("type", "message")
        self.messages.setdefault(channel, []).append(msg)
        return msg

    def handle(self, method: str, p: dict, auth: str) -> dict:
        self.calls.append((method, p))
        if auth != f"Bearer {TOKEN}":
            return {"ok": False, "error": "invalid_auth"}
        ch = p.get("channel")
        if method == "chat.postMessage":
            text = p["text"]
            for emoji, code in slack.SHORTCODES.items():
                text = text.replace(emoji, code)
            m = self.add(ch, text=text, username=p.get("username"), bot_id=BOT, **({"thread_ts": p["thread_ts"]} if p.get("thread_ts") else {}))
            return {"ok": True, "channel": ch, "ts": m["ts"], "message": m}
        if method == "chat.getPermalink":
            m = next((m for m in self.messages.get(ch, []) if m["ts"] == p["message_ts"]), {})
            reply = f"?thread_ts={m['thread_ts']}&cid={ch}" if m.get("thread_ts") not in (None, m.get("ts")) else ""
            return {"ok": True, "permalink": f"https://compute.slack.com/archives/{ch}/p{p['message_ts'].replace('.', '')}{reply}"}
        if method in ("conversations.history", "conversations.replies"):
            msgs = self.messages.get(ch, [])
            if method == "conversations.replies":
                msgs = [m for m in msgs if m["ts"] == p["ts"] or m.get("thread_ts") == p["ts"]]
            else:
                msgs = [m for m in msgs if not m.get("thread_ts") or m["thread_ts"] == m["ts"]][::-1]     # newest first, like Slack
            if "oldest" in p:
                keep = p["ts"] if method == "conversations.replies" and self.replies_keep_root else None   # as Slack does
                msgs = [m for m in msgs if float(m["ts"]) >= float(p["oldest"]) or m["ts"] == keep]
            if "latest" in p:
                msgs = [m for m in msgs if float(m["ts"]) <= float(p["latest"])]
            start = int(p.get("cursor") or 0)
            page = msgs[start:start + min(int(p.get("limit", 100)), self.page)]
            nxt = start + len(page)
            return {"ok": True, "messages": page, "response_metadata": {"next_cursor": str(nxt) if nxt < len(msgs) else ""}}
        if method == "chat.update":
            m = next((m for m in self.messages.get(ch, []) if m["ts"] == p["ts"]), None)
            if not m:
                return {"ok": False, "error": "message_not_found"}
            m["text"] = p["text"]
            return {"ok": True, "channel": ch, "ts": m["ts"]}
        if method == "reactions.add":
            if (ch, p["timestamp"], p["name"]) in self.reactions:
                return {"ok": False, "error": "already_reacted"}                     # one reaction per emoji per user, like Slack
            self.reactions.append((ch, p["timestamp"], p["name"]))
            return {"ok": True}
        if method == "auth.test":
            return {"ok": True, "bot_id": BOT, "user_id": "U0VERITYBOT"}
        if method == "usergroups.list":
            return {"ok": True, "usergroups": self.groups}
        if method == "usergroups.create":
            g = {"id": f"S{len(self.groups) + 1:04d}", "name": p["name"], "handle": p["handle"], "description": p.get("description", ""), "date_delete": 0}
            self.groups.append(g)
            return {"ok": True, "usergroup": g}
        if method in ("usergroups.update", "usergroups.enable", "usergroups.disable"):
            g = next(g for g in self.groups if g["id"] == p["usergroup"])
            g.update({k: p[k] for k in ("name", "handle", "description") if k in p}, **({"date_delete": 0} if method.endswith("enable") else {}),
                     **({"date_delete": 1700000000} if method.endswith("disable") else {}))
            return {"ok": True, "usergroup": g}
        return {"ok": False, "error": "unknown_method"}


@pytest.fixture(autouse=True)
def routes_dirs(tmp_path, monkeypatch):
    """No test reads the real notes clone's routes or the published ones: `local` is the clone's routes.d, `remote` the
    published files, `seen` the read marks; nor asks the real gh for a PR's branch.  `--as` takes any name here; the
    holder check has tests of its own (`ACTING_AS`)."""
    local, remote = tmp_path / "notes" / slack.ROUTES_DIR, tmp_path / "published"
    local.mkdir(parents=True)
    remote.mkdir()
    monkeypatch.setenv("RESEARCH_NOTES", str(tmp_path / "notes"))
    monkeypatch.setenv("RESEARCH_ROUTES_URL", f"file://{remote}")
    monkeypatch.setenv("RESEARCH_SLACK_SEEN", str(tmp_path / "seen"))
    monkeypatch.setenv("RESEARCH_PR_BRANCHES", str(tmp_path / "pr-branches.json"))
    monkeypatch.setattr(slack, "GH", str(tmp_path / "no-gh"))
    monkeypatch.setattr(slack, "acting_as", lambda ns, reg, name: slack.name_entry(ns, reg, name))  # tests post as any name
    return local, remote


def add_route(where: Path, name: str, inbox_ts: str, **kw) -> dict:
    r = {"schema": slack.ROUTE_SCHEMA, "name": name, "holder": f"bc-{name}", "channel": COORD, "inbox_ts": inbox_ts,
         "opened": "2026-10-01T17:00Z", "closed": None, "forward_to": None, **kw}
    (where / f"{name}.json").write_text(json.dumps(r))
    return r


@pytest.fixture
def fake(monkeypatch):
    state = FakeSlack()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"])).decode()
            params = {k: v[0] for k, v in urllib.parse.parse_qs(body).items()}
            out = json.dumps(state.handle(self.path.strip("/").split("/")[-1], params, self.headers.get("Authorization", ""))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    monkeypatch.setenv("SLACK_API_URL", f"http://127.0.0.1:{srv.server_address[1]}/api/")
    monkeypatch.setenv("SLACK_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("RESEARCH_AGENT_URL", "https://cursor.com/agents/bc-test")
    yield state
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def registry(tmp_path, monkeypatch):
    reg = json.loads(REAL_REGISTRY.read_text())
    reg["bot_id"] = BOT
    reg["channels"]["agent-coordination"]["id"] = COORD
    reg["channels"]["agent-alerts"]["id"] = ALERTS
    reg["channels"]["agent-alerts"]["bots"][0]["bot_id"] = "BGRAFANA"
    reg["humans"][0]["slack_user"] = "UDANIEL"
    for h in reg["handles"]:
        h["usergroup"] = "SINFRA" if h["handle"] == "infra" else None
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(reg))
    monkeypatch.setenv("RESEARCH_SLACK_REGISTRY", str(path))
    return path


def run(capsys, *argv) -> tuple[int, str, str]:
    rc = slack.main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


def match(capsys, handle, ev) -> tuple[int, str]:
    rc, _, err = run(capsys, "match", "--as", handle, "--event", ev if isinstance(ev, str) else json.dumps(ev), "--why")
    return rc, err


def test_the_repository_registry_loads_and_seeds_every_handle_and_channel():
    reg = slack.load_registry(REAL_REGISTRY)
    assert [h["handle"] for h in reg["handles"]] == HANDLES
    assert set(reg["channels"]) == {"agent-coordination", "agent-alerts", "ask-daniel"}
    assert slack.subscribers(reg, "agent-coordination") == ["old-accounting", "old-circuits-and-proofs", "console"]
    assert slack.subscribers(reg, "agent-alerts") == slack.subscribers(reg, "ask-daniel") == ["infra"]
    assert reg["channels"]["ask-daniel"]["id"] == "C0C5UCA0S0Z" and slack.resolve_key(reg, "approvals") == "ask-daniel"


def test_the_old_channel_name_still_selects_ask_daniel_and_an_alias_may_not_shadow_a_channel(fake, registry, capsys):
    assert run(capsys, "pickup", "--channel", "approvals", "--thread", "1.0")[0] == 0
    assert fake.calls == [("reactions.add", {"channel": "C0C5UCA0S0Z", "timestamp": "1.0", "name": "eyes"})]
    reg = json.loads(registry.read_text())
    reg["channels"]["ask-daniel"]["aliases"] = ["agent-alerts"]
    registry.write_text(json.dumps(reg))
    with pytest.raises(slack.SlackError, match="alias"):
        slack.load_registry(registry)


def test_a_remit_longer_than_a_user_group_description_is_refused(registry):
    reg = json.loads(registry.read_text())
    reg["handles"][0]["remit"] = "x" * (slack.DESCRIPTION_MAX + 1)
    registry.write_text(json.dumps(reg))
    with pytest.raises(slack.SlackError, match="remit"):
        slack.load_registry(registry)


def test_handles_and_subscriptions(registry, capsys):
    rc, out, _ = run(capsys, "handles")
    assert rc == 0 and "@infra" in out and "SINFRA" in out and "#agent-alerts" in out
    rc, out, _ = run(capsys, "subscriptions", "--as", "infra")
    assert rc == 0 and f'channelId="{ALERTS}", topLevelOnly=true' in out and COORD not in out
    rc, out, _ = run(capsys, "subscriptions", "--as", "old-accounting")
    assert rc == 0 and f'channelId="{COORD}", topLevelOnly=true' in out and ALERTS not in out


def test_owners_pick_the_longest_prefix(registry, capsys, monkeypatch):
    rc, out, _ = run(capsys, "owners", "tools/research/src/research/slack.py", "tools/research/src/research/cli.py",
                     "verity/ml/circuits/ops/tc.py", "verity/ml/circuits/cut.py", "AGENTS.md", "nowhere/x")
    assert rc == 0 and [line.split() for line in out.splitlines()] == [
        ["@comms", "tools/research/src/research/slack.py"], ["@infra", "tools/research/src/research/cli.py"],
        ["@circuits", "verity/ml/circuits/ops/tc.py"], ["@proofs", "verity/ml/circuits/cut.py"],
        ["@top", "AGENTS.md"], ["-", "nowhere/x"]]
    monkeypatch.setattr("sys.stdin", io.StringIO("tools/research/src/research/slack.pyc\n./AGENTS.md\n"))
    rc, out, _ = run(capsys, "owners")
    assert [line.split()[0] for line in out.splitlines()] == ["@infra", "@top"]


def test_registry_refuses_a_path_two_handles_own(registry, capsys):
    reg = json.loads(registry.read_text())
    reg["handles"][1]["owns"] = [reg["handles"][0]["owns"][0]]
    registry.write_text(json.dumps(reg))
    rc, _, err = run(capsys, "owners", "AGENTS.md")
    assert rc == 2 and "no other handle owns" in err


def test_subscriptions_fail_visibly_without_channel_ids(registry, capsys):
    reg = json.loads(registry.read_text())
    reg["channels"]["agent-coordination"]["id"] = None
    registry.write_text(json.dumps(reg))
    rc, out, _ = run(capsys, "subscriptions", "--as", "old-accounting")
    assert rc == 2 and "id not recorded" in out


def test_ask_tags_one_owner_with_its_user_group_and_carries_no_footer(fake, registry, capsys):
    rc, out, err = run(capsys, "ask", "--as", "compute-accounting", "--to", "@infra", "--text", "4 GPUs on node 2 for 6 h, by 20:00Z. Spec: notes/x.md")
    assert rc == 0
    ts, link = out.split()
    m = fake.messages[COORD][0]
    assert m["ts"] == ts and m["username"] == "compute-accounting" and link.endswith(ts.replace(".", ""))
    assert m["text"].startswith("<!subteam^SINFRA|@infra> 4 GPUs")
    assert m["text"].endswith("Spec: notes/x.md") and "cursor.com/agents" not in m["text"]   # its name is its author
    assert TOKEN not in out + err
    assert run(capsys, "ask", "--as", "compute-accounting", "--to", "@compute-accounting", "--text", "x")[0] == 2
    assert run(capsys, "ask", "--as", "pous-worker-3", "--to", "@infra", "--text", "x")[0] == 2      # workers aren't on Slack


def test_a_post_that_went_out_stands_when_its_permalink_fails(capsys):
    """A caller that retries on a raise (the monitors' relay) would otherwise post it twice."""
    class Api:
        dry_run = False

        def call(self, method, **kw):
            if method == "chat.getPermalink":
                raise slack.SlackError("chat.getPermalink: HTTP 429 (rate limited; retry after 1 s)")
            return {"ok": True, "ts": "1700000000.000100"}

    assert slack._post(Api(), ALERTS, "infra", "hello") == "1700000000.000100"
    assert capsys.readouterr().out == "1700000000.000100 -\n"


def test_ask_without_a_group_id_uses_plain_text_and_the_body_can_come_from_a_file(fake, registry, capsys, tmp_path):
    body = tmp_path / "ask.md"
    body.write_text("a frontend for the pous panel: spec in PR 12")
    assert run(capsys, "ask", "--as", "compute-accounting", "--to", "memory-accounting", "--file", str(body))[0] == 0
    assert fake.messages[COORD][0]["text"].startswith("@memory-accounting a frontend")


def test_no_token_and_no_cursor_identity_is_an_error_and_dry_run_sends_nothing(fake, registry, capsys, monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN")
    rc, _, err = run(capsys, "ask", "--as", "memory-accounting", "--to", "infra", "--text", "x")
    assert rc == 2 and "Cursor identity socket" in err and not fake.calls                  # conftest points it at no socket
    monkeypatch.setenv("RESEARCH_SLACK_VIA", "direct")
    rc, _, err = run(capsys, "ask", "--as", "memory-accounting", "--to", "infra", "--text", "x")
    assert rc == 2 and "SLACK_BOT_TOKEN is not set" in err
    rc, _, err = run(capsys, "ask", "--as", "memory-accounting", "--to", "infra", "--text", "x", "--dry-run")
    assert rc == 0 and "dry-run: chat.postMessage" in err and not fake.calls
    monkeypatch.setenv("RESEARCH_SLACK_VIA", "carrier-pigeon")
    rc, _, err = run(capsys, "pickup", "--thread", "1.0")
    assert rc == 2 and "RESEARCH_SLACK_VIA" in err


DRY_RUNS = [
    ["ask", "--as", "memory-accounting", "--to", "infra", "--text", "x"], ["announce", "--as", "infra", "--text", "x"],
    ["reply", "--as", "memory-accounting", "--thread", "1.0", "--text", "x"], ["pickup", "--thread", "1.0"],
    ["done", "--as", "memory-accounting", "--thread", "1.0", "--link", "https://x"], ["decline", "--as", "memory-accounting", "--thread", "1.0", "--why", "no"],
    ["roster", "--thread", "1.0"], ["read", "--since", "1.0"],
    ["match", "--as", "memory-accounting", "--event", '{"text":"x","ts":"2.0","thread_ts":"1.0","channel":"CCOORD"}'],
    ["verify-author", "--event", '{"ts":"1.0","channel":"CCOORD"}'], ["groups", "sync", "--apply"],
    ["ask-daniel", "status", "7", "--wait"], ["ask-daniel", "resolve", "7", "--choice", "yes"],
]


@pytest.mark.parametrize("argv", DRY_RUNS)
def test_every_subcommand_has_a_dry_run_that_sends_nothing(fake, registry, capsys, argv):
    slack.main([*argv, "--dry-run"])
    assert not fake.calls
    assert TOKEN not in capsys.readouterr().err


@pytest.mark.parametrize("argv", DRY_RUNS)
def test_every_dry_run_via_the_broker_sends_nothing_and_fetches_no_identity(broker, registry, capsys, argv):
    slack.main([*argv, "--dry-run"])
    err = capsys.readouterr().err
    assert not broker.requests and not broker.identities and not broker.state.calls
    assert ("(via the broker)" in err or argv[0] in ("verify-author", "ask-daniel")) and OIDC not in err   # neither calls Slack


# --- the broker transport ----------------------------------------------------------------------------------------------------

OIDC = "dummy-cursor-oidc-identity"


class FakeBroker:
    """The broker's route as a transport: checks the identity and the allowlist, then answers from the FakeSlack state."""

    def __init__(self, state: FakeSlack):
        self.state, self.requests, self.identities, self.fail, self.valid = state, [], [], None, OIDC

    def oidc(self, aud: str) -> str:
        self.identities.append(aud)
        return OIDC

    def __call__(self, url: str, data: bytes, headers: dict) -> dict:
        req = json.loads(data)
        self.requests.append((url, req, headers))
        if self.fail:
            raise self.fail
        if headers.get("Authorization") != f"Bearer {self.valid}":
            raise http_error(401, {"error": "invalid_identity"})
        if req["method"] not in slack.BROKER_METHODS:
            raise http_error(403, {"error": "method_not_allowed"})
        return self.state.handle(req["method"], req["args"], f"Bearer {TOKEN}")


def http_error(code: int, body: dict | str, **headers) -> urllib.error.HTTPError:
    msg = email.message.Message()
    for k, v in headers.items():
        msg[k.replace("_", "-")] = v
    raw = body if isinstance(body, str) else json.dumps(body)
    return urllib.error.HTTPError(slack.BROKER, code, "x", msg, io.BytesIO(raw.encode()))


@pytest.fixture
def broker(fake, monkeypatch):
    b = FakeBroker(fake)
    monkeypatch.delenv("SLACK_BOT_TOKEN")
    monkeypatch.setattr(slack, "post", b)
    monkeypatch.setattr(slack, "oidc_token", b.oidc)
    return b


def test_a_not_ok_answer_s_error_and_needed_scopes_are_echoed_only_as_codes():
    answers = iter([{"ok": False, "error": "missing_scope", "needed": "chat:write,users:read.email"},
                    {"ok": False, "error": "<html>refused dummy-verity-bot</html>", "needed": "see https://example.invalid/x\n"},
                    {"ok": False, "error": "invalid_auth\n"}, {"ok": False, "error": 7}])
    api = slack.Slack(TOKEN, transport=lambda url, data, headers: next(answers))
    said = []
    for _ in range(4):
        with pytest.raises(slack.SlackError) as e:
            api.call("auth.test")
        said.append(str(e.value))
    assert said == ["auth.test: missing_scope (needed scope chat:write,users:read.email)"] + ["auth.test: not ok"] * 3


def test_the_transport_is_direct_with_a_token_and_the_broker_without_or_when_forced(monkeypatch):
    assert slack.via(TOKEN) == "direct" and slack.via(None) == "broker" and slack.via("") == "broker"
    monkeypatch.setenv("RESEARCH_SLACK_VIA", "broker")
    assert slack.via(TOKEN) == "broker" and slack.Slack(TOKEN).token is None                # the bot token never goes to the broker
    monkeypatch.setenv("RESEARCH_SLACK_VIA", "direct")
    assert slack.via(None) == "direct"


def test_with_a_token_nothing_reaches_the_broker(fake, registry, capsys, monkeypatch):
    b = FakeBroker(fake)
    monkeypatch.setattr(slack, "oidc_token", b.oidc)
    assert run(capsys, "pickup", "--thread", "1.0")[0] == 0
    assert fake.calls == [("reactions.add", {"channel": COORD, "timestamp": "1.0", "name": "eyes"})] and not b.identities


def test_without_a_token_every_call_goes_through_the_broker(broker, registry, capsys):
    rc, out, err = run(capsys, "ask", "--as", "compute-accounting", "--to", "@infra", "--text", "4 GPUs?")
    assert rc == 0 and out.split()[0] == broker.state.messages[COORD][0]["ts"]
    url, req, headers = broker.requests[0]
    assert url == slack.BROKER and broker.identities == [slack.BROKER]                      # one identity per process, aud = the route
    assert req["method"] == "chat.postMessage" and set(req) == {"method", "args"}
    assert req["args"]["channel"] == COORD and req["args"]["username"] == "compute-accounting" and req["args"]["unfurl_links"] == "false"
    assert headers == {"Authorization": f"Bearer {OIDC}", "Content-Type": "application/json"}
    assert [r[1]["method"] for r in broker.requests] == ["chat.postMessage", "chat.getPermalink"]
    assert OIDC not in out + err and TOKEN not in json.dumps([r[1:] for r in broker.requests])
    t = broker.state.messages[COORD][0]["ts"]
    assert run(capsys, "done", "--as", "infra", "--thread", t, "--link", "https://x/pull/1")[0] == 0
    assert (COORD, t, "white_check_mark") in broker.state.reactions
    assert run(capsys, "verify-author", "--event", json.dumps({"ts": t, "channel": COORD}))[0] == 0


def test_forcing_the_broker_with_a_token_present(broker, registry, capsys, monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("RESEARCH_SLACK_VIA", "broker")
    assert run(capsys, "pickup", "--thread", "1.0")[0] == 0
    (_, req, headers), = broker.requests
    assert req == {"method": "reactions.add", "args": {"channel": COORD, "timestamp": "1.0", "name": "eyes"}}
    assert TOKEN not in json.dumps([req, headers])


@pytest.mark.parametrize("fail, says", [
    (http_error(401, {"error": "invalid_identity"}), "rejected this agent's Cursor identity (HTTP 401 invalid_identity)"),
    (http_error(403, {"error": "forbidden_repo"}), "refused the call (HTTP 403 forbidden_repo)"),
    (http_error(403, {"error": f"Bearer {OIDC}"}), "refused the call (HTTP 403)"),                  # only a short code is echoed
    (http_error(403, "<html>nope</html>"), "refused the call (HTTP 403)"),
    (http_error(413, {"error": "too_large"}), "larger than the broker takes"),
    (http_error(429, {"error": "rate_limited"}, Retry_After="30"), "rate limiting (HTTP 429 rate_limited; retry after 30 s)"),
    (http_error(503, {"error": "slack_disabled"}), "down or switched off (HTTP 503 slack_disabled)"),
    (http_error(502, ""), "down or switched off (HTTP 502)"),
    (urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")), "the Slack broker is unreachable"),
])
def test_broker_failures_are_clear_exit_2_and_never_show_the_identity(broker, registry, capsys, fail, says):
    broker.fail = fail
    rc, out, err = run(capsys, "pickup", "--thread", "1.0")
    assert rc == 2 and says in err and err.startswith("research msg: reactions.add: ")
    assert OIDC not in out + err


def test_a_long_running_client_gets_a_new_identity_once_the_broker_rejects_its_cached_one(broker, monkeypatch):
    issued = iter(["t1", "t2", "t3", "t4"])
    monkeypatch.setattr(slack, "oidc_token", lambda aud: next(issued))
    auths = lambda: [h["Authorization"].split()[1] for _, _, h in broker.requests]
    api, broker.valid = slack.Slack(None), "t1"
    api.call("chat.postMessage", channel=COORD, text="first")
    broker.valid = "t2"                                                                     # t1 expired (Cursor's live 300 s)
    api.call("chat.postMessage", channel=COORD, text="second")
    assert auths() == ["t1", "t1", "t2"] and [m["text"] for m in broker.state.messages[COORD]] == ["first", "second"]
    broker.valid = "none"
    with pytest.raises(slack.SlackError, match=re.escape("rejected this agent's Cursor identity (HTTP 401 invalid_identity)")):
        api.call("chat.postMessage", channel=COORD, text="third")
    assert auths()[3:] == ["t2", "t3"]                                                     # one retry, then the error stands
    with pytest.raises(slack.SlackError):
        slack.Slack(None).call("chat.postMessage", channel=COORD, text="fresh")
    assert auths()[5:] == ["t4"]                                                           # a token just issued isn't retried


def test_a_slack_error_through_the_broker_is_slacks_own(broker, registry, capsys):
    t = broker.state.add(COORD, text="x", user="UDANIEL")["ts"]
    assert run(capsys, "pickup", "--thread", t)[0] == 0
    rc, _, err = run(capsys, "pickup", "--thread", t)
    assert rc == 2 and "reactions.add: already_reacted" in err


def test_the_client_refuses_what_the_broker_would(broker, registry, capsys):
    with pytest.raises(slack.SlackError, match="not in BROKER_METHODS"):
        slack.Slack(None).call("chat.delete", channel=COORD, ts="1.0")
    with pytest.raises(slack.SlackError, match="more than the broker takes"):
        slack.Slack(None).call("chat.postMessage", channel=COORD, text="x" * slack.BROKER_MAX_BODY)
    assert not broker.requests


def test_broker_methods_are_the_methods_this_module_calls_but_the_user_group_writes():
    src = Path(slack.__file__).read_text()
    called = set(re.findall(r"\.(?:call|paged)\(\s*\"([a-z]+\.[A-Za-z]+)\"", src))
    assert set(slack.DIRECT_ONLY) == {"chat.update", "usergroups.create", "usergroups.disable", "usergroups.enable", "usergroups.update"}
    assert called == set(slack.BROKER_METHODS) | set(slack.DIRECT_ONLY) and not set(slack.BROKER_METHODS) & set(slack.DIRECT_ONLY)
    assert len(slack.BROKER_METHODS) == len(set(slack.BROKER_METHODS))


def test_groups_sync_plans_through_the_broker_but_applies_only_with_the_token(broker, registry, capsys):
    rc, out, _ = run(capsys, "groups", "sync")
    assert rc == 0 and "plan only" in out and [r[1]["method"] for r in broker.requests] == ["auth.test", "usergroups.list"]
    broker.requests.clear()
    rc, _, err = run(capsys, "groups", "sync", "--apply")
    assert rc == 2 and "groups sync --apply needs SLACK_BOT_TOKEN" in err and "a VM that has the token" in err
    assert not broker.requests and not broker.state.groups


@pytest.fixture
def identity_socket(monkeypatch):
    """A fake cursor-agent socket: answers POST /v1/tokens/oidc with `answer` (status, body) and records each request's body."""
    state = {"answer": (200, {"token": OIDC}), "seen": [], "drop": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def handle(self):
            if state["drop"]:
                state["drop"] -= 1
                return
            super().handle()

        def do_POST(self):
            state["seen"].append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            status, body = state["answer"]
            out = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def address_string(self):
            return "unix"

        def log_message(self, *a):
            pass

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    d = tempfile.mkdtemp(prefix="sock")                                   # AF_UNIX paths are short; pytest's tmp_path may not be
    srv = Server(os.path.join(d, "api.sock"), Handler)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    monkeypatch.setenv("CURSOR_AGENT_SOCKET", srv.server_address)
    yield state
    srv.shutdown()
    srv.server_close()
    shutil.rmtree(d, ignore_errors=True)


def test_oidc_token_asks_the_cursor_socket_for_the_brokers_audience(identity_socket):
    assert slack.oidc_token(slack.BROKER) == OIDC
    assert identity_socket["seen"] == [("/v1/tokens/oidc", {"aud": slack.BROKER})]
    for answer, says in [((403, {}), "Cursor refused"), ((500, {}), "unavailable (HTTP 500)"),
                         ((200, {"token": "two words"}), "invalid identity"), ((200, {}), "invalid identity")]:
        identity_socket["answer"] = answer
        with pytest.raises(slack.SlackError, match=re.escape(says)):
            slack.oidc_token(slack.BROKER)


def test_oidc_token_asks_again_when_the_socket_drops_the_connection(identity_socket, monkeypatch, tmp_path):
    waits: list[float] = []
    identity_socket["drop"] = 2
    assert slack.oidc_token(slack.BROKER, sleep=waits.append) == OIDC
    assert waits == list(slack.OIDC_RETRY_S) and len(identity_socket["seen"]) == 1
    identity_socket["drop"] = len(slack.OIDC_RETRY_S) + 1
    with pytest.raises(slack.SlackError, match="identity socket the Slack broker needs is unavailable"):
        slack.oidc_token(slack.BROKER, sleep=waits.append)
    assert len(identity_socket["seen"]) == 1 and identity_socket["drop"] == 0
    for absent in (tmp_path / "absent.sock", tmp_path / ("x" * 120) / "api.sock"):   # past AF_UNIX's ~108 bytes too
        monkeypatch.setenv("CURSOR_AGENT_SOCKET", str(absent))
        waits.clear()
        with pytest.raises(slack.SlackError, match="not a Cursor agent"):
            slack.oidc_token(slack.BROKER, sleep=waits.append)
        assert not waits


def test_announce_to_all_or_some_with_verify_and_pins(fake, registry, capsys):
    sha = "a" * 64
    rc, _, err = run(capsys, "announce", "--as", "infra", "--text", "curl the new helper and install it", "--verify", "helper --version")
    assert rc == 2 and "pin it" in err
    rc, _, err = run(capsys, "announce", "--as", "infra", "--text", "x", "--ships", "https://x/y.py")
    assert rc == 2 and "--ships" in err
    assert not fake.calls
    assert run(capsys, "announce", "--as", "infra", "--text", "Install the broker helper.", "--verify", "git ls-remote origin HEAD",
               "--ships", f"https://x/verity-github.py#sha256={sha}")[0] == 0
    m = fake.messages[COORD][-1]
    assert m["text"].startswith(":mega: *Announcement* from @infra to every handle")
    assert slack.SAFE_POINT in m["text"] and "`git ls-remote origin HEAD`" in m["text"] and sha in m["text"]
    assert run(capsys, "announce", "--as", "infra", "--to", "@compute-accounting", "circuits", "--text", "node 2 reboots at 22:00Z")[0] == 0
    m = fake.messages[COORD][-1]
    assert m["text"].split("\n")[0].endswith("to @compute-accounting @circuits") and slack.SAFE_POINT not in m["text"]
    assert slack.audience(m["text"], slack.load_registry(registry)) == ["circuits", "compute-accounting"]


def test_match_an_ask_only_for_its_owner(fake, registry, capsys):
    ask = {"text": "<!subteam^SINFRA|@infra> GPUs?", "ts": "5.0", "channel": COORD, "username": "compute-accounting", "bot_id": BOT}
    assert match(capsys, "infra", ask)[0] == 0
    assert all(match(capsys, h, ask)[0] == 1 for h in HANDLES if h != "infra")
    assert match(capsys, "infra", {"text": "<!subteam^SOTHER|@infra> hi", "ts": "5.0"})[0] == 0
    assert match(capsys, "memory-accounting", "ping @memory-accounting, please")[0] == 0
    assert match(capsys, "infra", "email infra@infra.dev or @infra-2")[0] == 1
    assert match(capsys, "infra", {"event": {"type": "message", "text": "@infra hi", "ts": "5.0"}})[0] == 0
    assert match(capsys, "infra", {"text": "@infra note to self", "ts": "5.0", "username": "infra", "bot_id": BOT})[0] == 1
    assert not fake.calls


def test_match_announcements_to_all_and_to_some(fake, registry, capsys):
    to_all = {"text": f"{slack.ANNOUNCE} *Announcement* from @infra to every handle\nbroker", "ts": "6.0", "channel": COORD,
              "username": "infra", "bot_id": BOT}
    assert [h for h in HANDLES if match(capsys, h, to_all)[0] == 0] == [h for h in HANDLES if h != "infra"]
    to_some = {**to_all, "text": f"{slack.ANNOUNCE} *Announcement* from @infra to @compute-accounting <!subteam^SX|@circuits>\nreboot"}
    assert [h for h in HANDLES if match(capsys, h, to_some)[0] == 0] == ["circuits", "compute-accounting"]


def test_match_in_threads_participants_and_status_lines(fake, registry, capsys):
    root = fake.add(COORD, text="<!subteam^SINFRA|@infra> a pod?", username="compute-accounting", bot_id=BOT)
    fake.add(COORD, text="which node?", username="infra", bot_id=BOT, thread_ts=root["ts"])
    reply = fake.add(COORD, text="node 2", username="compute-accounting", bot_id=BOT, thread_ts=root["ts"])
    rc, why = match(capsys, "infra", {**reply, "channel": COORD})
    assert rc == 0 and "posted in it" in why
    assert match(capsys, "memory-accounting", {**reply, "channel": COORD})[0] == 1
    human = fake.add(COORD, text="any news?", user="UDANIEL", thread_ts=root["ts"])
    assert match(capsys, "infra", {**human, "channel": COORD})[0] == 0 and match(capsys, "compute-accounting", {**human, "channel": COORD})[0] == 0

    ann = fake.add(COORD, text=f"{slack.ANNOUNCE} *Announcement* from @infra to every handle\nx", username="infra", bot_id=BOT)
    done = fake.add(COORD, text=f"{slack.DONE} done \u2014 v2", username="compute-accounting", bot_id=BOT, thread_ts=ann["ts"])
    rc, why = match(capsys, "infra", {**done, "channel": COORD})
    assert rc == 0 and "status line" in why
    fake.add(COORD, text=f"{slack.DONE} done", username="circuits", bot_id=BOT, thread_ts=ann["ts"])
    assert match(capsys, "circuits", {**done, "channel": COORD})[0] == 1                 # another handle's status line: not for you
    question = fake.add(COORD, text="does this need a restart?", username="circuits", bot_id=BOT, thread_ts=ann["ts"])
    assert match(capsys, "infra", {**question, "channel": COORD})[0] == 0
    assert match(capsys, "proofs", {**question, "channel": COORD})[0] == 1                # the root was to all, but proofs isn't in the thread


def test_match_alerts_are_for_infra_only(fake, registry, capsys):
    alert = fake.add(ALERTS, text="\U0001f534 FIRING [critical] vy-nebius-2: disk 95%", bot_id="BGRAFANA", username="grafana")
    assert match(capsys, "infra", {**alert, "channel": ALERTS})[0] == 0
    assert match(capsys, "compute-accounting", {**alert, "channel": ALERTS})[0] == 1
    resolved = fake.add(ALERTS, text="\u2705 RESOLVED disk 70%", bot_id="BGRAFANA", username="grafana", thread_ts=alert["ts"])
    assert match(capsys, "infra", {**resolved, "channel": ALERTS})[0] == 0


def test_match_reads_the_event_from_a_file_or_stdin(fake, registry, capsys, tmp_path, monkeypatch):
    p = tmp_path / "ev.json"
    p.write_text(json.dumps({"text": "@memory-accounting decide", "ts": "1.0"}))
    assert match(capsys, "memory-accounting", f"@{p}")[0] == 0
    monkeypatch.setattr("sys.stdin", io.StringIO("@memory-accounting again"))
    assert match(capsys, "memory-accounting", "-")[0] == 0


def test_the_ask_lifecycle_pickup_reply_done_and_decline(fake, registry, capsys):
    root = fake.add(COORD, text="<!subteam^SINFRA|@infra> need a pod", user="UDANIEL")
    assert run(capsys, "pickup", "--thread", root["ts"])[0] == 0
    assert run(capsys, "reply", "--as", "infra", "--thread", root["ts"], "--text", "buying one needs your OK: $3/h for 6 h")[0] == 0
    assert run(capsys, "done", "--as", "infra", "--thread", root["ts"], "--link", "https://github.com/x/pull/1")[0] == 0
    assert fake.reactions == [(COORD, root["ts"], "eyes"), (COORD, root["ts"], "white_check_mark")]
    assert fake.messages[COORD][-1]["text"].startswith(":white_check_mark: done \u2014 https://github.com/x/pull/1")
    other = fake.add(COORD, text="<!subteam^SINFRA|@infra> delete node 1", username="compute-accounting", bot_id=BOT)
    assert run(capsys, "decline", "--as", "infra", "--thread", other["ts"], "--why", "a node change needs Daniel")[0] == 0
    assert (COORD, other["ts"], "x") in fake.reactions


def test_an_announcements_status_lines_and_roster(fake, registry, capsys):
    run(capsys, "announce", "--as", "infra", "--to", "@compute-accounting", "@circuits", "@proofs", "--text", "node 2 reboots at 22:00Z")
    t = fake.messages[COORD][-1]["ts"]
    assert run(capsys, "done", "--as", "compute-accounting", "--thread", t)[0] == 0
    assert run(capsys, "decline", "--as", "circuits", "--thread", t, "--why", "a job runs until 7 Oct 23:00Z")[0] == 0
    assert run(capsys, "done", "--as", "memory-accounting", "--thread", t)[0] == 0         # not addressed: not counted
    assert not [r for r in fake.reactions if r[1] == t]                                    # one bot can't carry many agents' reactions
    fake.add(COORD, text=f"{slack.DONE} done", user="UINTRUDER", thread_ts=t)              # a human's line is no status of a handle
    rc, out, _ = run(capsys, "roster", "--thread", t, "--json")
    rows = {r["handle"]: r for r in json.loads(out)}
    assert rc == 1 and set(rows) == {"compute-accounting", "circuits", "proofs"}
    detail = rows["circuits"]["detail"]                                                  # the date token's fallback: Pacific (UTC)
    assert rows["compute-accounting"]["status"] == "done" and rows["circuits"]["status"] == "declined"
    assert detail.startswith("a job runs until ") and detail.endswith(" (23:00Z)") and "<!date" not in detail
    assert rows["proofs"]["status"] == "missing"
    run(capsys, "done", "--as", "circuits", "--thread", t)
    run(capsys, "done", "--as", "proofs", "--thread", t)
    assert run(capsys, "roster", "--thread", t)[0] == 0
    ask = fake.add(COORD, text="<!subteam^SINFRA|@infra> x", username="compute-accounting", bot_id=BOT)
    assert run(capsys, "roster", "--thread", ask["ts"])[0] == 2


def test_roster_of_an_announcement_to_all_expects_every_other_handle(fake, registry, capsys):
    run(capsys, "announce", "--as", "memory-accounting", "--text", "Friday: freeze main at 18:00Z")
    t = fake.messages[COORD][-1]["ts"]
    rc, out, _ = run(capsys, "roster", "--thread", t, "--json")
    assert rc == 1 and [r["handle"] for r in json.loads(out)] == [h for h in HANDLES if h != "memory-accounting"]


def test_read_since_and_thread_oldest_first_across_pages(fake, registry, capsys):
    fake.page = 2
    a = fake.add(COORD, text="first", user="U1")
    fake.add(COORD, text="second", user="U2")
    fake.add(COORD, text="reply", user="U3", thread_ts=a["ts"])
    fake.add(COORD, text="third", username="console", bot_id=BOT)
    rc, out, _ = run(capsys, "read", "--since", a["ts"])
    assert rc == 0 and [line.split(": ", 1)[1] for line in out.splitlines()] == ["second", "third"]
    rc, out, _ = run(capsys, "read", "--thread", a["ts"], "--json")
    assert [m["text"] for m in json.loads(out)] == ["first", "reply"]


def test_read_thread_since_reads_only_the_replies_after_ts(fake, registry, capsys):
    """`--since` with `--thread` was refused, so a lane following a thread re-read all of it each time."""
    fake.page = 2
    root = fake.add(COORD, text="question", user="U1")
    seen = fake.add(COORD, text="first answer", user="U2", thread_ts=root["ts"])
    fake.add(COORD, text="elsewhere", user="U2")
    fake.add(COORD, text="second answer", user="U3", thread_ts=root["ts"])
    fake.add(COORD, text="third answer", username="console", bot_id=BOT, thread_ts=root["ts"])
    rc, out, _ = run(capsys, "read", "--thread", root["ts"], "--since", seen["ts"])
    assert rc == 0 and [line.split(": ", 1)[1] for line in out.splitlines()] == ["second answer", "third answer"]
    fake.replies_keep_root = True
    rc, out, _ = run(capsys, "read", "--thread", root["ts"], "--since", seen["ts"], "--json")
    assert rc == 0 and [m["text"] for m in json.loads(out)] == ["second answer", "third answer"]


def test_read_since_takes_an_iso_time_as_its_ts_and_refuses_anything_else(fake, registry, capsys):
    """Slack answers a date passed as `oldest` with no messages, so a sweep reading `--since 2026-…Z` saw a quiet channel."""
    root = fake.add(COORD, text="before", user="U1", ts="1793525400.000100")                  # 2026-11-01T09:30:00Z
    fake.add(COORD, text="after", user="U2", ts="1793525460.000100")
    fake.add(COORD, text="reply after", user="U3", ts="1793525470.000100", thread_ts=root["ts"])
    rc, out, _ = run(capsys, "read", "--since", "2026-11-01T09:30:30Z")
    assert rc == 0 and [line.split(": ", 1)[1] for line in out.splitlines()] == ["after"]
    rc, out, _ = run(capsys, "read", "--thread", root["ts"], "--since", "2026-11-01T01:30:30-08:00")
    assert rc == 0 and [line.split(": ", 1)[1] for line in out.splitlines()] == ["reply after"]
    for bad in ("2026-11-01T09:30:30", "yesterday"):
        with pytest.raises(SystemExit) as e:
            run(capsys, "read", "--since", bad)
        assert e.value.code == 2 and "an ISO time with its zone" in capsys.readouterr().err


def test_a_post_carries_slack_date_tokens_and_read_shows_both_zones(fake, registry, capsys):
    text = "freeze 2026-11-01T09:30Z, see note:20260930T2020Z-handoff-infra and `--since 2026-11-01T09:30Z`"
    assert run(capsys, "ask", "--as", "infra", "--to", "@memory-accounting", "--text", text)[0] == 0
    posted = fake.messages[COORD][0]["text"]
    assert "freeze <!date^1793525400^{date_short_pretty} {time}|1:30 AM PST (09:30Z)>," in posted
    assert "note:20260930T2020Z-handoff-infra" in posted and "`--since 2026-11-01T09:30Z`" in posted
    fake.messages[COORD][0]["ts"] = "1793525400.000100"
    rc, out, _ = run(capsys, "read", "--thread", "1793525400.000100")
    assert rc == 0 and out.startswith("1793525400.000100 @infra: @memory-accounting freeze 1:30 AM PST (09:30Z), see note:")


def test_verify_author_fetches_the_message_and_fails_closed(fake, registry, capsys):
    ours = fake.add(COORD, text="x", username="infra", bot_id=BOT)
    renamed = fake.add(COORD, text="x", username="not-a-handle", bot_id=BOT)
    other_bot = fake.add(COORD, text="x", username="infra", bot_id="BOTHER")
    human = fake.add(COORD, text="x", user="UDANIEL")
    stranger = fake.add(COORD, text="x", user="USTRANGER")
    alert = fake.add(ALERTS, text="disk 95%", bot_id="BGRAFANA", username="grafana")

    def verify(ev, *extra):
        return run(capsys, "verify-author", "--event", json.dumps(ev), "--why", *extra)[0]

    assert verify({"ts": ours["ts"], "channel": COORD}) == 0
    assert verify({"ts": renamed["ts"], "channel": COORD}) == 1
    assert verify({"ts": other_bot["ts"], "channel": COORD, "bot_id": BOT}) == 1             # the payload's claim is not believed
    assert verify({"ts": human["ts"], "channel": COORD}) == 0
    assert verify({"ts": stranger["ts"], "channel": COORD}) == 1
    assert verify({"ts": alert["ts"], "channel": ALERTS}) == 0
    assert verify({"ts": "9.9", "channel": COORD}) == 1
    assert verify({"ts": ours["ts"], "channel": "CUNKNOWN"}, "--channel", "agent-coordination") == 1


def test_verify_author_allows_nobody_when_the_registry_has_no_ids(fake, registry, capsys):
    reg = json.loads(registry.read_text())
    reg["humans"][0]["slack_user"], reg["bot_id"] = None, None
    registry.write_text(json.dumps(reg))
    for m in (fake.add(COORD, text="x", user="UDANIEL"), fake.add(COORD, text="x", username="infra", bot_id=BOT)):
        assert run(capsys, "verify-author", "--event", json.dumps({"ts": m["ts"], "channel": COORD}))[0] == 1


def test_groups_sync_plans_by_default_then_creates_updates_enables_and_records_ids(fake, registry, capsys):
    reg = json.loads(registry.read_text())
    reg["bot_id"] = None
    registry.write_text(json.dumps(reg))
    fake.groups = [{"id": "SINFRA", "name": "infra", "handle": "infra", "description": "old", "date_delete": 0},
                   {"id": "SPROOF", "name": "proofs", "handle": "proofs", "description": "", "date_delete": 1700000000}]
    before = registry.read_text()
    rc, out, _ = run(capsys, "groups", "sync")
    assert rc == 0 and "create  @compute-accounting" in out and "update  @infra" in out and "enable  @proofs" in out
    assert "plan only" in out and registry.read_text() == before
    assert {m for m, _ in fake.calls} == {"auth.test", "usergroups.list"}
    rc, out, _ = run(capsys, "groups", "sync", "--apply")
    assert rc == 0 and f"record  bot {BOT}" in out and "wrote" in out
    assert json.loads(registry.read_text())["bot_id"] == BOT
    ids = {h["handle"]: h["usergroup"] for h in json.loads(registry.read_text())["handles"]}
    assert len(fake.groups) == len(HANDLES) and ids == {g["handle"]: g["id"] for g in fake.groups}
    assert all(not g["date_delete"] for g in fake.groups)
    rc, out, _ = run(capsys, "groups", "sync", "--apply")
    assert out.count("ok      @") == len(HANDLES) and "record" not in out and "wrote" not in out


def test_groups_sync_disables_a_retired_names_group_and_leaves_other_groups_alone(fake, registry, routes_dirs, capsys):
    add_route(routes_dirs[0], "pr-captain", "1.0", closed="2026-10-09T01:11Z", forward_to="top")
    fake.groups = [{"id": "SPRCAPTAIN", "name": "pr-captain", "handle": "pr-captain", "description": "", "date_delete": 0},
                   {"id": "SOTHER", "name": "design", "handle": "design", "description": "", "date_delete": 0}]
    rc, out, _ = run(capsys, "groups", "sync")
    assert rc == 0 and "disable @pr-captain (SPRCAPTAIN): retired, its inbox closed 2026-10-09T01:11Z and forwards to @top" in out
    assert "design" not in out and all(not g["date_delete"] for g in fake.groups)
    assert run(capsys, "groups", "sync", "--apply")[0] == 0
    assert {g["handle"]: bool(g["date_delete"]) for g in fake.groups if g["id"] in ("SPRCAPTAIN", "SOTHER")} == {"pr-captain": True, "design": False}
    rc, out, _ = run(capsys, "groups", "sync")
    assert "disable" not in out


def test_research_slack_is_a_cli_subcommand(registry, capsys):
    from research.cli import main

    assert main(["slack", "handles", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == slack.SCHEMA


# --- #ask-daniel -------------------------------------------------------------------------------------------------------------

NOW = datetime(2026, 9, 30, 20, 45, tzinfo=timezone.utc)                  # 1:45 PM PDT
ASK = "C0C5UCA0S0Z"


class FakeRoute:
    """The site's /api/agent-approvals: `fields` (the card's fields are deployed) or `old` (only {title, text, link, requested_by},
    and a 400 for anything without a title); `resolve` is deployed or answers 404. `statuses` are what successive GETs return."""

    def __init__(self):
        self.mode, self.resolve, self.requests, self.statuses, self.answer = "fields", False, [], [], None
        self.auths, self.valid = [], OIDC

    def handle(self, method: str, path: str, body: dict | None, auth: str) -> tuple[int, dict]:
        self.requests.append((method, path, body))
        self.auths.append(auth)
        if self.answer:
            return self.answer
        if auth != f"Bearer {self.valid}":
            return 401, {"error": "invalid_identity"}
        if method == "POST" and path == "/api/agent-approvals":
            if self.mode == "old" and not body.get("title"):
                return 400, {"error": "title_required"}
            return 200, {"id": "42", "ts": "1790800000.000100", **({"channel": ASK} if self.mode == "fields" else {})}
        if method == "POST" and path.endswith("/resolve"):
            return (200, {"status": "decided", "choice": body["choice"], "source": "thread"}) if self.resolve else (404, {"error": "not_found"})
        if method == "GET":
            return 200, self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return 404, {"error": "not_found"}


@pytest.fixture
def route(identity_socket, monkeypatch):
    state = FakeRoute()

    class Handler(http.server.BaseHTTPRequestHandler):
        def _answer(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n)) if n else None
            status, out = state.handle(self.command, self.path, body, self.headers.get("Authorization", ""))
            raw = json.dumps(out).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        do_GET = do_POST = _answer

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    monkeypatch.setenv("RESEARCH_ASK_DANIEL_URL", f"http://127.0.0.1:{srv.server_address[1]}/api/agent-approvals")
    monkeypatch.setenv("RESEARCH_AGENT_URL", "https://cursor.com/agents/bc-test")
    monkeypatch.setattr(slack, "_now", lambda: NOW)
    state.identity = identity_socket
    yield state
    srv.shutdown()
    srv.server_close()


def card_argv(*extra: str, kind: str = "default", deadline: str = "3:00 PM") -> list[str]:
    return ["ask-daniel", "--as", "infra", "--kind", kind, "--question", "Test: does the #ask-daniel card render correctly?",
            "--why", "First live card; checks the format.", "--option", "yes=Looks right", "--option", "no=Needs changes",
            "--recommend", "yes", "--default", "yes", "--deadline", deadline, "--link", "https://github.com/x/note.md", *extra]


def test_ask_daniel_posts_the_cards_fields_with_the_routes_identity(route, registry, capsys):
    rc, out, err = run(capsys, *card_argv())
    assert rc == 0 and out.split() == ["42", "1790800000.000100", f"https://computeverification.slack.com/archives/{ASK}/p1790800000000100"]
    (method, path, body), = route.requests
    assert (method, path) == ("POST", "/api/agent-approvals")
    assert body == {"kind": "default", "question": "Test: does the #ask-daniel card render correctly?", "why": "First live card; checks the format.",
                    "options": [{"value": "yes", "label": "Looks right", "recommended": True}, {"value": "no", "label": "Needs changes"}],
                    "default": "yes", "deadline": "2026-09-30T22:00:00Z", "link": "https://github.com/x/note.md", "requested_by": "infra",
                    "agent_url": "https://cursor.com/agents/bc-test"}
    assert route.identity["seen"][0] == ("/v1/tokens/oidc", {"aud": slack.ASK_DANIEL})
    assert "the card's fields" in err and f"threadTs=1790800000.000100" in err and OIDC not in out + err


def test_ask_daniel_mentions_him_on_the_card_so_slack_notifies_him(route, fake, registry, capsys):
    rc, _, err = run(capsys, *card_argv())
    human = json.loads(registry.read_text())["humans"][0]["slack_user"]
    (ping,) = [m for m in fake.messages.get(ASK, []) if m.get("thread_ts") == "1790800000.000100"]
    assert rc == 0 and ping["username"] == "infra" and ping["text"].startswith(f"<@{human}> a decision for you: Test: does the")
    assert "I recommend: Looks right. `yes` applies at 3:00 PM PDT" in ping["text"]


def test_send_to_daniel_mentions_him_and_nobody_posts_as_him(fake, registry, capsys):
    human = json.loads(registry.read_text())["humans"][0]["slack_user"]
    rc, out, _ = run(capsys, "send", "--as", "infra", "--to", "@daniel", "--text", "node 2 or node 3? I recommend node 3; node 3 by 5 PM PDT")
    assert rc == 0 and posted(fake, out)["text"].startswith(f"<@{human}> node 2 or node 3?")
    assert run(capsys, "send", "--as", "daniel", "--to", "@infra", "--text", "x")[0] == 2


def test_a_route_without_the_fields_gets_the_old_shape_with_the_card_in_its_text(route, registry, capsys):
    route.mode = "old"
    rc, out, err = run(capsys, *card_argv())
    assert rc == 0 and out.split()[:2] == ["42", "1790800000.000100"] and ASK in out            # the registry's channel id
    assert [r[2].get("title") for r in route.requests] == [None, "Test: does the #ask-daniel card render correctly?"]
    old = route.requests[1][2]
    assert set(old) == {"title", "text", "link", "requested_by"} and old["requested_by"] == "infra"
    assert old["text"].splitlines()[:5] == ["*DEFAULT*", "*Why:* First live card; checks the format.", "*Options:*",
                                            "\u2022 `yes`: Looks right (recommended)", "\u2022 `no`: Needs changes"]
    assert "*Default:* Looks right (`yes`), which applies at the deadline" in old["text"]
    assert "*Deadline:* 3:00 PM PDT, Wed 30 Sep" in old["text"] and "reply in the thread" in old["text"]
    assert "HTTP 400 title_required" in err and "the old shape" in err and "`yes` applies, and you record it" in err


def test_a_card_polled_past_its_identitys_life_gets_a_new_one(route, monkeypatch):
    issued = iter(["t1", "t2"])
    monkeypatch.setattr(slack, "oidc_token", lambda aud: next(issued))
    route.statuses, route.valid = [{"status": "pending"}], "t1"
    ask = slack.AskDaniel()
    assert ask.call("GET", "/42")["status"] == "pending"
    route.valid = "t2"
    assert ask.call("GET", "/42")["status"] == "pending"
    assert route.auths == ["Bearer t1", "Bearer t1", "Bearer t2"]


def test_only_a_400_falls_back(route, registry, capsys):
    route.answer = (401, {"error": "invalid_identity"})
    rc, _, err = run(capsys, *card_argv())
    assert rc == 2 and "rejected this agent's Cursor identity (HTTP 401 invalid_identity)" in err and len(route.requests) == 1


@pytest.mark.parametrize("change, says", [
    ({"--question": "two\nlines"}, "one line"),
    ({"--option": None}, "2 to 5 options"),
    ({"--recommend": "maybe"}, "--recommend 'maybe' is not an option"),
    ({"--default": "maybe"}, "--default 'maybe' is not an option"),
    ({"--deadline": "1:00 PM"}, "has passed"),
    ({"--deadline": "2026-10-08T21:00:00Z"}, "more than 7 days"),
    ({"--deadline": "2026-09-30T23:00:00"}, "needs its zone"),
    ({"--deadline": "tomorrow"}, "give `3:00 PM`"),
    ({"--deadline": "13:00 PM"}, "not a clock time"),
    ({"--link": "http://x"}, "an https URL"),
    ({"--link": None}, "needs --link"),
    ({"--as": "pous-worker-3"}, "unknown handle"),
])
def test_ask_daniel_checks_the_card_before_sending_anything(route, registry, capsys, change, says):
    argv = card_argv()
    for flag, value in change.items():
        i = argv.index(flag)
        if flag == "--option":
            del argv[argv.index("--option", i + 1):argv.index("--option", i + 1) + 2]
        elif value is None:
            del argv[i:i + 2]
        else:
            argv[i + 1] = value
    rc, _, err = run(capsys, *argv)
    assert rc == 2 and says in err and not route.requests and not route.identity["seen"]


def test_options_are_two_to_five_unique_values_with_button_sized_labels(route, registry, capsys):
    many = [x for v in "abcde" for x in ("--option", f"{v}=Option {v}")]
    assert run(capsys, *card_argv(*many))[0] == 2                                           # 7 with the two defaults
    rc, _, err = run(capsys, *card_argv("--option", "yes=Again"))
    assert rc == 2 and "appears twice" in err
    rc, _, err = run(capsys, *card_argv("--option", "c=" + "x" * (slack.LABEL_MAX + 1)))
    assert rc == 2 and "the label 1 to 75" in err and not route.requests


def test_deadlines_are_pacific_clock_times_today_or_zoned_iso():
    assert slack.parse_deadline("3:00 PM", NOW) == datetime(2026, 9, 30, 22, 0, tzinfo=timezone.utc)
    assert slack.parse_deadline("11:59 pm", NOW) == datetime(2026, 10, 1, 6, 59, tzinfo=timezone.utc)
    assert slack.parse_deadline("12:00 AM", NOW) == datetime(2026, 9, 30, 7, 0, tzinfo=timezone.utc)
    assert slack.parse_deadline("2026-09-30T16:00:00-07:00", NOW) == datetime(2026, 9, 30, 23, 0, tzinfo=timezone.utc)
    winter = datetime(2026, 12, 1, 18, 0, tzinfo=timezone.utc)                              # PST, UTC-8
    assert slack.parse_deadline("3:00 PM", winter) == datetime(2026, 12, 1, 23, 0, tzinfo=timezone.utc)
    assert slack.pacific(winter) == "10:00 AM PST, Tue 01 Dec"


def test_wait_polls_with_the_injected_sleeper_until_decided(route, registry, capsys, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(slack, "_sleep", slept.append)
    route.statuses = [{"status": "pending"}, {"status": "pending"},
                      {"status": "decided", "choice": "yes", "source": "default", "decided_at": "2026-09-30T22:00:05Z"}]
    rc, out, _ = run(capsys, *card_argv("--wait", "--every", "60"))
    assert rc == 0 and out.splitlines()[1] == "decided yes default 2026-09-30T22:00:05Z" and slept == [60, 60]
    assert [r[:2] for r in route.requests] == [("POST", "/api/agent-approvals")] + [("GET", "/api/agent-approvals/42")] * 3


def test_wait_gives_up_after_the_deadline_and_its_grace(route, registry, capsys, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(slack, "_sleep", slept.append)
    route.statuses = [{"status": "pending"}]
    rc, out, _ = run(capsys, *card_argv("--wait", "--every", "600", kind="blocking", deadline="2:45 PM"))
    assert rc == 1 and out.splitlines()[1].startswith("pending")
    assert len(slept) == 6 and sum(r[0] == "GET" for r in route.requests) == 7                   # (60 + 5) min in 10-minute polls


def test_status_reads_the_decision_and_exits_0_only_once_decided(route, registry, capsys):
    route.statuses = [{"status": "overdue"}]
    assert run(capsys, "ask-daniel", "status", "42")[:2] == (1, "overdue - - -\n")
    route.statuses = [{"status": "approved", "decided_at": "2026-09-30T21:00:00Z"}]
    rc, out, _ = run(capsys, "ask-daniel", "status", "42", "--json")
    assert rc == 0 and json.loads(out)["status"] == "approved"
    assert run(capsys, "ask-daniel", "status")[0] == 2


def test_status_404_says_no_card_has_that_id(route, registry, capsys):
    route.answer = (404, {"error": "not_found"})
    rc, _, err = run(capsys, "ask-daniel", "status", "1790800000.000100")
    assert rc == 2 and "HTTP 404: no card has that ID" in err and "not the card's Slack ts" in err


def test_resolve_records_a_thread_answer_and_says_clearly_when_there_is_no_such_card(route, registry, capsys):
    rc, _, err = run(capsys, "ask-daniel", "resolve", "42", "--choice", "no", "--note", "Daniel in the thread: needs changes")
    assert rc == 2 and "HTTP 404: no card has that ID" in err and "record Daniel's thread reply as a ruling" in err
    route.resolve = True
    rc, out, _ = run(capsys, "ask-daniel", "resolve", "42", "--choice", "no", "--note", "needs changes")
    assert rc == 0 and out == "decided no thread -\n"
    assert route.requests[-1] == ("POST", "/api/agent-approvals/42/resolve", {"choice": "no", "note": "needs changes"})
    assert run(capsys, "ask-daniel", "resolve", "42")[0] == 2
    route.answer = (403, {"error": "not_your_card"})
    rc, _, err = run(capsys, "ask-daniel", "resolve", "42", "--choice", "no")
    assert rc == 2 and "only a card's own asker" in err


def test_ask_daniels_dry_run_prints_both_shapes_and_sends_nothing(route, registry, capsys):
    rc, out, err = run(capsys, *card_argv("--dry-run"))
    assert rc == 0 and not out and not route.requests and not route.identity["seen"]
    new, old = [json.loads(line[line.index("{"):]) for line in err.splitlines()]
    assert new["kind"] == "default" and old["title"] == new["question"]


# ---- routing: `@name` at the start of a post rings that name's doorbell ---------------------------------------------------------------

VECTORS = Path(__file__).with_name("slack_routing_vectors.json")


@pytest.mark.parametrize("v", json.loads(VECTORS.read_text())["vectors"], ids=lambda v: v["name"])
def test_the_routing_rule_matches_every_vector(v):
    doc = json.loads(VECTORS.read_text())
    assert slack.route_targets(v["msg"], v["root"], doc["registry"], set(doc["inboxes"]), v.get("history", []), v.get("branches")) == v["targets"]


def bells(fake, inbox_ts):
    return [m for m in fake.messages.get(COORD, []) if m.get("thread_ts") == inbox_ts and m["ts"] != inbox_ts]


def inbox(fake, routes_dirs, name, where="local"):
    root = fake.add(COORD, text=f"{slack.INBOX_HEAD}{name}. pointers", username=name, bot_id=BOT)
    add_route(routes_dirs[0] if where == "local" else routes_dirs[1], name, root["ts"])
    return root["ts"]


def test_deliver_rings_for_a_persons_typed_post_once(fake, registry, routes_dirs, capsys, monkeypatch):
    monkeypatch.setattr(slack, "_now", lambda: datetime.fromtimestamp(1700000100, timezone.utc))
    box = inbox(fake, routes_dirs, "circuits")
    human = json.loads(registry.read_text())["humans"][0]["slack_user"]
    post = fake.add(COORD, text="@circuits is node 2 free?", user=human)
    for _ in range(2):
        assert run(capsys, "deliver", "--ts", post["ts"])[0] == 0
    (bell,) = bells(fake, box)
    assert bell["text"].startswith(f"from daniel, thread {post['ts']}: @circuits is node 2 free?")


def test_send_names_each_addressee_once(fake, registry, routes_dirs, capsys):
    rc, out, _ = run(capsys, "send", "--as", "infra", "--to", "@circuits", "--text", "@circuits #712 is ready")
    assert rc == 0 and posted(fake, out)["text"].startswith("@circuits #712 is ready")
    rc, out, _ = run(capsys, "send", "--as", "infra", "--to", "@circuits", "--text", "@proofs see this")
    assert rc == 0 and posted(fake, out)["text"].startswith("@circuits @proofs see this")
    rc, out, _ = run(capsys, "send", "--as", "infra", "--to", "@circuits", "--to", "@proofs", "--text", "both")
    assert rc == 0 and posted(fake, out)["text"].startswith("@circuits @proofs both")
    assert run(capsys, "announce", "--as", "infra", "--to", "@circuits", "--to", "@proofs", "--text", "x")[0] == 0
    assert fake.messages[COORD][-1]["text"].split("\n")[0].endswith("to @circuits @proofs")


def test_a_reply_to_a_replys_ts_lands_in_its_thread(fake, registry, routes_dirs, capsys):
    box = inbox(fake, routes_dirs, "circuits")
    root = fake.add(COORD, text="@infra a question", username="circuits", bot_id=BOT)
    first = fake.add(COORD, text="an answer", username="infra", bot_id=BOT, thread_ts=root["ts"])
    rc, out, _ = run(capsys, "send", "--as", "infra", "--thread", first["ts"], "--text", "and more")
    assert rc == 0 and posted(fake, out)["thread_ts"] == root["ts"]
    assert bells(fake, box)[-1]["text"].startswith(f"from infra, thread {root['ts']}, reply ")


def test_a_doorbell_carries_the_post_so_its_reader_needs_no_second_read(fake, registry, routes_dirs, capsys):
    box = inbox(fake, routes_dirs, "circuits")
    root = fake.add(COORD, text="@infra a question", username="circuits", bot_id=BOT)
    body = "node 2 is free &lt;from 3 PM&gt;, see <https://x/y|the plan> and <https://x/z>\n\n\n" + "a" * 700
    assert run(capsys, "send", "--as", "infra", "--thread", root["ts"], "--text", body)[0] == 0
    (bell,) = bells(fake, box)
    reply = next(m for m in fake.messages[COORD] if m.get("thread_ts") == root["ts"])
    head = f"from infra, thread {root['ts']}, reply {reply['ts']}: node 2 is free <from 3 PM>, see the plan (https://x/y) and https://x/z\n\naaa"
    assert bell["text"].startswith(head) and bell["text"].endswith(f"\u2026 (the rest: `research msg read --thread {root['ts']}`)")
    assert len(bell["text"]) < len(head) + slack.DOORBELL_MAX
    old = "from infra: https://x/p1700000001000001 @circuits hi\n\n_posted by <https://cursor.com/agents/bc-x|@infra's agent> \u00b7 ask_"
    assert slack._rang(old, "1700000001.000001", "https://x/p1700000001000001") and slack.tags_of(old) == ["ask"]
    assert slack.plain(old) == "from infra: https://x/p1700000001000001 @circuits hi"
    rc, out, _ = run(capsys, "read", "--as", "circuits")
    assert rc == 0 and out.split("\n", 1)[1].strip() == "- " + slack.plain(bell["text"])


def test_an_ask_rings_the_owners_doorbell_once_and_the_doorbell_cannot_route(fake, registry, routes_dirs, capsys):
    box = inbox(fake, routes_dirs, "circuits")
    rc, out, err = run(capsys, "ask", "--as", "infra", "--to", "@circuits", "--text", "can node 2 take a job?")
    assert rc == 0 and "route @circuits: rang" in err
    (bell,) = bells(fake, box)
    ts, link = out.split()
    assert bell["username"] == "router" and bell["text"] == f"from infra, thread {ts}: @circuits can node 2 take a job?"
    assert slack.route_targets(bell, fake.messages[COORD][0], json.loads(registry.read_text()), {box}) == []
    reg = json.loads(registry.read_text())
    slack.deliver(slack.client(argparse_ns()), reg, slack.Routes(), COORD, next(m for m in fake.messages[COORD] if m["ts"] == ts), link,
                  now=float(ts) + 60)
    assert len(bells(fake, box)) == 1                                                  # one doorbell per (post, target)


def test_a_reply_in_the_merge_thread_rings_the_owners_of_the_prs_it_names_and_no_one_else(fake, registry, routes_dirs, capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(slack, "_now", lambda: datetime.fromtimestamp(1700000100, timezone.utc))
    boxes = {n: inbox(fake, routes_dirs, n) for n in ("circuits", "proofs", "compute-accounting")}
    (tmp_path / "pr-branches.json").write_text(json.dumps({"1585": "cursor/gemm-rows-8c79"}))
    root = fake.add(COORD, text="\U0001f4e3 *Announcement* from @infra to every handle\nmerge thread", username="infra", bot_id=BOT)
    fake.add(COORD, text="#1600 is ready for the next train", username="proofs", bot_id=BOT, thread_ts=root["ts"])
    rc, _, err = run(capsys, "send", "--as", "infra", "--thread", root["ts"], "--text", "Landed #1585; #1600 failed its suites")
    assert rc == 0 and "route: PR owners unknown for #1600" in err
    assert [len(bells(fake, b)) for b in boxes.values()] == [1, 1, 0]                 # #1585's owner, #1600's namer; not compute-accounting
    rc, _, err = run(capsys, "send", "--as", "infra", "--thread", root["ts"], "--text", "the next train starts at 3 PM")
    assert rc == 0 and [len(bells(fake, b)) for b in boxes.values()] == [1, 1, 0]


def test_pr_branches_asks_gh_once_per_unknown_number_and_keeps_its_answers(tmp_path, monkeypatch):
    import sys
    calls = tmp_path / "calls"
    gh = tmp_path / "gh"
    gh.write_text(f"#!{sys.executable}\nimport json, pathlib\np = pathlib.Path({str(calls)!r})\np.write_text(p.read_text() + 'x' if p.exists() else 'x')\n"
                  "print(json.dumps([{'number': 1585, 'headRefName': 'cursor/gemm-rows-8c79'}]))\n")
    gh.chmod(0o755)
    monkeypatch.setattr(slack, "GH", str(gh))
    assert slack.pr_branches({"1585"}) == {"1585": "cursor/gemm-rows-8c79"}
    assert slack.pr_branches({"1585"}) == {"1585": "cursor/gemm-rows-8c79"} and calls.read_text() == "x"
    log = io.StringIO()
    assert slack.pr_branches({"1585", "1999"}, log) == {"1585": "cursor/gemm-rows-8c79"} and calls.read_text() == "xx"
    assert log.getvalue() == ""
    gh.write_text("#!/bin/sh\necho 'HTTP 401: Bad credentials' >&2\nexit 1\n")
    assert slack.pr_branches({"1999"}, log) == {}
    assert "PR owners unknown for #1999 (gh: HTTP 401: Bad credentials)" in log.getvalue()


def test_as_takes_only_a_name_this_agent_holds(fake, registry, routes_dirs, capsys, monkeypatch):
    monkeypatch.setattr(slack, "acting_as", ACTING_AS)
    reg = json.loads(registry.read_text())
    held = {h["handle"]: slack.AGENT_ID_RE.findall(h["holder"]) for h in reg["handles"]}
    monkeypatch.setenv("RESEARCH_AGENT_URL", f"https://cursor.com/agents/{held['infra'][0]}")
    assert run(capsys, "send", "--as", "infra", "--text", "node 2 is free")[0] == 0
    before = len(fake.messages[COORD])
    rc, _, err = run(capsys, "send", "--as", "ci", "--thread", fake.messages[COORD][0]["ts"], "--text", "x")
    assert rc == 2 and f"@ci is held by {', '.join(held['ci'])}, and this agent is {held['infra'][0]}" in err
    rc, _, err = run(capsys, "close", "--as", "ci", "--thread", fake.messages[COORD][0]["ts"])
    assert rc == 2 and "post as your own handle" in err and len(fake.messages[COORD]) == before
    lander = next(a for a in held["old-circuits-and-proofs"] if len(a) > 11)
    monkeypatch.setenv("RESEARCH_AGENT_URL", f"https://cursor.com/agents/{lander}")
    assert "bc-8ece7cde" in held["proofs"] and run(capsys, "send", "--as", "proofs", "--text", "the train is up")[0] == 0
    add_route(routes_dirs[0], "circuits", "1700000900.000200", holder="bc-11111111-2222-3333-4444-555555555555")
    monkeypatch.setenv("RESEARCH_AGENT_URL", "https://cursor.com/agents/bc-11111111-2222-3333-4444-555555555555")
    assert run(capsys, "send", "--as", "circuits", "--text", "a successor's first post")[0] == 0
    monkeypatch.delenv("RESEARCH_AGENT_URL")
    monkeypatch.setattr(slack, "self_agent_id", lambda: None)
    rc, _, err = run(capsys, "send", "--as", "infra", "--text", "x")
    assert rc == 2 and "this agent is unknown (set RESEARCH_AGENT_URL)" in err
    rc, _, err = run(capsys, "send", "--dry-run", "--as", "infra", "--text", "x")
    assert rc == 0 and "a real --as infra would be refused" in err


def test_retract_blanks_only_the_names_own_post(fake, registry, capsys):
    rc, out, _ = run(capsys, "send", "--as", "infra", "--text", "x")
    ts = out.split()[0]
    reply = fake.add(COORD, text="a stray reply", username="infra", bot_id=BOT, thread_ts=ts)
    other = fake.add(COORD, text="mine", username="ci", bot_id=BOT)
    assert run(capsys, "retract", "--as", "infra", "--ts", ts, "--why", "sent by mistake")[0] == 0
    assert next(m for m in fake.messages[COORD] if m["ts"] == ts)["text"] == "_retracted by @infra_: sent by mistake"
    fake.replies_keep_root = True
    assert run(capsys, "retract", "--as", "infra", "--ts", reply["ts"], "--thread", ts)[0] == 0
    assert reply["text"] == "_retracted by @infra_"
    rc, _, err = run(capsys, "retract", "--as", "infra", "--ts", other["ts"])
    assert rc == 2 and "was posted by @ci, not @infra" in err and other["text"] == "mine"
    rc, _, err = run(capsys, "retract", "--as", "infra", "--ts", "1700000999.000001")
    assert rc == 2 and "a reply needs --thread ROOT" in err


def argparse_ns(**kw):
    import argparse
    return argparse.Namespace(dry_run=False, registry=None, channel=None, channel_id=None, **kw)


def test_a_handle_without_an_inbox_is_left_to_top_level_and_an_unknown_name_is_refused(fake, registry, routes_dirs, capsys):
    rc, _, err = run(capsys, "ask", "--as", "infra", "--to", "@proofs", "--text", "x")
    assert rc == 0 and "route @proofs: no inbox (top-level forwards it)" in err
    rc, _, err = run(capsys, "ask", "--as", "infra", "--to", "@nobody-here", "--text", "x")
    assert rc == 2 and "neither a registry handle nor a name with an inbox" in err and "names open --as nobody-here" in err


def test_a_worker_with_a_published_inbox_can_post_and_be_addressed(fake, registry, routes_dirs, capsys):
    box = inbox(fake, routes_dirs, "vllm-epoch-run", where="published")               # registered after this clone was pulled
    rc, _, err = run(capsys, "ask", "--as", "circuits", "--to", "@vllm-epoch-run", "--urgent", "--text", "stop using node 2")
    assert rc == 0 and "route @vllm-epoch-run: rang" in err
    assert bells(fake, box)[0]["text"].startswith(f"from circuits (URGENT), thread {fake.messages[COORD][1]['ts']}: @vllm-epoch-run stop using node 2")
    rc, out, err = run(capsys, "reply", "--as", "vllm-epoch-run", "--thread", fake.messages[COORD][1]["ts"], "--text", "stopped")
    assert rc == 0 and "route @circuits: no inbox" in err


def test_a_reply_rings_the_root_poster_and_a_quiet_one_rings_no_one(fake, registry, routes_dirs, capsys):
    box = inbox(fake, routes_dirs, "infra")
    root = fake.add(COORD, text="<!subteam^SINFRA|@infra> can node 2 take a job?", username="circuits", bot_id=BOT)
    assert run(capsys, "reply", "--as", "infra", "--thread", root["ts"], "--text", "yes")[0] == 0
    assert bells(fake, box) == []                                                      # the replier is not its own target
    circuits = inbox(fake, routes_dirs, "circuits")
    assert run(capsys, "reply", "--as", "infra", "--thread", root["ts"], "--text", "from 3 PM")[0] == 0
    assert len(bells(fake, circuits)) == 1
    assert run(capsys, "reply", "--as", "infra", "--thread", root["ts"], "--quiet", "--text", "thanks")[0] == 0
    assert len(bells(fake, circuits)) == 1
    assert run(capsys, "reply", "--as", "infra", "--thread", root["ts"], "--quiet", "--text", "@circuits started, no reply needed")[0] == 0
    assert len(bells(fake, circuits)) == 2                                             # a quiet post still reaches the names it starts with
    assert run(capsys, "done", "--as", "infra", "--thread", root["ts"], "--link", "https://x/pr/1")[0] == 0
    assert len(bells(fake, circuits)) == 3                                             # a status line reaches the root's poster


def test_a_long_post_goes_out_whole_with_its_rest_as_quiet_replies(fake, registry, routes_dirs, capsys):
    box = inbox(fake, routes_dirs, "infra")
    lines = [f"*@infra:* row {i} " + "x" * 90 for i in range(80)] + ["see <!date^1791361800^{time}|1:30 AM PDT (08:30Z)> -> " + "y " * 900]
    assert run(capsys, "send", "--as", "circuits", "--to", "@infra", "--text", "\n".join(lines))[0] == 0
    root = fake.messages[COORD][1]
    rest = [m for m in fake.messages[COORD] if m.get("thread_ts") == root["ts"]]
    assert len(rest) >= 2 and not [m for m in fake.messages[COORD] if not m.get("thread_ts") and m["ts"] not in (box, root["ts"])]
    assert all(len(m["text"].encode()) <= slack.POST_MAX + 20 for m in [root, *rest])
    assert all(m["text"].startswith(f"({i}/{len(rest) + 1}) ") and slack.tags_of(m["text"]) == [slack.QUIET]
               for i, m in enumerate(rest, 2))
    body = "\n".join(slack.FOOTER_RE.sub("", re.sub(r"^\(\d+/\d+\) ", "", m["text"])) for m in [root, *rest])
    assert body.split() == f"<!subteam^SINFRA|@infra> {chr(10).join(lines)}".split()
    assert "<!date^1791361800^{time}|1:30 AM PDT (08:30Z)>" in body
    assert len(bells(fake, box)) == 1


def test_send_refuses_a_post_that_names_only_its_author(registry, capsys):
    rc, _, err = run(capsys, "send", "--as", "infra", "--dry-run", "--text", "<!subteam^SINFRA|@infra> steward: is exit 1 harmless?")
    assert rc == 2 and "never wakes its author" in err
    assert run(capsys, "send", "--as", "infra", "--dry-run", "--text", "@infra @ci is exit 1 harmless?")[0] == 0


def test_an_announcement_to_every_handle_refuses_leading_names(registry, capsys):
    rc, _, err = run(capsys, "send", "--as", "infra", "--kind", "announce", "--dry-run", "--text", "@lean @ci daily wasters")
    assert rc == 2 and "pass them as --to" in err
    assert run(capsys, "send", "--as", "infra", "--kind", "announce", "--dry-run", "--text", "Daily wasters. @lean, yours")[0] == 0
    assert run(capsys, "send", "--as", "infra", "--kind", "announce", "--to", "@lean", "@ci", "--dry-run", "--text", "daily")[0] == 0
    assert run(capsys, "announce", "--as", "infra", "--dry-run", "--text", "@lean @ci daily wasters")[0] == 2


def test_a_handle_named_past_the_start_is_refused_since_nothing_rings_it(registry, capsys):
    reply = ["reply", "--as", "top", "--thread", "1700000000.000001", "--dry-run", "--text"]
    rc, _, err = run(capsys, *reply, "@proofs stack point is 4. @lean, rebase onto it")
    assert rc == 2 and "names @lean past its start" in err and "Move them to the start" in err
    rc, _, err = run(capsys, *reply, "@proofs two things:\ncompute-accounting: your ask is open\n- @lean: rebase")
    assert rc == 2 and "names @lean @compute-accounting past its start" in err
    for bad in ("@proofs compute-accounting: 98c5 is yours", "@proofs the slot is #1574 (9987, 2000). lean: stack on 8cf4. done"):
        assert run(capsys, *reply, bad)[0] == 2, bad
    assert run(capsys, "send", "--as", "top", "--to", "@ci", "--dry-run", "--text", "can @ci rebuild? not @lean")[0] == 2
    for ok in ("@proofs @lean stack point is 4", "@proofs stack point is 4; lean rebases onto it",
               "@proofs run `research msg read --as @lean`", "@proofs\n> @lean said 4\nagreed", "top: @top is me",
               "@proofs mail a@lean.example", "@proofs @lean stack point is 4. @lean, rebase", "lean: stack point is 4",
               "@proofs note: the slot is 4"):
        assert run(capsys, *reply, ok)[0] == 0, ok
    assert run(capsys, "send", "--as", "top", "--to", "@ci", "--dry-run", "--text", "can @ci rebuild?")[0] == 0
    assert run(capsys, "ask", "--as", "top", "--to", "@ci", "--dry-run", "--text", "@ci, can you rebuild")[0] == 0
    assert run(capsys, "ask", "--as", "top", "--to", "@ci", "--dry-run", "--text", "rebuild for @lean")[0] == 2


def test_an_fyi_from_the_starter_wakes_only_its_to_names(fake, registry, routes_dirs, capsys):
    circuits, lean = inbox(fake, routes_dirs, "circuits"), inbox(fake, routes_dirs, "lean")
    root = fake.add(COORD, text="@circuits @lean the move starts at 7", username="top", bot_id=BOT)
    assert run(capsys, "send", "--as", "top", "--thread", root["ts"], "--kind", "fyi", "--text", "shard 3/5 passed")[0] == 0
    assert bells(fake, circuits) == [] and bells(fake, lean) == []                    # a progress note is for the digest
    note = next(m for m in reversed(fake.messages[COORD]) if m.get("username") == "top" and m is not root)
    assert slack.tags_of(note["text"]) == ["fyi", slack.QUIET]
    assert run(capsys, "send", "--as", "top", "--thread", root["ts"], "--kind", "fyi", "--to", "@lean",
               "--text", "your shard passed")[0] == 0
    assert bells(fake, circuits) == [] and len(bells(fake, lean)) == 1


def test_one_author_rings_one_inbox_from_a_thread_at_most_cap_times_an_hour_then_says_so_once(fake, registry, routes_dirs, capsys, monkeypatch):
    monkeypatch.setattr(slack, "_now", lambda: datetime.fromtimestamp(1700000100, timezone.utc))
    box = inbox(fake, routes_dirs, "circuits")
    root = fake.add(COORD, text="@infra a long one", username="circuits", bot_id=BOT)
    for i in range(slack.CAP_PER_HOUR + 2):
        assert run(capsys, "reply", "--as", "infra", "--thread", root["ts"], "--text", f"point {i}")[0] == 0
    rung = bells(fake, box)
    assert sum(m["text"].startswith("from infra,") for m in rung) == slack.CAP_PER_HOUR and sum("paused" in m["text"] for m in rung) == 1
    notices = [m for m in fake.messages[COORD] if m.get("thread_ts") == root["ts"] and m["username"] == "router"]
    assert len(notices) == 1 and notices[0]["text"].startswith("routing @infra to circuits paused here for the hour")
    assert slack.QUIET in notices[0]["text"]
    for who in ("proofs", "lean", "network-accounting", "comms", "top", "ci", "memory-accounting"):    # a roll call is not a loop
        assert run(capsys, "reply", "--as", who, "--thread", root["ts"], "--text", "subscribed")[0] == 0
    assert sum(m["text"].startswith("from ") and "paused" not in m["text"] for m in bells(fake, box)) == slack.CAP_PER_HOUR + 7


def test_a_long_multi_line_merge_thread_reply_rings_only_the_names_it_starts_with(fake, registry, routes_dirs, capsys, monkeypatch):
    monkeypatch.setattr(slack, "_now", lambda: datetime.fromtimestamp(1700000100, timezone.utc))
    infra, proofs, lean = (inbox(fake, routes_dirs, n) for n in ("infra", "proofs", "lean"))
    root = fake.add(COORD, text=f"{slack.ANNOUNCE} *Announcement* from @top to every handle\nmerge thread: land outcomes go here",
                    username="top", bot_id=BOT)
    fake.add(COORD, text="#1590 rides r20261009-035518-dd21; 6350 is its suites run", username="proofs", bot_id=BOT, thread_ts=root["ts"])
    fake.add(COORD, text="lean: #1601 waits on dd21 too", username="lean", bot_id=BOT, thread_ts=root["ts"])
    fake.add(COORD, text="vy-nebius-1: the OOM killer took python of the run for #1600", username="vy-monitors", bot_id=BOT,
             thread_ts=root["ts"])
    body = ("@infra circuits, on your placement: yes, r20261011-044421-6350 is ours, launched before it reached us.\n"
            "\u2022 Delete its scratch the moment 6350's pytest exits; nothing of ours needs it.\n"
            "\u2022 Its next suites run goes on vy-cpu-2 after dd21, beside #1590, #1600 and #1601, with --basetemp under scratch.\n"
            "\u2022 The tree is main + #1590, pushed as `cursor/sweep-2-tree-95d4`.\n\n"
            "Nothing else changes; the rest is in the run's notes.")
    rc, _, err = run(capsys, "reply", "--as", "circuits", "--thread", root["ts"], "--text", body)
    assert rc == 0 and "route @infra: rang" in err and "vy-monitors" not in err
    assert len(bells(fake, infra)) == 1 and bells(fake, proofs) == [] and bells(fake, lean) == []
    rc, _, err = run(capsys, "reply", "--as", "circuits", "--thread", root["ts"], "--text", "#1600 and #1601 are on the next train")
    assert rc == 0 and len(bells(fake, lean)) == 1 and "vy-monitors" not in err   # naming no one, a PR's namer is rung


def test_a_finished_name_forwards_and_the_server_router_switches_the_cli_off(fake, registry, routes_dirs, capsys):
    box = inbox(fake, routes_dirs, "circuits")
    add_route(routes_dirs[0], "old-worker", "1.0", closed="2026-10-01T18:00Z", forward_to="circuits")
    rc, _, err = run(capsys, "ask", "--as", "infra", "--to", "@old-worker", "--text", "where is the tree?")
    assert rc == 0 and "route @circuits: rang (for @old-worker, finished)" in err and len(bells(fake, box)) == 1
    router = {"schema": slack.ROUTE_SCHEMA, "name": "router", "mode": "server", "since": "2026-10-01T20:14Z"}   # no inbox
    (routes_dirs[1] / "router.json").write_text(json.dumps(router))                    # published, not yet in the clone
    rc, _, err = run(capsys, "ask", "--as", "infra", "--to", "@circuits", "--text", "again")
    assert rc == 0 and "route: the router rings @circuits" in err and len(bells(fake, box)) == 1
    t = fake.messages[COORD][-1]["ts"]
    rc, _, err = run(capsys, "reply", "--as", "circuits", "--thread", t, "--quiet", "--text", "thanks")
    assert rc == 0 and "route: this rings no one" in err
    (routes_dirs[0] / "router.json").write_text(json.dumps(router))
    rc, _, err = run(capsys, "send", "--as", "infra", "--to", "@circuits", "--text", "and again")
    assert rc == 0 and "rang" not in err and len(bells(fake, box)) == 1
    (routes_dirs[1] / "router.json").write_text(json.dumps({**router, "mode": "cli"}))   # switched back; the clone not pulled
    rc, _, err = run(capsys, "send", "--as", "infra", "--to", "@circuits", "--text", "once more")
    assert rc == 0 and "route @circuits: rang" in err and len(bells(fake, box)) == 2


def test_a_long_lived_routes_reads_the_routers_switch_back_to_cli(tmp_path, monkeypatch):
    clock = [1700000000.0]
    monkeypatch.setattr(slack, "_now", lambda: datetime.fromtimestamp(clock[0], timezone.utc))
    router = {"schema": slack.ROUTE_SCHEMA, "name": "router", "mode": "server", "since": "2026-10-01T20:14Z"}
    routes = slack.Routes(root=tmp_path, base="https://example.invalid/routes.d",
                          get=lambda url: json.dumps(router).encode() if url.endswith("/router.json") else None)
    assert routes.server_delivers()
    router["mode"] = "cli"
    clock[0] += slack.ROUTER_RECHECK_S - 1
    assert routes.server_delivers()
    clock[0] += 1
    assert not routes.server_delivers(), "a relay that started in server mode would otherwise ring no one for good"


def test_a_server_router_whose_route_stops_answering_falls_back_to_the_sender_and_alerts_its_owner(fake, registry, routes_dirs, capsys, monkeypatch):
    monkeypatch.setattr(slack, "_now", lambda: datetime.fromtimestamp(1700000100, timezone.utc))
    status = [405]

    class Route(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status[0])
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Route)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    box, top = inbox(fake, routes_dirs, "circuits"), inbox(fake, routes_dirs, "top")
    router = {"schema": slack.ROUTE_SCHEMA, "name": "router", "mode": "server", "since": "2026-10-01T20:14Z",
              "url": f"http://127.0.0.1:{srv.server_address[1]}/api/agent-slack/events", "owner": "top"}
    for d in routes_dirs[:2]:
        (d / "router.json").write_text(json.dumps(router))
    rc, _, err = run(capsys, "send", "--as", "infra", "--to", "@circuits", "--text", "answered")
    assert rc == 0 and "route: the router rings @circuits" in err and bells(fake, box) == [] and bells(fake, top) == []
    status[0] = 404
    rc, _, err = run(capsys, "send", "--as", "infra", "--to", "@circuits", "--text", "not answered")
    assert rc == 0 and "route @circuits: rang" in err and "route @top: alerted, the router's events route doesn't answer (HTTP 404)" in err
    alert = bells(fake, top)
    assert len(bells(fake, box)) == 1 and len(alert) == 1 and alert[0]["text"].startswith("from router (URGENT), thread ")
    assert router["url"] in alert[0]["text"] and "routes.d/router.json" in alert[0]["text"]
    status[0] = 503
    rc, _, err = run(capsys, "send", "--as", "proofs", "--to", "@circuits", "--text", "still down")
    assert rc == 0 and "route @top: the router's events route doesn't answer (HTTP 503), so this sender rang; already alerted" in err
    assert len(bells(fake, box)) == 2 and len(bells(fake, top)) == 1
    rc, _, err = run(capsys, "send", "--as", "top", "--to", "@circuits", "--text", "from its owner")
    assert rc == 0 and "route @top: the router's events route doesn't answer (HTTP 503), so this sender rang" in err
    assert len(bells(fake, box)) == 3 and len(bells(fake, top)) == 1
    srv.shutdown()


def test_names_list_shows_a_handles_published_route_that_the_clone_lacks(fake, registry, routes_dirs, capsys):
    add_route(routes_dirs[1], "circuits", "1.0", holder="bc-11111111")
    rc, out, _ = run(capsys, "names", "list")
    assert rc == 0 and out.startswith("@circuits") and "bc-11111111" in out and "open" in out


def test_http_get_reads_the_private_notes_raw_files_with_the_notes_token(monkeypatch):
    """The notes repository is private: its raw files answer 404 without a token, so the published-route fallback (and
    `names list` on a clone that lacks the route) only works when the request carries RESEARCH_NOTES_TOKEN."""
    seen = []

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    def urlopen(req, timeout):
        seen.append((req.full_url, req.get_header("Authorization")))
        return Reply(b'{"name": "x"}')

    monkeypatch.setattr(slack.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("RESEARCH_NOTES_TOKEN", "t0k")
    assert slack.http_get(f"{slack.ROUTES_RAW}x.json") == b'{"name": "x"}'
    assert slack.http_get("https://example.invalid/routes.d/x.json") == b'{"name": "x"}'
    monkeypatch.delenv("RESEARCH_NOTES_TOKEN")
    assert slack.http_get(f"{slack.ROUTES_RAW}x.json") == b'{"name": "x"}'
    assert seen == [(f"{slack.ROUTES_RAW}x.json", "Bearer t0k"), ("https://example.invalid/routes.d/x.json", None),
                    (f"{slack.ROUTES_RAW}x.json", None)]


def notes_clone(tmp_path) -> Path:
    import subprocess
    bare, clone = tmp_path / "notes.git", tmp_path / "notes"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    subprocess.run(["git", "-C", str(clone), "init", "-q", "-b", "main"], check=True)
    for k, v in (("user.name", "t"), ("user.email", "t@t"), ("commit.gpgsign", "false")):
        subprocess.run(["git", "-C", str(clone), "config", k, v], check=True)
    subprocess.run(["git", "-C", str(clone), "remote", "add", "origin", str(bare)], check=True)
    return clone


def test_inbox_open_posts_the_root_publishes_the_route_and_guards_it(fake, registry, routes_dirs, tmp_path, capsys, monkeypatch):
    clone = notes_clone(tmp_path)
    monkeypatch.setenv("RESEARCH_NOTES_CITE_CHECK", "0")
    rc, _, err = run(capsys, "names", "open", "--as", "vllm-epoch-run", "--holder", "bc-11111111")
    assert rc == 2 and "only leads get names" in err and not (clone / slack.ROUTES_DIR / "vllm-epoch-run.json").exists()
    rc, out, err = run(capsys, "inbox", "open", "--as", "circuits", "--holder", "bc-11111111", "--remit", "the  vLLM epoch run")
    assert rc == 0, err
    r = json.loads((clone / slack.ROUTES_DIR / "circuits.json").read_text())
    root = fake.messages[COORD][-1]
    assert r["remit"] == "the vLLM epoch run" and "\n  the vLLM epoch run" in run(capsys, "names", "list")[1]
    assert r["holder"] == "bc-11111111" and r["inbox_ts"] == root["ts"] and root["text"].startswith("Inbox for @circuits.")
    assert "pushed" in out and f'threadTs="{root["ts"]}"' in out
    assert run(capsys, "inbox", "open", "--as", "circuits", "--holder", "bc-11111111")[1].startswith("@circuits is open already")
    rc, _, err = run(capsys, "inbox", "open", "--as", "circuits", "--holder", "bc-22222222")
    assert rc == 2 and "held by bc-11111111; a successor passes --succeeds" in err
    assert run(capsys, "inbox", "open", "--as", "circuits", "--holder", "bc-22222222", "--succeeds")[0] == 0
    r = json.loads((clone / slack.ROUTES_DIR / "circuits.json").read_text())
    assert (r["holder"], r["inbox_ts"], r["succeeds"], r["remit"]) == ("bc-22222222", root["ts"], "bc-11111111", "the vLLM epoch run")
    assert sum(m["text"].startswith("Inbox for") for m in fake.messages[COORD]) == 1
    assert run(capsys, "inbox", "close", "--as", "circuits", "--forward-to", "@circuits")[0] == 0
    assert json.loads((clone / slack.ROUTES_DIR / "circuits.json").read_text())["forward_to"] == "circuits"


def test_inbox_read_shows_unread_doorbells_urgent_first_and_marks_them_read(fake, registry, routes_dirs, capsys):
    box = inbox(fake, routes_dirs, "vllm-epoch-run")
    run(capsys, "ask", "--as", "circuits", "--to", "@vllm-epoch-run", "--text", "next tree")
    run(capsys, "ask", "--as", "circuits", "--to", "@vllm-epoch-run", "--urgent", "--text", "stop node 2")
    rc, out, _ = run(capsys, "inbox", "read", "--as", "vllm-epoch-run", "--urgent", "--peek")
    lines = out.splitlines()
    assert rc == 0 and lines[0].startswith("SLACK @vllm-epoch-run: 2 new") and "(URGENT)" in lines[1] and "next tree" in lines[2]
    assert run(capsys, "inbox", "read", "--as", "vllm-epoch-run")[0] == 0
    rc, out, _ = run(capsys, "inbox", "read", "--as", "vllm-epoch-run", "--urgent")
    assert rc == 1 and out.strip() == "SLACK @vllm-epoch-run: nothing new"
    assert box


def test_read_as_since_rereads_the_doorbells_after_ts_and_never_moves_the_mark_back(fake, registry, routes_dirs, capsys):
    inbox(fake, routes_dirs, "vllm-epoch-run")
    run(capsys, "ask", "--as", "circuits", "--to", "@vllm-epoch-run", "--text", "first")
    run(capsys, "ask", "--as", "circuits", "--to", "@vllm-epoch-run", "--text", "second")
    run(capsys, "ask", "--as", "circuits", "--to", "@vllm-epoch-run", "--text", "third")
    rc, out, _ = run(capsys, "read", "--as", "vllm-epoch-run", "--json")
    first, second, third = json.loads(out)
    assert rc == 0 and "third" in third["text"]
    rc, out, _ = run(capsys, "read", "--as", "vllm-epoch-run", "--since", first["ts"], "--json")
    assert rc == 0 and [m["ts"] for m in json.loads(out)] == [second["ts"], third["ts"]]
    assert json.loads(run(capsys, "read", "--as", "vllm-epoch-run", "--json")[1]) == []
    assert slack.seen_path("vllm-epoch-run").read_text().strip() == third["ts"]
    slack.mark_read("vllm-epoch-run", [second])
    assert slack.seen_path("vllm-epoch-run").read_text().strip() == third["ts"]
    rc, _, err = run(capsys, "read", "--thread", first["ts"], "--as", "vllm-epoch-run")
    assert rc == 2 and "--thread reads one thread; it doesn't take --as" in err
    rc, _, err = run(capsys, "read")
    assert rc == 2 and "one of --thread TS, --since TS or --as NAME" in err


def test_traffic_counts_doorbells_and_reports_dropped_names_and_unanswered_asks(fake, registry, routes_dirs, capsys):
    box = inbox(fake, routes_dirs, "circuits")
    add_route(routes_dirs[0], "old-circuits", box, closed="2026-10-01T18:00Z", forward_to="circuits")   # renamed: one inbox
    run(capsys, "ask", "--as", "infra", "--to", "@circuits", "--text", "one")
    fake.add(COORD, text="circuits 06:00Z: 3 open", username="circuits", bot_id=BOT, thread_ts=box)   # its own note, no doorbell
    old = fake.add(COORD, text="@proofs ancient ask", username="infra", bot_id=BOT, ts="1600000000.000001")
    fyi = fake.add(COORD, text="@proofs ancient fyi (quiet)", username="infra", bot_id=BOT, ts="1600000000.000002")
    bare = fake.add(COORD, text="@proofs an older client's fyi\n\n_fyi_", username="infra", bot_id=BOT, ts="1600000000.000003")
    fake.add(COORD, text="@no-such-lane hello", username="infra", bot_id=BOT)
    retired = fake.add(COORD, text="<!subteam^SRETIRED> how does Daniel like reports?", username="top", bot_id=BOT,
                       ts="1600000000.000004")                     # a retired name's group, tagged by its id alone
    rc, out, _ = run(capsys, "traffic", "--hours", "1e6", "--now", "1700100000", "--json")
    rep = json.loads(out)
    assert rc == 0 and [x for x in rep["inboxes"] if x["doorbells"]] == [{"name": "circuits", "doorbells": 1, "peak_per_hour": 1, "paused": 0}]
    assert {d["name"] for d in rep["dropped"]} == {"no-such-lane", "subteam^SRETIRED"}
    assert {"ts": retired["ts"], "by": "top", "to": []} in rep["unanswered"]
    rc, out, _ = run(capsys, "traffic", "--hours", "1e6", "--now", "1700100000")
    assert f"UNANSWERED {retired['ts']} by @top to no one it can ring" in out
    stale = [s["ts"] for s in rep["unanswered"]]
    assert old["ts"] in stale and fyi["ts"] not in stale and bare["ts"] not in stale


def test_verify_author_accepts_a_routed_workers_name_and_the_router(fake, registry, routes_dirs, capsys):
    inbox(fake, routes_dirs, "vllm-epoch-run")
    for who in ("vllm-epoch-run", "router"):
        m = fake.add(COORD, text="x", username=who, bot_id=BOT)
        assert run(capsys, "verify-author", "--event", json.dumps({"channel": COORD, "ts": m["ts"]}))[0] == 0
    m = fake.add(COORD, text="x", username="impostor", bot_id=BOT)
    assert run(capsys, "verify-author", "--event", json.dumps({"channel": COORD, "ts": m["ts"]}))[0] == 1


def test_slack_py_runs_as_a_single_file_from_any_checkout(tmp_path, registry):
    import subprocess
    import sys
    single = tmp_path / "vslack.py"
    shutil.copy(slack.__file__, single)
    out = subprocess.run([sys.executable, str(single), "handles", "--registry", str(registry)], capture_output=True, text=True,
                         cwd=tmp_path, env={**os.environ, "PYTHONPATH": ""})
    assert out.returncode == 0, out.stderr
    assert "@circuits" in out.stdout


@pytest.mark.parametrize("v", json.loads(VECTORS.read_text())["subscriptions"]["cases"], ids=lambda v: v["name"])
def test_the_delivery_rule_matches_every_subscription_vector(v):
    doc = json.loads(VECTORS.read_text())
    got = slack.deliveries(v["msg"], v["root"], doc["registry"], doc["subscriptions"]["book"], set(doc["inboxes"]))
    assert [[d["name"], d["deliver"]] for d in got] == v["deliveries"]
    assert [d["name"] for d in got if d["why"] == "addressed"] == slack.route_targets(v["msg"], v["root"], doc["registry"], set(doc["inboxes"]), v.get("history", []), v.get("branches"))


@pytest.mark.parametrize("v", json.loads(VECTORS.read_text())["doorbells"]["cases"], ids=lambda v: v["name"])
def test_a_doorbell_matches_every_doorbell_vector(v):
    bell = slack.doorbell(v["author"], v["thread"], v["ts"], v["text"], v["urgent"])
    assert bell == v["doorbell"] and slack._rang(bell, v["ts"], "-") and not slack._rang(bell, "1.000000", "-")
    assert slack._mentions_thread(bell, v["thread"]) and slack._from(bell, v["author"]) and not bell.startswith("@")


# ---- the messages API: send, close, list, subscribe, read --as ----------------------------------------------------------------------

def posted(fake, out: str) -> dict:
    """The message a post printed `<ts> <permalink>` for (the doorbells it rang come after it)."""
    return next(m for m in fake.messages[COORD] if m["ts"] == out.split()[0])


def test_send_starts_a_thread_or_replies_and_its_footer_carries_the_kind(fake, registry, routes_dirs, capsys):
    box = inbox(fake, routes_dirs, "circuits")
    rc, out, err = run(capsys, "send", "--as", "infra", "--to", "@circuits", "--text", "can node 2 take a job?")
    root = posted(fake, out)
    assert rc == 0 and "route @circuits: rang" in err and root["text"].startswith("@circuits can node 2 take a job?")
    assert slack.tags_of(root["text"]) == ["ask"] and slack.kind_of(root) == "ask" and len(bells(fake, box)) == 1
    rc, out, err = run(capsys, "send", "--as", "circuits", "--thread", root["ts"], "--urgent", "--text", "yes, from 3 PM")
    reply = posted(fake, out)
    assert rc == 0 and reply["thread_ts"] == root["ts"] and slack.tags_of(reply["text"]) == ["urgent"] and slack.kind_of(reply) == "message"
    rc, out, _ = run(capsys, "send", "--as", "proofs", "--kind", "fyi", "--quiet", "--text", "the tables are re-rendered")
    assert rc == 0 and slack.tags_of(posted(fake, out)["text"]) == ["fyi", slack.QUIET] and len(bells(fake, box)) == 1
    rc, out, err = run(capsys, "send", "--as", "proofs", "--kind", "announce", "--to", "@circuits", "--text", "a new verifier")
    ann = posted(fake, out)
    assert rc == 0 and slack.is_announcement(ann) and slack.kind_of(ann) == "announce" and len(bells(fake, box)) == 2
    for argv, says in ((["--to", "@infra"], "someone else"), (["--kind", "Two words"], "one lower-case word"),
                       (["--kind", "urgent"], "one lower-case word"), (["--verify", "x"], "go with --kind announce"),
                       (["--kind", "announce", "--thread", root["ts"]], "starts its own thread")):
        rc, _, err = run(capsys, "send", "--as", "infra", *argv, "--text", "x")
        assert rc == 2 and says in err, argv


def test_close_posts_a_status_line_and_reacts_and_list_shows_the_state(fake, registry, routes_dirs, capsys, monkeypatch):
    monkeypatch.setattr(slack, "_now", lambda: datetime.fromtimestamp(1700000500, timezone.utc))
    inbox(fake, routes_dirs, "circuits")
    first = posted(fake, run(capsys, "send", "--as", "infra", "--to", "@circuits", "--text", "can node 2 take a job?")[1])["ts"]
    second = posted(fake, run(capsys, "send", "--as", "proofs", "--to", "@circuits", "--text", "which tree?")[1])["ts"]
    run(capsys, "send", "--as", "proofs", "--to", "@infra", "--text", "not for circuits")
    assert run(capsys, "close", "--as", "circuits", "--thread", first, "--link", "https://x/pr/1")[0] == 0
    assert run(capsys, "close", "--as", "circuits", "--thread", second, "--declined", "not mine")[0] == 0
    assert (COORD, first, "white_check_mark") in fake.reactions and (COORD, second, "x") in fake.reactions
    lines = [m["text"] for m in fake.messages[COORD] if m.get("thread_ts") in (first, second)]
    assert lines[0].startswith(":white_check_mark: done") and lines[1].startswith(":x: declined \u2014 not mine")
    for m in fake.messages[COORD]:                                                     # what Slack's history adds to a root
        m["reply_count"] = sum(x.get("thread_ts") == m["ts"] and x["ts"] != m["ts"] for x in fake.messages[COORD])
        m["reactions"] = [{"name": n} for c, t, n in fake.reactions if t == m["ts"]]
    rc, out, _ = run(capsys, "list", "--as", "circuits", "--json")
    rows = json.loads(out)
    assert rc == 0 and [(r["ts"], r["kind"], r["from"], r["to"], r["state"]) for r in rows] == [
        (second, "ask", "proofs", ["circuits"], "declined"), (first, "ask", "infra", ["circuits"], "done")]
    assert len(json.loads(run(capsys, "list", "--json")[1])) == 3 and json.loads(run(capsys, "list", "--open", "--json")[1])[0]["to"] == ["infra"]


def test_a_listener_subscribes_and_reads_its_digest_and_a_wake_subscription_rings(fake, registry, routes_dirs, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(slack, "_now", lambda: datetime.fromtimestamp(1700000500, timezone.utc))
    monkeypatch.setenv("RESEARCH_NOTES_CITE_CHECK", "0")
    clone = notes_clone(tmp_path)
    assert run(capsys, "names", "open", "--as", "comms", "--holder", "bc-comms")[0] == 0
    assert run(capsys, "subscribe", "--as", "comms", "--filter", "all")[0] == 0
    assert json.loads((clone / slack.ROUTES_DIR / "comms.json").read_text())["subscriptions"] == [{"filter": "all", "deliver": "digest"}]
    rc, _, err = run(capsys, "subscribe", "--as", "comms", "--filter", "knd:ask")
    assert rc == 2 and "terms are" in err
    box = inbox(fake, routes_dirs, "circuits")
    rc, out, err = run(capsys, "send", "--as", "infra", "--to", "@circuits", "--text", "one")
    assert rc == 0 and "route @circuits: rang" in err and "route @comms: in its digest" in err
    root = posted(fake, out)
    root["reply_count"], root["latest_reply"] = 1, "1700000400.000000"
    run(capsys, "send", "--as", "circuits", "--thread", root["ts"], "--quiet", "--text", "on it")
    rc, out, _ = run(capsys, "read", "--as", "comms")
    assert rc == 0 and "SLACK @comms: nothing new" in out and "DIGEST @comms: 2 post(s)" in out
    assert "@infra ask: @circuits one" in out and "@circuits message: on it" in out
    assert "DIGEST @comms: nothing new" in run(capsys, "read", "--as", "comms")[1]
    rc, out, _ = run(capsys, "subscribe", "--as", "comms", "--filter", "kind:announce", "--deliver", "wake")
    assert rc == 0 and "@comms wake   kind:announce" in out
    comms_box = json.loads((clone / slack.ROUTES_DIR / "comms.json").read_text())["inbox_ts"]
    run(capsys, "send", "--as", "proofs", "--kind", "announce", "--to", "@circuits", "--text", "a new verifier")
    assert len(bells(fake, comms_box)) == 0 and len(bells(fake, box)) == 2              # `all` matched first: a digest, no doorbell
    assert run(capsys, "subscribe", "--as", "comms", "--remove", "all")[1].splitlines()[-1] == "@comms wake   kind:announce"
    run(capsys, "send", "--as", "proofs", "--kind", "announce", "--to", "@circuits", "--text", "another")
    assert len(bells(fake, comms_box)) == 1


def test_research_msg_is_the_front_door(registry, capsys):
    from research import cli
    assert cli.main(["msg", "handles"]) == 0 and "@circuits" in capsys.readouterr().out
    assert cli.main(["msg", "send", "--as", "infra", "--text", "x", "--to", "@infra"]) == 2
    assert capsys.readouterr().err.startswith("research msg: ")



def test_a_single_file_in_a_checkout_older_than_the_registry_reads_it_from_origin(tmp_path, registry):
    import subprocess
    import sys
    single, repo = tmp_path / "vmsg.py", tmp_path / "old-checkout"
    shutil.copy(slack.__file__, single)
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    repo.mkdir()
    git("init", "-q", "-b", "main")
    (repo / slack.REGISTRY_REL).parent.mkdir(parents=True)
    shutil.copy(registry, repo / slack.REGISTRY_REL)
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "registry")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    git("rm", "-q", "-r", ".agents")                                                # this tree predates the registry
    env = {k: v for k, v in os.environ.items() if k != "RESEARCH_SLACK_REGISTRY"} | {"PYTHONPATH": "", "TMPDIR": str(tmp_path)}
    out = subprocess.run([sys.executable, str(single), "handles"], capture_output=True, text=True, cwd=repo, env=env)
    assert out.returncode == 0, out.stderr
    assert "@circuits" in out.stdout
    git("update-ref", "-d", "refs/remotes/origin/main")
    out = subprocess.run([sys.executable, str(single), "handles"], capture_output=True, text=True, cwd=repo, env=env)
    assert out.returncode == 2 and "after `git fetch origin main`" in out.stderr


def test_watch_fires_once_per_new_doorbell_and_marks_nothing_read(fake, registry, routes_dirs, capsys, monkeypatch):
    inbox(fake, routes_dirs, "console")
    sent = []
    monkeypatch.setattr(slack, "_sleep", lambda s: sent or sent.append(run(capsys, "send", "--as", "infra", "--to", "@console", "--text", "panel?")))
    rc, out, _ = run(capsys, "watch", "--as", "console", "--polls", "3")
    assert rc == 0 and len(sent) == 1 and out.startswith("SLACK @console: 1 new") and "panel?" in out
    assert run(capsys, "watch", "--as", "console", "--polls", "2")[0] == 1                # the same doorbell fires once
    rc, out, _ = run(capsys, "read", "--as", "console")
    assert rc == 0 and "panel?" in out                                                   # still unread for the agent
