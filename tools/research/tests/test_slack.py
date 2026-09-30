"""`research slack` against a fake Slack Web API: a local HTTP server that keeps channels, threads, reactions and user groups the
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
TOKEN = "dummy-verity-bot"
BOT = "B0VERITY"
COORD, ALERTS = "CCOORD", "CALERTS"
HANDLES = ["infra", "proofs", "circuits", "compute-accounting", "memory-accounting", "network-accounting", "console",
           "old-accounting", "old-circuits-and-proofs"]


class FakeSlack:
    def __init__(self):
        self.messages: dict[str, list[dict]] = {}
        self.groups: list[dict] = []
        self.reactions: list[tuple[str, str, str]] = []
        self.calls: list[tuple[str, dict]] = []
        self._ts = itertools.count(1)
        self.page = 200

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
            return {"ok": True, "permalink": f"https://compute.slack.com/archives/{ch}/p{p['message_ts'].replace('.', '')}"}
        if method in ("conversations.history", "conversations.replies"):
            msgs = self.messages.get(ch, [])
            if method == "conversations.replies":
                msgs = [m for m in msgs if m["ts"] == p["ts"] or m.get("thread_ts") == p["ts"]]
            else:
                msgs = [m for m in msgs if not m.get("thread_ts") or m["thread_ts"] == m["ts"]][::-1]     # newest first, like Slack
            if "oldest" in p:
                msgs = [m for m in msgs if float(m["ts"]) >= float(p["oldest"])]
            if "latest" in p:
                msgs = [m for m in msgs if float(m["ts"]) <= float(p["latest"])]
            start = int(p.get("cursor") or 0)
            page = msgs[start:start + min(int(p.get("limit", 100)), self.page)]
            nxt = start + len(page)
            return {"ok": True, "messages": page, "response_metadata": {"next_cursor": str(nxt) if nxt < len(msgs) else ""}}
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
        if method in ("usergroups.update", "usergroups.enable"):
            g = next(g for g in self.groups if g["id"] == p["usergroup"])
            g.update({k: p[k] for k in ("name", "handle", "description") if k in p}, **({"date_delete": 0} if method.endswith("enable") else {}))
            return {"ok": True, "usergroup": g}
        return {"ok": False, "error": "unknown_method"}


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
    assert slack.subscribers(reg, "agent-coordination") == ["old-accounting", "old-circuits-and-proofs"]
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


def test_subscriptions_fail_visibly_without_channel_ids(registry, capsys):
    reg = json.loads(registry.read_text())
    reg["channels"]["agent-coordination"]["id"] = None
    registry.write_text(json.dumps(reg))
    rc, out, _ = run(capsys, "subscriptions", "--as", "old-accounting")
    assert rc == 2 and "id not recorded" in out


def test_ask_tags_one_owner_with_its_user_group_and_links_the_agent(fake, registry, capsys):
    rc, out, err = run(capsys, "ask", "--as", "compute-accounting", "--to", "@infra", "--text", "4 GPUs on node 2 for 6 h, by 20:00Z. Spec: notes/x.md")
    assert rc == 0
    ts, link = out.split()
    m = fake.messages[COORD][0]
    assert m["ts"] == ts and m["username"] == "compute-accounting" and link.endswith(ts.replace(".", ""))
    assert m["text"].startswith("<!subteam^SINFRA|@infra> 4 GPUs")
    assert "<https://cursor.com/agents/bc-test|@compute-accounting's agent>" in m["text"]
    assert TOKEN not in out + err
    assert run(capsys, "ask", "--as", "compute-accounting", "--to", "@compute-accounting", "--text", "x")[0] == 2
    assert run(capsys, "ask", "--as", "pous-worker-3", "--to", "@infra", "--text", "x")[0] == 2      # workers aren't on Slack


def test_ask_without_a_group_id_uses_plain_text_and_the_body_can_come_from_a_file(fake, registry, capsys, tmp_path):
    body = tmp_path / "ask.md"
    body.write_text("a frontend for the pous panel: spec in PR 12")
    assert run(capsys, "ask", "--as", "compute-accounting", "--to", "console", "--file", str(body))[0] == 0
    assert fake.messages[COORD][0]["text"].startswith("@console a frontend")


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
        self.state, self.requests, self.identities, self.fail = state, [], [], None

    def oidc(self, aud: str) -> str:
        self.identities.append(aud)
        return OIDC

    def __call__(self, url: str, data: bytes, headers: dict) -> dict:
        req = json.loads(data)
        self.requests.append((url, req, headers))
        if self.fail:
            raise self.fail
        if headers.get("Authorization") != f"Bearer {OIDC}":
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
    assert rc == 2 and says in err and err.startswith("research slack: reactions.add: ")
    assert OIDC not in out + err


def test_a_slack_error_through_the_broker_is_slacks_own(broker, registry, capsys):
    t = broker.state.add(COORD, text="x", user="UDANIEL")["ts"]
    assert run(capsys, "pickup", "--thread", t)[0] == 0
    rc, _, err = run(capsys, "pickup", "--thread", t)
    assert rc == 2 and "reactions.add: already_reacted" in err


def test_the_client_refuses_what_the_broker_would(broker, registry, capsys):
    with pytest.raises(slack.SlackError, match="not in BROKER_METHODS"):
        slack.Slack(None).call("chat.delete", channel=COORD, ts="1.0")
    rc, _, err = run(capsys, "reply", "--as", "infra", "--thread", "1.0", "--text", "x" * slack.BROKER_MAX_BODY)
    assert rc == 2 and "more than the broker takes" in err and not broker.requests


def test_broker_methods_are_the_methods_this_module_calls_but_the_user_group_writes():
    src = Path(slack.__file__).read_text()
    called = set(re.findall(r"\.(?:call|paged)\(\s*\"([a-z]+\.[A-Za-z]+)\"", src))
    assert set(slack.DIRECT_ONLY) == {"usergroups.create", "usergroups.enable", "usergroups.update"}
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
    state = {"answer": (200, {"token": OIDC}), "seen": []}

    class Handler(http.server.BaseHTTPRequestHandler):
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
    assert match(capsys, "console", "ping @console, please")[0] == 0
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
    assert match(capsys, "console", {**reply, "channel": COORD})[0] == 1
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
    assert run(capsys, "decline", "--as", "circuits", "--thread", t, "--why", "a job runs until 23:00Z")[0] == 0
    assert run(capsys, "done", "--as", "console", "--thread", t)[0] == 0                   # not addressed: not counted
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


def test_a_post_carries_slack_date_tokens_and_read_shows_both_zones(fake, registry, capsys):
    text = "freeze 2026-11-01T09:30Z, see note:20260930T2020Z-handoff-infra and `--since 2026-11-01T09:30Z`"
    assert run(capsys, "ask", "--as", "infra", "--to", "@console", "--text", text)[0] == 0
    posted = fake.messages[COORD][0]["text"]
    assert "freeze <!date^1793525400^{date_short_pretty} {time}|1:30 AM PST (09:30Z)>," in posted
    assert "note:20260930T2020Z-handoff-infra" in posted and "`--since 2026-11-01T09:30Z`" in posted
    fake.messages[COORD][0]["ts"] = "1793525400.000100"
    rc, out, _ = run(capsys, "read", "--thread", "1793525400.000100")
    assert rc == 0 and out.startswith("1793525400.000100 1:30 AM PST (09:30Z) @infra: @console freeze 1:30 AM PST (09:30Z), see note:")


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

    def handle(self, method: str, path: str, body: dict | None, auth: str) -> tuple[int, dict]:
        self.requests.append((method, path, body))
        if self.answer:
            return self.answer
        if auth != f"Bearer {OIDC}":
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
    assert route.identity["seen"] == [("/v1/tokens/oidc", {"aud": slack.ASK_DANIEL})]
    assert "the card's fields" in err and f"threadTs=1790800000.000100" in err and OIDC not in out + err


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


def test_resolve_records_a_thread_answer_and_says_clearly_when_the_route_is_missing(route, registry, capsys):
    rc, _, err = run(capsys, "ask-daniel", "resolve", "42", "--choice", "no", "--note", "Daniel in the thread: needs changes")
    assert rc == 2 and "HTTP 404: the resolve route isn't deployed yet" in err and "record it as a ruling" in err
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
