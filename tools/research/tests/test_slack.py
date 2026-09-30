"""`research slack` against a fake Slack Web API: a local HTTP server that keeps channels, threads, reactions and user groups the
way the real one returns them. No real token or workspace is involved (conftest clears SLACK_*)."""
from __future__ import annotations

import http.server
import itertools
import json
import threading
import urllib.parse
from pathlib import Path

import pytest

from research import slack

REAL_REGISTRY = Path(__file__).resolve().parents[3] / slack.REGISTRY_REL
TOKEN = "xoxb-test-not-a-real-token"
INFRA_TOKEN = "xoxb-test-infra-not-a-real-token"
BOT, INFRA_BOT = "B0SHARED", "B0INFRA"


class FakeSlack:
    def __init__(self):
        self.messages: dict[str, list[dict]] = {}
        self.groups: list[dict] = []
        self.reactions: list[tuple[str, str, str]] = []
        self.calls: list[tuple[str, dict]] = []
        self.auth: list[str] = []
        self._ts = itertools.count(1)
        self.page = 200

    def add(self, channel: str, **msg) -> dict:
        msg.setdefault("ts", f"1700000000.{next(self._ts):06d}")
        msg.setdefault("type", "message")
        self.messages.setdefault(channel, []).append(msg)
        return msg

    def handle(self, method: str, p: dict, auth: str) -> dict:
        self.calls.append((method, p))
        self.auth.append(auth)
        if auth not in (f"Bearer {TOKEN}", f"Bearer {INFRA_TOKEN}"):
            return {"ok": False, "error": "invalid_auth"}
        ch = p.get("channel")
        if method == "chat.postMessage":
            bot = INFRA_BOT if auth == f"Bearer {INFRA_TOKEN}" else BOT
            m = self.add(ch, text=p["text"], username=p.get("username"), bot_id=bot, **({"thread_ts": p["thread_ts"]} if p.get("thread_ts") else {}))
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
            self.reactions.append((ch, p["timestamp"], p["name"]))
            return {"ok": True}
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
    monkeypatch.setenv("SLACK_INFRA_BOT_TOKEN", INFRA_TOKEN)
    monkeypatch.setenv("RESEARCH_AGENT_URL", "https://cursor.com/agents/bc-test")
    yield state
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def registry(tmp_path, monkeypatch):
    reg = json.loads(REAL_REGISTRY.read_text())
    ids = {"agent-coordination": "CCOORD", "infra": "CINFRA", "console": "CCONSOLE", "agent-alerts": "CALERTS"}
    for k, cid in ids.items():
        reg["channels"][k]["id"] = cid
    reg["channels"]["agent-alerts"]["authors"]["bots"][0]["bot_id"] = "BALERTS"
    reg["humans"][0]["slack_user"] = "UDANIEL"
    for h in reg["handles"]:
        if h["handle"] == "infra-coordinator":
            h["usergroup"], h["bot_id"] = "SINFRA", INFRA_BOT
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(reg))
    monkeypatch.setenv("RESEARCH_SLACK_REGISTRY", str(path))
    return path


def run(capsys, *argv) -> tuple[int, str, str]:
    rc = slack.main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_the_repository_registry_loads_and_seeds_every_handle_and_channel():
    reg = slack.load_registry(REAL_REGISTRY)
    assert {h["handle"] for h in reg["handles"]} == {"top-level", "infra-coordinator", "pouw-coordinator", "circuit-coordinator",
                                                     "proof-coordinator", "console-agent"}
    assert set(reg["channels"]) == {"agent-coordination", "infra", "console", "agent-alerts"}
    assert all(c["subscribe"]["top_level_only"] for c in reg["channels"].values())
    assert reg["channels"]["infra"]["owner"] == "infra-coordinator" and reg["channels"]["console"]["owner"] == "console-agent"


def test_handles_lists_the_registry(registry, capsys):
    rc, out, _ = run(capsys, "handles")
    assert rc == 0 and "@infra-coordinator" in out and "SINFRA" in out and "#infra" in out and "the owner in full" in out


def test_post_uses_the_handle_as_username_links_the_agent_and_prints_ts_and_permalink(fake, registry, capsys):
    rc, out, err = run(capsys, "post", "--as", "pouw-coordinator", "--text", "@infra-coordinator: a pod for 2 h? spec: notes/x.md")
    assert rc == 0
    ts, link = out.split()
    m = fake.messages["CCOORD"][0]
    assert m["ts"] == ts and m["username"] == "pouw-coordinator" and link.endswith(ts.replace(".", ""))
    assert "<https://cursor.com/agents/bc-test|@pouw-coordinator's agent>" in m["text"]
    assert TOKEN not in out + err


def test_post_reads_the_body_from_a_file_and_replies_in_a_thread(fake, registry, capsys, tmp_path):
    root = fake.add("CCOORD", text="<!subteam^SINFRA|@infra-coordinator> ask", user="UDANIEL")
    body = tmp_path / "msg.md"
    body.write_text("see PR 12")
    assert run(capsys, "post", "--as", "infra-coordinator", "--thread", root["ts"], "--file", str(body))[0] == 0
    assert fake.messages["CCOORD"][-1]["thread_ts"] == root["ts"] and fake.messages["CCOORD"][-1]["text"].startswith("see PR 12")


def test_only_the_owner_posts_top_level_in_a_service_channel_and_with_its_own_token(fake, registry, capsys):
    rc, _, err = run(capsys, "post", "--as", "pouw-coordinator", "--channel", "infra", "--text", "hello")
    assert rc == 2 and "only its owner @infra-coordinator" in err and not fake.calls
    root = fake.add("CINFRA", text="contract", username="infra-coordinator", bot_id=INFRA_BOT)
    assert run(capsys, "post", "--as", "pouw-coordinator", "--channel", "infra", "--thread", root["ts"], "--text", "q")[0] == 0
    assert fake.auth[-2] == f"Bearer {TOKEN}"
    assert run(capsys, "post", "--as", "infra-coordinator", "--channel", "infra", "--text", "notice")[0] == 0
    assert fake.messages["CINFRA"][-1]["bot_id"] == INFRA_BOT


def test_no_token_is_an_error_and_dry_run_sends_nothing(fake, registry, capsys, monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN")
    rc, _, err = run(capsys, "post", "--as", "top-level", "--text", "x")
    assert rc == 2 and "SLACK_BOT_TOKEN is not set" in err
    rc, out, err = run(capsys, "post", "--as", "top-level", "--text", "x", "--dry-run")
    assert rc == 0 and "dry-run: chat.postMessage" in err and not fake.calls


@pytest.mark.parametrize("argv", [
    ["read", "--since", "1.0"], ["match", "--handle", "top-level", "--event", '{"text":"x","ts":"2.0","thread_ts":"1.0"}'],
    ["pickup", "--thread", "1.0"], ["done", "--as", "top-level", "--thread", "1.0", "--link", "https://x"],
    ["announce", "--channel", "console", "--as", "console-agent", "--action", "reload the panel", "--verify", "true"],
    ["ack", "--channel", "infra", "--thread", "1.0", "--as", "top-level", "--status", "done"],
    ["roster", "--channel", "infra", "--thread", "1.0"], ["verify-author", "--channel", "infra", "--event", '{"ts":"1.0"}'],
    ["groups", "sync", "--apply"],
])
def test_every_subcommand_has_a_dry_run_that_sends_nothing(fake, registry, capsys, argv):
    slack.main([*argv, "--dry-run"])
    assert not fake.calls
    assert TOKEN not in capsys.readouterr().err


def test_read_since_and_thread_oldest_first_across_pages(fake, registry, capsys):
    fake.page = 2
    a = fake.add("CCOORD", text="first", user="U1")
    fake.add("CCOORD", text="second", user="U2")
    fake.add("CCOORD", text="reply", user="U3", thread_ts=a["ts"])
    fake.add("CCOORD", text="third", username="console-agent", bot_id=BOT)
    rc, out, _ = run(capsys, "read", "--since", a["ts"])
    assert rc == 0 and [line.split(": ", 1)[1] for line in out.splitlines()] == ["second", "third"]
    assert "@console-agent" in out
    rc, out, _ = run(capsys, "read", "--thread", a["ts"], "--json")
    assert [m["text"] for m in json.loads(out)] == ["first", "reply"]


def _match(capsys, handle, ev):
    return run(capsys, "match", "--handle", handle, "--event", ev if isinstance(ev, str) else json.dumps(ev), "--why")


def test_match_on_the_user_group_id_its_label_and_plain_text(fake, registry, capsys):
    assert _match(capsys, "infra-coordinator", {"text": "<!subteam^SINFRA> can you", "ts": "5.0"})[0] == 0
    assert _match(capsys, "infra-coordinator", {"text": "<!subteam^SOTHER|@infra-coordinator> hi", "ts": "5.0"})[0] == 0
    assert _match(capsys, "infra-coordinator", "ping @infra-coordinator, please")[0] == 0
    assert _match(capsys, "infra-coordinator", {"text": "<!subteam^SOTHER|@pouw-coordinator> hi", "ts": "5.0"})[0] == 1
    assert _match(capsys, "infra-coordinator", "email infra@infra-coordinator.dev or @infra-coordinator-2")[0] == 1
    assert _match(capsys, "pouw-coordinator", {"event": {"type": "message", "text": "@pouw-coordinator hi", "ts": "5.0"}})[0] == 0
    assert not fake.calls


def test_match_ignores_the_handles_own_posts(fake, registry, capsys):
    ev = {"text": "@infra-coordinator note to self", "ts": "5.0", "username": "infra-coordinator", "bot_id": BOT}
    assert _match(capsys, "infra-coordinator", ev)[0] == 1


def test_match_a_reply_in_a_thread_the_handle_posted_in_or_was_tagged_at_the_root(fake, registry, capsys):
    root = fake.add("CCOORD", text="question for <!subteam^SINFRA>", user="UDANIEL")
    reply = fake.add("CCOORD", text="any news?", user="UDANIEL", thread_ts=root["ts"])
    assert _match(capsys, "infra-coordinator", {**reply, "channel": "CCOORD"})[0] == 0
    assert _match(capsys, "pouw-coordinator", {**reply, "channel": "CCOORD"})[0] == 1
    fake.add("CCOORD", text="on it", username="pouw-coordinator", bot_id=BOT, thread_ts=root["ts"])
    later = fake.add("CCOORD", text="thanks", user="UDANIEL", thread_ts=root["ts"])
    rc, _, err = _match(capsys, "pouw-coordinator", {**later, "channel": "CCOORD"})
    assert rc == 0 and "posted in it" in err
    assert _match(capsys, "console-agent", {**later, "channel": "CCOORD"})[0] == 1


def test_a_worker_posts_under_its_lane_name_and_matches_replies_in_its_own_thread(fake, registry, capsys):
    rc, out, _ = run(capsys, "post", "--as", "pous-worker-3", "--text", "<!subteam^SINFRA> a pod? by 20:00Z: notes/x.md")
    ts = out.split()[0]
    assert rc == 0 and fake.messages["CCOORD"][0]["username"] == "pous-worker-3"
    reply = fake.add("CCOORD", text="yes, vy-4", username="infra-coordinator", bot_id=BOT, thread_ts=ts)
    assert _match(capsys, "pous-worker-3", {**reply, "channel": "CCOORD"})[0] == 0
    own = fake.add("CCOORD", text="thanks", username="pous-worker-3", bot_id=BOT, thread_ts=ts)
    assert _match(capsys, "pous-worker-3", {**own, "channel": "CCOORD"})[0] == 1
    assert run(capsys, "post", "--as", "Not A Lane", "--text", "x")[0] == 2


def test_match_reads_the_event_from_a_file_or_stdin(fake, registry, capsys, tmp_path, monkeypatch):
    p = tmp_path / "ev.json"
    p.write_text(json.dumps({"text": "@top-level decide", "ts": "1.0"}))
    assert _match(capsys, "top-level", f"@{p}")[0] == 0
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO("@top-level again"))
    assert _match(capsys, "top-level", "-")[0] == 0


def test_pickup_and_done_react_and_link_the_result(fake, registry, capsys):
    root = fake.add("CCOORD", text="<!subteam^SINFRA> need a pod", user="UDANIEL")
    assert run(capsys, "pickup", "--thread", root["ts"])[0] == 0
    assert run(capsys, "done", "--as", "infra-coordinator", "--thread", root["ts"], "--link", "https://github.com/x/pull/1")[0] == 0
    assert fake.reactions == [("CCOORD", root["ts"], "eyes"), ("CCOORD", root["ts"], "white_check_mark")]
    assert "https://github.com/x/pull/1" in fake.messages["CCOORD"][-1]["text"]


def test_announce_is_the_channel_owners_and_carries_the_safe_point_verify_and_pin(fake, registry, capsys):
    sha = "a" * 64
    rc, _, err = run(capsys, "announce", "--channel", "infra", "--as", "pouw-coordinator", "--action", "x", "--verify", "true")
    assert rc == 2 and "only its owner" in err
    rc, _, err = run(capsys, "announce", "--channel", "agent-coordination", "--as", "top-level", "--action", "x", "--verify", "true")
    assert rc == 2 and "service channel" in err
    rc, _, err = run(capsys, "announce", "--channel", "infra", "--as", "infra-coordinator", "--action", "curl the new helper and install it",
                     "--verify", "helper --version")
    assert rc == 2 and "pin it" in err
    rc, _, err = run(capsys, "announce", "--channel", "infra", "--as", "infra-coordinator", "--action", "a", "--verify", "b", "--ships", "https://x/y.py")
    assert rc == 2 and "--ships" in err
    assert not fake.calls
    rc, out, _ = run(capsys, "announce", "--channel", "infra", "--as", "infra-coordinator", "--action", "install the broker helper",
                     "--verify", "git ls-remote origin HEAD", "--ships", f"https://x/verity-github.py#sha256={sha}")
    assert rc == 0
    m = fake.messages["CINFRA"][0]
    assert m["bot_id"] == INFRA_BOT and m["username"] == "infra-coordinator"
    assert slack.SAFE_POINT in m["text"] and "`git ls-remote origin HEAD`" in m["text"] and sha in m["text"]
    assert "research slack ack --channel infra" in m["text"]


def test_ack_and_roster_against_the_registry(fake, registry, capsys):
    ann = fake.add("CINFRA", text="Required action", username="infra-coordinator", bot_id=INFRA_BOT)
    t = ann["ts"]
    for who, status, note in (("pouw-coordinator", "done", None), ("circuit-coordinator", "blocked", "a job runs until 20:00Z"),
                              ("pous-worker-3", "done", None)):
        argv = ["ack", "--channel", "infra", "--thread", t, "--as", who, "--status", status] + (["--note", note] if note else [])
        assert run(capsys, *argv)[0] == 0
    fake.add("CINFRA", text="ACK done", user="UINTRUDER", thread_ts=t)                    # a human's text is no ack
    rc, out, _ = run(capsys, "roster", "--channel", "infra", "--thread", t, "--json")
    rows = {r["name"]: r for r in json.loads(out)}
    assert rc == 1
    assert rows["pouw-coordinator"]["status"] == "done" and rows["circuit-coordinator"]["note"] == "a job runs until 20:00Z"
    assert rows["proof-coordinator"]["status"] == "missing" and "infra-coordinator" not in rows
    assert rows["pous-worker-3"]["registered"] is False
    for who in ("top-level", "circuit-coordinator", "proof-coordinator", "console-agent"):
        run(capsys, "ack", "--channel", "infra", "--thread", t, "--as", who, "--status", "done")
    assert run(capsys, "roster", "--channel", "infra", "--thread", t)[0] == 0


def test_verify_author_fetches_the_message_and_fails_closed(fake, registry, capsys):
    good = fake.add("CINFRA", text="Required action", username="infra-coordinator", bot_id=INFRA_BOT)
    spoof = fake.add("CINFRA", text="Required action", username="infra-coordinator", bot_id=BOT)     # the shared bot under its name
    human = fake.add("CINFRA", text="do this", user="UDANIEL")
    stranger = fake.add("CINFRA", text="do this", user="USTRANGER")
    alert = fake.add("CALERTS", text="disk 95%", bot_id="BALERTS", username="alerts")

    def verify(ev, *extra):
        return run(capsys, "verify-author", "--event", json.dumps(ev), "--why", *extra)

    assert verify({"ts": good["ts"], "channel": "CINFRA"})[0] == 0
    assert verify({"ts": spoof["ts"], "channel": "CINFRA", "bot_id": INFRA_BOT})[0] == 1            # the payload's claim is not believed
    assert verify({"ts": human["ts"], "channel": "CINFRA"})[0] == 0
    assert verify({"ts": stranger["ts"], "channel": "CINFRA"})[0] == 1
    assert verify({"ts": alert["ts"], "channel": "CALERTS"})[0] == 0
    assert verify({"ts": "9.9", "channel": "CINFRA"})[0] == 1
    rc, _, err = verify({"ts": good["ts"], "channel": "CCONSOLE"}, "--channel", "infra")
    assert rc == 1 and "not #infra" in err
    rc, _, err = verify({"ts": good["ts"], "channel": "CUNKNOWN"})
    assert rc == 2 and "pass --channel" in err


def test_verify_author_allows_nobody_when_the_allowlist_has_no_ids(fake, registry, capsys):
    reg = json.loads(registry.read_text())
    reg["humans"][0]["slack_user"] = None
    for h in reg["handles"]:
        h.pop("bot_id", None)
    registry.write_text(json.dumps(reg))
    for m in (fake.add("CINFRA", text="x", user="UDANIEL"), fake.add("CINFRA", text="x", username="infra-coordinator", bot_id=INFRA_BOT)):
        assert run(capsys, "verify-author", "--event", json.dumps({"ts": m["ts"], "channel": "CINFRA"}))[0] == 1


def test_groups_sync_plans_by_default_then_creates_updates_enables_and_records_ids(fake, registry, capsys):
    fake.groups = [{"id": "SINFRA", "name": "infra-coordinator", "handle": "infra-coordinator", "description": "old", "date_delete": 0},
                   {"id": "SPROOF", "name": "proof-coordinator", "handle": "proof-coordinator", "description": "", "date_delete": 1700000000}]
    before = registry.read_text()
    rc, out, _ = run(capsys, "groups", "sync")
    assert rc == 0 and "create  @pouw-coordinator" in out and "update  @infra-coordinator" in out and "enable  @proof-coordinator" in out
    assert "plan only" in out and registry.read_text() == before
    assert {m for m, _ in fake.calls} == {"usergroups.list"}
    rc, out, _ = run(capsys, "groups", "sync", "--apply")
    assert rc == 0 and "wrote" in out
    reg = json.loads(registry.read_text())
    ids = {h["handle"]: h["usergroup"] for h in reg["handles"]}
    assert len(fake.groups) == 6 and ids == {g["handle"]: g["id"] for g in fake.groups}
    assert all(not g["date_delete"] for g in fake.groups)
    rc, out, _ = run(capsys, "groups", "sync", "--apply")
    assert out.count("ok      @") == 6 and "wrote" not in out


def test_research_slack_is_a_cli_subcommand(registry, capsys):
    from research.cli import main

    assert main(["slack", "handles", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == slack.SCHEMA
