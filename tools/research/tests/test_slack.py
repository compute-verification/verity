"""`research slack` against a fake Slack Web API: a local HTTP server that keeps channels, threads, reactions and user groups the
way the real one returns them. No real token or workspace is involved (conftest clears SLACK_*)."""
from __future__ import annotations

import http.server
import io
import itertools
import json
import threading
import urllib.parse
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
    assert set(reg["channels"]) == {"agent-coordination", "agent-alerts", "approvals"}
    assert slack.subscribers(reg, "agent-coordination") == [h["handle"] for h in reg["handles"]]
    assert slack.subscribers(reg, "agent-alerts") == slack.subscribers(reg, "approvals") == ["infra"]


def test_handles_and_subscriptions(registry, capsys):
    rc, out, _ = run(capsys, "handles")
    assert rc == 0 and "@infra" in out and "SINFRA" in out and "#agent-alerts" in out
    rc, out, _ = run(capsys, "subscriptions", "--as", "infra")
    assert rc == 0 and f'channelId="{COORD}", topLevelOnly=true' in out and f'channelId="{ALERTS}", topLevelOnly=true' in out
    rc, out, _ = run(capsys, "subscriptions", "--as", "compute-accounting")
    assert rc == 0 and COORD in out and ALERTS not in out


def test_subscriptions_fail_visibly_without_channel_ids(registry, capsys):
    reg = json.loads(registry.read_text())
    reg["channels"]["agent-coordination"]["id"] = None
    registry.write_text(json.dumps(reg))
    rc, out, _ = run(capsys, "subscriptions", "--as", "compute-accounting")
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


def test_no_token_is_an_error_and_dry_run_sends_nothing(fake, registry, capsys, monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN")
    rc, _, err = run(capsys, "ask", "--as", "memory-accounting", "--to", "infra", "--text", "x")
    assert rc == 2 and "SLACK_BOT_TOKEN is not set" in err
    rc, _, err = run(capsys, "ask", "--as", "memory-accounting", "--to", "infra", "--text", "x", "--dry-run")
    assert rc == 0 and "dry-run: chat.postMessage" in err and not fake.calls


@pytest.mark.parametrize("argv", [
    ["ask", "--as", "memory-accounting", "--to", "infra", "--text", "x"], ["announce", "--as", "infra", "--text", "x"],
    ["reply", "--as", "memory-accounting", "--thread", "1.0", "--text", "x"], ["pickup", "--thread", "1.0"],
    ["done", "--as", "memory-accounting", "--thread", "1.0", "--link", "https://x"], ["decline", "--as", "memory-accounting", "--thread", "1.0", "--why", "no"],
    ["roster", "--thread", "1.0"], ["read", "--since", "1.0"],
    ["match", "--as", "memory-accounting", "--event", '{"text":"x","ts":"2.0","thread_ts":"1.0","channel":"CCOORD"}'],
    ["verify-author", "--event", '{"ts":"1.0","channel":"CCOORD"}'], ["groups", "sync", "--apply"],
])
def test_every_subcommand_has_a_dry_run_that_sends_nothing(fake, registry, capsys, argv):
    slack.main([*argv, "--dry-run"])
    assert not fake.calls
    assert TOKEN not in capsys.readouterr().err


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
    assert rows["compute-accounting"]["status"] == "done" and rows["circuits"] == {**rows["circuits"], "status": "declined", "detail": "a job runs until 23:00Z"}
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
