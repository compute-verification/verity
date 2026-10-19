"""The service against fakes: a Slack Web API in memory (the part of `tools/research/tests/test_slack.py`'s FakeSlack the
service calls, answered through `Slack`'s transport, so no socket), a Redis, a queue and a clock the tests move.  No real
token, workspace, Redis or queue is involved."""
from __future__ import annotations

import io
import itertools
import json
import urllib.parse
from pathlib import Path

import pytest

from slack_events import vendor
from slack_events.deps import Deps

rs = vendor.slack()
REPO = Path(__file__).resolve().parents[3]
TOKEN = "dummy-bot-token"
SIGNING = "dummy-signing-secret"
BOT, COORD, TEAM = "B0VERITY", "CCOORD", "C0CA1JZSYNR"
LUKE, DANIEL = "U0LUKE", "UDANIEL"
CHAT = "CCHATLUKE"
T0 = 1_800_000_000.0                                           # 2027-01-15 08:00 UTC


class Clock:
    def __init__(self, t: float = T0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeSlack:
    def __init__(self, clock: Clock):
        self.clock, self.messages, self.channels = clock, {}, {}
        self.calls: list[tuple[str, dict]] = []
        self.fail: set[str] = set()                            # methods that answer an error
        self._n = itertools.count(1)

    def add(self, channel: str, **msg) -> dict:
        msg.setdefault("ts", f"{int(self.clock())}.{next(self._n):06d}")
        msg.setdefault("type", "message")
        self.messages.setdefault(channel, []).append(msg)
        return msg

    def transport(self, url: str, data: bytes, headers: dict) -> dict:
        p = {k: v[0] for k, v in urllib.parse.parse_qs(data.decode()).items()}
        return self.handle(url.rsplit("/", 1)[-1], p, headers.get("Authorization", ""))

    def handle(self, method: str, p: dict, auth: str) -> dict:
        self.calls.append((method, p))
        if auth != f"Bearer {TOKEN}":
            return {"ok": False, "error": "invalid_auth"}
        if method in self.fail:
            return {"ok": False, "error": "internal_error"}
        ch = p.get("channel")
        if method == "chat.postMessage":
            m = self.add(ch, text=p["text"], username=p.get("username"), bot_id=BOT,
                         **({"thread_ts": p["thread_ts"]} if p.get("thread_ts") else {}))
            return {"ok": True, "channel": ch, "ts": m["ts"]}
        if method in ("conversations.history", "conversations.replies"):
            msgs = self.messages.get(ch, [])
            if method == "conversations.replies":
                msgs = [m for m in msgs if m["ts"] == p["ts"] or m.get("thread_ts") == p["ts"]]
            else:
                msgs = [self._root(ch, m) for m in msgs if not m.get("thread_ts") or m["thread_ts"] == m["ts"]][::-1]
            if "oldest" in p:
                msgs = [m for m in msgs if float(m["ts"]) >= float(p["oldest"])]
            return {"ok": True, "messages": msgs[:int(p.get("limit", 100))], "response_metadata": {"next_cursor": ""}}
        if method == "conversations.list":
            return {"ok": True, "channels": list(self.channels.values()), "response_metadata": {"next_cursor": ""}}
        if method == "conversations.info":
            c = self.channels.get(ch)
            return {"ok": True, "channel": c} if c else {"ok": False, "error": "channel_not_found"}
        return {"ok": False, "error": "unknown_method"}

    def _root(self, ch: str, m: dict) -> dict:
        replies = [r for r in self.messages.get(ch, []) if r.get("thread_ts") == m["ts"] and r["ts"] != m["ts"]]
        return {**m, "reply_count": len(replies), "latest_reply": replies[-1]["ts"]} if replies else m

    def thread(self, ch: str, ts: str) -> list[dict]:
        return [m for m in self.messages.get(ch, []) if m["ts"] == ts or m.get("thread_ts") == ts]


class FakeStore:
    """Redis, with expiry by the tests' clock."""

    def __init__(self, clock: Clock):
        self.clock, self.kv, self.z = clock, {}, {}

    def _live(self, key):
        v = self.kv.get(key)
        if v and v[1] is not None and v[1] <= self.clock():
            del self.kv[key]
            return None
        return v

    def get(self, key):
        v = self._live(key)
        return v[0] if v else None

    def set(self, key, value, ttl_s=None):
        self.kv[key] = (str(value), None if ttl_s is None else self.clock() + ttl_s)

    def claim(self, key, value, ttl_s):
        if self._live(key):
            return False
        self.set(key, value, ttl_s)
        return True

    def release(self, key, value):
        if self.get(key) == value:
            del self.kv[key]

    def incr(self, key):
        n = int(self.get(key) or 0) + 1
        self.set(key, n)
        return n

    def zadd(self, key, member, score):
        self.z.setdefault(key, {})[member] = score

    def zrem(self, key, member):
        self.z.get(key, {}).pop(member, None)

    def zmin(self, key):
        return min(self.z.get(key, {}).values(), default=None)


class FakeQueue:
    """A topic's messages, each idempotency key accepted once."""

    def __init__(self):
        self.sent: list[tuple[str, dict, str]] = []
        self.fail = False

    def send(self, topic, payload, *, idempotency_key):
        if self.fail:
            raise ConnectionError("queue unreachable")
        if any(k == idempotency_key and t == topic for t, _, k in self.sent):
            return
        self.sent.append((topic, json.loads(json.dumps(payload)), idempotency_key))

    def on(self, topic: str) -> list[dict]:
        return [p for t, p, _ in self.sent if t == topic]


def add_route(where: Path, name: str, inbox_ts: str) -> None:
    (where / f"{name}.json").write_text(json.dumps({
        "schema": rs.ROUTE_SCHEMA, "name": name, "holder": f"bc-{name}", "channel": COORD, "inbox_ts": inbox_ts,
        "opened": "2026-10-01T17:00Z", "closed": None, "forward_to": None}))


@pytest.fixture
def registry() -> dict:
    reg = json.loads((REPO / rs.REGISTRY_REL).read_text())
    reg["bot_id"] = BOT
    reg["channels"]["agent-coordination"]["id"] = COORD
    reg["channels"]["ask-the-team"]["id"] = TEAM
    reg["humans"][0]["slack_user"] = DANIEL
    for h in reg["handles"]:
        for k in ("holder", "branch_suffixes", "posters"):
            h.pop(k, None)
    return reg


@pytest.fixture
def world(tmp_path, monkeypatch, registry):
    """The deps the service runs on, with Luke's chat (people.d published beside routes.d) and inboxes for @top and
    @circuits."""
    notes, published = tmp_path / "notes", tmp_path / "published"
    (notes / rs.ROUTES_DIR).mkdir(parents=True)
    published.mkdir()
    (tmp_path / rs.PEOPLE_DIR).mkdir()
    (tmp_path / rs.PEOPLE_DIR / "luke.json").write_text(json.dumps({
        "schema": rs.PERSON_SCHEMA, "name": "Luke Example", "slack_user": LUKE, "email": None, "note": "x",
        "channel": {"key": "chat-luke", "id": CHAT}, "opened": "2026-10-19T17:30Z", "closed": None}))
    for k in ("RESEARCH_SLACK_REGISTRY", "RESEARCH_AGENT_URL", "ROUTER_MODEL_ENABLED"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RESEARCH_NOTES", str(notes))
    monkeypatch.setenv("RESEARCH_ROUTES_URL", f"file://{published}")
    monkeypatch.setenv("RESEARCH_PR_BRANCHES", str(tmp_path / "pr-branches.json"))
    monkeypatch.setattr(rs, "GH", str(tmp_path / "no-gh"))
    clock = Clock()
    slack = FakeSlack(clock)
    slack.channels[CHAT] = {"id": CHAT, "name": "chat-luke", "is_private": True, "is_archived": False}
    boxes = {}
    for name in ("top", "circuits"):
        root = slack.add(COORD, text=f"{rs.INBOX_HEAD}{name}. pointers", username=name, bot_id=BOT, ts=f"{int(T0) - 86400 * 7}.{len(boxes) + 1:06d}")
        add_route(notes / rs.ROUTES_DIR, name, root["ts"])
        boxes[name] = root["ts"]
    d = Deps(rs=rs, registry=registry, store=FakeStore(clock), queue=FakeQueue(),
             slack=rs.Slack(TOKEN, transport=slack.transport, direct=True), clock=clock,
             env={"SLACK_SIGNING_SECRET": SIGNING, "HEALTH_TOKEN": "dummy-health", "CRON_SECRET": "dummy-cron"},
             out=io.StringIO())
    return World(d, slack, clock, boxes)


class World:
    def __init__(self, deps: Deps, slack: FakeSlack, clock: Clock, boxes: dict[str, str]):
        self.deps, self.slack, self.clock, self.boxes = deps, slack, clock, boxes

    def rung(self, name: str) -> list[str]:
        """The doorbells in `name`'s inbox."""
        return [m["text"] for m in self.slack.thread(COORD, self.boxes[name]) if m["ts"] != self.boxes[name]]

    def logs(self) -> list[dict]:
        return [json.loads(line) for line in self.deps.out.getvalue().splitlines()]

    def event(self, channel: str, msg: dict, eid: str | None = None) -> dict:
        """The queue's payload for Slack's event of `msg`."""
        return {"kind": "event", "event_id": eid or f"Ev{msg['ts'].replace('.', '')}",
                "event": {"type": "message", "channel": channel, **msg}, "source": "events"}
