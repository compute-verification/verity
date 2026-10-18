"""Infra's relay posts each line the agent watch adds to events.jsonl on the control pod, once, from the offset it last posted;
what the watch leaves for a person leads with @top and rings @top's inbox."""
import importlib.util
import json
from pathlib import Path

import pytest

from verity_ops import agent_watch as W

RELAY = Path(__file__).resolve().parents[2] / "control" / "agent_watch_relay.py"
TOP_INBOX = {"name": "top", "holder": "bc-af075d78-d5b9-590b-acbb-eb47d6f88f39", "inbox_ts": "1760000000.000100", "closed": None}


class Routes:
    route = dict(TOP_INBOX)

    def resolve(self, name):
        return name, (self.route if name == "top" else None)


class Api:
    def __init__(self):
        self.calls = []

    def call(self, method, **kw):
        self.calls.append((method, kw))
        return {}


@pytest.fixture
def relay(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("agent_watch_relay", RELAY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    log, posts, wrote, api = {"text": ""}, [], [], Api()
    monkeypatch.setattr(m, "OFFSET", tmp_path / "offset")
    monkeypatch.setattr(m, "DRY", False)
    monkeypatch.setattr(m, "bells", lambda api, now: {"t": "2026-10-18T22:00:00Z", "inboxes": {}})
    monkeypatch.setattr(m, "pod", lambda cmd, data=None: log["text"] if data is None else wrote.append((cmd, data)) or "")
    monkeypatch.setattr(m.slack, "Slack", lambda token: api)
    monkeypatch.setattr(m.slack, "Routes", Routes)
    monkeypatch.setattr(m.slack, "_post", lambda api, ch, handle, text, *a, **k: posts.append(text) or f"1760000{len(posts)}.0")
    m.log_text, m.posts, m.wrote, m.api = log, posts, wrote, api
    return m


def line(run: str, kind: str = "held", **detail) -> str:
    return json.dumps({"kind": kind, "agent": "bc-x", "name": "x", "run": run, "since": "2026-10-18T20:00:00Z", "minutes": 15,
                       "action": "cancelled and continued", **detail})


def rings(relay):
    return [kw for method, kw in relay.api.calls if method == "chat.postMessage"]


def test_a_line_still_being_written_waits_for_the_next_pass(relay):
    relay.log_text["text"] = line("run-1") + "\n" + line("run-2")[:30]
    relay.once()
    relay.log_text["text"] = line("run-1") + "\n" + line("run-2") + "\n"
    relay.once()
    assert [p.count("run-1") + 2 * p.count("run-2") for p in relay.posts] == [1, 2], "each posted once, whole"
    assert relay.OFFSET.read_text() == "2"
    assert [cmd for cmd, _ in relay.wrote] == [f"cat > {relay.BELLS}.tmp && mv {relay.BELLS}.tmp {relay.BELLS}"] * 2


def test_a_log_replaced_with_a_shorter_one_is_posted_from_the_top(relay):
    relay.OFFSET.write_text("5")
    relay.log_text["text"] = line("run-a") + "\n" + line("run-b") + "\n"
    relay.once()
    assert len(relay.posts) == 2 and relay.OFFSET.read_text() == "2"


def test_a_refused_post_is_posted_again_next_pass_and_the_relay_goes_on(relay, monkeypatch):
    relay.log_text["text"] = line("run-1") + "\n" + line("run-2") + "\n"

    def refuse_second(api, ch, handle, text, *a, **k):
        if "run-2" in text:
            raise relay.slack.SlackError("ratelimited")
        relay.posts.append(text)
        return "1760000001.0"
    monkeypatch.setattr(relay.slack, "_post", refuse_second)
    monkeypatch.setattr(relay.sys, "argv", ["agent_watch_relay.py", "--once"])
    assert relay.main() == 0
    assert relay.OFFSET.read_text() == "1"


def test_what_is_left_for_a_person_leads_with_top_and_rings_its_inbox(relay):
    top = relay.slack.tag(relay.slack.entry(relay.REG, "top"))
    person = [line("run-e", "erroring", restarts=3), line("run-s", "stuck", status="CREATING"), line("run-r", "cancel_refused"),
              line("run-h", "held_again"), line("run-goal-u", "unreadable", status="HTTP 400")]
    channel = [line("run-1"), line("run-c", "crashed", action="restarted (1 of 3)"), line("run-l", "long_call"),
               line("run-w", "ready_to_wake", handle="vllm", reason="nothing has run for 40 min", action="recorded (--wake is off)"),
               json.dumps({"kind": "unreadable", "agent": "bc-y", "run": "run-goal-v", "status": "HTTP 400"})]
    relay.log_text["text"] = "\n".join(person + channel) + "\n"
    relay.once()
    assert [p.startswith(f"{top} *Agent watch:*") for p in relay.posts] == [True] * 5 + [False] * 4, \
        "the first unreadable line is posted nowhere"
    assert "@vllm's coordinator" in relay.posts[-1] and top not in relay.posts[-1]
    assert set(W.WATCH_STATES) - {"ok", "followed"} == set(relay.SAY), "each kind the watch records has its sentence"
    ring = rings(relay)
    assert len(ring) == 5 and {r["thread_ts"] for r in ring} == {TOP_INBOX["inbox_ts"]}
    assert {r["channel"] for r in ring} == {relay.slack.channel_id_for(relay.REG, relay.slack.FRONT_DOOR)}
    assert all(relay.slack.DOORBELL_RE.match(r["text"])[1] == "infra" for r in ring), "a doorbell the watch's rule reads"
    assert f"/archives/{relay.ALERTS}/p17600001" in ring[0]["text"] and "run-e" in ring[0]["text"]


def test_a_ring_that_fails_leaves_the_alert_posted_once(relay, monkeypatch):
    def refuse(method, **kw):
        raise relay.slack.SlackError("ratelimited")
    monkeypatch.setattr(relay.api, "call", refuse)
    relay.log_text["text"] = line("run-e", "erroring", restarts=3) + "\n"
    relay.once()
    relay.once()
    assert len(relay.posts) == 1 and relay.OFFSET.read_text() == "1"


def test_no_open_top_inbox_rings_nothing(relay, monkeypatch):
    monkeypatch.setattr(Routes, "route", {**TOP_INBOX, "closed": "2026-10-18T21:00Z"})
    relay.log_text["text"] = line("run-e", "erroring", restarts=3) + "\n"
    relay.once()
    assert len(relay.posts) == 1 and rings(relay) == []
