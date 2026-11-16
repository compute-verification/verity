import json
from datetime import datetime, timezone

import pytest

from conftest import BOT, COORD, T0, add_route, rs
from slack_events import comms, consumer
from test_app import callback, client, post_event

RELAY = "https://relay.example/api/comms-relay"


def test_only_comms_router_doorbells_go_to_the_platform(world, monkeypatch, tmp_path):
    d = world.deps
    inbox, source, event_ts = world.boxes["circuits"], f"{int(T0)-3}.000001", f"{int(T0)}.000001"
    add_route(tmp_path / "notes" / rs.ROUTES_DIR, "comms", inbox)
    d.env.update(COMMS_RELAY_INBOX=f"{COORD}/{inbox}", COMMS_RELAY_URL=RELAY, COMMS_RELAY_CANARY_THREAD=source,
                 COMMS_RELAY_UNTIL=datetime.fromtimestamp(T0 + 80000, timezone.utc).isoformat())
    raw = callback()
    ev = raw["event"]
    ev.update(type="message", channel=COORD, thread_ts=inbox, ts=event_ts, username="router", bot_id=BOT,
              text=f"from codex, thread {source}: @comms [comms relay test] Please check this one item.")
    for changed in ({"channel": "COTHER"}, {"thread_ts": world.boxes["top"]}, {"username": "comms"},
                    {"bot_id": "another"}, {"subtype": "message_changed"}):
        assert not comms.candidate(d, {**ev, **changed})
    assert comms.candidate(d, ev)
    assert post_event(client(world), world, raw).status_code == 200
    data = d.queue.sent[-1][1]
    assert data["kind"] == "comms-relay"
    sent = []
    def exchange(url, body):
        assert url == RELAY
        sent.append(body)
        return {"status": "answer", "reply": "Checked."} if len(sent) == 2 else {"status": "pending" if len(sent) == 1 else "check_receipt"}
    monkeypatch.setattr(comms, "exchange", exchange)
    with pytest.raises(consumer.Retry):
        consumer.consume(d, data, 1)
    assert consumer.consume(d, data, 2) == "comms: answered"
    assert consumer.consume(d, data, 3) == "comms: reply already delivered"
    assert {k for k in sent[0]} == {"channel", "inbox", "ts", "text"}
    assert len([m for m in world.slack.thread(COORD, source) if m.get("username") == "comms"]) == 1
    orig = data["event"]["text"]
    data["event"]["text"] = orig.replace("thread ", "#chat-someone thread ")
    assert consumer.process(d, data) == "comms: Cursor holds inbox"
    data["event"]["text"] = orig
    data["event"]["text"] = orig.replace("[comms relay test]", "regular")
    assert consumer.process(d, data) == "comms: Cursor holds inbox"
    data["event"]["text"] = orig.replace("from codex", "from infra")
    assert consumer.process(d, data) == "comms: Cursor holds inbox"
    data["event"]["text"] = orig
    del d.env["COMMS_RELAY_CANARY_THREAD"]
    assert consumer.process(d, data) == "comms: Cursor holds inbox"
    assert len(sent) == 3


@pytest.fixture
def platform_delivery(world, tmp_path):
    inbox = world.boxes["circuits"]
    route_file = tmp_path / "notes" / rs.ROUTES_DIR / "comms.json"
    add_route(route_file.parent, "comms", inbox)
    route_file.write_text(json.dumps({**json.loads(route_file.read_text()), "holder": comms.SEAT}))
    world.deps.env.update(COMMS_RELAY_INBOX=f"{COORD}/{inbox}", COMMS_RELAY_URL=RELAY,
                          COMMS_RELAY_UNTIL=datetime.fromtimestamp(T0 + 3000, timezone.utc).isoformat())
    ev = dict(type="message", channel=COORD, thread_ts=inbox, ts=f"{int(T0)}.000001",
              username="router", bot_id=BOT, text=f"from infra, thread {int(T0)-5}.000001: please check")
    return {"kind": "comms-relay", "event_id": "comms:" + ev["ts"], "event": ev}, route_file


@pytest.mark.parametrize("text", ["from ci, #chat-private thread 1790000000.000001: private detail",
                                  "from ci, Linear COMM-112 (<https://linear.app/test>): private detail"])
def test_unsupported_source_with_platform_holder_rings_top_without_sharing_content(world, platform_delivery,
                                                                                  monkeypatch, text):
    data, route_file = platform_delivery
    data["event"]["text"] = text
    monkeypatch.setattr(comms, "exchange", lambda *_: pytest.fail("unsupported source forwarded"))
    assert comms.candidate(world.deps, data["event"])
    assert consumer.consume(world.deps, data, 1) == "dead-lettered"
    assert consumer.consume(world.deps, data, 2) == "dead-lettered"
    assert len(world.rung("top")) == 1 and data["event"]["ts"] in world.rung("top")[0]
    assert "private detail" not in str(world.rung("top")) + str(world.deps.queue.sent)
    add_route(route_file.parent, "comms", data["event"]["thread_ts"])
    assert consumer.consume(world.deps, data, 3) == "comms: Cursor holds inbox"


def test_pending_polls_do_not_exhaust_retries_for_a_transient_failure(world, platform_delivery, monkeypatch):
    data, _ = platform_delivery
    results = iter([{"status": "pending"}, TimeoutError(), {"status": "answer", "reply": "recovered"}])
    def exchange(*_):
        current = next(results)
        if isinstance(current, Exception):
            raise current
        return current
    monkeypatch.setattr(comms, "exchange", exchange)
    for count in (9, 10):
        with pytest.raises(consumer.Retry):
            consumer.consume(world.deps, data, count)
    assert consumer.consume(world.deps, data, 11) == "comms: answered"


def test_route_failure_after_many_polls_retries_until_deadline(world, platform_delivery, monkeypatch):
    data, route_file = platform_delivery
    route_file.unlink()
    monkeypatch.setattr(comms, "exchange", lambda *_: pytest.fail("unverified route"))
    with pytest.raises(consumer.Retry):
        consumer.consume(world.deps, data, 28)
    world.clock.t += comms.MAX_WAIT_S + 1
    assert consumer.consume(world.deps, data, 29) == "dead-lettered"
    assert len(world.rung("top")) == 1


def test_window_or_missing_configuration_cannot_forward(world, platform_delivery, monkeypatch):
    data, _ = platform_delivery
    monkeypatch.setattr(comms, "exchange", lambda *_: pytest.fail("closed relay"))
    for key in ("COMMS_RELAY_INBOX", "COMMS_RELAY_UNTIL", "COMMS_RELAY_URL"):
        with monkeypatch.context() as m:
            m.delitem(world.deps.env, key)
            assert not comms.candidate(world.deps, data["event"])
    with monkeypatch.context() as m:
        m.setitem(world.deps.env, "COMMS_RELAY_URL", "http://relay.example/api/comms-relay")
        assert not comms.candidate(world.deps, data["event"])
    world.clock.t += 3001
    assert not comms.candidate(world.deps, data["event"])
    assert consumer.consume(world.deps, data, 5) == "dead-lettered"
    assert len(world.rung("top")) == 1


def test_holder_changes_while_answer_is_in_flight(world, platform_delivery, monkeypatch):
    data, route_file = platform_delivery
    def exchange(*_):
        add_route(route_file.parent, "comms", data["event"]["thread_ts"])
        return {"status": "answer", "reply": "from former holder"}
    monkeypatch.setattr(comms, "exchange", exchange)
    assert consumer.consume(world.deps, data, 1) == "dead-lettered"
    assert len(world.rung("top")) == 1
    assert not any(m.get("username") == "comms" for m in world.slack.messages[COORD])
