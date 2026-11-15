from datetime import datetime, timezone

import pytest

from conftest import BOT, COORD, T0, add_route, rs
from slack_events import comms, consumer
from test_app import callback, client, post_event


def test_only_comms_router_doorbells_go_to_the_platform(world, monkeypatch, tmp_path):
    d = world.deps
    inbox, source, event_ts = world.boxes["circuits"], f"{int(T0)-3}.000001", f"{int(T0)}.000001"
    add_route(tmp_path / "notes" / rs.ROUTES_DIR, "comms", inbox)
    d.env.update(COMMS_RELAY_INBOX=f"{COORD}/{inbox}", COMMS_RELAY_CANARY_THREAD=source,
                 COMMS_RELAY_UNTIL=datetime.fromtimestamp(T0 + 80000, timezone.utc).isoformat())
    raw = callback()
    ev = raw["event"]
    ev.update(type="message", channel=COORD, thread_ts=inbox, ts=event_ts, username="router", bot_id=BOT,
              text=f"from infra, thread {source}: Please check this one item.")
    for changed in ({"channel": "COTHER"}, {"thread_ts": world.boxes["top"]}, {"username": "comms"},
                    {"bot_id": "another"}, {"subtype": "message_changed"}):
        assert not comms.candidate(d, {**ev, **changed})
    assert comms.candidate(d, ev)
    assert post_event(client(world), world, raw).status_code == 200
    data = d.queue.sent[-1][1]
    assert data["kind"] == "comms-relay"
    sent = []
    def exchange(body):
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
    assert consumer.process(d, data) == "comms: source outside coordination"
    data["event"]["text"] = orig
    del d.env["COMMS_RELAY_CANARY_THREAD"]
    assert consumer.process(d, data) == "comms: Cursor holds inbox"
    assert len(sent) == 3
