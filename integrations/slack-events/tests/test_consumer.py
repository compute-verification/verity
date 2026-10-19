"""The consumer: the lease per event id, routing by `research msg`'s `deliver`, no-model mode's @top, and the dead letter."""
from __future__ import annotations

import pytest

from conftest import CHAT, COORD, DANIEL, LUKE, TEAM
from slack_events import consumer, router
from slack_events.consumer import FAILED, LEASE_S, MAX_DELIVERIES, NO_MODEL, Retry, consume, event_key
from slack_events.deps import DEAD_COUNT, DEAD_TOPIC, LAST_SUCCESS, PENDING


def private(ts: str) -> str:
    return f"from luke, #chat-luke thread {ts}: (a post in a private channel: research msg read --channel chat-luke --thread {ts})"


def router_lines(world, ch: str, ts: str) -> list[str]:
    return [m["text"] for m in world.slack.thread(ch, ts) if m.get("username") == "router"]


def test_an_unaddressed_chat_post_rings_top_and_says_so_in_its_thread(world):
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    assert consume(world.deps, world.event(CHAT, post), 1) == "unaddressed: @top"
    assert world.rung("top") == [private(post["ts"])] and world.rung("circuits") == []
    assert router_lines(world, CHAT, post["ts"]) == [NO_MODEL]
    assert world.deps.store.get(LAST_SUCCESS) == f"{world.clock():.3f}"


def test_an_addressed_chat_post_rings_the_names_it_starts_with_and_posts_no_routing_line(world):
    post = world.slack.add(CHAT, text="@circuits is the TP8 sweep done?", user=LUKE)
    assert consume(world.deps, world.event(CHAT, post), 1) == "rule"
    assert world.rung("circuits") == [private(post["ts"])] and world.rung("top") == []
    assert router_lines(world, CHAT, post["ts"]) == []


def test_an_unaddressed_ask_the_team_post_rings_top_with_its_text_and_an_addressed_one_its_names(world):
    plain = world.slack.add(TEAM, text="what is the TP8 status?", user=DANIEL)
    assert consume(world.deps, world.event(TEAM, plain), 1) == "unaddressed: @top"
    assert world.rung("top") == [f"from daniel, #ask-the-team thread {plain['ts']}: what is the TP8 status?"]
    assert router_lines(world, TEAM, plain["ts"]) == [NO_MODEL]
    named = world.slack.add(TEAM, text="@circuits when does the sweep land?", user=DANIEL)
    assert consume(world.deps, world.event(TEAM, named), 1) == "rule"
    assert world.rung("circuits") == [f"from daniel, #ask-the-team thread {named['ts']}: @circuits when does the sweep land?"]


@pytest.mark.parametrize("where,msg,why", [
    (COORD, {"text": "anyone?", "user": DANIEL}, "front door: the sender delivers"),
    (CHAT, {"text": "pointers", "bot_id": "B0VERITY", "username": "circuits"}, "not a person's post"),
    (CHAT, {"text": "hi", "user": "U0STRANGER"}, "not a person's post"),
])
def test_what_the_consumer_leaves_alone(world, where, msg, why):
    post = world.slack.add(where, **msg)
    assert consume(world.deps, world.event(where, post), 1) == why
    assert world.rung("top") == world.rung("circuits") == []


def test_an_edit_or_another_event_type_is_acknowledged_and_ignored(world):
    edit = {"kind": "event", "event_id": "Ev9", "event": {"type": "message", "subtype": "message_changed", "channel": CHAT, "ts": "1.2"}}
    assert consume(world.deps, edit, 1) == "ignored: message/message_changed"
    assert consume(world.deps, {"kind": "event", "event_id": "Ev10", "event": {"type": "reaction_added"}}, 1).startswith("ignored")
    assert world.deps.store.get(event_key("Ev9")) == "done"


def test_done_blocks_a_second_run_of_the_same_event(world):
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    world.deps.store.zadd(PENDING, "EvA", world.clock())
    assert consume(world.deps, world.event(CHAT, post, "EvA"), 1) == "unaddressed: @top"
    assert consume(world.deps, world.event(CHAT, post, "EvA"), 2) == "duplicate"
    assert len(world.rung("top")) == 1 and len(router_lines(world, CHAT, post["ts"])) == 1
    assert world.deps.store.zmin(PENDING) is None


def test_a_held_lease_retries_and_an_expired_one_is_taken_over(world):
    """A delivery that crashed mid-way leaves its lease; the next waits it out, then takes it and finishes the work."""
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    payload = world.event(CHAT, post, "EvB")
    assert world.deps.store.claim(event_key("EvB"), "processing:crashed", LEASE_S)
    with pytest.raises(Retry) as r:
        consume(world.deps, payload, 2)
    assert r.value.delay == LEASE_S and world.rung("top") == []
    world.clock.t += LEASE_S - 1
    with pytest.raises(Retry):
        consume(world.deps, payload, 3)
    world.clock.t += 1
    assert consume(world.deps, payload, 4) == "unaddressed: @top" and len(world.rung("top")) == 1
    assert world.deps.store.get(event_key("EvB")) == "done"


def test_the_same_post_by_two_event_ids_is_routed_once(world):
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    assert consume(world.deps, world.event(CHAT, post, "EvC"), 1) == "unaddressed: @top"
    assert consume(world.deps, world.event(CHAT, post, f"reconcile:{CHAT}:{post['ts']}"), 1) == "already routed"
    assert len(world.rung("top")) == 1 and len(router_lines(world, CHAT, post["ts"])) == 1


def test_a_failing_post_retries_with_backoff_then_is_dead_lettered_said_in_its_thread_and_rung_to_top(world, monkeypatch):
    post = world.slack.add(CHAT, text="@circuits is the TP8 sweep done?", user=LUKE)
    payload = world.event(CHAT, post, "EvD")

    def broken(*a, **kw):
        raise RuntimeError("secret text must not be logged")
    monkeypatch.setattr(consumer, "route", broken)
    delays = []
    for n in range(1, MAX_DELIVERIES):
        with pytest.raises(Retry) as r:
            consume(world.deps, payload, n)
        delays.append(r.value.delay)
    assert delays == sorted(delays) and delays[0] > 0 and world.deps.queue.on(DEAD_TOPIC) == []
    assert consume(world.deps, payload, MAX_DELIVERIES) == "dead-lettered"
    [dead] = world.deps.queue.on(DEAD_TOPIC)
    assert dead == {"payload": payload, "error": "RuntimeError", "deliveries": MAX_DELIVERIES}
    assert world.deps.store.get(DEAD_COUNT) == "1" and world.deps.store.get(event_key("EvD")) == "dead"
    assert router_lines(world, CHAT, post["ts"]) == [FAILED] and world.rung("top") == [private(post["ts"])]
    assert consume(world.deps, payload, MAX_DELIVERIES + 1) == "duplicate"
    assert "secret text" not in world.deps.out.getvalue()


def test_a_doorbell_that_does_not_ring_is_a_failure_and_the_retry_rings_only_what_is_missing(world):
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    payload = world.event(CHAT, post, "EvE")
    world.slack.fail = {"chat.postMessage"}
    with pytest.raises(Retry):
        consume(world.deps, payload, 1)
    world.slack.fail = set()
    assert consume(world.deps, payload, 2) == "unaddressed: @top"
    assert world.rung("top") == [private(post["ts"])] and router_lines(world, CHAT, post["ts"]) == [NO_MODEL]


def test_dead_lettering_when_slack_itself_is_down_still_parks_the_message(world):
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    world.slack.fail = {"chat.postMessage"}
    assert consume(world.deps, world.event(CHAT, post, "EvF"), MAX_DELIVERIES) == "dead-lettered"
    assert len(world.deps.queue.on(DEAD_TOPIC)) == 1 and world.rung("top") == []


def test_the_model_hook_is_off_unless_enabled_and_its_failure_sends_the_post_to_top(world, monkeypatch):
    assert router.route_unaddressed({"text": "x"}, [], world.deps.registry) is None
    monkeypatch.setenv("ROUTER_MODEL_ENABLED", "1")
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    assert consume(world.deps, world.event(CHAT, post), 1) == "unaddressed: @top"
    assert router_lines(world, CHAT, post["ts"]) == [consumer.MODEL_DOWN]


@pytest.mark.parametrize("got,names,line", [
    (["circuits", "top", "nobody"], ["circuits"], "routed to @circuits"),
    (["top"], ["top"], consumer.NO_VALID),
    ([], ["top"], consumer.NO_VALID),
])
def test_a_models_answer_rings_only_valid_handles_and_never_top_by_choice(world, monkeypatch, got, names, line):
    monkeypatch.setattr(router, "route_unaddressed", lambda post, thread, reg: got)
    assert consumer.pick(world.deps, {"text": "x"}, [], world.deps.registry) == (names, line)


def test_logs_carry_ids_and_decisions_never_a_posts_text(world):
    texts = ["how do I read a PoUW receipt?", "@circuits is the TP8 sweep done?"]
    for i, t in enumerate(texts):
        post = world.slack.add(CHAT, text=t, user=LUKE)
        consume(world.deps, world.event(CHAT, post, f"EvL{i}"), 1)
    logs, raw = world.logs(), world.deps.out.getvalue()
    assert {line["event"] for line in logs} >= {"deliver", "done"}
    assert not any(w in raw for t in texts for w in t.split() if len(w) > 3 and not w.startswith("@"))
