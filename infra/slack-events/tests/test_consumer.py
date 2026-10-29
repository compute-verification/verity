"""The consumer: routing by `research msg`'s `deliver` with Slack as the record (the routing line read back on a
redelivery), no-model mode's @top, the dead letter and the newcomers' `team_join`."""
from __future__ import annotations

import json

import pytest

from conftest import ACCT, BOT, BOT_USER, CHAT, COORD, DANIEL, LUKE, TEAM
from slack_events import consumer, router
from slack_events.consumer import FAILED, MAX_DELIVERIES, NO_MODEL, ROUTING_LINE, START_HERE, Retry, consume
from slack_events.deps import DEAD, DEAD_TOPIC, LAST_SUCCESS


def private(ts: str, key: str = "chat-luke", who: str = "luke") -> str:
    return f"from {who}, #{key} thread {ts}: (a post in a private channel: research msg read --channel {key} --thread {ts})"


def router_lines(world, ch: str, ts: str) -> list[str]:
    return [m["text"] for m in world.lines(ch, ts)]


class Model:
    """The model step, answering `names` and counting its calls."""

    def __init__(self, monkeypatch, names: list[str]):
        self.names, self.calls = names, 0
        monkeypatch.setattr(router, "route_unaddressed", self)

    def __call__(self, post, thread, reg):
        self.calls += 1
        return self.names


def test_an_unaddressed_chat_post_rings_top_after_a_routing_line_that_records_it(world):
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    assert consume(world.deps, world.event(CHAT, post), 1) == "unaddressed: @top"
    assert world.rung("top") == [private(post["ts"])] and world.rung("circuits") == []
    [line] = world.lines(CHAT, post["ts"])
    assert line["text"] == NO_MODEL and line["metadata"] == {"event_type": ROUTING_LINE,
                                                             "event_payload": {"post": post["ts"], "names": ["top"]}}
    assert world.deps.cache.get(LAST_SUCCESS) == world.clock()


def test_an_addressed_chat_post_rings_the_names_it_starts_with_and_posts_no_routing_line(world):
    post = world.slack.add(CHAT, text="@circuits is the TP8 sweep done?", user=LUKE)
    assert consume(world.deps, world.event(CHAT, post), 1) == "rule"
    assert world.rung("circuits") == [private(post["ts"])] and world.rung("top") == []
    assert router_lines(world, CHAT, post["ts"]) == []


@pytest.mark.parametrize("ch,key", [(TEAM, "ask-the-team"), (ACCT, "resource-accounting")])
def test_an_unaddressed_ask_post_rings_top_with_its_text_and_an_addressed_one_its_names(world, ch, key):
    plain = world.slack.add(ch, text="what is the TP8 status?", user=DANIEL)
    assert consume(world.deps, world.event(ch, plain), 1) == "unaddressed: @top"
    assert world.rung("top") == [f"from daniel, #{key} thread {plain['ts']}: what is the TP8 status?"]
    assert router_lines(world, ch, plain["ts"]) == [NO_MODEL]
    named = world.slack.add(ch, text="@circuits when does the sweep land?", user=DANIEL)
    assert consume(world.deps, world.event(ch, named), 1) == "rule"
    assert world.rung("circuits") == [f"from daniel, #{key} thread {named['ts']}: @circuits when does the sweep land?"]


@pytest.mark.parametrize("where,msg,why", [
    (COORD, {"text": "anyone?", "user": DANIEL}, "front door: the sender delivers"),
    (CHAT, {"text": "pointers", "bot_id": "B0VERITY", "username": "circuits"}, "not a person's post"),
    (CHAT, {"text": "hi", "user": "U0STRANGER"}, "not a person's post"),
])
def test_what_the_consumer_leaves_alone(world, where, msg, why):
    post = world.slack.add(where, **msg)
    assert consume(world.deps, world.event(where, post), 1) == why
    assert world.rung("top") == world.rung("circuits") == []


def test_an_external_posters_post_through_our_bot_rings_by_the_rule_even_at_the_front_door(world):
    """Codex, Daniel's desktop agent, posts through Verity's bot without `research msg`, so no sender rings for its asks to
    lanes.  The service delivers them, by the names they start with only."""
    codex = {"bot_id": BOT, "username": "Codex", "subtype": "bot_message"}
    ask = world.slack.add(COORD, text="@circuits @kernels both of you claimed the launch; who owns it?", **codex)
    assert consume(world.deps, world.event(COORD, ask), 1) == "rule"
    assert world.rung("circuits") == world.rung("kernels") == [
        f"from codex, thread {ask['ts']}: @circuits @kernels both of you claimed the launch; who owns it?"]
    assert consume(world.deps, world.event(COORD, ask, "EvAgain"), 1) == "rule" and len(world.rung("circuits")) == 1
    note = world.slack.add(COORD, text="readback done, nothing changed", **codex)
    assert consume(world.deps, world.event(COORD, note), 1) == "rule"
    assert world.rung("top") == [] and router_lines(world, COORD, note["ts"]) == []
    lane = world.slack.add(COORD, text="@kernels x", bot_id=BOT, username="circuits", subtype="bot_message")
    assert consume(world.deps, world.event(COORD, lane), 1) == "ignored: message/bot_message"


def test_an_edit_or_another_event_type_is_acknowledged_and_ignored(world):
    edit = {"kind": "event", "event_id": "Ev9", "event": {"type": "message", "subtype": "message_changed", "channel": CHAT, "ts": "1.2"}}
    assert consume(world.deps, edit, 1) == "ignored: message/message_changed"
    assert consume(world.deps, {"kind": "event", "event_id": "Ev10", "event": {"type": "reaction_added"}}, 1).startswith("ignored")


def test_a_redelivery_after_the_routing_line_rings_only_the_missing_lanes_with_no_second_pick(world, monkeypatch):
    """The first delivery picked, posted its line and rang @circuits, then failed to ring @kernels.  The redelivery finds
    the line, so the model isn't asked again, and rings @kernels alone; @circuits keeps its one doorbell."""
    model = Model(monkeypatch, ["circuits", "kernels"])
    post = world.slack.add(CHAT, text="how is the TP8 kernel validated?", user=LUKE)
    payload = world.event(CHAT, post, "EvR")
    world.slack.fail_threads = {world.boxes["kernels"]}
    with pytest.raises(Retry):
        consume(world.deps, payload, 1)
    assert router_lines(world, CHAT, post["ts"]) == ["routed to @circuits @kernels"]
    assert world.rung("circuits") == [private(post["ts"])] and world.rung("kernels") == []
    world.slack.fail_threads = set()
    model.names = ["top"]                                                  # a second pick would answer otherwise
    assert consume(world.deps, payload, 2) == "unaddressed: @circuits @kernels"
    assert model.calls == 1 and router_lines(world, CHAT, post["ts"]) == ["routed to @circuits @kernels"]
    assert world.rung("circuits") == world.rung("kernels") == [private(post["ts"])] and world.rung("top") == []


def test_a_crash_before_the_routing_line_is_handled_from_scratch_by_the_redelivery(world, monkeypatch):
    model = Model(monkeypatch, ["circuits"])
    post = world.slack.add(CHAT, text="how is the TP8 kernel validated?", user=LUKE)
    payload = world.event(CHAT, post, "EvS")
    world.slack.fail = {"chat.postMessage"}
    with pytest.raises(Retry):
        consume(world.deps, payload, 1)
    assert router_lines(world, CHAT, post["ts"]) == [] and world.rung("circuits") == []
    world.slack.fail = set()
    assert consume(world.deps, payload, 2) == "unaddressed: @circuits"
    assert model.calls == 2 and router_lines(world, CHAT, post["ts"]) == ["routed to @circuits"]
    assert world.rung("circuits") == [private(post["ts"])]


def test_a_line_someone_else_posted_or_one_for_another_post_is_not_the_record(world):
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    world.slack.add(CHAT, text="routed to @circuits", username="router", user=LUKE, thread_ts=post["ts"],
                    metadata={"event_type": ROUTING_LINE, "event_payload": {"post": post["ts"], "names": ["circuits"]}})
    world.slack.add(CHAT, text="routed to @circuits", username="router", bot_id="B0VERITY", thread_ts=post["ts"],
                    metadata={"event_type": ROUTING_LINE, "event_payload": {"post": "1.000001", "names": ["circuits"]}})
    assert consume(world.deps, world.event(CHAT, post), 1) == "unaddressed: @top"
    assert world.rung("circuits") == []


def test_the_same_post_by_two_event_ids_is_routed_once(world):
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    assert consume(world.deps, world.event(CHAT, post, "EvC"), 1) == "unaddressed: @top"
    assert consume(world.deps, world.event(CHAT, post, f"reconcile:{CHAT}:{post['ts']}"), 1) == "unaddressed: @top"
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
    assert world.deps.cache.get(DEAD) == {"count": 1, "since": world.clock()}
    [line] = world.lines(CHAT, post["ts"])
    assert line["text"] == FAILED and line["metadata"]["event_payload"] == {"post": post["ts"], "names": ["top"]}
    assert world.rung("top") == [private(post["ts"])]
    assert "secret text" not in world.deps.out.getvalue()


def test_a_doorbell_that_does_not_ring_is_a_failure_and_the_retry_rings_only_what_is_missing(world):
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    payload = world.event(CHAT, post, "EvE")
    world.slack.fail_threads = {world.boxes["top"]}
    with pytest.raises(Retry):
        consume(world.deps, payload, 1)
    world.slack.fail_threads = set()
    assert consume(world.deps, payload, 2) == "unaddressed: @top"
    assert world.rung("top") == [private(post["ts"])] and router_lines(world, CHAT, post["ts"]) == [NO_MODEL]


def test_dead_lettering_when_slack_itself_is_down_still_parks_the_message(world):
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    world.slack.fail = {"chat.postMessage"}
    world.deps.cache.fail = True                                           # its count is lost; the message isn't
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
    assert {line["event"] for line in logs} >= {"deliver", "done", "unaddressed"}
    assert not any(w in raw for t in texts for w in t.split() if len(w) > 3 and not w.startswith("@"))


# --- newcomers: `team_join` -------------------------------------------------------------------------------------------------

GRACE, CURSOR_BOT = "U0GRACE", "U0CURSOR"


def expect(world, name: str, email: str) -> None:
    """`research msg chat expect NAME --email E`, as published."""
    (world.people / f"{name}.json").write_text(json.dumps({
        "schema": "verity/slack-person/v1", "name": name.capitalize(), "slack_user": None, "email": email,
        "note": "a newcomer", "status": "expected", "channel": {"key": f"chat-{name}", "id": None},
        "expected": "2027-01-14T17:00Z", "opened": None, "closed": None}))


def join(uid: str) -> dict:
    return {"kind": "event", "event_id": f"EvJ{uid}", "source": "events",
            "event": {"type": "team_join", "user": {"id": uid, "name": "x", "is_bot": False}}}


def users(world) -> None:
    world.slack.users = [
        {"id": CURSOR_BOT, "name": "cursor", "real_name": "Cursor", "is_bot": True, "profile": {}},
        {"id": GRACE, "name": "grace", "real_name": "Grace Hopper", "is_bot": False,
         "profile": {"email": "Grace@Example.org", "real_name": "Grace Hopper"}},
        {"id": "U0OTHER", "name": "other", "real_name": "Some One", "is_bot": False, "profile": {"email": "other@example.org"}}]


def test_an_expected_persons_team_join_opens_their_chat_and_their_posts_route_there(world):
    users(world)
    expect(world, "grace", "grace@example.org")
    assert consume(world.deps, join(GRACE), 1) == "team_join: opened #chat-grace"
    [c] = [c for c in world.slack.channels.values() if c["name"] == "chat-grace"]
    assert c["is_private"] and c["members"] == [BOT_USER, CURSOR_BOT, GRACE]
    [welcome] = world.slack.messages[c["id"]]
    assert world.slack.pins == [(c["id"], welcome["ts"])] and welcome["username"] == "top"
    assert world.slack.bookmarks == [{"channel_id": c["id"], "title": "Start here", "type": "link", "link": START_HERE}]
    assert world.pushed == ["people.d/grace.json"]
    assert json.loads((world.people / "grace.json").read_text()) == {
        "schema": "verity/slack-person/v1", "name": "Grace Hopper", "slack_user": GRACE, "email": "Grace@Example.org",
        "note": "a newcomer", "channel": {"key": "chat-grace", "id": c["id"]}, "opened": "2027-01-15T08:00Z", "closed": None}
    post = world.slack.add(c["id"], text="where do I start?", user=GRACE)
    assert consume(world.deps, world.event(c["id"], post), 1) == "unaddressed: @top"
    assert world.rung("top") == [private(post["ts"], "chat-grace", "grace")]
    assert consume(world.deps, join(GRACE), 2) == "team_join: not expected"           # a redelivery opens nothing twice
    assert len(world.slack.pins) == 1 and world.pushed == ["people.d/grace.json"]


@pytest.mark.parametrize("uid", ["U0OTHER", "U0NOBODY", CURSOR_BOT])
def test_an_unexpected_join_posts_nothing_and_opens_nothing(world, uid):
    users(world)
    expect(world, "grace", "grace@example.org")
    before = {k: list(v) for k, v in world.slack.messages.items()}
    decision = None
    try:
        decision = consume(world.deps, join(uid), 1)
    except Retry:
        assert uid == "U0NOBODY"                                           # users.info refuses: tried again, then dead
    assert decision in (None, "team_join: not expected", "team_join: not a person with an email")
    assert [m for m, _ in world.slack.calls if m not in ("users.info",)] == []
    assert world.slack.messages == before and world.pushed == [] and world.slack.pins == []
    assert json.loads((world.people / "grace.json").read_text())["status"] == "expected"
