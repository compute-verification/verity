"""The watchdog and reconciliation, against Slack as they find it."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from conftest import BOT, CHAT, DANIEL, LUKE, TEAM
from slack_events import crons
from slack_events.consumer import NO_MODEL, consume, post_line
from slack_events.crons import DAILY_HEAD, ESCALATE_S, ESCALATED, ESCALATION, REMIND_S, REMINDER, SETTLE_S
from slack_events.deps import CANARY_DONE, CANARY_SENT, DEAD, LAST_DONE, LAST_ENQUEUED, LAST_SUCCESS, OLDEST, STALL_RUNG, TOPIC


def private(ts: str, reminder: str | None = None) -> str:
    return (f"from luke, #chat-luke thread {ts}: " + (f"reminder ({reminder}): " if reminder else "")
            + f"(a post in a private channel: research msg read --channel chat-luke --thread {ts})")


def routed(world, text: str, **kw) -> dict:
    post = world.slack.add(CHAT, text=text, user=LUKE, **kw)
    consume(world.deps, world.event(CHAT, post), 1)
    return post


def test_a_chat_post_with_no_eyes_and_no_lane_reply_rings_again_after_15_min_once(world):
    post = routed(world, "@circuits is the TP8 sweep done?")
    world.clock.t += REMIND_S - 1
    assert crons.watchdog(world.deps)["reminded"] == 0
    world.clock.t += 2
    assert crons.watchdog(world.deps)["reminded"] == 1
    world.clock.t += 300
    crons.watchdog(world.deps)
    assert world.rung("circuits") == [private(post["ts"]), private(post["ts"], REMINDER)] and world.rung("top") == []
    assert world.deps.cache.get(LAST_SUCCESS) == world.clock()


def test_eyes_or_a_lane_reply_stops_the_reminder(world):
    eyed = routed(world, "@circuits is the TP8 sweep done?")
    eyed["reactions"] = [{"name": "eyes", "users": ["U0CURSOR"], "count": 1}]
    answered = routed(world, "@circuits and the Lean side?")
    world.slack.add(CHAT, text="it is", username="circuits", bot_id=BOT, thread_ts=answered["ts"])
    world.clock.t += REMIND_S + 60
    assert crons.watchdog(world.deps) == {"open": 1, "reminded": 0, "escalated": 0, "canary": True, "stalled": False}
    assert world.rung("circuits") == [private(eyed["ts"]), private(answered["ts"])]


def test_after_60_min_with_no_lane_reply_it_rings_top_and_says_so_in_the_thread_once(world):
    post = routed(world, "@circuits is the TP8 sweep done?")
    post["reactions"] = [{"name": "eyes", "users": ["U0CURSOR"], "count": 1}]
    world.clock.t += ESCALATE_S + 1
    assert crons.watchdog(world.deps)["escalated"] == 1
    world.clock.t += 300
    assert crons.watchdog(world.deps)["escalated"] == 0
    assert world.rung("top") == [private(post["ts"], ESCALATION)]
    assert [m["text"] for m in world.slack.thread(CHAT, post["ts"]) if m.get("username") == "router"] == [ESCALATED]


def test_an_escalation_that_did_not_ring_posts_no_line_and_the_next_run_tries_again(world):
    post = routed(world, "@circuits is the TP8 sweep done?")
    world.clock.t += ESCALATE_S + 1
    world.slack.fail = {"chat.postMessage"}
    assert crons.watchdog(world.deps)["escalated"] == 0
    world.slack.fail = set()
    world.clock.t += 300
    assert crons.watchdog(world.deps)["escalated"] == 1 and world.rung("top") == [private(post["ts"], ESCALATION)]


def test_an_unaddressed_post_already_at_top_is_reminded_to_top(world):
    post = routed(world, "how do I read a PoUW receipt?")
    world.clock.t += REMIND_S + 1
    crons.watchdog(world.deps)
    assert world.rung("top") == [private(post["ts"]), private(post["ts"], REMINDER)]


def test_once_a_day_top_gets_the_list_of_open_threads(world):
    day = datetime.fromtimestamp(world.clock(), timezone.utc).replace(hour=crons.DAILY_AT[0], minute=0, second=0)
    world.clock.t = day.timestamp() - 2 * 3600
    post = routed(world, "@circuits is the TP8 sweep done?")
    post["reactions"] = [{"name": "eyes", "users": ["U0CURSOR"], "count": 1}]
    world.clock.t = day.timestamp() + 60
    assert crons.watchdog(world.deps)["daily"] == 1
    world.clock.t += 120
    assert crons.watchdog(world.deps)["daily"] == 0
    world.clock.t += crons.EVERY_S
    assert "daily" not in crons.watchdog(world.deps)
    daily = [t for t in world.rung("top") if t.startswith(DAILY_HEAD)]
    assert daily == [f"{DAILY_HEAD} #chat-luke thread {post['ts']} (2.0 h)"]


def test_reconciliation_enqueues_a_missed_post_once_and_the_consumer_routes_it(world):
    missed = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    asked = world.slack.add(TEAM, text="what is the TP8 status?", user=DANIEL)
    world.clock.t += SETTLE_S + 1
    young = world.slack.add(CHAT, text="still there?", user=LUKE)
    assert crons.reconcile(world.deps)["queued"] == 2
    assert world.deps.cache.get(OLDEST) == {"ts": missed["ts"], "at": world.clock()}
    assert crons.reconcile(world.deps)["queued"] == 2                      # sent again, and the queue keeps the first
    sent = world.deps.queue.on(TOPIC)
    assert [p["event_id"] for p in sent] == [f"reconcile:{CHAT}:{missed['ts']}", f"reconcile:{TEAM}:{asked['ts']}"]
    assert young["ts"] not in {p["event"]["ts"] for p in sent}
    for p in sent:
        consume(world.deps, p, 1)
    assert [m["text"] for m in world.lines(CHAT, missed["ts"])] == [NO_MODEL]
    assert len(world.rung("top")) == 2
    world.clock.t += SETTLE_S + 1
    before = len(world.deps.queue.sent)
    assert crons.reconcile(world.deps)["queued"] == 1                      # only the young post, now settled
    assert len(world.deps.queue.sent) == before + 1 and world.deps.cache.get(OLDEST)["ts"] == young["ts"]


def test_reconciliation_takes_a_routing_line_or_a_doorbell_as_the_record_and_skips_a_post_that_rings_no_one(world):
    """Slack is the record: an unaddressed post's line, an addressed post's doorbell.  A quiet post naming no one, or one
    naming only a lane with no inbox, leaves neither, and isn't enqueued for it."""
    routed(world, "how do I read a PoUW receipt?")
    routed(world, "@circuits is the TP8 sweep done?")
    lined = world.slack.add(CHAT, text="anyone?", user=LUKE)
    post_line(world.deps, CHAT, lined["ts"], lined["ts"], NO_MODEL, ["top"])
    world.slack.add(TEAM, text="thanks, no reply needed (quiet)", user=DANIEL)
    world.slack.add(TEAM, text="@proofs is the bound tight?", user=DANIEL)          # a lane with no inbox
    world.clock.t += SETTLE_S + 1
    assert crons.reconcile(world.deps)["queued"] == 0
    assert world.deps.cache.get(OLDEST) == {"ts": None, "at": world.clock()}
    assert world.deps.cache.get(LAST_SUCCESS) == world.clock()


def test_a_router_line_without_its_metadata_is_not_a_record(world):
    other = world.slack.add(CHAT, text="anyone?", user=LUKE)
    world.slack.add(CHAT, text=NO_MODEL, username="router", bot_id=BOT, thread_ts=other["ts"])
    world.clock.t += SETTLE_S + 1
    assert crons.reconcile(world.deps)["queued"] == 1


def test_reconciliation_starts_the_dead_letter_count_where_the_cache_has_none(world):
    crons.reconcile(world.deps)
    assert world.deps.cache.get(DEAD) == {"count": 0, "since": world.clock()}
    world.deps.cache.set(DEAD, {"count": 3, "since": 1.0})
    crons.reconcile(world.deps)
    assert world.deps.cache.get(DEAD) == {"count": 3, "since": 1.0}


def test_reconciliation_outside_its_hour_leaves_old_posts(world):
    world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    world.clock.t += crons.RECONCILE_S + 1
    assert crons.reconcile(world.deps)["queued"] == 0


def test_a_queue_with_events_waiting_and_nothing_done_rings_comms_hourly_and_a_working_one_never(world, tmp_path):
    """Health looks clean while nothing is consumed (its last_success_at counts the crons), so the watchdog compares the
    app's last publish with the consumer's last done."""
    import json
    from conftest import COORD, add_route
    box = world.slack.add(COORD, text=f"{world.deps.rs.INBOX_HEAD}comms. pointers", username="comms", bot_id=BOT)
    add_route(Path(world.deps.rs.notes_root()) / world.deps.rs.ROUTES_DIR, "comms", box["ts"])
    stalls = lambda: [m["text"] for m in world.slack.thread(COORD, box["ts"]) if m["ts"] != box["ts"]]
    now = world.clock()
    world.deps.write(CANARY_SENT, now)                                           # this hour's canary went out already
    world.deps.write(LAST_ENQUEUED, now - crons.STALL_S + 60)                    # nine minutes waiting: not yet
    assert crons.watchdog(world.deps)["stalled"] is False and stalls() == []
    world.clock.t += 60
    assert crons.watchdog(world.deps)["stalled"] is True
    [line] = stalls()
    assert line.startswith(crons.STALL_HEAD) and "consumer's last message done at unknown" in line
    world.clock.t += 300                                                         # rung within the hour: once
    assert crons.watchdog(world.deps)["stalled"] is False and len(stalls()) == 1
    world.deps.write(LAST_DONE, world.clock())                                   # the consumer caught up
    world.clock.t += crons.STALL_EVERY_S
    assert crons.watchdog(world.deps)["stalled"] is False and len(stalls()) == 1
    world.deps.cache.clear()                                                     # evicted: unknown rings nothing
    assert crons.watchdog(world.deps)["stalled"] is False


def test_the_consumer_stamps_its_own_done_and_the_app_its_publish(world):
    post = world.slack.add(CHAT, text="@circuits is the sweep done?", user=LUKE)
    assert consume(world.deps, world.event(CHAT, post), 1) == "rule"
    assert world.deps.cache.get(LAST_DONE) == world.clock()


def test_the_hourly_canary_proves_the_consumer_on_a_quiet_day_and_its_loss_rings_comms(world):
    """No person posted in the people's channels for the drain's first two days, so nothing proved it; the canary is a
    synthetic event only the consumer can stamp done."""
    from conftest import COORD, add_route
    box = world.slack.add(COORD, text=f"{world.deps.rs.INBOX_HEAD}comms. pointers", username="comms", bot_id=BOT)
    add_route(Path(world.deps.rs.notes_root()) / world.deps.rs.ROUTES_DIR, "comms", box["ts"])
    rung = lambda: [m for m in world.slack.thread(COORD, box["ts"]) if m["ts"] != box["ts"]]
    assert crons.watchdog(world.deps)["canary"] is True and crons.watchdog(world.deps)["canary"] is False
    [payload] = [p for p in world.deps.queue.on(TOPIC) if p["kind"] == "canary"]
    assert consume(world.deps, payload, 1) == "canary" and world.deps.cache.get(CANARY_DONE) == world.clock()
    world.clock.t += crons.STALL_S
    assert crons.watchdog(world.deps)["stalled"] is False and rung() == []                 # consumed: nothing to say
    world.clock.t += crons.CANARY_EVERY_S                                                    # the next one, never consumed
    assert crons.watchdog(world.deps)["canary"] is True
    world.clock.t += crons.STALL_S
    assert crons.watchdog(world.deps)["stalled"] is True and len(rung()) == 1
    assert world.rung("top") == []                                                          # a canary rings no lane
