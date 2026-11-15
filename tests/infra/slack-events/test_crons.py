"""The watchdog and reconciliation, against Slack as they find it."""
from __future__ import annotations

import json
import shutil
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pytest

from conftest import BOT, CHAT, DANIEL, LUKE, TEAM
from slack_events import crons
from slack_events.consumer import NO_MODEL, consume, post_line
from slack_events.crons import DAILY_HEAD, ESCALATE_S, ESCALATED, ESCALATION, LANE_QUIET, REMIND_S, REMINDER, SETTLE_S
from slack_events.deps import CANARY_DONE, CANARY_SENT, DEAD, LAST_DONE, LAST_ENQUEUED, LAST_SUCCESS, OLDEST, STALL_RUNG, TOPIC


def private(ts: str, reminder: str | None = None, waited: str = "under a minute") -> str:
    """Luke's doorbell for his post `ts`, rung `waited` after he posted it; the watchdog's `reminder`s are urgent."""
    return (f"from luke (a person, waiting {waited}){' (URGENT)' if reminder else ''}, #chat-luke thread {ts}: "
            + (f"reminder ({reminder}): " if reminder else "")
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
    assert world.rung("circuits") == [private(post["ts"]), private(post["ts"], REMINDER, "15 min")] and world.rung("top") == []
    assert world.deps.cache.get(LAST_SUCCESS) == world.clock()


def test_a_reminder_rings_once_though_slack_reads_its_emoji_back_as_shortcodes(world, monkeypatch):
    """Slack may return a post's emoji as `:shortcode:`s, so a reason with an emoji in it would never match its own
    doorbell, and the reminder would ring every run.  Its reasons carry none."""
    real = world.slack.add
    monkeypatch.setattr(world.slack, "add", lambda ch, **m: real(ch, **{**m, "text": m.get("text", "").replace("👀", ":eyes:")}))
    with_lane(world, "circuits")
    post = routed(world, "how do I read a PoUW receipt?")
    world.clock.t += REMIND_S + 1
    for _ in range(4):
        crons.watchdog(world.deps)
        world.clock.t += crons.EVERY_S
    assert world.rung("circuits") == [private(post["ts"]), private(post["ts"], REMINDER, "15 min")]
    assert world.rung("top") == [private(post["ts"], LANE_QUIET.format(lane="circuits"), "15 min")]
    assert all(r.isascii() for r in (REMINDER, ESCALATION, LANE_QUIET))


def test_eyes_or_a_lane_reply_stops_the_reminder(world):
    eyed = routed(world, "@circuits is the TP8 sweep done?")
    eyed["reactions"] = [{"name": "eyes", "users": ["U0CURSOR"], "count": 1}]
    answered = routed(world, "@circuits and the Lean side?")
    world.slack.add(CHAT, text="it is", username="circuits", bot_id=BOT, thread_ts=answered["ts"])
    world.clock.t += REMIND_S + 60
    assert crons.watchdog(world.deps) == {"open": 1, "reminded": 0, "escalated": 0, "lane_unanswered": 0, "canary": True,
                                          "stalled": False}
    assert world.rung("circuits") == [private(eyed["ts"]), private(answered["ts"])]


def test_after_60_min_with_no_lane_reply_it_rings_top_and_says_so_in_the_thread_once(world):
    post = routed(world, "@circuits is the TP8 sweep done?")
    post["reactions"] = [{"name": "eyes", "users": ["U0CURSOR"], "count": 1}]
    world.clock.t += ESCALATE_S + 1
    assert crons.watchdog(world.deps)["escalated"] == 1
    world.clock.t += 300
    assert crons.watchdog(world.deps)["escalated"] == 0
    assert world.rung("top") == [private(post["ts"], ESCALATION, "60 min")]
    assert [m["text"] for m in world.slack.thread(CHAT, post["ts"]) if m.get("username") == "router"] == [ESCALATED]


def test_an_escalation_that_did_not_ring_posts_no_line_and_the_next_run_tries_again(world):
    post = routed(world, "@circuits is the TP8 sweep done?")
    world.clock.t += ESCALATE_S + 1
    world.slack.fail = {"chat.postMessage"}
    assert crons.watchdog(world.deps)["escalated"] == 0
    world.slack.fail = set()
    world.clock.t += 300
    assert crons.watchdog(world.deps)["escalated"] == 1 and world.rung("top") == [private(post["ts"], ESCALATION, "65 min")]


def with_lane(world, lane: str | None) -> None:
    """Luke's people.d record with `lane`, the lane he usually works with (`chat open luke --lane LANE`)."""
    path = world.people / "luke.json"
    r = {k: v for k, v in json.loads(path.read_text()).items() if k != "lane"}
    path.write_text(json.dumps({**r, **({"lane": lane} if lane else {})}))


def test_a_post_sent_to_a_persons_usual_lane_rings_top_as_well_once_when_that_lane_is_quiet_15_min(world):
    """No model: Luke's unaddressed post goes to the circuits lane, his usual lane.  After 15 min with no 👀 and no lane
    reply the reminder rings that lane, and the admin as well, once, both urgent; after 60 the escalation rings the admin
    again."""
    with_lane(world, "circuits")
    post = routed(world, "how do I read a PoUW receipt?")
    assert [m["text"] for m in world.lines(CHAT, post["ts"])] == ["sent to @circuits: no lane named; luke's usual lane"]
    assert world.rung("circuits") == [private(post["ts"])] and world.rung("top") == []
    world.clock.t += REMIND_S + 1
    assert crons.watchdog(world.deps)["lane_unanswered"] == 1
    quiet = LANE_QUIET.format(lane="circuits")
    assert world.rung("circuits") == [private(post["ts"]), private(post["ts"], REMINDER, "15 min")]
    assert world.rung("top") == [private(post["ts"], quiet, "15 min")]
    for _ in range(3):
        world.clock.t += crons.EVERY_S
        assert crons.watchdog(world.deps)["lane_unanswered"] == 0
    assert len(world.rung("top")) == len(world.rung("circuits")) - 1 == 1
    world.clock.t = float(post["ts"]) + ESCALATE_S + 1
    assert crons.watchdog(world.deps)["escalated"] == 1
    assert world.rung("top") == [private(post["ts"], quiet, "15 min"), private(post["ts"], ESCALATION, "60 min")]


def test_a_usual_lane_that_picks_it_up_or_replies_keeps_top_out(world):
    with_lane(world, "circuits")
    eyed = routed(world, "how do I read a PoUW receipt?")
    eyed["reactions"] = [{"name": "eyes", "users": ["U0CURSOR"], "count": 1}]
    answered = routed(world, "and the Lean side?")
    world.slack.add(CHAT, text="here is how", username="circuits", bot_id=BOT, thread_ts=answered["ts"])
    world.clock.t += REMIND_S + 1
    assert crons.watchdog(world.deps)["lane_unanswered"] == 0 and world.rung("top") == []


@pytest.mark.parametrize("lane", [None, "nobody"])
def test_with_no_usual_lane_or_one_the_registry_lacks_the_post_goes_to_top_as_before(world, lane):
    with_lane(world, lane)
    post = routed(world, "how do I read a PoUW receipt?")
    assert [m["text"] for m in world.lines(CHAT, post["ts"])] == [NO_MODEL]
    world.clock.t += REMIND_S + 1
    assert crons.watchdog(world.deps)["lane_unanswered"] == 0
    assert world.rung("top") == [private(post["ts"]), private(post["ts"], REMINDER, "15 min")]


def test_an_unaddressed_post_already_at_top_is_reminded_to_top(world):
    post = routed(world, "how do I read a PoUW receipt?")
    world.clock.t += REMIND_S + 1
    crons.watchdog(world.deps)
    assert world.rung("top") == [private(post["ts"]), private(post["ts"], REMINDER, "15 min")]


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


def test_reconciliation_enqueues_a_post_whose_lanes_route_could_not_be_read(world, monkeypatch):
    """A lane whose route couldn't be read may have an inbox: its post is one to ring, not one that rings no one."""
    rs = world.deps.rs
    published = Path(urllib.parse.unquote(rs.Routes().base[len("file://"):]))
    shutil.move(rs.notes_root() / rs.ROUTES_DIR / "circuits.json", published / "circuits.json")
    world.slack.add(CHAT, text="@circuits is the TP8 sweep done?", user=LUKE)
    world.clock.t += SETTLE_S + 1
    real = rs.Routes

    def get(url: str) -> bytes | None:
        if url.endswith("/circuits.json"):
            raise rs.SlackError(f"{url}: HTTP 503")
        return rs.http_get(url)
    monkeypatch.setattr(rs, "Routes", lambda *a, **kw: real(*a, get=get, **kw))
    assert crons.reconcile(world.deps)["queued"] == 1
    monkeypatch.setattr(rs, "Routes", real)
    shutil.move(published / "circuits.json", rs.notes_root() / rs.ROUTES_DIR / "circuits.json")
    rs.deliver(world.deps.slack, crons.chats(world.deps)[0], rs.Routes(), CHAT, world.slack.messages[CHAT][-1],
               now=world.clock())
    world.deps.queue.sent.clear()
    assert crons.reconcile(world.deps)["queued"] == 0                     # rung now: its doorbell is the record


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
