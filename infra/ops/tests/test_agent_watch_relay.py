"""Infra's relay posts each line the agent watch adds to events.jsonl on the control pod, once, from the offset it last posted;
what the watch leaves for a person leads with @top and rings @top's inbox."""
import importlib.util
import json
from pathlib import Path

import pytest

from verity_ops import agent_watch as W

RELAY = Path(__file__).resolve().parents[2] / "control" / "agent_watch_relay.py"
TOP_INBOX = {"name": "top", "holder": "bc-af075d78-d5b9-590b-acbb-eb47d6f88f39", "inbox_ts": "1760000000.000100", "closed": None}


PROOFS_INBOX = {"name": "proofs", "holder": "bc-11111111-1111-1111-1111-111111111111", "inbox_ts": "1760000000.000200",
                "closed": None}
INFRA_INBOX = {"name": "infra", "holder": "bc-44444444-4444-4444-4444-444444444444", "inbox_ts": "1760000000.000300",
               "closed": None}
KERNELS_INBOX = {"name": "kernels", "holder": "bc-55555555-5555-5555-5555-555555555555", "inbox_ts": "1760000000.000400",
                 "closed": None}


class Routes:
    route = dict(TOP_INBOX)

    def resolve(self, name):
        return name, {"top": self.route, "proofs": PROOFS_INBOX, "infra": INFRA_INBOX, "kernels": KERNELS_INBOX}.get(name)


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


def test_a_stopped_worker_leads_with_its_lanes_handle_and_rings_that_inbox(relay):
    """Only the coordinator that launched a worker can resume it (Task): the watch can't start a worker's run."""
    proofs = relay.slack.tag(relay.slack.entry(relay.REG, "proofs"))
    w = "bc-22222222-2222-2222-2222-222222222222"
    relay.log_text["text"] = "\n".join(json.dumps(r) for r in [
        {"kind": "worker_idle", "agent": w, "name": "ZKS10", "run": "", "lane": "proofs", "handle": "proofs",
         "at": "2026-10-18T20:06:00Z", "since": "2026-10-18T20:06:00Z"},
        {"kind": "worker_long", "agent": w, "name": "W3", "run": "", "lane": "proofs", "handle": "proofs",
         "since": "2026-10-18T19:31:00Z", "hours": 6}]) + "\n"
    relay.once()
    assert relay.posts == [
        f"{proofs} *Agent watch:* worker ZKS10 (<https://cursor.com/agents/{w}|{w}>) of lane proofs went idle at "
        "2026-10-18T20:06:00Z; if it didn't report back, resume it with Task.",
        f"{proofs} *Agent watch:* worker W3 (<https://cursor.com/agents/{w}|{w}>) of lane proofs has been ACTIVE since "
        "2026-10-18T19:31:00Z (6 h); if it isn't working, stop it and resume it with Task."]
    ring = rings(relay)
    assert [r["thread_ts"] for r in ring] == [PROOFS_INBOX["inbox_ts"]] * 2
    assert relay.slack.DOORBELL_RE.match(ring[0]["text"])[1] == "infra" and "ZKS10" in ring[0]["text"]


def test_a_worker_whose_lane_no_handle_owns_rings_infra_and_says_so_never_posted_unrung(relay):
    """#agent-alerts 1792438328.270299 (19 Oct): G2-1, learned from a Task call of proofs' generation 3 after generation 4 took
    the inbox, was posted with lane '' and no handle, and rang nobody."""
    infra, kernels = (relay.slack.tag(relay.slack.entry(relay.REG, h)) for h in ("infra", "kernels"))
    g2, h1, p = ("bc-a8b63bc4-8d5b-53be-aaf8-35339162d81a", "bc-33333333-3333-3333-3333-333333333333",
                 "bc-8f4b652b-17bc-5055-a120-b56940b7c7a9")
    relay.log_text["text"] = "\n".join(json.dumps(r) for r in [
        {"kind": "worker_idle", "agent": g2, "name": "G2-1 → LiveEnd for FinExe", "run": "", "lane": "", "handle": None,
         "parent": p, "at": "2026-10-19T19:22:00Z", "wake": "I'll check back in 30 min.", "wake_due": "2026-10-19T19:52:00Z"},
        {"kind": "worker_idle", "agent": h1, "name": "H1", "run": "", "lane": "hash", "at": "2026-10-18T20:06:00Z"},
        {"kind": "workers_failed", "agent": p, "name": "", "run": "", "lane": "", "handle": None, "minutes": 4,
         "members": [{"kind": "worker_idle", "agent": g2, "name": "G2-1", "at": "2026-10-19T19:22:00Z"},
                     {"kind": "worker_idle", "agent": h1, "name": "G2-2", "at": "2026-10-19T19:25:00Z"}]}]) + "\n"
    relay.once()
    assert relay.posts[0] == (
        f"{infra} *Agent watch:* worker G2-1 → LiveEnd for FinExe (<https://cursor.com/agents/{g2}|{g2}>) went idle at "
        "2026-10-19T19:22:00Z; if it didn't report back, resume it with Task. Before it stopped, it set its own wake-up "
        "(\u201cI'll check back in 30 min.\u201d, due about 19:52Z). Its lane is unknown (no handle owns it or its parent), "
        "so @infra is rung to find its coordinator.")
    assert relay.posts[1] == (f"{kernels} *Agent watch:* worker H1 (<https://cursor.com/agents/{h1}|{h1}>) of lane hash went "
                              "idle at 2026-10-18T20:06:00Z; if it didn't report back, resume it with Task."), \
        "a lane's old name is its renamed handle"
    assert relay.posts[2].startswith(f"{infra} *Agent watch:* 2 workers of the coordinator") and \
        relay.posts[2].endswith("Its lane is unknown (no handle owns it or its parent), so @infra is rung to find its coordinator.")
    assert [r["thread_ts"] for r in rings(relay)] == [INFRA_INBOX["inbox_ts"], KERNELS_INBOX["inbox_ts"], INFRA_INBOX["inbox_ts"]]


def test_a_wake_cursor_refuses_with_409_posts_nothing(relay):
    """Cursor answers 409 for an agent waiting on background work, every hour the watch tries; the post would say nothing new."""
    relay.log_text["text"] = "\n".join([
        line("run-w", "ready_to_wake", handle="vllm", reason="nothing has run for 40 min", action="refused: HTTP 409 (busy)"),
        line("run-v", "ready_to_wake", handle="vllm", reason="nothing has run for 40 min", action="started run-z")]) + "\n"
    relay.once()
    assert len(relay.posts) == 1 and "run-v" in relay.posts[0] and relay.OFFSET.read_text() == "2"


def test_a_silent_agent_and_an_ended_run_lead_with_the_records_handle_and_ring_it(relay):
    proofs = relay.slack.tag(relay.slack.entry(relay.REG, "proofs"))
    w, co = "bc-22222222-2222-2222-2222-222222222222", PROOFS_INBOX["holder"]
    relay.log_text["text"] = "\n".join(json.dumps(r) for r in [
        {"kind": "silent", "agent": w, "name": "ZKS10", "run": "", "handle": "proofs", "lane": "proofs", "minutes": 90,
         "active_minutes": 95, "window": 1, "since": "2026-10-19T03:00:00Z"},
        {"kind": "silent", "agent": w, "name": "ZKS10", "run": "", "handle": "proofs", "lane": "proofs", "minutes": 180,
         "active_minutes": 185, "window": 2, "since": "2026-10-19T03:00:00Z"},
        {"kind": "run_ended", "agent": w, "name": "ZKS10", "run": "r20261019-040000-ab12", "handle": "proofs", "role": "worker",
         "node": "vy-nebius-1", "what": "sweep-a", "rc": 0, "state": "done", "after": "1 h 06 min"},
        {"kind": "run_ended", "agent": co, "name": "verity: proofs", "run": "r20261019-040000-cd34", "handle": "proofs",
         "role": "coordinator", "node": "vy-nebius-2", "what": "lean build", "rc": 1, "state": "failed", "after": "12 min"},
        {"kind": "run_ended", "agent": "", "name": "", "run": "r20261019-040000-ef56", "handle": "proofs", "role": "lane",
         "node": "vy-nebius-2", "what": "slot --by proofs --", "rc": 0, "state": "done", "after": "30 s"}]) + "\n"
    relay.once()
    link = f"<https://cursor.com/agents/{w}|{w}>"
    assert relay.posts == [
        f"{proofs} *Agent watch:* agent ZKS10 ({link}) has been ACTIVE 95 min with no note and no push for 90 min; check on it.",
        f"{proofs} *Agent watch:* agent ZKS10 ({link}) has been ACTIVE 185 min with no note and no push for 180 min; check on "
        "it. Second window: the last ring until it shows activity.",
        f"{proofs} *Agent watch:* run r20261019-040000-ab12 (sweep-a) of worker ZKS10 ({link}) ended rc=0 after 1 h 06 min on "
        "vy-nebius-1; resume ZKS10 with Task.",
        f"{proofs} *Agent watch:* your run r20261019-040000-cd34 (lean build) (<https://cursor.com/agents/{co}|{co}>) ended rc=1 "
        "after 12 min on vy-nebius-2; resume.",
        f"{proofs} *Agent watch:* run r20261019-040000-ef56 (slot --by proofs --) of lane proofs ended rc=0 after 30 s on "
        "vy-nebius-2."]
    assert [r["thread_ts"] for r in rings(relay)] == [PROOFS_INBOX["inbox_ts"]] * 5


def test_a_failure_the_watch_acted_on_leads_with_its_owner_and_rings_that_inbox(relay):
    """Daniel, 19 Oct: ring the owner at once. A coordinator's owner is top; a worker's, its launcher's handle."""
    top, proofs = (relay.slack.tag(relay.slack.entry(relay.REG, h)) for h in ("top", "proofs"))
    w, co = "bc-22222222-2222-2222-2222-222222222222", PROOFS_INBOX["holder"]
    relay.log_text["text"] = "\n".join(json.dumps(r) for r in [
        {"kind": "stalled", "agent": co, "name": "verity: proofs", "run": "run-s", "since": "2026-10-19T10:00:00Z", "minutes": 30,
         "action": "cancelled and continued", "new_run": "run-n", "owner": "top"},
        {"kind": "idle_after_run", "agent": co, "name": "verity: proofs", "run": "run-b", "role": "coordinator", "handle": "proofs",
         "ended_run": "r1", "result": "run r1 (lean) on vy-nebius-2: done rc=0 after 12 min at 2026-10-19T10:30:00Z",
         "minutes": 10, "action": "cancelled its background wait (run-b) and resumed it with the run's result", "owner": "top"},
        {"kind": "idle_after_run", "agent": w, "name": "ZKS10", "run": "", "role": "worker", "lane": "zk", "handle": "proofs",
         "ended_run": "r2", "result": "run r2 (sweep) on vy-nebius-1: failed rc=3 after 1 h 00 min at 2026-10-19T10:30:00Z",
         "minutes": 10, "action": "rang @proofs to resume it with Task", "owner": "proofs"},
        {"kind": "relaunch", "agent": w, "name": "ZKS10", "run": "", "role": "worker", "lane": "zk", "handle": "proofs",
         "reason": "idle since 10:20Z", "action": "relaunched it fresh as bc-7", "new_agent": "bc-7", "owner": "proofs"},
        {"kind": "held", "agent": "bc-x", "name": "x", "run": "run-h", "since": "2026-10-19T10:00:00Z", "minutes": 15,
         "action": "cancelled and continued"}]) + "\n"
    relay.once()
    link = f"<https://cursor.com/agents/{co}|{co}>"
    assert relay.posts[:2] == [
        f"{top} *Agent watch:* verity: proofs ({link}, run run-s) streamed nothing for 30 min since 2026-10-19T10:00:00Z "
        "while RUNNING (a hung call or a suspended VM); the watch cancelled and continued. New run run-n.",
        f"{top} *Agent watch:* @proofs's coordinator ({link}, run run-b) sat idle after its run r1 ended (run r1 (lean) on "
        "vy-nebius-2: done rc=0 after 12 min at 2026-10-19T10:30:00Z), with nothing resuming it for 10 min; the watch cancelled "
        "its background wait (run-b) and resumed it with the run's result."]
    assert relay.posts[2].startswith(f"{proofs} *Agent watch:* worker ZKS10 (") and "of lane zk sat idle after its run r2" in relay.posts[2]
    assert relay.posts[3].endswith("needs a fresh launch (idle since 10:20Z); the watch relaunched it fresh as bc-7. New agent bc-7.")
    assert relay.posts[4].startswith("*Agent watch:*"), "a failure with no owner is posted in the channel only"
    assert [r["thread_ts"] for r in rings(relay)] == [TOP_INBOX["inbox_ts"]] * 2 + [PROOFS_INBOX["inbox_ts"]] * 2


def test_a_renamed_handle_is_rung_by_its_new_name(relay):
    """hash became kernels on 18 Oct; lanes' notes and workers.d still carry hash."""
    assert relay.RENAMED.get("hash") == "kernels"
    relay.log_text["text"] = json.dumps({"kind": "worker_idle", "agent": "bc-x", "name": "H1", "run": "", "lane": "hash",
                                         "handle": "hash", "at": "2026-10-18T20:06:00Z"}) + "\n"
    relay.once()
    assert relay.posts[0].startswith(relay.slack.tag(relay.slack.entry(relay.REG, "kernels")))


def test_workers_that_stopped_together_are_one_post_to_the_lane_naming_each_and_its_wake_up(relay):
    """INF-126: 14 of proofs' workers stopped within 14 min. One post leads with @proofs and rings its inbox once, naming each
    worker and the wake-ups two said they set; a lone worker's record carries its wake-up too."""
    proofs, co = relay.slack.tag(relay.slack.entry(relay.REG, "proofs")), PROOFS_INBOX["holder"]
    names = ["REC1", "G2-1", "REC2", "G2-7", "ZKS10", "M18b", "G2-4", "INT7", "KSF1", "FIX2595", "G2-3", "G2-2", "G2-6", "G2-8"]
    members = [{"kind": "worker_idle", "agent": f"bc-{i + 1:08x}-0000-0000-0000-{i + 1:012x}", "name": n, "run": "",
                "at": f"2026-10-19T{16 + (49 + i) // 60}:{(49 + i) % 60:02d}:00Z", "lane": "proofs", "handle": "proofs"}
               for i, n in enumerate(names)]
    members[7].update(wake="Setting my wake-up timer before ending the turn.")
    members[1].update(wake="A one-off wake-up fires in 15 minutes.", wake_due="2026-10-19T17:05:46Z")
    relay.log_text["text"] = "\n".join(json.dumps(r) for r in [
        {"kind": "workers_failed", "agent": co, "name": "verity: proofs", "run": "", "role": "workers", "lane": "proofs",
         "handle": "proofs", "owner": "proofs", "count": 14, "minutes": 14, "members": members},
        {**members[1], "t": "2026-10-19T17:10:00Z"}]) + "\n"
    relay.once()
    burst, alone = relay.posts
    assert burst.startswith(f"{proofs} *Agent watch:* 14 workers of @proofs's coordinator verity: proofs "
                            f"(<https://cursor.com/agents/{co}|{co}>) stopped within 14 min: REC1 (")
    assert all(f"{n} (<https://cursor.com/agents/" in burst for n in names) and burst.endswith("with Task.")
    assert "INT7 (<https://cursor.com/agents/bc-00000008-0000-0000-0000-000000000008|bc-00000008>) idle at 16:56Z; it set its own " \
           "wake-up (\u201cSetting my wake-up timer before ending the turn.\u201d)" in burst
    assert "it set its own wake-up (\u201cA one-off wake-up fires in 15 minutes.\u201d, due about 17:05Z)" in burst
    assert alone.endswith("Before it stopped, it set its own wake-up (\u201cA one-off wake-up fires in 15 minutes.\u201d, due "
                          "about 17:05Z).")
    assert [r["thread_ts"] for r in rings(relay)] == [PROOFS_INBOX["inbox_ts"]] * 2, "the burst rings its lane once"
