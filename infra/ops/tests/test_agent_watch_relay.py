"""Infra's relay posts each line the agent watch adds to events.jsonl on the control pod, once, from the offset it last posted;
what the watch leaves for a person leads with @top and rings @top's inbox."""
import importlib.util
import json
import subprocess
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
    m.real_bells = m.bells
    log, posts, wrote, api = {"text": ""}, [], [], Api()
    monkeypatch.setattr(m, "OFFSET", tmp_path / "offset")
    monkeypatch.setattr(m, "DRY", False)
    monkeypatch.setattr(m, "bells", lambda api, now: {"t": "2026-10-18T22:00:00Z", "inboxes": {}})
    monkeypatch.setattr(m, "pod", lambda cmd, data=None: log["text"] if data is None else wrote.append((cmd, data)) or "")
    monkeypatch.setattr(m.slack, "Slack", lambda token: api)
    monkeypatch.setattr(m.slack, "Routes", Routes)
    monkeypatch.setattr(m.slack, "_post", lambda api, ch, handle, text, *a, **k: posts.append(text) or f"1760000{len(posts)}.0")
    ring_runs, refuse = [], []

    def call(argv):
        ring_runs.append(argv[len(m.RING_RUN):])
        rc = 2 if refuse and "--to" not in argv else 0
        return subprocess.CompletedProcess(argv, rc, "", "research: no handle holds bc-x: pass --to @LANE" if rc else "")
    monkeypatch.setattr(m, "call", call)
    m.log_text, m.posts, m.wrote, m.api, m.ring_runs, m.refuse = log, posts, wrote, api, ring_runs, refuse
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


def test_on_the_pod_its_reads_and_writes_run_in_a_local_shell_and_no_ssh(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("agent_watch_relay", RELAY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setenv("AGENT_WATCH_ON_POD", "1")
    monkeypatch.setattr(m.connect, "ssh_base", lambda host: pytest.fail("the relay on the pod reached for ssh"))
    f = tmp_path / "bells.json"
    m.pod(f"cat > {f}", '{"t": 1}')
    assert f.read_text() == '{"t": 1}' and m.pod(f"cat {f}") == '{"t": 1}'


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


def test_runs_whose_sender_no_handle_owns_are_one_post_that_rings_infra(relay):
    """Before, the watch dropped these to its log ("which no handle owns: not rung", 753 runs of 93 senders on 19 Oct)."""
    infra, aede = relay.slack.tag(relay.slack.entry(relay.REG, "infra")), "bc-aede327f-c2cf-5f24-a40a-309e7bb815f4"
    runs = [{"run": "r20261019-052718-24a9", "node": "vy-nebius-2", "sender": f"https://cursor.com/agents/{aede}",
             "what": "r20261019-052718-24a9", "rc": 8, "state": "failed", "after": "6 s", "at": "2026-10-19T05:28:12Z"},
            {"run": "r20261019-195000-0001", "node": "vy-nebius-1", "sender": "", "what": "python -m bench", "rc": 0,
             "state": "done", "after": "1 min", "at": "2026-10-19T19:50:00Z"}]
    many = [{**runs[1], "run": f"r20261019-2000{i:02d}-0001"} for i in range(relay.UNSENT_LISTED + 2)]
    relay.log_text["text"] = "\n".join(json.dumps({"kind": "run_ended", "agent": "", "name": "", "run": "2 runs", "handle": None,
                                                   "role": "unknown", "count": len(r), "runs": r}) for r in (runs, many)) + "\n"
    relay.once()
    assert relay.posts[0] == (
        f"{infra} *Agent watch:* 2 research runs ended whose sender no handle owns (not a coordinator, a worker the watch knows, "
        f"its parent, or a lane): r20261019-052718-24a9 on vy-nebius-2 (r20261019-052718-24a9, failed rc=8) sent by "
        f"<https://cursor.com/agents/{aede}|bc-aede327f>; r20261019-195000-0001 on vy-nebius-1 (python -m bench, done rc=0) sent "
        "by no `submitted_by` or `--by`. @infra is rung to find who sent them.")
    assert relay.posts[1].count(" sent by ") == relay.UNSENT_LISTED and "; and 2 more. @infra is rung" in relay.posts[1]
    assert [r["thread_ts"] for r in rings(relay)] == [INFRA_INBOX["inbox_ts"]] * 2


def test_a_wake_cursor_refuses_with_409_posts_nothing(relay):
    """Cursor answers 409 for an agent waiting on background work, every hour the watch tries; the post would say nothing new."""
    relay.log_text["text"] = "\n".join([
        line("run-w", "ready_to_wake", handle="vllm", reason="nothing has run for 40 min", action="refused: HTTP 409 (busy)"),
        line("run-v", "ready_to_wake", handle="vllm", reason="nothing has run for 40 min", action="started run-z")]) + "\n"
    relay.once()
    assert len(relay.posts) == 1 and "run-v" in relay.posts[0] and relay.OFFSET.read_text() == "2"


def test_a_silent_coordinator_rings_top(relay):
    """TOP-90 row 37: memory-accounting's coordinator hung 01:47-04:39Z, and only its own handle was rung."""
    co = PROOFS_INBOX["holder"]
    rec = {"kind": "silent", "agent": co, "name": "verity: proofs", "run": "run-r", "handle": "proofs", "lane": "proofs",
           "minutes": 60, "active_minutes": 300, "window": 1, "since": "2026-10-26T02:00:00Z", "owner": "top"}
    assert relay.rung(rec) == "top" and relay.rung({**rec, "owner": None}) == "proofs"


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
    assert [r["thread_ts"] for r in rings(relay)] == [PROOFS_INBOX["inbox_ts"]] * 2, "the silent posts; run ends ring-run"
    permalink = relay.slack.permalink(relay.REG, relay.ALERTS, "17600003.0")
    assert relay.ring_runs == [
        ["--run", "r20261019-040000-cd34", "--rc", "1", "--text", "lean build failed after 12 min on vy-nebius-2: "
         + relay.slack.permalink(relay.REG, relay.ALERTS, "17600004.0"), "--agent", co],
        ["--run", "r20261019-040000-ef56", "--rc", "0", "--text", "slot --by proofs -- done after 30 s on vy-nebius-2: "
         + relay.slack.permalink(relay.REG, relay.ALERTS, "17600005.0"), "--to", "@proofs"]], \
        "a lane's run names the handle; a coordinator's own is found by its route; a worker's rc 0 run (ab12) rings no one"


def test_a_run_whose_sender_is_in_a_turn_is_posted_and_rings_no_one(relay):
    """INF-277: the leads' survey (27 Oct 02:00Z) put run-end rings at most of their inboxes. A sender ACTIVE at its run's end
    sees the end itself; one that then idles is resumed by the watch (`idle_after_run`), so nobody is rung."""
    w, co = "bc-22222222-2222-2222-2222-222222222222", PROOFS_INBOX["holder"]
    relay.log_text["text"] = "\n".join(json.dumps(r) for r in [
        {"kind": "run_ended", "agent": w, "name": "ZKS10", "run": "r20261027-020000-ab12", "handle": "proofs", "role": "worker",
         "node": "vy-nebius-1", "what": "sweep-a", "rc": 1, "state": "failed", "after": "3 min", "sender_status": "ACTIVE"},
        {"kind": "run_ended", "agent": co, "name": "verity: proofs", "run": "r20261027-020000-cd34", "handle": "proofs",
         "role": "coordinator", "node": "vy-nebius-2", "what": "lean build", "rc": 1, "state": "failed", "after": "12 min",
         "sender_status": "ACTIVE"},
        {"kind": "run_ended", "agent": co, "name": "verity: proofs", "run": "r20261027-020000-ef56", "handle": "proofs",
         "role": "coordinator", "node": "vy-nebius-2", "what": "lean build", "rc": 1, "state": "failed", "after": "12 min",
         "sender_status": "IDLE"}]) + "\n"
    relay.once()
    link = f"<https://cursor.com/agents/{w}|{w}>"
    assert relay.posts[:2] == [
        f"*Agent watch:* run r20261027-020000-ab12 (sweep-a) of worker ZKS10 ({link}) ended rc=1 after 3 min on vy-nebius-1; its "
        "sender is in a turn, so no one is rung.",
        f"*Agent watch:* run r20261027-020000-cd34 (lean build) of verity: proofs (<https://cursor.com/agents/{co}|{co}>) ended "
        "rc=1 after 12 min on vy-nebius-2; its sender is in a turn, so no one is rung."]
    assert [a[a.index("--run") + 1] for a in relay.ring_runs] == ["r20261027-020000-ef56"], "an idle sender is still rung"


def test_a_former_coordinators_run_refused_by_ring_run_is_rung_again_with_the_watchs_handle(relay):
    """ring-run finds a coordinator by its open route; one a handover replaced holds none, so it refuses without --to."""
    relay.refuse.append(True)
    co = "bc-21f3d3de-28db-5e5d-964b-d357187449aa"
    relay.log_text["text"] = json.dumps({"kind": "run_ended", "agent": co, "name": "", "run": "r20261019-043737-f6de",
                                         "handle": "proofs", "role": "coordinator", "node": "vy-nebius-2", "what": "census",
                                         "rc": 130, "state": "failed", "after": "51 min"}) + "\n"
    relay.once()
    assert [("--to" in a, a[a.index("--agent") + 1]) for a in relay.ring_runs] == [(False, co), (True, co)]
    assert relay.ring_runs[1][-2:] == ["--to", "@proofs"] and len(relay.posts) == 1


def test_a_ring_run_that_fails_leaves_the_alert_posted_once(relay):
    relay.refuse.append(True)
    relay.log_text["text"] = json.dumps({"kind": "run_ended", "agent": "", "name": "", "run": "r20261019-040000-ef56",
                                         "handle": "proofs", "role": "lane", "node": "vy-nebius-2", "what": "x", "rc": 0,
                                         "state": "done", "after": "30 s"}) + "\n"
    relay.call = lambda argv: subprocess.CompletedProcess(argv, 2, "", "research: @proofs: no open inbox")
    relay.once()
    relay.once()
    assert len(relay.posts) == 1 and relay.OFFSET.read_text() == "1"


def test_a_record_the_relay_cant_render_goes_to_infra_as_raw_json_never_a_sentence_of_question_marks(relay):
    """#agent-alerts 20:14:59Z, 19 Oct: "*Agent watch:* run 6 runs (?) of lane None ended rc=? after ? on ?." was the watch's
    grouped unknown-sender record through a relay that didn't know it. Now this relay renders it, and anything it doesn't know,
    or that lacks what its sentence reads, is posted raw to @infra."""
    infra = relay.slack.tag(relay.slack.entry(relay.REG, "infra"))
    grouped = {"t": "2026-10-19T20:18:45Z", "kind": "run_ended", "agent": "", "name": "", "run": "r20261019-195914-0b94",
               "handle": None, "role": "unknown", "count": 1, "runs": [
                   {"run": "r20261019-195914-0b94", "node": "vy-nebius-cpu-2", "what": "r20261019-195914-0b94", "rc": 0,
                    "sender": "https://cursor.com/agents/bc-c8838f5a-e10e-54ad-be4f-bbad4534f555", "state": "done",
                    "after": "11 min", "at": "2026-10-19T20:15:51Z"}],
               "senders": ["https://cursor.com/agents/bc-c8838f5a-e10e-54ad-be4f-bbad4534f555"]}
    odd = [{"kind": "brand_new", "agent": "bc-x", "detail": "a kind added to the watch first"},
           {"kind": "run_ended", "agent": "", "name": "", "run": "6 runs", "handle": None, "role": "unknown", "count": 6},
           {"kind": "run_ended", "agent": "bc-x", "run": "r1", "handle": "proofs", "role": "lane", "what": "x"},
           {"kind": "run_ended", "agent": "bc-x", "run": "r2", "handle": "proofs", "role": "mystery", "node": "n", "rc": 0,
            "after": "1 s", "what": "x"}]
    relay.log_text["text"] = "\n".join(json.dumps(r) for r in [grouped, *odd]) + "\n{not json\n"
    relay.once()
    assert relay.posts[0].startswith(f"{infra} *Agent watch:* 1 research run ended whose sender no handle owns")
    for post, rec in zip(relay.posts[1:], [*odd, None]):
        assert post.startswith(f"{infra} *Agent watch:* a record this relay can't render (kind `")
        assert "?" not in post.split("```")[0] and post.endswith("```")
        if rec:
            assert json.loads(post.split("```")[1]) == rec
    assert "{not json" in relay.posts[-1] and "`garbled`" in relay.posts[-1]
    assert [r["thread_ts"] for r in rings(relay)] == [INFRA_INBOX["inbox_ts"]] * 6 and relay.ring_runs == []


def test_doorbells_count_ring_runs_doorbell_for_ready_to_wake(relay, monkeypatch):
    """The watch's ready-to-wake rule reads each inbox's latest doorbell: ring-run's `from runs, run ...` is one."""
    class Bells:
        local = frozenset({"proofs"})

        def get_route(self, name, fresh=False):
            return PROOFS_INBOX if name == "proofs" else None
    monkeypatch.setattr(relay.slack, "Routes", Bells)
    box = PROOFS_INBOX["inbox_ts"]
    msgs = [{"ts": box, "text": "inbox"},
            {"ts": "1760000100.000001", "username": "router", "text": f"from infra, thread {box}: an alert"},
            {"ts": "1760000200.000001", "username": "router", "text": "from runs, run r20261019-040000-ab12 rc 0 (bc-x): done"},
            {"ts": "1760000300.000001", "username": "someone", "text": "from runs, run r1 rc 0 (bc-x): typed by hand"}]
    api = type("A", (), {"paged": lambda self, *a, **k: msgs})()
    assert relay.real_bells(api, 1760000400.0)["inboxes"] == {"proofs": {"agent": PROOFS_INBOX["holder"], "last": 1760000200.000001}}


def test_a_failure_the_watch_acted_on_leads_with_its_owner_and_rings_that_inbox(relay):
    """Ring the owner at once. A coordinator's owner is top; a worker's, its launcher's handle."""
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
        f"{top} *Agent watch:* verity: proofs ({link}, run run-s) streamed nothing for 30 min since 2026-10-19T10:00:00Z, "
        "its agent not moving either (a hung call or a suspended VM); the watch cancelled and continued. New run run-n.",
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


def test_a_run_its_own_lane_cancelled_and_a_relaunch_declined_are_posted_quietly_and_ring_no_one(relay):
    """INF-141 (b): a run its lane cancelled ended rc=143 by its own hand: no "resume it", no ring, no handle tagged. A relaunch
    the watch declined (the worker reported back) is the same: on record in the channel, nothing for anyone to do."""
    w = "bc-6f3f0e29-409b-53b4-9bb0-5f9d68b71c23"
    relay.log_text["text"] = "\n".join(json.dumps(r) for r in [
        {"kind": "run_ended", "agent": w, "name": "SALT-WIRE: salt tag", "run": "r20261019-182846-7d10", "handle": "proofs",
         "role": "worker", "node": "vy-nebius-cpu-2", "what": "quick", "rc": 143, "state": "failed", "after": "1 h 35 min",
         "at": "2026-10-19T20:04:18Z", "lane": "proofs", "quiet": True, "cancelled_by": "ubuntu",
         "cancel_reason": "superseded at SALT-WIRE's final head (research cancel by ubuntu)"},
        {"kind": "not_relaunched", "agent": w, "name": "STRAND: land #2822", "run": "", "role": "worker", "lane": "proofs",
         "handle": "proofs", "since": "2026-10-19T20:13:52Z", "ended_run": "r20261019-200828-85fc", "quiet": True,
         "why": "its last message reads as a report back (\u201c\u2026Nothing changed: #2822's head stays\u201d)"}]) + "\n"
    relay.once()
    cancel, declined = relay.posts
    assert cancel == (f"*Agent watch:* run r20261019-182846-7d10 (quick) of worker SALT-WIRE: salt tag "
                      f"(<https://cursor.com/agents/{w}|{w}>) ended rc=143 after 1 h 35 min on vy-nebius-cpu-2, cancelled by its "
                      "own lane (ubuntu: \u201csuperseded at SALT-WIRE's final head (research cancel by ubuntu)\u201d): nothing "
                      "to resume.")
    assert declined.startswith("*Agent watch:* worker STRAND: land #2822 (") and declined.endswith(
        "was idle after its run r20261019-200828-85fc ended, and the watch did not relaunch it: its last message reads as a "
        "report back (\u201c\u2026Nothing changed: #2822's head stays\u201d).")
    assert rings(relay) == [] and relay.ring_runs == []


def test_a_listed_workers_run_end_says_the_watch_resumes_it(relay):
    """A relaunched worker is an agent the API lists: its coordinator can't Task it, and the watch starts a run on it."""
    rec = {"kind": "run_ended", "agent": "bc-7", "name": "proofs worker ZKS10 (relaunch of 1111)", "run": "r1", "handle": "proofs",
           "role": "worker", "node": "n", "what": "x", "rc": 0, "state": "done", "after": "1 s", "listed": True}
    relay.log_text["text"] = json.dumps(rec) + "\n"
    relay.once()
    assert relay.posts[0].endswith("; the watch starts a run on it with the result if it is still idle in 10 min.")
    assert relay.ring_runs == [], "rc 0: posted unrung (INF-216)"
    rec.update(run="r2", rc=1, state="failed")
    relay.log_text["text"] += json.dumps(rec) + "\n"
    relay.once()
    assert relay.ring_runs[0][-2:] == ["--to", "@proofs"], "a worker's failed run still rings its lane"


def test_a_restart_never_posts_a_line_again_even_with_its_offset_lost(relay):
    """INF-141 (a): the sent-set beside the offset keeps each line posted; a relay restarted from offset 0 (a lost or reset
    offset file, a new pod tree) skips every line it already posted and posts only the new one."""
    relay.log_text["text"] = line("run-1") + "\n" + line("run-2") + "\n"
    relay.once()
    relay.OFFSET.unlink()
    relay.log_text["text"] += line("run-3") + "\n"
    relay.once()
    assert [p.split("run ")[1].split(")")[0] for p in relay.posts] == ["run-1", "run-2", "run-3"]
    assert relay.OFFSET.read_text() == "3" and len(json.loads(relay.sent_path().read_text())) == 3


def test_a_slack_read_cut_short_skips_the_doorbells_and_still_posts(relay, monkeypatch):
    """22:21Z 19 Oct: conversations.replies came back short (http.client.IncompleteRead, no OSError) and the relay exited."""
    def short(api, now):
        raise relay.http.client.IncompleteRead(b"x" * 10, 5)
    monkeypatch.setattr(relay, "bells", short)
    relay.log_text["text"] = line("run-1") + "\n"
    relay.once()
    assert len(relay.posts) == 1 and relay.wrote == []


def test_a_holder_held_unread_is_posted_and_rings_top(relay):
    """top, 24 Oct 21:50Z: a holder that sat in a streaming worker's call for an hour rings @top."""
    top = relay.slack.tag(relay.slack.entry(relay.REG, "top"))
    co = PROOFS_INBOX["holder"]
    relay.log_text["text"] = json.dumps(
        {"kind": "held_unread", "agent": co, "name": "verity: proofs", "run": "run-u", "since": "2026-10-24T17:25:00Z",
         "minutes": 60, "status": "RUNNING", "action": "cancelled and continued", "new_run": "run-n", "owner": "top",
         "handle": "proofs", "reason": "a doorbell rang at 2026-10-24T17:40:00Z"}) + "\n"
    relay.once()
    (post,) = relay.posts
    assert post.startswith(f"{top} *Agent watch:* ") and "did nothing itself for 60 min since 2026-10-24T17:25:00Z" in post
    assert "(a doorbell rang at 2026-10-24T17:40:00Z); the watch cancelled and continued" in post
    assert [r["thread_ts"] for r in rings(relay)] == [TOP_INBOX["inbox_ts"]]
