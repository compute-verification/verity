"""Top's loops on the control pod (infra/control/top_watch.py): the asks relay applies Daniel's ticks to his Asks pages and rings
the lane and @top once each, quoting only his own new reply text; the sweep rings each finding's owner once."""
import calendar
import importlib.util
import sys
from pathlib import Path

CONTROL = Path(__file__).resolve().parents[3] / "infra" / "control"
_spec = importlib.util.spec_from_file_location("top_watch", CONTROL / "top_watch.py")
T = sys.modules["top_watch"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
NOW = float(calendar.timegm((2026, 10, 19, 20, 0, 0)))


def rt(s):
    return [{"plain_text": s}]


def page(pid, **p):
    kinds = {"Status": "status", "Approve": "checkbox", "Choose": "select", "Kind": "select", "Deadline": "date", "Lane": "select",
             "Ask": "title", "Slack card": "url"}
    props = {}
    for k, v in {"Status": "Waiting on Daniel", "Approve": False, "Lane": "proofs", "Ask": f"ask {pid}", "Kind": "Blocking",
                 "Card ID": f"card-{pid}", "Recommendation": "A. the first", "Slack card": "https://x/archives/C1/p1792000000000100",
                 **p}.items():
        kind = kinds.get(k, "rich_text")
        val = {"status": {"name": v}, "select": {"name": v} if v else None, "checkbox": v, "date": {"start": v}, "url": v,
               "title": rt(v), "rich_text": rt(v)}[kind]
        props[k] = {"type": kind, kind: val}
    return {"id": pid, "url": f"https://notion/{pid}", "properties": props}


def callout(*pieces):
    return [{"id": "co", "type": "callout", "has_children": True,
             "callout": {"color": "yellow_background", "rich_text": rt("Your reply. Write below this line.")}},
            {"id": "h-a", "type": "heading_3", "heading_3": {"rich_text": rt("A. Keep it (recommended)")}},
            {"id": "h-b", "type": "heading_3", "heading_3": {"rich_text": rt("B. Drop it")}}], \
        [{"id": bid, "type": "paragraph", "paragraph": {"rich_text": rt(text)}} for bid, text in pieces]


class Notion:
    def __init__(self, *pages):
        self.pages = {p["id"]: p for p in pages}
        self.blocks, self.children, self.comments, self.patches = {}, {}, {}, []

    def call(self, method, path, body=None):
        if method == "POST":
            conds = (body["filter"].get("and") or [])
            hit = [p for p in self.pages.values() if not conds or (
                T.props(p)["Status"] == "Waiting on Daniel" and T.props(p)["Kind"] == "Default" and T.props(p).get("Deadline")
                and T.props(p)["Deadline"] < conds[2]["date"]["before"])]
            return {"results": hit}
        if method == "PATCH":
            pid = path.split("/")[2]
            self.patches.append((pid, body["properties"]["Status"]["status"]["name"],
                                 body["properties"]["Decision"]["rich_text"][0]["text"]["content"]))
            return {}
        if path.startswith("/comments"):
            return {"results": self.comments.get(path.split("block_id=")[1].split("&")[0], [])}
        bid = path.split("/")[2]
        return {"results": self.children.get(bid, []) if bid == "co" else self.blocks.get(bid, [])}

    def body(self, pid, *pieces):
        self.blocks[pid], self.children["co"] = callout(*pieces)


class Ring:
    def __init__(self):
        self.rings = []

    def __call__(self, name, body, thread=None, reply=None):
        self.rings.append((name, body, thread))
        return True


def test_approve_ticked_sets_approved_and_rings_the_lane_and_top_once():
    n, ring = Notion(page("p1", Approve=True)), Ring()
    n.body("p1")
    state = {}
    T.asks_pass(state, n, ring, "ds", NOW, False)
    assert n.patches == [("p1", "Approved", "Approved by Daniel (Approve ticked, 1:00 PM PDT, Mon 19 Oct)")]
    (lane, to_lane, thread), (top, to_top, _) = ring.rings
    assert lane == "proofs" and to_lane.startswith("proofs: Approved by Daniel") and thread == "1792000000.000100"
    assert "research msg ask-daniel resolve card-p1 --choice A --note https://notion/p1" in to_lane
    assert top == "top" and "proofs's ask" in to_top
    T.asks_pass(state, n, ring, "ds", NOW + 900, False)
    assert len(n.patches) == 1 and len(ring.rings) == 2


def test_choose_takes_that_options_heading_and_a_tick_on_a_lanes_decision_endorses_it():
    n, ring = Notion(page("p1", Choose="B"), page("p2", Status="Decided by lane", Approve=True, Decision="Lane took A.")), Ring()
    n.body("p1")
    n.body("p2")
    T.asks_pass({}, n, ring, "ds", NOW, False)
    assert ("p1", "Decided otherwise", "Daniel chose B: Drop it") in n.patches
    assert ("p2", "Approved", "Lane took A. Endorsed by Daniel.") in n.patches
    assert any("--choice B" in body for _, body, _ in ring.rings)


def test_a_default_page_past_its_deadline_gets_its_default_and_one_before_it_waits():
    late = page("p1", Kind="Default", Deadline="2026-10-19T19:00:00.000+00:00", **{"If no answer": "A applies at noon"})
    early = page("p2", Kind="Default", Deadline="2026-10-21T16:00:00.000+00:00")
    n, ring = Notion(late, early), Ring()
    n.body("p1")
    n.body("p2")
    T.asks_pass({}, n, ring, "ds", NOW, False)
    assert n.patches == [("p1", "Default applied", "Default applied 1:00 PM PDT, Mon 19 Oct: A applies at noon")]


def test_only_daniels_new_reply_text_is_quoted_and_a_lanes_answer_under_it_is_not():
    n, ring, state = Notion(page("p1")), Ring(), {}
    n.body("p1", ("b1", "old text, already relayed"))
    T.asks_pass(state, n, None, "ds", NOW, False)                # the seed: nothing rung
    assert ring.rings == [] and n.patches == []
    n.body("p1", ("b1", "old text, already relayed"), ("b2", "Go with B, but keep the tests."))
    T.asks_pass(state, n, ring, "ds", NOW + 900, False)
    assert [(to, "Go with B, but keep the tests." in body) for to, body, _ in ring.rings] == [("proofs", True), ("top", True)]
    ring.rings.clear()
    n.body("p1", ("b1", "old text, already relayed"), ("b2", "Go with B, but keep the tests."), ("b3", "@proofs: Taken, as B."),
           ("b4", "The tests stay."))
    T.asks_pass(state, n, ring, "ds", NOW + 1800, False)
    assert ring.rings == []
    n.body("p1", ("b1", "x"), ("b2", "x"), ("b3", "x"), ("b4", "x"), ("b5", "And one more thing."))
    T.asks_pass(state, n, ring, "ds", NOW + 2700, False)
    assert len(ring.rings) == 2 and "And one more thing." in ring.rings[0][1]


def test_a_lanes_answer_signed_with_its_name_and_a_colon_is_not_quoted_as_daniels():
    n, ring, state = Notion(page("p1")), Ring(), {}
    n.body("p1")
    T.asks_pass(state, n, None, "ds", NOW, False)
    n.body("p1", ("b1", "proofs: Default A applied (6:05 PM PDT, Fri 23 Oct, under your night orders)."),
           ("b2", "top: logged."))
    T.asks_pass(state, n, ring, "ds", NOW + 900, False)
    assert ring.rings == []
    n.body("p1", ("b1", "x"), ("b2", "x"), ("b3", "Note: keep TileChecks on the same plan."))
    T.asks_pass(state, n, ring, "ds", NOW + 1800, False)
    assert [to for to, _, _ in ring.rings] == ["proofs", "top"] and "Note: keep TileChecks" in ring.rings[0][1]


def test_a_comment_signed_with_the_lanes_name_and_a_time_is_the_lanes():
    n, ring, state = Notion(page("p1")), Ring(), {}
    n.body("p1")
    n.comments["p1"] = [{"id": "c1", "rich_text": rt("proofs, 26 Oct 03:20Z: provisionally accepted A now.")},
                        {"id": "c2", "rich_text": rt("proofs 03:25Z: the lock DM will show it.")},
                        {"id": "c3", "rich_text": rt("top, 26 Oct 03:40Z: logged.")}]
    T.asks_pass(state, n, ring, "ds", NOW, False)
    assert ring.rings == []
    n.comments["p1"].append({"id": "c4", "rich_text": rt("proofs, please wait for the Fri default.")})
    T.asks_pass(state, n, ring, "ds", NOW + 900, False)
    assert [to for to, _, _ in ring.rings] == ["proofs", "top"] and "please wait" in ring.rings[0][1]


def test_a_new_comment_is_quoted_unless_it_starts_with_a_handle():
    n, ring, state = Notion(page("p1")), Ring(), {}
    n.body("p1")
    n.comments["p1"] = [{"id": "c1", "rich_text": rt("@proofs: done")}, {"id": "c2", "rich_text": rt("Why not C?")}]
    T.asks_pass(state, n, ring, "ds", NOW, False)
    assert [to for to, _, _ in ring.rings] == ["proofs", "top"] and "Why not C?" in ring.rings[0][1]
    T.asks_pass(state, n, ring, "ds", NOW + 900, False)
    assert len(ring.rings) == 2


def test_a_dry_run_writes_nothing_to_notion():
    n, ring = Notion(page("p1", Approve=True)), Ring()
    n.body("p1")
    T.asks_pass({}, n, ring, "ds", NOW, True)
    assert n.patches == [] and len(ring.rings) == 2


REG = {"workspace": "w.slack.com", "channels": {"agent-coordination": {"id": "C1"}}}


def test_an_unanswered_ask_rings_each_addressee_once_and_a_machines_none():
    state, ring = {}, Ring()
    traffic = lambda: {"unanswered": [{"ts": "1.1", "by": "proofs", "to": ["top", "infra"]}, {"ts": "2.2", "by": "vy-monitors",
                                                                                               "to": ["ci"]}]}
    T.unanswered(state, ring, REG, NOW, traffic)
    T.unanswered(state, ring, REG, NOW + 3600, traffic)
    assert [to for to, _, _ in ring.rings] == ["top", "infra"] and "https://w.slack.com/archives/C1/p11" in ring.rings[0][1]


def test_an_active_agent_with_no_handle_rings_top_once():
    state, ring = {}, Ring()
    agents = lambda: [{"id": "bc-a", "status": "ACTIVE", "name": "x"}, {"id": "bc-b", "status": "ACTIVE"},
                      {"id": "bc-c", "status": "IDLE"}]
    T.holderless(state, ring, NOW, agents, {"bc-b"})
    T.holderless(state, ring, NOW + 3600, agents, {"bc-b"})
    assert [(to, "x (bc-a)" in body) for to, body, _ in ring.rings] == [("top", True)]


def test_main_stalled_rings_ci_only_while_next_is_ahead_and_once_per_main_head():
    state, ring = {}, Ring()
    ahead = {"n": 0}

    def read():
        return {"sha": "abc1234567", "committed": "2026-10-19T18:30:00Z", "ahead": ahead["n"]}

    T.main_stalled(state, ring, NOW - 3600, read)
    T.main_stalled(state, ring, NOW, read)
    assert ring.rings == []
    ahead["n"] = 3
    T.main_stalled(state, ring, NOW, read)
    T.main_stalled(state, ring, NOW + 3600, read)
    assert [to for to, _, _ in ring.rings] == ["ci"] and "60 min" in ring.rings[0][1]


def test_a_train_that_just_landed_is_not_a_stall_though_drive_built_its_commit_long_ago():
    state, ring = {}, Ring()
    heads = iter(["old", "landed", "landed", "landed"])

    def read():
        return {"sha": next(heads), "committed": "2026-10-19T16:00:00Z", "ahead": 700}

    for t in (NOW - 3600, NOW, NOW + 1800):
        T.main_stalled(state, ring, t, read)
    assert ring.rings == []
    T.main_stalled(state, ring, NOW + 3600, read)
    assert [body.split(" (")[0] for _, body, _ in ring.rings] == ["ci: main hasn't moved for 60 min"]


def test_an_idle_h200_rings_top_once_a_day_after_1500z():
    state, ring = {}, Ring()
    rows = [{"name": "vy-glm53-eu", "platform": "gpu-h200-sxm", "read": True, "gpu_util": [0, 2, 4, 0]},
            {"name": "vy-glm53-eu2", "platform": "gpu-h200-sxm", "read": True, "gpu_util": [90, 80]},
            {"name": "vy-nebius-1", "platform": "gpu-b200", "read": True, "gpu_util": [0]}]
    morning = float(calendar.timegm((2026, 10, 19, 14, 0, 0)))
    T.h200s(state, ring, morning, lambda a, b: rows)
    assert ring.rings == []
    T.h200s(state, ring, NOW, lambda a, b: rows)
    T.h200s(state, ring, NOW + 3600, lambda a, b: rows)
    assert len(ring.rings) == 1 and "vy-glm53-eu 1.5%" in ring.rings[0][1] and "eu2" not in ring.rings[0][1]


def test_a_seeded_sweep_records_its_findings_and_rings_nobody():
    state = {}
    T.unanswered(state, None, REG, NOW, lambda: {"unanswered": [{"ts": "1.1", "by": "proofs", "to": ["top"]}]})
    ring = Ring()
    T.unanswered(state, ring, REG, NOW, lambda: {"unanswered": [{"ts": "1.1", "by": "proofs", "to": ["top"]}]})
    assert ring.rings == [] and "unanswered:1.1:top" in state["rung"]
    T.prune(state, NOW + T.KEEP_S)
    assert state["rung"] == {}


def finding(issue, kind, lane):
    return {"issue": issue, "url": f"https://linear.app/x/issue/{issue}", "title": "t", "team": issue.split("-")[0], "lane": lane,
            "state": "In Progress", "kind": kind, "fix": "fix it"}


FOUND = [finding("INF-1", "orphan", "infra"), finding("INF-2", "quiet", "infra"), finding("INF-3", "quiet", "infra"),
         finding("PRF-1", "undated", "proofs"), finding("DAN-1", "lane", "-")]
VOLUME = {"totals": {"posts": 1758, "words": 197955, "over": 997, "median": 90}}


def at(day, hour, minute=0):
    return float(calendar.timegm((2026, 10, day, hour, minute, 0)))


def test_the_linear_audit_rings_each_lane_once_a_window_and_issues_with_no_lane_to_top(tmp_path):
    state, ring, log = {}, Ring(), tmp_path / T.AUDIT_LOG
    for t in (at(19, 12, 30), at(19, 14, 30)):                             # one 12 h window, before @top's daily line
        T.linear_audit(state, ring, t, lambda: FOUND, lambda: VOLUME, log)
    assert [to for to, _, _ in ring.rings] == ["top", "infra", "proofs"]
    (_, no_lane, _), (_, infra, _), _ = ring.rings
    assert infra == ("infra: the Linear audit finds 3 on your open issues (orphan 1, quiet 2): `research msg linear audit "
                     "--lane infra` lists them, one line each with its fix.")
    assert no_lane.startswith("top: the Linear audit finds 1 on issues with no lane (lane 1): `research msg linear audit --lane -`")
    T.linear_audit(state, ring, at(19, 20), lambda: FOUND, lambda: VOLUME, log)
    assert [to for to, _, _ in ring.rings[3:]] == ["top"] and ring.rings[3][1].startswith("top: the Linear audit at")
    T.linear_audit(state, ring, at(20, 0, 30), lambda: FOUND, lambda: VOLUME, log)  # the next window
    assert [to for to, _, _ in ring.rings[4:]] == ["top", "infra", "proofs"]


def test_a_clean_audit_rings_nobody_and_tops_daily_line_gives_the_change_from_the_line_24_h_earlier(tmp_path):
    state, ring, log = {}, Ring(), tmp_path / T.AUDIT_LOG
    T.linear_audit(state, ring, at(19, 20), lambda: FOUND, lambda: VOLUME, log)
    assert ring.rings[-1][1] == ("top: the Linear audit at 2026-10-19T20:00:00Z: 5 findings (orphan 1, undated 1, quiet 2, lane 1); "
                                 "#agent-coordination's last 24 h: 1758 posts, 197955 words, 997 over 80 words; "
                                 "no line 24 h earlier yet; by lane: infra 3, - 1, proofs 1")
    rung = len(ring.rings)
    quieter = {"totals": {"posts": 1000, "words": 90000, "over": 400, "median": 60}}
    T.linear_audit(state, ring, at(20, 14), lambda: [], lambda: quieter, log)
    assert len(ring.rings) == rung                                          # clean, and before 15:00Z
    T.linear_audit(state, ring, at(20, 20), lambda: [], lambda: quieter, log)
    assert [to for to, _, _ in ring.rings[rung:]] == ["top"]
    assert ring.rings[-1][1] == ("top: the Linear audit at 2026-10-20T20:00:00Z: 0 findings (-5) (orphan 0 (-1), undated 0 (-1), "
                                 "quiet 0 (-2), lane 0 (-1)); #agent-coordination's last 24 h: 1000 posts (-758), 90000 words "
                                 "(-107955), 400 over 80 words (-597); changes from 2026-10-19T20:00:00Z")
    T.linear_audit(state, ring, at(20, 21), lambda: [], lambda: quieter, log)
    assert [to for to, _, _ in ring.rings[rung:]] == ["top"]


def test_each_audit_pass_appends_its_line_with_the_volume_null_when_unread_and_a_dry_run_appends_none(tmp_path):
    log = tmp_path / T.AUDIT_LOG

    def unread():
        raise RuntimeError("no Slack token")

    T.linear_audit({}, Ring(), at(19, 20), lambda: FOUND, lambda: VOLUME, log)
    T.linear_audit({}, Ring(), at(19, 21), lambda: FOUND[:1], unread, log, lambda: ["HASH", "TRN"])
    T.linear_audit({}, Ring(), at(19, 22), lambda: FOUND, lambda: VOLUME, None)
    first, second = [T.json.loads(x) for x in log.read_text().splitlines()]
    assert first == {"t": "2026-10-19T20:00:00Z", "findings": 5,
                     "kinds": {"orphan": 1, "milestone": 0, "undated": 1, "overdue": 0, "quiet": 2, "stale": 0, "title": 0,
                               "landed": 0, "lane": 1},
                     "lanes": {"-": {"lane": 1}, "infra": {"orphan": 1, "quiet": 2}, "proofs": {"undated": 1}},
                     "volume": {"posts": 1758, "words": 197955, "over": 997}, "unread": []}
    assert second["findings"] == 1 and second["lanes"] == {"infra": {"orphan": 1}} and second["volume"] is None
    assert second["unread"] == ["HASH", "TRN"]


def test_tops_daily_line_names_the_teams_the_audit_cant_read(tmp_path):
    ring = Ring()
    T.linear_audit({}, ring, at(19, 20), lambda: [], lambda: VOLUME, tmp_path / T.AUDIT_LOG, lambda: ["HASH", "TRN"])
    (to, body, _), = ring.rings
    assert to == "top" and body.endswith("; no line 24 h earlier yet; not audited, the app can't read team(s) HASH, TRN")
