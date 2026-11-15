"""`research goals blockers` over a hand-made page and a given time: the goals above a blocker are counted through two levels
of links and stop at a done goal, a parked, done or blank blocker is left out, a cycle of links ends, the ranking is by
tier-1 goals above, then all goals above, then due, and `--since` names what is standing, new, changed and cleared."""
from __future__ import annotations

import json
from datetime import datetime

from research import goals, goals_blockers

NOW = "2026-11-14T00:00-08:00"


def goal(gid: str, owner: str = "a", due: str = "2026-11-18", **more) -> dict:
    return {"id": gid, "title": "A plain headline", "status": "off-track", "due": due, "tags": [], "owner": owner} | more


#: `a/leaf` sits under `a/mid` under the tier-1 `t/project`. `b/leaf` is tier 1 itself, and the tier-1 `t/other` waits on
#: it only through the done `b/done`. `c/one` and `c/two` wait on each other. `z/early` and `z/late` have nothing above.
#: Left out: `p/paused` (only under the parked goal), `d/done` (done) and `e/blank` (a blank text).
PAGE = [
    goal("t/project", "t", "2026-11-30", tier=1, status="on-track", waitingOn=[{"goal": "a/mid"}, {"ask": "card-1"}]),
    goal("a/mid", due="2026-11-20", waitingOn=[{"goal": "a/leaf"}], blockers="mid waits on the leaf",
         updatedAt="2026-11-14T06:30:00.000Z"),
    goal("a/leaf", blockers="the leaf's\n  text " + "x" * 200, updatedAt="2026-11-13T05:00:00.000Z"),
    goal("t/other", "t", "2026-11-30", tier=1, status="on-track", waitingOn=[{"goal": "b/done"}]),
    goal("b/done", "b", status="done", waitingOn=[{"goal": "b/leaf"}]),
    goal("b/leaf", "b", tier=1, blockers="b's text", updatedAt="2026-11-14T03:00:00.000Z"),
    goal("c/one", "c", waitingOn=[{"goal": "c/two"}], blockers="c's text"),
    goal("c/two", "c", status="on-track", waitingOn=[{"goal": "c/one"}]),
    goal("z/late", "z", "2026-11-20", blockers="late"),
    goal("z/early", "z", "2026-11-15T09:00-08:00", blockers="early"),
    goal("top/parked", "top", "2026-12-01", tier=1, status="on-track", waitingOn=[{"goal": "p/paused"}]),
    goal("p/paused", "p", blockers="paused"),
    goal("d/done", "d", status="done", blockers="was blocked"),
    goal("e/blank", "e", blockers="  "),
]


def ranked(page: list[dict], **kw) -> list[tuple[str, list[str], int]]:
    return [(r["id"], r["tier1"], r["above"]) for r in goals_blockers.rows(page, datetime.fromisoformat(NOW), **kw)]


def test_goals_above_are_counted_through_links_and_stop_at_a_done_goal_and_a_cycle_ends():
    assert ranked(PAGE) == [
        ("a/leaf", ["t/project"], 2),
        ("a/mid", ["t/project"], 1),
        ("b/leaf", ["b/leaf"], 0),
        ("c/one", [], 1),
        ("z/early", [], 0),
        ("z/late", [], 0),
    ], "t/other waits on b/leaf only through a done goal, so it is not above it; a parked, done or blank blocker is left out"
    rows = {r["id"]: r for r in goals_blockers.rows(PAGE, datetime.fromisoformat(NOW))}
    assert (rows["a/leaf"]["hours"], rows["a/mid"]["hours"], rows["b/leaf"]["hours"], rows["c/one"]["hours"]) == (27, 1, 5, None)
    assert goals_blockers.summary(list(rows.values())) == {"blockers": 6, "under_tier1": 3, "median_hours": 5, "oldest_hours": 27}


def test_since_names_standing_new_changed_and_cleared():
    earlier = [
        goal("a/leaf", blockers=PAGE[2]["blockers"]),
        goal("a/mid", blockers="an older text"),
        goal("b/leaf", "b", status="on-track"),
        goal("d/done", "d", blockers="was blocked"),
        goal("e/blank", "e", blockers="was blocked"),
        goal("r/removed", "r", blockers="was blocked"),
        goal("p/paused", "p", blockers="paused"),
        goal("q/finished", "q", status="done", blockers="done then too"),
    ]
    rows = goals_blockers.rows(PAGE, datetime.fromisoformat(NOW))
    moved = goals_blockers.since(rows, PAGE, earlier)
    assert {r["id"]: r["since"] for r in rows} == {"a/leaf": "standing", "a/mid": "changed", "b/leaf": "new", "c/one": "new",
                                                   "z/early": "new", "z/late": "new"}
    assert moved == {"standing": 1, "new": 4, "changed": 1, "cleared": 3, "cleared_ids": ["d/done", "e/blank", "r/removed"]}, \
        "a parked goal still carrying its blocker is not cleared, and a goal done in both dumps carried none then"


def test_the_command_ranks_filters_limits_and_compares_with_an_earlier_dump(capsys, tmp_path):
    page = tmp_path / "page.json"
    page.write_text(json.dumps(PAGE))
    assert goals.main(["blockers", "--from", str(page), "--now", NOW, "--top", "2"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == " T1 ABOVE  GOAL  OWNER  STATUS  DUE  WRITTEN  BLOCKER"
    assert out[1] == "  1     2  a/leaf  a  off-track  due Wed 18 Nov  27h  the leaf's text " + "x" * 143 + "…"
    assert out[2] == "  1     1  a/mid  a  off-track  due Fri 20 Nov  1h  mid waits on the leaf"
    assert out[3] == ("research goals blockers: 6 open goal(s) carry a blocker, showing 2; 3 under a tier-1 goal or tier 1, "
                      "written a median 5 h and at most 27 h ago") and len(out) == 4

    earlier = tmp_path / "goals-20261114T0730Z.json"
    earlier.write_text(json.dumps([goal("z/late", "z", blockers="late"), goal("z/gone", "z", blockers="gone")]))
    assert goals.main(["blockers", "--from", str(page), "--now", NOW, "--owner", "z", "--since", str(earlier)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        " T1 ABOVE  SINCE    GOAL  OWNER  STATUS  DUE  WRITTEN  BLOCKER",
        "  0     0  new      z/early  z  off-track  due Sun 15 Nov  ?  early",
        "  0     0  standing z/late  z  off-track  due Fri 20 Nov  ?  late",
        f"since {earlier} (taken Fri 13 Nov 11:30 PM PST): 1 standing, 1 new, 0 changed, 1 cleared: z/gone",
        "research goals blockers: 2 open goal(s) carry a blocker, showing 2; 0 under a tier-1 goal or tier 1",
    ]

    assert goals.main(["blockers", "--from", str(page), "--now", NOW, "--owner", "a", "--owner", "b", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in doc["rows"]] == ["a/leaf", "a/mid", "b/leaf"] and "since" not in doc
    assert doc["rows"][0]["blockers"] == PAGE[2]["blockers"] and doc["rows"][0]["tier1"] == ["t/project"]
    assert goals.main(["blockers", "--from", str(page), "--now", "2026-11-14T00:00"]) == 1
    assert "give its offset" in capsys.readouterr().err
