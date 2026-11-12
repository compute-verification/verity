"""`research goals audit` over a page of goals and a given time: each kind of finding names its goal and its fix, a lead with
nothing wrong says so, `--owner` narrows the audit and reports a lead with no open goal, and the exit code says whether
anything was found."""
from __future__ import annotations

import json
from datetime import datetime

from research import goals, goals_audit

NOW = "2026-11-10T22:00-08:00"


def goal(gid: str, owner: str, due: str, **more) -> dict:
    return {"id": gid, "title": "Main moves within two hours", "status": "on-track", "due": due, "tags": ["ci"], "owner": owner} | more


#: ci: a project waiting on a piece due in 6 hours (nothing found). lean: an overdue goal under no project with a long title,
#: and a goal due before the one it waits on and waiting on one the page doesn't hold. vllm: only a done goal.
PAGE = [
    goal("ci/landing", "ci", "2026-11-13", waitingOn=[{"goal": "ci/shards"}, {"ask": "card-1"}]),
    goal("ci/shards", "ci", "2026-11-11T04:00-08:00"),
    goal("lean/audit", "lean", "2026-11-10T18:00-08:00", status="off-track",
         title="The Lean audit keeps its warm build when it is stopped early"),
    goal("lean/first", "lean", "2026-11-12", waitingOn=[{"goal": "lean/later"}, {"goal": "lean/gone"}, {"goal": "lean/done"}]),
    goal("lean/later", "lean", "2026-11-14", tier=2),
    goal("lean/done", "lean", "2026-11-20", status="done"),
    goal("vllm/old", "vllm", "2026-11-01", status="done"),
]


class Page:
    def call(self, body: dict | None = None) -> list[dict]:
        assert body is None, "the audit only reads the page"
        return PAGE


def kinds(lead: dict) -> list[tuple[str, str | None]]:
    return [(f["kind"], f["goal"]) for f in lead["findings"]]


def test_each_kind_names_its_goal_and_its_fix():
    leads = goals_audit.findings(PAGE, datetime.fromisoformat(NOW))
    assert list(leads) == ["ci", "lean"], "a lead whose goals are all done owns no open goal, so it isn't listed unasked"
    assert leads["ci"] == {"open": 2, "off_track": 0, "findings": []}
    assert (leads["lean"]["open"], leads["lean"]["off_track"]) == (3, 1)
    assert kinds(leads["lean"]) == [("overdue", "lean/audit"), ("unplaced", "lean/audit"), ("order", "lean/first"),
                                    ("order", "lean/first"), ("title", "lean/audit"), ("no-step", None)]
    text = {(f["kind"], f["text"].split("`")[3] if f["kind"] == "order" else None): f["text"] for f in leads["lean"]["findings"]}
    assert "was due Tue 10 Nov, 6:00 PM" in text["overdue", None] and "research goals done lean/audit" in text["overdue", None]
    assert "research goals set PROJECT --wait goal:lean/audit" in text["unplaced", None] and "--tier 1" in text["unplaced", None]
    assert "(due Thu 12 Nov, end of day) waits on `lean/later` (due Sat 14 Nov, end of day)" in text["order", "lean/later"]
    assert "waits on `lean/gone`, which the page doesn't hold" in text["order", "lean/gone"]
    assert "has 12 words" in text["title", None] and "--title T" in text["title", None]
    assert "due in the next 12 hours" in text["no-step", None] and "--wait goal:STEP" in text["no-step", None]
    assert goals_audit.tally(leads) == {"overdue": 1, "unplaced": 1, "order": 2, "title": 1, "no-step": 1}


def test_a_day_is_due_at_its_end_in_pacific_time_and_a_set_tier_or_a_link_places_a_goal():
    page = [goal("a/day", "a", "2026-11-10", tier=1), goal("a/piece", "a", "2026-11-12"),
            goal("a/project", "a", "2026-11-13", waitingOn=[{"goal": "a/piece"}])]
    assert goals_audit.findings(page, datetime.fromisoformat("2026-11-10T23:00-08:00"))["a"]["findings"] == []
    late = goals_audit.findings(page, datetime.fromisoformat("2026-11-11T00:30-08:00"))["a"]
    assert kinds(late) == [("overdue", "a/day"), ("no-step", None)]


def test_the_command_prints_each_lead_and_exits_1_when_anything_is_found(capsys):
    assert goals.main(["audit", "--now", NOW], Page()) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "ci: 2 open, 0 off track: nothing found" and out[1] == "lean: 3 open, 1 off track"
    assert out[2].startswith("  overdue   `lean/audit` was due")
    assert out[-1] == "research goals audit: 6 finding(s) on 1 of 2 lead(s): overdue 1, unplaced 1, order 2, title 1, no-step 1"
    assert goals.main(["audit", "--now", NOW, "--owner", "ci"], Page()) == 0
    assert capsys.readouterr().out.splitlines() == ["ci: 2 open, 0 off track: nothing found",
                                                    "research goals audit: 0 finding(s) on 0 of 1 lead(s)"]


def test_owner_names_a_lead_with_no_open_goal_and_from_reads_a_saved_page(capsys, tmp_path):
    saved = tmp_path / "page.json"
    saved.write_text(json.dumps(PAGE))
    assert goals.main(["audit", "--from", str(saved), "--now", NOW, "--owner", "vllm", "--json"]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert list(doc["leads"]) == ["vllm"] and doc["counts"] == {"no-goal": 1}
    assert doc["leads"]["vllm"]["findings"][0]["text"].startswith("no open goal: the page says nothing")
    saved.write_text(json.dumps({"goals": PAGE}))
    assert goals.main(["audit", "--from", str(saved), "--now", NOW]) == 1
    assert "expected a JSON list of goals" in capsys.readouterr().err
    assert goals.main(["audit", "--now", "2026-11-10T22:00"], Page()) == 1
    assert "give its offset" in capsys.readouterr().err
