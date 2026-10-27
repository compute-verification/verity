"""`research queue stats`: a landed PR's time from its ready mark to `main`, split into queue, rework, machine, check and land."""
import json
from datetime import datetime, timezone

from research import queue_stats as qs

LOG = """\
2026-10-19T10:00:00Z
drive code 5d78b25b13 (main)
research queue drive: checking aaaaaaaaaaaa (#1 #2): r20261019-100100-0001 with agreement
2026-10-19T10:30:00Z
drive code 5d78b25b13 (main)
research queue drive: cancelled r20261019-100100-0001: aaaaaaaaaaaa is no longer on `next`
research queue drive: checking bbbbbbbbbbbb (#1 #2 #3): r20261019-103100-0002 with agreement
2026-10-19T12:00:00Z
drive code 5d78b25b13 (main)
research queue drive: landed #1 #2 #3 on `main` at bbbbbbbbbbbb (check r20261019-103100-0002)
"""


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_each_landed_pr_is_split_into_its_five_steps(tmp_path):
    """Why `main` is a bottleneck: a PR waited 11 h from its mark, under 1.5 h of it in machines and checks."""
    d = tmp_path / "r20261019-103100-0002"
    d.mkdir()
    (d / "slot.json").write_text(json.dumps({"state": "held", "took": "2026-10-19T10:40:00Z"}))
    (d / "steps.jsonl").write_text("\n".join(json.dumps(s) for s in [
        {"step": "lean-audit", "status": "passed", "ended_utc": "2026-10-19T11:40:00Z"},
        {"step": "pytest", "status": "passed", "ended_utc": "2026-10-19T13:00:00Z"},   # deferred: ended after the landing
    ]) + "\n")
    marks = {1: utc("2026-10-19T09:00:00"), 2: utc("2026-10-19T09:30:00"), 3: utc("2026-10-19T10:20:00")}
    launches, landings = qs.parse(LOG)
    assert [x.tip for x in launches] == ["aaaaaaaaaaaa", "bbbbbbbbbbbb"] and landings[0].check == "r20261019-103100-0002"
    rows = qs.rows(launches, landings, utc("2026-10-19T00:00:00"), utc("2026-10-19T23:00:00"), marks.get, lambda rid: tmp_path / rid)
    by = {r["pr"]: qs.spans(r) for r in rows}
    one = by[1]
    assert one["total"] == 3.0
    assert one["queue"] == 61 / 60                         # 09:00 -> 10:01, the first train with it
    assert round(one["rework"], 4) == round(0.5, 4)        # 10:01 -> 10:31, the train it landed with
    assert round(one["machine"], 4) == round(9 / 60, 4)    # 10:31 -> 10:40
    assert one["check"] == 1.0                             # 10:40 -> 11:40: pytest ended after the landing, so it was deferred
    assert round(one["land"], 4) == round(20 / 60, 4)
    assert by[3]["queue"] == 11 / 60 and by[3]["rework"] == 0  # marked after the first train: only the landing check had it
    assert {r["pr"]: r["trains"] for r in rows} == {1: 2, 2: 2, 3: 1}
    s = qs.summary(rows)
    assert s["prs"] == 3 and s["landings"] == 1 and s["rechecked"] == 2
    assert "3 PR(s) landed by drive in 1 landing(s)" in qs.render(s, utc("2026-10-19T00:00:00"), utc("2026-10-19T23:00:00"))


def test_a_landing_outside_the_window_or_a_missing_record_counts_as_no_data(tmp_path):
    launches, landings = qs.parse(LOG)
    assert qs.rows(launches, landings, utc("2026-10-19T12:30:00"), utc("2026-10-19T23:00:00"), {}.get, lambda r: tmp_path) == []
    rows = qs.rows(launches, landings, utc("2026-10-19T00:00:00"), utc("2026-10-19T23:00:00"), {}.get, lambda r: tmp_path / "none")
    s = qs.summary(rows)
    assert s["steps"]["machine"] == {"n": 0} and s["steps"]["total"] == {"n": 0}
    assert "no data" in qs.render(s, utc("2026-10-19T00:00:00"), utc("2026-10-19T23:00:00"))


def test_the_control_pods_log_stamps_a_landing_with_the_end_of_its_tick():
    """drive.sh on the control pod writes `<time> tick rc=N <sha>` after each tick, not a time before it: from 20 Oct `stats`
    found no tick, so no landing (22 Oct: '0 PR(s) landed' over a day of landings)."""
    pod = """\
research queue drive: checking bbbbbbbbbbbb (#1 #2 #3): r20261019-103100-0002 with agreement
2026-10-19T10:32:00Z tick rc=0 5d78b25b1
research queue drive: landed #1 #2 #3 on `main` at bbbbbbbbbbbb (check r20261019-103100-0002)
research merge: 3 `landed` label(s) on the check runs of 5d78b25b13..bbbbbbbbbbbb
2026-10-19T12:04:00Z tick rc=0 5d78b25b1
research queue drive: landed #4 on `main` at cccccccccccc (check r20261019-120000-0003)
"""
    launches, landings = qs.parse(pod)
    assert [(l.tip, sorted(l.prs)) for l in launches] == [("bbbbbbbbbbbb", [1, 2, 3])]
    assert [(l.at, l.prs, l.check) for l in landings] == [(utc("2026-10-19T12:04:00"), [1, 2, 3], "r20261019-103100-0002")]


def test_rows_read_each_landings_record_and_each_prs_mark_once_and_at_once(tmp_path):
    """22 Oct: read one at a time, a day of 30 landings and 134 PRs took over 20 min. Each is read once, from `READ_JOBS` threads,
    however many of its PRs or repeated landings name it."""
    import threading

    launches, landings = qs.parse(LOG)
    seen, lock = [], threading.Lock()

    def run_dir(rid):
        with lock:
            seen.append(rid)
        return tmp_path / rid

    def marked(pr):
        with lock:
            seen.append(pr)
        return None

    rows = qs.rows(launches, landings + landings[:1], utc("2026-10-19T00:00:00"), utc("2026-10-19T23:00:00"), marked, run_dir)
    assert sorted(map(str, seen)) == sorted(["r20261019-103100-0002", "1", "2", "3"]) and len(rows) == 6
