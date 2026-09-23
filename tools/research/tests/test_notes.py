"""`research notes checkpoint/status`: the structured line, the legacy lines lanes already write, STALE vs final."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pytest

from research import notes

NOW = dt.datetime(2026, 9, 23, 21, 0, tzinfo=dt.timezone.utc)
FRONT = "---\nlane: {lane}\nkind: report\ncreated: 2026-09-23T17:00Z\nstatus: {status}\n---\n\n"


def _report(root: Path, lane: str, body: str, minutes_ago: float, status: str = "open") -> Path:
    p = root / "lanes" / lane / f"20260923T1700Z-report-{lane}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(FRONT.format(lane=lane, status=status) + body)
    t = (NOW - dt.timedelta(minutes=minutes_ago)).timestamp()
    os.utime(p, (t, t))
    return p


def test_checkpoint_goes_under_the_front_matter_newest_first_and_sets_status(tmp_path):
    p = _report(tmp_path, "lane-a", "CHECKPOINT 1111111 (17:40Z) older\n# lane-a\n", 5)
    notes.checkpoint(tmp_path, "lane-a", "blocked", "waiting   on\nthe pod", commit="abc1234", now=NOW)
    os.utime(p, (NOW.timestamp(), NOW.timestamp()))
    text = p.read_text()
    front, body = text.split("\n---\n", 1)
    assert "status: blocked" in front
    assert body.splitlines()[1:3] == ["CHECKPOINT abc1234 (21:00Z) [blocked] waiting on the pod", "CHECKPOINT 1111111 (17:40Z) older"]
    row = notes.lane_status(p, NOW, 45)
    assert (row["state"], row["commit"], row["checkpoint_at"], row["text"]) == ("blocked", "abc1234", "21:00Z", "waiting on the pod")


def test_checkpoint_creates_the_report_for_a_new_lane(tmp_path):
    p = notes.checkpoint(tmp_path, "fresh", "open", "started", commit="none", now=NOW)
    assert p.name == "20260923T2100Z-report-fresh.md"
    assert p.read_text() == FRONT.format(lane="fresh", status="open").replace("2026-09-23T17:00Z", "2026-09-23T21:00Z") + \
        "CHECKPOINT none (21:00Z) [open] started\n"
    with pytest.raises(ValueError):
        notes.checkpoint(tmp_path, "fresh", "done", "x")


@pytest.mark.parametrize("line,commit,at,text", [
    ("CHECKPOINT fc49219 (18:50Z) Rust verifier verifies the pair", "fc49219", "18:50Z", "Rust verifier verifies the pair"),
    ("CHECKPOINT none (17:10Z) — brief + code read", "none", "17:10Z", "brief + code read"),
    ("CHECKPOINT 71dbab04b8b3c1720c3ce087286ba13a5b002df5 18:51Z: bbr default", "71dbab04b8b3c1720c3ce087286ba13a5b002df5", "18:51Z", "bbr default"),
    ("CHECKPOINT 82f3a6e 20:20Z — reference RTX 4090 numbers", "82f3a6e", "20:20Z", "reference RTX 4090 numbers"),
    ("(CHECKPOINT ff9d4c3): accept, 2^-128.026, 6 ms.", "ff9d4c3", None, "accept, 2^-128.026, 6 ms."),
    ("CHECKPOINT d35a52f D0 LANDED (17:52Z) — rebase", "d35a52f", "17:52Z", "D0 LANDED (17:52Z) — rebase"),
])
def test_legacy_checkpoint_lines_parse(tmp_path, line, commit, at, text):
    row = notes.lane_status(_report(tmp_path, "l", line + "\n", 10), NOW, 45)
    assert (row["commit"], row["checkpoint_at"], row["text"]) == (commit, at, text)


def test_newest_checkpoint_wins_wherever_it_sits_and_times_before_midnight_are_yesterday(tmp_path):
    body = "CHECKPOINT aaaaaaa (18:00Z) first\n...\nCHECKPOINT bbbbbbb (19:30Z) appended at the bottom\n"
    assert notes.lane_status(_report(tmp_path, "l", body, 30), NOW, 45)["commit"] == "bbbbbbb"
    late = _report(tmp_path, "m", "CHECKPOINT ccccccc (23:50Z) last night\nCHECKPOINT ddddddd (20:10Z) stamped ahead\n", 60)
    assert notes.lane_status(late, NOW, 45)["commit"] == "ddddddd"     # mtime 20:00Z: 20:10Z is today, 23:50Z the day before


def test_stale_means_not_final_and_silent_past_the_threshold(tmp_path):
    _report(tmp_path, "dead", "CHECKPOINT 1234567 (19:00Z) bench running\n", 120)
    _report(tmp_path, "alive", "CHECKPOINT 2345678 (20:50Z) measuring\n", 10)
    _report(tmp_path, "done-section", "CHECKPOINT 3456789 (18:00Z) x\n## FINAL\nnumbers here\n", 180)
    _report(tmp_path, "done-status", "text\n", 300, status="done")
    _report(tmp_path, "done-line", "CHECKPOINT 5d88d70 FINAL (14:50Z; mergeable: yes)\n", 300)
    _report(tmp_path, "placeholder", "CHECKPOINT 4567890 (18:45Z) rebased\n## FINAL\n(pending)\n", 130)
    rows = {r["lane"]: r for r in notes.status(tmp_path, 45, None, now=NOW)}
    assert {k for k, r in rows.items() if r["stale"]} == {"dead", "placeholder"}
    assert {k for k, r in rows.items() if r["state"] == "final"} == {"done-section", "done-status", "done-line"}
    assert [r["lane"] for r in notes.status(tmp_path, 45, None, now=NOW)][:2] == ["placeholder", "dead"]   # stale first, then oldest
    assert [r["lane"] for r in notes.status(tmp_path, 45, 1.0, now=NOW)] == ["alive"]
    assert {r["lane"] for r in notes.status(tmp_path, 45, None, now=NOW, lanes=["d*"], exclude=["done-*"])} == {"dead"}


def test_cli_status_exits_1_when_a_lane_is_stale(tmp_path, capsys):
    _report(tmp_path, "dead", "CHECKPOINT 1234567 (19:00Z) bench running\n", 10_000)
    assert notes.main(["status", "--root", str(tmp_path), "--all"]) == 1
    assert "STALE" in capsys.readouterr().out
    assert notes.main(["status", "--root", str(tmp_path)]) == 0     # outside the default 24 h window
