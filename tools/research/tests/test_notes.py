"""`research notes checkpoint/status`: the structured line, the legacy lines lanes already write, STALE vs final."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
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


# ---- signs of life beyond the report --------------------------------------------------------------------------------------------------------

def _git(repo: Path, *args: str, when: dt.datetime | None = None) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    if when is not None:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = when.isoformat()
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout


def _touch(p: Path, minutes_ago: float, text: str = "x\n") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    t = (NOW - dt.timedelta(minutes=minutes_ago)).timestamp()
    os.utime(p, (t, t))
    return p


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _touch(r / "a.py", 500)
    _git(r, "add", "a.py")
    _git(r, "commit", "-qm", "base", when=NOW - dt.timedelta(minutes=500))
    return r


def test_uncommitted_edits_and_commits_keep_a_lane_alive_when_its_report_is_old(tmp_path, repo):
    root = tmp_path / "notes"
    _git(repo, "checkout", "-qb", "lane/busy")
    _touch(repo / "b.py", 50)
    _git(repo, "add", "b.py")
    _git(repo, "commit", "-qm", "step", when=NOW - dt.timedelta(minutes=50))
    _touch(repo / "a.py", 5, "edited\n")                 # modified, not committed
    _touch(repo / "new_test.py", 12)                      # untracked
    _report(root, "busy", "CHECKPOINT 1111111 (19:40Z) coding\n", 80)
    row = notes.status(root, 30, None, now=NOW, repo=repo)[0]
    assert (row["branch"], row["worktree"]) == ("lane/busy", str(repo))
    assert (row["life_via"], row["life_min"], row["commit_min"], row["edit_min"], row["dirty"]) == ("edit", 5.0, 50.0, 5.0, 2)
    assert not row["stale"]


def test_a_lane_that_died_mid_edit_is_stale_and_says_how_much_it_would_lose(tmp_path, repo):
    root = tmp_path / "notes"
    _git(repo, "checkout", "-qb", "lane/dead")
    _touch(repo / "a.py", 40, "half done\n")
    _git(repo, "mv", "a.py", "moved.py")                  # a staged rename is one dirty path, and its mtime still counts
    _report(root, "dead", "CHECKPOINT 2222222 (20:25Z) merging\n", 35)
    row = notes.status(root, 30, None, now=NOW, repo=repo)[0]
    assert row["stale"] and row["dirty"] == 1
    assert (row["life_via"], row["life_min"]) == ("report", 35.0)


def test_binding_names_a_predecessors_branch_and_a_lane_with_no_report_still_shows(tmp_path, repo):
    root = tmp_path / "notes"
    _git(repo, "checkout", "-qb", "lane/old-name")
    _touch(repo / "wip.py", 3)
    notes.bind(root, "successor", branch="lane/old-name", worktree=str(repo), pod="vy-old-name")
    notes.bind(root, "successor", pod="")                 # empty clears a field
    assert json.loads((root / "lanes" / "successor" / "binding.json").read_text()) == {"branch": "lane/old-name", "worktree": str(repo)}
    os.utime(root / "lanes" / "successor" / "binding.json", (NOW.timestamp() - 600, NOW.timestamp() - 600))
    row = notes.status(root, 30, None, now=NOW, repo=repo)[0]
    assert (row["lane"], row["text"], row["branch"], row["pod"]) == ("successor", "(no report yet)", "lane/old-name", "vy-successor-veritor-campaign")
    assert (row["life_via"], row["life_min"], row["stale"]) == ("edit", 3.0, False)


def test_superseded_lanes_are_done_however_old(tmp_path):
    p = _report(tmp_path, "pred", "CHECKPOINT 1234567 (15:00Z) running\n", 400)
    notes.checkpoint(tmp_path, "pred", "superseded", "by pred-2", commit="none", now=NOW - dt.timedelta(minutes=300))
    os.utime(p, (NOW.timestamp() - 18000, NOW.timestamp() - 18000))
    row = notes.status(tmp_path, 30, None, now=NOW)[0]
    assert (row["state"], row["stale"]) == ("superseded", False)
    assert "supersd" in notes.render([row])


def test_pod_probe_parses_processes_gpu_and_newest_file():
    out = "GPU 87, 17928\nPROC 25\nPROC 4\nPROC 3120\nFILE 1727134000.5 1727134600\n"
    assert notes.parse_pod_probe(out) == {"busy": 3, "newest_proc_s": 4, "gpu_util": 87, "gpu_mem_mb": 17928, "file_age_s": 600}
    assert notes.parse_pod_probe("GPU none\nFILE none 1727134600\n") == \
        {"busy": 0, "newest_proc_s": None, "gpu_util": None, "gpu_mem_mb": None, "file_age_s": None}


def test_a_busy_pod_counts_as_now_and_an_idle_one_by_its_newest_file(tmp_path):
    _report(tmp_path, "waiting", "CHECKPOINT 1111111 (20:00Z) campaign running\n", 60)
    _report(tmp_path, "idle", "CHECKPOINT 2222222 (20:10Z) bench\n", 50)
    probes = {"waiting": {"name": "vy-waiting", "busy": 2, "gpu_util": 90, "file_age_s": 30},
              "idle": {"name": "vy-idle", "busy": 0, "gpu_util": 0, "file_age_s": 2400}}
    rows = {r["lane"]: r for r in notes.status(tmp_path, 30, None, now=NOW, pods=lambda names: {k: probes[k] for k in names})}
    assert (rows["waiting"]["life_via"], rows["waiting"]["life_min"], rows["waiting"]["stale"]) == ("pod", 0.0, False)
    assert (rows["idle"]["life_via"], rows["idle"]["life_min"], rows["idle"]["stale"]) == ("pod", 40.0, True)
    text = notes.render(list(rows.values()))
    assert "busy2 90%" in text and "idle 0%" in text


def test_watch_prints_each_transition_once(tmp_path):
    rows = [{"lane": "a", "state": "open", "stale": True, "age_min": 50, "life_min": 50, "life_via": "edit", "dirty": 6},
            {"lane": "b", "state": "open", "stale": False, "age_min": 5}]
    lines, prev = notes.transitions(set(), rows, NOW)
    assert lines == ["21:00Z STALE a life=50m via=edit dirty=6 pod=- branch=-"] and prev == {"a"}
    assert notes.transitions(prev, rows, NOW) == ([], {"a"})
    rows[0].update(stale=False, state="superseded")
    assert notes.transitions(prev, rows, NOW) == (["21:00Z ALIVE a (superseded)"], set())


# ---- the inbox and the finish checks --------------------------------------------------------------------------------------------------------

def test_the_inbox_is_handoffs_changed_since_last_shown_by_mtime_not_by_the_name_stamp(tmp_path):
    _report(tmp_path, "lane-a", "CHECKPOINT 1111111 (20:00Z) [open] x\n", 60)
    d = tmp_path / "lanes" / "lane-a"
    _touch(d / "20260923T1900Z-handoff-coordinator.md", 30, "---\nfrom: coordinator\n---\n\n# coordinator -> lane-a: fix H1\n")
    _touch(d / "20260923T2050Z-asks-from-verify.md", 10, "# verify asks\n")
    items = notes.inbox(tmp_path, "lane-a")
    assert [(x["path"].name, x["title"]) for x in items] == [("20260923T1900Z-handoff-coordinator.md", "coordinator -> lane-a: fix H1"),
                                                             ("20260923T2050Z-asks-from-verify.md", "verify asks")]
    notes.mark_seen(tmp_path, "lane-a", items)
    assert notes.inbox(tmp_path, "lane-a") == []
    _touch(d / "20260923T1900Z-handoff-coordinator.md", 1, "# coordinator -> lane-a: fix H1 (and H2)\n")   # edited in place
    again = notes.inbox(tmp_path, "lane-a")
    assert [x["path"].name for x in again] == ["20260923T1900Z-handoff-coordinator.md"]
    assert notes.render_inbox("lane-a", again).splitlines() == [
        "INBOX lane-a: 1 new -- act on each, or say in your next checkpoint why not",
        '  20:59Z 20260923T1900Z-handoff-coordinator.md  "coordinator -> lane-a: fix H1 (and H2)"']


def test_a_successor_gets_its_predecessors_unread_handoffs_and_they_are_not_mail(tmp_path):
    _report(tmp_path, "live-2b", "CHECKPOINT 1111111 (20:00Z) [superseded] by live-2c (coordinator)\n", 60, status="superseded")
    _touch(tmp_path / "lanes" / "live-2b" / "20260923T2030Z-handoff-from-relation.md", 20, "# relation -> live-2b: may I run?\n")
    _report(tmp_path, "live-2c", "CHECKPOINT 2222222 (20:50Z) [open] x\n", 10)
    assert notes.successors(tmp_path) == {"live-2b": "live-2c"}
    assert [(x["lane_dir"], x["path"].name) for x in notes.inbox(tmp_path, "live-2c")] == [
        ("live-2b", "20260923T2030Z-handoff-from-relation.md")]
    rows = {r["lane"]: r for r in notes.status(tmp_path, 30, None, now=NOW)}
    assert (rows["live-2b"]["mail"], rows["live-2c"]["inbox"], rows["live-2c"]["inbox_oldest_min"]) == ([], 1, 20.0)
    notes.bind(tmp_path, "live-2d", succeeds="live-2c")
    assert notes.chain(tmp_path, "live-2d") == ["live-2d", "live-2c", "live-2b"]


def test_handoffs_to_a_finished_lane_with_no_successor_are_mail_and_watch_says_so_once(tmp_path):
    _report(tmp_path, "blake3", "CHECKPOINT 1111111 (20:00Z) [final] done\n", 55, status="final")
    _touch(tmp_path / "lanes" / "blake3" / "20260923T1900Z-handoff-coordinator.md", 120, "# read before FINAL\n")
    _touch(tmp_path / "lanes" / "blake3" / "20260923T2058Z-handoff-coordinator-H2.md", 2, "# H2\n")
    rows = notes.status(tmp_path, 30, None, now=NOW)
    assert rows[0]["mail"] == ["20260923T2058Z-handoff-coordinator-H2.md"] and "MAIL 1" in notes.render(rows)
    lines, prev = notes.mail_lines({}, rows, NOW)
    assert lines == ["21:00Z MAIL blake3 got 1 handoff(s) after it finished: 20260923T2058Z-handoff-coordinator-H2.md"]
    assert notes.mail_lines(prev, rows, NOW) == ([], prev)


def test_cli_checkpoint_prints_the_inbox_once_and_superseded_leaves_it_for_the_successor(tmp_path, capsys):
    _report(tmp_path, "a", "", 60)
    _touch(tmp_path / "lanes" / "a" / "20260923T2050Z-handoff-coordinator.md", 10, "# do X\n")
    assert notes.main(["checkpoint", "a", "open", "step", "--commit", "abc1234", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "INBOX a: 1 new" in out and '"do X"' in out
    notes.main(["checkpoint", "a", "open", "step 2", "--commit", "abc1234", "--root", str(tmp_path)])
    assert "INBOX a: nothing new" in capsys.readouterr().out
    _report(tmp_path, "b", "", 60)
    _touch(tmp_path / "lanes" / "b" / "20260923T2055Z-handoff-coordinator.md", 5, "# do Y\n")
    assert notes.main(["checkpoint", "b", "superseded", "by", "b2", "--commit", "none", "--root", str(tmp_path)]) == 0
    assert "INBOX" not in capsys.readouterr().out
    assert [x["path"].name for x in notes.inbox(tmp_path, "b2")] == ["20260923T2055Z-handoff-coordinator.md"]


def test_finish_checks_custody_pod_worktree_and_handoffs_never_named(tmp_path, repo):
    root = tmp_path / "notes"
    _git(repo, "checkout", "-qb", "lane/fin")
    p = _report(root, "fin", "CHECKPOINT 1111111 (20:55Z) [final] done\n## FINAL\nart:1234abcd, art:feedbeef00 and art:1234abcd again;"
                " answered 20260923T2000Z-handoff-a.md\n", 5, status="final")
    _touch(root / "lanes" / "fin" / "20260923T2000Z-handoff-a.md", 60, "# a\n")
    _touch(root / "lanes" / "fin" / "20260923T2040Z-handoff-b.md", 20, "# b\n")
    cited = []

    def custody(arts):
        cited.append(arts)
        return False, "1/2 cited artifacts not preserved: art:feedbeef00"

    text, rc = notes.finish(root, "fin", repo, custody=custody, pods=lambda lane, bound, lanes: (False, "vy-fin RUNNING"))
    head = _git(repo, "rev-parse", "HEAD")[:8]
    assert (rc, cited) == (3, [["1234abcd", "feedbeef00"]])
    assert text.splitlines() == [
        "FINISH fin: 3 problem(s) (FINAL line written; exit 3)",
        "  custody   1/2 cited artifacts not preserved: art:feedbeef00",
        "  pod       vy-fin RUNNING",
        f"  worktree  clean, HEAD {head} == lane/fin {head}",
        "  handoffs  1 never named in the report (answer or say why not): 20260923T2040Z-handoff-b.md"]
    p.write_text(p.read_text() + "b: not doing it, out of scope (20260923T2040Z-handoff-b.md)\n")
    _touch(repo / "a.py", 1, "dirty\n")
    text, rc = notes.finish(root, "fin", repo, keep_pod="the shared live verifier", custody=lambda arts: (True, "2 preserved"))
    assert rc == 3 and "  pod       kept: the shared live verifier" in text and "1 uncommitted files" in text
    _git(repo, "checkout", "-q", "a.py")
    assert notes.finish(root, "fin", repo, keep_pod="x", custody=lambda arts: (True, "ok"))[1] == 0


def test_a_pod_belongs_to_the_longest_lane_name_it_starts_with():
    lanes = {"fp4-decode", "fp4-decode-2", "live-2", "live-2c"}
    assert notes.pod_owner("vy-fp4-decode-2-5090b-veritor-campaign", lanes) == "fp4-decode-2"
    assert notes.pod_owner("vy-fp4-decode-veritor-campaign", lanes) == "fp4-decode"
    assert notes.pod_owner("vy-live-2c-veritor-campaign", lanes) == "live-2c"
    assert notes.pod_owner("vy-live-2-verifier", lanes) == "live-2"
    assert notes.pod_owner("vy-control", lanes) is None
