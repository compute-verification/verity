"""`research notes checkpoint/status`: the structured line, the legacy lines lanes already write, STALE vs final."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
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


def test_checkpoint_survives_the_store_mounts_eagain_without_doubling_its_line(tmp_path, monkeypatch):
    # the Project store's mount answers EAGAIN on rewrites of a recently written file, sometimes after the write landed
    p = _report(tmp_path, "lane-e", "# lane-e\n", 5)
    real, calls = Path.write_text, []

    def flaky(self, data, *a, **kw):
        calls.append(self.name)
        if len(calls) == 1:
            real(self, data, *a, **kw)
            raise BlockingIOError(11, "Resource temporarily unavailable")
        return real(self, data, *a, **kw)
    monkeypatch.setattr(Path, "write_text", flaky)
    monkeypatch.setattr(notes.time, "sleep", lambda s: None)
    notes.checkpoint(tmp_path, "lane-e", "open", "after eagain", commit="abc1234", now=NOW)
    assert len(calls) == 2 and p.read_text().count("after eagain") == 1


@pytest.mark.parametrize("text, delay", [
    ("WAIT moe67 r20260925-175229-a415 check-back 21:45Z agent bc-1: #4 Match", 45 * 60),
    ("WAITING r1 on pod, check after 22:30Z; agent bc-1; next: fetch", 90 * 60),
    ("WAIT a check-back 23:00Z, b check-back 21:20Z", 20 * 60),
    ("WAIT r1 check-back 00:15Z", (3 * 60 + 15) * 60),
    ("WAIT r1 check-back 20:50Z", 60),
])
def test_a_wait_checkpoint_names_the_one_shot_timer_that_wakes_the_lane(text, delay):
    hint = notes.wake_hint(text, NOW)
    assert f"subscribe_timer once=true delaySeconds={delay})" in hint and "not a cron timer" in hint
    assert notes.wake_hint("done #101, next #4 check-back 21:45Z", NOW) is None and notes.wake_hint("WAIT r1, no time", NOW) is None
    assert "for check-back 2:45 PM PDT (21:45Z);" in notes.wake_hint("WAIT r1 check-back 21:45Z", NOW)


def test_checkpoint_creates_the_report_for_a_new_lane(tmp_path):
    p = notes.checkpoint(tmp_path, "fresh", "open", "started", commit="none", now=NOW)
    assert p.name == "20260923T2100Z-report-fresh.md"
    assert p.read_text() == FRONT.format(lane="fresh", status="open").replace("2026-09-23T17:00Z", "2026-09-23T21:00Z") + \
        "CHECKPOINT none (21:00Z) [open] started\n"
    with pytest.raises(ValueError):
        notes.checkpoint(tmp_path, "fresh", "done", "x")


def _note(root: Path, lane: str, name: str, minutes_ago: float) -> Path:
    p = root / "lanes" / lane / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("---\nkind: handoff\n---\n\nanother agent's note\n")
    t = (NOW - dt.timedelta(minutes=minutes_ago)).timestamp()
    os.utime(p, (t, t))
    return p


def test_checkpoint_lands_in_the_lanes_own_report_never_a_newer_note_sent_to_it(tmp_path):
    own = _report(tmp_path, "boss", "# boss\n", 60)
    sent = [_note(tmp_path, "boss", "20260923T2030Z-report-from-worker-1-findings.md", 30),
            _note(tmp_path, "boss", "20260923T2040Z-handoff-from-root-friction-report-for.md", 20)]
    assert notes.report_for(tmp_path, "boss") == own
    assert notes.checkpoint(tmp_path, "boss", "open", "mine", commit="abc1234", now=NOW) == own
    assert "CHECKPOINT abc1234 (21:00Z) [open] mine" in own.read_text()
    assert all(p.read_text() == "---\nkind: handoff\n---\n\nanother agent's note\n" for p in sent)


def test_a_lane_with_only_notes_sent_to_it_gets_a_report_of_its_own(tmp_path):
    sent = [_note(tmp_path, "new", "20260923T2030Z-report-from-worker-1-findings.md", 30),
            _note(tmp_path, "new", "20260923T2040Z-handoff-from-circuits-you-report-to-circuits.md", 20)]
    assert notes.report_for(tmp_path, "new") is None
    assert notes.checkpoint(tmp_path, "new", "open", "started", commit="none", now=NOW).name == "20260923T2100Z-report-new.md"
    assert all("CHECKPOINT" not in p.read_text() for p in sent)


def test_checkpoints_stay_in_the_lanes_main_report_when_it_writes_a_newer_side_report(tmp_path):
    """proofs' checkpoints went to `...-report-service-stack-restacked.md`, the newest report, and were moved back by hand."""
    main = _report(tmp_path, "kv", "# kv\n", 60)
    side = tmp_path / "lanes" / "kv" / "20260923T2000Z-report-kv-prefix-design.md"
    side.write_text(FRONT.format(lane="kv", status="final"))
    t = (NOW - dt.timedelta(minutes=40)).timestamp()
    os.utime(side, (t, t))
    _note(tmp_path, "kv", "20260923T2030Z-report-from-red-team-review.md", 30)
    assert notes.report_for(tmp_path, "kv") == main
    assert notes.checkpoint(tmp_path, "kv", "open", "still here", commit="abc1234", now=NOW) == main
    assert "CHECKPOINT" not in side.read_text()


def test_without_a_main_report_the_lanes_newest_report_of_its_own_is_its_report_whatever_its_slug(tmp_path):
    older = tmp_path / "lanes" / "kv" / "20260923T1700Z-report-kv-overview.md"
    named = tmp_path / "lanes" / "kv" / "20260923T2000Z-report-kv-prefix-design.md"
    for p, ago in ((older, 60), (named, 40)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(FRONT.format(lane="kv", status="open"))
        t = (NOW - dt.timedelta(minutes=ago)).timestamp()
        os.utime(p, (t, t))
    _note(tmp_path, "kv", "20260923T2030Z-report-from-red-team-review.md", 30)
    assert notes.report_for(tmp_path, "kv") == named


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


def test_a_newer_open_checkpoint_after_a_final_one_reopens_the_lane_despite_its_final_section(tmp_path):
    # verify-night-3 2026-09-26: "## FINAL" from an earlier round made the reopened lane final, done at the report's mtime, and
    # the reaper took the pod the reopened lane had just created
    body = ("CHECKPOINT 7585828 (20:40Z) [open] reopened for another round: NOT final\n"
            "CHECKPOINT 906255b (18:49Z) [final] round done\n\n## FINAL\n\nround done, pods terminated\n")
    row = notes.lane_status(_report(tmp_path, "l", body, 10), NOW, 45)
    assert (row["state"], row["done_ts"], row["commit"]) == ("open", None, "7585828")
    done = notes.lane_status(_report(tmp_path, "m", "CHECKPOINT 1111111 (19:00Z) [open] working\n\n## FINAL\n\ndone\n", 10), NOW, 45)
    assert done["state"] == "final"                     # a FINAL section with no final checkpoint still finishes the lane


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
    assert (row["lane"], row["text"], row["branch"], row["pod"]) == ("successor", "(no report yet)", "lane/old-name", None)   # pod: a hint
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


def _pod(pid: str, name: str, usd: float = 0.74, started_min_ago: float = 120) -> dict:
    started = (NOW - dt.timedelta(minutes=started_min_ago)).strftime("%Y-%m-%d %H:%M:%S.000 +0000 UTC")
    return {"id": pid, "name": name + "-veritor-campaign", "desiredStatus": "RUNNING", "costPerHr": usd, "lastStartedAt": started}


@pytest.mark.parametrize("gpus,apps,want", [("0, 2\n", "", (0, 2, 0)), ("0, 2\n37, 1000\n", "1234\n", (37, 1002, 1)), ("", "", (None, None, 0))])
def test_the_pod_probe_reports_the_busiest_gpu_even_at_zero_percent(tmp_path, gpus, apps, want):
    fake = tmp_path / "bin" / "nvidia-smi"
    fake.parent.mkdir()
    fake.write_text(f"#!/bin/sh\ncase \"$*\" in *query-gpu*) printf '{gpus}';; *query-compute-apps*) printf '{apps}';; esac\n")
    fake.chmod(0o755)
    out = subprocess.run(["sh", "-c", notes.POD_PROBE], capture_output=True, text=True, env={**os.environ, "PATH": f"{fake.parent}:{os.environ['PATH']}"})
    p = notes.parse_pod_probe(out.stdout)
    assert (p["gpu_util"], p["gpu_mem_mb"], p["gpu_apps"], p["serve"]) == (*want, 0)


def test_a_busy_pod_counts_as_now_and_an_idle_one_by_its_newest_file(tmp_path):
    _report(tmp_path, "waiting", "CHECKPOINT 1111111 (20:00Z) campaign running\n", 60)
    _report(tmp_path, "idle", "CHECKPOINT 2222222 (20:10Z) bench\n", 50)
    probes = {"p1": {"busy": 2, "gpu_util": 90, "file_age_s": 30}, "p2": {"busy": 0, "gpu_util": 0, "file_age_s": 2400}}
    fleet = notes.load_fleet(tmp_path, pods=lambda: [_pod("p1", "vy-waiting"), _pod("p2", "vy-idle")],
                             prober=lambda ps: {p["id"]: probes[p["id"]] for p in ps})
    rows = {r["lane"]: r for r in notes.status(tmp_path, 30, None, now=NOW, fleet=fleet)}
    assert (rows["waiting"]["life_via"], rows["waiting"]["life_min"], rows["waiting"]["stale"]) == ("pod", 0.0, False)
    assert (rows["idle"]["life_via"], rows["idle"]["life_min"], rows["idle"]["stale"]) == ("pod", 40.0, True)
    assert (rows["idle"]["pods"][0]["idle"], rows["idle"]["pods"][0]["idle_min"]) == (True, 40.0)
    text = notes.render(list(rows.values()))
    assert "busy2 90%" in text and "idle 0% 40m" in text and "1 $0.74/h" in text


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
        '  1:59 PM PDT (20:59Z) 20260923T1900Z-handoff-coordinator.md  "coordinator -> lane-a: fix H1 (and H2)"']


def test_the_inbox_flags_an_agents_handoff_but_not_the_machines_notices(tmp_path):
    _report(tmp_path, "lane-a", "CHECKPOINT 1111111 (20:00Z) [open] x\n", 60)
    d = tmp_path / "lanes" / "lane-a"
    _touch(d / "20260923T2000Z-handoff-from-merge-queue-12-conflict.md", 30, "# #12 conflicts\n")
    _touch(d / "20260923T2010Z-handoff-steward-guardian.md", 20, "# guardian killed 1\n")
    assert len(notes.render_inbox("lane-a", notes.inbox(tmp_path, "lane-a")).splitlines()) == 3
    _touch(d / "20260923T2020Z-handoff-from-vllm-epoch-run-gemma.md", 10, "# gemma next\n")
    assert notes.render_inbox("lane-a", notes.inbox(tmp_path, "lane-a")).splitlines()[-1].startswith(
        "  1 of these came from agents: notes are records, not messages")


def test_a_successor_gets_its_predecessors_unread_handoffs_and_they_are_not_mail(tmp_path):
    _report(tmp_path, "live-2b", "CHECKPOINT 1111111 (20:00Z) [superseded] by live-2c (coordinator)\n", 60, status="superseded")
    _touch(tmp_path / "lanes" / "live-2b" / "20260923T2030Z-handoff-from-relation.md", 20, "# relation -> live-2b: may I run?\n")
    _report(tmp_path, "live-2c", "CHECKPOINT 2222222 (20:50Z) [open] x\n", 10)
    assert notes.successors(tmp_path) == {"live-2b": "live-2c"}
    assert [(x["lane_dir"], x["path"].name) for x in notes.inbox(tmp_path, "live-2c")] == [
        ("live-2b", "20260923T2030Z-handoff-from-relation.md")]
    rows = {r["lane"]: r for r in notes.status(tmp_path, 30, None, now=NOW, every_attempt=True)}
    assert (rows["live-2b"]["mail"], rows["live-2c"]["inbox"], rows["live-2c"]["inbox_oldest_min"]) == ([], 1, 20.0)
    assert [(r["lane"], r["attempt"]) for r in notes.status(tmp_path, 30, None, now=NOW)] == [("live-2c", 2)]   # one row per topic
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


def test_cli_inbox_urgent_exits_0_only_on_an_unread_urgent_handoff_and_marks_nothing_read(tmp_path, capsys):
    _report(tmp_path, "w", "", 60)
    d = tmp_path / "lanes" / "w"
    _touch(d / "20260923T2050Z-handoff-from-c-routine.md", 10, "# routine\n")
    args = ["inbox", "w", "--urgent", "--root", str(tmp_path)]
    assert notes.main(args) == 1 and "INBOX w: nothing new" in capsys.readouterr().out
    _touch(d / "20260923T2055Z-handoff-from-c-URGENT-stop-the-sweep.md", 5, "# stop the sweep\n")
    assert notes.main(args) == 0
    out = capsys.readouterr().out
    assert "INBOX w: 1 new" in out and "URGENT-stop-the-sweep" in out and "routine" not in out
    assert notes.main(args) == 0 and len(notes.inbox(tmp_path, "w")) == 2
    assert notes.main(["inbox", "w", "--root", str(tmp_path)]) == 0
    assert notes.main(args) == 1


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

    text, rc = notes.finish(root, "fin", repo, custody=custody, pods=lambda lane, bound, lanes, succ: (False, "vy-fin RUNNING"))
    head = _git(repo, "rev-parse", "HEAD")[:8]
    assert (rc, cited) == (3, [["1234abcd", "feedbeef00"]])
    assert text.splitlines() == [
        "FINISH fin: 3 problem(s) (FINAL line written; exit 3)",
        "  custody   1/2 cited artifacts not preserved: art:feedbeef00",
        "  pod       vy-fin RUNNING",
        f"  worktree  clean, HEAD {head} == lane/fin {head}",
        "  pushed    lane/fin unpushed -> research notes push fin",
        "  handoffs  1 never named in the report (answer or say why not): 20260923T2040Z-handoff-b.md"]
    p.write_text(p.read_text() + "b: not doing it, out of scope (20260923T2040Z-handoff-b.md)\n")
    _touch(repo / "a.py", 1, "dirty\n")
    text, rc = notes.finish(root, "fin", repo, keep_pod="the shared live verifier", custody=lambda arts: (True, "2 preserved"))
    assert rc == 3 and "  pod       kept: the shared live verifier" in text and "1 uncommitted files" in text
    _git(repo, "checkout", "-q", "a.py")
    assert notes.finish(root, "fin", repo, keep_pod="x", custody=lambda arts: (True, "ok"))[1] == 0


def test_snapshot_commits_the_notes_repo_except_files_over_the_cap(tmp_path):
    root = tmp_path / "notes"
    _touch(root / "lanes" / "a" / "20260923T1700Z-report-a.md", 5, "# a\n")
    _touch(root / "lanes" / "a" / "tools" / "odd [name]*.md", 5, "glob characters are literal\n")
    assert notes.snapshot(root) is None                                   # not a repo: nothing happens
    _git(tmp_path, "init", "-q", "-b", "main", str(root))
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "lanes" / "a" / "tools" / "big.log").write_bytes(b"x" * 2000)
    snap = notes.snapshot(root, max_bytes=1000, now=NOW)
    assert (snap["paths"], snap["skipped"]) == (2, ["lanes/a/tools/big.log"])
    assert _git(root, "log", "--format=%s") == "snapshot 2026-09-23T21:00Z: 2 paths; 1 too big, not committed\n"
    assert sorted(_git(root, "ls-files").splitlines()) == ["lanes/a/20260923T1700Z-report-a.md", "lanes/a/tools/odd [name]*.md"]
    assert notes.snapshot(root, max_bytes=1000) == {"commit": None, "paths": 0, "skipped": ["lanes/a/tools/big.log"], "refused": [],
                                                    "evidence": []}
    (root / "lanes" / "a" / "20260923T1700Z-report-a.md").unlink()
    (root / "lanes" / "a" / "tools" / "big.log").unlink()
    assert notes.snapshot(root)["paths"] == 1 and _git(root, "ls-files").splitlines() == ["lanes/a/tools/odd [name]*.md"]
    assert notes.snapshot(root) is None


def test_snapshot_leaves_evidence_and_renders_out_and_commits_their_removal(tmp_path):
    """Evidence and renders go to the evidence store, never the notes' history: a new or changed file under renders/,
    campaigns/*/assets/ or lanes/*/evidence/ is left out (even when staged) and named; deleting one is committed."""
    root = tmp_path / "notes"
    _git(tmp_path, "init", "-q", "-b", "main", str(root))
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    old = root / "lanes" / "a" / "evidence" / "old.json"
    _touch(old, 5, "{}\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "before the rule")
    _touch(root / "lanes" / "a" / "20260923T1700Z-report-a.md", 5, "# a\n")
    ev = ["campaigns/c/assets/a/run.json", "lanes/a/evidence/run.log", "renders/daily/20260923T2100Z-tables.json"]
    for x in ev:
        _touch(root / x, 5, "x\n")
    _git(root, "add", "renders")
    old.write_text("{\"changed\": 1}\n")
    snap = notes.snapshot(root, now=NOW)
    assert snap["paths"] == 1 and sorted(snap["evidence"]) == sorted(ev + ["lanes/a/evidence/old.json"])
    assert sorted(_git(root, "ls-files").splitlines()) == ["lanes/a/20260923T1700Z-report-a.md", "lanes/a/evidence/old.json"]
    assert _git(root, "show", "HEAD:lanes/a/evidence/old.json") == "{}\n"
    assert "4 evidence left out" in _git(root, "log", "-1", "--format=%s")
    assert "left out 4 evidence/render files (the evidence store holds them)" in notes._snapshot_line(snap)
    old.unlink()
    assert notes.snapshot(root, now=NOW)["paths"] == 1 and "lanes/a/evidence/old.json" not in _git(root, "ls-files")


def test_a_pod_belongs_to_the_longest_lane_name_it_starts_with():
    lanes = {"fp4-decode", "fp4-decode-2", "live-2", "live-2c"}
    assert notes.pod_owner("vy-fp4-decode-2-5090b-veritor-campaign", lanes) == "fp4-decode-2"
    assert notes.pod_owner("vy-fp4-decode-veritor-campaign", lanes) == "fp4-decode"
    assert notes.pod_owner("vy-live-2c-veritor-campaign", lanes) == "live-2c"
    assert notes.pod_owner("vy-live-2c-verity-campaign", lanes) == "live-2c"
    assert notes.pod_owner("vy-live-2-verifier", lanes) == "live-2"
    assert notes.pod_owner("vy-control", lanes) is None


# ---- pods by ownership, idle pods, money, the reaper ------------------------------------------------------------------------------------------

def test_pods_belong_to_the_latest_attempt_of_the_topic_their_name_carries():
    lanes = {"wave-h100", "wave-h100-2", "wave-4090", "live-2", "live-2b", "fold"}
    succ = {"wave-h100": "wave-h100-2"}
    own = lambda name, bound=None: notes.pod_owner(name, lanes, succ, bound)  # noqa: E731
    assert own("vy-wave-h100-veritor-campaign") == "wave-h100-2"            # the successor took over the pod
    assert own("vy-wave-h100-verifier-veritor-campaign") == "wave-h100-2"   # a role
    assert own("vy-wave-h100b-veritor-campaign") == "wave-h100-2"           # a sibling: the second H100
    assert own("vy-wave-h1000") is None and own("vy-wave-4090x2y") is None  # not one letter (+ digits)
    assert own("vy-live-2b-verifier") == "live-2b" and own("vy-live-2c") == "live-2"
    assert own("vyv-sw-73b-veritor-campaign") is None and own("vy-control-verity") is None
    assert own("vy-live2b-verifier-ro-veritor-campaign", {"vy-live2b-verifier-ro": "fold"}) == "fold"   # the binding hint
    assert own("vy-live2b-verifier-ro-verity-campaign", {"vy-live2b-verifier-ro": "fold"}) == "fold"
    assert own("vy-wave-4090", {"vy-wave-4090": "fold"}) == "wave-4090"    # a name that carries a lane beats a hint


def _fleet_rows(tmp_path, probes, listing=None, **kw):
    listing = listing or [_pod("a", "vy-wave-h100", 3.49), _pod("b", "vy-wave-h100b", 3.49), _pod("c", "vy-wave-h100-verifier", 0.49),
                          _pod("d", "vyv-sw-73b", 6.98), _pod("e", "vy-control-verity", 0.03)]
    fleet = notes.load_fleet(tmp_path, pods=lambda: listing, prober=lambda ps: {p["id"]: probes.get(p["id"], {}) for p in ps})
    return fleet, notes.status(tmp_path, 30, None, now=NOW, fleet=fleet, **kw)


def test_status_shows_every_pod_a_lane_owns_with_count_rate_and_probe(tmp_path):
    _report(tmp_path, "wave-h100", "CHECKPOINT 1111111 (20:00Z) [superseded] by wave-h100-2\n", 60, status="superseded")
    _report(tmp_path, "wave-h100-2", "CHECKPOINT 2222222 (20:55Z) [open] sweeping\n", 5)
    probes = {"a": {"busy": 2, "gpu_util": 97, "file_age_s": 10}, "b": {"busy": 0, "gpu_util": 0, "gpu_apps": 0, "file_age_s": 900},
              "c": {"busy": 0, "serve": 1, "gpu_util": None, "file_age_s": 3000}}
    fleet, rows = _fleet_rows(tmp_path, probes)
    assert [(r["lane"], r["attempt"]) for r in rows] == [("wave-h100-2", 2)]
    r = rows[0]
    assert [(p["name"], p.get("idle"), p.get("idle_min")) for p in r["pods"]] == [
        ("vy-wave-h100", False, None), ("vy-wave-h100b", True, 15.0), ("vy-wave-h100-verifier", False, None)]
    text = notes.render(rows)
    assert "wave-h100-2 #2" in text and "3 $7.47/h" in text
    assert "busy2 97%" in text and "idle 0% 15m" in text and "serve1" in text
    assert sorted(fleet.owner) == ["a", "b", "c"]                            # vyv-* and vy-control: nobody's
    line, runway = notes.account_line(fleet, {"balance": 117.0, "usd_h": 18.0}, NOW)
    assert line == "21:00Z ACCOUNT $117.00 at $18.00/h, runway 6.5h; lane pods $7.47/h (3), other pods $7.01/h (2)" and runway == 6.5
    assert notes.account_line(fleet, None, NOW)[1] is None
    assert len(notes.status(tmp_path, 30, None, now=NOW, fleet=fleet, every_attempt=True)) == 2


def test_a_lane_owning_a_running_pod_shows_however_old_and_a_failed_listing_says_so(tmp_path):
    _report(tmp_path, "old", "CHECKPOINT 1111111 (09:00Z) [final] done\n", 60 * 30, status="final")
    rows = notes.status(tmp_path, 30, 8, now=NOW, fleet=notes.Fleet([_pod("x", "vy-old")], {"x": "old"}, {}))
    assert [(r["lane"], len(r["pods"])) for r in rows] == [("old", 1)]
    assert notes.status(tmp_path, 30, 8, now=NOW, fleet=notes.Fleet([], {}, {})) == []
    _report(tmp_path, "new", "CHECKPOINT 1111111 (20:50Z) [open] x\n", 10)
    rows = notes.status(tmp_path, 30, 8, now=NOW, fleet=notes.load_fleet(tmp_path, pods=lambda: 1 / 0))
    assert "?" in notes.render(rows) and rows[0]["pods_error"].startswith("pod listing failed")


def test_idle_pod_alerts_once_per_idle_spell_of_an_open_lane(tmp_path):
    rows = [{"lane": "w", "state": "open", "pods": [{"id": "a", "name": "vy-w", "usd_h": 0.74, "idle": True, "idle_min": 2.0, "gpu_util": 0}]},
            {"lane": "f", "state": "final", "pods": [{"id": "b", "name": "vy-f", "usd_h": 0.74, "idle": True, "idle_min": 60.0}]}]
    since, alerted = {}, set()
    assert notes.idle_lines(since, alerted, rows, NOW, 5) == []               # idle 2 min so far
    later = NOW + dt.timedelta(minutes=3)
    assert notes.idle_lines(since, alerted, rows, later, 5) == ["21:03Z IDLE-POD w vy-w 5m ($0.74/h, GPU 0%, no work process)"]
    assert notes.idle_lines(since, alerted, rows, later + dt.timedelta(minutes=2), 5) == []   # once
    rows[0]["pods"][0]["idle"] = False                                        # busy again: the next spell alerts again
    assert notes.idle_lines(since, alerted, rows, later, 5) == [] and since == {}
    rows[0]["pods"][0].update(idle=True, idle_min=9.0)
    assert notes.idle_lines(since, alerted, rows, later, 5)[0].startswith("21:03Z IDLE-POD w vy-w 9m")


def test_runway_alerts_under_the_threshold_then_each_half_hour_shorter():
    assert notes.runway_alert(3.4, 3, None, NOW) == ([], None)
    lines, last = notes.runway_alert(2.9, 3, None, NOW)
    assert lines == ["21:00Z RUNWAY 2.9h (under 3h)"] and last == 2.9
    assert notes.runway_alert(2.6, 3, last, NOW) == ([], 2.9)
    assert notes.runway_alert(2.4, 3, last, NOW)[0] == ["21:00Z RUNWAY 2.4h (under 3h)"]
    assert notes.runway_alert(None, 3, 2.4, NOW) == ([], None)


def test_the_reaper_takes_only_pods_of_final_lanes_past_the_grace_that_are_not_kept(tmp_path):
    pod = {"id": "p1", "name": "vy-done", "usd_h": 0.74, "started_ts": NOW.timestamp() - 3600}
    rows = [{"lane": "done", "state": "final", "done_ts": NOW.timestamp() - 900, "pods": [pod]},
            {"lane": "fresh", "state": "final", "done_ts": NOW.timestamp() - 300, "pods": [{**pod, "id": "p2"}]},
            {"lane": "kept", "state": "final", "done_ts": NOW.timestamp() - 900, "keep_pod": "live verifier", "pods": [{**pod, "id": "p3"}]},
            {"lane": "busy", "state": "open", "pods": [{**pod, "id": "p4"}]}]
    noted, killed, asked = set(), [], []
    lines = notes.reap_lines(tmp_path, rows, NOW, 10, noted)
    assert lines == ["21:00Z FINAL-POD done vy-done p1 $0.74/h (final 15m ago): --reap terminates it, checkpoint --keep-pod WHY keeps it"]
    assert notes.reap_lines(tmp_path, rows, NOW, 10, noted) == [] and killed == []     # said once; nothing terminated
    ok = lambda lane, p: asked.append((lane, p["id"])) or (True, "2 cited art(s) preserved")
    lines = notes.reap_lines(tmp_path, rows, NOW, 10, noted, reap=True, terminate=killed.append, custody=ok)
    assert killed == ["p1"] and asked == [("done", "p1")]      # custody asked only for the reapable pod; kept/fresh/open untouched
    assert lines == ["21:00Z REAPED done vy-done p1 $0.74/h (final 15m ago); custody: 2 cited art(s) preserved"]
    assert (tmp_path / "reaper.log").read_text() == lines[0] + "\n"


def test_the_reaper_leaves_a_pod_started_after_the_final_it_means_the_lane_was_reopened(tmp_path):
    done = NOW.timestamp() - 64 * 60
    pods = [{"id": "fzs8", "name": "vy-red-team-flock", "usd_h": 0.48, "started_ts": NOW.timestamp() - 120},
            {"id": "nots", "name": "vy-red-team-flock", "usd_h": 0.48}]                     # no start time in the listing: same
    rows = [{"lane": "red-team-flock", "state": "final", "done_ts": done, "pods": pods}]
    noted, killed, state = set(), [], {}
    never = lambda lane, p: pytest.fail("no custody check for a resumed pod")
    lines = notes.reap_lines(tmp_path, rows, NOW, 10, noted, reap=True, terminate=killed.append, custody=never, state=state)
    assert killed == [] and lines == [
        "21:00Z RESUMED-POD red-team-flock vy-red-team-flock fzs8 (created 20:58Z, after FINAL 19:56Z): not reaped",
        "21:00Z RESUMED-POD red-team-flock vy-red-team-flock nots (created unknown, after FINAL 19:56Z): not reaped"]
    assert notes.reap_lines(tmp_path, rows, NOW, 10, noted, reap=True, terminate=killed.append, custody=never, state=state) == []
    assert killed == [] and not (tmp_path / "reaper.log").exists()


def test_the_reaper_says_nothing_about_a_pod_runpod_no_longer_knows(tmp_path):
    pod = {"id": "gone", "name": "vy-done", "usd_h": 0.48, "started_ts": NOW.timestamp() - 3600, "idle": True, "idle_min": 60}
    gone = lambda lane, p: (False, "its run dirs' custody is unknown: PodError: GET /pods/gone -> HTTP 404: not found (terminated, or never existed)")
    final = [{"lane": "done", "state": "final", "done_ts": NOW.timestamp() - 900, "pods": [pod]}]
    stale = [{"lane": "slow", "state": "open", "stale": True, "age_min": 90, "pods": [pod]}]
    killed: list = []
    assert notes.reap_lines(tmp_path, final, NOW, 10, set(), reap=True, terminate=killed.append, custody=gone, state={}) == []
    assert notes.stale_reap_lines(tmp_path, stale, NOW, 30, {}, reap=True, custody=gone, terminate=killed.append) == []
    assert killed == [] and not (tmp_path / "reaper.log").exists()


def test_the_reaper_blocks_a_final_lanes_pod_that_fails_custody_and_rechecks_after_30_min(tmp_path):
    pod = {"id": "p1", "name": "vy-done", "usd_h": 0.74, "started_ts": NOW.timestamp() - 3600}
    rows = [{"lane": "done", "state": "final", "done_ts": NOW.timestamp() - 900, "pods": [pod]}]
    noted, killed, state = set(), [], {}
    blocked = lambda lane, p: (False, "art:deadbeef not preserved: 404")
    lines = notes.reap_lines(tmp_path, rows, NOW, 10, noted, reap=True, terminate=killed.append, custody=blocked, state=state)
    assert killed == [] and lines == ["21:00Z REAP-BLOCKED done vy-done: art:deadbeef not preserved: 404"]
    assert (tmp_path / "reaper.log").read_text() == lines[0] + "\n"
    soon = NOW + dt.timedelta(minutes=29)
    assert notes.reap_lines(tmp_path, rows, soon, 10, noted, reap=True, terminate=killed.append, custody=blocked, state=state) == []
    later = NOW + dt.timedelta(minutes=31)
    lines = notes.reap_lines(tmp_path, rows, later, 10, noted, reap=True, terminate=killed.append,
                             custody=lambda lane, p: (True, "ok"), state=state)
    assert killed == ["p1"] and lines == ["21:31Z REAPED done vy-done p1 $0.74/h (final 46m ago); custody: ok"]
    boom = lambda lane, p: 1 / 0
    rows[0]["pods"] = [{**pod, "id": "p2"}]
    assert notes.reap_lines(tmp_path, rows, later, 10, noted, reap=True, terminate=killed.append, custody=boom, state=state) == [
        "21:31Z REAP-BLOCKED done vy-done: custody not checked (ZeroDivisionError: division by zero)"] and killed == ["p1"]


def test_pod_view_takes_the_later_of_created_and_last_started():
    p = {**_pod("x", "vy-x", started_min_ago=5), "createdAt": "2026-09-23T19:00:00Z"}
    assert notes.pod_view(p, None, NOW)["started_ts"] == (NOW - dt.timedelta(minutes=5)).timestamp()
    assert "started_ts" not in notes.pod_view({"id": "x", "name": "vy-x"}, None, NOW)


def test_cli_keep_pod_is_recorded_for_the_reaper_and_the_finish_check(tmp_path, capsys):
    _report(tmp_path, "lv", "", 60)
    notes.main(["checkpoint", "lv", "open", "serving", "--commit", "abc1234", "--keep-pod", "shared live verifier", "--root", str(tmp_path)])
    assert notes.binding(tmp_path, "lv")["keep_pod"] == "shared live verifier"
    row = notes.status(tmp_path, 30, None, now=NOW)[0]
    assert row["keep_pod"] == "shared live verifier"
    assert notes.main(["bind", "lv", "--keep-pod", "", "--root", str(tmp_path)]) == 0 and "keep_pod" not in notes.binding(tmp_path, "lv")


def test_watch_prints_no_mail_on_its_first_pass_and_never_for_a_superseded_lane(tmp_path, capsys):
    _report(tmp_path, "done", "CHECKPOINT 1111111 (20:00Z) [final] done\n", 55, status="final")
    _touch(tmp_path / "lanes" / "done" / "20260923T2058Z-handoff-coordinator.md", 2, "# late\n")
    _report(tmp_path, "gone", "CHECKPOINT 1111111 (20:00Z) [superseded] folded into done\n", 55, status="superseded")
    _touch(tmp_path / "lanes" / "gone" / "20260923T2058Z-handoff-x.md", 2, "# late too\n")
    rows = {r["lane"]: r for r in notes.status(tmp_path, 30, None, now=NOW, every_attempt=True)}
    assert (rows["done"]["mail"], rows["gone"]["mail"]) == (["20260923T2058Z-handoff-coordinator.md"], [])
    assert notes.main(["watch", "--root", str(tmp_path), "--all", "--passes", "1"]) == 0
    out = capsys.readouterr().out
    assert "MAIL" not in out and "pass 1: 0 open, 0 stale" in out


# ---- relaunch, bind fields, worktree gc -------------------------------------------------------------------------------------------------------

def test_bind_records_brief_final_budget_and_refuses_a_final_that_is_not_hhmmz(tmp_path):
    notes.bind(tmp_path, "l", branch="lane/l", brief="campaigns/w/BRIEF.md", final="06:30Z", budget="$5")
    assert notes.binding(tmp_path, "l") == {"branch": "lane/l", "brief": "campaigns/w/BRIEF.md", "final": "06:30Z", "budget": "$5"}
    with pytest.raises(ValueError):
        notes.bind(tmp_path, "l", final="6:30")
    assert notes.main(["bind", "l", "--final", "0630", "--root", str(tmp_path)]) == 2


def test_relaunch_saves_the_work_supersedes_binds_the_successor_and_prints_its_launch_message(tmp_path, repo, monkeypatch):
    root = tmp_path / "notes"
    monkeypatch.setenv("RESEARCH_RELAUNCH", str(tmp_path / "relaunch"))
    _git(repo, "checkout", "-qb", "other")
    _touch(repo / "a.py", 1, "theirs\n")
    _git(repo, "commit", "-qam", "theirs")
    _git(repo, "checkout", "-qb", "lane/wave-4090", "main")
    _touch(repo / "a.py", 1, "mine\n")
    _git(repo, "commit", "-qam", "mine")
    subprocess.run(["git", "-C", str(repo), "merge", "-q", "other"], capture_output=True)   # conflict: a.py UU, MERGE_HEAD
    _touch(repo / "notes.txt", 1, "untracked work\n")
    _report(root, "wave-4090", "CHECKPOINT 1111111 (20:00Z) [open] sweeping\n", 60)
    notes.bind(root, "wave-4090", branch="lane/wave-4090", worktree=str(repo), pod="vy-wave-4090", brief="campaigns/w/BRIEF.md",
               final="06:30Z", budget="$5")
    listing = [_pod("p1", "vy-wave-4090"), _pod("p2", "vy-wave-4090-verifier", 0.24), _pod("p3", "vyv-sw-1", 6.98)]
    fleet = lambda: notes.load_fleet(root, probe=False, pods=lambda: listing)  # noqa: E731
    s, done, msg = notes.relaunch(root, "wave-4090", why="stall 4", repo=repo, now=NOW, fleet=fleet)
    ev = tmp_path / "relaunch" / "wave-4090"
    assert not (root / "lanes" / "wave-4090" / "evidence").exists(), "a copy of code never goes in the notes"
    assert s == "wave-4090-2" and sorted(p.name for p in ev.iterdir()) == [
        "uncommitted-2100Z-untracked.tgz", "uncommitted-2100Z.merge", "uncommitted-2100Z.patch", "uncommitted-2100Z.status"]
    assert "UU a.py" in (ev / "uncommitted-2100Z.status").read_text() and "notes.txt" in (ev / "uncommitted-2100Z.status").read_text()
    import tarfile
    assert tarfile.open(ev / "uncommitted-2100Z-untracked.tgz").getnames() == ["notes.txt"]
    assert (ev / "uncommitted-2100Z.merge").read_text().strip() == _git(repo, "rev-parse", "other").strip()
    report = notes.report_for(root, "wave-4090")
    os.utime(report, (NOW.timestamp(), NOW.timestamp()))   # written at NOW: a checkpoint's HH:MM is dated from its file's time
    row = notes.lane_status(report, NOW, 30)
    assert row["state"] == "superseded" and row["text"] == "by wave-4090-2 (relaunch): stall 4"
    assert notes.binding(root, "wave-4090-2") == {"branch": "lane/wave-4090", "worktree": str(repo), "pod": "vy-wave-4090",
                                                  "succeeds": "wave-4090", "brief": "campaigns/w/BRIEF.md", "final": "06:30Z", "budget": "$5"}
    assert done[1:] == ["checkpoint wave-4090 superseded by wave-4090-2", f"bound wave-4090-2: lane/wave-4090 {repo} succeeds wave-4090"]
    lines = msg.splitlines()
    assert lines[0].startswith("You are lane `wave-4090-2` (attempt 2 of wave-4090), taking over `wave-4090`.")
    assert "`~/.research/bin/research notes inbox wave-4090-2`" in lines[0]
    assert lines[1] == "- Goal: the brief `campaigns/w/BRIEF.md`; continue where wave-4090 stopped (its report says what is done and what is left)."
    assert lines[2].startswith(f"- Base: `lane/wave-4090` @ {_git(repo, 'rev-parse', '--short=8', 'HEAD').strip()}, worktree `{repo}`")
    assert "- Pods (yours now): `vy-wave-4090` (p1, $0.74/h), `vy-wave-4090-verifier` (p2, $0.24/h)" in msg
    assert "- FINAL: 06:30Z hard; budget: $5." in lines and lines[-1] == "- MID-MERGE: finish the merge first (a.py unmerged)."
    with pytest.raises(ValueError, match="already taken over by wave-4090-2"):
        notes.relaunch(root, "wave-4090", repo=repo, now=NOW)
    s3, _, msg3 = notes.relaunch(root, "wave-4090-2", repo=repo, now=NOW, final="07:00Z")
    assert s3 == "wave-4090-3" and "(attempt 3 of wave-4090)" in msg3 and "- FINAL: 07:00Z hard; budget: $5." in msg3
    assert notes.chain(root, "wave-4090-3") == ["wave-4090-3", "wave-4090-2", "wave-4090"]


def test_successor_names_count_attempts_from_the_topic(tmp_path):
    for lane in ("dev-4090", "verify-rs-4", "b-v2"):
        _report(tmp_path, lane, "x\n", 5)
    assert [notes.successor_name(tmp_path, x) for x in ("dev-4090", "verify-rs-4", "b-v2")] == ["dev-4090-2", "verify-rs-5", "b-v2-2"]
    (tmp_path / "lanes" / "dev-4090-2").mkdir()
    assert notes.successor_name(tmp_path, "dev-4090") == "dev-4090-3"


def test_push_sends_the_lane_branch_never_main_and_never_forces(tmp_path, repo):
    root = tmp_path / "notes"
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "worktree", "add", "-q", "-b", "lane/p", str(tmp_path / "p"), "main")
    _report(root, "p", "CHECKPOINT 1111111 (20:00Z) [open] x\n", 30)
    line, rc = notes.push_branch(root, "p", repo)
    tip = _git(repo, "rev-parse", "lane/p").strip()
    assert (rc, line) == (0, f"PUSHED lane/p {tip[:8]} -> origin") and _git(origin, "rev-parse", "lane/p").strip() == tip
    assert notes.sync_state(repo, "lane/p")["sync"] == "in sync"

    other = tmp_path / "other"                                   # someone else pushed to lane/p: a rejected push, nothing forced
    _git(tmp_path, "clone", "-q", "-b", "lane/p", str(origin), str(other))
    _touch(other / "theirs.py", 1)
    _git(other, "add", "theirs.py")
    _git(other, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "theirs")
    _git(other, "push", "-q", "origin", "lane/p")
    _touch(tmp_path / "p" / "mine.py", 1)
    _git(tmp_path / "p", "add", "mine.py")
    _git(tmp_path / "p", "commit", "-qm", "mine")
    line, rc = notes.push_branch(root, "p", repo)
    assert rc == 1 and line.startswith("PUSH-FAILED lane/p ") and "git fetch origin lane/p and rebase or merge origin/lane/p" in line
    assert _git(origin, "rev-parse", "lane/p").strip() == _git(other, "rev-parse", "HEAD").strip()

    notes.bind(root, "m", branch="main")
    assert notes.push_branch(root, "m", repo)[1] == 2 and "never main" in notes.push_branch(root, "m", repo)[0]
    assert notes.push_branch(root, "nobranch", repo) == (f"REFUSED nobranch: no local branch lane/nobranch in {repo}", 2)


def test_push_in_the_notes_clone_lands_the_lane_on_the_notes_main_not_a_lane_branch(tmp_path):
    origin = tmp_path / "notes.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    root = tmp_path / "notes"
    _git(tmp_path, "clone", "-q", str(origin), str(root))
    _git(root, "checkout", "-q", "-b", "main")
    _report(root, "p", "CHECKPOINT 1111111 (20:00Z) [open] x\n", 30)
    (root / "lanes" / "q").mkdir(parents=True)
    (root / "lanes" / "q" / "other.md").write_text("another lane's draft\n")
    line, rc = notes.push_branch(root, "p", root)
    assert rc == 0 and "; pushed" in line, line
    assert _git(origin, "ls-tree", "-r", "--name-only", "main").split() == [str(notes.report_for(root, "p").relative_to(root))]
    assert not _git(origin, "for-each-ref", "refs/heads/lane/").strip()


@pytest.mark.slow
def test_branches_lists_what_origin_lacks_and_blocks_only_on_open_lanes(tmp_path, repo):
    root = tmp_path / "notes"
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    for lane, state in (("synced", "open"), ("ahead", "open"), ("local", "open"), ("old", "final"), ("behind", "open")):
        _git(repo, "branch", f"lane/{lane}", "main")
        _report(root, lane, f"CHECKPOINT 1111111 (20:00Z) [{state}] x\n", 30, status=state)
        if lane not in ("local", "old"):
            _git(repo, "push", "-q", "origin", f"lane/{lane}")
    _git(repo, "branch", "work-x", "main")                       # a bound branch outside lane/*
    notes.bind(root, "bound", branch="work-x")
    _report(root, "bound", "CHECKPOINT 1111111 (20:00Z) [open] x\n", 30)
    _git(repo, "checkout", "-q", "lane/ahead")
    _touch(repo / "n.py", 1)
    _git(repo, "add", "n.py")
    _git(repo, "commit", "-qm", "n")
    _git(repo, "push", "-q", "origin", "lane/ahead:lane/behind")
    _git(repo, "fetch", "-q", "origin")
    _git(repo, "checkout", "-q", "main")
    rows = {r["branch"]: r for r in notes.branches(root, repo, now=NOW)}
    assert {k: (r["sync"], r["blocking"]) for k, r in rows.items()} == {
        "lane/ahead": ("ahead", True), "lane/behind": ("behind", False), "lane/local": ("unpushed", True),
        "lane/old": ("unpushed", False), "lane/synced": ("in sync", False), "work-x": ("unpushed", True)}
    assert rows["lane/ahead"]["ahead"] == 1 and rows["work-x"]["lanes"] == {"bound": "open"}
    text = notes.render_branches(list(rows.values()))
    assert text.splitlines()[0].startswith("BLOCKING lane/ahead ") and "lane/synced" not in text
    assert text.endswith("6 branches: 1 ahead, 1 behind, 1 in sync, 3 unpushed; 3 blocking (an open lane's work origin lacks: "
                         "research notes push <lane>)")
    assert [r["branch"] for r in notes.branches(root, repo, ["old"], now=NOW)] == ["lane/old"]
    assert notes.main(["branches", "--root", str(root), "--repo", str(repo)]) == 1


def test_finish_requires_the_push_only_when_asked(tmp_path, repo):
    root = tmp_path / "notes"
    _git(repo, "checkout", "-qb", "lane/q")
    _report(root, "q", "CHECKPOINT 1111111 (20:55Z) [final] done art:1234abcd\n", 5, status="final")
    kw = {"custody": lambda arts: (True, "ok"), "keep_pod": "x"}
    text, rc = notes.finish(root, "q", repo, **kw)
    assert rc == 0 and "  pushed    lane/q unpushed -> research notes push q" in text
    assert notes.finish(root, "q", repo, require_pushed=True, **kw)[1] == 3
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", "lane/q")
    text, rc = notes.finish(root, "q", repo, require_pushed=True, **kw)
    assert rc == 0 and "  pushed    lane/q on origin" in text


def test_gc_worktrees_removes_only_clean_pushed_worktrees_of_done_lanes(tmp_path, repo, monkeypatch):
    git = notes._git
    monkeypatch.setattr(notes, "_git", lambda p, *a, **kw: git(p, *a, timeout=None))   # a loaded pod can't time a status out here
    root = tmp_path / "notes"
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    for lane, state in (("done", "final"), ("dirty", "final"), ("local", "final"), ("busy", "open"), ("pred", "superseded")):
        _git(repo, "worktree", "add", "-q", "-b", f"lane/{lane}", str(tmp_path / lane), "main")
        _report(root, lane, f"CHECKPOINT 1111111 (20:00Z) [{state}] x\n", 30, status=state)
        if lane != "local":
            _git(repo, "push", "-q", "origin", f"lane/{lane}")
    _git(repo, "worktree", "add", "-q", "-b", "scratch", str(tmp_path / "scratch"), "main")
    _git(repo, "push", "-q", "origin", "scratch")
    _touch(tmp_path / "local" / "x.py", 1)
    _git(tmp_path / "local", "add", "x.py")
    _git(tmp_path / "local", "commit", "-qm", "not pushed")
    _touch(tmp_path / "dirty" / "wip.py", 1)
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "exclude").write_text("target/\n")
    _touch(tmp_path / "done" / "target" / "build.o", 1)                             # ignored: not dirty, goes with the worktree
    _report(root, "pred-2", "CHECKPOINT 1111111 (20:50Z) [open] x\n", 10)          # took over pred's worktree: still in use
    notes.bind(root, "pred-2", worktree=str(tmp_path / "pred"), succeeds="pred")
    rows = {Path(r["worktree"]).name: r for r in notes.gc_worktrees(root, repo, now=NOW)}
    assert {k: r["keep"] for k, r in rows.items()} == {
        "busy": "lane busy is open", "dirty": "dirty (1 paths)", "done": None, "local": "branch tip not on origin",
        "pred": "lane pred-2 is open", "scratch": "no lane"}
    assert (tmp_path / "done").exists()                                             # listing removes nothing
    with monkeypatch.context() as m:                                                # a failed status keeps it, and says so
        m.setattr(notes, "_git", lambda p, *a, **kw: None if a[0] == "status" else git(p, *a, timeout=None))
        assert {Path(r["worktree"]).name: r["keep"] for r in notes.gc_worktrees(root, repo, now=NOW)}["done"] == "git status failed"
    rows = {Path(r["worktree"]).name: r for r in notes.gc_worktrees(root, repo, now=NOW, apply=True)}
    assert rows["done"]["removed"] is True and "removed" not in rows["dirty"]
    assert not (tmp_path / "done").exists() and (tmp_path / "dirty" / "wip.py").exists()
    assert _git(repo, "branch", "--list", "lane/done").strip() == "lane/done"          # the branch stays
    text = notes.render_gc(list(rows.values()), True)
    assert "\nremoved  " in "\n" + text and "keep     " in text and text.endswith("1 removed; kept 5: 2 in use by an open lane, 1 with no lane")


# ---- the steward rules: stale-lane pods, OVERDUE, OVER-BUDGET, scheduled runs, guardian kills ------------------------------------------

def _set_mtime(p: Path, when: dt.datetime) -> None:
    os.utime(p, (when.timestamp(), when.timestamp()))


def test_the_steward_reaps_an_idle_pod_of_a_stale_lane_only_with_reap_and_after_custody(tmp_path):
    _report(tmp_path, "dead", "CHECKPOINT 1111111 (19:00Z) [open] bench running\n", 120)
    fleet = notes.load_fleet(tmp_path, pods=lambda: [_pod("p1", "vy-dead")],
                             prober=lambda ps: {"p1": {"busy": 0, "gpu_util": 0, "gpu_apps": 0, "serve": 0, "file_age_s": 3000}})
    rows = notes.status(tmp_path, 30, None, now=NOW, fleet=fleet)
    assert (rows[0]["stale"], rows[0]["pods"][0]["idle"], rows[0]["pods"][0]["idle_min"]) == (True, True, 50.0)
    pod = {"id": "p2", "name": "vy-x", "usd_h": 0.74, "idle": True, "idle_min": 45.0}
    rows += [{"lane": "short", "state": "open", "stale": True, "age_min": 90, "pods": [{**pod, "id": "p2", "idle_min": 20.0}]},
             {"lane": "busy", "state": "open", "stale": True, "age_min": 90, "pods": [{**pod, "id": "p3", "idle": False, "busy": 1}]},
             {"lane": "kept", "state": "open", "stale": True, "keep_pod": "verifier", "age_min": 90, "pods": [{**pod, "id": "p4"}]},
             {"lane": "alive", "state": "open", "stale": False, "age_min": 5, "pods": [{**pod, "id": "p5"}]},
             {"lane": "done", "state": "final", "stale": False, "age_min": 5, "pods": [{**pod, "id": "p6"}]},
             {"lane": "unreached", "state": "open", "stale": True, "age_min": 90, "pods": [{**pod, "id": "p7", "idle": None, "error": "ssh rc 255"}]}]
    state, killed, asked = {}, [], []

    def blocked(lane, p):
        asked.append((lane, p["id"]))
        return False, "art:feedbeef00 not preserved: absent"

    assert notes.stale_reap_lines(tmp_path, rows, NOW, 30, state, custody=blocked, terminate=killed.append) == []   # no --reap
    lines = notes.stale_reap_lines(tmp_path, rows, NOW, 30, state, reap=True, custody=blocked, terminate=killed.append)
    assert (asked, killed, lines) == ([("dead", "p1")], [], ["21:00Z REAP-BLOCKED dead vy-dead: art:feedbeef00 not preserved: absent"])
    assert notes.stale_reap_lines(tmp_path, rows, NOW + dt.timedelta(minutes=29), 30, state, reap=True, custody=blocked,
                                  terminate=killed.append) == [] and len(asked) == 1           # once per pod per 30 min, custody too
    later = NOW + dt.timedelta(minutes=30)
    lines = notes.stale_reap_lines(tmp_path, rows, later, 30, state, reap=True, custody=lambda lane, p: (True, "1 cited art(s) preserved"),
                                   terminate=killed.append)
    assert killed == ["p1"] and lines == ["21:30Z REAPED dead vy-dead p1 $0.74/h (STALE 50m, idle 50m); custody: 1 cited art(s) preserved"]
    assert (tmp_path / "reaper.log").read_text().splitlines() == ["21:00Z REAP-BLOCKED dead vy-dead: art:feedbeef00 not preserved: absent", lines[0]]

    def boom(lane, p):
        raise RuntimeError("catalog locked")

    def refuse(pod_id):
        raise RuntimeError("HTTP 500")

    state = {}
    assert notes.stale_reap_lines(tmp_path, rows[:1], NOW, 30, state, reap=True, custody=boom, terminate=killed.append) == [
        "21:00Z REAP-BLOCKED dead vy-dead: custody not checked (RuntimeError: catalog locked)"]
    assert notes.stale_reap_lines(tmp_path, rows[:1], later, 30, state, reap=True, custody=lambda lane, p: (True, "ok"), terminate=refuse) == [
        "21:30Z REAP-FAILED dead vy-dead p1 $0.74/h (STALE 50m, idle 50m): HTTP 500"]


def test_reap_custody_wants_the_cited_arts_preserved_and_every_run_on_the_pod_fetched_or_registered(tmp_path):
    root, runs = tmp_path / "notes", tmp_path / "runs"
    _report(root, "dead", "CHECKPOINT 1111111 (20:00Z) [open] x\nresults art:1234abcd and art:feedbeef00, again art:1234abcd\n", 60)

    def run(rid, pod_id, phase="launched", files=()):
        (runs / rid).mkdir(parents=True)
        (runs / rid / "remote.json").write_text(json.dumps({"run_id": rid, "machine": "vy-dead", "pod_id": pod_id, "phase": phase}))
        for f in files:
            (runs / rid / f).write_text("{}\n")

    run("r20260923-190000-aaaa", "p1", files=["job.json", "preserved.json"])     # fetched with --all
    run("r20260923-191000-bbbb", "p1", files=["job.json"])                       # registered in the store
    run("r20260923-192000-cccc", "p1", phase="refused")                          # these three never ran on the pod
    run("r20260923-192500-dddd", "p1", phase="shipping")
    run("r20260923-192600-eeee", "p1", files=["refused.json"])
    run("r20260923-193000-ffff", "p2")                                           # another pod's
    cited = []

    def arts(a):
        cited.append(a)
        return True, f"{len(a)} cited art(s) preserved"

    runs_ok = lambda pod_id: notes.run_custody(pod_id, runs, registered={"r20260923-191000-bbbb"}.__contains__)  # noqa: E731
    assert notes.reap_custody(root, "dead", "p1", arts=arts, runs=runs_ok) == (
        True, "2 cited art(s) preserved; 2 run(s) on it fetched or registered")
    assert cited == [["1234abcd", "feedbeef00"]]
    assert notes.reap_custody(root, "dead", "p1", arts=lambda a: (False, "art:feedbeef00 not preserved: absent"), runs=runs_ok) == (
        False, "art:feedbeef00 not preserved: absent")
    run("r20260923-194000-abcd", "p1", phase="launching")                         # may have started: it must be fetched
    assert notes.reap_custody(root, "dead", "p1", arts=arts, runs=runs_ok) == (
        False, "run r20260923-194000-abcd on it is neither fetched nor in the store -> research fetch --all r20260923-194000-abcd")
    _report(root, "empty", "CHECKPOINT 1111111 (20:00Z) [open] died before any result\n", 60)
    assert notes.reap_custody(root, "empty", "p9", arts=arts, runs=runs_ok) == (True, "no art: cited; 0 run(s) on it fetched or registered")


def test_art_custody_is_the_preserved_check_and_caps_what_it_would_read_back(tmp_path, monkeypatch):
    from research.store.local import LocalStore

    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    st = LocalStore()
    arts = []
    for name in ("pushed", "local-only", "unrecorded"):
        (tmp_path / f"{name}.txt").write_text(f"result {name}\n")
        arts.append(st.put_artifact("fixture/v1", {}, file=tmp_path / f"{name}.txt"))
    st.push(arts[0])
    st.push(arts[2], preserve=False)                            # on the remote, but no verified replica recorded
    for h, _ in st.get_manifest(arts[2]).blobs():
        st.object_path(h).unlink()                              # and not held locally: the head check would read it back
    pushed, local, unrecorded = (a.split(":")[1][:12] for a in arts)
    assert notes.art_custody([pushed]) == (True, "1 cited art(s) preserved")
    ok, why = notes.art_custody([pushed, local])
    assert not ok and why.startswith(f"art:{local} not preserved: ")
    assert notes.art_custody(["0000000000"]) == (False, "art:0000000000 is unknown in the local catalog and on the remote")
    assert notes.art_custody([unrecorded], max_readback_mb=0) == (
        False, f"art:{unrecorded}: no verified remote replica recorded, 0 MB to read back -> research data preserved art:{unrecorded}")
    assert notes.art_custody([unrecorded]) == (True, "1 cited art(s) preserved")          # small: read back, verified, recorded
    assert notes.art_custody([unrecorded], max_readback_mb=0) == (True, "1 cited art(s) preserved")


def test_r2_env_fills_only_unset_variables(tmp_path):
    p = tmp_path / "r2.env"
    p.write_text("# R2\nAWS_ACCESS_KEY_ID=abc\nexport R2_BUCKET='store-bucket'\nR2_ENDPOINT=\"https://x\"\nAWS_SECRET_ACCESS_KEY=theirs\n"
                 "R2_ACCOUNT_ID=acct\n\nnot a pair\n")
    env = {"AWS_SECRET_ACCESS_KEY": "mine", "R2_ACCOUNT_ID": ""}
    assert notes.load_env_file(p, env) == ["AWS_ACCESS_KEY_ID", "R2_BUCKET", "R2_ENDPOINT", "R2_ACCOUNT_ID"]
    assert env == {"AWS_SECRET_ACCESS_KEY": "mine", "R2_ACCOUNT_ID": "acct", "AWS_ACCESS_KEY_ID": "abc", "R2_BUCKET": "store-bucket",
                   "R2_ENDPOINT": "https://x"}
    assert notes.load_env_file(tmp_path / "absent.env", env) == []


def test_overdue_fires_once_thirty_minutes_after_the_bindings_final_unless_the_lane_is_done(tmp_path):
    for lane, final in (("late", "20:00Z"), ("done", "20:00Z"), ("early", "21:20Z"), ("overnight", "06:30Z"), ("nofinal", None)):
        notes.bind(tmp_path, lane, final=final, branch=f"lane/{lane}")
        _set_mtime(tmp_path / "lanes" / lane / "binding.json", NOW - dt.timedelta(hours=4))       # bound at 17:00Z
    rows = [{"lane": "late", "state": "open", "stale": True, "age_min": 50, "life_min": 50.0, "life_via": "edit"},
            {"lane": "done", "state": "final", "stale": False, "age_min": 5},
            {"lane": "early", "state": "open", "stale": False, "age_min": 5, "life_min": 2.0, "life_via": "commit"},
            {"lane": "overnight", "state": "open", "stale": False, "age_min": 5},
            {"lane": "nofinal", "state": "open", "stale": False, "age_min": 5}]
    state = {}
    assert notes.overdue_lines(tmp_path, rows, NOW - dt.timedelta(minutes=31), state) == []                  # 20:29Z: in the grace
    assert notes.overdue_lines(tmp_path, rows, NOW, state) == ["21:00Z OVERDUE late FINAL 20:00Z passed 60m ago: STALE, life 50m via edit"]
    assert notes.overdue_lines(tmp_path, rows, NOW + dt.timedelta(minutes=5), state) == []                   # once
    notes.bind(tmp_path, "early", pod="vy-early")                           # rebound after its deadline, same final: the deadline stays
    _set_mtime(tmp_path / "lanes" / "early" / "binding.json", NOW + dt.timedelta(minutes=30))
    assert notes.overdue_lines(tmp_path, rows, NOW + dt.timedelta(minutes=51), state) == [
        "21:51Z OVERDUE early FINAL 21:20Z passed 31m ago: open, life 2m via commit"]
    notes.bind(tmp_path, "late", final="22:00Z")                            # the coordinator extends it: a new deadline, a new alert
    _set_mtime(tmp_path / "lanes" / "late" / "binding.json", NOW + dt.timedelta(minutes=10))
    assert notes.overdue_lines(tmp_path, rows, NOW + dt.timedelta(minutes=89), state) == []
    assert notes.overdue_lines(tmp_path, rows, NOW + dt.timedelta(minutes=91), state)[0].startswith("22:31Z OVERDUE late FINAL 22:00Z passed 31m")
    assert notes.final_deadline(tmp_path, "overnight", "06:30Z") == dt.datetime(2026, 9, 24, 6, 30, tzinfo=dt.timezone.utc)


def test_budget_accumulates_each_lanes_pod_dollars_per_pass_and_alerts_once_over_the_binding(tmp_path):
    notes.bind(tmp_path, "w", budget="$2: 1x 4090 + a verifier")
    notes.bind(tmp_path, "free", budget="$0, laptop only")
    listing = [_pod("a", "vy-w", 0.60, started_min_ago=120), _pod("b", "vy-w-verifier", 0.20, started_min_ago=60), _pod("c", "vy-nob", 1.0, 30)]
    fleet = notes.Fleet(listing, {"a": "w", "b": "w", "c": "nob"}, {})
    state = {}
    at = lambda m: NOW + dt.timedelta(minutes=m)  # noqa: E731
    assert notes.budget_lines(tmp_path, fleet, NOW, state) == []            # first sight: each pod counted since it started
    assert state["spend"] == {"w": 1.4, "nob": 0.5}
    assert notes.budget_lines(tmp_path, fleet, at(45), state) == [] and state["spend"]["w"] == 2.0         # at the budget, not over
    assert notes.budget_lines(tmp_path, fleet, at(60), state) == ["22:00Z OVER-BUDGET w $2.20 on pods > $2 (budget: $2: 1x 4090 + a verifier); now $0.80/h"]
    assert notes.budget_lines(tmp_path, fleet, at(62), state) == []         # once
    spent = state["spend"]["w"]
    assert notes.budget_lines(tmp_path, notes.Fleet([], {}, {}, error="pod listing failed"), at(70), state) == [] and state["spend"]["w"] == spent
    notes.budget_lines(tmp_path, notes.Fleet(listing[1:], fleet.owner, {}), at(72), state)                 # pod a missing from one listing
    notes.budget_lines(tmp_path, fleet, at(74), state)                                                    # ... is not recounted from its start
    assert state["spend"]["w"] == pytest.approx(spent + 0.2 * 12 / 60 + 0.6 * 12 / 60, abs=1e-3)
    notes.bind(tmp_path, "w", budget="$5")                                  # a raised budget re-arms the alert
    assert notes.budget_lines(tmp_path, notes.Fleet([_pod("a", "vy-w", 30.0)], {"a": "w"}, {}), at(80), state)[0].startswith(
        "22:20Z OVER-BUDGET w $5.")
    assert (notes.budget_usd("$22: 1x A100"), notes.budget_usd("$1.5"), notes.budget_usd("1x 4090 <= 30 min")) == (22.0, 1.5, None)


FRESH = ["--on", "vy-lean", "--project", "verity", "--source", ".", "--cwd", "source", "--tool", "lean_audit", "--",
         "python", "tools/verity/lean/check.py", "--all", "--build", "--fresh"]


def _run_toml(*entries: str) -> str:
    return "\n".join(f"[[run]]\n{e}\n" for e in entries)


def test_scheduled_runs_launch_once_per_utc_day_or_iso_week_at_or_after_their_time(tmp_path):
    (tmp_path / "steward.toml").write_text(_run_toml(
        f'name = "lean-fresh"\nat = "20:30Z"\nsource = "~/src"\nargs = {json.dumps(FRESH)}',
        'name = "lean-upstream"\nat = "10:00Z"\nweekday = "thu"\nsource = "/src"\nargs = ["--on=vy-small", "--project", "verity", "--", "true"]',
        'name = "later"\nat = "22:00Z"\nsource = "/src"\nargs = ["--on", "vy-x", "--", "true"]'))
    calls, ids = [], iter(f"r{i}" for i in range(1, 99))

    def run(cmd, env, cwd, timeout):
        calls.append((cmd, env["PYTHONPATH"], cwd, timeout))
        return (3, "", "research run --on vy-x: refused: machine 'vy-x' is not listed") if "vy-x" in cmd else (0, "", "research: run launched")

    def at(days: float = 0, minutes: float = 0, state: dict | None = None) -> list[str]:
        return notes.run_lines(tmp_path, NOW + dt.timedelta(days=days, minutes=minutes), state, run=run, new_id=lambda: next(ids))

    state, home = {}, Path.home()                                  # NOW: Wednesday 2026-09-23 21:00Z, ISO week 39
    assert at(state=state) == ["21:00Z LAUNCHED lean-fresh r1 on vy-lean"]
    assert calls == [([sys.executable, "-m", "research", "run", "--id", "r1", *FRESH], f"{home}/src", home / "src", notes.RUN_LAUNCH_TIMEOUT_S)]
    assert state["run"]["lean-fresh"] == {"period": "2026-09-23", "run": "r1", "rc": 0}
    assert at(minutes=50, state=state) == []                       # once a day; `later` is not due before 22:00Z
    assert at(minutes=60, state=state) == ["22:00Z RUN-FAILED later r2: research run --on vy-x: refused: machine 'vy-x' is not listed"]
    assert at(minutes=70, state=state) == [] and len(calls) == 2   # a failed launch waits for the next period
    assert at(days=1, state=state) == ["21:00Z LAUNCHED lean-fresh r3 on vy-lean", "21:00Z LAUNCHED lean-upstream r4 on vy-small"]
    assert at(days=2, state=state) == ["21:00Z LAUNCHED lean-fresh r5 on vy-lean"]             # the weekly one: once in week 39
    assert at(days=7, state=state) == ["21:00Z LAUNCHED lean-fresh r6 on vy-lean"]             # week 40, before its Thursday
    assert at(days=8, state=state)[1] == "21:00Z LAUNCHED lean-upstream r8 on vy-small"
    assert at(days=3, minutes=-11 * 60, state={}) == ["10:00Z LAUNCHED lean-upstream r9 on vy-small"]   # a steward down on Thursday: Saturday
    lines = notes.run_lines(tmp_path, NOW, {}, run=lambda *a: (None, "", "timeout after 600s"), new_id=lambda: "r0")
    assert lines[0] == "21:00Z RUN-FAILED lean-fresh r0: timeout after 600s"


def test_an_invalid_run_entry_is_one_line_until_it_changes(tmp_path):
    ok = 'at = "20:00Z"\nsource = "/src"\nargs = ["--on", "vy-x", "--", "true"]'
    (tmp_path / "steward.toml").write_text(_run_toml(
        'name = "no-args"\nat = "20:00Z"\nsource = "/src"', 'name = "nine"\nat = "9am"\nsource = "/src"\nargs = ["--on", "vy-x", "--", "true"]',
        f'name = "monday"\nweekday = "monday"\n{ok}', 'name = "local"\nat = "20:00Z"\nsource = "/src"\nargs = ["--project", "p", "--", "research", "--on", "x"]',
        'name = "named"\nat = "20:00Z"\nsource = "/src"\nargs = ["--on", "vy-x", "--id", "r1", "--", "true"]',
        f'name = "twice"\n{ok}', f'name = "twice"\n{ok}'))
    calls, state = [], {}

    def run(*a):
        calls.append(a)
        return 0, "", ""

    assert notes.run_lines(tmp_path, NOW, state, run=run, new_id=lambda: "r9") == [
        '21:00Z RUN-FAILED no-args: an entry needs name, at = "HH:MMZ", source and args (a list of strings)',
        '21:00Z RUN-FAILED nine: an entry needs name, at = "HH:MMZ", source and args (a list of strings)',
        "21:00Z RUN-FAILED monday: weekday = 'monday': one of mon, tue, wed, thu, fri, sat, sun",
        '21:00Z RUN-FAILED local: args put the run on a machine: ["--on", MACHINE, ..., "--", CMD...]',
        "21:00Z RUN-FAILED named: args set --id: the steward names the run",
        "21:00Z LAUNCHED twice r9 on vy-x", "21:00Z RUN-FAILED twice: name 'twice' is used by another entry"]
    assert len(calls) == 1 and notes.run_lines(tmp_path, NOW, state, run=run) == []             # each said once
    (tmp_path / "steward.toml").write_text(_run_toml(f'name = "monday"\nweekday = "mon"\n{ok}'))
    assert notes.run_lines(tmp_path, NOW, state, run=run, new_id=lambda: "r10") == ["21:00Z LAUNCHED monday r10 on vy-x"]   # fixed: it runs
    (tmp_path / "steward.toml").write_text("[[run]\n")
    assert notes.run_lines(tmp_path, NOW, state, run=run)[0].startswith("21:00Z RUN-FAILED steward.toml: ")
    assert notes.run_lines(tmp_path, NOW, state, run=run) == []                                 # a broken file is said once
    assert notes.run_lines(tmp_path / "nowhere", NOW, {}, run=run) == [] and len(calls) == 2


def test_guardian_kills_go_to_the_owning_lanes_inbox_by_machine_pod_hint_or_cwd(tmp_path):
    root, log, wt = tmp_path / "notes", tmp_path / "mem_guardian.log", tmp_path / "wt" / "tables-fix"
    for lane in ("sp1-formats", "agkr-table", "tables-fix", "tables-fix-2"):
        _report(root, lane, "CHECKPOINT 1111111 (20:50Z) [open] x\n", 10)
    notes.bind(root, "agkr-table", pod="vy-agkr-a100,vy-agkr-h100")
    notes.bind(root, "tables-fix", worktree=str(wt))
    notes.bind(root, "tables-fix-2", worktree=str(wt), succeeds="tables-fix")
    (wt / "sub").mkdir(parents=True)
    _touch(log, 0, "2026-09-23T20:00:00+00:00 heartbeat loops=30\n2026-09-23T20:01:00+00:00 KILLED pid=1 mem=0.58GB reason=swap cmd=python -m "
                   "research run --on box-a\n")
    fleet = notes.Fleet([_pod("p4090", "vy-sp1f-4090")], {"p4090": "sp1-formats"}, {})
    machines = lambda: {"box-a": {"pod_id": "p4090"}}  # noqa: E731
    state = {}
    assert notes.guardian_lines(root, [], fleet, NOW, state, log, machines) == []             # the first look only marks the end
    py = "/Users/u/projects/verity-main-wt/main/.venv/bin/python -m research"
    kills = [f"2026-09-23T20:55:00+00:00 KILLED pid=12046 mem=0.58GB reason=swap 8.6GB used (95%) > 95% cmd={py} run --on box-a --source .",
             f"2026-09-23T20:55:10+00:00 KILLED pid=12047 mem=0.58GB reason=disk floor 3.4GB free cwd=/ cmd={py} run --on vy-agkr-h100 --source .",
             f"2026-09-23T20:56:00+00:00 KILLED pid=12048 mem=1.20GB reason=per-proc cap 1.0GB cwd={wt}/sub cmd={py} data reindex",
             f"2026-09-23T20:57:00+00:00 KILLED pid=12049 mem=0.69GB reason=disk floor 3.5GB free cwd=/Users/u cmd={py} data reindex --remote",
             f"2026-09-23T20:57:30+00:00 KILLED pid=12050 mem=0.60GB reason=total cap 3GB (sum 3.2GB) cmd={py} run --on box-a --stage b"]
    with log.open("a") as f:
        f.write("\n".join(kills) + "\n2026-09-23T20:58:00+00:00 heartbeat loops=60\n2026-09-23T20:58:01+00:00 KILLED pid=9 mem=0.6GB reason=swap")
    name = "20260923T2100Z-handoff-steward-guardian.md"
    assert notes.guardian_lines(root, [], fleet, NOW, state, log, machines) == [
        f"21:00Z KILLED-UNOWNED {kills[3]}", f"21:00Z ROUTED agkr-table 1: {name}", f"21:00Z ROUTED sp1-formats 2: {name}",
        f"21:00Z ROUTED tables-fix-2 1: {name}"]
    text = (root / "lanes" / "sp1-formats" / name).read_text()
    title = "the laptop guardian killed 2 process(es) of sp1-formats (swap 8.6GB used (95%) > 95%): rerun"
    assert text.startswith(f"---\nlane: sp1-formats\nkind: handoff\nfrom: steward\ncreated: 2026-09-23T21:00Z\n---\n\n# {title}\n")
    assert f"~~~text\n{kills[0]}\n{kills[4]}\n~~~\n" in text and "killed on the laptop by the guardian, not on the pod: rerun it" in text
    assert [x["title"] for x in notes.inbox(root, "sp1-formats")] == [title]
    assert notes.guardian_lines(root, [], fleet, NOW, state, log, machines) == []             # each kill once; the partial line waits
    with log.open("a") as f:
        f.write(" cmd=python -m research run --on box-a\n")
    assert notes.guardian_lines(root, [], None, NOW, state, log, machines) == [                 # no fleet: the machine name as a pod name
        "21:00Z KILLED-UNOWNED 2026-09-23T20:58:01+00:00 KILLED pid=9 mem=0.6GB reason=swap cmd=python -m research run --on box-a"]
    log.write_text(kills[2] + "\n")                                          # truncated (rotated): read from the start again
    assert notes.guardian_lines(root, [], fleet, NOW, state, log, machines) == [f"21:00Z ROUTED tables-fix-2 1: {name[:-3]}-2.md"]
    with log.open("a") as f:                                                  # the owner is final: the coordinator's inbox (contract §5)
        f.write(kills[2].replace("pid=12048", "pid=12051") + "\n")
    rows = [{"lane": "tables-fix-2", "state": "final"}]
    assert notes.guardian_lines(root, rows, fleet, NOW, state, log, machines) == [f"21:00Z ROUTED coordinator 1: {name} (tables-fix-2 is final)"]
    text = (root / "lanes" / "coordinator" / name).read_text()
    assert text.startswith("---\nlane: coordinator\nkind: handoff\nfrom: steward\ncreated: 2026-09-23T21:00Z\n---\n\n"
                           "# the laptop guardian killed 1 process(es) of tables-fix-2 (final) (per-proc cap 1.0GB): rerun\n")


def test_snapshot_refuses_git_data_by_name_and_by_content(tmp_path):
    """The notes repo is public: a git bundle, pack or index (14 bundles of the private repo were found there, 2026-09-27) is never
    committed, whatever its name, and is reported as refused."""
    root = tmp_path / "notes"
    _git(tmp_path, "init", "-q", "-b", "main", str(root))
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    _touch(root / "lanes" / "a" / "20260927T1100Z-report-a.md", 5, "# a\n")
    (root / "lanes" / "a" / "evidence").mkdir(parents=True)
    (root / "lanes" / "a" / "evidence" / "main-abc.bundle").write_bytes(b"anything")
    (root / "lanes" / "a" / "evidence" / "renamed.txt").write_bytes(b"# v2 git bundle\nabc refs/heads/main\n")
    (root / "lanes" / "a" / "evidence" / "objects.dat").write_bytes(b"PACK\x00\x00\x00\x02")
    snap = notes.snapshot(root, now=NOW)
    assert snap["paths"] == 1 and sorted(snap["refused"]) == ["lanes/a/evidence/main-abc.bundle", "lanes/a/evidence/objects.dat",
                                                              "lanes/a/evidence/renamed.txt"] and snap["evidence"] == []
    assert _git(root, "ls-files").splitlines() == ["lanes/a/20260927T1100Z-report-a.md"]
    assert "3 git data refused" in _git(root, "log", "--format=%s")


def test_snapshot_leaves_the_steward_state_out(tmp_path):
    root = tmp_path / "notes"
    _touch(root / "lanes" / "a" / "20260923T1700Z-report-a.md", 5, "# a\n")
    _git(tmp_path, "init", "-q", "-b", "main", str(root))
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    notes.save_state(root, {"spend": {"a": 1.0}})
    assert notes.snapshot(root, now=NOW)["paths"] == 1 and _git(root, "ls-files").splitlines() == ["lanes/a/20260923T1700Z-report-a.md"]
    notes.save_state(root, {"spend": {"a": 2.0}})
    assert notes.snapshot(root) is None and notes.load_state(root) == {"spend": {"a": 2.0}}


def test_a_failing_steward_rule_is_a_line_not_a_dead_watcher():
    assert notes.steward_rule("ok", lambda a, b=0: [f"{a}{b}"], "x", b=1) == ["x1"]
    [line] = notes.steward_rule("run", lambda: 1 / 0)
    assert line.endswith(" STEWARD-ERROR run: ZeroDivisionError: division by zero")


def test_cli_watch_pass_runs_the_steward_rules_on_a_temp_root(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(notes, "_utcnow", lambda: NOW)
    root, log, src = tmp_path / "notes", tmp_path / "guardian.log", tmp_path / "src"
    _report(root, "late", "CHECKPOINT 1111111 (12:00Z) [open] x\n", 60 * 48)
    wt = tmp_path / "wt" / "late"
    wt.mkdir(parents=True)
    notes.bind(root, "late", final="12:00Z", worktree=str(wt))
    _set_mtime(root / "lanes" / "late" / "binding.json", NOW - dt.timedelta(hours=48))
    src.mkdir()
    (src / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = []\n')
    (root / "steward.toml").write_text(f'[[run]]\nname = "here"\nat = "00:00Z"\nsource = "{src}"\nargs = ["--", "true"]\n')
    log.write_text(f"2026-09-24T10:00:00+00:00 KILLED pid=7 mem=1.1GB reason=per-proc cap 1.0GB cwd={wt} cmd=python -m pytest\n")
    notes.save_state(root, {"guardian": {"path": str(log), "offset": 0, "ino": log.stat().st_ino}})
    assert notes.main(["watch", "--root", str(root), "--all", "--passes", "1", "--guardian-log", str(log)]) == 1    # late is STALE
    out = capsys.readouterr().out
    assert " OVERDUE late FINAL 12:00Z passed " in out and " ROUTED late 1: " in out and "pass 1: 1 open, 1 stale" in out
    assert ' RUN-FAILED here: args put the run on a machine: ["--on", MACHINE, ..., "--", CMD...]' in out
    state = notes.load_state(root)
    assert state["overdue"]["late"]["alerted"]
    assert notes.main(["watch", "--root", str(root), "--all", "--passes", "1", "--guardian-log", "none"]) == 1
    assert "OVERDUE" not in capsys.readouterr().out                             # said once, across watcher restarts too
