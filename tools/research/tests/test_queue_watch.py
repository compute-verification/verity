"""`research queue watch`: landing's watchdog pages @ci when `main` stalls, the ready-to-landed p90 is over 2 h, drive stops
ticking, or a train check stalls in its slot or waits unstarted beside a free one; each breach once, again every 30 min while it
lasts, and a resolved line."""
import json
import subprocess

import pytest

from research import queue, queue_stats as qs, queue_watch as W
from research.queue import Member

T = 1792800000.0                                   # 2026-10-24T00:00:00Z


def at(t: float) -> str:
    return W.iso(t)[:-1] + ":00Z"


def drive_log(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def tick(t: float, rc: int = 0) -> str:
    return f"{at(t)} tick rc={rc} e50f71f76"


class Fake:
    """Readers over values a test sets, and the posts `page` sends."""

    def __init__(self, log="", main="m" * 40, main_time=T - 7200, waiting=(), heads=None, marks=None, drive=None, runs=None,
                 slots=None):
        self.log, self.main, self.main_time, self.waiting, self.heads = log, main, main_time, list(waiting), heads or {}
        self.marks, self.drive, self.runs, self.slots = marks or {}, drive or {}, runs or {}, slots or {}
        self.fail: set[str] = set()
        self.posts: list[str] = []
        self.post_ok = True
        self.asked_slots: list[list[str]] = []

    def _maybe(self, name):
        if name in self.fail:
            raise OSError(f"{name} unreadable")

    def readers(self) -> W.Readers:
        def slots(ms):
            self.asked_slots.append(ms)
            return {m: self.slots.get(m, []) for m in ms}
        return W.Readers(
            log=lambda: (self._maybe("log"), (self.log, None))[1],
            drive=lambda: (self._maybe("drive"), self.drive)[1],
            repo=lambda: (self._maybe("repo"), W.Repo(self.main, self.main_time, self.waiting, self.heads))[1],
            marks=lambda ts: {k: self.marks[k] for k in ts if k in self.marks},
            runs=lambda ids: {i: self.runs.get(i) for i in ids},
            slots=slots)

    def post(self, text: str) -> bool:
        self.posts.append(text)
        return self.post_ok

    def run(self, state: dict, now: float) -> tuple[list[W.Breach], str | None]:
        found = W.watch(self.readers(), state, now)
        return found, W.page(state, found, now, self.post)


def waiting(*prs: int) -> list[Member]:
    return [Member(pr, f"{pr:040x}", "ci", f"c{pr:039x}") for pr in prs]


def test_main_still_for_an_hour_with_prs_waiting_pages_once_again_after_30_min_and_says_when_it_resolved():
    """24 Oct: `main` stood still 01:38Z-05:15Z with 70+ ready PRs waiting, and nothing alerted."""
    f = Fake(log=drive_log(tick(T)), main_time=T, waiting=waiting(1, 2, 3))
    state: dict = {}
    assert f.run(state, T) == ([], None)
    assert state["main"] == {"sha": f.main, "moved": T, "checked": T}
    f.log += tick(T + W.MAIN_STALL_S - 60) + "\n"
    found, _ = f.run(state, T + W.MAIN_STALL_S - 60)
    assert found == [] and f.posts == []
    f.log += tick(T + W.MAIN_STALL_S) + "\n"
    found, sent = f.run(state, T + W.MAIN_STALL_S)
    assert [b.key for b in found] == ["main"] and sent == f.posts[-1]
    assert "`main` hasn't moved for 60 min while 3 ready PR(s) wait" in sent and sent.startswith(W.HEADER)
    f.log += tick(T + W.MAIN_STALL_S + 300) + "\n"
    f.run(state, T + W.MAIN_STALL_S + 300)
    assert len(f.posts) == 1                       # once per breach...
    f.log += tick(T + W.MAIN_STALL_S + W.REPEAT_S) + "\n"
    _, sent = f.run(state, T + W.MAIN_STALL_S + W.REPEAT_S)
    assert len(f.posts) == 2 and "(still, since 01:00Z)" in sent   # ...again every REPEAT_S while it lasts
    f.main, f.main_time = "n" * 40, T + 5000
    _, sent = f.run(state, T + W.MAIN_STALL_S + W.REPEAT_S + 60)
    assert sent == W.HEADER + "\n- resolved after 31 min: `main` hasn't moved for 90 min while 3 ready PR(s) wait on `next`"
    assert state["open"] == {}


def test_main_counts_as_moved_when_drive_landed_it_not_when_its_merge_was_made():
    """A train's merge commit is hours older than its landing: its commit time alone would page a `main` that just moved."""
    landed = drive_log(f"research queue drive: landed #7 on `main` at {'a' * 12} (check r20261024-002000-0001)", tick(T - 60))
    landings = qs.parse(landed)[1]
    repo = W.Repo("a" * 40, T - 6 * 3600)
    assert W.main_moved(repo, landings, {}, T)["moved"] == T - 60
    assert W.main_moved(repo, [], {"sha": "b" * 40, "moved": T - 9000, "checked": T - 300}, T)["moved"] == T - 300
    assert W.main_moved(repo, [], {"sha": "a" * 40, "moved": T - 9000, "checked": T - 300}, T)["moved"] == T - 9000
    assert W.main_stalled({"sha": "a" * 40, "moved": T - 9000}, 0, T) is None    # nothing waits: a still `main` is fine


def test_the_waiting_count_falls_back_to_drives_last_sync_line_when_next_holds_no_queue_chain():
    log = drive_log("research queue: `next` at 3fcbf81357d2: 84 PR(s) above `main`, kept 30, merged 54", tick(T))
    f = Fake(log=log, main_time=T - W.MAIN_STALL_S)
    found, _ = f.run({}, T)
    assert [b.title for b in found] == ["`main` hasn't moved for 60 min while 84 ready PR(s) wait on `next`"]


def test_p90_counts_landed_prs_from_their_marks_and_waiting_prs_by_their_age():
    log = drive_log(f"research queue drive: landed #1 #2 on `main` at {'a' * 12} (check r20261024-000000-0001)", tick(T - 600))
    heads = {1: "1" * 40, 2: "2" * 40, 9: "9" * 40}
    marks = {(1, "1" * 40): T - 3600, (2, "2" * 40): T - 1800}
    f = Fake(log=log, main="a" * 40, main_time=T - 600, heads=heads, marks=marks)
    assert f.run({}, T)[0] == []
    f.waiting = waiting(3)
    f.marks[(3, f"{3:040x}")] = T - 3 * 3600                # waiting 3 h: the p90 of (50 min, 20 min, 3 h)
    found, sent = f.run({}, T)
    assert [b.key for b in found] == ["p90"]
    assert "ready-to-landed p90 is 3.0 h, over the 2 h SLO" in sent
    assert "over 3 PR(s): 2 landed in the last 6 h, 1 waiting; the longest #3 3.0 h waiting, #1 0.8 h, #2 0.3 h" in sent
    late = T + 7 * 3600                                        # #1 and #2 landed out of the window: only #3 counts
    assert W.slow_landings(late, 6, qs.parse(log)[1], W.Repo("m" * 40, T, [], heads), f.readers().marks) is None


def test_a_drive_that_stopped_ticking_pages_and_names_its_stop_file():
    f = Fake(log=drive_log(tick(T - W.TICK_STALE_S), f"{at(T - 1500)} stopped: /data/drive/stopped exists (drive.sh start)"),
             main_time=T)
    found, sent = f.run({}, T)
    assert [b.key for b in found] == ["tick"]
    assert "no drive tick has ended for 30 min: drive is stopped (/data/drive/stopped)" in sent
    assert f"stopped since {at(T - 1500)[:-4]}Z" in sent
    assert [b.key for b in f.run({}, T - 60)[0]] == []
    assert W.tick_stale("", None, T).title == "no drive tick has ended in the tail of drive's log"


def run(state="running", slot="held", machine="vy-nebius-cpu-1", wrote=T, note=""):
    return W.Run(state, {"state": slot, "slot": "check-1", "took": "2026-10-24T00:00:00Z"} if slot else {}, machine, wrote, note)


def test_a_check_that_holds_its_slot_and_writes_nothing_for_10_min_pages_with_its_train():
    """24 Oct: n2's Quiet SIGSTOPped a train's check in its slot, sampler and all, and `main` waited on it."""
    c = "3fcbf81357d2" + "0" * 28
    log = drive_log("research queue drive: checking 3fcbf81357d2 (#1 #2): r20261024-010000-aaaa with agreement", tick(T))
    drive = {"drive": {"launched": {c: {"at": T - 3600, "ids": ["r20261024-010000-aaaa"], "touched": ["r20261024-010400-bbbb"]}}}}
    f = Fake(log=log, main_time=T, drive=drive, runs={"r20261024-010000-aaaa": run(wrote=T - W.STALLED_S, note="mem 3 GiB"),
                                                       "r20261024-010400-bbbb": run(wrote=T - 60)})
    found, sent = f.run({}, T)
    assert [b.key for b in found] == ["stalled:r20261024-010000-aaaa"]
    assert ("train 3fcbf81357d2 (2 PR(s), up to #2)'s check r20261024-010000-aaaa holds vy-nebius-cpu-1's slot check-1 and has "
            "written nothing in its run dir for 10 min") in sent and "mem 3 GiB" in sent
    f.runs["r20261024-010400-bbbb"] = run(wrote=None)
    assert [b.key for b in f.run({}, T)[0]] == ["stalled:r20261024-010000-aaaa", "stalled:r20261024-010400-bbbb"]
    f.runs = {"r20261024-010000-aaaa": run(slot="waiting", wrote=T - 9999), "r20261024-010400-bbbb": run(state="passed", wrote=0)}
    assert W.stalled_checks(drive["drive"]["launched"], f.runs, {}, T) == []


def test_a_check_unstarted_for_30_min_beside_a_free_slot_pages_only_once_it_held_two_passes():
    c = "d" * 40
    drive = {"drive": {"launched": {c: {"at": T - W.UNSTARTED_S, "ids": ["r1", "r2"]}}}}
    log = drive_log("research queue drive: flags from tools/verity/check/drive.toml at main e50f71f76277: on=cpu1,cpu2 "
                    "max_running=8", tick(T))
    f = Fake(log=log, main_time=T, drive=drive, runs={"r1": run(slot="waiting", machine="cpu1"), "r2": run(slot="waiting", machine="cpu2")},
             slots={"cpu1": [{"name": "check-1", "holder": "r9"}], "cpu2": [{"name": "check-1", "holder": None, "blocked": "window"}]})
    state: dict = {}
    assert f.run(state, T)[0] == [] and f.asked_slots == [["cpu1", "cpu2"]]      # every slot taken or blocked
    f.slots["cpu2"] = [{"name": "check-2", "holder": None}]
    found, sent = f.run(state, T)
    assert [b.key for b in found] == [f"unstarted:{c}"] and sent is None          # a snapshot: it must hold UNSTARTED_HOLD_S
    _, sent = f.run(state, T + W.UNSTARTED_HOLD_S)
    assert "train dddddddddddd's check has waited 34 min unstarted while a check slot is free" in sent
    assert "its twins wait on cpu1, cpu2; free now: cpu2 check-2" in sent
    f.runs["r2"] = run(slot="held", machine="cpu2")                                 # a twin took it
    assert f.run(state, T + W.UNSTARTED_HOLD_S + 60)[0] == []
    f.drive["drive"]["launched"][c]["at"] = T
    f.runs["r2"] = run(slot="waiting", machine="cpu2")
    assert W.unstarted(f.drive["drive"]["launched"], f.runs, T + W.UNSTARTED_S - 1) == {}


def test_a_reader_that_fails_skips_only_its_checks_and_pages_once_it_failed_for_30_min():
    f = Fake(log=drive_log(tick(T - W.TICK_STALE_S)), main_time=T)
    f.fail = {"repo"}
    state: dict = {}
    found, sent = f.run(state, T)
    assert [b.key for b in found] == ["unread:the repository (main and next)", "tick"]
    assert "the landing watch can't read" not in sent and "no drive tick has ended" in sent
    _, sent = f.run(state, T + W.UNREAD_S)
    assert "- the landing watch can't read the repository (main and next): OSError: repo unreadable" in sent


def test_a_post_that_fails_is_sent_again_next_pass_with_its_resolved_lines():
    f = Fake(log=drive_log(tick(T - W.TICK_STALE_S)), main_time=T)
    state: dict = {}
    f.run(state, T)
    f.log += tick(T) + "\n"
    f.post_ok = False
    assert f.run(state, T + 60) == ([], None) and state["open"]["tick"]["paged"] == T
    f.post_ok = True
    _, sent = f.run(state, T + 120)
    assert sent == W.HEADER + "\n- resolved after 2 min: no drive tick has ended for 30 min" and state["open"] == {}
    f.log = drive_log(tick(T - W.TICK_STALE_S))
    assert f.run(state, T + 180)[1] is None and len(f.posts) == 3   # back within REPEAT_S of its last page: not paged yet
    f.post_ok = False
    f.run(state, T + W.REPEAT_S)
    assert len(f.posts) == 4 and state["paged"]["tick"] == T and "paged" not in state["open"]["tick"]
    f.post_ok = True
    _, sent = f.run(state, T + W.REPEAT_S + 300)
    assert "no drive tick has ended for 65 min" in sent and state["paged"]["tick"] == T + W.REPEAT_S + 300


def test_free_slots_skip_held_and_blocked_slots_and_unread_machines():
    docs = {"a": [{"name": "s1", "holder": "r"}, {"name": "s2", "holder": None}], "b": "no answer in 20 s",
            "c": [{"name": "s1", "holder": None, "blocked": "lean window"}]}
    assert W.free_slots(docs) == [("a", "s2")]


def test_read_log_reads_the_tail_from_a_whole_line(tmp_path, monkeypatch):
    p = tmp_path / "drive.log"
    p.write_text("first line\n" + tick(T) + "\n")
    assert W.read_log(p)[0].startswith("first line")
    monkeypatch.setattr(W, "LOG_TAIL", len(tick(T)) + 4)
    assert W.read_log(p)[0] == tick(T) + "\n"


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True, capture_output=True)


def test_read_repo_fetches_main_and_next_and_reads_the_heads_main_landed_and_the_prs_next_holds(tmp_path):
    origin, clone = tmp_path / "origin", tmp_path / "clone"
    origin.mkdir()
    git(origin, "init", "-q", "-b", "main")
    git(origin, "commit", "-q", "--allow-empty", "-m", "root")
    git(origin, "commit", "-q", "--allow-empty", "-m", f"Merge #5\n\nPR: #5\nHead: {'5' * 40}\nOwner: ci")
    git(origin, "checkout", "-q", "-b", "next")
    git(origin, "commit", "-q", "--allow-empty", "-m", f"Merge #6\n\nPR: #6\nHead: {'6' * 40}\nOwner: ci")
    git(tmp_path, "clone", "-q", "--no-checkout", str(origin), str(clone))
    git(origin, "commit", "-q", "--allow-empty", "-m", f"Merge #7\n\nPR: #7\nHead: {'7' * 40}\nOwner: ci")
    repo = W.read_repo(clone)
    assert repo.heads == {5: "5" * 40}
    assert [(m.pr, m.head) for m in repo.waiting] == [(6, "6" * 40), (7, "7" * 40)]    # fetched this pass


def test_the_cli_refuses_to_page_as_ci_whose_own_posts_never_ring_its_inbox(capsys):
    with pytest.raises(SystemExit) as e:
        queue.main(["watch", "--as", "ci", "--once"])
    assert e.value.code == 2 and "--as ci" in capsys.readouterr().err


def test_dry_run_prints_the_page_and_saves_no_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(W, "read_repo", lambda p: W.Repo("m" * 40, T, waiting(1)))
    monkeypatch.setattr(W, "read_marks", lambda cache, ts: {})
    log = tmp_path / "drive.log"
    log.write_text(tick(T + W.MAIN_STALL_S) + "\n")
    (tmp_path / "drive.json").write_text(json.dumps({}))
    rc = W.main(["--once", "--dry-run", "--log", str(log), "--drive-state", str(tmp_path / "drive.json"), "--repo",
                 str(tmp_path / "repo"), "--state", str(tmp_path / "w" / "state.json"), "--drive-runs", str(tmp_path / "runs")],
                now=lambda: T + W.MAIN_STALL_S)
    out = capsys.readouterr().out
    assert rc == 1 and "`main` hasn't moved for 60 min while 1 ready PR(s) wait" in out and "queue watch: main" in out
    assert not (tmp_path / "w").exists()
