"""`research digest`: a lane's sync on one screen, every source faked and the clock fixed."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from research import digest as D

NOW = 1792300000.0
LANE = "memory-accounting"
GIB = 2**30


def row(pr: int, state: str, head: str = "a" * 40, owner: str = LANE, why: str = "", position: int | None = None) -> dict:
    return {"pr": pr, "head": head, "owner": owner, "state": state, "why": why, "position": position}


class Fake:
    """Sources over plain data; `fail` names the ones that raise."""

    def __init__(self) -> None:
        self.bells: list[dict] = []
        self.replies: dict[str, list[dict]] = {}
        self.main = ("m1", [])
        self.landing_times: list[float] = []
        self.rows: list[dict] = []
        self.states: dict[int, str] = {}
        self.note_files: tuple[str, list[str]] = ("n1", [])
        self.note_texts: dict[tuple[str, str], str] = {}
        self.vm = {"mem_total": 16 * GIB, "mem_available": 10 * GIB, "load": (0.5, 0.25, 0.125), "cpus": 4, "oom": 0,
                   "oom_error": "", "disks": [{"label": "/", "path": "/", "dev": 1, "pct": 8, "free": 234 * GIB}]}
        self.ws = {"branch": None, "head": "19f54cae0", "dirty": 0, "behind": 0}
        self.ages: dict[str, str] = {}
        self.inbox_list: str | None = ""
        self.prs: list[dict] = []
        self.lake_dirs: list[dict] = []
        self.run_rows: list[dict] = []
        self.fail: set[str] = set()

    def _check(self, name: str) -> None:
        if name in self.fail:
            raise RuntimeError(f"{name} is down")

    def _get(self, name: str, value):
        self._check(name)
        return value

    def sources(self) -> D.Sources:
        def bells(since: str) -> list[dict]:
            self._check("doorbells")
            return [m for m in self.bells if float(m["ts"]) > float(since)]

        def thread(channel: str, ts: str, since: str) -> list[dict]:
            return [m for m in self.replies.get(ts, []) if float(m["ts"]) > float(since)]

        def landings(last):
            self._check("landings")
            return self.main

        def queue():
            self._check("queue")
            return self.rows

        return D.Sources(doorbells=bells, thread=thread, landings=landings,
                         landed_since=lambda t: sum(1 for x in self.landing_times if x >= t), queue=queue,
                         pr_state=lambda n: self.states.get(n, ""), notes=lambda last: self.note_files,
                         note_text=lambda tip, path: self.note_texts.get((tip, path)),
                         vm=lambda: self._get("vm", self.vm), workspace=lambda: self._get("workspace", self.ws),
                         branch_age=lambda b: self._get("branch_age", self.ages.get(b)),
                         inbox=lambda: None if self.inbox_list is None else D.parse_inbox(self._get("inbox", self.inbox_list)),
                         lane_prs=lambda: self._get("lane_prs", self.prs), lakes=lambda: self._get("lakes", self.lake_dirs),
                         runs=lambda: self._get("runs", self.run_rows), now=lambda: NOW)


def stamp(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def health(out: list[str]) -> list[str]:
    return out[out.index("## Health") + 1:]


def test_a_doorbell_shows_once_and_only_from_the_last_hour_on_a_first_digest():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    f.bells = [{"ts": f"{NOW - 7200:.6f}", "text": "two hours ago"}, {"ts": f"{NOW - 60:.6f}", "text": "a minute ago"}]
    out, errors = D.digest(LANE, s, f.sources())
    assert not errors
    assert "## Doorbells (1)" in out and any("a minute ago" in x for x in out)
    assert not any("two hours ago" in x for x in out)
    out, _ = D.digest(LANE, s, f.sources())
    assert not any(x.startswith("## Doorbells") for x in out)


def test_a_thread_shows_other_handles_replies_but_not_the_lanes_own_nor_a_reply_its_doorbell_showed():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    s["threads"] = {"pairing": ["agent-coordination", "1.0"]}
    f.bells = [{"ts": f"{NOW - 10:.6f}", "text": f"from ci, thread 1.0, reply {NOW - 20:.6f}: @memory-accounting queue"}]
    f.replies["1.0"] = [
        {"ts": f"{NOW - 30:.6f}", "username": LANE, "text": "our own status"},
        {"ts": f"{NOW - 20:.6f}", "username": "ci", "text": "already rung"},
        {"ts": f"{NOW - 15:.6f}", "username": "compute-accounting", "text": "new from them"},
    ]
    out, _ = D.digest(LANE, s, f.sources())
    threads = out[out.index("## Threads") + 1:]
    assert len([x for x in threads if x.startswith("- pairing")]) == 1
    assert "new from them" in threads[0]


def test_landings_name_the_queue_merges_and_a_lane_pr_that_left_the_queue_says_whether_it_landed():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    f.rows = [row(10, "queued", position=3), row(11, "queued"), row(12, "waiting", owner="ci")]
    D.digest(LANE, s, f.sources())
    assert set(s["queue"]) == {"10", "11"}
    f.main = ("m2", ["Queue: #10 the first", "a direct commit", "Queue: #12 another lane's"])
    f.rows, f.states = [], {11: "CLOSED"}
    out, _ = D.digest(LANE, s, f.sources())
    assert "## Landed on main (2)" in out
    assert "- #10 landed" in out and "- #11 left the queue (CLOSED)" in out
    assert s["main"] == "m2" and s["queue"] == {}


def test_a_watched_pr_triggers_when_it_lands_or_its_state_changes_but_not_when_first_seen_or_only_its_head_moves():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    s["watch"] = {"20": "merge main into #21", "30": "re-mark #31"}
    f.rows = [row(20, "queued", owner="ci"), row(30, "queued", owner="ci")]
    out, _ = D.digest(LANE, s, f.sources())
    assert not any("TRIGGER" in x for x in out)
    f.rows = [row(20, "queued", owner="ci", head="b" * 40), row(30, "queued", owner="ci", head="c" * 40)]
    out, _ = D.digest(LANE, s, f.sources())
    assert not any("TRIGGER" in x for x in out)
    f.rows = [row(30, "refused", owner="ci", why="conflicts with next")]
    f.main = ("m2", ["Queue: #20 ci's change"])
    out, _ = D.digest(LANE, s, f.sources())
    assert "- TRIGGER #20 landed: merge main into #21" in out
    assert any(x.startswith("- TRIGGER #30 is now refused") and "re-mark #31" in x for x in out)


def test_a_reverted_landing_is_listed_and_leaves_landed_so_a_watch_on_it_does_not_trigger_as_landed():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    s["watch"] = {"40": "merge main into #41", "50": "start #51"}
    f.rows = [row(40, "queued", owner="ci"), row(50, "queued"), row(60, "queued")]
    D.digest(LANE, s, f.sources())
    f.rows = []
    f.main = ("m2", ["Queue: #40 ci's change", "Queue: #50 ours", "Queue: #60 ours", 'Revert "Queue: #40 ci\'s change"',
                     'Revert "Queue: #2382 landed before the last digest"', 'Revert "Queue: #60 ours"', "Queue: #60 ours, fixed"])
    out, errors = D.digest(LANE, s, f.sources())
    assert not errors
    i = out.index("## Reverted on main (3)")
    assert out[i + 1:i + 4] == ['- Revert "Queue: #40 ci\'s change"', '- Revert "Queue: #2382 landed before the last digest"',
                                '- Revert "Queue: #60 ours"']
    assert "- #50 landed" in out and "- #60 landed" in out
    assert "- TRIGGER #50 landed: start #51" in out
    assert not any(x.startswith("- TRIGGER #40") for x in out)


def test_the_queue_line_counts_every_lane_by_state_with_the_landings_of_the_last_six_hours():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    f.rows = [row(10, "queued"), row(11, "queued", owner="ci"), row(12, "waiting", owner="vllm")]
    f.landing_times = [NOW - 600, NOW - 5 * 3600, NOW - 7 * 3600]
    out, _ = D.digest(LANE, s, f.sources())
    assert "## Queue: 2 queued, 1 waiting; 2 landed in the last 6 h" in out
    f.rows = [row(10, "queued", head="d" * 40)]
    out, _ = D.digest(LANE, s, f.sources())
    assert any(x.startswith("- #10 dddddddd queued (new head)") for x in out)


def test_a_failing_source_is_reported_keeps_its_cursor_and_hides_nothing_else():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    s["doorbell"] = f"{NOW - 100:.6f}"
    f.fail = {"doorbells"}
    f.rows = [row(10, "queued")]
    f.note_files = ("n2", ["lanes/memory-accounting/20261018T0500Z-report-x.md"])
    out, errors = D.digest(LANE, s, f.sources())
    assert errors == ["doorbells: doorbells is down"]
    assert s["doorbell"] == f"{NOW - 100:.6f}"
    assert "## Our PRs: 1 queued" in out and "- 20261018T0500Z-report-x.md" in out


def test_a_changed_note_shows_three_lines_of_its_result_so_far_at_the_tip_and_a_note_without_one_only_its_name():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    d = f"lanes/{LANE}"
    f.note_files = ("n2", [f"{d}/a-report.md", f"{d}/b-draft.md", f"{d}/c-gone.md"])
    f.note_texts = {
        ("n2", f"{d}/a-report.md"): "---\nid: a\n---\n# Report\n\n## Result so far\n\nfirst\n\nsecond\nthird\n"
                                    "fourth, not shown\n\n## Next\nnot shown\n",
        ("n1", f"{d}/b-draft.md"): "## Result so far\nan older tip's text, not shown\n",
        ("n2", f"{d}/b-draft.md"): "# Draft\n\nno result section\n",
    }
    out, errors = D.digest(LANE, s, f.sources())
    assert not errors
    notes = out[out.index("## Notes changed (3)") + 1:out.index("## Health")]
    assert notes == ["- a-report.md", "    first", "    second", "    third", "- b-draft.md", "- c-gone.md"]
    assert D.result_so_far("## Result so far (17 Oct)\none\n## Next\ntwo\n") == ["one"]
    assert D.result_so_far(None) == []


def test_result_so_far_reads_the_first_such_heading_of_level_two_to_four_and_stops_at_the_next_heading_of_any_level():
    text = ("# Result so far, a title\nnot shown\n##### Result so far, too deep\nnot shown\n\n"
            "### result so far (round 2)\n\nnewest\nnext newest\n## Result so far (round 1)\nstale\n")
    assert D.result_so_far(text) == ["newest", "next newest"]
    assert D.result_so_far("#### Result so far\none\n##### detail\ntwo\n") == ["one"]


def test_the_roster_parser_reads_tab_separated_workers_and_names_a_bad_line():
    text = "# agent_id\tlabel\tbranch\n\nbc-1\tLean prover\tcursor/a-93c1\n  bc-2\tgrader\tcursor/b-93c1  \n"
    assert D.parse_roster(text) == [("bc-1", "Lean prover", "cursor/a-93c1"), ("bc-2", "grader", "cursor/b-93c1")]
    with pytest.raises(ValueError, match=r"^r.tsv:2: want agent_id<TAB>label<TAB>branch"):
        D.parse_roster("bc-1\tok\tcursor/a\nbc-2 grader cursor/b\n", "r.tsv")
    with pytest.raises(ValueError, match=r"^roster:1: "):
        D.parse_roster("bc-1\t\tcursor/a\n")


def test_health_shows_the_vm_workspace_workers_inboxes_prs_off_the_queue_finished_lakes_and_live_runs():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    f.vm["oom"], f.vm["oom_error"] = None, "dmesg: read kernel buffer failed: Operation not permitted; it may need root"
    f.vm["disks"] += [{"label": "notes", "path": "/home/u/.research/notes"},
                      {"label": "workspace", "path": "/workspace", "dev": 1, "pct": 8, "free": 234 * GIB}]
    f.ws = {"branch": "cursor/coord-93c1", "head": "abc1234", "dirty": 2, "behind": 7}
    f.ages = {"cursor/a-93c1": "3 hours ago 1a2b3c4"}
    f.inbox_list = (f"pending   bc-1  312 bytes  since {stamp(NOW - 45 * 60)}\ndelivered bc-1  80 bytes\n"
                    f"pending   bc-2  90 bytes  since {stamp(NOW - 5 * 60)}\npending   bc-9  40 bytes\n")
    f.rows = [row(10, "queued")]
    f.prs = [{"number": 10, "title": "queued", "baseRefName": "main"},
             {"number": 2463, "title": "Properties.PoubCheck", "baseRefName": "cursor/x-93c1", "isDraft": True}]
    f.lake_dirs = [{"path": f"/w/wt{i}/Security/.lake", "bytes": i * GIB, "branch": f"cursor/w{i}", "age_h": 6.5 + i}
                   for i in range(1, 7)]
    f.run_rows = [{"id": "r-live", "state": "running", "remote": {"machine": "nebius-1"}, "command": ["bash", "-lc", "uv run x"]},
                  {"id": "r-wait", "state": "waiting", "remote": None, "command": ["uv", "run", "y"]},
                  {"id": "r-done", "state": "done", "remote": None, "command": ["z"]}]
    roster = [("bc-1", "prover", "cursor/a-93c1"), ("bc-2", "grader", "cursor/gone-93c1")]
    out, errors = D.digest(LANE, s, f.sources(), roster)
    assert not errors
    h = health(out)
    assert h[0] == ("- VM: memory 6.0 GiB used, 10.0 GiB available; load 0.50 0.25 0.12 on 4 CPUs; disk /, workspace 8% used, "
                    "234.0 GiB free; notes /home/u/.research/notes absent; OOM kills unreadable (dmesg: read kernel buffer "
                    "failed: Operation not permitted; it may need root)")
    assert h[1] == ("- WARNING /workspace on branch cursor/coord-93c1, dirty (2 paths), 7 behind origin/main: the shell hook "
                    "and the inbox run from this checkout")
    assert h[2] == ("- WARNING worker prover bc-1: cursor/a-93c1 3 hours ago 1a2b3c4; inbox: 312 bytes pending since "
                    f"{D.hhmm(NOW - 45 * 60)} (45 min), over 30 min")
    assert h[3] == (f"- worker grader bc-2: no branch cursor/gone-93c1 on origin; inbox: 90 bytes pending since "
                    f"{D.hhmm(NOW - 5 * 60)} (5 min)")
    assert h[4] == "- bc-9 (not in the roster): inbox: 40 bytes pending since an unknown time"
    assert h[5] == "- WARNING #2463 is not in the queue (base cursor/x-93c1, draft): Properties.PoubCheck"
    assert h[6] == "- .lake of finished worktrees, the largest 5 (listed, not deleted):"
    assert h[7:12] == [f"    {i}.0 GiB /w/wt{i}/Security/.lake (cursor/w{i}, tip {6.5 + i:.0f} h old)" for i in range(6, 1, -1)]
    assert h[12:] == ["- run r-live running on nebius-1: bash -lc uv run x", "- run r-wait waiting on local: uv run y"]


def test_a_failing_health_check_prints_its_error_in_its_place_and_hides_nothing_else():
    f, s = Fake(), D.load(D.state_path(LANE, "/nonexistent/x.json"))
    f.fail = {"vm", "branch_age", "queue"}
    f.inbox_list = f"pending   bc-1  10 bytes  since {stamp(NOW - 60)}\n"
    f.prs = [{"number": 11, "title": "t", "baseRefName": "main"}]
    f.run_rows = [{"id": "r-live", "state": "running", "command": "bash -lc 'uv run x'"}]
    out, errors = D.digest(LANE, s, f.sources(), [("bc-1", "prover", "cursor/a-93c1")])
    h = health(out[:out.index("## Errors")])
    assert h[0] == "- VM: FAILED: vm is down"
    assert h[1] == "- /workspace detached at 19f54cae0, clean, 0 behind origin/main"
    assert h[2].startswith("- worker prover bc-1: branch cursor/a-93c1: FAILED: branch_age is down; inbox: 10 bytes pending")
    assert h[3] == "- PRs: FAILED: the queue is unread (see Errors)"
    assert h[4:] == ["- .lake: none in finished worktrees (tip on origin, over 6 h old)",
                     "- run r-live running on local: bash -lc 'uv run x'"]
    assert errors == ["queue: queue is down", "health VM: vm is down", "health branch cursor/a-93c1: branch_age is down",
                      "health PRs: the queue is unread (see Errors)"]
    assert out[out.index("## Errors") + 1:] == ["- queue: queue is down"]
    f.fail, f.inbox_list = set(), None
    out, errors = D.digest(LANE, s, f.sources(), [("bc-1", "prover", "cursor/a-93c1")])
    assert not errors and "- inbox: no tools/agent-guard/inbox.py in /workspace" in health(out)
    assert "- worker prover bc-1: no branch cursor/a-93c1 on origin; inbox unread" in health(out)
    assert "- WARNING #11 is not in the queue (base main): t" in health(out)
    f.rows = [row(11, "waiting")]
    out, _ = D.digest(LANE, s, f.sources())
    assert "- PRs: all 1 of the lane's open PRs are in the queue" in health(out)


def test_the_watch_list_and_threads_are_edited_in_the_state_file(tmp_path, capsys):
    p = tmp_path / "d.json"
    assert D.main(["--as", f"@{LANE}", "--state", str(p), "--watch", "#2446", "merge main into #2421"]) == 0
    assert D.main(["--as", LANE, "--state", str(p), "--thread", "cards", "9.5", "--channel", "ask-daniel"]) == 0
    s = json.loads(p.read_text())
    assert s["watch"] == {"2446": "merge main into #2421"} and s["threads"] == {"cards": ["ask-daniel", "9.5"]}
    assert D.main(["--as", LANE, "--state", str(p), "--unwatch", "2446", "--drop-thread", "cards"]) == 0
    s = json.loads(p.read_text())
    assert s["watch"] == {} and s["threads"] == {}
    capsys.readouterr()
