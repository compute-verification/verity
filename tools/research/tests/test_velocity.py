"""`research velocity` on two tiny dumps of the Goals page taken six hours apart: what finished and what moved, by scope; the
stalls; the other rows from stand-ins for their records; and the command itself, which exits 0 and says `not read` for a
row it can't read."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from research import velocity as V

FIX = Path(__file__).parent / "fixtures" / "velocity"
A, B = FIX / "goals-20261109T2000Z.json", FIX / "goals-20261110T0200Z.json"
NOW = datetime(2026, 11, 10, 2, 0, tzinfo=UTC)       # when B was taken, 6 PM Pacific
REG = {"handles": [{"handle": "top", "linear": "TOP"}, {"handle": "kernels", "linear": "HASH"},
                   {"handle": "compute-accounting", "linear": "CMP"},
                   {"handle": "compute-research", "linear": "CMP", "linear_label": "lane:compute-research"}]}


def goals(path: Path) -> list[dict]:
    return json.loads(path.read_text())


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No pod, no Linear, no queue table, and a `velocity.json` that names no dump but A as the baseline."""
    def no_pod(cmd, timeout=0):
        raise V.Unread("a test has no pod")

    def no_landing(since, until, live):
        raise V.Unread("a test has no queue table")
    data = tmp_path / "velocity.json"
    data.write_text(json.dumps({"baseline": str(A), "dumps": {}}))
    monkeypatch.setattr(V, "DATA", data)
    monkeypatch.setattr(V, "pod", no_pod)
    monkeypatch.setattr(V, "landing", no_landing)
    monkeypatch.setattr(V, "read_asks", lambda now: pytest.fail("a window that ended is not read from Linear"))


def test_what_finished_and_what_moved_between_two_dumps_by_scope():
    a, b = goals(A), goals(B)
    # set by hand: delta finished, alpha came a day earlier, beta went from the end of the 13th (Pacific) to noon of the 14th
    assert V.standing(b, "hand") == {"goals": 5, "open": 3, "off_track": 1}
    assert V.moved(a, b, "hand") == {"finished": 1, "earlier": 1, "earlier_days": 1.0, "later": 1, "later_days": 0.5,
                                     "net_days": 0.5, "new_done": 1, "removed": 0}
    # priority 1 adds their steps and zeta, which finished a day after its first date
    assert V.standing(b, "p1") == {"goals": 9, "open": 6, "off_track": 1}
    assert V.moved(a, b, "p1") == {"finished": 2, "earlier": 1, "earlier_days": 1.0, "later": 2, "later_days": 1.5,
                                   "net_days": -0.5, "new_done": 1, "removed": 0}
    # every goal adds eta, two days earlier, and theta, which was open and left the page
    assert V.standing(b, "all") == {"goals": 11, "open": 7, "off_track": 1}
    assert V.moved(a, b, "all") == {"finished": 2, "earlier": 2, "earlier_days": 3.0, "later": 2, "later_days": 1.5,
                                    "net_days": 1.5, "new_done": 1, "removed": 1}
    assert V.moved(a, a, "all") == {"finished": 0, "earlier": 0, "earlier_days": 0, "later": 0, "later_days": 0, "net_days": 0,
                                    "new_done": 0, "removed": 0}


def test_a_goal_stalls_on_a_step_past_its_hour_and_not_updated_or_with_no_step_due_within_a_day():
    b = goals(B)
    # beta waits, through its piece, on a step due at 3 PM that nobody touched since; gamma is due in 11 days and waits on nothing
    assert V.stalls(b, NOW) == [{"goal": "top/beta", "why": "past its hour and not updated since", "steps": ["top/beta-step"]},
                                {"goal": "top/gamma", "why": "no step due within 24 hours", "steps": []}]
    # half an hour past its hour a step is not dead yet, and alpha's step, due at 5:30 PM, is within its hour of grace at 6 PM
    assert [s["goal"] for s in V.stalls(b, NOW - timedelta(hours=2, minutes=30))] == ["top/gamma"]
    touched = [g | {"updatedAt": "2026-11-09T23:30:00.000Z"} if g["id"] == "top/beta-step" else g for g in b]
    assert [s["goal"] for s in V.stalls(touched, NOW)] == ["top/gamma"]
    assert V.stalls([g | {"prioritySet": False} for g in b], NOW) == []


def test_agents_are_counted_by_lane_from_their_names():
    listed = [("TOP-403 SCORE: the score", "RUNNING"), ("HASH-112 KHSHARE", "RUNNING"), ("CMP-385 NAME", "RUNNING"),
              ("verity: kernels", "RUNNING"), ("top", "RUNNING"), ("CERTNEG: negative tests (NET-340)", "RUNNING"),
              ("XYZ-1 a team with no lane", "RUNNING"), ("TOP-9 stopped", "IDLE")]
    text = "Listing cloud agents: 8 shown.\n" + json.dumps({"hasMore": False, "agents": [{"name": n, "status": s} for n, s in listed]})
    assert V.agents(text, REG) == {"running": 7, "limit": 100, "more": False,
                                   "lanes": {"kernels": 2, "top": 2, "unnamed": 2, "compute-accounting": 1}}
    with pytest.raises(V.Unread, match="list-cloud-agents"):
        V.agents("no listing here", REG)


def test_machines_are_the_fleets_report_over_the_pods_samples_of_the_window(monkeypatch):
    def sample(t: str, name: str, busy: bool, rate: float) -> str:
        return json.dumps({"t": t, "name": name, "read": True, "gpu_util": [90 if busy else 0], "load": 8 if busy else 0, "cores": 16,
                           "mem_used_gib": 100 if busy else 1, "mem_total_gib": 200, "rate_usd_per_hour": rate, "preset": "p"})
    hour = [f"2026-11-09T21:{m:02d}:00Z" for m in range(0, 60, 5)]
    lines = [sample(t, "vm-busy", True, 12.0) for t in hour] + [sample(t, "vm-idle", False, 6.0) for t in hour]
    lines.append(sample("2026-11-10T02:00:00Z", "vm-busy", True, 12.0))      # at the window's end: outside it
    asked = []

    def pod(cmd, timeout=0):
        asked.append(cmd)
        return "\n".join(lines) + "\n"
    monkeypatch.setattr(V, "pod", pod)
    since = NOW - timedelta(hours=6)
    assert V.machines(since, NOW) == {"usd": 18, "idle_usd": 6, "vms": 2, "no_rate": []}
    assert asked == ["cat /data/watchers/fleet/samples-20261109.jsonl /data/watchers/fleet/samples-20261110.jsonl 2>/dev/null; true"]
    monkeypatch.setattr(V, "pod", lambda cmd, timeout=0: "")
    with pytest.raises(V.Unread, match="holds no sample of the window"):
        V.machines(since, NOW)

    def unreachable(cmd, timeout=0):
        raise V.Unread("the pod isn't reached from here")
    monkeypatch.setattr(V, "pod", unreachable)
    with pytest.raises(V.Unread, match=r"fleet.py report --hours 6 --now 2026-11-10T02:00:00Z"):
        V.machines(since, NOW)


def test_the_pod_is_asked_over_one_ssh_line_a_reading_and_run_here_when_this_machine_is_the_pod(monkeypatch, tmp_path):
    from research import remote
    from research.pods import connect

    asked, ran = [], []

    def ssh_base(target):
        asked.append(target)
        return ["ssh", "root@pod"]

    def run(argv, **kw):
        ran.append(argv)
        return V.subprocess.CompletedProcess(argv, 0 if argv[-1] != "false" else 1, stdout="out\n", stderr="no such file")
    monkeypatch.setattr(V, "SWEEP_GOALS", str(tmp_path / "watchers" / "goals"))
    monkeypatch.setattr(connect, "ssh_base", ssh_base)
    monkeypatch.setattr(V.subprocess, "run", run)
    V.control_ssh.cache_clear()
    assert V.pod("ls goals") == V.pod("cat samples") == "out\n"
    assert ran == [["ssh", "root@pod", "ls goals"], ["ssh", "root@pod", "cat samples"]] and asked == [V.CONTROL]
    with pytest.raises(V.Unread, match="isn't reached from here \\(no such file\\)"):
        V.pod("false")
    (tmp_path / "watchers").mkdir()                              # this machine is the pod: its own shell
    assert V.pod("ls goals") == "out\n" and ran[-1] == ["sh", "-c", "ls goals"]
    (tmp_path / "watchers").rmdir()
    V.control_ssh.cache_clear()

    def refused(target):
        raise remote.Refused("runpod API unavailable: HTTP 429\n")
    monkeypatch.setattr(connect, "ssh_base", refused)
    with pytest.raises(V.Unread, match="vy-control-verity isn't reached from here \\(runpod API unavailable: HTTP 429\\)"):
        V.pod("ls goals")
    V.control_ssh.cache_clear()


def test_asks_off_track_are_the_audits_findings_on_the_asks_not_delivered():
    def ask(issue: str, kind: str, **more) -> dict:
        return {"issue": issue, "url": "", "title": issue, "team": "TOP", "lane": "top", "asker": "daniel", "state": kind, "type": kind,
                "due": None, "created": NOW.timestamp() - 3600, "closed": None, "dates": [], "slips": 0, "explained": False,
                "again": 0, "postmortems": 0, "ok": False} | more
    rows = [ask("TOP-1", "backlog"), ask("TOP-2", "started"), ask("TOP-3", "completed"), ask("TOP-4", "started", again=1)]
    assert V.asks(rows, NOW.timestamp()) == {"open": 3, "off_track": 2, "kinds": {"parked": 1, "repeated": 1}, "issues": ["TOP-1", "TOP-4"]}


def test_landing_is_the_queue_tables_long_window_started_at_the_readings_start(monkeypatch):
    from research import queue_table

    def stamp(t: datetime) -> str:
        return f"{t:%Y-%m-%dT%H:%M:%SZ}"

    def landed(t: datetime, pr: int, head: str) -> dict:
        return {"action": "advance", "at": stamp(t), "sha": head * 40, "shadow": False, "done": True, "members": [pr], "heads": [head]}

    def marked(t: datetime) -> list[dict]:
        return [{"key": "ready", "value": True, "by": "x", "ts": stamp(t)}]
    # two landings in the table's 24 hours: one 8 h ago, 30 min after its mark, and one 2 h ago, 5 h after its mark
    early, late = NOW - timedelta(hours=8), NOW - timedelta(hours=2)
    doc = {"day": stamp(NOW - timedelta(hours=24)), "errors": ["vy-nebius-cpu-3: no answer in 100 s", "gh pr list: no list of open PRs"],
           "check_nodes": [], "off_gate": [], "nodes": {}, "open": {}, "lean": {},
           "control": {"decisions": [landed(early, 1, "a"), landed(late, 2, "b")], "ticks": [], "args": "", "executor": []},
           "labels": {"1@a": marked(early - timedelta(minutes=30)), "2@b": marked(late - timedelta(hours=5))}}
    asked = []

    def collect(store_root=None, now=None, until=None):
        asked.append((now, until))
        return doc | {"now": until or now}
    monkeypatch.setattr(queue_table, "collect", collect)
    six = V.landing(NOW - timedelta(hours=6), NOW, live=False)                # the table's own rows, over the reading's window
    assert six == {"share": "0 of 1 landed (0%)", "mark_to_main": "p50 300, p90 300 min (n=1)", "unread": ["gh pr list: no list of open PRs"]}
    assert V.landing(NOW - timedelta(hours=12), NOW, live=True)["share"] == "1 of 2 landed (50%)"
    assert asked == [(None, NOW.timestamp()), (NOW.timestamp(), None)]         # a window that ended is cut there; one to now is not
    with pytest.raises(V.Unread, match="gathers the 24 hours to a window's end, and this window is 40.3 h"):
        V.landing(NOW - timedelta(hours=40.3), NOW, live=False)
    monkeypatch.setattr(queue_table, "collect", lambda **kw: {"control": {}, "errors": ["vy-control-verity: no answer in 100 s"]})
    with pytest.raises(V.Unread, match=r"didn't read the queue's loop \(vy-control-verity: no answer in 100 s\)"):
        V.landing(NOW - timedelta(hours=6), NOW, live=False)


def test_the_command_compares_two_dumps_and_exits_0_with_the_rows_it_cant_read_saying_so(offline, capsys):
    assert V.main(["--from", str(A), "--to", str(B), "--baseline", str(B)], now=NOW + timedelta(days=3)) == 0
    out = capsys.readouterr().out
    assert out.startswith("research velocity: Mon 9 Nov 12:00 PM to 6:00 PM PST (6 h). A reading, not a gate.")
    assert "top priority, set by hand: 5 goals, 3 open, 1 of them off track" in out
    assert "  net days (earlier less later)     +0.5" in out and "since the baseline not read: its dump" in out
    assert "stalled now                       2 of 3 open: 1 with a step past its hour and not updated since, 1 with no step" in out
    assert "    top/beta: top/beta-step past its hour and not updated since" in out
    for name, why in (("LANDING", "a test has no queue table"), ("AGENTS", "give --agents FILE.json"),
                      ("MACHINES", "a test has no pod"), ("ASKS", "this window ends in the past")):
        assert f"{name:<10}not read: " in out and why in out
    assert V.main(["--from", str(A), "--to", str(B), "--json"], now=NOW) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["goals"]["all"]["window"]["finished"] == 2 and r["goals"]["all"]["window"]["net_days"] == 1.5
    assert [s["goal"] for s in r["stalls"]["goals"]] == ["top/beta", "top/gamma"] and r["hours"] == 6.0
    assert V.main(["--from", str(A), "--to", str(B), "--line"], now=NOW) == 0
    assert capsys.readouterr().out == (
        "Velocity, Mon 9 Nov 12:00 PM to 6:00 PM PST. Top-priority goals set by hand, since Mon 9 Nov 12:00 PM PST: 1 finished, "
        "1.0 days earlier less 0.5 later (net +0.5); 1 of 3 open off track; 2 stalled. Since the baseline (Mon 9 Nov 12:00 PM PST): "
        "1 finished, 1.0 days earlier less 0.5 later (net +0.5). Landing not read; agents not read; machines not read; asks not "
        "read.\n")
    assert V.main(["--since", "yesterday"], now=NOW) == 2
    assert V.main(["--until", "2026-11-09T18:01-08:00"], now=NOW) == 2          # a minute after now: no window ends there yet
    assert V.main(["--since", "2026-11-09T18:00-08:00"], now=NOW) == 2
    err = capsys.readouterr().err
    assert "--until 2026-11-09T18:01-08:00: it is not in the past" in err and "it is not before the window's end" in err
    assert V.main(["--from", "sweep:goals.json; true", "--to", str(B)], now=NOW) == 0     # not a dump's name: nothing is run
    assert "the sweep's dumps are named goals-YYYYMMDDTHHMMZ.json.gz" in capsys.readouterr().out
    assert V.main(["--since", "2026-11-09T12:00-08:00", "--until", "2026-11-09T18:00-08:00", "--json"], now=NOW) == 0
    assert [n for n in json.loads(capsys.readouterr().out)["notes"] if "sweep" in n] == ["the sweep's dumps not read: a test has no pod"]


def test_a_reading_now_saves_the_page_and_starts_from_the_newest_known_dump_at_the_windows_start(offline, monkeypatch, tmp_path, capsys):
    from research.store.local import LocalStore

    st = LocalStore()                                         # the suite's own store: a test's has no remote
    old = st.put_artifact(V.KIND, {"taken": "20261109T1400Z", "what": "a dump six hours before A"}, file=A)
    first = st.put_artifact(V.KIND, {"taken": "20261109T2010Z", "what": "ten minutes into the window: it stands for its start"}, file=A)
    st.put_artifact(V.KIND, {"taken": "20261109T2300Z", "what": "inside the window"}, file=A)
    monkeypatch.setattr(V, "read_page", lambda: goals(B))
    monkeypatch.setattr(V, "read_asks", lambda now: [])
    listing = tmp_path / "agents.json"
    listing.write_text(json.dumps({"agents": [{"name": "TOP-403 SCORE", "status": "RUNNING"}]}))
    assert V.main(["--agents", str(listing), "--json"], now=NOW) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["dumps"]["earlier"]["source"] == first and r["goals"]["hand"]["window"]["finished"] == 1
    saved = r["dumps"]["later"]
    assert saved["source"] == "the Goals page" and saved["unpreserved"]        # no remote here: it says so, and still reads
    assert goals(st.fetch(saved["art"])) == goals(B) and st.get_manifest(saved["art"]).meta["taken"] == "20261110T0200Z"
    assert r["agents"]["lanes"] == {"top": 1} and r["asks"] == {"open": 0, "off_track": 0, "kinds": {}, "issues": []}
    assert any("the sweep's dumps not read" in n for n in r["notes"])
    # a count in place of the listing, and the queue table's cells as the landing row and in the chat line
    monkeypatch.setattr(V, "landing", lambda since, until, live: {
        "share": "7 of 10 landed (70%); counting the 2 marked in the window and waiting over 2 h as misses, 58%",
        "mark_to_main": "p50 95, p90 200 min (n=10)", "unread": []})
    assert V.main(["--agents", "93", "--no-save"], now=NOW) == 0
    out = capsys.readouterr().out
    assert "AGENTS    93 running of 100 (the count --agents gave)\n" in out
    assert ("LANDING   within 2 h of the mark, 7 of 10 landed (70%); counting the 2 marked in the window and waiting over 2 h as "
            "misses, 58%; mark to `main` p50 95, p90 200 min (n=10) (`research queue stats`, the 6 h to Mon 9 Nov 6:00 PM PST)\n") in out
    assert V.main(["--agents", "93", "--no-save", "--line"], now=NOW) == 0
    assert "Landing within 2 h of the mark: 7 of 10 landed (70%); 93 of 100 agents running; " in capsys.readouterr().out
    # a window that ended: the newest known dump at or before each end, and nothing saved
    assert V.main(["--since", "2026-11-09T06:00-08:00", "--until", "2026-11-09T12:05-08:00", "--json"], now=NOW) == 0
    r = json.loads(capsys.readouterr().out)
    assert (r["dumps"]["earlier"]["source"], r["dumps"]["later"]["source"]) == (old, first) and "art" not in r["dumps"]["later"]
    assert not any("before the window" in n for n in r["notes"])
    assert V.main(["--since", "2026-11-09T10:00-08:00", "--until", "2026-11-09T12:05-08:00"], now=NOW) == 0     # no dump near 10 AM
    assert ("note: the goals' first column starts 4.0 h before the window, at its dump: none nearer Mon 9 Nov 10:00 AM PST is known"
            in capsys.readouterr().out)
    assert V.main(["--since", "2026-11-08T06:00-08:00", "--until", "2026-11-08T12:00-08:00"], now=NOW) == 0
    assert "GOALS     not read: no dump of the Goals page is known at or before Sun 8 Nov 12:00 PM PST" in capsys.readouterr().out
