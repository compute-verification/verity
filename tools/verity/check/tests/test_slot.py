"""`tools/verity/check/slot.py`: a `--record --on` check takes a free check slot's lock and cores, as the lander's train checks do."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("check_slot", ROOT / "tools" / "verity" / "check" / "slot.py")
SLOT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SLOT)
NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


class Exec(Exception):
    pass


@pytest.fixture(autouse=True)
def any_core(monkeypatch):
    """main() keeps only slots within the process's own cores; these tests name cores the machine running them may not give
    this process (a check slot pins it), so they see every core. The real-exec test runs a fresh slot.py and sees its own."""
    monkeypatch.setattr(SLOT, "allowed", lambda: set(range(1024)))


@pytest.fixture(autouse=True)
def own_ramlock(monkeypatch, tmp_path):
    """A slot with mem= writes its research.mem reservation here, never into the machine's /workspace/ramlock."""
    monkeypatch.setattr(SLOT, "RAMLOCK", tmp_path / "ramlock")


@pytest.fixture(autouse=True)
def outside_a_run(monkeypatch):
    """A check runs these tests inside its own `research run`: main() must not write that run's slot record."""
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)


def _hold(path: Path) -> int:
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def test_the_first_free_slot_is_taken_and_a_held_one_skipped(tmp_path):
    slots = SLOT.parse("# name cpus\na 32-63\nb 64-95   # train b\nc 8-31\n")
    assert [(s["name"], s["cpus"]) for s in slots] == [("a", "32-63"), ("b", "64-95"), ("c", "8-31")]
    held = _hold(tmp_path / "check-a.lock")
    slot, fd = SLOT.take(slots, tmp_path, NOW)
    assert slot["name"] == "b"
    assert SLOT.take(slots, tmp_path, NOW)[0]["name"] == "c"
    os.close(held)
    os.close(fd)


def test_a_slot_with_windows_pauses_from_its_lead_to_the_windows_end(tmp_path):
    w = tmp_path / "windows"
    slot = SLOT.parse(f"d 0-47 windows={w} lead=30")[0]
    for line, why in (("2026-10-01T09:20Z 30 # ca", True), ("2026-10-01T09:40Z 30", False), ("2026-10-01T08:45Z 30", True),
                      ("2026-10-01T08:00Z 30", False)):
        w.write_text(line + "\n")
        assert bool(SLOT.blocked(slot, NOW)) is why, line
    w.write_text("2026-10-01T09:20Z 30\n")
    assert SLOT.take([slot], tmp_path, NOW) is None
    assert SLOT.parse("a 0-1")[0]["windows"] is None and SLOT.blocked(SLOT.parse("a 0-1")[0], NOW) is None


def test_with_every_slot_busy_it_waits_then_runs_in_the_slot_that_frees(tmp_path, monkeypatch):
    (tmp_path / "slots").write_text("a 0\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    monkeypatch.setenv("LEASE_DIR", "/run/lease")
    monkeypatch.setenv(SLOT.SLOT_MEM_ENV, "560")
    held = [_hold(tmp_path / "check-a.lock")]
    ran = {}

    def free(_s):
        os.close(held.pop())

    def execvpe(f, argv, env):
        ran.update(argv=argv, env=env)
        raise Exec

    monkeypatch.setattr(SLOT.os, "sched_setaffinity", lambda pid, cpus: ran.update(cpus=cpus))
    monkeypatch.setattr(SLOT.os, "execvpe", execvpe)
    with pytest.raises(Exec):
        SLOT.main(["--", "uv", "run", "check.py"], now=lambda: NOW, sleep=free)
    assert ran["argv"] == ["uv", "run", "check.py"] and ran["cpus"] == {0}
    env = ran["env"]
    assert "LEASE_DIR" not in env and env["LEASE_DEADLINE_FILE"] == "/nonexistent/research-deadline" and env["CHECK_SLOT"] == "a"
    assert env["VERITY_TEST_CACHE"].endswith("/.cache/verity/tests-slot-a") and SLOT.SLOT_MEM_ENV not in env


def test_a_slot_added_while_a_check_waits_is_one_it_takes(tmp_path, monkeypatch):
    """Node 1's slot d (2 Oct 06:58Z) sat free behind the head of the line, a check that had read the slots file before d was in it."""
    slots = tmp_path / "slots"
    slots.write_text("a 0\n")
    monkeypatch.setattr(SLOT, "SLOTS", slots)
    held = _hold(tmp_path / "check-a.lock")
    ran = {}

    tries = []

    def add_d(_s):                                   # d appears on the first retry; a waiter that never rereads gives up at 5
        tries.append(1)
        assert len(tries) < 5, "the waiter never took the slot added while it waited"
        slots.write_text("a 0\nd 1\n")

    def execvpe(f, argv, env):
        ran.update(env=env)
        raise Exec

    monkeypatch.setattr(SLOT.os, "sched_setaffinity", lambda pid, cpus: ran.update(cpus=cpus))
    monkeypatch.setattr(SLOT.os, "execvpe", execvpe)
    with pytest.raises(Exec):
        SLOT.main(["--", "true"], now=lambda: NOW, sleep=add_d)
    assert ran["env"]["CHECK_SLOT"] == "d" and ran["cpus"] == {1}
    os.close(held)
    assert SLOT.reread([{"name": "x"}]) == SLOT.parse("a 0\nd 1\n")
    slots.write_text("# being rewritten\n")
    assert SLOT.reread([{"name": "x"}]) == [{"name": "x"}]


def test_a_slot_outside_the_processs_cores_is_skipped_never_pinned_onto(tmp_path, monkeypatch, capsys):
    """Node 1's slot d (128-159) killed every check that took it: a `research run` session there is held to cores 0-127."""
    (tmp_path / "slots").write_text("d 128-159\na 0\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    monkeypatch.setattr(SLOT, "allowed", lambda: set(range(128)))
    ran = {}

    def execvpe(f, argv, env):
        ran.update(env=env)
        raise Exec

    monkeypatch.setattr(SLOT.os, "sched_setaffinity", lambda pid, cpus: ran.update(cpus=cpus))
    monkeypatch.setattr(SLOT.os, "execvpe", execvpe)
    with pytest.raises(Exec):
        SLOT.main(["--", "true"], now=lambda: NOW)
    assert ran["env"]["CHECK_SLOT"] == "a" and ran["cpus"] == {0}
    err = capsys.readouterr().err
    assert "skipping d (128-159): outside this process's cores" in err and "add the slot's cores there" in err
    assert "/etc/vy/direct-cpus" in err                     # the fix, named where the skip is seen
    (tmp_path / "slots").write_text("d 128-159\n")
    with pytest.raises(SystemExit, match="none of .* slots is within this process's cores"):
        SLOT.main(["--", "true"], now=lambda: NOW)


def test_a_waiter_goes_after_the_live_ones_that_came_before_it_and_a_dead_ones_ticket_is_cleared(tmp_path, monkeypatch, capsys):
    (tmp_path / "slots").write_text("a 0\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    line = tmp_path / SLOT.LINE
    line.mkdir()
    (line / f"{1:020d}-1").write_text("")                                    # a waiter that died: nobody holds its ticket
    first = _hold(line / f"{2:020d}-2")                                      # a live one that came first; slot a is free
    seen, ran = [], {}

    def sleep(_s):
        seen.append(sorted(p.name for p in line.iterdir()))
        (line / f"{2:020d}-2").unlink()                                      # it takes its slot and runs
        os.close(first)

    def execvpe(f, argv, env):
        ran.update(slot=env["CHECK_SLOT"])
        raise Exec

    monkeypatch.setattr(SLOT.os, "sched_setaffinity", lambda pid, cpus: None)
    monkeypatch.setattr(SLOT.os, "execvpe", execvpe)
    with pytest.raises(Exec):
        SLOT.main(["--", "true"], now=lambda: NOW, sleep=sleep)
    assert len(seen) == 1 and seen[0][0] == f"{2:020d}-2" and len(seen[0]) == 2, seen     # the dead ticket gone, ours behind
    assert ran == {"slot": "a"} and list(line.iterdir()) == []
    assert "slot: 1 check(s) ahead in line; retrying every 20 s" in capsys.readouterr().err


def test_a_waiter_names_the_slots_holders_and_the_waiters_ahead_of_it_each_time_its_reason_changes(tmp_path, monkeypatch, capsys):
    """A check's log says whom it waits behind, not only how many (proofs, 2 Oct: "3-4 check(s) ahead in line" for 45 min)."""
    (tmp_path / "slots").write_text("a 0\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    held = _hold(tmp_path / "check-a.lock")
    (tmp_path / "check-a.who").write_text(json.dumps({"pid": os.getpid(), "run": "r-train", "tree": "/src/6b3d704c8", "by": "lander"}))
    line = tmp_path / SLOT.LINE
    line.mkdir()
    ahead = line / f"{int((NOW.timestamp() - 600) * 1e9):020d}-11"
    ahead.write_text(json.dumps({"pid": 11, "run": "r-ahead", "tree": "/src/740daff33", "by": "proofs"}))
    afd = _hold(ahead)
    real_join = SLOT.join

    def join(d, priority=False, who=None):
        ticket, fd = real_join(d, priority, who)
        later = line / f"{int(ticket.name.split('-')[0]) + 1:020d}-12"
        later.write_text(json.dumps({"pid": 12, "run": "r-later"}))
        fds.append(_hold(later))
        return ticket, fd

    fds, polls = [], []

    def sleep(_s):
        polls.append(1)
        if len(polls) == 1:
            ahead.unlink()                                                   # the waiter ahead takes the slot when it frees
            os.close(afd)
        elif len(polls) == 2:
            os.close(held)

    def execvpe(f, argv, env):
        raise Exec

    monkeypatch.setattr(SLOT, "join", join)
    monkeypatch.setattr(SLOT.os, "sched_setaffinity", lambda pid, cpus: None)
    monkeypatch.setattr(SLOT.os, "execvpe", execvpe)
    with pytest.raises(Exec):
        SLOT.main(["--", "true"], now=lambda: NOW, sleep=sleep)
    err = capsys.readouterr().err.splitlines()
    assert err[0] == "slot: 1 check(s) ahead in line; retrying every 20 s"
    assert err[1].startswith("slot:   a  cores 0  held by pid ") and "(run r-train, tree 6b3d704c8, by lander)" in err[1]
    assert err[2] == "slot:   waiting since 2026-10-01T08:50Z (10 min): pid 11 (run r-ahead, tree 740daff33, by proofs)"
    assert err[3].startswith("slot: every check slot is taken (a: busy)") and err[4].startswith("slot:   a  cores 0  held by pid ")
    assert not any("r-later" in ln for ln in err) and err[5] == "slot: check slot a (cores 0)"
    for fd in fds:
        os.close(fd)


def test_without_a_slots_file_the_command_runs_as_it_is(tmp_path, monkeypatch):
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "absent")

    def execvp(f, argv):
        raise Exec(argv)

    monkeypatch.setattr(SLOT.os, "execvp", execvp)
    with pytest.raises(Exec) as e:
        SLOT.main(["--", "true"], now=lambda: NOW)
    assert e.value.args[0] == ["true"]


def test_a_waiter_past_its_wait_max_leaves_the_line_and_exits_not_started(tmp_path, monkeypatch, capsys):
    """Quick tiers sat 3 h in node 1's line and their run's timeout reported them failed (#1344, 6 Oct): --wait-max ends the
    wait first, with an exit that says nothing ran."""
    (tmp_path / "slots").write_text("a 0\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    held = _hold(tmp_path / "check-a.lock")
    t = [NOW]
    monkeypatch.setattr(SLOT.os, "execvpe", lambda *a: pytest.fail("CMD ran without a slot"))
    rc = SLOT.main(["--wait-max", "60", "--", "true"], now=lambda: t[0],
                   sleep=lambda s: t.__setitem__(0, t[0] + timedelta(seconds=s)))
    assert rc == SLOT.NOT_STARTED == 75 and t[0] - NOW == timedelta(seconds=3 * SLOT.POLL_S)
    assert "slot: not started: no check slot within --wait-max 60 s (every check slot is taken (a: busy)); exit 75" in capsys.readouterr().err
    assert list((tmp_path / SLOT.LINE).iterdir()) == []                         # its ticket is gone: nobody waits behind it
    os.close(held)


def test_run_max_counts_from_the_slot_taken_through_the_runs_timeout_control(tmp_path, monkeypatch, capsys):
    """The run's --timeout counted the wait; once the slot is taken, the run has --run-max from then (its harness re-reads
    control/timeout_s, from the run's start on this machine's monotonic clock)."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "status.json").write_text(json.dumps({"state": "running", "started_mono": 1000.0}))
    assert SLOT.limit_run(10800, "d", str(run), mono=lambda: 1000.0 + 5400) == \
        "slot: the run's timeout is now 16200 s from its start: 10800 s from taking slot d"
    doc = json.loads((run / SLOT.RUN_TIMEOUT).read_text())
    assert doc == {"timeout_s": 16200.0, "actor": "timeout", "reason": "slot.py --run-max: 10800 s from taking check slot d"}
    assert "not applied: not in a `research run`" in SLOT.limit_run(60, "d", None)
    (run / "status.json").write_text("{}")
    assert "has no started_mono; the run keeps its timeout" in SLOT.limit_run(60, "d", str(run))
    (tmp_path / "slots").write_text("a 0\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run))
    (run / "status.json").write_text(json.dumps({"started_mono": 0.0}))
    monkeypatch.setattr(SLOT.os, "sched_setaffinity", lambda pid, cpus: None)
    monkeypatch.setattr(SLOT.os, "execvpe", lambda f, argv, env: (_ for _ in ()).throw(Exec(argv)))
    with pytest.raises(Exec):
        SLOT.main(["--by", "infra", "--wait-max", "21600", "--run-max", "10800", "--", "true"], now=lambda: NOW)
    assert "slot: the run's timeout is now" in capsys.readouterr().err
    assert json.loads((run / SLOT.RUN_TIMEOUT).read_text())["timeout_s"] >= 10800
    for bad in (["--wait-max", "0"], ["--run-max", "soon"], ["--wait-max"]):
        with pytest.raises(SystemExit):
            SLOT.main([*bad, "--", "true"])


def test_a_run_s_slot_record_says_it_waits_until_it_takes_a_slot_or_leaves_the_line(tmp_path, monkeypatch):
    """af2d (10 Oct) waited 95 min in node 1's line and was reported running since its launch: its run's slot.json says waiting,
    and why, until the slot is taken, then which slot and when; a waiter that gives up says it left, and why."""
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "slots").write_text("a 0\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run))
    monkeypatch.setattr(SLOT.os, "sched_setaffinity", lambda pid, cpus: None)
    held, t, seen = [_hold(tmp_path / "check-a.lock")], [NOW], []

    def sleep(s):
        seen.append(json.loads((run / SLOT.RUN_SLOT).read_text()))
        t[0] += timedelta(seconds=s)
        os.close(held.pop())

    monkeypatch.setattr(SLOT.os, "execvpe", lambda *a: (_ for _ in ()).throw(Exec()))
    with pytest.raises(Exec):
        SLOT.main(["--priority", "--", "true"], now=lambda: t[0], sleep=sleep)
    assert seen == [{"state": "waiting", "since": "2026-10-01T09:00:00Z", "priority": True,
                     "why": "every check slot is taken (a: busy)"}]
    assert json.loads((run / SLOT.RUN_SLOT).read_text()) == {"state": "held", "since": "2026-10-01T09:00:00Z", "priority": True,
                                                              "slot": "a", "cores": "0", "took": "2026-10-01T09:00:20Z"}
    (tmp_path / "again").mkdir()
    (tmp_path / "again" / "slots").write_text("a 0\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "again" / "slots")
    held = [_hold(tmp_path / "again" / "check-a.lock")]
    monkeypatch.setattr(SLOT.os, "execvpe", lambda *a: pytest.fail("CMD ran without a slot"))
    assert SLOT.main(["--wait-max", "20", "--", "true"], now=lambda: t[0],
                     sleep=lambda s: t.__setitem__(0, t[0] + timedelta(seconds=s))) == SLOT.NOT_STARTED
    doc = json.loads((run / SLOT.RUN_SLOT).read_text())
    assert doc["state"] == "left" and doc["since"] == "2026-10-01T09:00:20Z" and doc["why"].startswith("no check slot within --wait-max 20 s")
    os.close(held[0])
    (run / SLOT.RUN_SLOT).unlink()
    run.rmdir()
    SLOT.record_slot(str(run), {"state": "waiting"})                       # an unwritable run dir never stops the check
    assert not run.exists()


def test_the_command_holds_the_slots_lock_on_the_slots_cores(tmp_path):
    """The real exec: the child runs pinned, and the slot's lock is held for as long as it runs."""
    cpu = min(os.sched_getaffinity(0))
    (tmp_path / "slots").write_text(f"a {cpu}\n")
    child = ("import fcntl, os; fd = os.open(os.environ['LOCK'], os.O_RDWR)\n"
             "try:\n    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); held = False\n"
             "except BlockingIOError:\n    held = True\n"
             "print(sorted(os.sched_getaffinity(0)), held, os.environ['CHECK_SLOT'])")
    env = {**os.environ, "CHECK_SLOTS": str(tmp_path / "slots"), "LOCK": str(tmp_path / "check-a.lock")}
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "verity" / "check" / "slot.py"), "--", sys.executable, "-c", child],
                         env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == [f"[{cpu}]", "True", "a"], out.stderr
    assert f"check slot a (cores {cpu})" in out.stderr


def test_status_names_each_slots_holder_or_free(tmp_path, monkeypatch):
    """Who holds check-a/b/c and since when, without `fuser` on the lock files (old-circuits-and-proofs, 1 Oct)."""
    w = tmp_path / "windows"
    w.write_text("2026-10-01T09:20Z 30\n")
    slots = SLOT.parse(f"a 32-63\nb 64-95\nd 0-47 windows={w}\n")
    held = _hold(tmp_path / "check-a.lock")
    a, b, d, dw = SLOT.status(slots, tmp_path, NOW)
    me = Path(f"/proc/{os.getpid()}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()[:200]
    assert a.startswith(f"a  cores 32-63  held by pid {os.getpid()} since ") and a.endswith(f": {me}")
    assert b == "b  cores 64-95  free"
    assert d.startswith("d  cores 0-47  free; blocked: window 2026-10-01T09:20Z")
    assert dw == "d  window 2026-10-01T09:20Z (30 min), starts in 20 min"
    os.close(held)
    assert SLOT.status(slots[:1], tmp_path, NOW) == ["a  cores 32-63  free"]
    (tmp_path / "slots").write_text("a 0\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    assert SLOT.main(["--status"], now=lambda: NOW) == 0


def test_status_lists_a_windowed_slots_windows_open_now_or_within_six_hours(tmp_path):
    """A check slot paused for a timed window nobody saw coming (node 2, 3 Oct: a PR sat about 20 min on the 18:05Z window)."""
    w = tmp_path / "windows"
    w.write_text("# start UTC, minutes\n"
                 "2026-10-01T15:01Z 30 # beyond six hours\n"
                 "2026-10-01T08:00Z 30 # ended at 08:30Z\n"
                 "2026-10-01T09:25Z 20 # served qwen3-8b timed, compute-accounting: quiet timed FP8 Qwen3-8B pass, 8 GPUs\n"
                 "2026-10-01T08:50Z 30 # circuits TP8\n"
                 "2026-10-01T14:59Z 10\n")
    slot = SLOT.parse(f"d 0-47 windows={w} lead=30")[0]
    assert SLOT.upcoming(slot, NOW) == [
        {"start": "2026-10-01T08:50Z", "minutes": 30, "starts_in_min": -10, "note": "circuits TP8"},
        {"start": "2026-10-01T09:25Z", "minutes": 20, "starts_in_min": 25, "note": "served qwen3-8b timed, compute-accounting: quiet timed ..."},
        {"start": "2026-10-01T14:59Z", "minutes": 10, "starts_in_min": 359, "note": ""}]
    assert SLOT.status([slot], tmp_path, NOW) == [
        "d  cores 0-47  free; blocked: window 2026-10-01T09:25Z (20 min) is open or starts within 30 min",
        "d  window 2026-10-01T08:50Z (30 min), open now: circuits TP8",
        "d  window 2026-10-01T09:25Z (20 min), starts in 25 min: served qwen3-8b timed, compute-accounting: quiet timed ...",
        "d  window 2026-10-01T14:59Z (10 min), starts in 359 min"]
    assert SLOT.upcoming(SLOT.parse("a 0")[0], NOW) == []


def test_status_json_is_the_same_state_as_one_object(tmp_path, monkeypatch, capsys):
    """`--status --json`, which `check.py --slots` reads from each machine instead of parsing the lines."""
    w = tmp_path / "windows"
    w.write_text("2026-10-01T09:20Z 30 # Pearl-C4 row\n")
    (tmp_path / "slots").write_text(f"a 0\nd 1 windows={w} lead=30\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    held = _hold(tmp_path / "check-a.lock")
    (tmp_path / "check-a.who").write_text(json.dumps({"pid": os.getpid(), "run": "r-train", "tree": "/src/6b3d704c8", "by": "lander"}))
    line = tmp_path / SLOT.LINE
    line.mkdir()
    waiter = line / f"{SLOT.PRIORITY}{int((NOW.timestamp() - 300) * 1e9):020d}-4242"
    waiter.write_text(json.dumps({"pid": 4242, "run": "r-waiting", "by": "lander"}))
    wfd = _hold(waiter)
    assert SLOT.main(["--status", "--json"], now=lambda: NOW) == 0
    doc = json.loads(capsys.readouterr().out)
    a, d = doc["slots"]
    assert doc["file"] == str(tmp_path / "slots") and doc["now"] == "2026-10-01T09:00Z"
    assert a["holder"]["pid"] == os.getpid() and (a["holder"]["run"], a["holder"]["by"]) == ("r-train", "lander")
    assert a["holder"]["cmd"] and a["blocked"] is None and a["upcoming"] == []
    assert d["holder"] is None and d["blocked"].startswith("window 2026-10-01T09:20Z (30 min)") and d["lead"] == 30
    assert d["upcoming"] == [{"start": "2026-10-01T09:20Z", "minutes": 30, "starts_in_min": 20, "note": "Pearl-C4 row"}]
    assert doc["waiters"] == [{"since": "2026-10-01T08:55Z", "minutes": 5, "priority": True, "pid": 4242, "run": "r-waiting",
                               "by": "lander"}]
    os.close(held)
    os.close(wfd)
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "absent")
    assert SLOT.main(["--status", "--json"], now=lambda: NOW) == 0
    assert json.loads(capsys.readouterr().out) == {"file": str(tmp_path / "absent"), "now": "2026-10-01T09:00Z", "slots": None,
                                                    "waiters": []}
    with pytest.raises(SystemExit, match="usage: slot.py --status"):
        SLOT.main(["--status", "--yaml"], now=lambda: NOW)


def test_status_names_who_asked_for_each_held_slot_and_each_live_waiter(tmp_path):
    """Which run, tree and agent hold or wait for a slot, without asking around (old-circuits-and-proofs, 2 Oct)."""
    slots = SLOT.parse("a 0\nb 1\n")
    held = _hold(tmp_path / "check-a.lock")
    me = {"pid": os.getpid(), "run": "r20261002-040109-a7ed", "tree": "/workspace/research/src/6a1d0bf19", "by": "proofs"}
    (tmp_path / "check-a.who").write_text(json.dumps(me))
    (tmp_path / "check-b.who").write_text(json.dumps({**me, "run": "r-gone"}))     # b is free: its last holder isn't shown
    line = tmp_path / SLOT.LINE
    line.mkdir()
    (line / f"{1:020d}-1").write_text(json.dumps({**me, "run": "r-dead"}))           # a waiter that died: nobody holds its ticket
    waiter = line / f"{SLOT.PRIORITY}{int((NOW.timestamp() - 300) * 1e9):020d}-4242"   # the lander's train check
    waiter.write_text(json.dumps({"pid": 4242, "run": "r-waiting", "tree": "/workspace/research/src/bc63d108f", "by": "lander"}))
    wfd = _hold(waiter)
    a, b, w = SLOT.status(slots, tmp_path, NOW)
    assert f"held by pid {os.getpid()} since " in a and " (run r20261002-040109-a7ed, tree 6a1d0bf19, by proofs): " in a
    assert b == "b  cores 1  free"
    assert w == "waiting since 2026-10-01T08:55Z (5 min) with priority: pid 4242 (run r-waiting, tree bc63d108f, by lander)"
    (tmp_path / "check-a.who").write_text(json.dumps({**me, "pid": 1}))              # left by an earlier holder: a hand-run flock now
    assert "(run " not in SLOT.status(slots, tmp_path, NOW)[0]
    os.close(held)
    os.close(wfd)
    assert SLOT.status(slots, tmp_path, NOW) == ["a  cores 0  free", "b  cores 1  free"]


def test_the_check_that_takes_a_slot_records_who_asked_for_it(tmp_path, monkeypatch):
    (tmp_path / "slots").write_text("a 0\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setenv("RESEARCH_AGENT_URL", "https://cursor.com/agents/bc-1")
    monkeypatch.chdir(tmp_path)
    real, tickets = SLOT.join, []

    def join(d, priority=False, who=None):
        ticket, fd = real(d, priority, who)
        tickets.append(json.loads(ticket.read_text()))
        return ticket, fd

    def execvpe(f, argv, env):
        raise Exec(argv)

    monkeypatch.setattr(SLOT, "join", join)
    monkeypatch.setattr(SLOT.os, "sched_setaffinity", lambda pid, cpus: None)
    monkeypatch.setattr(SLOT.os, "execvpe", execvpe)
    with pytest.raises(Exec) as e:
        SLOT.main(["--by", "lander", "--", "true"], now=lambda: NOW)
    assert e.value.args[0] == ["true"]
    me = {"pid": os.getpid(), "run": "r1", "tree": os.getcwd(), "by": "lander"}
    who = tmp_path / "check-a.who"
    assert tickets == [me] and json.loads(who.read_text()) == me and stat.S_IMODE(who.stat().st_mode) == 0o664
    assert sorted(p.name for p in tmp_path.iterdir()) == ["check-a.lock", "check-a.who", SLOT.LINE, "slots"]
    assert SLOT.asker()["by"] == "https://cursor.com/agents/bc-1"


def test_a_waiters_line_in_proc_locks_is_not_the_holder(tmp_path):
    lock = tmp_path / "check-a.lock"
    lock.touch()
    st = lock.stat()
    dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "locks").write_text(f"1: FLOCK  ADVISORY  WRITE 4242 {dev} 0 EOF\n1: -> FLOCK  ADVISORY  WRITE 777 {dev} 0 EOF\n"
                                f"2: POSIX  ADVISORY  WRITE 99 00:01:1 0 EOF\n")
    assert SLOT.holder(lock, proc) == 4242
    (proc / "locks").write_text(f"1: -> FLOCK  ADVISORY  WRITE 777 {dev} 0 EOF\n")
    assert SLOT.holder(lock, proc) is None
    assert SLOT.holder(tmp_path / "absent.lock", proc) is None


def test_a_check_recorded_on_a_machine_runs_in_a_slot(monkeypatch):
    spec2 = importlib.util.spec_from_file_location("repo_check_slot", ROOT / "tools" / "verity" / "check" / "check.py")
    check = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(check)
    monkeypatch.setattr(check, "gather_verdicts", lambda d, given=None: None)
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(check.subprocess, "run", run)
    monkeypatch.setattr(check, "lean_deps_url_file", lambda d: None)
    monkeypatch.delenv("RESEARCH_AGENT_URL", raising=False)
    monkeypatch.delenv("RESEARCH_LANE", raising=False)
    assert check.record(["--on", "vy-nebius-2"]) == 0
    remote = calls[-1][calls[-1].index("--") + 1:]
    assert remote[:3] == check.SLOT_WRAP[:3] == ["python3", "tools/verity/check/slot.py", "--"] and remote[3:5] == ["uv", "run"]
    monkeypatch.setenv("RESEARCH_LANE", "proofs")
    assert check.record(["--on", "vy-nebius-2"]) == 0
    assert calls[-1][calls[-1].index("--") + 1:][:5] == ["python3", "tools/verity/check/slot.py", "--by", "proofs", "--"]
    assert check.record([]) == 0 and "slot.py" not in " ".join(calls[-1])


def test_agreement_sends_the_upstream_build_whatever_the_commit_changes(monkeypatch, tmp_path):
    """`--record --agreement`: the lander's check of a train that changes a pinned guarantee outside verity/ml/flock/ still runs
    the agreement step, with the pinned build and inputs as the run's inputs."""
    spec2 = importlib.util.spec_from_file_location("repo_check_agreement", ROOT / "tools" / "verity" / "check" / "check.py")
    check = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(check)
    import research.store.local as local

    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    class Store:
        def fetch(self, art):
            f = tmp_path / art.replace(":", "_")
            f.write_bytes(b"")
            return f

    monkeypatch.setattr(check.subprocess, "run", run)
    monkeypatch.setattr(check, "lean_deps_url_file", lambda d: None)
    monkeypatch.setattr(local, "LocalStore", Store)
    assert check.record(["--on", "vy-nebius-1", "--base", "HEAD"]) == 0
    assert "--input" not in calls[-1]
    assert check.record(["--on", "vy-nebius-1", "--base", "HEAD", "--agreement"]) == 0
    cmd = calls[-1]
    assert [cmd[i + 1].split("=")[0] for i, x in enumerate(cmd) if x == "--input"] == ["flock_upstream", "flock_inputs"]
    assert "--agreement" not in cmd


def test_a_priority_waiter_goes_ahead_of_waiting_lane_checks_and_behind_older_priority_ones(tmp_path):
    """Merges unblock everyone (root, 2 Oct): the lander's train check takes the next slot that frees ahead of lane checks
    already waiting, never one a check is running in, and priority waiters keep their own arrival order."""
    line = tmp_path / SLOT.LINE
    line.mkdir()
    lane = _hold(line / f"{1:020d}-1")                                       # a lane check that has waited longest
    train, tfd = SLOT.join(tmp_path, priority=True)
    assert train.name.startswith(SLOT.PRIORITY) and SLOT.ahead(train) == 0  # the lane waiter is no obstacle
    lane2, lfd = SLOT.join(tmp_path)
    assert SLOT.ahead(lane2) == 2                                            # behind the train and the older lane waiter
    train2, t2 = SLOT.join(tmp_path, priority=True)
    assert SLOT.ahead(train2) == 1                                           # behind the older train only
    for fd in (lane, tfd, lfd, t2):
        os.close(fd)


def test_a_trains_slot_is_taken_only_by_a_priority_waiter(tmp_path, monkeypatch, capsys):
    """Node 2, 6 Oct 02:31Z: the lander's train check waited 36 min with priority while quick tiers held both slots; priority
    orders the line but never frees a slot a lane check is running in."""
    (tmp_path / "slots").write_text("d 0 only=trains\ne 1\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    assert [s["trains"] for s in SLOT.parse((tmp_path / "slots").read_text())] == [True, False]
    held = [_hold(tmp_path / "check-e.lock")]
    ran = {}

    def free(_s):
        if held:
            os.close(held.pop())

    def execvpe(f, argv, env):
        ran.update(slot=env["CHECK_SLOT"])
        raise Exec

    monkeypatch.setattr(SLOT.os, "sched_setaffinity", lambda pid, cpus: None)
    monkeypatch.setattr(SLOT.os, "execvpe", execvpe)
    with pytest.raises(Exec):
        SLOT.main(["--", "check.py"], now=lambda: NOW, sleep=free)              # a lane check waits for e, though d is free
    assert ran["slot"] == "e" and "d: for train checks; e: busy" in capsys.readouterr().err
    with pytest.raises(Exec):
        SLOT.main(["--priority", "--", "check.py"], now=lambda: NOW, sleep=free)
    assert ran["slot"] == "d"
    assert SLOT.status(SLOT.parse("d 0 only=trains\n"), tmp_path, NOW)[0].startswith("d  cores 0  trains only  ")
    (tmp_path / "slots").write_text("d 0 only=trains\n")
    with pytest.raises(SystemExit, match="every slot in .* is for train checks"):
        SLOT.main(["--", "check.py"], now=lambda: NOW, sleep=free)


def test_priority_reaches_slot_py_from_record_and_unknown_options_are_refused(monkeypatch):
    spec2 = importlib.util.spec_from_file_location("repo_check_priority", ROOT / "tools" / "verity" / "check" / "check.py")
    check = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(check)
    monkeypatch.setattr(check, "gather_verdicts", lambda d, given=None: None)
    calls = []
    monkeypatch.setattr(check.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(check, "lean_deps_url_file", lambda d: None)
    monkeypatch.delenv("RESEARCH_AGENT_URL", raising=False)
    monkeypatch.delenv("RESEARCH_LANE", raising=False)
    assert check.record(["--on", "vy-nebius-1", "--priority"]) == 0
    remote = calls[-1][calls[-1].index("--") + 1:]
    assert remote[:4] == ["python3", "tools/verity/check/slot.py", "--priority", "--"] and "--priority" not in remote[4:]
    monkeypatch.setenv("RESEARCH_LANE", "lander")
    assert check.record(["--on", "vy-nebius-1", "--priority"]) == 0
    assert calls[-1][calls[-1].index("--") + 1:][:6] == ["python3", "tools/verity/check/slot.py", "--priority", "--by", "lander", "--"]
    with pytest.raises(SystemExit):
        SLOT.main(["--fast", "--", "true"])


def test_memory_gb_is_the_least_of_memavailable_and_each_cgroup_limits_headroom(tmp_path):
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc/meminfo").write_text(f"MemTotal: {2000 << 20} kB\nMemAvailable: {1500 << 20} kB\n")
    (tmp_path / "proc/self").mkdir()
    (tmp_path / "proc/self/cgroup").write_text("0::/user.slice/user-1001.slice/run-1.scope\n")
    cg = tmp_path / "sys/fs/cgroup"
    for d, mx, cur in (("user.slice", "max", 0), ("user.slice/user-1001.slice", 1600 << 30, 1200 << 30),
                       ("user.slice/user-1001.slice/run-1.scope", 900 << 30, 10 << 30)):
        (cg / d).mkdir(parents=True)
        (cg / d / "memory.max").write_text(f"{mx}\n")
        (cg / d / "memory.current").write_text(f"{cur}\n")
    assert SLOT.memory_gb(tmp_path) == 400
    (cg / "user.slice/user-1001.slice/memory.current").write_text(f"{1700 << 30}\n")
    assert SLOT.memory_gb(tmp_path) == 0
    assert SLOT.memory_gb(tmp_path / "absent") == 0


def test_memory_gb_counts_a_cgroups_page_cache_as_left(tmp_path):
    """vy-nebius-cpu-1, 9 Oct 20:50Z: 273 GB of page cache in the user slice held slot a (mem=560) shut with 677 GB available."""
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc/meminfo").write_text(f"MemAvailable: {677 << 20} kB\n")
    (tmp_path / "proc/self").mkdir()
    (tmp_path / "proc/self/cgroup").write_text("0::/user.slice\n")
    cg = tmp_path / "sys/fs/cgroup/user.slice"
    cg.mkdir(parents=True)
    (cg / "memory.max").write_text(f"{700 << 30}\n")
    (cg / "memory.current").write_text(f"{347 << 30}\n")
    assert SLOT.memory_gb(tmp_path) == 353
    (cg / "memory.stat").write_text(f"anon {74 << 30}\nfile {273 << 30}\nactive_file {200 << 30}\ninactive_file {73 << 30}\n"
                                    f"shmem 0\n")
    assert SLOT.memory_gb(tmp_path) == 626


def test_a_slot_with_mem_is_taken_only_while_that_much_memory_is_free(tmp_path, monkeypatch, capsys):
    """Node 1, 8 Oct 02:00Z: OOM kills at 1,605 of 1,716 GB across ~8 runs under the user slice; a check whose peak doesn't fit
    waits in line instead of starting and being killed with the others."""
    (tmp_path / "slots").write_text("a 0 mem=560\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    assert SLOT.parse("a 0 mem=560\nb 1\n")[0]["mem"] == 560 and SLOT.parse("b 1")[0]["mem"] is None
    free = [300, 600]
    monkeypatch.setattr(SLOT, "memory_gb", lambda: free[0])
    assert SLOT.blocked(SLOT.parse("a 0 mem=560")[0], NOW) == "300 GB of memory free, under the 560 GB a check here may peak at"
    assert SLOT.blocked(SLOT.parse("b 1")[0], NOW) is None
    assert SLOT.status(SLOT.parse("a 0 mem=560"), tmp_path, NOW)[0].startswith("a  cores 0  mem 560 GB  free; blocked: 300 GB")
    ran = {}

    def execvpe(f, argv, env):
        ran.update(slot=env["CHECK_SLOT"], mem=env.get(SLOT.SLOT_MEM_ENV))
        raise Exec

    monkeypatch.setattr(SLOT.os, "sched_setaffinity", lambda pid, cpus: None)
    monkeypatch.setattr(SLOT.os, "execvpe", execvpe)
    with pytest.raises(Exec):
        SLOT.main(["--", "check.py"], now=lambda: NOW, sleep=lambda _s: free.pop(0))
    assert ran == {"slot": "a", "mem": "560"} and free == [600]
    assert "a: 300 GB of memory free, under the 560 GB a check here may peak at" in capsys.readouterr().err


def _twin_ssh(monkeypatch, here: Path, home: Path, sent: list):
    """slot.py's ssh to HOME, run by sh on this machine with HOME's lock directory in place of this one's (both nodes keep their
    locks at the same path)."""
    real = subprocess.run

    def run(argv, **kw):
        if argv[:1] != ["ssh"]:
            return real(argv, **kw)
        sent.append(argv[:-1])
        return real(["sh", "-c", argv[-1].replace(str(here), str(home))], **kw)

    monkeypatch.setattr(SLOT.subprocess, "run", run)


def test_a_twin_runs_only_if_it_claims_its_pair_first_and_its_twins_then_leave_their_lines(tmp_path, monkeypatch, capsys):
    """9 Oct: four checks waited up to 54 min in node 1's line while a node 2 slot was free, until each was moved by hand. Twins
    wait in both lines; the first to hold a slot claims the pair on HOME and runs, and the other leaves its line (75)."""
    n1, n2 = tmp_path / "n1", tmp_path / "n2"
    for d in (n1, n2):
        d.mkdir()
        (d / "slots").write_text("a 0\n")
    peers = tmp_path / "peers"
    peers.write_text("# machine, ssh options and destination\nvy-nebius-1 -i /k -o IdentitiesOnly=yes research@10.0.0.1\n")
    monkeypatch.setattr(SLOT, "PEERS", peers)
    sent = []
    _twin_ssh(monkeypatch, n2, n1, sent)
    monkeypatch.setattr(SLOT.os, "sched_setaffinity", lambda pid, cpus: None)
    monkeypatch.setattr(SLOT.os, "execvpe", lambda *a: (_ for _ in ()).throw(Exec()))
    held = _hold(n1 / "check-a.lock")                                              # node 1's slot is busy: its twin waits there
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-n2")
    monkeypatch.setattr(SLOT, "SLOTS", n2 / "slots")
    with pytest.raises(Exec):                                                      # node 2's frees first: its twin claims and runs
        SLOT.main(["--twin", "p1@vy-nebius-1", "--", "check"], now=lambda: NOW, sleep=lambda s: None)
    assert sent[-1] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-i", "/k", "-o", "IdentitiesOnly=yes", "research@10.0.0.1"]
    claim = json.loads((n1 / "twins" / "p1" / "who").read_text())
    assert (claim["run"], claim["twin"], claim["took"]) == ("r-n2", "p1@vy-nebius-1", "2026-10-01T09:00Z")
    assert not (n2 / "twins").exists()
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-n1")
    monkeypatch.setattr(SLOT, "SLOTS", n1 / "slots")
    assert SLOT.main(["--twin", "p1", "--", "check"], now=lambda: NOW, sleep=lambda s: pytest.fail("waited")) == SLOT.NOT_STARTED
    assert "slot: not started: its twin took a slot first (run r-n2, " in capsys.readouterr().err
    assert list((n1 / SLOT.LINE).iterdir()) == []
    os.close(held)


def test_a_twin_that_holds_a_slot_after_its_pair_was_claimed_or_can_t_claim_it_lets_the_slot_go(tmp_path, monkeypatch, capsys):
    (tmp_path / "slots").write_text("a 0\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    monkeypatch.setattr(SLOT, "PEERS", tmp_path / "absent-peers")
    monkeypatch.setattr(SLOT.os, "execvpe", lambda *a: pytest.fail("CMD ran"))
    assert SLOT.main(["--twin", "p2@vy-nebius-1", "--", "check"], now=lambda: NOW, sleep=lambda s: None) == SLOT.NOT_STARTED
    assert (f"slot: not started: twin p2@vy-nebius-1 can't be claimed (vy-nebius-1 isn't in {tmp_path / 'absent-peers'}), "
            "so its twin there runs instead; exit 75") in capsys.readouterr().err
    monkeypatch.setattr(SLOT, "twin_here", lambda *a: ("unreachable", {"why": "PermissionError: twins"}))
    assert SLOT.main(["--twin", "p2", "--", "check"], now=lambda: NOW, sleep=lambda s: None) == SLOT.NOT_STARTED
    assert ("twin p2's claim can't be written here (PermissionError: twins), and no twin runs without it: rerun the check"
            in capsys.readouterr().err)
    monkeypatch.undo()                                                             # undoes any_core's pin too
    monkeypatch.setattr(SLOT, "allowed", lambda: set(range(1024)))
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    monkeypatch.setattr(SLOT.os, "execvpe", lambda *a: pytest.fail("CMD ran"))
    assert SLOT.twin_here("claim", tmp_path, "p3", {"run": "r-other"})[0] == "claimed"   # its twin claimed p3 here first
    t = [NOW]
    peeks = []
    real = SLOT.twin

    def twin(op, spec, who):
        peeks.append(op)
        return ("free", {}) if op == "peek" else real(op, spec, who)              # the claim lands between a peek and the take

    monkeypatch.setattr(SLOT, "twin", twin)
    assert SLOT.main(["--twin", "p3", "--", "check"], now=lambda: t[0], sleep=lambda s: None) == SLOT.NOT_STARTED
    assert peeks == ["peek", "claim"] and "its twin took a slot first (run r-other)" in capsys.readouterr().err
    slot, fd = SLOT.take(SLOT.parse("a 0"), tmp_path, NOW)                         # the slot it held is free again
    os.close(fd)


def test_twin_is_parsed_and_shown_and_a_bad_pair_is_refused(tmp_path, capsys):
    assert SLOT.options(["--by", "x", "--twin", "20261009T0440Z-ab12@vy-nebius-2"])[4] == "20261009T0440Z-ab12@vy-nebius-2"
    with pytest.raises(SystemExit, match="is not PAIR or PAIR@HOME"):
        SLOT.options(["--twin", "p;rm -rf /"])
    assert "twin p1" in SLOT.describe({"run": "r1", "twin": "p1"})


def test_a_slot_with_mem_holds_a_research_mem_reservation_that_research_mem_counts_while_the_check_runs(tmp_path, monkeypatch):
    """Node 1, 8:21Z on 10 Oct: a check slot checked free memory once, at its start, and reserved nothing, so research.mem
    admitted work beside two full checks until the node ran out of memory. A slot with mem=GB writes `check-NAME.json` in
    research.mem's format, naming the check's own pid (CMD's, after the exec) and run; research.mem counts it as live, the GB in
    its 10^9-byte units less what the process tree holds, and not once the run has ended. A slot without mem= reserves nothing."""
    from research import mem

    (tmp_path / "slots").write_text("a 0 mem=500\nb 1\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    monkeypatch.setattr(SLOT, "memory_gb", lambda: 1000)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261010-170000-abcd")
    monkeypatch.setattr(SLOT.os, "sched_setaffinity", lambda pid, cpus: None)
    monkeypatch.setattr(SLOT.os, "execvpe", lambda f, argv, env: (_ for _ in ()).throw(Exec(env["CHECK_SLOT"])))
    with pytest.raises(Exec, match="a"):
        SLOT.main(["--", "check.py"], now=lambda: NOW, sleep=lambda _s: pytest.fail("slot a is free"))
    doc = json.loads((tmp_path / "ramlock" / "check-a.json").read_text())
    assert doc["owner"] == "check" and doc["pid"] == os.getpid() and doc["gb"] == 536.9 and doc["run_id"] == "r20261010-170000-abcd"
    (row,) = mem.reservations(tmp_path / "ramlock")
    assert row["live"] and row["pid"] == os.getpid() and 0 < row["outstanding"] <= 536.9e9, row
    (run_dir / "status.json").write_text(json.dumps({"state": "done"}))
    (row,) = mem.reservations(tmp_path / "ramlock")
    assert not row["live"] and row["why"] == "run r20261010-170000-abcd ended"

    (tmp_path / "slots").write_text("b 1\n")
    with pytest.raises(Exec, match="b"):
        SLOT.main(["--", "check.py"], now=lambda: NOW, sleep=lambda _s: pytest.fail("slot b is free"))
    assert [f.name for f in (tmp_path / "ramlock").glob("*.json")] == ["check-a.json"]
