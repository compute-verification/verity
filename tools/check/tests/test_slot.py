"""`tools/check/slot.py`: a `--record --on` check takes a free check slot's lock and cores, as the lander's train checks do."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("check_slot", ROOT / "tools" / "check" / "slot.py")
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
    assert env["VERITY_TEST_CACHE"].endswith("/.cache/verity/tests-slot-a")


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


def test_the_command_holds_the_slots_lock_on_the_slots_cores(tmp_path):
    """The real exec: the child runs pinned, and the slot's lock is held for as long as it runs."""
    cpu = min(os.sched_getaffinity(0))
    (tmp_path / "slots").write_text(f"a {cpu}\n")
    child = ("import fcntl, os; fd = os.open(os.environ['LOCK'], os.O_RDWR)\n"
             "try:\n    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); held = False\n"
             "except BlockingIOError:\n    held = True\n"
             "print(sorted(os.sched_getaffinity(0)), held, os.environ['CHECK_SLOT'])")
    env = {**os.environ, "CHECK_SLOTS": str(tmp_path / "slots"), "LOCK": str(tmp_path / "check-a.lock")}
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "check" / "slot.py"), "--", sys.executable, "-c", child],
                         env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == [f"[{cpu}]", "True", "a"], out.stderr
    assert f"check slot a (cores {cpu})" in out.stderr


def test_status_names_each_slots_holder_or_free(tmp_path, monkeypatch):
    """Who holds check-a/b/c and since when, without `fuser` on the lock files (old-circuits-and-proofs, 1 Oct)."""
    w = tmp_path / "windows"
    w.write_text("2026-10-01T09:20Z 30\n")
    slots = SLOT.parse(f"a 32-63\nb 64-95\nd 0-47 windows={w}\n")
    held = _hold(tmp_path / "check-a.lock")
    a, b, d = SLOT.status(slots, tmp_path, NOW)
    me = Path(f"/proc/{os.getpid()}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()[:200]
    assert a.startswith(f"a  cores 32-63  held by pid {os.getpid()} since ") and a.endswith(f": {me}")
    assert b == "b  cores 64-95  free"
    assert d.startswith("d  cores 0-47  free; blocked: window 2026-10-01T09:20Z")
    os.close(held)
    assert SLOT.status(slots[:1], tmp_path, NOW) == ["a  cores 32-63  free"]
    (tmp_path / "slots").write_text("a 0\n")
    monkeypatch.setattr(SLOT, "SLOTS", tmp_path / "slots")
    assert SLOT.main(["--status"], now=lambda: NOW) == 0


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
    spec2 = importlib.util.spec_from_file_location("repo_check_slot", ROOT / "tools" / "check" / "check.py")
    check = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(check)
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
    assert remote[:3] == check.SLOT_WRAP[:3] == ["python3", "tools/check/slot.py", "--"] and remote[3:5] == ["uv", "run"]
    monkeypatch.setenv("RESEARCH_LANE", "proofs")
    assert check.record(["--on", "vy-nebius-2"]) == 0
    assert calls[-1][calls[-1].index("--") + 1:][:5] == ["python3", "tools/check/slot.py", "--by", "proofs", "--"]
    assert check.record([]) == 0 and "slot.py" not in " ".join(calls[-1])


def test_agreement_sends_the_upstream_build_whatever_the_commit_changes(monkeypatch, tmp_path):
    """`--record --agreement`: the lander's check of a train that changes a pinned guarantee outside backends/flock/ still runs
    the agreement step, with the pinned build and inputs as the run's inputs."""
    spec2 = importlib.util.spec_from_file_location("repo_check_agreement", ROOT / "tools" / "check" / "check.py")
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


def test_priority_reaches_slot_py_from_record_and_unknown_options_are_refused(monkeypatch):
    spec2 = importlib.util.spec_from_file_location("repo_check_priority", ROOT / "tools" / "check" / "check.py")
    check = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(check)
    calls = []
    monkeypatch.setattr(check.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(check, "lean_deps_url_file", lambda d: None)
    monkeypatch.delenv("RESEARCH_AGENT_URL", raising=False)
    monkeypatch.delenv("RESEARCH_LANE", raising=False)
    assert check.record(["--on", "vy-nebius-1", "--priority"]) == 0
    remote = calls[-1][calls[-1].index("--") + 1:]
    assert remote[:4] == ["python3", "tools/check/slot.py", "--priority", "--"] and "--priority" not in remote[4:]
    monkeypatch.setenv("RESEARCH_LANE", "lander")
    assert check.record(["--on", "vy-nebius-1", "--priority"]) == 0
    assert calls[-1][calls[-1].index("--") + 1:][:6] == ["python3", "tools/check/slot.py", "--priority", "--by", "lander", "--"]
    with pytest.raises(SystemExit):
        SLOT.main(["--fast", "--", "true"])
