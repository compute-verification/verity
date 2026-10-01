"""`tools/check/slot.py`: a `--record --on` check takes a free check slot's lock and cores, as the lander's train checks do."""
from __future__ import annotations

import fcntl
import importlib.util
import os
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
    assert check.record(["--on", "vy-nebius-2"]) == 0
    remote = calls[-1][calls[-1].index("--") + 1:]
    assert remote[:3] == check.SLOT_WRAP[:3] == ["python3", "tools/check/slot.py", "--"] and remote[3:5] == ["uv", "run"]
    assert check.record([]) == 0 and "slot.py" not in " ".join(calls[-1])
