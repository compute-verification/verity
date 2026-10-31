"""`tools/verity/check/lean_slot.py`: a Lean audit takes one of the node's N slots, waits for one naming their holders, and frees it on
exit; with no cap set it runs at once. And check's lean-audit step takes one only when it audits a package."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("check_lean_slot", ROOT / "tools" / "verity" / "check" / "lean_slot.py")
LS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LS)
NOW = datetime(2026, 10, 3, 17, 0, tzinfo=timezone.utc)


class Exec(Exception):
    pass


@pytest.fixture
def slots(tmp_path, monkeypatch):
    """A node whose steward set `audit 2` and `check 2`, with no slot held and nobody asking from a check slot."""
    f = tmp_path / "lean-slots"
    f.write_text("# vy-nebius-1: 4, split for proofs and the lander\naudit 2\ncheck 2   # check's lean-audit step\n")
    monkeypatch.setattr(LS, "SLOTS", f)
    for v in (LS.HELD, "LEAN_SLOT_POOL", "CHECK_SLOT", "RESEARCH_LANE", "RESEARCH_RUN_ID", "RESEARCH_AGENT_URL"):
        monkeypatch.delenv(v, raising=False)
    return tmp_path


def _hold(path: Path, who: dict | None = None) -> int:
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if who:
        path.with_suffix(".who").write_text(json.dumps(who))
    return fd


def _held(path: Path) -> bool:
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def test_n_requests_take_the_n_slots_and_the_next_waits_until_one_frees(slots):
    first = LS.wait("audit", now=lambda: NOW, log=lambda s: None)
    second = LS.wait("audit", now=lambda: NOW, log=lambda s: None)
    assert [first[:2], second[:2]] == [("audit", 0), ("audit", 1)]
    assert LS.take(slots, "audit", 2) is None
    polls = []

    def sleep(_s):                                          # the third request polls; slot 1's holder exits on the third poll
        polls.append(1)
        if len(polls) == 3:
            os.close(second[2])

    third = LS.wait("audit", now=lambda: NOW, sleep=sleep, log=lambda s: None)
    assert third[:2] == ("audit", 1) and len(polls) == 3
    other = LS.wait("check", now=lambda: NOW, log=lambda s: None)
    assert other[:2] == ("check", 0)                                                      # another pool's slots are its own
    assert not list((slots / "lean-line-audit").iterdir())                                # its ticket left the line
    for fd in (first[2], third[2], other[2]):
        os.close(fd)


def test_a_waiter_names_each_holder_and_repeats_the_list_every_few_minutes(slots, monkeypatch):
    """The holders' pid, run, who and since when, so a waiting audit's log says whom it waits behind."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-me")
    me = os.getpid()
    held = [_hold(slots / f"lean-audit-{i}.lock", {"pid": me, "run": f"r-{i}", "tree": "/src/740daff33", "by": by,
                                                    "took": "2026-10-03T16:40Z"})
            for i, by in enumerate(("zk-cz", "pc-interp"))]
    lines, polls = [], []

    def sleep(_s):
        polls.append(1)
        if len(polls) == 2 * LS.REPORT_EVERY + 1:
            os.close(held.pop())

    got = LS.wait("audit", now=lambda: NOW, sleep=sleep, log=lines.append)
    assert got[:2] == ("audit", 1)
    heads = [ln for ln in lines if not ln.startswith("lean-slot:   ")]
    assert heads[:3] == [f"lean-slot: all 2 Lean audit slot(s) of pool audit are taken; waiting, with no time limit, and retrying "
                         f"every {LS.POLL_S} s ({LS.SLOTS})"] * 3                                # at once, then every 3 min
    assert heads[3] == f"lean-slot: Lean audit slot audit/1 of 2 ({LS.SLOTS})" and len(heads) == 4
    a0, a1 = lines[1:3]
    assert a0.startswith(f"lean-slot:   audit/0  held by pid {me} since ")
    assert "(run r-0, tree 740daff33, by zk-cz, slot taken 2026-10-03T16:40Z): " in a0 and "by pc-interp" in a1
    who = json.loads((slots / "lean-audit-1.who").read_text())
    assert who == {"pid": me, "run": "r-me", "tree": os.getcwd(), "by": None, "took": "2026-10-03T17:00Z"}
    os.close(held.pop())
    os.close(got[2])


def test_a_waiter_goes_behind_the_live_waiters_that_came_first(slots):
    line = slots / "lean-line-audit"
    line.mkdir()
    (line / f"{1:020d}-1").write_text("")                                    # a waiter that died: nobody holds its ticket
    first = _hold(line / f"{2:020d}-2")                                      # a live one; both slots are free
    lines = []

    def sleep(_s):
        (line / f"{2:020d}-2").unlink()                                      # it takes its slot
        os.close(first)

    got = LS.wait("audit", now=lambda: NOW, sleep=sleep, log=lines.append)
    assert got[:2] == ("audit", 0) and lines[0].startswith("lean-slot: 1 Lean audit(s) ahead in line; ")
    assert lines[1].startswith("lean-slot:   audit/0  free") and "audit: waiting since " in lines[3]
    os.close(got[2])


def test_without_the_steward_s_file_there_is_no_cap(tmp_path, monkeypatch):
    """A laptop, or a node whose steward set no N: the audit runs at once, and nothing is written."""
    monkeypatch.setattr(LS, "SLOTS", tmp_path / "absent")
    monkeypatch.delenv(LS.HELD, raising=False)
    monkeypatch.setattr(LS.os, "execvp", lambda f, argv: (_ for _ in ()).throw(Exec(argv)))
    with pytest.raises(Exec) as e:
        LS.main(["--", "python", "tools/verity/lean/check.py", "--build", "pkg"])
    assert e.value.args[0] == ["python", "tools/verity/lean/check.py", "--build", "pkg"]
    assert LS.hold("audit") is None and LS.HELD not in os.environ and list(tmp_path.iterdir()) == []
    assert LS.main(["--status"]) == 0


def test_a_request_inside_a_slot_takes_no_second_one(slots, monkeypatch):
    """check.py under check's lean-audit step: the step holds the slot, so the nested request neither waits nor counts."""
    held = [_hold(slots / f"lean-audit-{i}.lock") for i in range(2)]
    monkeypatch.setenv(LS.HELD, "check/0")
    assert LS.hold("audit") is None
    monkeypatch.setattr(LS.os, "execvp", lambda f, argv: (_ for _ in ()).throw(Exec(argv)))
    with pytest.raises(Exec):
        LS.main(["--", "true"])
    for fd in held:
        os.close(fd)


def test_a_pool_the_file_doesn_t_list_takes_from_any_or_is_refused_by_name(slots, monkeypatch, capsys):
    LS.SLOTS.write_text("any 1\n")
    assert LS.resolve(LS.parse(LS.SLOTS.read_text()), "audit") == ("any", 1)
    with pytest.raises(LS.Refused, match="lists no pool 'build', the only one this request takes; the steward adds `build N`"):
        LS.hold("build", or_any=False)
    LS.SLOTS.write_text("check 2\n")
    with pytest.raises(LS.Refused, match="lists neither pool 'audit' nor 'any'; the steward adds `audit N`"):
        LS.hold("audit")
    with pytest.raises(SystemExit, match="lean-slot: .* lists neither pool 'audit'"):
        LS.main(["--", "true"])
    LS.SLOTS.write_text("audit two\n")
    with pytest.raises(LS.Refused, match="'audit two' is not `POOL N`"):
        LS.hold("audit")
    with pytest.raises(SystemExit, match="usage"):
        LS.main(["--fast", "--", "true"])


def test_the_steward_can_change_n_while_requests_wait(slots):
    held = _hold(slots / "lean-audit-0.lock")
    LS.SLOTS.write_text("audit 1\n")

    def sleep(_s):
        LS.SLOTS.write_text("audit 2\n")

    got = LS.wait("audit", now=lambda: NOW, sleep=sleep, log=lambda s: None)
    assert got[:2] == ("audit", 1)
    LS.SLOTS.write_text("audit 1\n")                             # lowered while slot 1 is held: still listed, past N
    assert any(ln.startswith("audit/1 (past N)  held by pid") for ln in LS.status(LS.parse("audit 1"), slots, NOW))
    os.close(held)
    os.close(got[2])
    assert LS.status(LS.parse("audit 1"), slots, NOW) == ["audit/0  free"]


def test_who_asked_is_the_lane_else_the_check_slot_s_holder(slots, monkeypatch, tmp_path):
    checks = tmp_path / "checks"
    checks.mkdir()
    (checks / "check-b.who").write_text(json.dumps({"pid": 1, "by": "lander"}))
    monkeypatch.setattr(LS.SLOT, "SLOTS", checks / "slots")
    monkeypatch.setenv("CHECK_SLOT", "b")
    assert LS.asker()["by"] == "lander"
    monkeypatch.setenv("RESEARCH_LANE", "lean-slots")
    assert LS.asker()["by"] == "lean-slots" and LS.asker("me")["by"] == "me"


def test_status_lists_each_pool_s_slots_and_waiters(slots, monkeypatch, capsys):
    held = _hold(slots / "lean-check-1.lock", {"pid": os.getpid(), "run": "r-train", "by": "lander"})
    lines = LS.status(LS.parse(LS.SLOTS.read_text()), slots, NOW)
    assert lines[:3] == ["audit/0  free", "audit/1  free", "check/0  free"]
    assert lines[3].startswith(f"check/1  held by pid {os.getpid()} since ") and "(run r-train, by lander): " in lines[3]
    assert LS.main(["--status"], now=lambda: NOW) == 0 and "check/1  held by pid" in capsys.readouterr().out
    os.close(held)


def test_status_json_is_each_pool_s_slots_holders_and_waiters(slots, monkeypatch, capsys):
    """`--status --json`, which `check.py --slots` reads from each machine; null pools where no cap is set (node 2)."""
    held = _hold(slots / "lean-check-1.lock", {"pid": os.getpid(), "run": "r-train", "by": "lander", "took": "2026-10-03T16:40Z"})
    line = slots / "lean-line-audit"
    line.mkdir()
    waiter = line / f"{int((NOW.timestamp() - 120) * 1e9):020d}-77"
    waiter.write_text(json.dumps({"pid": 77, "run": "r-audit", "by": "proofs"}))
    wfd = _hold(waiter)
    past = _hold(slots / "lean-audit-2.lock")                                  # the steward lowered N while it ran
    assert LS.main(["--status", "--json"], now=lambda: NOW) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["file"] == str(LS.SLOTS) and list(doc["pools"]) == ["audit", "check"]
    audit, check = doc["pools"]["audit"], doc["pools"]["check"]
    assert [(s["index"], s["past_n"], s["holder"] is None) for s in audit["slots"]] == [(0, False, True), (1, False, True), (2, True, False)]
    assert audit["waiters"] == [{"since": "2026-10-03T16:58Z", "minutes": 2, "priority": False, "pid": 77, "run": "r-audit", "by": "proofs"}]
    assert check["n"] == 2 and check["slots"][0] == {"index": 0, "past_n": False, "holder": None} and check["waiters"] == []
    h = check["slots"][1]["holder"]
    assert h["pid"] == os.getpid() and (h["run"], h["by"], h["took"]) == ("r-train", "lander", "2026-10-03T16:40Z")
    for fd in (held, wfd, past):
        os.close(fd)
    monkeypatch.setattr(LS, "SLOTS", slots / "absent")
    assert LS.main(["--status", "--json"], now=lambda: NOW) == 0
    assert json.loads(capsys.readouterr().out) == {"file": str(slots / "absent"), "now": "2026-10-03T17:00Z", "pools": None}
    with pytest.raises(SystemExit, match="usage: lean_slot.py --status"):
        LS.main(["--status", "--all"])


def test_the_command_holds_the_slot_while_it_runs_and_sigterm_frees_it(slots):
    """The real exec, ended as `research cancel` ends it: the kernel drops the flock with the process."""
    child = ("import os, sys\nprint(os.environ['LEAN_SLOT'], flush=True)\nsys.stdin.read()\n")
    env = {**os.environ, "LEAN_SLOTS": str(LS.SLOTS)}
    p = subprocess.Popen([sys.executable, str(ROOT / "tools" / "verity" / "check" / "lean_slot.py"), "--", sys.executable, "-c", child],
                         env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "audit/0"                     # running, in slot audit/0
        assert _held(slots / "lean-audit-0.lock")
        assert json.loads((slots / "lean-audit-0.who").read_text())["pid"] == p.pid
        p.send_signal(signal.SIGTERM)
        assert p.wait() == -signal.SIGTERM
        assert not _held(slots / "lean-audit-0.lock")
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        p.stdin.close()
        p.stdout.close()
        p.stderr.close()


def _check():
    s = importlib.util.spec_from_file_location("repo_check_lean_slot", ROOT / "tools" / "verity" / "check" / "check.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_check_s_lean_audit_takes_a_slot_only_when_it_will_audit_a_package(tmp_path, monkeypatch):
    """A step that only reuses passes would otherwise wait behind an hour-long audit for seconds of work."""
    check = _check()
    keys = {"p": "kp", "q": "kq"}
    monkeypatch.setattr(check, "_lean_audit", lambda: SimpleNamespace(packages=lambda files: sorted(keys), FLAGS=["--build"],
                                                                      key=lambda p, files, flags: keys[p]))
    monkeypatch.setattr(check, "tracked_files", lambda root: ["p/lean-audit.json", "q/lean-audit.json"])
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path))
    (tmp_path / "lean-audit").mkdir()
    (tmp_path / "lean-audit" / "kp.json").write_text("{}")
    cmd = [sys.executable, "tools/verity/lean/lean_audit.py", "--out", "o"]
    assert check.lean_slot(cmd)[:len(check.LEAN_SLOT_WRAP)] == check.LEAN_SLOT_WRAP           # q misses
    assert check.LEAN_SLOT_WRAP[1].endswith("tools/verity/check/lean_slot.py") and check.LEAN_SLOT_WRAP[2:4] == ["--pool", "check"]
    (tmp_path / "lean-audit" / "kq.json").write_text("{}")
    assert check.lean_slot(cmd) == cmd                                                         # every pass reused
    assert check.lean_slot([*cmd, "--fresh"])[:len(check.LEAN_SLOT_WRAP)] == check.LEAN_SLOT_WRAP
    monkeypatch.setattr(check.os, "execve", lambda f, argv, env: (_ for _ in ()).throw(Exec(argv, env)))
    with pytest.raises(Exec) as e:
        check.main(["--lean-slot", *cmd])
    assert e.value.args[0] == cmd and e.value.args[1]["LEAN_SLOT_POOL"] == "check"


def _audit_py():
    s = importlib.util.spec_from_file_location("repo_lean_check_slot", ROOT / "tools" / "verity" / "lean" / "check.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_the_lean_audit_run_by_hand_with_build_takes_a_slot_of_pool_audit(slots, monkeypatch):
    """`tools/verity/lean/check.py --build` outside check and lean_changed: a full audit that took no slot overran the cap."""
    audit = _audit_py()
    monkeypatch.setenv("LEAN_SLOTS", str(slots / "lean-slots"))
    held = [_hold(slots / "lean-audit-0.lock")]
    fd = audit.lean_slot(ROOT)
    assert os.environ.pop(LS.HELD) == "audit/1" and fd is not None
    os.close(fd)
    monkeypatch.setenv(LS.HELD, "check/0")
    assert audit.lean_slot(ROOT) is None                                 # nested: check's step or lean_changed holds one
    monkeypatch.delenv(LS.HELD)
    (slots / "lean-slots").write_text("check 2\n")
    with pytest.raises(RuntimeError):
        audit.lean_slot(ROOT)                                            # neither `audit` nor `any`: refused, never run
    assert audit.main(["--build", "--out", str(slots / "out"), str(slots)]) == 2
    monkeypatch.setattr(audit, "LEAN_SLOTS", slots / "lean-slots")
    with pytest.raises(RuntimeError, match="missing"):
        audit.lean_slot(slots)                                           # a repository without lean_slot.py, on a capped node
    monkeypatch.setattr(audit, "LEAN_SLOTS", slots / "absent")
    assert audit.lean_slot(slots) is None                                # a laptop
    for h in held:
        os.close(h)
