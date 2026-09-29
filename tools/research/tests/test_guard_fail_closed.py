"""Guards never fail open (2026-09-28 18:10 PT: the control pod's disk filled and every guard died on its state write):
`research pods guard` and `pods/budget_cap.py` keep enforcing through failed writes, keep the last cap when the cap file goes, survive a
corrupt state file and restart after a crash.  Every clock and sleep is injected."""
from __future__ import annotations

import errno
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.pods import guard as G

from test_pods_guard import T0, FakeRunPod, pod


@pytest.fixture(autouse=True)
def guard_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_GUARD_DIR", str(tmp_path / "guard"))
    return tmp_path / "guard"


def clocked(step_s: float = 1800.0):
    t = [T0]
    return (lambda: t[0]), (lambda s: t.__setitem__(0, t[0] + step_s))


def disk_full(*_a, **_k):
    raise OSError(errno.ENOSPC, "No space left on device")


# ---- research pods guard ------------------------------------------------------------------------------------------------------------------

def test_a_guard_whose_disk_is_full_still_trips_at_its_cap_and_terminates(monkeypatch, capsys):
    monkeypatch.setattr(G, "save_state", disk_full)
    monkeypatch.setattr(G, "_append", disk_full)
    api = FakeRunPod([pod("a", "vyv-a", 2.0, 60), pod("b", "vyv-b", 2.0, 30)])
    clock, sleep = clocked()
    assert G.run(G.Config(prefix="vyv-", cap_usd=3.0), api, clock=clock, sleep=sleep, iterations=3) == 0
    assert sorted(api.deleted) == ["a", "b"], "$4/h x 1 h >= $3: the trip terminates from memory, with no write succeeding"
    err = capsys.readouterr().err
    assert "state write failed" in err and "log write failed" in err and "enforcing from memory" in err
    assert err.count("state write failed") == 1, "a failure is reported when it starts, not on every poll"


def test_a_guard_whose_state_directory_cannot_exist_still_trips(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("a file where the guard dir would be")
    monkeypatch.setenv("RESEARCH_GUARD_DIR", str(blocker / "pods"))
    api = FakeRunPod([pod("a", "vyv-a", 6.0, 60)])
    clock, sleep = clocked()
    assert G.run(G.Config(prefix="vyv-", cap_usd=2.0), api, clock=clock, sleep=sleep, iterations=2) == 0
    assert api.deleted == ["a"]
    assert "directory write failed" in capsys.readouterr().err


def test_a_write_that_recovers_records_the_fault_it_had(guard_dir, monkeypatch, capsys):
    guard_dir.mkdir()
    real, fail = G.save_state, [True]
    monkeypatch.setattr(G, "save_state", lambda path, state: disk_full() if fail[0] else real(path, state))
    api = FakeRunPod([pod("a", "vyv-a", 1.0, 60)])
    clock, sleep = clocked(60)
    cfg = G.Config(prefix="vyv-", cap_usd=100.0)
    state = G.new_state(cfg, T0)
    rec = G.Recorder(G.paths("vyv-"), clock)
    for _ in range(2):
        G.poll(api, cfg, state, clock(), rec.log)
        rec.save(state)
        sleep(60)
    fail[0] = False
    rec.save(state)
    assert "state writes work again after 2 failure(s)" in capsys.readouterr().err
    written = json.loads((guard_dir / "guard-vyv-.json").read_text())
    assert written["faults"]["state"]["count"] == 2, "the state records the fault it had once a write succeeds"
    rec.save(state)
    assert json.loads((guard_dir / "guard-vyv-.json").read_text())["faults"] == {}


def test_an_unreadable_cap_file_keeps_the_cap_in_force_and_so_does_a_restart(tmp_path, guard_dir):
    cap_file = tmp_path / "CAP"
    cap_file.write_text("10\n")
    api = FakeRunPod([pod("a", "vyv-a", 4.0, 60)])
    clock, sleep = clocked()
    cfg = G.Config(prefix="vyv-", cap_usd=1.0, cap_file=str(cap_file))
    assert G.run(cfg, api, clock=clock, sleep=sleep, iterations=1) == 0
    cap_file.unlink()
    assert G.run(cfg, api, clock=clock, sleep=sleep, iterations=2) == 0
    state = json.loads((guard_dir / "guard-vyv-.json").read_text())
    assert state["cap"] == 10.0 and state["spent"] == pytest.approx(2.0) and api.deleted == [], \
        "$2 spent: over --cap-usd 1, under the file's $10 still in force"
    assert "FileNotFoundError" in state["cap_file_error"] and "keeping the cap in force, $10" in (guard_dir / "guard-vyv-.log").read_text()
    cap_file.write_text("garbage")
    assert G.run(cfg, api, clock=clock, sleep=sleep, iterations=5) == 0
    assert api.deleted == ["a"] and "CAP $10 reached" in json.loads((guard_dir / "guard-vyv-.json").read_text())["tripped"]["why"]


def test_a_cap_file_alone_that_was_never_readable_is_refused_at_start(tmp_path, capsys):
    from research.cli import main as research_main

    assert research_main(["pods", "guard", "--prefix", "vyv-", "--cap-file", str(tmp_path / "absent")]) == 2
    assert "no cap to enforce" in capsys.readouterr().err


def test_a_corrupt_state_file_is_set_aside_and_the_guard_keeps_enforcing(guard_dir):
    guard_dir.mkdir()
    (guard_dir / "guard-vyv-.json").write_text('{"spent": 3.0, "pods": {"a": ')
    api = FakeRunPod([pod("a", "vyv-a", 4.0, 60)])
    clock, sleep = clocked()
    assert G.run(G.Config(prefix="vyv-", cap_usd=2.0), api, clock=clock, sleep=sleep, iterations=2) == 0
    assert api.deleted == ["a"]
    state = json.loads((guard_dir / "guard-vyv-.json").read_text())
    kept = Path(state["tally_restarted"]["kept"])
    assert "does not parse" in state["tally_restarted"]["why"] and kept.read_text().startswith('{"spent": 3.0')


def test_the_pod_list_is_logged_after_the_terminations(guard_dir):
    api, lines = FakeRunPod([pod("a", "vyv-a", 4.0, 60)]), []
    cfg = G.Config(prefix="vyv-", cap_usd=0.0)
    G.poll(api, cfg, G.new_state(cfg, T0), T0, lines.append)
    assert [ln.split()[0] for ln in lines] == ["TRIPPED:", "TERMINATED", "1"]


def test_the_supervisor_restarts_a_crashed_guard_one_poll_later_and_stops_when_it_is_stopped():
    rcs, slept = [1, -9, 0], []

    def call(cmd):
        return rcs.pop(0)

    assert G.supervise(["guard"], 60.0, call=call, sleep=slept.append, clock=lambda: T0) == 0
    assert slept == [60.0, 60.0] and rcs == []
    for stop in G.STOP_CODES:
        slept.clear()
        assert G.supervise(["guard"], 60.0, call=lambda cmd, rc=stop: rc, sleep=slept.append, clock=lambda: T0) == stop and slept == []
    tries = [OSError(errno.EAGAIN, "fork"), 0]

    def flaky(cmd):
        x = tries.pop(0)
        if isinstance(x, Exception):
            raise x
        return x

    slept.clear()
    assert G.supervise(["guard"], 5.0, call=flaky, sleep=slept.append, clock=lambda: T0) == 0 and slept == [5.0]


# ---- pods/budget_cap.py ------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def budget_cap(tmp_path, monkeypatch):
    from research.pods import budget_cap as B

    monkeypatch.setattr(B, "HOME", tmp_path / "runpod")
    (tmp_path / "runpod").mkdir()
    fleet = {"a": {"id": "a", "name": "cp-a", "costPerHr": 4.0, "desiredStatus": "RUNNING"}}
    deleted: list[str] = []

    def request(method, path, body=None):
        if method == "GET":
            return [p for p in fleet.values() if p["desiredStatus"] == "RUNNING"]
        deleted.append(path.rsplit("/", 1)[1])
        fleet[deleted[-1]]["desiredStatus"] = "TERMINATED"

    monkeypatch.setattr(B.runpod, "_request", request)
    t = [T0]
    monkeypatch.setattr(B, "_now", lambda: t[0])
    return SimpleNamespace(B=B, deleted=deleted, t=t, home=tmp_path / "runpod")


def test_budget_cap_terminates_at_its_cap_while_no_write_succeeds(budget_cap, monkeypatch, capsys):
    B = budget_cap.B
    monkeypatch.setattr(B, "HOME", budget_cap.home / "gone" / "deeper")
    state = B._load("tag")
    B._poll("tag", 3.0, 50.0, ["cp-"], None, None, state)
    budget_cap.t[0] += 3600
    B._poll("tag", 3.0, 50.0, ["cp-"], None, None, state)
    assert budget_cap.deleted == ["a"] and "CAP $3.0 reached" in state["tripped"]
    err = capsys.readouterr().err
    assert "state write failed" in err and "log write failed" in err


def test_budget_cap_keeps_the_cap_in_force_across_a_restart_when_the_cap_file_goes(budget_cap, tmp_path, monkeypatch):
    B = budget_cap.B
    cap_file = tmp_path / "CAP"
    cap_file.write_text("10")
    state = B._load("tag")
    cap = B._poll("tag", 1.0, 50.0, ["cp-"], str(cap_file), None, state)
    assert cap == 10.0 and json.loads((budget_cap.home / "budget-tag.json").read_text())["cap"] == 10.0
    cap_file.unlink()

    class Stop(Exception):
        pass

    def sleep(s):
        budget_cap.t[0] += 3600
        if budget_cap.t[0] > T0 + 2 * 3600:
            raise Stop

    monkeypatch.setattr(B, "time", SimpleNamespace(sleep=sleep, time=lambda: budget_cap.t[0], strptime=__import__("time").strptime))
    monkeypatch.setattr(B, "_other_daemons", lambda: [])
    with pytest.raises(Stop):
        B.run("tag", 1.0, 50.0, prefixes=["cp-"], cap_file=str(cap_file))
    state = json.loads((budget_cap.home / "budget-tag.json").read_text())
    assert state["spent"] == pytest.approx(8.0) and state["tripped"] is None and budget_cap.deleted == [], \
        "$8 spent: over --cap 1, under the $10 in force when the file went"


def test_budget_cap_sets_a_corrupt_state_aside(budget_cap, capsys):
    B = budget_cap.B
    (budget_cap.home / "budget-tag.json").write_text("{not json")
    state = B._load("tag")
    assert state["spent"] == 0.0 and state["tally_restarted"] and "set aside" in capsys.readouterr().err
    assert any(p.name.startswith("budget-tag.json.corrupt-") for p in budget_cap.home.iterdir())


def test_budget_cap_start_runs_the_daemon_under_a_respawn_loop(budget_cap, monkeypatch):
    B = budget_cap.B
    seen = {}

    class Proc:
        pid = 4242

    def popen(cmd, **kw):
        seen["cmd"] = cmd
        return Proc()

    monkeypatch.setattr(B, "_other_daemons", lambda: [])
    monkeypatch.setattr(B.subprocess, "Popen", popen)
    B.start(SimpleNamespace(tag="tag", cap=5.0, rate_max=50.0, own_prefix=["cp-"], allow_multiple=False, cap_file=None, pod_max_hours=None))
    cmd = seen["cmd"]
    assert cmd[:3] == ["sh", "-c", B.RESPAWN] and cmd[4] == str(B.POLL) and "run" in cmd[5:]
    assert (budget_cap.home / "budget-tag.pid").read_text() == "4242"


def test_budget_cap_respawn_loop_restarts_until_the_daemon_says_another_runs(tmp_path):
    import subprocess

    count = tmp_path / "count"
    script = tmp_path / "daemon.sh"
    script.write_text(f'n=$(cat {count} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {count}; [ $n -ge 3 ] && exit 3; exit 1\n')
    from research.pods import budget_cap as B

    r = subprocess.run(["sh", "-c", B.RESPAWN, "respawn", "0", "sh", str(script)], capture_output=True, text=True)
    assert r.returncode == 3 and count.read_text().strip() == "3" and r.stdout.count("restarting") == 2
