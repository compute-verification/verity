"""The guard never fails open (2026-09-28 18:10 PT: the control pod's disk filled and every guard died on its state write): it
keeps enforcing through failed writes, keeps the last good budgets when budgets.toml goes, survives a corrupt state file and restarts
after a crash.  Every clock and sleep is injected."""
from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from research.pods import guard as G

from test_budgets_guard import T0, TOML, FakeRunPod, cfg, pod


@pytest.fixture(autouse=True)
def guard_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_GUARD_DIR", str(tmp_path / "guard"))
    return tmp_path / "guard"


@pytest.fixture
def budgets(tmp_path) -> Path:
    (tmp_path / "budgets.toml").write_text(TOML)
    return tmp_path


def clocked(step_s: float = 1800.0):
    t = [T0]
    return (lambda: t[0]), (lambda s: t.__setitem__(0, t[0] + step_s))


def disk_full(*_a, **_k):
    raise OSError(errno.ENOSPC, "No space left on device")


def test_a_guard_whose_disk_is_full_still_trips_at_its_cap_and_terminates(budgets, monkeypatch, capsys):
    monkeypatch.setattr(G, "save_state", disk_full)
    monkeypatch.setattr(G, "_append", disk_full)
    api = FakeRunPod([pod("a", "vy-a-1", 2.0, 60), pod("b", "vy-a-2", 2.0, 30)])
    clock, sleep = clocked()
    assert G.run(cfg(budgets), api, clock=clock, sleep=sleep, iterations=3) == 0
    assert sorted(api.deleted) == ["a", "b"], "$4/h x 1 h >= $3: the trip terminates from memory, with no write succeeding"
    err = capsys.readouterr().err
    assert "state write failed" in err and "log write failed" in err and "enforcing from memory" in err
    assert err.count("state write failed") == 1, "a failure is reported when it starts, not on every poll"


def test_a_guard_whose_state_directory_cannot_exist_still_trips(budgets, tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("a file where the guard dir would be")
    monkeypatch.setenv("RESEARCH_GUARD_DIR", str(blocker / "pods"))
    api = FakeRunPod([pod("a", "vy-a-1", 6.0, 60)])
    clock, sleep = clocked()
    assert G.run(cfg(budgets), api, clock=clock, sleep=sleep, iterations=2) == 0
    assert api.deleted == ["a"]
    assert "directory write failed" in capsys.readouterr().err


def test_a_write_that_recovers_records_the_fault_it_had(budgets, guard_dir, monkeypatch, capsys):
    guard_dir.mkdir()
    real, fail = G.save_state, [True]
    monkeypatch.setattr(G, "save_state", lambda path, state: disk_full() if fail[0] else real(path, state))
    rec = G.Recorder(G.paths(), lambda: T0)
    state = G.new_state(T0)
    rec.save(state)
    rec.save(state)
    fail[0] = False
    rec.save(state)
    assert "state writes work again after 2 failure(s)" in capsys.readouterr().err
    assert json.loads((guard_dir / "guard-budgets.json").read_text())["faults"]["state"]["count"] == 2, \
        "the state records the fault it had once a write succeeds"
    rec.save(state)
    assert json.loads((guard_dir / "guard-budgets.json").read_text())["faults"] == {}


def test_an_unreadable_budgets_file_keeps_its_limit(budgets, guard_dir):
    api = FakeRunPod([pod("a", "vy-a-1", 4.0, 60)])
    clock, sleep = clocked()
    c = cfg(budgets)
    assert G.run(c, api, clock=clock, sleep=sleep, iterations=1) == 0
    (budgets / "budgets.toml").write_text("[budgets\n")
    assert G.run(c, api, clock=clock, sleep=sleep, iterations=3) == 0
    assert api.deleted == ["a"], "$4/h x 1 h >= $3 an hour on: the cap in force is the last good file's"
    assert "does not parse" in json.loads((guard_dir / "guard-budgets.json").read_text())["budgets"]["error"]


def test_a_corrupt_state_file_is_set_aside_and_the_guard_keeps_enforcing(budgets, guard_dir):
    guard_dir.mkdir()
    (guard_dir / "guard-budgets.json").write_text('{"schema": "research/budgets-guard/v1", "lines": {"vy-a-": ')
    api = FakeRunPod([pod("a", "vy-a-1", 8.0, 60)])
    clock, sleep = clocked()
    assert G.run(cfg(budgets), api, clock=clock, sleep=sleep, iterations=2) == 0
    assert api.deleted == ["a"]
    state = json.loads((guard_dir / "guard-budgets.json").read_text())
    kept = Path(state["tally_restarted"]["kept"])
    assert "does not parse" in state["tally_restarted"]["why"] and kept.read_text().startswith('{"schema"')


def test_each_poll_terminates_before_it_logs():
    from research.pods import budgets as B

    api, lines = FakeRunPod([pod("a", "vy-a-1", 4.0, 60)]), []
    state = G.new_state(T0)
    order = []
    real_terminate = api.terminate
    api.terminate = lambda pid: (order.append(("terminate", pid)), real_terminate(pid))
    b = B.parse(TOML.replace("cap_usd = 3,", "cap_usd = 0,"))
    G.poll(api, b, cfg(), state, T0, lambda m: order.append(("log", m.split()[0])))
    assert order[0] == ("terminate", "a") and [o[1] for o in order[1:]] == ["TRIPPED:", "TERMINATED", "1"]


def test_the_supervisor_restarts_a_crashed_guard_one_poll_later_and_stops_when_it_is_stopped():
    rcs, slept = [1, -9, 0], []
    assert G.supervise(["guard"], 60.0, call=lambda cmd: rcs.pop(0), sleep=slept.append, clock=lambda: T0) == 0
    assert slept == [60.0, 60.0] and rcs == []
    for stop in G.STOP_CODES:
        slept.clear()
        assert G.supervise(["guard"], 60.0, call=lambda cmd, rc=stop: rc, sleep=slept.append, clock=lambda: T0) == stop and slept == []
    tries: list = [OSError(errno.EAGAIN, "fork"), 0]

    def flaky(cmd):
        x = tries.pop(0)
        if isinstance(x, Exception):
            raise x
        return x

    slept.clear()
    assert G.supervise(["guard"], 5.0, call=flaky, sleep=slept.append, clock=lambda: T0) == 0 and slept == [5.0]


def test_the_supervisor_marks_the_guard_it_runs_so_a_restart_never_refuses_on_a_broken_file(budgets, monkeypatch, capsys):
    from research.cli import main as research_main

    (budgets / "budgets.toml").write_text("[budgets\n")
    monkeypatch.setenv(G.SUPERVISED_ENV, "1")
    ran = []
    monkeypatch.setattr(G, "run", lambda c, **kw: ran.append(c) or 0)
    assert research_main(["pods", "guard", "--budgets", str(budgets / "budgets.toml")]) == 0 and ran, \
        "a supervised restart runs on the budgets in its state instead of exiting 2 (which would end the supervisor)"
