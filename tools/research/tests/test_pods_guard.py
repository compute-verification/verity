"""`research pods guard` (pods/guard.py) against a fake RunPod API: prefix scoping, newest-first rate shedding that never touches the
guard's own host, sticky trips (deadline / cap / balance floor), the spend tally across a restart, dry runs, and the CLI's refusals."""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys

import pytest

from research.cli import main as research_main
from research.pods import guard as G

T0 = G.parse_time("2026-09-23T20:00:00Z")


def pod(pid: str, name: str, rate: float, age_min: float, status: str = "RUNNING") -> dict:
    t = dt.datetime.fromtimestamp(T0 - age_min * 60, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " +0000 UTC"
    return {"id": pid, "name": name, "costPerHr": rate, "createdAt": t, "lastStartedAt": t, "desiredStatus": status}


class FakeRunPod:
    """GET /pods, DELETE /pods/<id>, clientBalance.  `lag`: a DELETEd pod is still listed RUNNING by the next list (RunPod does this)."""

    def __init__(self, pods: list[dict], balance: float = 100.0, lag: bool = False):
        self.fleet = {p["id"]: dict(p) for p in pods}
        self.deleted: list[str] = []
        self.bal, self.lag, self.fail_list, self._lagging = balance, lag, 0, set()

    def pods(self):
        if self.fail_list:
            self.fail_list -= 1
            raise OSError("temporary failure in name resolution")
        out = [dict(p) for p in self.fleet.values()]
        for pid in self._lagging:
            self.fleet[pid]["desiredStatus"] = "TERMINATED"
        self._lagging = set()
        return out

    def terminate(self, pid):
        self.deleted.append(pid)
        if self.lag:
            self._lagging.add(pid)
        else:
            self.fleet[pid]["desiredStatus"] = "TERMINATED"

    def balance(self):
        if isinstance(self.bal, Exception):
            raise self.bal
        return self.bal

    def add(self, p):
        self.fleet[p["id"]] = dict(p)


def fleet(**kw) -> FakeRunPod:
    return FakeRunPod([pod("a", "vyv-a", 2.0, 300), pod("b", "vyv-b", 2.0, 120), pod("c", "vyv-c", 2.0, 30), pod("me", "vyv-control", 1.0, 5),
                       pod("f", "vy-ligerito-h100", 10.0, 1), pod("x", "xvyv-lookalike", 5.0, 0), pod("d", "vyv-done", 3.0, 0, "EXITED")], **kw)


@pytest.fixture(autouse=True)
def guard_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_GUARD_DIR", str(tmp_path / "guard"))
    return tmp_path / "guard"


def step(api, cfg, state, t, lines):
    return G.poll(api, cfg, state, t, lines.append)


def test_runpod_and_iso_times_parse_and_a_zoneless_time_does_not():
    assert G.parse_time("2026-09-23 20:49:55.033 +0000 UTC") == dt.datetime(2026, 9, 23, 20, 49, 55, 33000, dt.timezone.utc).timestamp()
    assert G.parse_time("2026-09-23 20:49:55 +0000 UTC") == G.parse_time("2026-09-23T20:49:55Z")
    assert G.parse_time("2026-09-23T20:49:55") is None and G.parse_time("soon") is None and G.parse_time(None) is None


def test_rate_over_the_limit_sheds_exactly_the_newest_scoped_pods_and_never_the_guards_host():
    api, lines = fleet(), []
    cfg = G.Config(prefix="vyv-", rate_max=4.5, self_pod="me")
    state = G.new_state(cfg, T0)
    done = step(api, cfg, state, T0, lines)
    # scoped: a b c me = $7/h (f, x and the EXITED d are not the guard's); newest non-host first: c ($7 -> $5), b ($5 -> $3 <= $4.5)
    assert state["rate_now"] == 7.0
    assert api.deleted == ["c", "b"] == [d["id"] for d in done]
    assert [ln.split(" -- ")[0] for ln in lines if ln.startswith("TERMINATED")] == ["TERMINATED c vyv-c $2/h", "TERMINATED b vyv-b $2/h"]
    assert all("newest first" in d["why"] for d in done)
    step(api, cfg, state, T0 + 60, lines)
    assert api.deleted == ["c", "b"], "under the limit now: nothing more"

    alone = FakeRunPod([pod("me", "vyv-control", 9.0, 5)])
    lines = []
    step(alone, cfg, G.new_state(cfg, T0), T0, lines)
    assert alone.deleted == [] and any("never shed" in ln for ln in lines)


def test_a_pod_the_api_still_lists_after_delete_does_not_cost_another_pod():
    api, lines = fleet(lag=True), []
    cfg = G.Config(prefix="vyv-", rate_max=4.5, self_pod="me")
    state = G.new_state(cfg, T0)
    step(api, cfg, state, T0, lines)
    step(api, cfg, state, T0 + 60, lines)                                   # c and b are listed RUNNING once more
    assert api.deleted == ["c", "b"]
    assert state["pods"]["c"]["spent"] == pytest.approx(2.0 / 60), "still billed while it goes: counted in spend, not in the rate"


def test_only_prefix_pods_are_counted_or_terminated_even_when_everything_trips():
    api, lines = fleet(), []
    cfg = G.Config(prefix="vyv-", cap_usd=1.0, pod_max_hours=0.1, deadline=T0 + 1800, self_pod="me")
    state = G.new_state(cfg, T0)
    step(api, cfg, state, T0, lines)
    assert set(api.deleted) == {"a", "b", "c"}, "pod-max-hours: every scoped pod but the host (f is 1 min old but not ours; x is a lookalike)"
    step(api, cfg, state, T0 + 1800, lines)
    assert state["spent"] == pytest.approx(0.5), "only the host was left to bill: $1/h x 0.5 h; f's $10/h and x's $5/h never count"
    assert state["tripped"]["why"].startswith("DEADLINE") and api.deleted[-1] == "me"
    assert "f" not in api.deleted and "x" not in api.deleted and "d" not in api.deleted


def test_the_deadline_terminates_every_scoped_pod_own_host_last_and_keeps_terminating_newcomers():
    api, lines = fleet(), []
    cfg = G.Config(prefix="vyv-", deadline=T0 + 120, self_pod="me")
    state = G.new_state(cfg, T0)
    step(api, cfg, state, T0, lines)
    assert api.deleted == [] and state["tripped"] is None
    step(api, cfg, state, T0 + 120, lines)
    assert api.deleted == ["a", "b", "c", "me"], "the guard's own host goes last"
    api.add(pod("late", "vyv-late", 4.0, -5))
    step(api, cfg, state, T0 + 300, lines)
    assert api.deleted[-1] == "late" and set(api.deleted) == {"a", "b", "c", "me", "late"}


def test_the_cap_is_rate_times_time_seeded_by_baseline_and_a_restart_continues_the_tally(guard_dir):
    api = FakeRunPod([pod("a", "vyv-a", 2.0, 60), pod("b", "vyv-b", 2.0, 30), pod("f", "other", 50.0, 1)])
    t = [T0]
    clock, sleep = (lambda: t[0]), (lambda s: t.__setitem__(0, t[0] + 1800))
    cfg = G.Config(prefix="vyv-", cap_usd=5.0, baseline=1.0)
    assert G.run(cfg, api, clock=clock, sleep=sleep, iterations=2) == 0
    state = json.loads((guard_dir / "guard-vyv-.json").read_text())
    assert state["spent"] == pytest.approx(3.0) and state["tripped"] is None and api.deleted == []   # $1 baseline + $4/h x 0.5 h

    assert G.run(G.Config(prefix="vyv-", cap_usd=5.0, baseline=7.0), api, clock=clock, sleep=sleep, iterations=2) == 0
    state = json.loads((guard_dir / "guard-vyv-.json").read_text())
    assert state["spent"] == pytest.approx(5.0) and "CAP $5 reached" in state["tripped"]["why"]
    assert sorted(api.deleted) == ["a", "b"]
    log = (guard_dir / "guard-vyv-.log").read_text()
    assert "--baseline 7 ignored" in log and "TRIPPED: CAP $5" in log


def test_a_cap_file_raises_the_cap_each_poll(tmp_path):
    api, lines = FakeRunPod([pod("a", "vyv-a", 4.0, 60)]), []
    cap_file = tmp_path / "CAP"
    cap_file.write_text("50\n")
    cfg = G.Config(prefix="vyv-", cap_usd=1.0, cap_file=str(cap_file))
    state = G.new_state(cfg, T0)
    step(api, cfg, state, T0, lines)
    step(api, cfg, state, T0 + 3600, lines)
    assert state["spent"] == pytest.approx(4.0) and state["cap"] == 50.0 and state["tripped"] is None
    cap_file.write_text("3")
    step(api, cfg, state, T0 + 3660, lines)
    assert state["tripped"] and api.deleted == ["a"]


def test_pod_max_hours_terminates_old_scoped_pods_but_never_the_guards_host():
    api, lines = FakeRunPod([pod("a", "vyv-a", 1.0, 5 * 60), pod("me", "vyv-control", 1.0, 10 * 60), pod("b", "vyv-b", 1.0, 30)]), []
    cfg = G.Config(prefix="vyv-", pod_max_hours=4, self_pod="me")
    step(api, cfg, G.new_state(cfg, T0), T0, lines)
    assert api.deleted == ["a"] and "POD AGE 5.00 h > 4 h" in lines[-1]


def test_the_balance_floor_trips_and_an_unreadable_balance_does_not():
    cfg = G.Config(prefix="vyv-", balance_floor=5.0)
    api, lines = FakeRunPod([pod("a", "vyv-a", 1.0, 5), pod("f", "other", 1.0, 5)], balance=RuntimeError("graphql down")), []
    state = G.new_state(cfg, T0)
    step(api, cfg, state, T0, lines)
    assert state["tripped"] is None and any("balance read failed" in ln for ln in lines)
    api.bal = 4.0
    step(api, cfg, state, T0 + 60, lines)
    assert "BALANCE $4.00 <= floor $5" in state["tripped"]["why"] and api.deleted == ["a"]


def test_a_dry_run_calls_nothing_and_keeps_its_own_files(guard_dir):
    api = fleet()
    assert G.run(G.Config(prefix="vyv-", rate_max=4.5, deadline=T0, self_pod="me", dry_run=True), api, clock=lambda: T0,
                 sleep=lambda s: None, iterations=2) == 0
    assert api.deleted == []
    assert not (guard_dir / "guard-vyv-.json").exists(), "a dry run's sticky trip must never arm the real guard"
    state = json.loads((guard_dir / "guard-vyv-.dry-run.json").read_text())
    assert [e["id"] for e in state["terminated"]] == ["a", "b", "c", "me"] and all(e["dry_run"] for e in state["terminated"])
    log = (guard_dir / "guard-vyv-.dry-run.log").read_text()
    assert "WOULD-TERMINATE me vyv-control" in log and "\nTERMINATED" not in log and log.count("WOULD-TERMINATE") == 4


def test_a_failed_list_is_logged_and_the_guard_keeps_polling(guard_dir):
    api = fleet()
    api.fail_list = 1
    assert G.run(G.Config(prefix="vyv-", rate_max=4.5, self_pod="me"), api, clock=lambda: T0, sleep=lambda s: None, iterations=2) == 0
    log = (guard_dir / "guard-vyv-.log").read_text()
    assert "poll failed: OSError: temporary failure in name resolution" in log and api.deleted == ["c", "b"]


def test_the_cli_refuses_a_second_guard_or_no_limit_and_status_stop_read_the_recorded_pid(guard_dir, capsys):
    assert research_main(["pods", "guard", "--prefix", "vyv-"]) == 2
    assert research_main(["pods", "guard", "--prefix", "vyv-", "--deadline", "2026-09-24T03:00"]) == 2, "a zoneless deadline is refused"
    assert research_main(["pods", "guard", "status", "--prefix", "vyv-"]) == 2
    G.run(G.Config(prefix="vyv-", rate_max=4.5, self_pod="me"), fleet(), clock=lambda: T0, sleep=lambda s: None, iterations=1)

    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "guard"])
    try:
        (guard_dir / "guard-vyv-.pid").write_text(f"{other.pid}\n")
        assert research_main(["pods", "guard", "--prefix", "vyv-", "--rate-max", "5"]) == 3
        assert f"guard pid {other.pid}" in capsys.readouterr().err
        assert research_main(["pods", "guard", "status", "--prefix", "vyv-"]) == 0
        out = capsys.readouterr().out
        assert f"alive (pid {other.pid})" in out and "rate $3.00/h kept ($7.00/h listed) (max $4.5/h)" in out and "terminated 2026-09-23T20:00:00Z c vyv-c" in out
        assert research_main(["pods", "guard", "stop", "--prefix", "vyv-"]) == 0
        assert other.wait(timeout=10) == -15
    finally:
        other.kill()
    assert research_main(["pods", "guard", "stop", "--prefix", "vyv-"]) == 1


def test_reset_clears_a_sticky_trip_keeps_the_tally_logs_who_and_refuses_while_the_guard_runs(guard_dir, capsys):
    """The vLLM coordinator hand-edited state.tripped after a lapsed deadline (2026-09-26); `reset` does it with a record."""
    api = fleet()
    t = [T0]
    G.run(G.Config(prefix="vyv-", deadline=T0 + 60, self_pod="me"), api, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + 120),
          iterations=2)
    before = json.loads((guard_dir / "guard-vyv-.json").read_text())
    assert "DEADLINE" in before["tripped"]["why"] and before["spent"] > 0

    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "guard"])
    try:
        (guard_dir / "guard-vyv-.pid").write_text(f"{other.pid}\n")
        assert research_main(["pods", "guard", "reset", "--prefix", "vyv-", "--by", "vllm-coordinator"]) == 3
        assert "stop --prefix vyv-" in capsys.readouterr().err
        assert json.loads((guard_dir / "guard-vyv-.json").read_text())["tripped"] == before["tripped"]
    finally:
        other.kill()
        other.wait(timeout=10)

    assert research_main(["pods", "guard", "reset", "--prefix", "vyv-", "--by", "vllm-coordinator"]) == 0
    after = json.loads((guard_dir / "guard-vyv-.json").read_text())
    assert after["tripped"] is None and after["spent"] == before["spent"] and after["terminated"] == before["terminated"]
    assert after["resets"][-1]["by"] == "vllm-coordinator" and after["resets"][-1]["cleared"] == before["tripped"]
    assert "RESET by vllm-coordinator: cleared the trip" in (guard_dir / "guard-vyv-.log").read_text()
    assert research_main(["pods", "guard", "reset", "--prefix", "vyv-"]) == 0 and "not tripped" in capsys.readouterr().out
    assert research_main(["pods", "guard", "reset", "--prefix", "none-"]) == 2

    api.add(pod("new", "vyv-new", 1.0, -1))                                  # a restart with a new deadline no longer kills new pods
    G.run(G.Config(prefix="vyv-", deadline=T0 + 10 * 3600, self_pod="me"), api, clock=lambda: t[0], sleep=lambda s: None, iterations=1)
    assert "new" not in api.deleted and json.loads((guard_dir / "guard-vyv-.json").read_text())["spent"] >= before["spent"]

