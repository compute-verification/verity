"""`research pods guard` (pods/guard.py), the one guard over budgets.toml, against a fake RunPod API: what is in scope, per-line caps
that hold until raised and spare pods that appear later, the balance floor, each line's max pod age, uncovered project pods, a
removed line, never the clock, the last good budgets kept, dry runs, and the CLI."""
from __future__ import annotations

import datetime as dt
import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from research.cli import main as research_main
from research.pods import budgets as B
from research.pods import guard as G

T0 = G.parse_time("2026-09-23T20:00:00Z")
TOML = """
[guard]
project = ["vy"]
balance_floor = 5
uncovered_grace_min = 15
exempt = ["vy-control-"]

[budgets]
"vy-a-" = { cap_usd = 3, expires = "2026-09-24T20:00Z", max_pod_hours = 4, by = "test" }
"vy-a-big-" = { cap_usd = 100, expires = "2026-09-24T20:00Z", max_pod_hours = 48 }
"""


def pod(pid: str, name: str, rate: float, age_min: float, status: str = "RUNNING") -> dict:
    t = dt.datetime.fromtimestamp(T0 - age_min * 60, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " +0000 UTC"
    return {"id": pid, "name": f"{name}-veritor-campaign", "costPerHr": rate, "createdAt": t, "lastStartedAt": t, "desiredStatus": status}


class FakeRunPod:
    """GET /pods, DELETE /pods/<id>, clientBalance.  `lag`: a DELETEd pod is still listed RUNNING by the next list (RunPod does this)."""

    def __init__(self, pods: list[dict], balance: float | Exception = 100.0, lag: bool = False):
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


@pytest.fixture(autouse=True)
def guard_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_GUARD_DIR", str(tmp_path / "guard"))
    return tmp_path / "guard"


def cfg(tmp_path: Path | None = None, **kw) -> G.Config:
    return G.Config(source=(tmp_path or Path("/nonexistent")) / "budgets.toml", self_pod=kw.pop("self_pod", "me"), **kw)


def step(api, budgets, state, t, lines, c=None):
    return G.poll(api, budgets if isinstance(budgets, B.Budgets) else B.parse(budgets), c or cfg(), state, t, lines.append)


def test_scope_is_the_lines_and_the_project_minus_exempt_names_the_host_and_foreign_pods():
    api = FakeRunPod([pod("a", "vy-a-1", 1, 5), pod("g", "vy-a-big-1", 1, 5), pod("u", "vy-x", 1, 5), pod("c", "vy-control-verity", 1, 5),
                      pod("me", "vy-a-host", 1, 5), pod("f", "ligerito-h100", 9, 5), pod("d", "vy-a-done", 1, 5, "EXITED")])
    state, lines = G.new_state(T0), []
    step(api, TOML, state, T0, lines)
    step(api, TOML, state, T0 + 600, lines)
    assert set(state["pods"]) == {"a", "g", "u"} and state["rate_now"] == 3.0
    assert state["pods"]["a"]["line"] == "vy-a-" and state["pods"]["g"]["line"] == "vy-a-big-" and state["pods"]["u"]["line"] is None
    assert state["lines"]["vy-a-"]["spent"] == pytest.approx(1 / 6) and state["lines"]["vy-a-big-"]["spent"] == pytest.approx(1 / 6), \
        "the longest prefix takes the pod's spend"
    assert state["last_ok_poll"] == T0 + 600 and api.deleted == []


def test_a_line_at_its_cap_terminates_its_pods_and_holds_until_the_cap_is_raised():
    api = FakeRunPod([pod("a", "vy-a-1", 2, 5), pod("b", "vy-a-2", 2, 5), pod("g", "vy-a-big-1", 2, 5)])
    state, lines = G.new_state(T0), []
    step(api, TOML, state, T0, lines)
    step(api, TOML, state, T0 + 3600, lines)
    assert sorted(api.deleted) == ["a", "b"], "$4/h x 1 h >= $3: the line trips; the big line's pod is not its"
    assert "'vy-a-' CAP $3 reached (spent $4.00)" in state["lines"]["vy-a-"]["tripped"]["why"]
    api.add(pod("n", "vy-a-3", 1, -60))
    step(api, TOML, state, T0 + 3700, lines)
    assert "n" not in api.deleted and any("OVER-CAP n" in ln for ln in lines), "a pod that appears under a tripped line is not killed"
    raised = TOML.replace("cap_usd = 3,", "cap_usd = 10,")
    step(api, raised, state, T0 + 3760, lines)
    assert state["lines"]["vy-a-"]["tripped"] is None and any("UNTRIPPED 'vy-a-'" in ln for ln in lines)


def test_a_failed_delete_is_tried_again_and_a_lagging_listing_costs_no_second_delete():
    api = FakeRunPod([pod("a", "vy-a-1", 4, 5)], lag=True)
    real, fails = api.terminate, [1]

    def flaky(pid):
        if fails[0]:
            fails[0] -= 1
            raise RuntimeError("HTTP 502")
        real(pid)

    api.terminate = flaky
    state, lines = G.new_state(T0), []
    step(api, TOML, state, T0, lines)
    step(api, TOML, state, T0 + 3600, lines)
    assert api.deleted == [] and "terminate a vy-a-1-veritor-campaign failed: RuntimeError: HTTP 502" in "\n".join(lines)
    step(api, TOML, state, T0 + 3660, lines)
    step(api, TOML, state, T0 + 3720, lines)
    assert api.deleted == ["a"], "retried once, then left alone while the API still lists it"
    assert state["lines"]["vy-a-"]["spent"] == pytest.approx(4 + 4 / 60 * 2), "still billed while it goes"


def test_the_balance_floor_terminates_every_pod_in_scope_and_an_unreadable_balance_keeps_the_last_reading():
    api = FakeRunPod([pod("a", "vy-a-1", 0.1, 5), pod("u", "vy-x", 0.1, 5), pod("c", "vy-control-verity", 1, 5), pod("f", "other", 1, 5)],
                     balance=6.0)
    state, lines = G.new_state(T0), []
    step(api, TOML, state, T0, lines)
    api.bal = RuntimeError("graphql down")
    step(api, TOML, state, T0 + 60, lines)
    assert api.deleted == [] and any("judging the last reading, $6.00" in ln for ln in lines)
    api.bal = 4.0
    step(api, TOML, state, T0 + 120, lines)
    assert sorted(api.deleted) == ["a", "u"] and all("BALANCE $4.00 <= floor $5" in t["why"] for t in state["terminated"])
    api.bal = RuntimeError("graphql down")
    api.add(pod("n", "vy-a-2", 0.1, -2))
    step(api, TOML, state, T0 + 180, lines)
    assert "n" in api.deleted, "the last reading is under the floor: a failed read does not lift it"


def test_a_pod_older_than_its_lines_max_age_is_terminated_and_the_clock_never_trips_a_line():
    api = FakeRunPod([pod("old", "vy-a-1", 0.01, 5 * 60), pod("young", "vy-a-2", 0.01, 60), pod("big", "vy-a-big-1", 0.01, 5 * 60)])
    state, lines = G.new_state(T0), []
    step(api, TOML, state, T0, lines)
    assert api.deleted == ["old"] and "POD AGE" in state["terminated"][0]["why"] and "max_pod_hours 4" in state["terminated"][0]["why"]
    long_lived = TOML.replace("max_pod_hours = 4,", "max_pod_hours = 1000,").replace("max_pod_hours = 48 ", "max_pod_hours = 1000 ")
    step(api, long_lived, state, T0 + 2 * 86400, lines)
    assert api.deleted == ["old"], "the lines expired a day ago: that ends creates, never pods"


def test_an_uncovered_project_pod_goes_after_the_grace_and_only_if_the_file_says_so():
    api = FakeRunPod([pod("u", "vy-x", 1, 5)])
    state, lines = G.new_state(T0), []
    step(api, TOML, state, T0, lines)
    step(api, TOML, state, T0 + 14 * 60, lines)
    assert api.deleted == [] and sum("UNCOVERED u" in ln for ln in lines) == 1
    step(api, TOML, state, T0 + 15 * 60, lines)
    assert api.deleted == ["u"] and "UNCOVERED: no budget line covers" in state["terminated"][0]["why"]
    api, state, lines = FakeRunPod([pod("u", "vy-x", 1, 5)]), G.new_state(T0), []
    report_only = TOML.replace("[guard]\n", "[guard]\nterminate_uncovered = false\n")
    for t in (T0, T0 + 3600, T0 + 7200):
        step(api, report_only, state, t, lines)
    assert api.deleted == [] and sum("reported only" in ln for ln in lines) == 1


def test_a_removed_line_leaves_its_pods_uncovered_and_its_tally_kept():
    api = FakeRunPod([pod("g", "vy-a-big-1", 1, 5)])
    state, lines = G.new_state(T0), []
    step(api, TOML, state, T0, lines)
    step(api, TOML, state, T0 + 3600, lines)
    without = TOML.replace('"vy-a-big-" = { cap_usd = 100, expires = "2026-09-24T20:00Z", max_pod_hours = 48 }\n', "")
    step(api, without, state, T0 + 3660, lines)
    assert state["pods"]["g"]["line"] == "vy-a-", "the next-longest line covers it now"
    only_other = "[guard]\nproject = [\"vy\"]\n[budgets]\n\"vy-z-\" = { cap_usd = 1, expires = \"2026-09-24T20:00Z\", max_pod_hours = 1 }\n"
    step(api, only_other, state, T0 + 3720, lines)
    step(api, only_other, state, T0 + 3720 + 15 * 60, lines)
    assert api.deleted == ["g"] and state["lines"]["vy-a-big-"]["spent"] == pytest.approx(1.0)


def test_a_dry_run_calls_nothing_and_keeps_its_own_files(tmp_path, guard_dir):
    (tmp_path / "budgets.toml").write_text(TOML)
    api = FakeRunPod([pod("a", "vy-a-1", 4, 5)])
    t = [T0]
    assert G.run(cfg(tmp_path, dry_run=True), api, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + 3600), iterations=2) == 0
    assert api.deleted == [] and not (guard_dir / "guard-budgets.json").exists()
    state = json.loads((guard_dir / "guard-budgets.dry-run.json").read_text())
    assert state["terminated"][0]["dry_run"] and "WOULD-TERMINATE a" in (guard_dir / "guard-budgets.dry-run.log").read_text()


def test_the_last_good_budgets_stay_in_force_when_the_file_breaks_and_across_a_restart(tmp_path, guard_dir):
    f = tmp_path / "budgets.toml"
    f.write_text(TOML)
    api = FakeRunPod([pod("a", "vy-a-1", 8, 5)])
    t = [T0]
    clock, sleep = (lambda: t[0]), (lambda s: t.__setitem__(0, t[0] + 1800))
    assert G.run(cfg(tmp_path), api, clock=clock, sleep=sleep, iterations=1) == 0
    f.write_text(TOML.replace("cap_usd = 3,", "cap = 300,"))
    assert G.run(cfg(tmp_path), api, clock=clock, sleep=sleep, iterations=1) == 0
    state = json.loads((guard_dir / "guard-budgets.json").read_text())
    assert "cap not a line key" in state["budgets"]["error"] and state["budgets"]["lines"]["vy-a-"]["cap_usd"] == 3.0
    f.unlink()
    assert G.run(cfg(tmp_path), api, clock=clock, sleep=sleep, iterations=2) == 0
    assert api.deleted == ["a"], "a restart with no readable file enforces the budgets its state carries"
    log = (guard_dir / "guard-budgets.log").read_text()
    assert log.count("BUDGETS unreadable (BudgetError") == 1 and "BUDGETS unreadable (FileNotFoundError" in log


def test_a_guard_with_no_budgets_ever_read_enforces_nothing_and_says_it_is_not_ready(tmp_path, guard_dir):
    api = FakeRunPod([pod("a", "vy-a-1", 2, 5)])
    assert G.run(cfg(tmp_path), api, clock=lambda: T0, sleep=lambda s: None, iterations=2) == 0
    state = json.loads((guard_dir / "guard-budgets.json").read_text())
    assert state["last_ok_poll"] is None and "nothing is enforced yet" in (guard_dir / "guard-budgets.log").read_text()
    assert "never" in B.gate("vy-a-1", 1, state, T0), "`pods create` refuses under a guard that has no budgets"


def test_a_failed_list_is_logged_and_the_guard_keeps_polling(tmp_path, guard_dir):
    (tmp_path / "budgets.toml").write_text(TOML)
    api = FakeRunPod([pod("a", "vy-a-1", 1, 5)])
    api.fail_list = 1
    assert G.run(cfg(tmp_path), api, clock=lambda: T0, sleep=lambda s: None, iterations=2) == 0
    assert "poll failed: OSError: temporary failure" in (guard_dir / "guard-budgets.log").read_text()
    assert json.loads((guard_dir / "guard-budgets.json").read_text())["last_ok_poll"] == T0


def live_helper() -> subprocess.Popen:
    """A process whose command line names `guard`, returned once its Python runs.  `Popen` returns while the child is still inside
    execve, before the kernel records the new argument area, so /proc/<pid>/cmdline can read empty then (measured 2026-09-29: 40 %
    of spawns on an idle VM, 95 % under load); the child's own line on stdout comes after it is in place."""
    p = subprocess.Popen([sys.executable, "-c", "import os, signal; os.write(1, b'ready\\n'); signal.pause()", "guard"], stdout=subprocess.PIPE)
    assert p.stdout is not None and p.stdout.readline() == b"ready\n", "the helper did not start"
    return p


def one_fake_poll(monkeypatch) -> None:
    """The CLI's `run` becomes one poll against a fake API: whatever a test's pid check says, it cannot reach a real loop."""
    real = G.run
    monkeypatch.setattr(G, "run", lambda c, api=None, **kw: real(c, FakeRunPod([]), poll_s=kw.get("poll_s", 60.0), log_path=kw.get("log_path"),
                                                                 clock=lambda: T0, sleep=lambda s: pytest.fail("a second poll"), iterations=1))


def test_the_guard_refuses_to_start_without_readable_budgets_or_a_runpod_key(tmp_path, monkeypatch, capsys):
    from research.pods import runpod

    monkeypatch.setattr(runpod, "CONFIG", tmp_path / "no-runpod-config.toml")      # a laptop's ~/.runpod/config.toml is not this test's
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert research_main(["pods", "guard", "--budgets", str(tmp_path / "absent.toml")]) == 2
    assert "nothing to enforce" in capsys.readouterr().err
    (tmp_path / "budgets.toml").write_text(TOML)
    assert research_main(["pods", "guard", "--budgets", str(tmp_path / "budgets.toml")]) == G.EXIT_NO_KEY
    assert "RUNPOD_API_KEY is not set" in capsys.readouterr().err
    assert G.run(cfg(tmp_path), clock=lambda: T0, sleep=lambda s: pytest.fail("a guard without a key polled")) == G.EXIT_NO_KEY, \
        "run itself refuses before its first poll when it would call the real API"
    slept: list[float] = []
    rcs = [0, G.EXIT_NO_KEY]
    assert G.supervise(["guard"], 60.0, call=lambda c: rcs.pop(), sleep=slept.append, clock=lambda: T0) == 0 and slept == [60.0], \
        "supervised, a guard without a key is tried again a poll later"


def test_the_cli_refuses_a_second_guard_and_status_stop_read_the_recorded_pid(tmp_path, guard_dir, monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "not-a-real-key")
    one_fake_poll(monkeypatch)
    assert research_main(["pods", "guard", "status"]) == 2
    (tmp_path / "budgets.toml").write_text(TOML)
    G.run(cfg(tmp_path), FakeRunPod([pod("a", "vy-a-1", 1, 5)]), clock=lambda: T0, sleep=lambda s: None, iterations=1)
    other = live_helper()
    monkeypatch.setattr(G, "_alive", lambda pid: pid == other.pid and other.poll() is None)
    try:
        (guard_dir / "guard-budgets.pid").write_text(f"{other.pid}\n")
        assert research_main(["pods", "guard", "--budgets", str(tmp_path / "budgets.toml")]) == 3
        assert f"guard pid {other.pid}" in capsys.readouterr().err
        assert G.status(clock=lambda: T0 + 60) == 0
        out = capsys.readouterr().out
        assert f"alive (pid {other.pid}), last completed poll 1.0 min ago" in out and "'vy-a-': $0.00 of $3" in out and "(by test)" in out
        assert G.status(clock=lambda: T0 + 600) == 0 and "STALE: `pods create` refuses" in capsys.readouterr().out
        assert research_main(["pods", "guard", "stop"]) == 0
        assert other.wait() == -signal.SIGTERM
    finally:
        other.kill()
        other.stdout.close()
    assert research_main(["pods", "guard", "stop"]) == 1


