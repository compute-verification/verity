"""Standing campaign targets (pods/nebius/keeper.py, `vy-keeper` on the Nebius nodes): each tick tops a target up to its n live
chunks, a fresh seed each, until its deadline, outside booked windows and the daily hold, pausing after failures in a row; a stop
starts nothing more and cancels only the target's own chunks. The node's systemd is a fake (units started and ended by the test)
and the clock is injected; the chunks' requests, run dirs and the keeper's files are under tmp_path."""
from __future__ import annotations

import importlib.util
import json
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from research import remote
from research.pods import registry

KEEPER = Path(registry.__file__).resolve().parent / "nebius" / "keeper.py"
TOOL = Path(remote.__file__).resolve().parents[1]


def t(s: str) -> float:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


class Systemd:
    """The research user's systemd as the keeper sees it: units it started, live until the test ends them."""

    def __init__(self) -> None:
        self.units: dict[str, bool | None] = {}
        self.started: list[tuple[str, dict]] = []
        self.refuse: str | None = None
        self.cancelled: list[str] = []

    def launch(self, unit: str, request: Path, description: str) -> str | None:
        if self.refuse:
            return self.refuse
        self.units[unit] = True
        self.started.append((unit, json.loads(request.read_text())))
        return None

    def unit_live(self, unit: str) -> bool | None:
        return self.units.get(unit, False)

    def cancel_chunk(self, chunk: dict, reason: str) -> str | None:
        self.cancelled.append(Path(chunk["run_dir"]).name)
        return None


@pytest.fixture
def node(tmp_path, monkeypatch):
    """A fresh load of vy-keeper (as a restarted tick would be) on vy-nebius-2, its clock at 2026-10-04T00:00Z."""
    sd = Systemd()
    now = [t("2026-10-04T00:00")]

    def load():
        spec = importlib.util.spec_from_file_location(f"keeper_{len(loaded)}", KEEPER)
        k = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(k)
        monkeypatch.setattr(k, "DIR", tmp_path / "keep")
        monkeypatch.setattr(k, "ROOT", tmp_path / "research")
        monkeypatch.setattr(k, "NODE", "vy-nebius-2")
        monkeypatch.setattr(k, "GPU_LEASE", ["/bin/gpu-lease"])
        monkeypatch.setattr(k, "WINDOWS", str(tmp_path / "windows"))
        monkeypatch.setattr(k, "DAILY_HOLD", "")
        monkeypatch.setattr(k, "clock", lambda: now[0])
        for seam in ("launch", "unit_live", "cancel_chunk"):
            monkeypatch.setattr(k, seam, getattr(sd, seam))
        loaded.append(k)
        return k

    loaded: list = []
    k = load()
    return k, sd, now, load


def target(k, **over) -> dict:
    root = k.ROOT
    doc = {"owner": "@proofs", "node": "vy-nebius-2", "n": 2, "until": "2026-10-05T00:00Z", "command": ["python3", "-m", "x", "--seed", "{seed}"],
           "request": {"schema": "research/request/v0.1", "runs_root": str(root / "runs"), "tool": str(root / "tool" / "abc"), "python": "python3",
                       "cwd": str(root / "src" / "sha"), "env": {"OMP_NUM_THREADS": "8"}, "harness_args": ["--stage", "run", "--project", "verity"],
                       "remote": {"machine": "vy-nebius-2", "source_sha": "sha"}}}
    return {**doc, **over}


def end(k, sd: Systemd, unit: str, state: str | None = "done", rc: int | None = 0) -> None:
    """The chunk's runner exits (with that status, or none: it died) and its unit ends."""
    rd = k.ROOT / "runs" / unit.removeprefix("keeper-")
    rd.mkdir(parents=True, exist_ok=True)
    if state:
        (rd / "status.json").write_text(json.dumps({"state": state, "rc": rc}))
    sd.units[unit] = False


def events(k, event: str | None = None) -> list[dict]:
    p = k.DIR / "events.jsonl"
    lines = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    return [e for e in lines if event is None or e["event"] == event]


def seeds(sd: Systemd) -> list[int]:
    return [req["remote"]["keeper"]["seed"] for _, req in sd.started]


def test_a_tick_tops_the_target_up_to_n_and_a_waiting_chunk_counts_live(node):
    k, sd, now, _ = node
    k.set_target("sweep", target(k, n=3, max_min=25))
    k.tick()
    assert seeds(sd) == [0, 1, 2]
    unit, req = sd.started[1]
    assert req["command"] == ["/bin/gpu-lease", "1", "--wait", "--preemptible", "--max-min", "25", "--", "python3", "-m", "x", "--seed", "1"]
    assert req["run_id"] == unit.removeprefix("keeper-") and req["runs_root"] == str(k.ROOT / "runs")
    assert req["remote"]["keeper"] == {"target": "sweep", "owner": "@proofs", "seed": 1, "node": "vy-nebius-2", "unit": unit}
    assert req["remote"]["source_sha"] == "sha" and req["harness_args"] == ["--stage", "run", "--project", "verity"]
    assert req["env"] == {"OMP_NUM_THREADS": "8", "GPU_LEASE_WHO": "keeper:@proofs", "RESEARCH_KEEPER_TARGET": "sweep",
                          "RESEARCH_KEEPER_SEED": "1"}
    assert (k.ROOT / "requests" / req["run_id"] / "request.json").is_file()
    k.tick()                                      # all three still running or waiting for their lease
    assert len(sd.started) == 3
    end(k, sd, sd.started[0][0])
    k.tick()
    assert seeds(sd) == [0, 1, 2, 3]
    assert [(e["seed"], e["outcome"]) for e in events(k, "exit")] == [(0, "done")]
    st = k.status("sweep")[0]
    assert st["next_seed"] == 4 and sorted(c["seed"] for c in st["live"]) == [1, 2, 3] and st["failures"] == 0


def test_the_seed_counter_only_rises_across_restarts_a_crash_and_a_re_set(node):
    k, sd, now, load = node
    k.set_target("sweep", target(k, n=2))
    k.tick()
    k2 = load()                                   # the next tick is a fresh process: everything it knows is on disk
    for unit, _ in sd.started:
        end(k2, sd, unit, "failed", 143)
    k2.set_target("sweep", target(k2, n=2, seed_from=0))
    sd.started.clear()

    def crash(unit, request, description):        # the tick dies after taking the seed, before the unit starts
        raise KeyboardInterrupt

    k2.launch = crash
    with pytest.raises(KeyboardInterrupt):
        k2.tick()
    k3 = load()
    k3.tick()
    assert seeds(sd) == [3, 4]                    # 2 was taken by the crashed tick and is never handed out again
    assert [e["seed"] for e in events(k3, "abandoned")] == [2]
    k3.set_target("sweep", target(k3, n=2, seed_from=10))
    for unit, _ in sd.started:
        end(k3, sd, unit)
    k3.tick()
    assert seeds(sd)[-2:] == [10, 11]


def test_nothing_starts_past_until_and_live_chunks_are_left_to_finish(node):
    k, sd, now, _ = node
    k.set_target("sweep", target(k, n=2, until="2026-10-04T06:00Z"))
    k.tick()
    end(k, sd, sd.started[0][0])
    now[0] = t("2026-10-04T06:00")
    k.tick()
    k.tick()
    assert len(sd.started) == 2 and sd.units[sd.started[1][0]] is True and not sd.cancelled
    assert [e["why"] for e in events(k, "idle")] == ["past its until 2026-10-04T06:00Z"]


def test_nothing_starts_in_a_booked_window_or_where_its_max_min_would_reach_one(node, tmp_path):
    k, sd, now, _ = node
    (tmp_path / "windows").write_text("# start UTC, minutes\n2026-10-04T01:00Z 30 # served window, bc-x\n")
    k.set_target("sweep", target(k, n=1, max_min=30))
    now[0] = t("2026-10-04T00:45")                # a 30-minute chunk would run into the window at 01:00
    k.tick()
    now[0] = t("2026-10-04T01:10")
    k.tick()
    assert sd.started == []
    assert [e["why"] for e in events(k, "idle")] == ["a chunk of max_min 30 would reach the window at 2026-10-04T01:00:00Z",
                                                    "in the window 2026-10-04T01:00:00Z-2026-10-04T01:30:00Z"]
    now[0] = t("2026-10-04T01:30")
    k.tick()
    assert seeds(sd) == [0]


def test_nothing_starts_in_node_1s_daily_hold_and_an_unreadable_windows_file_holds_everything(node, tmp_path, monkeypatch):
    k, sd, now, _ = node
    monkeypatch.setattr(k, "DAILY_HOLD", "12:10-13:30")
    k.set_target("sweep", target(k, n=1, max_min=90, until="2026-10-05T20:00Z"))
    for hhmm in ("2026-10-04T12:30", "2026-10-04T11:00", "2026-10-05T12:09"):   # in the hold; 90 min reach it; so does the next day's
        now[0] = t(hhmm)
        k.tick()
    assert sd.started == []
    (tmp_path / "windows").mkdir()                # there, but not a file it can read: fail closed
    now[0] = t("2026-10-05T14:00")
    k.tick()
    assert sd.started == [] and "the holds can't be read" in events(k, "idle")[-1]["why"]
    (tmp_path / "windows").rmdir()
    k.tick()
    assert seeds(sd) == [0]


def test_failures_in_a_row_pause_the_target_until_it_is_set_again(node):
    k, sd, now, _ = node
    k.set_target("sweep", target(k, n=1))
    k.tick()
    end(k, sd, sd.started[-1][0], "failed", 1)
    k.tick()
    end(k, sd, sd.started[-1][0], "failed", 143)  # preempted: frees the place, counts nothing
    k.tick()
    end(k, sd, sd.started[-1][0], "done", 0)      # a success clears the count
    k.tick()
    end(k, sd, sd.started[-1][0], "failed", 124)  # its max_min
    k.tick()
    end(k, sd, sd.started[-1][0], state=None)     # the runner died without a final status
    k.tick()
    (k.ROOT / "runs" / sd.started[-1][0].removeprefix("keeper-")).mkdir(parents=True)
    (k.ROOT / "runs" / sd.started[-1][0].removeprefix("keeper-") / "refused.json").write_text(json.dumps({"reason": "SourceNotReady"}))
    sd.units[sd.started[-1][0]] = False
    k.tick()
    assert [e["outcome"] for e in events(k, "exit")] == ["failed", "preempted", "done", "max_min", "lost", "refused"]
    assert len(sd.started) == 6 and k.status("sweep")[0]["paused"].startswith("3 failures in a row, the last: seed 5")
    k.tick()
    assert len(sd.started) == 6 and len(events(k, "paused")) == 1
    k.set_target("sweep", target(k, n=1))
    k.tick()
    assert seeds(sd)[-1] == 6 and k.status("sweep")[0]["failures"] == 0


def test_a_failed_launch_counts_a_failure_and_starts_nothing_more_that_tick(node):
    k, sd, now, _ = node
    k.set_target("sweep", target(k, n=4))
    sd.refuse = "Failed to connect to bus"
    for _ in range(3):
        k.tick()
    assert sd.started == [] and [e["seed"] for e in events(k, "launch-failed")] == [0, 1, 2]
    assert k.status("sweep")[0]["paused"] and k.status("sweep")[0]["live"] == []


def test_a_malformed_target_is_skipped_and_logged_once_and_the_others_still_run(node):
    k, sd, now, _ = node
    k.set_target("good", target(k, n=1))
    (k.DIR / "targets" / "garbled.json").write_text("{not json")
    other = {**target(k), "schema": k.SCHEMA, "name": "elsewhere", "set_utc": "2026-10-04T00:00Z", "node": "vy-nebius-1"}
    (k.DIR / "targets" / "elsewhere.json").write_text(json.dumps(other))
    long = {**target(k), "schema": k.SCHEMA, "name": "long", "set_utc": "2026-10-04T00:00Z", "until": "2026-10-06T00:01Z"}
    (k.DIR / "targets" / "long.json").write_text(json.dumps(long))
    k.tick()
    k.tick()
    assert [req["remote"]["keeper"]["target"] for _, req in sd.started] == ["good"]
    skipped = {e["target"]: e["why"] for e in events(k, "skipped")}
    assert len(events(k, "skipped")) == 3
    assert skipped["garbled"].startswith("the target can't be used: Expecting")
    assert "is not this node (vy-nebius-2)" in skipped["elsewhere"] and "more than 48 h after it was set" in skipped["long"]


def test_set_refuses_a_bad_target_and_another_owners_name(node):
    k, sd, now, _ = node
    for over, why in (({"until": "2026-10-06T00:01Z"}, "more than 48 h"), ({"until": "2026-10-03T23:00Z"}, "has passed"),
                      ({"until": "2026-10-05T00:00"}, "not in UTC"), ({"command": ["x", "--seed", "1"]}, "no {seed}"),
                      ({"n": 0}, "n 0 is not a whole number"), ({"node": "vy-nebius-1"}, "not this node"),
                      ({"request": {**target(k)["request"], "tool": "/tmp/tool"}}, "not a snapshot under")):
        with pytest.raises(ValueError, match=re.escape(why)):
            k.set_target("sweep", target(k, **over))
    k.set_target("sweep", target(k))
    with pytest.raises(ValueError, match="is @proofs's"):
        k.set_target("sweep", target(k, owner="@circuits"))
    assert [e["event"] for e in events(k)] == ["set"]


def test_stop_starts_nothing_more_and_cancel_signals_only_the_targets_own_chunks(node):
    k, sd, now, _ = node
    k.set_target("sweep", target(k, n=2))
    k.set_target("other", target(k, n=1, owner="@circuits"))
    k.tick()
    mine = sorted(u.removeprefix("keeper-") for u, req in sd.started if req["remote"]["keeper"]["target"] == "sweep")
    out = k.stop_target("sweep", "@proofs", "campaign done", cancel=False)
    assert out["live"] == mine and out["cancels"] == {} and sd.cancelled == []
    end(k, sd, f"keeper-{mine[0]}")
    k.tick()
    assert len(sd.started) == 3
    out = k.stop_target("sweep", "@proofs", "campaign done", cancel=True)
    assert sd.cancelled == [mine[1]] and out["cancels"] == {mine[1]: "SIGTERM delivered"}
    assert [e["event"] for e in events(k) if e["target"] == "sweep"][-3:] == ["idle", "stop", "cancel"]
    k.set_target("sweep", target(k, n=2))         # set again: re-armed, the counter where it was
    end(k, sd, f"keeper-{mine[1]}", "failed", 143)
    k.tick()
    assert sorted(seeds(sd))[-2:] == [2, 3]
    with pytest.raises(ValueError, match="no target nope"):
        k.stop_target("nope", "@proofs", "x", cancel=False)


def test_a_unit_systemd_cant_report_on_counts_live(node):
    k, sd, now, _ = node
    k.set_target("sweep", target(k, n=1))
    k.tick()
    sd.units[sd.started[0][0]] = None
    k.tick()
    assert len(sd.started) == 1 and k.status("sweep")[0]["live"][0]["seed"] == 0


def test_cancel_chunk_terms_the_workloads_group_as_a_manual_cancel(tmp_path):
    spec = importlib.util.spec_from_file_location("keeper_cancel", KEEPER)
    k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k)
    k.ROOT = tmp_path / "research"
    rd = k.ROOT / "runs" / "r20261004-000000-abcd"
    rd.mkdir(parents=True)
    workload = subprocess.Popen(["cat"], stdin=subprocess.PIPE, start_new_session=True)
    try:
        (rd / "status.json").write_text(json.dumps({"state": "running", "pid": workload.pid, "pgid": workload.pid}))
        (rd / "events.jsonl").write_text("")
        req = k.ROOT / "requests" / rd.name / "request.json"
        req.parent.mkdir(parents=True)
        req.write_text(json.dumps({"tool": str(TOOL), "python": sys.executable}))
        assert k.cancel_chunk({"run_dir": str(rd)}, "campaign done (vy-keeper stop sweep by @proofs)") is None
        assert workload.wait() == -signal.SIGTERM
        intent = [json.loads(x) for x in (rd / "events.jsonl").read_text().splitlines()][-2]
        assert intent["actor"] == "manual" and intent["reason"] == "campaign done (vy-keeper stop sweep by @proofs)"
        assert k.cancel_chunk({"run_dir": str(k.ROOT / "runs" / "r-none")}, "x") == "no running workload to signal (state None)"
    finally:
        workload.stdin.close()
        workload.wait()


def test_vy_keeper_set_reads_the_target_on_stdin_and_status_reports_it(tmp_path):
    env = {"KEEPER_DIR": str(tmp_path / "keep"), "KEEPER_ROOT": str(tmp_path / "research"), "KEEPER_NODE": "vy-nebius-2", "PATH": "/usr/bin:/bin"}
    doc = target(SimpleNamespace(ROOT=tmp_path / "research"), until="2099-01-01T00:00Z")
    r = subprocess.run([sys.executable, str(KEEPER), "set", "sweep"], input=json.dumps(doc), env=env, capture_output=True, text=True)
    assert r.returncode == 3 and "more than 48 h" in json.loads(r.stdout)["refused"]
    r = subprocess.run([sys.executable, str(KEEPER), "status", "--json"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout) == []
