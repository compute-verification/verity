"""Standing campaign targets (pods/nebius/keeper.py, `vy-keeper` on the Nebius nodes): each tick tops a target up to its n live
chunks, a fresh seed each, until its deadline, outside booked windows and the daily hold, pausing after failures in a row; a stop
starts nothing more and cancels only the target's own chunks. The node's systemd is a fake (units started and ended by the test)
and the clock is injected; the chunks' requests, run dirs and the keeper's files are under tmp_path."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
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
GPU_LEASE_SH = Path(registry.__file__).resolve().parent / "sh" / "gpu_lease.sh"
needs_flock = pytest.mark.skipif(shutil.which("flock") is None or shutil.which("bash") is None, reason="gpu-lease needs bash and flock(1)")


def t(s: str) -> float:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


class Systemd:
    """The research user's systemd as the keeper sees it: units it started, live until the test ends them."""

    def __init__(self) -> None:
        self.units: dict[str, bool | None] = {}
        self.started: list[tuple[str, dict]] = []
        self.refuse: str | None = None
        self.cancelled: list[str] = []
        self.cancel_refuses: str | None = None

    def launch(self, unit: str, request: Path, description: str) -> str | None:
        if self.refuse:
            return self.refuse
        self.units[unit] = True
        self.started.append((unit, json.loads(request.read_text())))
        return None

    def unit_live(self, unit: str) -> bool | None:
        return self.units.get(unit, False)

    def cancel_chunk(self, chunk: dict, reason: str) -> str | None:
        """As cancel-intent does: the manual cancel recorded in the run's events.jsonl (the unit ends when the test says)."""
        if self.cancel_refuses:
            return self.cancel_refuses
        rd = Path(chunk["run_dir"])
        rd.mkdir(parents=True, exist_ok=True)
        with open(rd / "events.jsonl", "a") as f:
            f.write(json.dumps({"kind": "cancel_intent", "actor": "manual", "reason": reason}) + "\n")
        self.cancelled.append(rd.name)
        return None


@pytest.fixture
def node(tmp_path, monkeypatch):
    """A fresh load of vy-keeper (as a restarted tick would be) on vy-nebius-2, its clock at 2026-10-04T00:00Z, nothing booked in
    its windows file and no lease in its gpu-lease dir."""
    sd = Systemd()
    now = [t("2026-10-04T00:00")]
    (tmp_path / "windows").write_text("")

    def load():
        spec = importlib.util.spec_from_file_location(f"keeper_{len(loaded)}", KEEPER)
        k = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(k)
        monkeypatch.setattr(k, "DIR", tmp_path / "keep")
        monkeypatch.setattr(k, "ROOT", tmp_path / "research")
        monkeypatch.setattr(k, "NODE", "vy-nebius-2")
        monkeypatch.setattr(k, "GPU_LEASE", ["/bin/gpu-lease"])
        monkeypatch.setattr(k, "LEASE_DIR", tmp_path / "lease")
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
    doc = {"owner": "@proofs", "node": "vy-nebius-2", "n": 2, "until": "2026-10-05T00:00Z", "max_min": 30,
           "command": ["python3", "-m", "x", "--seed", "{seed}"],
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


def run_of(unit: str) -> str:
    return unit.removeprefix("keeper-")


def grant(k, unit: str, since: str, gpu: int = 0) -> None:
    """gpu-lease takes a GPU for the chunk at `since`: its owner line, with a decoy run= and since= in the command it records."""
    k.LEASE_DIR.mkdir(parents=True, exist_ok=True)
    (k.LEASE_DIR / f"{gpu}.owner").write_text(f"pid=4242 run={run_of(unit)} who=keeper:@proofs uuid=GPU-{gpu} since={since} preempt=1 "
                                              "scope=gpu-lease-4242 cmd=python3 -m x run=r-other since=2026-10-01T00:00:00Z\n")


def preempt(k, unit: str) -> None:
    """A waiter takes the chunk's GPU back: gpu-lease's line in preempted.log."""
    k.LEASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(k.LEASE_DIR / "preempted.log", "a") as f:
        f.write(f"2026-10-04T00:30:00Z SIGTERM GPU 0 for pid 99 (n=8, who=timed): pid=4242 run={run_of(unit)} who=keeper:@proofs "
                "since=2026-10-04T00:00:00Z preempt=1 cmd=python3 -m x\n")


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
    assert req["command"] == ["/bin/gpu-lease", "1", "--wait", "--preemptible", "--max-min", "25", "--", "/usr/bin/env", "OMP_NUM_THREADS=8",
                              "python3", "-m", "x", "--seed", "1"]
    assert req["run_id"] == unit.removeprefix("keeper-") and req["runs_root"] == str(k.ROOT / "runs")
    assert req["remote"]["keeper"] == {"target": "sweep", "owner": "@proofs", "seed": 1, "node": "vy-nebius-2", "unit": unit}
    assert req["remote"]["source_sha"] == "sha" and req["harness_args"] == ["--stage", "run", "--project", "verity"]
    assert req["env"] == {"GPU_LEASE_WHO": "keeper:@proofs", "RESEARCH_KEEPER_TARGET": "sweep", "RESEARCH_KEEPER_SEED": "1"}
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


def test_past_until_nothing_starts_a_chunk_granted_before_it_finishes_and_the_rest_are_cancelled(node):
    k, sd, now, _ = node
    k.set_target("sweep", target(k, n=4, until="2026-10-04T06:00Z", max_min=600))
    k.tick()
    done, ran, late, waits = (u for u, _ in sd.started)
    end(k, sd, done)
    grant(k, ran, "2026-10-04T05:00:00Z")
    grant(k, late, "2026-10-04T06:00:30Z", gpu=1)
    now[0] = t("2026-10-04T06:01")
    k.tick()
    k.tick()
    assert len(sd.started) == 4 and sd.units[ran] is True and sorted(sd.cancelled) == sorted([run_of(late), run_of(waits)])
    assert [e["why"] for e in events(k, "idle")] == ["past its until 2026-10-04T06:00Z"]
    assert {e["run"]: e["why"] for e in events(k, "cancel")} == {
        run_of(late): "granted at 2026-10-04T06:00:30Z, at or after its until 2026-10-04T06:00:00Z",
        run_of(waits): "still waiting for its lease, at or after its until 2026-10-04T06:00:00Z"}


def test_nothing_starts_in_a_booked_window_or_where_its_max_min_would_reach_one(node, tmp_path):
    k, sd, now, _ = node
    (tmp_path / "windows").write_text("# start UTC, minutes\n2026-10-04T01:00Z 30 # served window, bc-x\n")
    k.set_target("sweep", target(k, n=1, max_min=30))
    now[0] = t("2026-10-04T00:45")                # a 30-minute chunk would run into the window at 01:00
    k.tick()
    now[0] = t("2026-10-04T01:10")
    k.tick()
    assert sd.started == []
    assert [e["why"] for e in events(k, "idle")] == ["a lease to 2026-10-04T01:15:00Z would reach the window at 2026-10-04T01:00:00Z",
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
    (tmp_path / "windows").unlink()
    (tmp_path / "windows").mkdir()                # there, but not a file it can read: fail closed
    now[0] = t("2026-10-05T14:00")
    k.tick()
    assert sd.started == [] and "the holds can't be read" in events(k, "idle")[-1]["why"]
    (tmp_path / "windows").rmdir()
    (tmp_path / "windows").write_text("")
    k.tick()
    assert seeds(sd) == [0]


def test_failures_in_a_row_pause_the_target_until_it_is_set_again(node):
    k, sd, now, _ = node
    k.set_target("sweep", target(k, n=1))
    k.tick()
    end(k, sd, sd.started[-1][0], "failed", 1)
    k.tick()
    preempt(k, sd.started[-1][0])
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
                      ({"max_min": None}, "max_min is missing"), ({"gpus": 9}, "gpus 9 is not a whole number from 1 to 8"),
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
    workload = subprocess.Popen(["cat"], stdin=subprocess.PIPE, start_new_session=True, env={**os.environ, "RESEARCH_RUN_DIR": str(rd.resolve())})
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


def test_research_keeper_set_status_and_stop_reach_the_nodes_keeper_over_ssh(tmp_path, monkeypatch, capsys):
    """The CLI from a VM: `set` ships the tool snapshot and writes the target through the node's vy-keeper (here the repository's,
    reached through `sh -c` for ssh), `status` reads it back, `stop` stops it. The node's clock is real, so its 48 h bound is
    lifted and the target's until is far ahead; the CLI's own check runs on an injected clock."""
    from verity_ops import cli
    from verity_ops import keeper as K

    root = tmp_path / "research"
    (tmp_path / "machines.toml").write_text(f'[machines.vy-nebius-2]\nprovider = "ssh"\nhost = "n2"\nuser = "research"\nproject = "verity"\n'
                                            f'root = "{root}"\npython = "{sys.executable}"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **k: ["sh", "-c"])
    pkg = tmp_path / "pkg" / "research"            # a stand-in for the research package `set` ships, which would read all of it
    pkg.mkdir(parents=True)
    for name, text in (("__init__.py", ""), ("cli.py", ""), ("remote.py", "def tool_identity(): ...\n")):
        (pkg / name).write_text(text)
    monkeypatch.setattr(remote, "source_tool_package", lambda src: pkg)
    monkeypatch.setattr(K, "NODE_KEEPER", str(KEEPER))
    monkeypatch.setattr(K, "clock", lambda: t("2098-12-31T12:00"))
    for k, v in (("KEEPER_DIR", tmp_path / "keep"), ("KEEPER_ROOT", root), ("KEEPER_MAX_AHEAD_H", 10**6)):
        monkeypatch.setenv(k, str(v))
    monkeypatch.delenv("KEEPER_NODE", raising=False)      # as over ssh: only vy-keeper.service sets it
    on = ["--on", "vy-nebius-2"]
    assert cli.main(["keeper", "set", "sweep", *on, "--project", "verity", "--n", "2", "--until", "2099-01-01T00:00Z", "--owner", "@proofs",
                     "--max-min", "25", "--lane", "proofs", "--declared-output", "out/*", "--", "python3", "-m", "x", "--seed", "{seed}"]) == 0
    assert "sweep on vy-nebius-2: 2 chunk(s) of `python3 -m x --seed '{seed}'`" in capsys.readouterr().out
    doc = json.loads((tmp_path / "keep" / "targets" / "sweep.json").read_text())
    req = doc["request"]
    assert (doc["owner"], doc["n"], doc["max_min"], doc["command"]) == ("@proofs", 2, 25, ["python3", "-m", "x", "--seed", "{seed}"])
    assert req["runs_root"] == str(root / "runs") and (Path(req["tool"]) / "research" / "cli.py").is_file()
    assert req["harness_args"] == ["--stage", "run", "--project", "verity", "--lane", "proofs", "--declared-output", "out/*"]
    assert req["remote"]["machine"] == "vy-nebius-2" and req["remote"]["tool_identity"] == remote.tool_identity(pkg)
    assert cli.main(["keeper", "status", *on, "--json"]) == 0
    row, = json.loads(capsys.readouterr().out)
    assert (row["name"], row["owner"], row["next_seed"], row["live"], row["stopped"]) == ("sweep", "@proofs", 0, [], None)
    assert cli.main(["keeper", "stop", "sweep", *on, "--by", "@proofs", "--reason", "campaign done"]) == 0
    assert "sweep on vy-nebius-2 stopped: no more chunks start (0 live)" in capsys.readouterr().out
    s = ["keeper", "set", "sweep", *on, "--project", "verity", "--n", "2", "--max-min", "25"]
    assert cli.main([*s, "--until", "2099-01-01T00:00Z", "--owner", "@circuits", "--", "x", "{seed}"]) == 1
    assert "vy-nebius-2 refuses it: target sweep is @proofs's" in capsys.readouterr().err
    assert cli.main([*s, "--until", "2099-01-03T00:00Z", "--owner", "@proofs", "--", "x", "{seed}"]) == 2
    assert "is not in the next 48 h" in capsys.readouterr().err
    assert cli.main([*s, "--until", "2099-01-01T00:00Z", "--owner", "@proofs", "--", "x", "1"]) == 2
    for flags, why in ((["--env", "GPU_LEASE_TIMED=1"], "--env GPU_LEASE_TIMED: the chunks' lease"),
                       (["--env", "CUDA_DEVICE_ORDER=FASTEST_FIRST"], "--env CUDA_DEVICE_ORDER: the chunks' lease"),
                       (["--env", "BASH_FUNC_timeout%%=() { :; }"], "--env BASH_FUNC_timeout%%: not a variable name"),
                       (["--env", "HF_TOKEN=hf_x"], "--env HF_TOKEN: a secret's name, and a chunk's env is set on its command line"),
                       (["--attempt-dir", str(tmp_path / "elsewhere")], "--attempt-dir: the runner sets each chunk's run dir"),
                       (["--job=j"], "--job: the runner sets")):
        assert cli.main([*s, "--until", "2099-01-01T00:00Z", "--owner", "@proofs", *flags, "--", "x", "{seed}"]) == 2
        assert why in capsys.readouterr().err
    with pytest.raises(SystemExit):               # a chunk without max_min would run through every hold it is granted into
        cli.main(["keeper", "set", "sweep", *on, "--project", "verity", "--n", "2", "--until", "2099-01-01T00:00Z", "--owner", "@proofs",
                  "--", "x", "{seed}"])
    assert "--max-min" in capsys.readouterr().err


def test_vy_keeper_set_reads_the_target_on_stdin_and_status_reports_it(tmp_path):
    env = {"KEEPER_DIR": str(tmp_path / "keep"), "KEEPER_ROOT": str(tmp_path / "research"), "KEEPER_NODE": "vy-nebius-2", "PATH": "/usr/bin:/bin"}
    doc = target(SimpleNamespace(ROOT=tmp_path / "research"), until="2099-01-01T00:00Z")
    r = subprocess.run([sys.executable, str(KEEPER), "set", "sweep"], input=json.dumps(doc), env=env, capture_output=True, text=True)
    assert r.returncode == 3 and "more than 48 h" in json.loads(r.stdout)["refused"]
    r = subprocess.run([sys.executable, str(KEEPER), "status", "--json"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout) == []
    r = subprocess.run([sys.executable, str(KEEPER), "status", "../state/sweep"], env=env, capture_output=True, text=True)
    assert r.returncode == 3 and "is not a target name" in json.loads(r.stdout)["refused"]


def test_a_chunk_still_waiting_is_cancelled_once_a_grant_then_would_reach_a_hold_and_only_once(node, tmp_path):
    """max_min counts from the lease's grant, so a chunk queued at 00:00 with max_min 30 that waited to 00:50 would run into a
    01:00 window: it is cancelled the first tick a grant would reach the window, and nothing takes its place."""
    k, sd, now, _ = node
    (tmp_path / "windows").write_text("2026-10-04T01:00Z 30 # served window\n")
    k.set_target("sweep", target(k, n=1, max_min=30))
    k.tick()
    unit = sd.started[0][0]
    now[0] = t("2026-10-04T00:30")
    k.tick()
    assert sd.cancelled == []
    sd.cancel_refuses = "no running workload to signal (state None)"
    now[0] = t("2026-10-04T00:31")
    k.tick()
    k.tick()
    sd.cancel_refuses = None
    k.tick()
    now[0] = t("2026-10-04T01:00")
    k.tick()
    assert sd.cancelled == [run_of(unit)] and len(sd.started) == 1
    assert [(e["why"], e["result"]) for e in events(k, "cancel")] == [
        ("still waiting for its lease: a lease to 2026-10-04T01:01:00Z would reach the window at 2026-10-04T01:00:00Z", r)
        for r in ("no running workload to signal (state None)", "SIGTERM delivered")]
    end(k, sd, unit, "cancelled", 143)
    k.tick()
    assert [e["outcome"] for e in events(k, "exit")] == ["cancelled"] and k.status("sweep")[0]["failures"] == 0


def test_a_chunk_granted_into_a_hold_is_cancelled_and_one_whose_lease_ends_before_it_is_left(node, tmp_path):
    """A window booked after the chunks were granted still binds them: the one whose lease reaches it is cancelled at the next
    tick, the one whose max_min ends before it runs on."""
    k, sd, now, _ = node
    k.set_target("sweep", target(k, n=2, max_min=30))
    k.tick()
    early, late = (u for u, _ in sd.started)
    grant(k, early, "2026-10-04T00:20:00Z", gpu=0)
    grant(k, late, "2026-10-04T00:40:00Z", gpu=1)
    now[0] = t("2026-10-04T00:41")
    (tmp_path / "windows").write_text("2026-10-04T01:00Z 30 # booked late\n")
    k.tick()
    assert sd.cancelled == [run_of(late)]
    assert events(k, "cancel")[0]["why"] == "granted at 2026-10-04T00:40:00Z: a lease to 2026-10-04T01:10:00Z would reach the window at " \
                                            "2026-10-04T01:00:00Z"


def test_holds_that_cant_be_read_cancel_nothing_and_start_nothing_but_until_still_binds(node, tmp_path):
    k, sd, now, _ = node
    k.set_target("sweep", target(k, n=2, until="2026-10-04T06:00Z"))
    k.tick()
    (tmp_path / "windows").unlink()               # a node that names a windows file holds without it
    now[0] = t("2026-10-04T01:00")
    end(k, sd, sd.started[0][0])
    k.tick()
    assert len(sd.started) == 2 and sd.cancelled == [] and "the holds can't be read" in events(k, "idle")[-1]["why"]
    (tmp_path / "windows").write_text("# start UTC, minutes\n\n2026-10-04T02:00Z 30min # a typo in its minutes\n")
    k.tick()
    assert len(sd.started) == 2 and "'2026-10-04T02:00Z 30min # a typo in its minutes' is not a booking" in events(k, "idle")[-1]["why"]
    now[0] = t("2026-10-04T06:00")
    k.tick()
    assert sd.cancelled == [run_of(sd.started[1][0])]


def test_an_exit_143_frees_the_place_only_with_a_cancel_or_a_preemption_on_record(node):
    """A workload whose own shell exits 143 is a failure like any other, so it pauses the target instead of restarting forever."""
    k, sd, now, _ = node
    k.set_target("sweep", target(k, n=1))
    k.tick()
    preempt(k, sd.started[-1][0])
    end(k, sd, sd.started[-1][0], "failed", 143)
    k.tick()
    sd.cancel_chunk({"run_dir": str(k.ROOT / "runs" / run_of(sd.started[-1][0]))}, "by hand")
    end(k, sd, sd.started[-1][0], "cancelled", 143)
    k.tick()
    for _ in range(3):
        preempt(k, sd.started[-1][0] + "0")       # another run, whose id begins with this one's
        end(k, sd, sd.started[-1][0], "failed", 143)
        k.tick()
    assert [e["outcome"] for e in events(k, "exit")] == ["preempted", "cancelled", "failed", "failed", "failed"]
    assert len(sd.started) == 5 and k.status("sweep")[0]["paused"].startswith("3 failures in a row, the last: seed 4")


@pytest.mark.parametrize("over,why", [
    ({"env": {"GPU_LEASE_TIMED": "1"}}, "env sets GPU_LEASE_TIMED: the lease"), ({"env": {"CUDA_VISIBLE_DEVICES": "0"}}, "sets CUDA_VISIBLE_DEVICES"),
    ({"env": {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}}, "sets CUDA_DEVICE_ORDER"), ({"env": {"RESEARCH_KEEPER_SEED": "0"}}, "sets RESEARCH_KEEPER_SEED"),
    ({"env": {"BASH_FUNC_timeout%%": "() { :; }"}}, "env has 'BASH_FUNC_timeout%%', not variable names"),
    ({remote.QUEUE_WORKLOAD_WRAP: ["env", "GPU_LEASE_TIMED=1"]}, "request has workload_wrap"),
    ({remote.QUEUE_WRAP: ["env", "GPU_LEASE_TIMED=1"]}, "request has wrap"), ({"exclusive": True}, "request is exclusive"),
    ({"harness_args": ["--stage", "run", "--attempt-dir", "/tmp/elsewhere"]}, "give --attempt-dir: the runner"),
    ({"harness_args": ["--attempt-dir=/tmp/elsewhere"]}, "give --attempt-dir: the runner"),
    ({"harness_args": ["--job", "j", "--id", "r-x", "--runs-dir", "/tmp"]}, "give --id, --job, --runs-dir: the runner"),
    ({"harness_args": ["--on", "vy-nebius-1"]}, "give --on"), ({"harness_args": ["--env=GPU_LEASE_TIMED=1"]}, "give --env"),
    ({"harness_args": ["--stage", "run", "--", "python3", "-m", "x"]}, "hold `--`: what follows it would be the workload"),
    ({"harness_args": ["--project", "verity", "--project", "other"]}, "give --project more than once")])
def test_a_targets_request_cannot_reach_the_lease_nor_give_the_runners_own_flags(node, over, why):
    """gpu-lease reads its knobs from the environment (GPU_LEASE_TIMED=1 would make every chunk a timed window to node_ops), a
    wrap would put something around the lease, argparse keeps the last of a flag (--attempt-dir moves status.json out of
    runs/<rid>, where no hold, until or stop reaches the chunk) and a `--` makes what follows it the workload, outside the lease.
    `set` refuses each, and a tick starts nothing for a target written by hand."""
    k, sd, now, _ = node
    doc = target(k, n=1, request={**target(k)["request"], **over})
    with pytest.raises(ValueError, match=re.escape(why)):
        k.set_target("sweep", doc)
    (k.DIR / "targets").mkdir(parents=True, exist_ok=True)
    (k.DIR / "targets" / "sweep.json").write_text(json.dumps({**doc, "schema": k.SCHEMA, "name": "sweep", "set_utc": "2026-10-04T00:00Z"}))
    k.tick()
    assert sd.started == [] and why in events(k, "skipped")[0]["why"]


@needs_flock
def test_a_targets_env_is_set_inside_the_lease_where_bash_env_cannot_make_it_timed(node, tmp_path, monkeypatch):
    """gpu-lease is `#!/usr/bin/env bash`: a BASH_ENV in its environment runs before its first line and can export any of its knobs
    (GPU_LEASE_TIMED=1 here, or the since= that the holds' cancels read). Set inside the lease, it reaches the workload alone."""
    k, sd, now, _ = node
    (tmp_path / "be.sh").write_text("export GPU_LEASE_TIMED=1\n")
    req = {**target(k)["request"], "env": {"BASH_ENV": str(tmp_path / "be.sh"), "PYTHONPATH": "/mine"}}
    k.set_target("sweep", target(k, n=1, request=req, command=["sh", "-c", 'echo "inside $BASH_ENV"; sed "s/ cmd=.*//" "$GPU_LEASE_DIR"/0.owner # {seed}']))
    monkeypatch.setattr(k, "GPU_LEASE", ["bash", str(GPU_LEASE_SH)])
    k.tick()
    _, chunk = sd.started[0]
    assert chunk["env"] == {"PYTHONPATH": "/mine", "GPU_LEASE_WHO": "keeper:@proofs", "RESEARCH_KEEPER_TARGET": "sweep", "RESEARCH_KEEPER_SEED": "0"}
    env = {key: v for key, v in os.environ.items() if not key.startswith(("GPU_LEASE_", "RESEARCH_RUN", "BASH_ENV"))}
    env.update(GPU_LEASE_DIR=str(tmp_path / "gl"), GPU_LEASE_COUNT="1", GPU_LEASE_UUID_TABLE="GPU-a0", RESEARCH_RUN_ID="r-x",
               GPU_LEASE_USAGE_LOG=str(tmp_path / "u.jsonl"), GPU_LEASE_UTIL_LOG=str(tmp_path / "util-*.jsonl"),
               GPU_LEASE_ALLOWED_FILE=str(tmp_path / "absent-direct-gpus"), GPU_LEASE_WINDOWS=str(tmp_path / "absent-windows"),
               **chunk["env"])       # the node's own direct-gpus and windows files would decide the lease (node 1: none)
    r = subprocess.run(chunk["command"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    owner = next(x for x in r.stdout.splitlines() if x.startswith("pid="))
    assert f"inside {tmp_path / 'be.sh'}" in r.stdout.splitlines() and "who=keeper:@proofs" in owner and " timed=1" not in owner


@pytest.mark.parametrize("unit", ["vy-keeper-n1.service", "vy-keeper-n2.service"])
def test_the_lease_wrapper_names_its_programs_by_path_and_a_targets_path_cannot_replace_them(node, tmp_path, monkeypatch, unit):
    """The runner looks its workload's argv[0] up in its own PATH (node 1's wrapper starts with env), as do gpu-lease's shebang
    and its flock and timeout calls; a target's PATH, set inside the lease, decides none of them."""
    k, sd, now, _ = node
    line, = [x for x in (KEEPER.parent / unit).read_text().splitlines() if "KEEPER_GPU_LEASE=" in x]
    wrapper = shlex.split(shlex.split(line.removeprefix("Environment="))[0].partition("=")[2])
    progs = [a for a in wrapper if "=" not in a]
    assert progs[-1].endswith("/gpu-lease") and all(os.path.isabs(p) for p in progs), wrapper
    fake = tmp_path / "bin"
    fake.mkdir()
    for prog in ("env", "bash", "flock", "timeout", "gpu-lease"):
        (fake / prog).write_text(f"#!/bin/sh\ntouch {tmp_path / 'replaced'}\n")
        (fake / prog).chmod(0o755)
    (tmp_path / "gpu-lease").write_text(f"#!/bin/sh\ntouch {tmp_path / 'leased'}\n")
    (tmp_path / "gpu-lease").chmod(0o755)
    monkeypatch.setattr(k, "GPU_LEASE", [str(tmp_path / "gpu-lease") if a == progs[-1] else a for a in wrapper])
    k.set_target("sweep", target(k, n=1, request={**target(k)["request"], "env": {"PATH": f"{fake}:{os.environ['PATH']}"}}, command=["true", "{seed}"]))
    k.tick()
    _, chunk = sd.started[0]
    assert "PATH" not in chunk["env"] and f"PATH={fake}:{os.environ['PATH']}" in chunk["command"]
    subprocess.run(chunk["command"], env={**os.environ, **chunk["env"]}, check=True)
    assert (tmp_path / "leased").exists() and not (tmp_path / "replaced").exists()


def test_a_chunks_runner_sees_only_the_keepers_env_and_pythonpath_and_runs_the_lease_as_its_workload(node, monkeypatch):
    """launch-request applies the request's env to the runner, which hands it on to gpu-lease: only the keeper's keys and
    PYTHONPATH (behind the tool snapshot, as with --on) go there, and the workload after `--` is the keeper's lease."""
    k, sd, now, _ = node
    req = {**target(k)["request"], "env": {"BASH_ENV": "/x", "PATH": "/evil", "PYTHONPATH": "/mine"}, "remote": {"machine": "vy-nebius-2"}}
    k.set_target("sweep", target(k, n=1, request=req))
    k.tick()
    seen = []

    class Proc:
        pid = 4242

    monkeypatch.setattr(remote.subprocess, "Popen", lambda argv, **kw: seen.append((argv, kw["env"])) or Proc())
    monkeypatch.setattr(remote, "await_first_record", lambda proc, fd: remote.FIRST_RECORD)     # the fake runner never acknowledges
    assert remote._launch_request(k.ROOT / "requests" / run_of(sd.started[0][0]) / "request.json")["started"]
    (argv, env), = seen
    assert argv[:4] == ["python3", "-m", "research", "run"]
    assert argv[argv.index("--") + 1:] == ["/bin/gpu-lease", "1", "--wait", "--preemptible", "--max-min", "30", "--", "/usr/bin/env",
                                           "BASH_ENV=/x", "PATH=/evil", "python3", "-m", "x", "--seed", "0"]
    assert env.get("BASH_ENV") != "/x" and env["PATH"] != "/evil" and env["PYTHONPATH"] == f"{k.ROOT / 'tool' / 'abc'}:/mine"
    assert (env["GPU_LEASE_WHO"], env["RESEARCH_KEEPER_SEED"]) == ("keeper:@proofs", "0")


def test_node_2_keeps_a_chunk_off_all_8_gpus(node, monkeypatch):
    """node_ops reads one holder of every lock on node 2 as a timed window and pauses research's CPU-heavy work for it."""
    k, sd, now, _ = node
    assert "Environment=KEEPER_MAX_GPUS=7" in (KEEPER.parent / "vy-keeper-n2.service").read_text().splitlines()
    monkeypatch.setattr(k, "MAX_GPUS", 7)
    with pytest.raises(ValueError, match="gpus 8 is not a whole number from 1 to 7"):
        k.set_target("sweep", target(k, gpus=8))
    k.set_target("sweep", target(k, n=1, gpus=7))
    k.tick()
    assert sd.started[0][1]["command"][:2] == ["/bin/gpu-lease", "7"]


def test_cancel_chunk_refuses_a_stale_pgid_that_is_now_another_groups(tmp_path):
    """A runner that died leaves status.json `running`; the group number it names may be another job's by now."""
    spec = importlib.util.spec_from_file_location("keeper_stale", KEEPER)
    k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k)
    k.ROOT = tmp_path / "research"
    rd = k.ROOT / "runs" / "r20261004-000000-abcd"
    rd.mkdir(parents=True)
    other = subprocess.Popen(["cat"], stdin=subprocess.PIPE, start_new_session=True,
                             env={**os.environ, "RESEARCH_RUN_DIR": str(tmp_path / "research" / "runs" / "r-another")})
    try:
        (rd / "status.json").write_text(json.dumps({"state": "running", "pid": other.pid, "pgid": other.pid}))
        (rd / "events.jsonl").write_text("")
        req = k.ROOT / "requests" / rd.name / "request.json"
        req.parent.mkdir(parents=True)
        req.write_text(json.dumps({"tool": str(TOOL), "python": sys.executable}))
        assert k.cancel_chunk({"run_dir": str(rd)}, "campaign done").startswith(f"refused: no process of group {other.pid} is this run's")
        assert other.poll() is None and (rd / "events.jsonl").read_text() == ""
    finally:
        other.stdin.close()
        other.wait()


def test_stop_and_status_take_only_a_target_name(node):
    k, sd, now, _ = node
    k.set_target("sweep", target(k, n=1))
    for call in (lambda: k.stop_target("../state/sweep", "@x", "traversal", cancel=False), lambda: k.status("../state/sweep")):
        with pytest.raises(ValueError, match="is not a target name"):
            call()
    assert [e["event"] for e in events(k)] == ["set"]


def test_a_chunk_launched_just_before_a_crash_is_reaped_with_its_outcome(node):
    """The tick died after systemd-run started the unit and before it saved `launched`: the run dir says the chunk ran."""
    k, sd, now, load = node
    k.set_target("sweep", target(k, n=1))
    real = k.launch

    def launch_then_die(unit, request, description):
        real(unit, request, description)
        raise KeyboardInterrupt

    k.launch = launch_then_die
    with pytest.raises(KeyboardInterrupt):
        k.tick()
    k2 = load()
    end(k2, sd, sd.started[0][0], "failed", 1)
    k2.tick()
    assert [e["outcome"] for e in events(k2, "exit")] == ["failed"] and events(k2, "abandoned") == []
    assert k2.status("sweep")[0]["failures"] == 1
