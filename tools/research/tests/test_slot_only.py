"""pods/nebius/slot_only.py: on a machine that takes runs only through its check slots, which commands count, and which live runs it
stops (cpu-1, 8 Oct: three runs outside slot.py beside the merge check, two from trees older than the launcher's refusal)."""
from __future__ import annotations

import json
import signal

from research.pods.nebius import slot_only as S


def test_only_a_command_that_runs_cmd_in_a_check_slot_counts():
    assert S.through_slot(["python3", "tools/verity/check/slot.py", "--", "uv", "run", "python", "tools/verity/check/check.py"])
    assert S.through_slot(["python3", "tools/verity/check/slot.py", "--priority", "--by", "x", "--", "true"])
    for c in (["python3", "tools/verity/check/slot.py", "--status"], ["python3", "tools/verity/check/slot.py", "--by", "x", "--"],
              ["uv", "run", "circuit-check", "--all"], ["bash", "-c", "python3 tools/verity/check/slot.py -- true"],
              ["python", "tools/verity/check/slot.py", "--", "true"], ["python3", "/elsewhere/slot.py", "--", "true"], None, "python3"):
        assert not S.through_slot(c), c


def _run(runs, rid, state, command, *, pgid=None, runner=None):
    d = runs / rid
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": state, "pgid": pgid}))
    (d / "job.json").write_text(json.dumps({"execution": {"command": command}}))
    if runner:
        (d / "launch.json").write_text(json.dumps({"runner_pid": runner}))
    return d


def _proc(proc, pid, pgid, run_dir=None):
    p = proc / str(pid)
    p.mkdir(parents=True)
    (p / "stat").write_text(f"{pid} (py thon) S 1 {pgid} {pgid} 0 -1\n")
    (p / "environ").write_bytes(b"PATH=/bin\0" + (f"RESEARCH_RUN_DIR={run_dir.resolve()}\0".encode() if run_dir else b""))


def test_it_stops_every_process_of_a_live_run_outside_the_slots_but_its_runner(tmp_path, monkeypatch, capsys):
    runs, proc, marker, log = tmp_path / "runs", tmp_path / "proc", tmp_path / "runs-via-slot", tmp_path / "slot-only.jsonl"
    slot = ["python3", "tools/verity/check/slot.py", "--", "true"]
    gate = _run(runs, "r-gate", "running", slot, pgid=10, runner=9)
    off = _run(runs, "r-off", "running", ["uv", "run", "circuit-check", "--all"], pgid=20, runner=19)
    _run(runs, "r-done", "done", ["true"], pgid=30)
    _proc(proc, 9, 9)
    _proc(proc, 10, 10, gate)
    _proc(proc, 19, 19, off)          # a runner never carries RESEARCH_RUN_DIR; were it to, it is still spared
    _proc(proc, 20, 20, off)
    _proc(proc, 21, 20, off)
    _proc(proc, 40, 40, off)          # left in another group (a lake build's setsid): stopped too
    _proc(proc, 30, 30)
    assert [o["run"] for o in S.offenders(runs)] == ["r-off"]
    sent = []
    monkeypatch.setattr(S.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(S, "stop", lambda o, dry_run=False, _stop=S.stop: _stop(o, proc=proc, kill=S.os.kill, dry_run=dry_run))
    assert S.main(["--runs", str(runs), "--marker", str(marker), "--log", str(log)]) == 0
    assert sent == [] and not log.exists(), "no marker: the machine takes any run"
    marker.write_text("")
    assert S.main(["--runs", str(runs), "--marker", str(marker), "--log", str(log), "--dry-run"]) == 0
    assert sent == [] and not log.exists() and '"r-off"' in capsys.readouterr().out
    assert S.main(["--runs", str(runs), "--marker", str(marker), "--log", str(log)]) == 0
    assert sent == [(20, signal.SIGTERM), (21, signal.SIGTERM), (40, signal.SIGTERM)]
    rec = json.loads(log.read_text())
    assert rec["run"] == "r-off" and rec["workload_group_seen"] and rec["sigterm"] == [20, 21, 40]
