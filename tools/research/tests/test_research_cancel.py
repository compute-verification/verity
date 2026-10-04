"""`research cancel`: a remote run stops through SIGTERM to its workload's process group, recorded as a manual cancel, never by
signalling its runner. Cancels that signalled the runner and then SIGKILLed it during its custody upload left 15 node 1 runs without
custody (1-2 Oct)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from research import cli, remote

TOOL = Path(remote.__file__).resolve().parents[1]


def _run(tmp_path: Path, monkeypatch, state: str = "running"):
    """A launched run on a 'machine' reached through `sh -c`: its workload and a stand-in runner, each `cat` on a pipe the test holds,
    in process groups of their own."""
    workload, runner = (subprocess.Popen(["cat"], stdin=subprocess.PIPE, start_new_session=True) for _ in range(2))
    d = tmp_path / "machine" / "r-x"
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": state, "pid": workload.pid, "pgid": workload.pid}))
    (d / "events.jsonl").write_text("")
    local = tmp_path / "local" / "r-x"
    local.mkdir(parents=True)
    (local / "remote.json").write_text(json.dumps({"run_id": "r-x", "phase": "launched", "host": "h", "port": 22, "remote_run_dir": str(d),
                                                   "tool": str(TOOL)}))
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **k: ["sh", "-c"])
    return local, d, workload, runner


def _end(*procs: subprocess.Popen) -> None:
    for p in procs:
        p.stdin.close()
        p.wait()


def test_cancel_terms_the_workloads_group_records_a_manual_cancel_and_leaves_the_runner(tmp_path, monkeypatch, capsys):
    local, d, workload, runner = _run(tmp_path, monkeypatch)
    try:
        assert cli.main(["cancel", str(local), "--reason", "train superseded", "--by", "lander", "--runs-dir", str(local.parent)]) == 0
        assert workload.wait() == -signal.SIGTERM and runner.poll() is None
        intent, result = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines()][-2:]
        assert intent["kind"] == "cancel_intent" and intent["actor"] == "manual" and intent["signal"] == signal.SIGTERM
        assert intent["target"]["pgid"] == workload.pid and intent["reason"] == "train superseded (research cancel by lander)"
        assert result["kind"] == "cancel_result" and result["delivered"] is True
        assert f"process group {workload.pid} (1 process(es))" in capsys.readouterr().out
    finally:
        _end(workload, runner)


def test_a_cancel_is_reported_delivered_when_the_runner_records_the_exit_right_after_it(tmp_path, monkeypatch, capsys):
    """The cancel read only the last line of events.jsonl, which by then was the runner's record of the workload's exit, so runs it
    stopped (exit 143) were reported as "the cancel did not run on the machine (rc 0)"."""
    local, d, workload, runner = _run(tmp_path, monkeypatch)
    (bin_ := tmp_path / "bin").mkdir()
    (bin_ / "python3").write_text(f'#!/bin/sh\n"{sys.executable}" "$@"; rc=$?\n'
                                  f"echo '{json.dumps({'kind': 'proc_end', 'rc': 143})}' >> {d}/events.jsonl\nexit $rc\n")
    (bin_ / "python3").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    try:
        assert cli.main(["cancel", str(local), "--reason", "x", "--by", "lander", "--runs-dir", str(local.parent)]) == 0
        assert workload.wait() == -signal.SIGTERM and runner.poll() is None
        assert json.loads((d / "events.jsonl").read_text().splitlines()[-1])["kind"] == "proc_end"
        assert f"SIGTERM to its workload's process group {workload.pid} (1 process(es))" in capsys.readouterr().out
    finally:
        _end(workload, runner)


def test_a_run_that_is_not_running_is_refused_and_nothing_is_signalled(tmp_path, monkeypatch, capsys):
    local, d, workload, runner = _run(tmp_path, monkeypatch, state="done")
    try:
        assert cli.main(["cancel", str(local), "--reason", "x", "--runs-dir", str(local.parent)]) == 1
        assert "r-x is done, so there is nothing to cancel" in capsys.readouterr().err
        assert (d / "events.jsonl").read_text() == "" and workload.poll() is None
    finally:
        _end(workload, runner)
