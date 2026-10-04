"""`research cancel`: a remote run stops through SIGTERM to its workload's process group, recorded as a manual cancel, never by
signalling its runner. Cancels that signalled the runner and then SIGKILLed it during its custody upload left 15 node 1 runs without
custody (1-2 Oct). It works from any VM: a run another VM launched is found on its machine (`remote.adopt`)."""
from __future__ import annotations

import json
import signal
import subprocess
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


RUN = "r20261003-101500-c0de"


def _launched_elsewhere(tmp_path: Path, monkeypatch, *, tool: bool = True):
    """A running run another VM launched on ssh machine vy-b (a directory reached over `sh -c`, as in test_remote_adopt): no local run
    dir, and the shipped tool named only where the runner records it, job.json `run.remote.tool`."""
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    workload = subprocess.Popen(["cat"], stdin=subprocess.PIPE, start_new_session=True)
    d = tmp_path / "b" / "runs" / RUN
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": "running", "pid": workload.pid, "pgid": workload.pid}))
    (d / "job.json").write_text(json.dumps({"run": {"remote": {"machine": "vy-b", "tool": str(TOOL) if tool else None}}}))
    (d / "events.jsonl").write_text("")
    ms = {name: {"provider": "ssh", "host": name, "user": "research", "root": str(tmp_path / name[-1]), "project": "verity"}
          for name in ("vy-a", "vy-b")}
    monkeypatch.setattr(remote, "load_machines", lambda path=None: ms)
    monkeypatch.setattr(remote, "pod_target", lambda m: {"host": m["host"], "port": 22, "user": m["user"]})
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **kw: ["sh", "-c"])
    return tmp_path / "here", d, workload


def test_cancel_from_any_vm_finds_the_run_on_its_machine_and_records_the_cancel_there(tmp_path, monkeypatch, capsys):
    here, d, workload = _launched_elsewhere(tmp_path, monkeypatch)
    try:
        assert cli.main(["cancel", RUN, "--on", "vy-a", "--reason", "x", "--runs-dir", str(here)]) == 1
        assert f"vy-a: no {tmp_path / 'a' / 'runs' / RUN}" in capsys.readouterr().err
        assert workload.poll() is None and not (here / RUN).exists(), "--on asks only the machine it names"

        assert cli.main(["cancel", RUN, "--reason", "superseded", "--by", "lander", "--runs-dir", str(here)]) == 0
        assert workload.wait() == -signal.SIGTERM
        rec = json.loads((here / RUN / "remote.json").read_text())
        assert (rec["machine"], rec["phase"], rec.get("tool")) == ("vy-b", "adopted", None), "the tool came from the machine's job.json"
        intent, result = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines()][-2:]
        assert intent["actor"] == "manual" and intent["reason"] == "superseded (research cancel by lander)"
        assert result["kind"] == "cancel_result" and result["delivered"] is True
        assert f"process group {workload.pid} (1 process(es))" in capsys.readouterr().out
    finally:
        _end(workload)


def test_a_run_adopted_earlier_is_cancelled_from_its_local_dir(tmp_path, monkeypatch, capsys):
    here, d, workload = _launched_elsewhere(tmp_path, monkeypatch)
    try:
        assert cli.main(["status", RUN, "--runs-dir", str(here)]) == 0
        assert cli.main(["cancel", RUN[:12], "--reason", "x", "--runs-dir", str(here)]) == 0, "a prefix resolves to the adopted dir"
        assert workload.wait() == -signal.SIGTERM
    finally:
        _end(workload)


def test_a_run_whose_records_name_no_tool_is_refused_and_nothing_is_signalled(tmp_path, monkeypatch, capsys):
    here, d, workload = _launched_elsewhere(tmp_path, monkeypatch, tool=False)
    try:
        assert cli.main(["cancel", RUN, "--on", "vy-b", "--reason", "x", "--runs-dir", str(here)]) == 1
        assert "nor its job.json names a shipped tool" in capsys.readouterr().err
        assert (d / "events.jsonl").read_text() == "" and workload.poll() is None
    finally:
        _end(workload)


def test_a_run_that_is_not_running_is_refused_and_nothing_is_signalled(tmp_path, monkeypatch, capsys):
    local, d, workload, runner = _run(tmp_path, monkeypatch, state="done")
    try:
        assert cli.main(["cancel", str(local), "--reason", "x", "--runs-dir", str(local.parent)]) == 1
        assert "r-x is done, so there is nothing to cancel" in capsys.readouterr().err
        assert (d / "events.jsonl").read_text() == "" and workload.poll() is None
    finally:
        _end(workload, runner)
