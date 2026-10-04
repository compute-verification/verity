"""`research run --on` says a run was launched only once the machine's runner has parsed its arguments and written its first record
(job.json): `launch-request` waits for the runner's acknowledgement on a FIFO beside the request, or for the runner's exit. A runner
that exits first is a refusal (`RunnerFailed`, exit 3) carrying its launcher.log's tail. The launch runs locally: ssh is `sh -c`
and the machine a directory."""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
from pathlib import Path

import pytest

from research import remote
from research import runs as R
from research.cli import main
from waits import Gate, wait_exit, wait_runner

PY = sys.executable
SRC = str(Path(remote.__file__).resolve().parents[1])


@pytest.fixture
def machine(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "machine"
    (tmp_path / "machines.toml").write_text(f'[machines.m]\nprovider = "runpod"\npod_id = "p1"\nproject = "p"\nroot = "{root}"\npython = "{PY}"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setattr(remote, "pod_target", lambda m: {"host": "h", "port": 22, "pod_name": "m", "cost_per_hr": 0.1,
                                                         "rate_observed_utc": "t0", "gpu": None})
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **k: ["sh", "-c"])
    return root


def _run(tmp_path: Path, *extra: str, command: tuple[str, ...] = (PY, "-c", "print('ran')")) -> int:
    return main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "m", "--project", "p", "--no-custody-r2", "--no-sampler",
                 *extra, "--", *command])


def test_a_runner_that_fails_at_argument_parsing_is_refused_with_its_error_not_launched(tmp_path, machine, capsys):
    rc = _run(tmp_path, "--no-such-harness-flag")
    err = capsys.readouterr().err
    assert rc == 3 and " launched on " not in err
    assert "refused on m (RunnerFailed)" in err and "unrecognized arguments: --no-such-harness-flag" in err, "the runner's own error"
    [rid] = [p.name for p in (tmp_path / "runs").iterdir()]
    run_dir = machine / "runs" / rid
    refused = json.loads((run_dir / "refused.json").read_text())
    assert refused["fault"] == "RunnerFailed" and refused["rc"] == 2 and not (run_dir / "job.json").exists()
    assert R.summarize(run_dir)["state"] == "refused"
    assert json.loads((tmp_path / "runs" / rid / "remote.json").read_text())["phase"] == "refused"
    assert not (machine / "requests" / rid / remote.LAUNCH_ACK).exists(), "the FIFO goes once the launch is settled"


def test_launched_is_said_only_after_the_runner_wrote_job_json(tmp_path, machine, capsys):
    gate = Gate(tmp_path / "gate")
    try:
        assert _run(tmp_path, command=(PY, "-c", Gate.WAIT, str(gate.path))) == 0
        err = capsys.readouterr().err
        [rid] = [p.name for p in (tmp_path / "runs").iterdir()]
        run_dir = machine / "runs" / rid
        assert f"run {rid} launched on m" in err
        assert (run_dir / "job.json").exists(), "the first record exists by the time the launcher says launched"
        assert R.summarize(run_dir)["state"] not in R.TERMINAL, "and the run is still going: the launch did not wait for it"
        gate.release()
        wait_runner(run_dir)
    finally:
        gate.close()
    assert R.summarize(run_dir)["state"] == "done"


def _request(tmp_path: Path, run_id: str, command: list[str], harness: tuple[str, ...] = ()) -> tuple[Path, Path]:
    root = tmp_path / "root"
    (root / "requests" / run_id).mkdir(parents=True)
    req = {"schema": remote.REQUEST_SCHEMA, "run_id": run_id, "runs_root": str(root / "runs"), "tool": SRC, "python": PY, "cwd": None,
           "env": {}, "harness_args": ["--stage", "t", "--no-sampler", *harness], "command": command, "exclusive": False,
           "cost": {}, "remote": {"machine": "m", "pod_id": "p1"}}
    p = root / "requests" / run_id / "request.json"
    p.write_text(json.dumps(req))
    return p, root / "runs"


def test_launch_request_waits_for_the_runners_first_record_even_behind_a_wrapper(tmp_path):
    """A wrapper (the queue's CPU scope) that starts the runner only once the test lets it: launch-request is still waiting when the
    wrapper reaches its gate, and returns once the runner, started through a fork the ack fd could not have crossed, wrote job.json."""
    at_gate, go, work = Gate(tmp_path / "at-gate"), Gate(tmp_path / "go"), Gate(tmp_path / "work")
    wrap = [PY, "-c", "import subprocess, sys; open(sys.argv[1], 'w').write('at\\n'); open(sys.argv[2], 'rb').read(1); "
            "sys.exit(subprocess.call(sys.argv[3:]))", str(at_gate.path), str(go.path)]
    req, runs_root = _request(tmp_path, "w1", [PY, "-c", Gate.WAIT, str(work.path)])
    doc = json.loads(req.read_text())
    doc[remote.QUEUE_WRAP] = wrap
    req.write_text(json.dumps(doc))
    got: dict = {}
    t = threading.Thread(target=lambda: got.update(remote.launch_request(req)))
    try:
        t.start()
        assert at_gate.readline() == b"at\n"
        assert t.is_alive() and not got, "no answer while the runner has not started"
        go.release()
        t.join()
        assert got["started"] and got["first_record"] == "job.json" and (runs_root / "w1" / "job.json").exists()
        work.release()
        wait_runner(runs_root / "w1")
    finally:
        for g in (at_gate, go, work):
            g.close()
    assert R.summarize(runs_root / "w1")["state"] == "done"


def test_a_wrapper_that_exits_before_the_runner_is_refused_with_its_output(tmp_path):
    req, runs_root = _request(tmp_path, "c1", [PY, "-c", "print('never')"])
    doc = json.loads(req.read_text())
    doc[remote.QUEUE_WRAP] = ["sh", "-c", "echo 'cluster: scope does not hold CPUs 0-3' >&2; exit 70", "cpuset-check"]
    req.write_text(json.dumps(doc))
    out = remote.launch_request(req)
    assert out["started"] is False and out["fault"] == "RunnerFailed" and "exited 70" in out["refused"]
    assert "cluster: scope does not hold CPUs 0-3" in out["refused"]
    again = remote.launch_request(req)                       # re-delivered after an ssh drop: the same refusal
    assert again["existing"] and again["fault"] == "RunnerFailed" and again["refused"] == out["refused"]


def test_a_run_held_for_disk_ahead_of_its_runner_counts_as_launched(tmp_path):
    """disk-hold runs before the runner and may hold for hours: it acknowledges the launch when it starts holding, so the launch
    stays detached; the runner's job.json comes once the disk frees."""
    req, runs_root = _request(tmp_path, "h1", [PY, "-c", "print('never')"])
    doc = json.loads(req.read_text())
    doc["queue"] = {"disk_gb": 10**12}
    req.write_text(json.dumps(doc))
    out = remote.launch_request(req)
    try:
        assert out["started"] and out["first_record"] == "disk-hold" and not (runs_root / "h1" / "job.json").exists()
        assert "research disk-hold: holding" in (runs_root / "h1" / "launcher.log").read_text()
    finally:
        os.killpg(out["runner_pid"], signal.SIGTERM)
        wait_exit(out["runner_pid"])


def test_the_ack_is_a_no_op_with_no_one_waiting(tmp_path):
    remote.ack_launch(tmp_path / "absent", "job.json")
    (tmp_path / "plain").write_text("")
    remote.ack_launch(tmp_path / "plain", "job.json")
    assert (tmp_path / "plain").read_text() == "", "only a FIFO is written"
    os.mkfifo(tmp_path / "fifo")
    remote.ack_launch(tmp_path / "fifo", "job.json")             # no reader: ENXIO, ignored, never blocks
