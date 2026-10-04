"""`research run --on` says a run was launched only once the machine's runner has parsed its arguments and written its first record
(job.json): `launch-request` waits for the runner's acknowledgement on a FIFO beside the request, or for the runner's exit. A runner
that exits first is a refusal (`RunnerFailed`, exit 3) carrying its launcher.log's tail. The launch runs locally: ssh is `sh -c`
and the machine a directory."""
from __future__ import annotations

import errno
import json
import os
import shlex
import signal
import subprocess
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


def _request(tmp_path: Path, run_id: str, command: list[str], harness: tuple[str, ...] = (), **extra) -> tuple[Path, Path]:
    root = tmp_path / "root"
    (root / "requests" / run_id).mkdir(parents=True)
    req = {"schema": remote.REQUEST_SCHEMA, "run_id": run_id, "runs_root": str(root / "runs"), "tool": SRC, "python": PY, "cwd": None,
           "env": {}, "harness_args": ["--stage", "t", "--no-sampler", *harness], "command": command, "exclusive": False,
           "cost": {}, "remote": {"machine": "m", "pod_id": "p1"}, **extra}
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


def test_without_a_pidfd_the_launch_still_waits_for_the_first_record(tmp_path, monkeypatch):
    """A kernel or seccomp profile without pidfd_open: the runner's exit is watched through a pipe a thread closes, and the launch is
    answered as before."""
    def nope(pid, flags=0):
        raise OSError(errno.ENOSYS, "Function not implemented")

    monkeypatch.setattr(remote.os, "pidfd_open", nope)
    req, _ = _request(tmp_path, "pf2", [PY, "-c", "print('never')"], harness=("--no-such-harness-flag",))
    assert remote.launch_request(req)["fault"] == "RunnerFailed"
    work = Gate(tmp_path / "work")
    req, runs_root = _request(tmp_path, "pf1", [PY, "-c", Gate.WAIT, str(work.path)])
    try:
        out = remote.launch_request(req)
        assert out["started"] and out["first_record"] == "job.json"
    finally:
        monkeypatch.undo()                                   # wait_runner watches with a pidfd of its own
        work.release()
        wait_runner(runs_root / "pf1")
        work.close()


def test_an_error_after_the_runner_started_leaves_the_launch_unconfirmed_never_refused(tmp_path, monkeypatch):
    def boom(proc, ack_fd):
        raise RuntimeError("the wait broke")

    monkeypatch.setattr(remote, "await_first_record", boom)
    work = Gate(tmp_path / "work")
    req, runs_root = _request(tmp_path, "u1", [PY, "-c", Gate.WAIT, str(work.path)])
    try:
        out = remote.launch_request(req)
        assert out["started"] and out["unconfirmed"] == "RuntimeError: the wait broke" and not (runs_root / "u1" / "refused.json").exists()
    finally:
        work.release()
        wait_runner(runs_root / "u1")
        work.close()


def test_a_redelivery_during_the_wait_answers_what_the_first_delivery_settled(tmp_path):
    """An ssh drop while launch-request waits for the ack: the launcher re-delivers the request.  The re-delivery waits on the first
    one's lock and answers its refusal, never `existing` with no refusal, which the launcher would print as launched."""
    at_gate, go = Gate(tmp_path / "at-gate"), Gate(tmp_path / "go")
    wrap = [PY, "-c", "import os, sys; open(sys.argv[1], 'w').write('at\\n'); open(sys.argv[2], 'rb').read(1); os.execvp(sys.argv[3], sys.argv[3:])",
            str(at_gate.path), str(go.path)]
    req, runs_root = _request(tmp_path, "rd1", [PY, "-c", "print('never')"], harness=("--no-such-harness-flag",), **{remote.QUEUE_WRAP: wrap})
    first: dict = {}
    again: dict = {}
    t1 = threading.Thread(target=lambda: first.update(remote.launch_request(req)))
    t2 = threading.Thread(target=lambda: again.update(remote.launch_request(req)))
    try:
        t1.start()
        assert at_gate.readline() == b"at\n"
        t2.start()
        assert not again, "no answer while the first delivery is still waiting"
        go.release()
        t1.join()
        t2.join()
        assert first["fault"] == "RunnerFailed"
        assert again["existing"] and again["fault"] == "RunnerFailed" and again["refused"] == first["refused"]
    finally:
        go.release()
        t1.join()
        t2.join()
        at_gate.close()
        go.close()


def test_a_launcher_whose_first_delivery_dropped_reports_the_refusal_not_launched(tmp_path, monkeypatch, capsys):
    """Through `research run --on`: the first launch-request's ssh drops (255) once its runner is up, and the request goes on on the
    machine; the runner then fails its argument parsing.  The re-delivery answers that refusal, so the launcher exits 3."""
    root = tmp_path / "machine"
    at_gate, go = Gate(tmp_path / "at-gate"), Gate(tmp_path / "go")
    pywrap = tmp_path / "py.sh"
    pywrap.write_text(f'#!/bin/sh\ncase " $* " in *" run "*" --claimed "*) echo at > {at_gate.path}; head -c1 {go.path} >/dev/null;; esac\n'
                      f'exec {PY} "$@"\n')
    pywrap.chmod(0o755)
    fakessh = tmp_path / "ssh.sh"
    fakessh.write_text(f'#!/bin/sh\ncase "$1" in *launch-request*) if [ ! -e {tmp_path}/dropped ]; then touch {tmp_path}/dropped; '
                       f'sh -c "$1" >/dev/null 2>&1 & echo $! > {tmp_path}/lr.pid; head -c3 {at_gate.path} >/dev/null; echo "Connection reset" >&2; exit 255; fi;; esac\n'
                       f'exec sh -c "$1"\n')
    (tmp_path / "machines.toml").write_text(f'[machines.m]\nprovider = "runpod"\npod_id = "p1"\nproject = "p"\nroot = "{root}"\npython = "{pywrap}"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setattr(remote, "pod_target", lambda m: {"host": "h", "port": 22, "pod_name": "m", "cost_per_hr": 0.1,
                                                         "rate_observed_utc": "t0", "gpu": None})
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **k: ["sh", str(fakessh)])
    monkeypatch.setattr(remote.time, "sleep", lambda s: go.release())     # the back-off between deliveries: the runner goes on
    try:
        rc = _run(tmp_path, "--no-such-harness-flag")
        err = capsys.readouterr().err
        [rid] = [p.name for p in (tmp_path / "runs").iterdir()]
        wait_exit(int((tmp_path / "lr.pid").read_text()))
        assert rc == 3 and " launched on " not in err and "refused on m (RunnerFailed)" in err
        assert json.loads((root / "runs" / rid / "refused.json").read_text())["fault"] == "RunnerFailed"
    finally:
        go.release()
        at_gate.close()
        go.close()


def test_a_launch_request_past_the_launchers_deadline_is_redelivered_and_then_unknown(tmp_path, machine, monkeypatch, capsys):
    """The launcher's deadline covers the runner's start-up, which a starved CPU allocation can stretch.  Past it the request goes on
    on the machine, so the launcher re-delivers it like an ssh drop, and after three deliveries records phase `unknown`: no crash."""
    real = remote.ssh_run
    started: list[subprocess.Popen] = []

    def ssh_run(ssh, argv, **kw):
        if "launch-request" in argv:
            cmd = [*ssh, shlex.join(argv)]
            started.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        return real(ssh, argv, **kw)

    monkeypatch.setattr(remote, "ssh_run", ssh_run)
    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    try:
        rc = _run(tmp_path)
        [rid] = [p.name for p in (tmp_path / "runs").iterdir()]
        rec = json.loads((tmp_path / "runs" / rid / "remote.json").read_text())
        assert rc == 3 and len(started) == 3 and rec["phase"] == "unknown" and "no answer from launch-request" in rec["note"]
    finally:
        for p in started:
            p.wait()
        for d in (machine / "runs").iterdir():
            if d.is_dir() and (d / "launch.json").exists():
                wait_runner(d)


DETACH = ("import os, sys\npid = os.fork()\nif pid:\n    open(sys.argv[1], 'w').write(str(pid))\n    sys.exit(0)\n"
          "open(sys.argv[2], 'rb').read(1)\nos.execvp(sys.argv[3], sys.argv[3:])\n")


def test_a_wrapper_that_forks_the_runner_off_leaves_the_launch_unconfirmed_and_releases_nothing(tmp_path):
    """Every wrap in the tree execs, so the watched process is the runner.  One that forks it off and exits (`setsid -f`-like, here
    within the session) leaves its process group alive: the launch is unconfirmed and nothing is released."""
    go, work = Gate(tmp_path / "go"), Gate(tmp_path / "work")
    pidf = tmp_path / "runner.pid"
    req, runs_root = _request(tmp_path, "d1", [PY, "-c", Gate.WAIT, str(work.path)], exclusive=True,
                              **{remote.QUEUE_WRAP: [PY, "-c", DETACH, str(pidf), str(go.path)]})
    out = remote.launch_request(req)
    runner = int(pidf.read_text())
    try:
        go.release()
        assert out["started"] and "process group lives on" in out["unconfirmed"] and not (runs_root / "d1" / "refused.json").exists()
        assert (runs_root / ".exclusive.lock").read_text().strip() == "d1", "nothing released"
    finally:
        work.release()
        wait_exit(runner)
        go.close()
        work.close()


def test_a_launch_that_cannot_make_its_fifo_is_refused_and_releases_what_it_took(tmp_path, monkeypatch):
    req, runs_root = _request(tmp_path, "mk1", [PY, "-c", "print('never')"], custody={"key": "k"}, exclusive=True)
    cred = req.parent / remote.CUSTODY_DIR / "cred.json"
    cred.parent.mkdir()
    cred.write_text("{}")

    def no_fifo(path, mode=0o666):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(remote.os, "mkfifo", no_fifo)
    out = remote.launch_request(req)
    assert out["started"] is False and "could not prepare the launch" in out["refused"]
    assert (runs_root / "mk1" / "refused.json").exists() and not cred.exists() and remote.exclusive_holder(runs_root) is None


def test_a_relative_request_path_still_reaches_the_runners_ack(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "w").mkdir()
    req, runs_root = _request(tmp_path, "rel1", [PY, "-c", "print('ran')"], cwd=str(tmp_path / "w"))
    out = remote.launch_request(req.relative_to(tmp_path))
    assert out["started"] and out["first_record"] == "job.json"
    wait_runner(runs_root / "rel1")


def test_a_runner_that_wrote_job_json_without_an_ack_is_unconfirmed_not_refused(tmp_path):
    """The ack goes astray (here a wrap removes the FIFO before the runner opens it): launch-request waits for the runner's exit,
    finds job.json, and answers started and unconfirmed, never a refusal beside the run's own status."""
    req, runs_root = _request(tmp_path, "na1", [PY, "-c", "print('ran')"])
    wrap = ["sh", "-c", 'rm -f "$0"; exec "$@"', str(req.parent / remote.LAUNCH_ACK)]
    doc = json.loads(req.read_text())
    doc[remote.QUEUE_WRAP] = wrap
    req.write_text(json.dumps(doc))
    out = remote.launch_request(req)
    assert out["started"] and "wrote job.json" in out["unconfirmed"] and not (runs_root / "na1" / "refused.json").exists()
    assert R.summarize(runs_root / "na1")["state"] == "done"
