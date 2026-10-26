"""`research cancel`: a remote run stops through SIGTERM to its workload's process group, recorded as a manual cancel, never by
signalling its runner. Cancels that signalled the runner and then SIGKILLed it during its custody upload left 15 node 1 runs without
custody (1-2 Oct). It works from any VM: a run another VM launched is found on its machine (`remote.adopt`). The group is signalled
only while one of its members is the run's workload (RESEARCH_RUN_DIR in its environment): a runner that died leaves status.json
`running`, and its pgid may be another job's group by then (red team of #1043, C1)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from research import cli, remote
from research.telemetry.procs import stat_fields

TOOL = Path(remote.__file__).resolve().parents[1]


def _workload(d: Path) -> subprocess.Popen:
    """A `cat` on a pipe the test holds, started as the runner starts a workload: its own process group, RESEARCH_RUN_DIR set."""
    return subprocess.Popen(["cat"], stdin=subprocess.PIPE, start_new_session=True, env={**os.environ, "RESEARCH_RUN_DIR": str(d.resolve())})


def _snapshot(root: Path, name: str = "t1") -> str:
    """The shipped tool where the machine keeps it, <root>/tool/<snapshot>: a link to this checkout's research package."""
    (root / "tool").mkdir(parents=True, exist_ok=True)
    (root / "tool" / name).symlink_to(TOOL)
    return str(root / "tool" / name)


def _run(tmp_path: Path, monkeypatch, state: str = "running"):
    """A launched run on a 'machine' reached through `sh -c`: its workload and a stand-in runner, each `cat` on a pipe the test holds,
    in process groups of their own."""
    root = tmp_path / "machine"
    d = root / "runs" / "r-x"
    d.mkdir(parents=True)
    workload, runner = _workload(d), subprocess.Popen(["cat"], stdin=subprocess.PIPE, start_new_session=True)
    stream = f"h:{runner.pid}:{stat_fields(runner.pid)['starttime_ticks']}"
    (d / "status.json").write_text(json.dumps({"state": state, "pid": workload.pid, "pgid": workload.pid, "transitions": [
        {"state": "running", "stream": stream, "pid": workload.pid, "pgid": workload.pid}]}))
    (d / "events.jsonl").write_text("")
    local = tmp_path / "local" / "r-x"
    local.mkdir(parents=True)
    (local / "remote.json").write_text(json.dumps({"run_id": "r-x", "phase": "launched", "host": "h", "port": 22, "remote_run_dir": str(d),
                                                   "root": str(root), "tool": _snapshot(root)}))
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
        assert intent["verified"] is True and intent["target"]["run_dir_members"] == [workload.pid]
        assert result["kind"] == "cancel_result" and result["delivered"] is True
        out = capsys.readouterr().out
        assert f"process group {workload.pid} (1 process(es))" in out
        assert "the runner SIGKILLs what is left of the run 20 s after the SIGTERM" in out
        assert json.loads((d / "control" / "cancel").read_text())["grace_s"] == remote.CANCEL_GRACE_S == 20
        assert result["scope"]["runner_kills_after_s"] == 20
    finally:
        _end(workload, runner)


def test_cancel_says_how_long_the_run_ran_and_how_much_of_it_waited_for_a_check_slot():
    """af2d (10 Oct) was launched at 16:56Z, waited in node 1's check line until 18:31Z and was cancelled at 18:43Z: 107 minutes
    in by its state, which reads `running` from launch, but 12 of checking."""
    from datetime import datetime, timezone

    now = datetime(2026, 10, 10, 18, 43, tzinfo=timezone.utc)
    s = {"started_utc": "2026-10-10T16:56:00.120Z", "slot": {"state": "held", "since": "2026-10-10T16:56:10Z", "took": "2026-10-10T18:31:00Z",
                                                              "slot": "a", "cores": "0-79"}}
    assert cli._ran_text(s, now) == ("ran 107 min since 2026-10-10T16:56Z: waited 95 min for a check slot, then checked 12 min in "
                                     "slot a")
    s["slot"] = {"state": "waiting", "since": "2026-10-10T16:56:10Z", "why": "every check slot is taken"}
    assert cli._ran_text(s, now).endswith("waiting for a check slot since 16:56Z (107 min), so its check never started")
    assert cli._ran_text({"started_utc": "2026-10-10T18:00:00Z", "slot": None}, now) == "ran 43 min since 2026-10-10T18:00Z"
    assert cli._ran_text({"started_utc": None}, now) == ""


RUN = "r20261003-101500-c0de"


def _launched_elsewhere(tmp_path: Path, monkeypatch, *, tool: bool = True):
    """A running run another VM launched on ssh machine vy-b (a directory reached over `sh -c`, as in test_remote_adopt): no local run
    dir, and the shipped tool named only where the runner records it, job.json `run.remote.tool`."""
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    d = tmp_path / "b" / "runs" / RUN
    d.mkdir(parents=True)
    workload = _workload(d)
    (d / "status.json").write_text(json.dumps({"state": "running", "pid": workload.pid, "pgid": workload.pid}))
    (d / "job.json").write_text(json.dumps({"run": {"remote": {"machine": "vy-b", "tool": _snapshot(tmp_path / "b") if tool else None}}}))
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


def test_a_run_held_for_disk_before_it_starts_is_withdrawn_and_recorded_so(tmp_path, monkeypatch, capsys):
    """A queued run waiting in its machine's `research disk-hold` has no status.json, and the cancel refused it (`not running
    yet`): on 19 Oct infra ended 20 such holds by hand. Here a real disk-hold holds it (at 0%, so the disk never has room)."""
    root = tmp_path / "machine"
    d, req = root / "runs" / "r-x", root / "requests" / "r-x"
    d.mkdir(parents=True)
    req.mkdir(parents=True)
    (req / "request.json").write_text("{}")
    hold = subprocess.Popen([sys.executable, "-m", "research", "disk-hold", "--gb", "1", "--path", str(tmp_path), "--ack",
                             str(req / remote.LAUNCH_ACK), "--", "true"], stderr=subprocess.PIPE, start_new_session=True,
                            env={**os.environ, "PYTHONPATH": str(TOOL), "RESEARCH_DISK_HOLD_PCT": "0"})
    try:
        assert b"research disk-hold: holding" in hold.stderr.readline()
        (d / "launch.json").write_text(json.dumps({"run_id": "r-x", "runner_pid": hold.pid, "request": str(req / "request.json")}))
        local = tmp_path / "local" / "r-x"
        local.mkdir(parents=True)
        (local / "remote.json").write_text(json.dumps({"run_id": "r-x", "phase": "launched", "host": "h", "port": 22, "remote_run_dir": str(d),
                                                       "root": str(root), "tool": _snapshot(root)}))
        monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **k: ["sh", "-c"])
        assert cli.main(["cancel", str(local), "--reason", "superseded", "--by", "infra", "--runs-dir", str(local.parent)]) == 0
        assert hold.wait() == -signal.SIGTERM
        assert f"withdrawn before it started: SIGTERM to its hold (research disk-hold pid {hold.pid})" in capsys.readouterr().out
        for where in (d, local):
            refused = json.loads((where / "refused.json").read_text())
            assert (refused["fault"], refused["reason"], refused["by"]) == ("Withdrawn", "superseded", "infra")
        intent, result = [json.loads(x) for x in (local / "events.jsonl").read_text().splitlines()]
        assert (intent["kind"], intent["actor"], intent["target"]["pids"]) == ("cancel_intent", "manual", [hold.pid])
        assert result["kind"] == "cancel_result" and result["target_still_present"] == []
        assert cli.main(["cancel", str(local), "--reason", "x", "--runs-dir", str(local.parent)]) == 1
        assert "r-x was refused on its machine (Withdrawn: superseded), so there is nothing to cancel" in capsys.readouterr().err
    finally:
        if hold.poll() is None:
            hold.kill()
            hold.wait()


def test_a_run_that_is_not_running_is_refused_and_nothing_is_signalled(tmp_path, monkeypatch, capsys):
    local, d, workload, runner = _run(tmp_path, monkeypatch, state="done")
    try:
        assert cli.main(["cancel", str(local), "--reason", "x", "--runs-dir", str(local.parent)]) == 1
        assert "r-x is done, so there is nothing to cancel" in capsys.readouterr().err
        assert (d / "events.jsonl").read_text() == "" and workload.poll() is None
    finally:
        _end(workload, runner)


def _held(tmp_path: Path, monkeypatch):
    """A run its machine holds for disk ahead of its runner, as remote._launch_request starts one: launch.json names the hold's pid
    and argv, `research disk-hold --gb <more than any disk> --ack <root>/requests/r-x/.launch-ack -- <runner argv>`, and no
    status.json.  The test returns once the hold acknowledged the launch on the FIFO, so it is holding."""
    local, d, workload, runner = _run(tmp_path, monkeypatch)
    _end(workload, runner)
    (d / "status.json").unlink()
    root, req = tmp_path / "machine", tmp_path / "machine" / "requests" / "r-x"
    (req / "custody").mkdir(parents=True)
    (req / "custody" / "cred.json").write_text("{}")
    (tmp_path / "ramlock").mkdir()
    (mem := tmp_path / "ramlock" / "r-x.json").write_text(json.dumps({"run_id": "r-x", "gb": 40}))
    (req / "request.json").write_text(json.dumps({"run_id": "r-x", "runs_root": str(root / "runs"), "custody": True}))
    ack = req / ".launch-ack"
    os.mkfifo(ack)
    fd = os.open(ack, os.O_RDONLY | os.O_NONBLOCK)
    keep = os.open(ack, os.O_WRONLY)                 # a writer held open, so the read below waits for the hold's line, not EOF
    argv = [sys.executable, "-m", "research", "disk-hold", "--gb", "1e12", "--path", str(root), "--ack", str(ack), "--",
            sys.executable, "-m", "research", "run", "--id", "r-x", "--remote-json", json.dumps({"mem": {"reservation": str(mem)}}), "--", "true"]
    hold = subprocess.Popen(argv, env={**os.environ, "PYTHONPATH": str(TOOL)}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    os.set_blocking(fd, True)
    assert os.read(fd, 64) == b"disk-hold\n"
    os.close(fd)
    os.close(keep)
    (d / "launch.json").write_text(json.dumps({"run_id": "r-x", "runner_pid": hold.pid, "argv": argv}))
    return local, d, req, mem, hold


def test_cancel_of_a_run_held_for_disk_ends_the_hold_and_records_it_withdrawn(tmp_path, monkeypatch, capsys):
    """INF-101: `research cancel` of a run in `research disk-hold` said "not running yet", so a held run could only wait."""
    local, d, req, mem, hold = _held(tmp_path, monkeypatch)
    try:
        assert cli.main(["cancel", str(local), "--reason", "superseded", "--by", "ci", "--runs-dir", str(local.parent)]) == 0
        assert hold.wait() == -signal.SIGTERM
        refused = json.loads((d / "refused.json").read_text())
        assert (refused["fault"], refused["reason"], refused["by"]) == ("Withdrawn", "superseded", "ci")
        assert [(h["hold"], h["pid"]) for h in refused["holds"]] == [("disk-hold", hold.pid)]
        assert not mem.exists() and not (req / "custody" / "cred.json").exists(), "what the launch took goes"
        assert json.loads((local / "refused.json").read_text())["fault"] == "Withdrawn", "fetched back"
        assert cli.R.summarize(local)["state"] == "refused"
        out = capsys.readouterr().out
        assert f"withdrawn before it started: SIGTERM to its hold (research disk-hold pid {hold.pid})" in out
        assert f"removed what its launch staged: " in out and str(mem) in out
        assert f"research cancel: withdrawn from its disk-hold (pid {hold.pid}): superseded" in (d / "launcher.log").read_text()

        assert cli.main(["cancel", str(local), "--reason", "again", "--runs-dir", str(local.parent)]) == 1
        assert "r-x was refused on its machine (Withdrawn: superseded), so there is nothing to cancel" in capsys.readouterr().err
    finally:
        if hold.poll() is None:
            hold.kill()
            hold.wait()


def test_a_held_runs_pid_that_is_no_longer_its_hold_is_not_signalled(tmp_path, monkeypatch, capsys):
    """The hold execs the runner in place, so its pid is the runner's once it releases; and a pid may be reused after it exits.
    The hold is found by its argv naming the run's ack FIFO, never by launch.json's pid."""
    local, d, _req, mem, hold = _held(tmp_path, monkeypatch)
    other = subprocess.Popen(["cat"], stdin=subprocess.PIPE, start_new_session=True)
    try:
        launch = json.loads((d / "launch.json").read_text())
        (d / "launch.json").write_text(json.dumps({**launch, "runner_pid": other.pid}))
        assert cli.main(["cancel", str(local), "--reason", "x", "--runs-dir", str(local.parent)]) == 0
        assert hold.wait() == -signal.SIGTERM and other.poll() is None, "the hold ends, the pid launch.json names is left alone"
        assert [h["pid"] for h in json.loads((d / "refused.json").read_text())["holds"]] == [hold.pid]
    finally:
        _end(other)
        if hold.poll() is None:
            hold.kill()
            hold.wait()


def test_a_stale_running_status_whose_pgid_is_another_jobs_group_is_refused(tmp_path, monkeypatch, capsys):
    """C1: the run's workload and runner are gone (SIGKILL, OOM, a reboot), status.json still says running, and another run's
    workload holds the pgid now."""
    here, d, workload = _launched_elsewhere(tmp_path, monkeypatch)
    _end(workload)
    other = _workload(tmp_path / "b" / "runs" / "r20261003-111111-0bed")
    try:
        (d / "status.json").write_text(json.dumps({"state": "running", "pid": other.pid, "pgid": other.pid}))
        assert cli.main(["cancel", RUN, "--on", "vy-b", "--reason", "x", "--runs-dir", str(here)]) == 1
        assert other.poll() is None, "SIGTERM went to a process group that is not the run's workload"
        assert f"process group {other.pid} not delivered (refused: no member of the group is the expected workload" in capsys.readouterr().err
        intent, result = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines()][-2:]
        assert intent["verified"] is False and intent["target"]["run_dir_members"] == [] and result["delivered"] is False
    finally:
        _end(other)


def test_a_run_whose_tool_predates_the_workload_check_is_refused_not_signalled_unchecked(tmp_path, monkeypatch, capsys):
    here, d, workload = _launched_elsewhere(tmp_path, monkeypatch)
    old = tmp_path / "b" / "tool" / "old" / "research" / "telemetry"
    old.mkdir(parents=True)
    (old.parent / "__init__.py").write_text("")
    (old / "__init__.py").write_text("")
    (old / "__main__.py").write_text("import sys\nsys.stderr.write('tele cancel-intent: error: unrecognized arguments: --expect-run-dir x\\n')\n"
                                     "sys.exit(2)\n")
    (d / "job.json").write_text(json.dumps({"run": {"remote": {"machine": "vy-b", "tool": str(tmp_path / "b" / "tool" / "old")}}}))
    try:
        assert cli.main(["cancel", RUN, "--on", "vy-b", "--reason", "x", "--runs-dir", str(here)]) == 1
        assert "predates the workload check, so nothing was signalled" in capsys.readouterr().err and workload.poll() is None
    finally:
        _end(workload)


def test_a_run_whose_tool_predates_the_scope_end_gets_the_group_sigterm_and_is_told_so(tmp_path, monkeypatch, capsys):
    """A run launched before the scope end (INF-93) ships a cancel-intent without --grace-s: the cancel is delivered the old way and
    says that nothing SIGKILLs what ignores it."""
    here, d, workload = _launched_elsewhere(tmp_path, monkeypatch)
    old = tmp_path / "b" / "tool" / "old" / "research" / "telemetry"
    old.mkdir(parents=True)
    (old.parent / "__init__.py").write_text("")
    (old / "__init__.py").write_text("")
    (old / "__main__.py").write_text(
        "import os, sys\n"
        "if '--grace-s' in sys.argv:\n"
        "    sys.stderr.write('tele cancel-intent: error: unrecognized arguments: --grace-s 20\\n')\n"
        "    sys.exit(2)\n"
        f"os.execve(sys.executable, [sys.executable, '-m', 'research.telemetry', *sys.argv[1:]], {{**os.environ, 'PYTHONPATH': {str(TOOL)!r}}})\n")
    (d / "job.json").write_text(json.dumps({"run": {"remote": {"machine": "vy-b", "tool": str(tmp_path / "b" / "tool" / "old")}}}))
    try:
        assert cli.main(["cancel", RUN, "--on", "vy-b", "--reason", "x", "--runs-dir", str(here)]) == 0
        assert workload.wait() == -signal.SIGTERM
        out = capsys.readouterr().out
        assert "predates the scope end: SIGTERM to its group and to the processes naming its run dir only" in out
        assert "the runner SIGKILLs" not in out and not (d / "control" / "cancel").exists()
    finally:
        _end(workload)


def test_a_tool_outside_the_machines_tool_snapshots_is_refused(tmp_path, monkeypatch, capsys):
    """C2: job.json's run.remote.tool becomes PYTHONPATH on the machine, so it must be one of the machine's <root>/tool/ snapshots."""
    here, d, workload = _launched_elsewhere(tmp_path, monkeypatch)
    evil = tmp_path / "elsewhere"
    (evil / "research" / "telemetry").mkdir(parents=True)
    (evil / "research" / "__init__.py").write_text("")
    (evil / "research" / "telemetry" / "__init__.py").write_text("")
    marker = tmp_path / "ran"
    (evil / "research" / "telemetry" / "__main__.py").write_text(f"open({str(marker)!r}, 'w').write('x')\n")
    try:
        for tool in (str(evil), f"{tmp_path / 'b' / 'tool' / 't1'}:{evil}", str(tmp_path / "b" / "tool" / "..")):
            (d / "job.json").write_text(json.dumps({"run": {"remote": {"machine": "vy-b", "tool": tool}}}))
            assert cli.main(["cancel", RUN, "--on", "vy-b", "--reason", "x", "--runs-dir", str(here)]) == 1
            assert "is not a snapshot under" in capsys.readouterr().err
        assert not marker.exists() and workload.poll() is None
    finally:
        _end(workload)


def test_on_naming_another_machine_than_the_local_dirs_is_refused(tmp_path, monkeypatch, capsys):
    """C3: once `research status RUN` adopted the run from vy-b, `cancel RUN --on vy-a` must not cancel it on vy-b."""
    here, d, workload = _launched_elsewhere(tmp_path, monkeypatch)
    try:
        assert cli.main(["status", RUN, "--runs-dir", str(here)]) == 0
        assert cli.main(["cancel", RUN, "--on", "vy-a", "--reason", "x", "--runs-dir", str(here)]) == 1
        assert f"{RUN} is on vy-b (its local run dir), not on vy-a" in capsys.readouterr().err and workload.poll() is None
    finally:
        _end(workload)


def test_a_run_id_two_machines_hold_is_refused_without_on(tmp_path, monkeypatch, capsys):
    """C4: adopt takes the first machine in name order; a cancel asks every machine and refuses an id two of them hold."""
    here, d, workload_b = _launched_elsewhere(tmp_path, monkeypatch)
    da = tmp_path / "a" / "runs" / RUN
    da.mkdir(parents=True)
    workload_a = _workload(da)
    (da / "status.json").write_text(json.dumps({"state": "running", "pid": workload_a.pid, "pgid": workload_a.pid}))
    (da / "job.json").write_text(json.dumps({"run": {"remote": {"machine": "vy-a", "tool": _snapshot(tmp_path / "a")}}}))
    (da / "events.jsonl").write_text("")
    try:
        assert cli.main(["cancel", RUN, "--reason", "x", "--runs-dir", str(here)]) == 1
        assert f"vy-a and vy-b each hold a run {RUN}: name one with --on" in capsys.readouterr().err
        assert workload_a.poll() is None and workload_b.poll() is None and not (here / RUN).exists()
        assert cli.main(["cancel", RUN, "--on", "vy-a", "--reason", "x", "--runs-dir", str(here)]) == 0
        assert workload_a.wait() == -signal.SIGTERM and workload_b.poll() is None
    finally:
        _end(workload_a, workload_b)
