"""`research run --on` failures that used to surface only on the machine: a --timeout with units, a runner that exited before
status.json, a relative command started outside the shipped tree, and push_branch's refusal on a worktree on another branch."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from research import cli, notes, remote
from research import runs as R
from research.pods.lease import RUN_MARGIN_S


def test_a_timeout_with_units_is_shipped_as_seconds_and_seen_by_the_guards(capsys):
    assert cli._normalize_timeout(["--stage", "s", "--timeout", "40m"]) == ["--stage", "s", "--timeout", "2400"]
    assert cli._normalize_timeout(["--timeout=1.5"]) == ["--timeout=1.5"]
    assert remote.lease_extend_s(["--timeout", "40m"]) == 2400 + RUN_MARGIN_S
    assert "outlives the custody key's default lifetime" in cli._custody_ttl_problem(None, cli._normalize_timeout(["--timeout", "7h"]))
    with pytest.raises(ValueError):
        cli._normalize_timeout(["--timeout", "forty"])
    assert cli.main(["run", "--on", "vy-nowhere", "--timeout", "forty", "--", "true"]) == 2
    assert "--timeout" in capsys.readouterr().err


def test_a_relative_command_with_source_and_no_cwd_is_refused_before_launch(tmp_path, capsys):
    rc = cli.main(["run", "--on", "vy-nowhere", "--project", "p", "--runs-dir", str(tmp_path), "--no-custody-r2",
                   "--source", str(tmp_path), "--", "./scripts/go.sh"])
    assert rc == 2 and "pass --cwd source" in capsys.readouterr().err


def _machine_run(tmp_path: Path, monkeypatch, *, launcher_log: str) -> Path:
    d = tmp_path / "machine" / "r-x"
    d.mkdir(parents=True)
    dead = subprocess.Popen(["true"])
    dead.wait()
    (d / "launch.json").write_text(json.dumps({"run_id": "r-x", "runner_pid": dead.pid}))
    (d / "launcher.log").write_text(launcher_log)
    local = tmp_path / "local" / "r-x"
    local.mkdir(parents=True)
    (local / "remote.json").write_text(json.dumps({"run_id": "r-x", "phase": "launched", "host": "h", "port": 22, "remote_run_dir": str(d)}))
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **k: ["sh", "-c"])
    return local


def test_fetch_reports_a_runner_that_exited_without_status(tmp_path, monkeypatch, capsys):
    local = _machine_run(tmp_path, monkeypatch, launcher_log="usage...\nresearch run: error: argument --timeout: invalid float value: '40m'\n")
    rec = remote.fetch(local)
    assert "invalid float value" in rec["launcher_exited"]["last_line"]
    assert R.summarize(local)["state"] == "launcher-exited"
    assert cli.main(["fetch", str(local), "--runs-dir", str(local.parent)]) == 0
    assert "exited without writing status.json" in capsys.readouterr().err


def test_fetch_reports_a_running_run_whose_machine_rebooted_or_whose_runner_is_gone(tmp_path, monkeypatch, capsys):
    import socket
    local = _machine_run(tmp_path, monkeypatch, launcher_log="")
    d = tmp_path / "machine" / "r-x"
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    dead = json.loads((d / "launch.json").read_text())["runner_pid"]

    def at(pid, **env):
        (d / "status.json").write_text(json.dumps({"state": "running", "pid": pid}))
        (d / "job.json").write_text(json.dumps({"environment": {"hostname": socket.gethostname(), "boot_id": boot, **env}}))
        return remote.fetch(local)

    assert "rebooted since its runner started" in at(os.getpid(), boot_id="0-old-boot")["runner_gone"]["why"]
    assert R.summarize(local)["state"] == "runner-gone"
    assert cli.main(["fetch", str(local), "--runs-dir", str(local.parent)]) == 0
    assert "ended without recording it: the machine rebooted" in capsys.readouterr().err
    assert at(dead)["runner_gone"]["why"] == f"its runner (pid {dead}) is gone without recording an end"
    assert "runner_gone" not in at(dead, hostname="a-kubernetes-pod")      # its runner is in a pid namespace the host can't see
    assert "runner_gone" not in at(os.getpid()) and R.summarize(local)["state"] == "running"


def test_push_refusal_names_the_bind_for_a_worktree_on_another_branch(tmp_path, monkeypatch):
    repo = tmp_path / "wt"
    subprocess.run(["git", "init", "-q", "-b", "feature", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    monkeypatch.setattr(notes, "lane_branch", lambda root, lane: ("lane/l", str(repo)))
    line, rc = notes.push_branch(tmp_path, "l", None)
    assert rc == 2 and "research notes bind l --branch feature --worktree" in line and "--pod none" in line
