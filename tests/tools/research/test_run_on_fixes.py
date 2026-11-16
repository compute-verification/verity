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


def test_a_bare_timeout_under_five_minutes_is_refused_for_a_machine_with_the_unit_to_write(capsys):
    """A PR's runs passed `--timeout 60` meaning minutes and were killed 60 s after taking their slots."""
    for argv in (["--on", "vy-nowhere"], ["--on", "auto"], ["--queue", "--source", ".", "--project", "p"]):
        assert cli.main(["run", *argv, "--timeout", "60", "--", "true"]) == 2
        err = capsys.readouterr().err
        assert "--timeout 60m" in err and "--timeout 60s" in err, err
    assert cli._short_bare_timeout(["--timeout", "90s"]) is None
    assert cli._short_bare_timeout(["--timeout=60m"]) is None
    assert cli._short_bare_timeout(["--timeout", "300"]) is None
    assert cli._short_bare_timeout(["--stage", "s"]) is None
    assert "--timeout 1.5m" in cli._short_bare_timeout(["--timeout=1.5"])


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


def test_fetch_of_a_file_named_twice_fetches_it_and_every_file_after_it(tmp_path, monkeypatch):
    """`research status RUN --also steps.jsonl` names steps.jsonl twice, once itself and once in `checkview.CHECK_FILES`: tar
    stores a second copy of a name as a hard link, which a stream can't extract, and every file after it (suites.failed.json,
    a running check's failures so far) never arrived."""
    local = _machine_run(tmp_path, monkeypatch, launcher_log="")
    d = tmp_path / "machine" / "r-x"
    (d / "status.json").write_text(json.dumps({"state": "running"}))
    for name in ("spans.jsonl", "steps.jsonl", "suites.failed.json"):
        (d / name).write_text(f"{name}\n")
    rec = remote.fetch(local, also=("steps.jsonl", "suites.failed.json", "spans.jsonl", "steps.jsonl"))
    assert "last_fetch_error" not in rec, rec
    assert (local / "suites.failed.json").read_text() == "suites.failed.json\n"


def test_a_run_that_ends_between_the_status_copy_and_the_runner_probe_is_its_end_not_runner_gone(tmp_path, monkeypatch):
    """`--wait` copied status.json while the run was running, then probed the runner after it had ended and exited, and called
    a passing run runner-gone. A failed probe rereads the status first."""
    import socket
    local = _machine_run(tmp_path, monkeypatch, launcher_log="")
    d = tmp_path / "machine" / "r-x"
    dead = json.loads((d / "launch.json").read_text())["runner_pid"]
    (d / "status.json").write_text(json.dumps({"state": "running", "pid": dead}))
    (d / "job.json").write_text(json.dumps({"environment": {"hostname": socket.gethostname(),
                                                            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}}))
    probe = remote._runner_gone

    def ended_meanwhile(*a, **k):
        (d / "status.json").write_text(json.dumps({"state": "done", "rc": 0, "pid": dead}))
        return probe(*a, **k)

    monkeypatch.setattr(remote, "_runner_gone", ended_meanwhile)
    rec = remote.fetch(local)
    assert "runner_gone" not in rec and rec["last_observed"] == {**rec["last_observed"], "state": "done", "rc": 0}
    assert R.summarize(local)["state"] == "done"


def test_push_refusal_names_the_bind_for_a_worktree_on_another_branch(tmp_path, monkeypatch):
    repo = tmp_path / "wt"
    subprocess.run(["git", "init", "-q", "-b", "feature", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    monkeypatch.setattr(notes, "lane_branch", lambda root, lane: ("lane/l", str(repo)))
    line, rc = notes.push_branch(tmp_path, "l", None)
    assert rc == 2 and "research notes bind l --branch feature --worktree" in line and "--pod none" in line
