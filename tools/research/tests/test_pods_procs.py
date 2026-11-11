"""`research pods procs`' reading (research.scopemem's `procs`) over a fake /proc and research directory under tmp_path: a
process is given its run by its cgroup's name, else by an ancestor that is the run's runner, else, once its parent is gone, by
the one runner in its session scope, else only by its working directory; a scope holding two runs' runners names both and
decides nothing; no command line is returned; and the command refuses to list a machine's whole process table."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research import scopemem as S
from research.pods import connect

CHECK, DIRECT, OTHER, ENDED = "r20261111-060040-65f1", "r20261111-050055-b652", "r20261110-230000-0a0a", "r20261109-120000-ffff"
CHECK_SCOPE = f"/user.slice/user-1001.slice/user@1001.service/app.slice/check-c-{CHECK}.scope"
SESSION, SHARED, LONE = (f"/user.slice/user-1001.slice/session-{n}.scope" for n in (136081, 136802, 137000))
NOW = 1_800_000_600.0
SECRET = "--token=never-printed"


def _proc(proc: Path, pid: int, ppid: int, comm: str, cgroup: str, argv: tuple[str, ...] = (), cwd: str = "/workspace") -> None:
    d = proc / str(pid)
    d.mkdir(parents=True)
    (d / "stat").write_text(f"{pid} ({comm}) S {ppid} {pid} {pid} " + " ".join(["0"] * 15) + " 6000 0 0\n")
    (d / "comm").write_text(comm + "\n")
    (d / "cgroup").write_text(f"0::{cgroup}\n")
    (d / "cmdline").write_bytes("\0".join((comm, SECRET, *argv)).encode() + b"\0")
    (d / "cwd").symlink_to(cwd)


def _run(root: Path, rid: str, state: str, **request) -> None:
    for sub, name, doc in (("runs", "status.json", {"state": state}), ("requests", "request.json", request)):
        (root / sub / rid).mkdir(parents=True)
        (root / sub / rid / name).write_text(json.dumps(doc))


@pytest.fixture
def machine(tmp_path):
    proc, root = tmp_path / "proc", tmp_path / "research"
    proc.mkdir()
    (proc / "stat").write_text("cpu 0 0 0 0\nbtime 1800000000\n")
    _run(root, CHECK, "running", submitted_by="https://cursor.com/agents/bc-1", env={"KEY": "never-printed"})
    _run(root, DIRECT, "running", command=["slot.py", "--by", "lean"])
    _run(root, ENDED, "failed", submitted_by="https://cursor.com/agents/bc-2")
    _proc(proc, 10, 1, "lake", CHECK_SCOPE, cwd=f"/workspace/research/scratch/{OTHER}/tree")        # its cgroup wins over its cwd
    _proc(proc, 20, 1, "python3", SESSION, ("-m", "research", "run", "--id", DIRECT))
    _proc(proc, 21, 20, "bash", SESSION)
    _proc(proc, 22, 21, "lake", SESSION)                                                            # an ancestor is the runner
    _proc(proc, 23, 1, "lake", SESSION)                                                             # orphaned: the scope's one runner
    _proc(proc, 30, 1, "python3", SHARED, ("run", "--id", DIRECT))
    _proc(proc, 31, 1, "python3", SHARED, ("run", "--id", OTHER))
    _proc(proc, 32, 1, "lake", SHARED)                                                              # two runners: undecided
    _proc(proc, 40, 1, "lake", LONE, cwd=f"/workspace/research/scratch/{ENDED}/tree/security")      # only its cwd, of an ended run
    _proc(proc, 41, 1, "lake", "/user.slice", ("--id", "not-a-run"))                                # nothing names a run
    _proc(proc, 50, 1, "lean", SESSION)
    return proc, root


def test_a_process_gets_its_run_from_the_strongest_thing_that_names_it(machine):
    proc, root = machine
    doc = S.procs(["lake"], [], proc, root, now=NOW)
    got = {r["pid"]: r for r in doc["procs"]}
    assert sorted(got) == [10, 22, 23, 32, 40, 41], "only the processes named, never the machine's table"
    assert (got[10]["run"], got[10]["via"], got[10]["state"], got[10]["by"]) == (CHECK, "cgroup", "running", "https://cursor.com/agents/bc-1")
    assert (got[22]["run"], got[22]["via"], got[22]["runner"], got[22]["by"]) == (DIRECT, "ancestor", 20, "lean")
    assert (got[23]["run"], got[23]["via"]) == (DIRECT, "scope")
    assert (got[32]["run"], got[32]["via"], got[32]["runs"]) == (None, None, [OTHER, DIRECT])
    assert (got[40]["run"], got[40]["via"], got[40]["state"]) == (ENDED, "cwd", "failed")
    assert (got[41]["run"], got[41]["via"]) == (None, None) and "runs" not in got[41]
    assert got[10]["started"] == 1_800_000_000 + 6000 / S.os.sysconf("SC_CLK_TCK")
    assert "never-printed" not in json.dumps(doc), "neither a command line nor a request's environment is returned"


def test_a_pid_is_read_whatever_its_name_and_the_lines_say_what_names_each_run(machine):
    proc, root = machine
    doc = S.procs(["lake"], [50], proc, root, now=NOW)
    lines = {int(line.split()[0]): line for line in S.render_procs(doc).splitlines()}
    assert sorted(lines) == [10, 22, 23, 32, 40, 41, 50]
    up = int(NOW - 1_800_000_000 - 6000 / S.os.sysconf("SC_CLK_TCK")) // 60
    assert lines[10].startswith(f"10 lake  up {up} min  run {CHECK} (running), sent by https://cursor.com/agents/bc-1: its cgroup names the run")
    assert f"run {DIRECT} (running), sent by lean: an ancestor is the run's runner" in lines[22]
    assert "its parent is gone; its session scope holds this run's runner" in lines[23] and lines[50].startswith("50 lean")
    assert "no run: neither its cgroup" in lines[32] and f"holds the runners of {OTHER}, {DIRECT}" in lines[32]
    assert "only its working directory names the run: not proof it belongs to it; the run has ended and this process outlived it" in lines[40]
    assert lines[41].endswith("[cwd /workspace; cgroup /user.slice]")
    assert S.render_procs(S.procs(["nothing"], [], proc, root, now=NOW)) == ""


def test_the_command_refuses_to_list_a_whole_process_table_and_sends_the_reader_itself(machine, monkeypatch, capsys):
    proc, root = machine
    assert connect.main_procs(["vy-nebius-cpu-7"]) == 2
    assert "--comm NAME" in capsys.readouterr().err
    sent = {}

    def ssh_run(ssh, argv, *, stdin=None, timeout=0):
        sent.update(ssh=ssh, argv=argv, stdin=stdin)
        return connect.subprocess.CompletedProcess(argv, 0, json.dumps(S.procs(["lake"], [50], proc, root, now=NOW)).encode(), b"")

    monkeypatch.setattr(connect, "ssh_base", lambda target: ["ssh", target])
    monkeypatch.setattr(connect.remote, "ssh_run", ssh_run)
    assert connect.main_procs(["vy-nebius-cpu-7", "--comm", "lake", "--pid", "50"]) == 0
    assert sent["argv"] == ["python3", "-", "--procs", "--root", S.RESEARCH_ROOT, "--comm", "lake", "--pid", "50"]
    assert sent["stdin"] == Path(S.__file__).read_bytes() and sent["ssh"] == ["ssh", "vy-nebius-cpu-7"]
    out = capsys.readouterr().out
    assert out.count("\n") == 7 and f"run {CHECK} (running)" in out and "never-printed" not in out
