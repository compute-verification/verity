"""`research status`'s memory line (research.scopemem): a fake /proc and a fake cgroup v2 tree under tmp_path, the run's own
resources.jsonl samples, and the remote read run through the script it sends (no ssh)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from research import remote
from research import scopemem as S
from research.cli import main

GIB = 1 << 30
SCOPE = "/system.slice/cluster-r1.scope"
LEASE = "/user.slice/user-0.slice/user@0.service/app.slice/gpu-lease-41.scope"


def _proc(proc: Path, pid: int, pgid: int, cgroup: str, start: int = 1000, cpus: str = "0-191") -> None:
    d = proc / str(pid)
    d.mkdir(parents=True)
    (d / "stat").write_text(f"{pid} (py thon) S 1 {pgid} {pgid} " + " ".join(["0"] * 15) + f" {start} 0 0\n")
    (d / "cgroup").write_text(f"0::{cgroup}\n")
    (d / "status").write_text(f"Name:\tpython\nCpus_allowed:\tff\nCpus_allowed_list:\t{cpus}\n")


def _cg(root: Path, path: str, *, current: int | None = None, peak: int | None = None, max_: str | None = None) -> None:
    d = root / path.lstrip("/")
    d.mkdir(parents=True, exist_ok=True)
    for name, v in (("memory.current", current), ("memory.peak", peak), ("memory.max", max_)):
        if v is not None:
            (d / name).write_text(f"{v}\n")


def _no_node(tmp_path: Path) -> dict[str, str]:
    """No gpu-lease directory and no slot files: the machine running the test has its own."""
    return {"RESEARCH_GPU_LEASE_DIRS": str(tmp_path / "no-leases"), "CHECK_SLOTS": str(tmp_path / "no-slots"),
            "LEAN_SLOTS": str(tmp_path / "no-lean-slots")}


@pytest.fixture
def node(tmp_path):
    proc, cg, run = tmp_path / "proc", tmp_path / "cgroup", tmp_path / "runs" / "r1"
    proc.mkdir()
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"state": "running", "pid": 40, "pgid": 40, "starttime_ticks": 1000}))
    _proc(proc, 40, 40, SCOPE)
    _cg(cg, SCOPE, current=100 * GIB, peak=120 * GIB, max_=str(400 * GIB))
    return proc, cg, run


def test_current_peak_and_cap(node):
    proc, cg, run = node
    (r,) = S.reading(run, proc, cg)
    assert (r["cgroup"], r["current_bytes"], r["peak_bytes"], r["cap_bytes"], r["cap_at"]) == (SCOPE, 100 * GIB, 120 * GIB, 400 * GIB, SCOPE)
    assert r["memory_max"] == str(400 * GIB) and not r["near_cap"] and r["growth"] is None
    assert S.render(r) == f"{SCOPE}  current 100.0 GiB  peak 120.0 GiB  cap 400.0 GiB (25% used)  growth -"


def test_max_means_no_cap(node):
    proc, cg, run = node
    _cg(cg, SCOPE, current=900 * GIB, max_="max")
    (r,) = S.reading(run, proc, cg)
    assert r["memory_max"] == "max" and r["cap_bytes"] is None and not r["near_cap"]
    assert "cap none (memory.max is max)" in S.render(r) and "NEAR CAP" not in S.render(r)


def test_cap_set_by_an_ancestor_slice(node):
    proc, cg, run = node
    _cg(cg, SCOPE, max_="max")
    _cg(cg, "/system.slice", max_=str(110 * GIB))
    (r,) = S.reading(run, proc, cg)
    assert (r["cap_bytes"], r["cap_at"]) == (110 * GIB, "/system.slice")
    assert "cap 110.0 GiB (at /system.slice, 91% used)" in S.render(r) and r["near_cap"]


@pytest.mark.parametrize("gib,near", [(359, False), (360, True), (399, True)])
def test_near_cap_marker_within_ten_percent(node, gib, near):
    proc, cg, run = node
    _cg(cg, SCOPE, current=gib * GIB)
    (r,) = S.reading(run, proc, cg)
    assert r["near_cap"] is near
    assert S.render(r).endswith("  NEAR CAP") is near


def test_growth_from_the_runs_own_samples(node):
    proc, cg, run = node
    rate = 32 * GIB // 60                                         # 32 GiB/min
    lines = []
    for i in range(10):                                           # t_mono 0..90 s every 10 s, another cgroup's and a process record between
        lines.append({"kind": "cgroup", "t_mono": 10.0 * i, "cgroup": {"path": SCOPE, "memory_current_bytes": 10 * GIB + rate * 10 * i}})
        lines.append({"kind": "cgroup", "t_mono": 10.0 * i + 1, "cgroup": {"path": "/elsewhere", "memory_current_bytes": 0}})
        lines.append({"kind": "process", "t_mono": 10.0 * i + 2, "processes": []})
    (run / "resources.jsonl").write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in lines) + '{"kind":"cgroup","t_mo')
    (r,) = S.reading(run, proc, cg)
    assert r["growth"] == {"bytes_per_s": pytest.approx(rate), "over_s": 60.0}
    assert "growth +32.0 GiB/min over 60 s of samples" in S.render(r)


def test_growth_needs_two_samples_of_that_cgroup(node):
    proc, cg, run = node
    (run / "resources.jsonl").write_text(json.dumps({"kind": "cgroup", "t_mono": 5.0, "cgroup": {"path": SCOPE, "memory_current_bytes": 1}}) + "\n")
    (r,) = S.reading(run, proc, cg)
    assert r["growth"] is None


def test_every_cgroup_of_the_workload_group_and_the_job_user_unit(node):
    proc, cg, run = node
    _proc(proc, 41, 40, SCOPE)                                    # gpu-lease's shell, beside the runner
    _proc(proc, 42, 40, LEASE)                                    # its CMD in gpu-lease-41.scope
    _proc(proc, 50, 50, "/system.slice/other.scope")              # another group
    _proc(proc, 43, 40, "/")                                      # the namespace root is the machine's, not the run's
    _cg(cg, LEASE, current=5 * GIB, max_=str(200 * GIB))
    _cg(cg, "/system.slice/research-job-r1.service", current=GIB, max_="max")
    got = [r["cgroup"] for r in S.reading(run, proc, cg)]
    assert got == [SCOPE, LEASE, "/system.slice/research-job-r1.service"]


def test_a_reused_pid_is_not_the_run(node):
    proc, cg, run = node
    (run / "status.json").write_text(json.dumps({"state": "running", "pid": 40, "pgid": 40, "starttime_ticks": 999}))
    assert S.reading(run, proc, cg) == []


def test_status_prints_a_local_active_runs_memory(node, monkeypatch, capsys):
    proc, cg, run = node
    (run / "job.json").write_text("{}")
    _cg(cg, SCOPE, current=380 * GIB)
    monkeypatch.setenv("RESEARCH_PROC_ROOT", str(proc))
    monkeypatch.setenv("RESEARCH_CGROUP_ROOT", str(cg))
    monkeypatch.setenv("RESEARCH_GPU_LEASE_DIRS", str(run.parent / "no-leases"))
    assert main(["status", "--runs-dir", str(run.parent)]) == 0
    out = capsys.readouterr().out
    assert f"mem  {SCOPE}  current 380.0 GiB  peak 120.0 GiB  cap 400.0 GiB (95% used)  growth -  NEAR CAP" in out
    assert main(["status", "--runs-dir", str(run.parent), "--json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)["runs"]
    assert row["scope_memory"][0]["near_cap"] is True


def test_status_reads_a_remote_runs_memory_with_the_script_it_sends(node, tmp_path, monkeypatch, capsys):
    """The remote read runs scopemem.py as the machine would (`python3 - RUN_DIR`, the file on stdin), with ssh replaced by a local
    subprocess: what is sent must run on its own, outside the package."""
    proc, cg, machine_run = node
    local = tmp_path / "control" / "r1"
    local.mkdir(parents=True)
    (local / "status.json").write_text(json.dumps({"state": "running"}))
    (local / "remote.json").write_text(json.dumps({"machine": "vy-nebius-2", "host": "h", "port": 22, "remote_run_dir": str(machine_run)}))
    sent = []

    def ssh_run(ssh, argv, *, stdin=None, timeout=120):
        sent.append(argv)
        assert argv[:2] == ["python3", "-"]
        return subprocess.run([sys.executable, "-", *argv[2:]], input=stdin, capture_output=True, timeout=timeout,
                              env={"RESEARCH_PROC_ROOT": str(proc), "RESEARCH_CGROUP_ROOT": str(cg), **_no_node(tmp_path)}, cwd=tmp_path)

    monkeypatch.setattr(remote, "ssh_run", ssh_run)
    monkeypatch.setattr(remote, "fetch", lambda d, **_: {})
    assert main(["status", "--runs-dir", str(local.parent)]) == 0
    out = capsys.readouterr().out
    assert sent == [] and "memory of 1 active remote run(s): `--refresh` or `research status RUN`" in out
    assert main(["status", "--runs-dir", str(local.parent), "--refresh"]) == 0
    out = capsys.readouterr().out
    assert sent == [["python3", "-", str(machine_run)]]
    assert f"mem  {SCOPE}  current 100.0 GiB  peak 120.0 GiB  cap 400.0 GiB (25% used)  growth -" in out


def test_status_says_why_a_remote_read_failed(node, tmp_path, monkeypatch, capsys):
    local = tmp_path / "control" / "r1"
    local.mkdir(parents=True)
    (local / "status.json").write_text(json.dumps({"state": "running"}))
    (local / "remote.json").write_text(json.dumps({"machine": "vy-nebius-2", "host": "h", "port": 22, "remote_run_dir": "/x"}))
    monkeypatch.setattr(remote, "ssh_run", lambda *a, **k: subprocess.CompletedProcess(a, 255, b"", b"ssh: connect refused"))
    monkeypatch.setattr(remote, "fetch", lambda d, **_: {})
    assert main(["status", "--runs-dir", str(local.parent), "--refresh"]) == 0
    assert "mem  unread: reading vy-nebius-2: rc 255 ssh: connect refused" in capsys.readouterr().out
