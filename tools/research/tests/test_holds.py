"""What `research status` / `inspect` say a live run got and waits on (research.holds over research.scopemem's `node`): a fake
/proc, cgroup tree, gpu-lease directory and slot files under tmp_path, read through the program `node_readings` sends (no ssh),
in the formats the nodes write (gpu-lease's owner and wait lines, n1-lease.json, slot.py's `.who` files and tickets)."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from research import holds as H
from research import remote
from research.cli import main

SCOPE = "/user.slice/user-1001.slice/user@1001.service/app.slice/cluster-r1.scope"
SESSION = "/user.slice/user-1001.slice/session-197561.scope"
#: vy-nebius-2's /run/gpu-lease/0.owner on 6 Oct, its run renamed r1
OWNER = ("pid=40 run=r1 who=bc-1beaff8e uuid=GPU-5f1149a4-e5f5-1007-ec78-7cd350a69a4f since=2026-10-06T16:46:00Z "
         "until=2026-10-06T17:11:00Z mem=1544G timed=1 scope=gpu-lease-40 cmd=env VY_PROVERS_CPUS=160-175 vy-provers taskset -c 160-175")
FENCE = "pid=70 run=- who=kubernetes uuid=- since=2026-10-06T11:53:08Z until=2026-10-07T11:53:08Z cmd=n1-lease-fence: not in node 1's lease pool (n1_lease.py)"
NOW = datetime(2026, 10, 7, 3, 10, tzinfo=UTC)


def _proc(proc: Path, pid: int, cgroup: str, cpus: str) -> None:
    d = proc / str(pid)
    d.mkdir(parents=True)
    (d / "stat").write_text(f"{pid} (python) S 1 {pid} {pid} " + " ".join(["0"] * 15) + " 1000 0 0\n")
    (d / "cgroup").write_text(f"0::{cgroup}\n")
    (d / "status").write_text(f"Name:\tpython\nCpus_allowed_list:\t{cpus}\n")


def _flock(path: Path, pid: int) -> str:
    st = os.stat(path)
    return f"1: FLOCK  ADVISORY  WRITE {pid} {os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino} 0 EOF"


@pytest.fixture
def machine(tmp_path):
    """Run r1 in a cluster scope pinned to 96-127, holding GPU 0, check slot b and Lean slot check/0; run r2 in a session
    scope under `taskset -c 64-95`, waiting in gpu-lease for 2 GPUs (one waiter ahead) and in the check line."""
    proc, cg, leases, locks, runs = (tmp_path / n for n in ("proc", "cgroup", "gpu-lease", "locks", "runs"))
    for d in (proc, leases, locks / "check-line"):
        d.mkdir(parents=True)
    (proc / "stat").write_text("cpu 0 0 0 0\nbtime 1791334800\n")
    for rid, pid, cgroup, cpus in (("r1", 40, SCOPE, "96-127"), ("r2", 60, SESSION, "64-95")):
        (runs / rid).mkdir(parents=True)
        (runs / rid / "status.json").write_text(json.dumps({"state": "running", "pid": pid, "pgid": pid, "starttime_ticks": 1000}))
        _proc(proc, pid, cgroup, cpus)
    _proc(proc, 70, "/kubepods.slice", "0-191")
    for path, own, eff in ((SCOPE, "96-127", "96-127"), (SESSION, "", "0-159")):
        d = cg / path.lstrip("/")
        d.mkdir(parents=True)
        (d / "memory.current").write_text(f"{8 << 30}\n")
        (d / "memory.max").write_text(f"{64 << 30}\n")
        (d / "cpuset.cpus").write_text(own + "\n")
        (d / "cpuset.cpus.effective").write_text(eff + "\n")
    for i in range(4):
        (leases / f"{i}.lock").touch()
    (leases / "0.owner").write_text(OWNER + "\n")
    (leases / "1.owner").write_text(OWNER.replace("pid=40", "pid=99") + "\n")          # a lease that ended: its lock is free
    (leases / "2.owner").write_text(FENCE + "\n")
    (leases / "wait.1791335000000000000.80").write_text("pid=80 n=1 run=rX who=other since=2026-10-07T03:00:00Z mem=8G\n")
    (leases / "wait.1791335060000000000.60").write_text("pid=60 n=2 run=r2 who=lane since=2026-10-07T03:01:00Z max=30 mem=193G preempt=1\n")
    (leases / "wait.1791334000000000000.81").write_text("pid=81 n=1 run=r2 who=gone since=2026-10-07T02:40:00Z\n")   # a dead waiter's
    (leases / "n1-lease.json").write_text(json.dumps({"t": "2026-10-07T03:09:58Z", "pool": 1, "pool_name": "node1", "blocked": None,
                                                      "pending_holders": ["gpu-pool-x"], "waiting": [], "gpus": []}))
    (locks / "slots").write_text("a 32-63 only=trains   # train checks\nb 64-95\n")
    (locks / "check-a.lock").touch()
    (locks / "check-b.lock").touch()
    (locks / "check-b.who").write_text(json.dumps({"pid": 40, "run": "r1", "tree": "/workspace/research/src/abc", "by": "infra"}))
    (locks / "lean-slots").write_text("check 1\n")
    (locks / "lean-check-0.lock").touch()
    (locks / "lean-check-0.who").write_text(json.dumps({"pid": 40, "run": "r1", "took": "2026-10-07T03:05Z"}))
    ticket = locks / "check-line" / "01791335070000000000-60"
    ticket.write_text(json.dumps({"pid": 60, "run": "r2", "tree": "/workspace/research/src/def", "by": "lane"}) + "\n")
    held = [leases / "0.lock", leases / "2.lock", leases / "wait.1791335000000000000.80", leases / "wait.1791335060000000000.60"]
    (proc / "locks").write_text("\n".join([*(_flock(p, 1) for p in held), _flock(locks / "check-b.lock", 40),
                                           _flock(locks / "lean-check-0.lock", 40)]) + "\n")
    fd = os.open(ticket, os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_EX)                                # the waiter's lock on its ticket, which slot.py probes
    env = {"RESEARCH_PROC_ROOT": str(proc), "RESEARCH_CGROUP_ROOT": str(cg), "RESEARCH_GPU_LEASE_DIRS": str(leases),
           "CHECK_SLOTS": str(locks / "slots"), "LEAN_SLOTS": str(locks / "lean-slots")}
    yield runs, leases, env
    os.close(fd)


def _control(tmp_path: Path, runs: Path, machine: str = "vy-nebius-2") -> Path:
    for rid in ("r1", "r2"):
        d = tmp_path / "control" / rid
        d.mkdir(parents=True)
        (d / "status.json").write_text(json.dumps({"state": "running"}))
        (d / "job.json").write_text("{}")
        (d / "remote.json").write_text(json.dumps({"machine": machine, "host": "h", "port": 22, "user": "research",
                                                   "remote_run_dir": str(runs / rid)}))
    return tmp_path / "control"


def test_status_reads_every_run_on_a_machine_with_one_ssh(machine, tmp_path, monkeypatch, capsys):
    runs, leases, env = machine
    sent = []

    def ssh_run(ssh, argv, *, stdin=None, timeout=120):
        sent.append(argv)
        return subprocess.run([sys.executable, "-", *argv[2:]], input=stdin, capture_output=True, timeout=timeout, env=env, cwd=tmp_path, check=False)

    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, user="root", **_: ["ssh", f"{user}@{host}"])
    monkeypatch.setattr(remote, "ssh_run", ssh_run)
    monkeypatch.setattr(remote, "fetch", lambda d, **_: {})
    assert main(["status", "--runs-dir", str(_control(tmp_path, runs)), "--refresh"]) == 0
    out = capsys.readouterr().out
    assert sent == [["python3", "-", str(runs / "r1"), str(runs / "r2")]]
    assert f"mem  {SCOPE}  current 8.0 GiB" in out
    assert f"cpus 96-127 (affinity); AllowedCPUs 96-127 at {SCOPE}\n" in out
    assert "cpus 64-95 (affinity); its cgroup allows 0-159\n" in out
    assert "gpus GPU 0 of " + str(leases) + "  pid 40 since 2026-10-06T16:46Z until 2026-10-06T17:11Z mem 1544G timed\n" in out
    assert "slot check slot b (cores 64-95) since " in out and "slot Lean slot check/0 since " in out
    assert (f"waits gpu-lease {leases}: 2 GPU(s) preemptible since 2026-10-07T03:01Z; 1 waiter(s) ahead, 2 of 4 GPUs unlocked; "
            "pool 1 GPU(s), 1 holder(s) pending\n") in out
    assert "waits check slot since " in out and ": 0 ahead in line; b held (a trains only)\n" in out
    assert out.count("waits") == 2 and "gpus GPU 1" not in out


def test_inspect_says_what_a_live_run_holds(machine, tmp_path, monkeypatch, capsys):
    runs, leases, env = machine
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, user="root", **_: ["ssh", f"{user}@{host}"])
    monkeypatch.setattr(remote, "ssh_run", lambda ssh, argv, *, stdin=None, timeout=120: subprocess.run(
        [sys.executable, "-", *argv[2:]], input=stdin, capture_output=True, timeout=timeout, env=env, cwd=tmp_path, check=False))
    control = _control(tmp_path, runs)
    assert main(["inspect", str(control / "r2"), "--runs-dir", str(control)]) == 0
    out = capsys.readouterr().out
    assert "cpus      64-95 (affinity); its cgroup allows 0-159\n" in out
    assert f"waits     gpu-lease {leases}: 2 GPU(s) preemptible" in out


def test_a_machine_that_cant_be_reached_leaves_the_others_read(machine, tmp_path, monkeypatch, capsys):
    def ssh_argv(host, port, user="root", **_):
        raise RuntimeError("RUNPOD_SSH_KEY_B64 not in this shell")

    monkeypatch.setattr(remote, "ssh_argv", ssh_argv)
    monkeypatch.setattr(remote, "fetch", lambda d, **_: {})
    assert main(["status", "--runs-dir", str(_control(tmp_path, machine[0])), "--refresh"]) == 0
    assert capsys.readouterr().out.count("mem  unread: reading vy-nebius-2: RuntimeError: RUNPOD_SSH_KEY_B64 not in this shell") == 2


DESC = """
[[workstreams]]
name = "provers"
node = "vy-nebius-1"
gpus = 2

[[workstreams]]
name = "deployments-gpu"
node = "vy-nebius-1"
gpus = 6

[policy.vy-nebius-1.lease-pool.provers]
max = 8

[policy.vy-nebius-1.lease-pool.circuits]
dir = "/run/gpu-lease-circuits"
queue = "deployments-gpu"
"""


def test_node1_cap_is_the_pool_queues_quota_unless_every_waiter_is_preemptible(tmp_path):
    desc = tmp_path / "nebius.toml"
    desc.write_text(DESC)
    plain, fill = [{"n": "4"}], [{"n": "4", "preempt": "1"}]
    assert H.n1_cap("/run/gpu-lease", 4, plain, None, NOW, desc) == (
        "node 1's provers pool grants at most the provers queue's quota, 2 GPU(s), less its other Workloads', while a waiter isn't "
        "--preemptible; this asks for 4, more than that")
    assert "the deployments-gpu queue's quota, 6 GPU(s)" in H.n1_cap("/run/gpu-lease-circuits", 1, plain, None, NOW, desc)
    assert "borrows more only while every waiter is --preemptible" in H.n1_cap("/run/gpu-lease", 4, fill, None, NOW, desc)
    lim = "max=8 borrow=6 borrow-until=2026-10-07T12:00:00Z from=policy"
    assert H.n1_cap("/run/gpu-lease", 4, fill, lim, NOW, desc).startswith(
        "node 1's provers pool grants at most 8 GPU(s): the provers queue's quota 2 + borrow 6, max 8")
    assert H.n1_cap("/run/gpu-lease", 4, plain, lim, NOW, desc).endswith("this asks for 4, more than that")
    assert H.n1_cap("/run/gpu-lease", 4, fill, lim.replace("10-07", "10-06"), NOW, desc).startswith("node 1's provers pool grants at most 2 GPU(s)")
    assert H.n1_cap("/run/other", 1, plain, None, NOW, desc) is None
    assert H.n1_cap("/run/gpu-lease", 1, plain, None, NOW) is not None          # the repository's description still reads


def test_node1_waits_name_their_cap():
    node = {"leases": {"/run/gpu-lease": {"gpus": 8, "locked": 8, "owners": [], "limits": None, "pool": None,
                                          "waits": [{"file": "wait.1.60", "pid": "60", "n": "1", "run": "r2", "since": "2026-10-07T03:01:00Z"}]}}}
    (line,) = H.gpu_waits("r2", "vy-nebius-1", node, NOW)
    assert "0 of 8 GPUs unlocked; node 1's provers pool grants at most the provers queue's quota" in line
    assert "pool grants" not in H.gpu_waits("r2", "vy-nebius-2", node, NOW)[0]
