"""The Nebius server (pods/nebius/): an ssh machine `research run --on` reaches by the host its registry file names, the lease loop
stopping a non-RunPod machine through its own command, gpu-lease sharing the GPUs, launch.sh's rendered cloud-init and VM spec,
and the weight list's pins.  No cloud: nothing here calls Nebius; the ssh probe is stubbed."""
from __future__ import annotations

import base64
import datetime
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from research import remote
from research.pods import lease as L
from research.pods import registry

NEBIUS = Path(registry.__file__).resolve().parent / "nebius"
GPU_LEASE = Path(registry.__file__).resolve().parent / "sh" / "gpu_lease.sh"
CHECKPOINTS = Path(__file__).resolve().parents[3] / "integrations" / "vllm" / "manifests" / "checkpoints.json"
HOST = "203.0.113.7"


@pytest.fixture
def reg(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "notes" / "machines.d"
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setenv("RESEARCH_MACHINES_D", str(d))
    return d


def reachable(_entry: dict) -> None:
    return None


# ---- the ssh provider --------------------------------------------------------------------------------------------------------------------

def test_register_host_writes_an_ssh_machine_that_run_on_resolves(reg):
    p, what = registry.register_ssh("vy-nebius-1", HOST, "verity", user="research", rate=14.40, root="/workspace/research", probe=reachable)
    assert what == "registered" and p == reg / "vy-nebius-1.toml"
    doc = tomllib.loads(p.read_text())
    assert doc["provider"] == "ssh" and doc["host"] == HOST and doc["user"] == "research" and doc["rate_usd_per_hour"] == 14.40
    m = remote.resolve_machine("vy-nebius-1", "verity")
    tgt = remote.pod_target(m)
    assert (tgt["host"], tgt["port"], tgt["user"], tgt["cost_per_hr"]) == (HOST, 22, "research", 14.40)
    ssh = remote.ssh_argv(tgt["host"], tgt["port"], user=tgt["user"])
    assert ssh[-1] == f"research@{HOST}" and ssh[ssh.index("-p") + 1] == "22"
    assert registry.register_ssh("vy-nebius-1", HOST, "verity", user="research", rate=14.40, root="/workspace/research",
                                 probe=reachable)[1] == "unchanged"


def test_register_host_refuses_an_unreachable_host_and_a_silent_rebind(reg):
    with pytest.raises(remote.Refused, match="does not answer over ssh"):
        registry.register_ssh("vy-nebius-1", HOST, "verity", probe=lambda e: "ssh rc 255: Connection refused")
    assert not reg.exists()
    registry.register_ssh("vy-nebius-1", HOST, "verity", probe=reachable)
    with pytest.raises(remote.Refused, match=f"binds it to host {HOST}.*--replace"):
        registry.register_ssh("vy-nebius-1", "198.51.100.9", "verity", probe=reachable)
    assert registry.register_ssh("vy-nebius-1", "198.51.100.9", "verity", replace=True, probe=reachable)[1] == "updated"


def test_an_ssh_entry_without_a_usable_host_is_refused(reg):
    reg.mkdir(parents=True)
    (reg / "vy-bad.toml").write_text('provider = "ssh"\nproject = "verity"\n')
    with pytest.raises(remote.Refused, match="host None is not a host"):
        remote.resolve_machine("vy-bad", "verity")
    with pytest.raises(remote.Refused, match="needs host"):
        remote.pod_target({"provider": "ssh", "name": "x"})
    with pytest.raises(remote.Refused, match="runpod or ssh"):
        remote.pod_target({"provider": "aws", "name": "x"})


# ---- the lease on a machine that is not a pod --------------------------------------------------------------------------------------------

def test_the_lease_loop_stops_a_non_runpod_machine_by_its_own_command(tmp_path):
    rec = tmp_path / "stop.sh"
    rec.write_text(f'#!/bin/sh\necho "$1" >> {tmp_path / "stopped"}\necho "TERMINATED $1"\n')
    (tmp_path / "environ").write_bytes(b"")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LEASE_", "RESEARCH_LEASE", "RUNPOD_"))}
    env |= {"LEASE_DIR": str(tmp_path / "lease"), "LEASE_ENVIRON": str(tmp_path / "environ"), "LEASE_TERMINATE_CMD": f"sh {rec}",
            "LEASE_DEADLINE_FILE": str(tmp_path / "no-deadline"),     # not the host's /etc/research/deadline (a Nebius server has one)
            "LEASE_MACHINE_ID": "computeinstance-e00test", "LEASE_ONCE": "1"}
    now = 1_800_000_000
    assert subprocess.run(["sh", str(L.LEASE_SH), "extend", "60", "test"], env={**env, "LEASE_NOW": str(now)}).returncode == 0
    not_due = subprocess.run(["sh", str(L.LEASE_SH), "loop"], env={**env, "LEASE_NOW": str(now + 60 + 899)})
    assert not_due.returncode == 1 and not (tmp_path / "stopped").exists()
    due = subprocess.run(["sh", str(L.LEASE_SH), "loop"], env={**env, "LEASE_NOW": str(now + 60 + 900)})
    assert due.returncode == 0 and (tmp_path / "stopped").read_text() == "computeinstance-e00test\n"


# ---- gpu-lease -----------------------------------------------------------------------------------------------------------------------------

needs_flock = pytest.mark.skipif(shutil.which("flock") is None or shutil.which("bash") is None, reason="gpu-lease needs bash and flock(1)")


def gpu_env(tmp_path: Path, count: int = 3) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "RESEARCH_RUN_DIR"}
    return {**env, "GPU_LEASE_DIR": str(tmp_path / "gl"), "GPU_LEASE_COUNT": str(count), "RESEARCH_RUN_ID": "r-test",
            "GPU_LEASE_USAGE_LOG": str(tmp_path / "lease-usage.jsonl"), "GPU_LEASE_UTIL_LOG": str(tmp_path / "util-*.jsonl")}


@needs_flock
def test_gpu_lease_reports_each_gpus_held_and_busy_time_from_the_sampler_log_and_keeps_cmds_status(tmp_path):
    # started long ago, so the lease's held time is far past its 30 s of readings and the cap at held time doesn't bite
    env = {**gpu_env(tmp_path, count=2), "GPU_LEASE_UUID_TABLE": "GPU-a0,GPU-b1", "GPU_LEASE_WHO": "bc-test",
           "RESEARCH_RUN_DIR": str(tmp_path / "run"), "GPU_LEASE_STARTED": "2026-01-01T00:00:00Z"}
    (tmp_path / "run").mkdir()
    # CMD writes the node sampler's readings as it runs, stamped inside the lease: GPU 0 busy (util), busy (SM activity 2%), idle;
    # the GPU it doesn't hold is busy throughout and must not count
    rows = [("50", "0.3"), ("0", "0.02"), ("0", "0")]
    lines = "; ".join(
        f'echo "{{\\"t\\":\\"$t\\",\\"gpus\\":[{{\\"i\\":0,\\"uuid\\":\\"GPU-a0\\",\\"util\\":{u},\\"smact\\":{s}}},'
        f'{{\\"i\\":1,\\"uuid\\":\\"GPU-b1\\",\\"util\\":99,\\"smact\\":1}}]}}" >> {tmp_path / "util-1.jsonl"}' for u, s in rows)
    script = f"t=$(date -u +%Y-%m-%dT%H:%M:%SZ); {lines}; exit 3"
    r = subprocess.run(["bash", str(GPU_LEASE), "1", "--on", "0", "--mem-gb", "0", "--", "sh", "-c", script], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 3, r.stderr
    assert "gpu-lease: GPU 0 (GPU-a0) held" in r.stderr and "busy 0m20s of 0m30s sampled (67%)" in r.stderr
    for log in (tmp_path / "lease-usage.jsonl", tmp_path / "run" / "gpu-lease-usage.jsonl"):
        (rec,) = [json.loads(ln) for ln in log.read_text().splitlines()]
        assert rec["schema"] == "gpu-lease/usage/v1" and rec["rc"] == 3 and rec["who"] == "bc-test" and rec["run"] == "r-test"
        assert rec["gpus"] == [{"index": 0, "uuid": "GPU-a0", "held_s": rec["gpus"][0]["held_s"], "busy_s": 20, "sampled_s": 30}]
    free = subprocess.run(["bash", str(GPU_LEASE), "status"], env=env, capture_output=True, text=True).stdout
    assert [ln.split()[:2] for ln in free.splitlines() if ln and not ln.startswith("#")] == [["0", "free"], ["1", "free"]]


@needs_flock
def test_gpu_lease_caps_busy_and_sampled_time_at_the_held_time(tmp_path):
    """Each reading counts a whole cadence, so readings near a lease's ends added up to more than it held (pous infra, 18:56Z).
    Two busy readings of a cadence longer than any held time must record busy_s == sampled_s == held_s."""
    env = {**gpu_env(tmp_path, count=1), "GPU_LEASE_UUID_TABLE": "GPU-a0", "GPU_LEASE_STARTED": "2026-01-01T00:00:00Z",
           "GPU_LEASE_UTIL_CADENCE_S": str(10**10)}
    reading = f'echo "{{\\"t\\":\\"$t\\",\\"gpus\\":[{{\\"i\\":0,\\"uuid\\":\\"GPU-a0\\",\\"util\\":90,\\"smact\\":0.9}}]}}" >> {tmp_path / "util-1.jsonl"}'
    script = f"t=$(date -u +%Y-%m-%dT%H:%M:%SZ); {reading}; {reading}"
    r = subprocess.run(["bash", str(GPU_LEASE), "1", "--mem-gb", "0", "--", "sh", "-c", script], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    (rec,) = [json.loads(ln) for ln in (tmp_path / "lease-usage.jsonl").read_text().splitlines()]
    (g,) = rec["gpus"]
    assert g["held_s"] > 0 and g["busy_s"] == g["sampled_s"] == g["held_s"], g


FILL_RUNNER = GPU_LEASE.parent.parent / "nebius" / "fill_runner.py"


@needs_flock
def test_fill_runner_counts_a_queued_waiter_but_not_a_lease_that_holds_its_gpus_with_wait_in_its_argv(tmp_path, monkeypatch):
    """A `gpu-lease 1 --wait` that holds its GPU keeps `--wait` in its argv. Counting it as a waiter froze fill across node 2 for as
    long as any such lease ran (bc-2aa33ad8, 17:56Z); only a request queued in gpu-lease's own wait files is a waiter."""
    env = gpu_env(tmp_path, count=1)
    lease = tmp_path / "bin" / "gpu-lease"
    lease.parent.mkdir()
    lease.symlink_to(GPU_LEASE)
    pause = [sys.executable, "-c", "import signal; print('ready', flush=True); signal.pause()"]
    holder = subprocess.Popen(["bash", str(lease), "1", "--wait", "--mem-gb", "0", "--", *pause], env=env,
                              stdout=subprocess.PIPE, text=True, start_new_session=True)
    waiter = None
    try:
        assert holder.stdout.readline().strip() == "ready"
        monkeypatch.setenv("GPU_LEASE_DIR", env["GPU_LEASE_DIR"])
        monkeypatch.setenv("FILL_DIR", str(tmp_path / "fill"))
        spec = importlib.util.spec_from_file_location("fill_runner_under_test", FILL_RUNNER)
        fr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fr)
        (tmp_path / "gl" / "wait.1.99999").write_text("pid=99999 n=4 run=- who=gone since=2026-09-30T00:00:00Z\n")
        assert fr.waiters() == []
        waiter = subprocess.Popen(["bash", str(lease), "1", "--wait", "--mem-gb", "0", "--", "true"],
                                  env={**env, "GPU_LEASE_POLL_S": "3600"}, stderr=subprocess.PIPE, text=True, start_new_session=True)
        line = waiter.stderr.readline()
        assert line.startswith("gpu-lease: waiting for 1 GPU(s), queued as wait."), line
        assert fr.waiters() == [1]
    finally:
        for p in (waiter, holder):
            if p is not None:
                stop_group(p)


def stop_group(p: subprocess.Popen) -> None:
    """SIGTERM a process started with start_new_session and everything in its group. Terminating only the bash leaves the
    `sleep` it waits on (GPU_LEASE_POLL_S=3600) or the leased command orphaned for the length of that sleep."""
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    p.wait()


def test_fill_runner_does_not_count_a_waiter_pinned_inside_keep_free(tmp_path, monkeypatch):
    """Memory accounting's `gpu-lease 1 --on 7 --wait` requests, on node 2's kept-free GPU 7, stopped fill from starting on the
    six free GPUs (pouw-node2, 1 Oct 13:05Z): fill never takes a kept-free GPU, so a waiter pinned inside keep-free holds none back.
    A waiter pinned partly outside it, unpinned, or pinned by UUID still counts."""
    gl, fill = tmp_path / "gl", tmp_path / "fill"
    gl.mkdir()
    fill.mkdir()
    (fill / "keep-free").write_text("6,7  # pous\n")
    monkeypatch.setenv("GPU_LEASE_DIR", str(gl))
    monkeypatch.setenv("FILL_DIR", str(fill))
    spec = importlib.util.spec_from_file_location("fill_runner_keep_free", FILL_RUNNER)
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    locks = []
    try:
        for i, on in enumerate(["7", "6,7", "6-7", "5,7", None, "GPU-af0bf9e0"]):
            f = gl / f"wait.{i}.{100 + i}"
            f.write_text(f"pid={100 + i} n={i + 1}{f' on={on}' if on else ''} run=- who=w since=2026-10-01T13:00:00Z\n")
            fd = os.open(f, os.O_RDONLY)
            fcntl.flock(fd, fcntl.LOCK_EX)
            locks.append(fd)
        assert fr.waiters() == [4, 5, 6]
        (fill / "keep-free").unlink()
        assert fr.waiters() == [1, 2, 3, 4, 5, 6]
    finally:
        for fd in locks:
            os.close(fd)


def test_fill_runner_leaves_waiters_to_the_agent_while_it_holds_agent_lock(tmp_path, monkeypatch):
    """node2-ops (21:15Z): while the agent decides, fill_runner stops no GPU fill for a waiter, since the agent evicts; a waiting
    or running window still stops it (infra, 1 Oct 06:50Z: the agent's eviction signals gpu-lease's pid only, which leaves a
    `research run` workload on the GPU), and once agent.lock is free the next tick is today's rules."""
    gl, fill = tmp_path / "gl", tmp_path / "fill"
    (fill / "queue").mkdir(parents=True)
    gl.mkdir()
    monkeypatch.setenv("GPU_LEASE_DIR", str(gl))
    monkeypatch.setenv("FILL_DIR", str(fill))
    spec = importlib.util.spec_from_file_location("fill_runner_agent", FILL_RUNNER)
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    fill_line = "pid=5 run=- who=fill:t since=2026-09-30T20:00:00Z preempt=1 cmd=x"
    held = {0: fill_line}
    want: list[int] = []
    stopped: list[str] = []
    monkeypatch.setattr(fr, "reap", lambda: None)
    monkeypatch.setattr(fr, "verity_pool", lambda queued: None)
    monkeypatch.setattr(fr, "lease", lambda i: held.get(i))
    monkeypatch.setattr(fr, "waiters", lambda: list(want))
    monkeypatch.setattr(fr, "stop", lambda k, why, requeue=True: stopped.append(why))
    fr.running["fill-1"] = {"t0": 0, "paused": False, "started": "-",
                            "h": {"gpus": 1, "project": "pous", "owner": "t", "max_min": 10**9}}

    def stops(waiting: list[int], timed: bool = False) -> list[str]:
        want[:] = waiting
        held.pop(1, None)
        if timed:
            held[1] = "pid=6 run=- who=w since=2026-09-30T20:00:00Z timed=1 cmd=y"
        stopped.clear()
        fr.tick(2)
        return list(stopped)

    assert stops([2]) == ["timed window"] and stops([1, 1]) == ["free 1 < needed 2"]
    agent = os.open(gl / "agent.lock", os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(agent, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert fr.agent_up() and stops([2]) == ["timed window"] and stops([1, 1]) == []
        assert stops([], timed=True) == ["timed window"]
    finally:
        os.close(agent)
    assert not fr.agent_up() and stops([2]) == ["timed window"]


def test_fill_runner_ranks_commits_first_reclaims_for_them_and_keeps_out_of_booked_windows(tmp_path, monkeypatch):
    """Node 2's order (Daniel, 1 Oct): circuits' Commits, then pous, then proofs. A queued Commit with no free GPU stops the
    newest proofs fill (one per Commit), never while a window waits; no GPU job starts whose max_min reaches a booked window;
    and a GPU job's lease scope is killed when it stops and swept once it exits."""
    gl, fill = tmp_path / "gl", tmp_path / "fill"
    for d in ("queue", "running", "logs"):
        (fill / d).mkdir(parents=True)
    gl.mkdir()
    monkeypatch.setenv("GPU_LEASE_DIR", str(gl))
    monkeypatch.setenv("FILL_DIR", str(fill))
    spec = importlib.util.spec_from_file_location("fill_runner_rank", FILL_RUNNER)
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    held: dict[int, str] = {}
    want: list[int] = []
    started, stopped = [], []
    monkeypatch.setattr(fr, "reap", lambda: None)
    monkeypatch.setattr(fr, "verity_pool", lambda queued: None)
    monkeypatch.setattr(fr, "lease", lambda i: held.get(i))
    monkeypatch.setattr(fr, "waiters", lambda: list(want))
    monkeypatch.setattr(fr, "start", lambda p, h, backfill=False: started.append(p.name) or p.unlink())
    monkeypatch.setattr(fr, "stop", lambda k, why, requeue=True: stopped.append(k) or fr.running[k].update(stopping=True, why=why))

    def queue(*names):
        for name in names:
            head = "gpus=1 project=verity prio=20" if name.startswith("verity-commit-") else (
                "gpus=1 project=verity" if name.startswith("pn2g") else "gpus=1")
            (fill / "queue" / name).write_text(f"# fill: owner=o max_min=30 {head}\n")

    def run(name, t0, gpu):
        held[gpu] = f"pid={10 + gpu} run=- who=fill:o since=2026-10-01T00:00:00Z preempt=1 cmd=x"
        fr.running[name] = {"t0": t0, "paused": False, "started": "-",
                            "h": {"gpus": 1, "project": "verity" if name.startswith("pn2g") else "pous", "owner": "o",
                                  "max_min": 10**9}}

    run("pous-a.sh", 1, 0)
    run("pn2g-old.sh", 2, 1)
    run("pn2g-new.sh", 3, 2)
    queue("verity-commit-c1.sh")
    fr.tick(3)
    assert stopped == ["pn2g-new.sh"] and started == []
    fr.tick(3)
    assert stopped == ["pn2g-new.sh"]                          # one GPU already freeing for the one Commit
    queue("verity-commit-c2.sh")
    fr.tick(3)
    assert stopped == ["pn2g-new.sh", "pn2g-old.sh"]
    for k, gpu in (("pn2g-new.sh", 2), ("pn2g-old.sh", 1)):
        fr.running.pop(k), held.pop(gpu)
    want[:] = [3]
    fr.tick(3)
    assert stopped == ["pn2g-new.sh", "pn2g-old.sh", "pous-a.sh"] and started == []   # a waiting window drains; no Commit starts
    want.clear()
    fr.running.clear()
    held.clear()
    queue("pn2g-x.sh", "pous-b.sh")
    fr.tick(3)
    assert started == ["verity-commit-c1.sh", "verity-commit-c2.sh", "pous-b.sh"]
    started.clear()
    stopped.clear()
    queue("verity-commit-c3.sh")
    t = 1_790_848_800.0                                        # 2026-10-01T10:00Z
    (fill / "windows").write_text("2026-10-01T10:00Z 30 # a booked window\n\nnot a line\n")
    assert fr.windows() == [(t, t + 1800)]
    monkeypatch.setattr(fr, "wall", lambda: t - 600)
    fr.tick(3)
    assert started == [] and stopped == []                     # 30 min from 09:50Z reach the window
    monkeypatch.setattr(fr, "wall", lambda: t - 1800)
    fr.tick(3)
    assert started == ["verity-commit-c3.sh", "pn2g-x.sh"]
    assert not fr.clears({"max_min": 30}, [(t, t + 1800)], t + 60) and fr.clears({"max_min": 30}, [(t, t + 1800)], t + 1800)

    calls = []
    queue("pous-c.sh")
    spec.loader.exec_module(fr)
    monkeypatch.setattr(fr.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setattr(fr.subprocess, "Popen", lambda cmd, **kw: type("P", (), {"pid": 4242, "poll": lambda self: 143})())
    monkeypatch.setattr(fr, "event", lambda *a, **kw: calls.append(a[:2]))
    fr.start(fill / "queue" / "pous-c.sh", fr.header(fill / "queue" / "pous-c.sh"))
    assert fr.running["pous-c.sh"]["unit"] == "gpu-lease-4242"
    assert (fill / "running" / ".pous-c.sh.unit").read_text() == "gpu-lease-4242\n"
    monkeypatch.setattr(fr.os, "killpg", lambda pid, sig: None)
    fr.stop("pous-c.sh", "test")
    assert ["systemctl", "--user", "kill", "--signal=SIGTERM", "gpu-lease-4242.scope"] in calls
    fr.reap()
    assert ["systemctl", "--user", "kill", "--signal=SIGKILL", "gpu-lease-4242.scope"] in calls
    assert ("scope-residue", "pous-c.sh") in calls and "pous-c.sh" not in fr.running
    assert (fill / "queue" / "pous-c.sh").exists()


@needs_flock
def test_gpu_lease_passes_stdin_to_cmd_and_a_sigterm_on_to_it_and_records_the_stop(tmp_path):
    env = {**gpu_env(tmp_path, count=1), "GPU_LEASE_UTIL_LOG": str(tmp_path / "absent-*.jsonl")}
    r = subprocess.run(["bash", str(GPU_LEASE), "1", "--mem-gb", "0", "--", "cat"], env=env, input="through\n",
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout == "through\n" and "no sampler log matches" in r.stderr
    pause = [sys.executable, "-c", "import signal; print('ready', flush=True); signal.pause()"]
    held = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--mem-gb", "0", "--", *pause], env=env, stdout=subprocess.PIPE, text=True)
    assert held.stdout.readline().strip() == "ready"
    held.terminate()
    assert held.wait() == 143
    recs = [json.loads(ln) for ln in (tmp_path / "lease-usage.jsonl").read_text().splitlines()]
    assert [x["rc"] for x in recs] == [0, 143] and recs[1]["gpus"][0]["sampled_s"] == 0


@needs_flock
def test_gpu_lease_hands_out_disjoint_gpus_and_frees_them_when_the_holder_exits(tmp_path):
    env = gpu_env(tmp_path)
    inner = (f'echo "outer=$CUDA_VISIBLE_DEVICES $CUDA_DEVICE_ORDER"; bash {GPU_LEASE} 2 -- true; echo "short=$?"; '
             f'bash {GPU_LEASE} 1 -- sh -c \'echo "inner=$CUDA_VISIBLE_DEVICES"\'; bash {GPU_LEASE} status')
    r = subprocess.run(["bash", str(GPU_LEASE), "2", "--", "sh", "-c", inner], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == "outer=0,1 PCI_BUS_ID" and "short=75" in lines and "inner=2" in lines
    assert [ln.split()[:2] for ln in lines[-3:]] == [["0", "held"], ["1", "held"], ["2", "free"]] and "run=r-test" in lines[-3]
    after = subprocess.run(["bash", str(GPU_LEASE), "status"], env=env, capture_output=True, text=True).stdout.split("\n")
    assert [ln.split()[:2] for ln in after if ln] == [["0", "free"], ["1", "free"], ["2", "free"]]


@needs_flock
def test_gpu_lease_refuses_more_gpus_than_the_machine_has_and_bad_arguments(tmp_path):
    env = gpu_env(tmp_path, count=2)
    assert subprocess.run(["bash", str(GPU_LEASE), "3", "--", "true"], env=env, capture_output=True).returncode == 2
    assert subprocess.run(["bash", str(GPU_LEASE), "0", "--", "true"], env=env, capture_output=True).returncode == 2
    assert subprocess.run(["bash", str(GPU_LEASE), "1", "true"], env=env, capture_output=True).returncode == 2


@needs_flock
def test_gpu_lease_hands_out_only_the_direct_gpus_when_kubernetes_owns_the_rest(tmp_path):
    allowed = tmp_path / "direct-gpus"
    allowed.write_text("1,2\n")
    env = {**gpu_env(tmp_path, count=4), "GPU_LEASE_ALLOWED_FILE": str(allowed)}
    r = subprocess.run(["bash", str(GPU_LEASE), "2", "--", "sh", "-c", "echo $CUDA_VISIBLE_DEVICES"], env=env, capture_output=True,
                       text=True)
    assert r.returncode == 0 and r.stdout.strip() == "1,2"
    assert subprocess.run(["bash", str(GPU_LEASE), "3", "--", "true"], env=env, capture_output=True).returncode == 2
    status = subprocess.run(["bash", str(GPU_LEASE), "status"], env=env, capture_output=True, text=True).stdout.splitlines()
    assert status[0].startswith("# direct-run GPUs") and [ln.split()[0] for ln in status[1:]] == ["1", "2"]
    allowed.write_text("none\n")
    r = subprocess.run(["bash", str(GPU_LEASE), "1", "--", "true"], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "submit through the cluster's queues" in r.stderr


@needs_flock
def test_gpu_lease_pins_by_index_or_uuid_and_status_names_the_holder(tmp_path):
    env = {**gpu_env(tmp_path, count=4), "GPU_LEASE_UUID_TABLE": "GPU-a0,GPU-b1,GPU-c2,GPU-d3", "GPU_LEASE_WHO": "bc-test"}
    show = 'echo "$CUDA_VISIBLE_DEVICES $GPU_LEASE_UUID"'
    inner = (f"{show}; bash {GPU_LEASE} 1 --on 2 -- true; echo pinned-busy=$?; "
             f"bash {GPU_LEASE} 1 --on 2,3 -- sh -c '{show}'; bash {GPU_LEASE} status")
    r = subprocess.run(["bash", str(GPU_LEASE), "1", "--on", "GPU-c2", "--", "sh", "-c", inner], env=env, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[:3] == ["2 GPU-c2", "pinned-busy=75", "3 GPU-d3"]
    held = [ln for ln in lines[3:] if ln.split()[1] == "held"]
    assert len(held) == 1 and held[0].startswith("2 held ") and "who=bc-test" in held[0] and "uuid=GPU-c2" in held[0]
    assert re.match(r"2 held \d+m pid=", held[0]) and "timed=1" not in held[0]
    for tag, extra in (("flag", ["--timed"]), ("env", [])):
        r = subprocess.run(["bash", str(GPU_LEASE), "1", *extra, "--mem-gb", "0", "--", "bash", str(GPU_LEASE), "status"],
                           env={**env, **({"GPU_LEASE_TIMED": "1"} if tag == "env" else {})}, capture_output=True, text=True)
        assert r.returncode == 0 and any("held" in ln and "timed=1" in ln for ln in r.stdout.splitlines()), (tag, r.stdout)
    for bad in (["--on", "7"], ["--on", "GPU-zz"], ["--on", "x1"]):
        assert subprocess.run(["bash", str(GPU_LEASE), "1", *bad, "--", "true"], env=env, capture_output=True).returncode == 2
    assert subprocess.run(["bash", str(GPU_LEASE), "2", "--on", "1", "--", "true"], env=env, capture_output=True).returncode == 2
    allowed = tmp_path / "direct-gpus"
    allowed.write_text("2,3\n")
    r = subprocess.run(["bash", str(GPU_LEASE), "1", "--on", "GPU-b1", "--", "true"],
                       env={**env, "GPU_LEASE_ALLOWED_FILE": str(allowed)}, capture_output=True, text=True)
    assert r.returncode == 2 and "not one gpu-lease hands out" in r.stderr


@needs_flock
def test_gpu_lease_leaves_the_gpus_to_an_older_waiter_and_forgets_a_dead_one(tmp_path):
    env = gpu_env(tmp_path, count=2)
    gl = tmp_path / "gl"
    gl.mkdir()
    assert subprocess.run(["bash", str(GPU_LEASE), "1", "--wait", "--", "true"], env=env).returncode == 0
    assert not list(gl.glob("wait.*"))                                  # met at once: never queued
    queued = gl / "wait.0000000000000000001.1"
    waiter = subprocess.Popen(["bash", "-c", f'exec 9>"{queued}"; flock -x 9; echo "pid=1 n=2 who=timed-window" >&9; echo ready; read x'],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert waiter.stdout.readline().strip() == "ready"
        r = subprocess.run(["bash", str(GPU_LEASE), "1", "--", "true"], env=env, capture_output=True, text=True)
        assert r.returncode == 75 and "waiting first" in r.stderr and "who=timed-window" in r.stderr
        status = subprocess.run(["bash", str(GPU_LEASE), "status"], env=env, capture_output=True, text=True).stdout.splitlines()
        assert status[-1].startswith("waiting ") and "who=timed-window" in status[-1]
    finally:
        waiter.stdin.close()
        waiter.wait()
    assert subprocess.run(["bash", str(GPU_LEASE), "1", "--", "true"], env=env).returncode == 0
    assert not queued.exists()


@needs_flock
def test_gpu_lease_backfills_a_short_request_ahead_of_a_waiter_that_cannot_start_yet(tmp_path):
    """A 3-GPU waiter can't start before GPU 0's lease (no until=, so its start plus 30 min) ends: a 10-minute request may use
    the two free GPUs meanwhile, a 40-minute one may not, and a backfilled lease records when it ends."""
    env = gpu_env(tmp_path, count=3)
    gl = tmp_path / "gl"
    gl.mkdir()
    holder = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--", "bash", "-c", "echo ready; read x"], env=env,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "ready"
    queued = gl / "wait.0000000000000000001.1"
    waiter = subprocess.Popen(["bash", "-c", f'exec 9>"{queued}"; flock -x 9; echo "pid=1 n=3 run=r-t who=timed-window" >&9; '
                               'echo ready; read x'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert waiter.stdout.readline().strip() == "ready"
        long = subprocess.run(["bash", str(GPU_LEASE), "1", "--max-min", "40", "--", "true"], env=env, capture_output=True, text=True)
        assert long.returncode == 75 and "waiting first" in long.stderr
        short = subprocess.run(["bash", str(GPU_LEASE), "1", "--max-min", "10", "--", "bash", str(GPU_LEASE), "status"], env=env,
                               capture_output=True, text=True)
        assert short.returncode == 0, short.stderr
        mine = [ln for ln in short.stdout.splitlines() if ln.startswith("1 held")]
        assert len(mine) == 1 and re.search(r" until=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ", mine[0])
        assert subprocess.run(["bash", str(GPU_LEASE), "1", "--", "true"], env=env, capture_output=True).returncode == 75
    finally:
        for p in (holder, waiter):
            p.stdin.close()
            p.wait()
    assert subprocess.run(["bash", str(GPU_LEASE), "1", "--max-min", "x", "--", "true"], env=env, capture_output=True).returncode == 2
    assert subprocess.run(["bash", str(GPU_LEASE), "1", "--max-min", "1", "--", "sh", "-c", "exit 3"], env=env).returncode == 3


@needs_flock
def test_a_waiter_stops_a_preemptible_lease_and_leaves_an_ordinary_one(tmp_path):
    env = {**gpu_env(tmp_path, count=1), "GPU_LEASE_POLL_S": "0.1"}
    pause = [sys.executable, "-c", "import signal; print('ready', flush=True); signal.pause()"]
    fill = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--preemptible", "--mem-gb", "0", "--", *pause], env=env,
                            stdout=subprocess.PIPE, text=True)
    assert fill.stdout.readline().strip() == "ready"
    held = subprocess.run(["bash", str(GPU_LEASE), "status"], env=env, capture_output=True, text=True).stdout
    assert "preempt=1" in held
    r = subprocess.run(["bash", str(GPU_LEASE), "1", "--wait", "--mem-gb", "0", "--", "echo", "got"], env=env, capture_output=True,
                       text=True)
    assert r.returncode == 0 and r.stdout.strip() == "got"
    assert fill.wait() != 0 and "SIGTERM GPU 0" in (tmp_path / "gl" / "preempted.log").read_text()
    ordinary = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--mem-gb", "0", "--", *pause], env=env, stdout=subprocess.PIPE,
                                text=True)
    try:
        assert ordinary.stdout.readline().strip() == "ready"
        r = subprocess.run(["bash", str(GPU_LEASE), "1", "--", "true"], env=env, capture_output=True, text=True)
        assert r.returncode == 75 and ordinary.poll() is None
    finally:
        ordinary.terminate()
        ordinary.wait()


def hold_agent_lock(gl: Path) -> int:
    """Stand in for `cluster agent --mode live`: hold agent.lock (the descriptor is not inherited by gpu-lease)."""
    gl.mkdir(exist_ok=True)
    fd = os.open(gl / "agent.lock", os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def queued(p: subprocess.Popen) -> str:
    """The wait file gpu-lease names on stderr once it has queued."""
    for line in p.stderr:
        if "queued as " in line:
            return line.rsplit("queued as ", 1)[1].strip()
    raise AssertionError("gpu-lease never queued")


@needs_flock
def test_under_the_agent_a_request_queues_at_once_with_its_flags_and_takes_exactly_the_gpus_it_is_granted(tmp_path):
    """GPU 0 is free, so gpu-lease's own rules would take it; under the agent it takes the granted GPU 2.  The wait file carries
    what the agent's planner reads (tools/cluster's nebius2.job_of)."""
    env = {**gpu_env(tmp_path), "GPU_LEASE_AGENT_POLL_S": "0.05"}
    gl = tmp_path / "gl"
    agent = hold_agent_lock(gl)
    try:
        p = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--wait", "--timed", "--max-min", "5", "--mem-gb", "0", "--", "sh", "-c",
                              'echo "$CUDA_VISIBLE_DEVICES"'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f = dict(kv.split("=", 1) for kv in (gl / queued(p)).read_text().split())
        assert f["pid"] == str(p.pid) and f["n"] == "1" and f["max"] == "5" and f["timed"] == "1" and "preempt" not in f
        (gl / f".grant.{p.pid}").write_text("2\n")
        os.replace(gl / f".grant.{p.pid}", gl / f"grant.{p.pid}")
        out, _ = p.communicate()
        assert p.returncode == 0 and out.strip() == "2"
        assert not list(gl.glob("wait.*")) and not list(gl.glob("grant.*"))
    finally:
        os.close(agent)


@needs_flock
def test_under_the_agent_gpu_lease_preempts_nothing_itself_and_falls_back_to_its_own_rules_once_the_agent_is_gone(tmp_path):
    """Its own rules stop the preemptible lease before queueing; under the agent it has queued and stopped nothing.  Once
    agent.lock is free it keeps its place, preempts by its own rules and runs."""
    env = {**gpu_env(tmp_path, count=1), "GPU_LEASE_POLL_S": "0.1", "GPU_LEASE_AGENT_POLL_S": "0.05"}
    gl = tmp_path / "gl"
    pause = [sys.executable, "-c", "import signal; print('ready', flush=True); signal.pause()"]
    fill = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--preemptible", "--mem-gb", "0", "--", *pause], env=env,
                            stdout=subprocess.PIPE, text=True)
    assert fill.stdout.readline().strip() == "ready"
    agent = hold_agent_lock(gl)
    p = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--wait", "--mem-gb", "0", "--", "echo", "got"], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        queued(p)
        assert not (gl / "preempted.log").exists() and fill.poll() is None
    finally:
        os.close(agent)
    out, _ = p.communicate()
    assert p.returncode == 0 and out.strip() == "got"
    assert fill.wait() != 0 and "SIGTERM GPU 0" in (gl / "preempted.log").read_text()
    assert not list(gl.glob("wait.*"))


@needs_flock
def test_under_the_agent_a_request_without_wait_gives_up_when_no_grant_comes_and_a_free_agent_lock_is_no_agent(tmp_path):
    env = {**gpu_env(tmp_path, count=2), "GPU_LEASE_AGENT_WAIT_S": "0"}
    gl = tmp_path / "gl"
    agent = hold_agent_lock(gl)
    try:
        r = subprocess.run(["bash", str(GPU_LEASE), "1", "--", "true"], env=env, capture_output=True, text=True)
        assert r.returncode == 75 and "agent granted no 1 GPU(s)" in r.stderr and not list(gl.glob("wait.*"))
    finally:
        os.close(agent)
    assert (gl / "agent.lock").exists()
    assert subprocess.run(["bash", str(GPU_LEASE), "1", "--", "true"], env=env, capture_output=True).returncode == 0


needs_taskset = pytest.mark.skipif(shutil.which("taskset") is None or shutil.which("flock") is None, reason="needs taskset and flock")


@needs_taskset
def test_check_slot_pins_a_check_to_a_free_slot_with_the_pods_python(tmp_path):
    slot = NEBIUS / "check_slot.sh"
    env = {**{k: v for k, v in os.environ.items() if k != "UV_PYTHON"}, "VY_CHECK_SLOTS": "0 0", "VY_CHECK_LOCKS": str(tmp_path),
           "VY_CHECK_WAIT": "0"}
    show = "echo $UV_PYTHON $(taskset -cp $$ | sed 's/.*: //')"
    first = subprocess.Popen(["bash", str(slot), "bash", "-c", f"{show}; read x"], env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert first.stdout.readline().split() == ["3.14.7", "0"]
        second = subprocess.run(["bash", str(slot), "bash", "-c", show], env=env, capture_output=True, text=True)
        assert second.returncode == 0 and "check-b" in second.stderr
        other = subprocess.Popen(["flock", str(tmp_path / "check-b.lock"), "bash", "-c", "echo ready; read x"], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True)
        try:
            assert other.stdout.readline().strip() == "ready"
            busy = subprocess.run(["bash", str(slot), "true"], env=env, capture_output=True, text=True)
            assert busy.returncode == 75 and "every check slot is busy" in busy.stderr
        finally:
            other.stdin.close()
            other.wait()
    finally:
        first.stdin.close()
        first.wait()
    slots_file = tmp_path / "check-slots"
    slots_file.write_text("0\n")
    r = subprocess.run(["bash", str(slot), "bash", "-c", show], env={**env, "VY_CHECK_SLOTS": "", "VY_CHECK_SLOTS_FILE": str(slots_file),
                       "UV_PYTHON": "3.12"}, capture_output=True, text=True)
    assert r.stdout.split() == ["3.12", "0"]
    path_file = tmp_path / "check-path"
    path_file.write_text("/opt/check-toolchain/bin\n")
    r = subprocess.run(["bash", str(slot), "bash", "-c", "echo $PATH"], env={**env, "VY_CHECK_PATH_FILE": str(path_file)},
                       capture_output=True, text=True)
    assert r.stdout.startswith("/opt/check-toolchain/bin:")



def test_a_short_check_takes_a_short_slot_while_every_train_slot_is_held(tmp_path):
    """--short: a lane's short check doesn't queue behind the trains. With both check slots held it still starts, in check-s1, on
    the check slots' CPUs, at nice 10, so the trains keep first claim on the cores."""
    slot = NEBIUS / "check_slot.sh"
    env = {**os.environ, "VY_CHECK_SLOTS": "0 0", "VY_CHECK_LOCKS": str(tmp_path), "VY_CHECK_WAIT": "0"}
    held = [subprocess.Popen(["flock", str(tmp_path / f"check-{s}.lock"), "bash", "-c", "echo ready; read x"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True) for s in ("a", "b")]
    try:
        assert [h.stdout.readline().strip() for h in held] == ["ready", "ready"]
        assert subprocess.run(["bash", str(slot), "true"], env=env, capture_output=True, text=True).returncode == 75
        r = subprocess.run(["bash", str(slot), "--short", "bash", "-c", "echo $(nice) $(taskset -cp $$ | sed 's/.*: //')"], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0 and "check-s1" in r.stderr and r.stdout.split() == ["10", "0"], (r.stdout, r.stderr)
    finally:
        for h in held:
            h.stdin.close()
            h.wait()

# ---- launch.sh ---------------------------------------------------------------------------------------------------------------------------

def launch(*args: str, **env: str) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items() if not k.startswith(("NEBIUS_", "RUNPOD_"))}
    return subprocess.run(["bash", str(NEBIUS / "launch.sh"), *args], env={**base, **env}, capture_output=True, text=True)


def test_dry_run_renders_the_cloud_init_and_vm_spec_without_credentials():
    pub = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIdummy research@test"
    r = launch("--dry-run", NEBIUS_SSH_PUBKEY=pub)
    assert r.returncode == 0, r.stderr
    ud, spec = r.stdout.split("# instance create request (ids are placeholders)\n")
    assert ud.splitlines()[1] == "#cloud-config" and f"      - {pub}" in ud and "PRIVATE KEY" not in r.stdout
    embedded = {m[0]: gzip.decompress(base64.b64decode(m[1]))
                for m in re.findall(r"  - path: (\S+)\n    permissions: '\d+'\n    encoding: gz\+b64\n    content: (\S+)", ud)}
    assert embedded["/usr/local/sbin/vy-setup"] == (NEBIUS / "vm_setup.sh").read_bytes()
    assert embedded["/usr/local/sbin/vy-nebius-stop"] == (NEBIUS / "vm_stop.sh").read_bytes()
    assert embedded["/usr/local/share/vy/lease.sh"] == L.LEASE_SH.read_bytes()
    assert embedded["/usr/local/bin/gpu-lease"] == GPU_LEASE.read_bytes()
    assert b"VY_USER=research\n" in embedded["/etc/vy/vy.env"]
    s = json.loads(spec)["spec"]
    assert s["resources"] == {"platform": "gpu-rtx6000-a", "preset": "8gpu-192vcpu-1744gb"}
    nic = s["network_interfaces"][0]
    assert nic["public_ip_address"] == {"static": True} and nic["security_groups"] and s["secondary_disks"][0]["device_id"] == "data"


def test_launch_refuses_to_spend_without_yes_and_a_lease():
    r = launch("--max-hours", "4")
    assert r.returncode == 1 and "pass --yes" in r.stderr
    r = launch("--yes")
    assert r.returncode == 1 and "--max-hours H" in r.stderr
    r = launch("--yes", "--max-hours", "4", NEBIUS_SSH_PUBKEY="ssh-ed25519 dummy", NEBIUS_BIN="/bin/true")
    assert r.returncode == 1 and "NEBIUS_SA_ID is not set" in r.stderr


# ---- the weight list -----------------------------------------------------------------------------------------------------------------------

def test_the_weight_list_pins_the_integrations_revisions():
    import importlib.util

    spec = importlib.util.spec_from_file_location("nebius_prefetch", NEBIUS / "prefetch.py")
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)
    rows = pf.rows()
    assert all(re.fullmatch(r"[0-9a-f]{40}", r["revision"]) for r in rows)
    sweep = [r for r in rows if r["group"] == "bf16-sweep"]
    assert len(sweep) == 16 and {r["format"] for r in sweep} == {"bf16"}
    pinned = {(c["repo"], c["revision"]) for c in json.loads(CHECKPOINTS.read_text())["checkpoints"]}
    assert [r["repo"] for r in rows if r["group"] != "frontier" and (r["repo"], r["revision"]) not in pinned] == []
    assert {r["role"] for r in rows if r["group"] == "frontier"} == {"DSV3_FP8", "DSV3_FP4", "K2_FP8", "K2_FP4"}
    assert sum(r["gb"] for r in rows) < 5022 * 2**30 / 1e9 * 0.6        # the cache leaves the disk's other 40% to /workspace


@needs_flock
def test_gpu_lease_hands_out_only_the_direct_run_gpus_the_cluster_leaves_it(tmp_path):
    allowed = tmp_path / "direct-gpus"
    env = {**gpu_env(tmp_path, count=8), "GPU_LEASE_ALLOWED_FILE": str(allowed)}
    allowed.write_text("4,5,6,7\n")
    inner = f'echo "outer=$CUDA_VISIBLE_DEVICES"; bash {GPU_LEASE} 3 -- true; echo "short=$?"'
    r = subprocess.run(["bash", str(GPU_LEASE), "2", "--", "sh", "-c", inner], env=env, capture_output=True, text=True)
    assert r.stdout.splitlines() == ["outer=4,5", "short=75"], r.stderr
    assert subprocess.run(["bash", str(GPU_LEASE), "5", "--", "true"], env=env, capture_output=True).returncode == 2
    status = subprocess.run(["bash", str(GPU_LEASE), "status"], env=env, capture_output=True, text=True).stdout.splitlines()
    assert [ln.split()[0] for ln in status if not ln.startswith("#")] == ["4", "5", "6", "7"]
    allowed.write_text("none\n")
    r = subprocess.run(["bash", str(GPU_LEASE), "1", "--", "true"], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "submit through the cluster's queues" in r.stderr


def test_a_direct_run_sees_no_gpu_once_kueue_owns_every_gpu(tmp_path, monkeypatch):
    """After the cutover (`/etc/vy/direct-gpus` = none) the runner blanks CUDA_VISIBLE_DEVICES for every direct run, even one whose
    --env names a GPU; before it (a direct set) or on a machine without the file, the run's environment is left alone."""
    split = tmp_path / "direct-gpus"
    monkeypatch.setenv("GPU_LEASE_ALLOWED_FILE", str(split))
    req = {"env": {"CUDA_VISIBLE_DEVICES": "0"}}
    for text, want in (("none\n", ""), ("4,5,6,7\n", "0")):
        split.write_text(text)
        rec: dict = {}
        assert remote._runner_env(req, tmp_path / "runs", None, rec)["CUDA_VISIBLE_DEVICES"] == want
        assert ("gpus" in rec) == (want == "")
    split.unlink()
    assert remote._runner_env(req, tmp_path / "runs", None, {})["CUDA_VISIBLE_DEVICES"] == "0"


def test_a_direct_run_starts_on_the_hosts_direct_cpus_and_may_pin_itself_elsewhere(tmp_path, monkeypatch):
    """With /etc/vy/direct-cpus (the Nebius CPU map's non-benchmark CPUs) a direct run's runner, and so its workload, starts on those
    CPUs, keeping the pinned benchmarks' cores clean; a command's own taskset still moves it. No file: the machine's own affinity."""
    mine = sorted(os.sched_getaffinity(0))
    cpus_file = tmp_path / "direct-cpus"
    monkeypatch.setenv("DIRECT_CPUS_FILE", str(cpus_file))
    assert remote.direct_cpus() == set()
    cpus_file.write_text(f"{mine[0]}\n")
    assert remote.direct_cpus() == {mine[0]} and remote.format_cpus({0, 1, 2, 5, 7, 8}) == "0-2,5,7-8"
    cpus_file.write_text("100000-100001\n")                   # none of this machine's: no default rather than an empty mask
    assert remote.direct_cpus() == set()


def test_fill_runner_gives_a_tp2_commit_two_gpus_on_one_numa_node_and_commits_a_longer_cap(tmp_path, monkeypatch):
    """Top-level's node-2-first order (1 Oct 12:14 AM PDT): a Commit guest may take 2 GPUs (a TP2 row), from one NUMA node when it
    can, through `gpu-lease 2 --on a,b`; it runs up to FILL_COMMIT_MAX_MIN (a Gemma-2 Commit's CPU phases outlast 30 min) while
    other GPU fill stays at 30 and gpus 1; and a TP2 Commit short of GPUs reclaims as many as it lacks."""
    gl, fill = tmp_path / "gl", tmp_path / "fill"
    for d in ("queue", "running", "logs"):
        (fill / d).mkdir(parents=True)
    gl.mkdir()
    monkeypatch.setenv("GPU_LEASE_DIR", str(gl))
    monkeypatch.setenv("FILL_DIR", str(fill))
    spec = importlib.util.spec_from_file_location("fill_runner_tp2", FILL_RUNNER)
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    for name, head, want in (("verity-commit-a.sh", "gpus=2 max_min=90", (2, 90)), ("verity-commit-b.sh", "gpus=4 max_min=500", (2, 120)),
                             ("pn2g-x.sh", "gpus=2 max_min=90 project=verity", (1, 30))):
        (tmp_path / name).write_text(f"# fill: owner=t {head}\n")
        h = fr.header(tmp_path / name)
        assert (h["gpus"], h["max_min"]) == want, name
    assert fr.pick_gpus([0, 3, 5], None, 2, 8) == [0, 3] and fr.pick_gpus([3, 4, 6, 7], None, 2, 8) == [6, 7]
    assert fr.pick_gpus([3, 5], None, 2, 8) == [3, 5] and fr.pick_gpus([5], None, 2, 8) is None
    held: dict[int, str] = {}
    started, stopped = [], []
    monkeypatch.setattr(fr, "reap", lambda: None)
    monkeypatch.setattr(fr, "verity_pool", lambda queued: None)
    monkeypatch.setattr(fr, "lease", lambda i: held.get(i))
    monkeypatch.setattr(fr, "waiters", lambda: [])
    monkeypatch.setattr(fr, "start", lambda p, h, backfill=False: started.append((p.name, h["on"])) or p.unlink())
    monkeypatch.setattr(fr, "stop", lambda k, why, requeue=True: stopped.append(k) or fr.running[k].update(stopping=True, why=why))
    for gpu in (0, 1, 2, 5):
        held[gpu] = "pid=1 run=- who=x since=2026-10-01T00:00:00Z cmd=x"
    (fill / "queue" / "verity-commit-tp2.sh").write_text("# fill: owner=o gpus=2 max_min=90 project=verity prio=20\n")
    (fill / "queue" / "verity-commit-tp1.sh").write_text("# fill: owner=o gpus=1 max_min=90 project=verity prio=20\n")
    os.utime(fill / "queue" / "verity-commit-tp2.sh", (1, 1))
    fr.tick(8)
    assert started == [("verity-commit-tp2.sh", "6,7"), ("verity-commit-tp1.sh", "4")]
    started.clear()
    held.update({i: "pid=2 run=- who=x since=2026-10-01T00:00:00Z cmd=x" for i in (3, 4, 6, 7)})
    held.pop(5)
    for k, gpu in (("pn2g-a.sh", 0), ("pn2g-b.sh", 1)):
        fr.running[k] = {"t0": gpu, "paused": False, "started": "-", "h": {"gpus": 1, "project": "verity", "owner": "o", "max_min": 10**9}}
    (fill / "queue" / "verity-commit-tp2b.sh").write_text("# fill: owner=o gpus=2 max_min=90 project=verity prio=20\n")
    fr.tick(8)
    assert started == [] and stopped == ["pn2g-b.sh"]          # one GPU free, one more reclaimed: the newest proofs fill's
    fr.tick(8)
    assert stopped == ["pn2g-b.sh"]


def test_fill_runner_lets_a_verity_cpu_job_run_a_whole_build_and_freezes_its_scope(tmp_path, monkeypatch):
    """A Verity CPU guest (a vLLM Build, 2-50 min and some 1-4 h) may set max_min up to 360; every other job stays capped at 30.
    Its pause and stop go through its systemd scope, since `research run` starts the workload as its own process group."""
    monkeypatch.setenv("FILL_DIR", str(tmp_path / "fill"))
    spec = importlib.util.spec_from_file_location("fill_runner_verity", FILL_RUNNER)
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    job = tmp_path / "job.sh"
    for head, want in (("gpus=0 max_min=300 project=verity", 300), ("gpus=0 max_min=900 project=verity", 360),
                       ("gpus=1 max_min=300 project=verity", 30), ("gpus=0 max_min=300 project=pous", 30)):
        job.write_text(f"#!/usr/bin/env bash\n# fill: owner=t {head}\n")
        assert fr.header(job)["max_min"] == want, head
    monkeypatch.setattr(fr, "MAX_MIN_EXCEPTIONS", {"gate.sh": 60})
    gate = tmp_path / "gate.sh"
    gate.write_text("# fill: owner=t gpus=1 max_min=90 project=verity\n")
    assert fr.header(gate)["max_min"] == 60
    calls = []
    monkeypatch.setattr(fr.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    assert fr.scope({"unit": "fill-verity-1"}, "freeze") and calls == [["systemctl", "--user", "freeze", "fill-verity-1.scope"]]
    assert not fr.scope({"unit": None}, "freeze") and len(calls) == 1


def test_the_verity_pool_lends_idle_slots_to_pous_cpu_jobs_and_takes_them_back(tmp_path, monkeypatch):
    """With no Verity job waiting, queued pous CPU jobs start in the Verity pool's free slots (its CPUs, a scope, its memory caps);
    a Verity job that finds no slot requeues the newest lent job, and nothing more is lent while it waits."""
    monkeypatch.setenv("FILL_DIR", str(tmp_path / "fill"))
    monkeypatch.setenv("FILL_VERITY_CPU_SLOTS", "2")
    spec = importlib.util.spec_from_file_location("fill_runner_lend", FILL_RUNNER)
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    monkeypatch.setattr(fr, "disk_pct", lambda path: 0.0)
    started, stopped = [], []
    monkeypatch.setattr(fr, "start", lambda p, h, backfill=False: started.append((p.name, h)) or fr.running.__setitem__(
        p.name, {"h": h, "t0": len(fr.running), "pid": 0}))
    monkeypatch.setattr(fr, "stop", lambda name, why, requeue=True: stopped.append(name))
    q = tmp_path / "q"
    q.mkdir()

    def job(name, head):
        p = q / name
        p.write_text(f"# fill: owner=o {head}\n")
        return (p, fr.header(p), 0.0)

    fr.verity_pool([job("v1.sh", "gpus=0 project=verity mem_gb=64"), job("p1.sh", "gpus=0 mem_gb=500"), job("p2.sh", "gpus=0")])
    assert [(n, h.get("lent"), h["mem_gb"]) for n, h in started] == [("v1.sh", None, 64), ("p1.sh", True, 256)]
    started.clear()
    fr.verity_pool([job("v2.sh", "gpus=0 project=verity"), job("p2.sh", "gpus=0")])
    assert stopped == ["p1.sh"] and started == []
    fr.running.pop("p1.sh")
    fr.verity_pool([job("v2.sh", "gpus=0 project=verity")])
    assert [n for n, _ in started] == ["v2.sh"]


def test_fill_runner_files_a_job_by_its_own_exit_status_across_a_restart_and_never_requeues_an_unknown_exit(tmp_path, monkeypatch):
    """A job that finishes while a restarted runner has adopted it goes to done/, not back to the queue, where its rerun deleted
    its result (pous-climb-a3, 1 Oct 08:50Z); exit 0 is done even when a stop raced it; and a job whose exit the runner can't learn
    is held, never rerun, while one that asks for more chunks (99) is still requeued."""
    fill = tmp_path / "fill"
    for d in ("queue", "running", "done", "failed", "logs", "held-unknown-exit"):
        (fill / d).mkdir(parents=True)
    monkeypatch.setenv("FILL_DIR", str(fill))
    monkeypatch.setenv("FILL_CPU_SET", str(sorted(os.sched_getaffinity(0))[0]))
    spec = importlib.util.spec_from_file_location("fill_runner_exit", FILL_RUNNER)
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    monkeypatch.setattr(fr.shutil, "which", lambda name: None)
    out = tmp_path / "out"

    def run(name: str, body: str) -> dict:
        p = fill / "queue" / name
        p.write_text(f"#!/usr/bin/env bash\n# fill: owner=o gpus=0\n{body}\n")
        fr.start(p, fr.header(p))
        j = fr.running[name]
        j["proc"].wait()
        return j

    def restart() -> None:
        fr.running.clear()
        fr.adopt()
        fr.reap()

    run("a.sh", f"rm -rf {out}; mkdir -p {out}; echo result > {out}/r; exit 0")
    restart()
    assert (fill / "done" / "a.sh").exists() and not (fill / "queue" / "a.sh").exists()
    assert (out / "r").read_text() == "result\n" and not list((fill / "running").iterdir())

    run("more.sh", "exit 99")
    restart()
    assert (fill / "queue" / "more.sh").exists()

    run("raced.sh", "exit 0")["stopping"] = fr.running["raced.sh"]["requeue"] = True
    fr.reap()
    assert (fill / "done" / "raced.sh").exists()

    gone = subprocess.Popen(["true"], start_new_session=True)
    gone.wait()
    (fill / "running" / "legacy.sh").write_text("# fill: owner=o gpus=0\n")
    (fill / "running" / ".legacy.sh.pgid").write_text(f"{gone.pid}\n")
    (fill / "running" / "orphan.sh").write_text("# fill: owner=o gpus=0\n")
    restart()
    assert sorted(p.name for p in (fill / "held-unknown-exit").iterdir()) == ["legacy.sh", "orphan.sh"]
    assert not list((fill / "queue").glob("legacy.sh")) and not list((fill / "queue").glob("orphan.sh"))
    kinds = [json.loads(l)["event"] for l in (fill / "events.jsonl").read_text().splitlines()]
    assert kinds.count("unknown-exit") == 2 and "adopted-exit" not in kinds


def _pool_n1(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    spec = importlib.util.spec_from_file_location("pool_n1_under_test", NEBIUS / "pool_n1.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_pool_n1_covers_node_2s_timed_window_pause_in_the_merged_file_for_at_most_3_hours(monkeypatch, tmp_path):
    m = _pool_n1(monkeypatch)
    merged = tmp_path / "infra-pool.json"
    t0 = 1_800_000_000.0
    merged.write_text(json.dumps({"generated_at": m.iso(t0), "nodes": {"n1": {"old": 1}, "n2": {"gpu_busy_1h": 0.5}}}))
    assert not m.cover_pause(merged, {"new": 1}, t0 + 300)
    assert m.cover_pause(merged, {"new": 1}, t0 + 900)
    doc = json.loads(merged.read_text())
    assert doc["generated_at"] == m.iso(t0 + 900) and doc["n2_generated_at"] == m.iso(t0)
    assert doc["nodes"]["n1"] == {"new": 1} and doc["nodes"]["n2"]["paused_since"] == m.iso(t0) and "gpu_busy_1h" not in doc["nodes"]["n2"]
    assert m.cover_pause(merged, {"new": 2}, t0 + 3 * 3600 - 60)
    assert not m.cover_pause(merged, {"new": 3}, t0 + 3 * 3600 + 60)
    assert json.loads(merged.read_text())["nodes"]["n1"] == {"new": 2}
    merged.write_text(json.dumps({"generated_at": m.iso(t0 + 4 * 3600), "nodes": {"n2": {"gpu_busy_1h": 0.4}}}))
    assert not m.cover_pause(merged, {"new": 4}, t0 + 4 * 3600 + 120)


def test_pool_n1_flags_an_idle_admitted_gpu_and_an_unleased_one_and_counts_waiting_workloads(monkeypatch):
    m = _pool_n1(monkeypatch)
    t = datetime.datetime(2026, 9, 30, 22, 0, tzinfo=datetime.timezone.utc).timestamp()

    def wl(owner, queue, admitted, gpus=1, kind="Job"):
        conds = [{"type": "QuotaReserved", "status": "True" if admitted else "False"}]
        return {"metadata": {"ownerReferences": [{"kind": kind, "name": owner}]}, "status": {"conditions": conds},
                "spec": {"queueName": queue, "podSets": [{"count": 1, "template": {"spec": {"containers": [
                    {"resources": {"requests": {"nvidia.com/gpu": str(gpus)} if gpus else {}}}]}}}]}}

    pod = lambda name, job, start: {"metadata": {"name": name, "labels": {"job-name": job}}, "status": {"startTime": start}}
    obs = {"t": t, "sampled_at": t - 10, "samples_5m": 20, "samples_1h": 240,
           "busy_5m": {0: 0.1, 1: 1.0, 2: 0.0, 3: 0.5, 4: 0.4}, "busy_1h": {0: 0.2, 1: 1.0, 2: 0.0, 3: 0.5, 4: 0.3},
           "util_5m": {0: 2.0, 1: 80.0, 2: 0.0, 3: 30.0, 4: 0.0}, "fb_mib": {0: 9000, 1: 20000, 2: 0, 3: 5000, 4: 50000},
           "held_busy_1h": 540, "all_busy_1h": 600, "cpu_busy_5m": 0.31,
           "pod_of_gpu": {0: "c-idle-x", 1: "c-busy-x", 2: None, 3: "rogue", 4: "done-x"},
           "pods": [pod("c-idle-x", "nd-vllm-commit-1", "2026-09-30T21:40:00Z"), pod("c-busy-x", "nd-po-1", "2026-09-30T21:40:00Z"),
                    pod("rogue", "rogue-job", "2026-09-30T21:00:00Z"),
                    {**pod("done-x", "done-job", "2026-09-30T21:00:00Z"), "status": {"phase": "Succeeded"}}],
           "jobs": [{"metadata": {"name": "nd-vllm-commit-1", "annotations": {"verity.dev/template": "config-run@9c6a"}}}],
           "workloads": [wl("nd-vllm-commit-1", "deployments-gpu", True), wl("nd-po-1", "pous-overflow", True),
                         wl("wait-g", "deployments-gpu", False), wl("wait-c", "deployments-cpu", False, gpus=0)]}
    n1 = m.build({**obs, "busy_5m": {**obs["busy_5m"], 5: 0.0, 6: 1.0}, "util_5m": {**obs["util_5m"], 5: 0.0, 6: 90.0},
                  "pod_of_gpu": {**obs["pod_of_gpu"], 5: "gpu-pool-1-a", 6: "gpu-pool-2-b"},
                  "lease_pool": {5: {"index": 5, "holder": "gpu-pool-1"},
                                 6: {"index": 6, "holder": "gpu-pool-2", "lease": {"who": "backend-sweep-2", "since": "2026-09-30T21:30:00Z",
                                                                                   "pod": "nd-bs-x"}}}})
    assert [(g["index"], g["holder"], g["class"]) for g in n1["gpus"]][5:] == [(5, "pool:free", "free"), (6, "lease:backend-sweep-2", "verity")]
    n1 = m.build(obs)
    assert [(g["index"], g["holder"], g["class"]) for g in n1["gpus"]] == [
        (0, "kueue:nd-vllm-commit-1", "verity"), (1, "kueue:nd-po-1", "pous"), (2, None, "free"), (3, "rogue", "direct"), (4, None, "free")]
    assert (n1["gpu_busy_5m"], n1["useful_share_1h"], n1["waiting"], n1["waiting_gpu"], n1["waiting_cpu"]) == (0.4, 0.9, 2, 1, 1)
    assert [(x["kind"], x["gpu"]) for x in n1["monitors"]] == [("gpu-idle-in-lease", 0), ("gpu-unleased", 3)]
    assert n1["monitors"][0]["job_kind"] == "config-run@9c6a" and n1["monitors"][0]["owner"] == "deployments-gpu"


def test_pool_n1_splits_leased_gpu_seconds_by_hour_and_counts_them_delivered_only_on_a_passing_attempt(tmp_path, monkeypatch):
    db = tmp_path / "catalog.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("create table attempts(id text, host text, state text, rc integer, validation text, start_utc text)")
        c.execute("create table attempt_outputs(attempt text, name text, artifact text)")
        c.executemany("insert into attempts values (?, ?, ?, ?, ?, ?)", [
            ("r1", "commit-ok-ab1cd", "done", 0, "passed", "2026-09-30T20:40:00Z"),
            ("r2", "commit-bad-ab1cd", "done", 0, "failed", "2026-09-30T21:10:00Z"),
            ("r3", "chunk-ab1cd", "done", 0, "not_run", "2026-09-30T21:20:00Z"),
            ("r4", "filler-ab1cd", "done", 0, "passed", "2026-09-30T21:20:00Z")])
        c.execute("insert into attempt_outputs values ('r3', 'result', 'art:x')")
    m = _pool_n1(monkeypatch, POOL_N1_STORE_DB=db)
    t = datetime.datetime(2026, 9, 30, 22, 30, tzinfo=datetime.timezone.utc).timestamp()

    def wl(job, admitted, finished=None, gpus=1):
        conds = [{"type": "QuotaReserved", "status": "True", "lastTransitionTime": admitted}]
        conds += [{"type": "Finished", "status": "True", "lastTransitionTime": finished}] if finished else []
        return {"metadata": {"ownerReferences": [{"kind": "Job", "name": job}]}, "status": {"conditions": conds},
                "spec": {"podSets": [{"count": 1, "template": {"spec": {"containers": [
                    {"resources": {"requests": {"nvidia.com/gpu": str(gpus)} if gpus else {}}}]}}}]}}

    obs = {"t": t, "attempts": m.attempts_by_job("2026-09-30T00:00:00Z"),
           "jobs": [{"metadata": {"name": "filler", "labels": {"verity.dev/filler": "true"}}}],
           "workloads": [wl("commit-ok", "2026-09-30T20:30:00Z", "2026-09-30T21:30:00Z"),       # 30 + 30 min, delivered
                         wl("commit-bad", "2026-09-30T21:00:00Z", "2026-09-30T21:30:00Z", gpus=2),
                         wl("chunk", "2026-09-30T21:15:00Z", "2026-09-30T21:45:00Z"),          # no validation, rc 0, an output
                         wl("filler", "2026-09-30T21:00:00Z", "2026-09-30T21:10:00Z"),
                         wl("running", "2026-09-30T22:00:00Z"),                                # no Attempt yet: leased only
                         wl("cpu-build", "2026-09-30T21:00:00Z", "2026-09-30T22:00:00Z", gpus=0)]}
    hours = m.delivery(obs)
    assert [h["hour"] for h in hours] == ["2026-09-30T19:00:00Z", "2026-09-30T20:00:00Z", "2026-09-30T21:00:00Z", "2026-09-30T22:00:00Z"]
    assert [(h["leased_gpu_s"], h["delivered_gpu_s"], h["provisional"]) for h in hours] == [
        (0, 0, False), (1800, 1800, False), (1800 + 3600 + 1800 + 600, 1800 + 1800, True), (1800, 0, True)]
    assert hours[0]["delivered_share"] is None and hours[2]["delivered_share"] == round(3600 / 7800, 4)


def test_pool_n1_reports_template_drift_against_the_pushed_infra_nebius_ref(tmp_path, monkeypatch):
    src, bare, sky = tmp_path / "src", tmp_path / "bare.git", tmp_path / "sky"
    run = lambda *a, cwd=None: subprocess.run(a, cwd=cwd, check=True, capture_output=True)
    (src / "tools/research/src/research/pods/nebius/sky/jobs").mkdir(parents=True)
    (src / "tools/research/src/research/pods/nebius/sky/kueue.yaml").write_text("a: 1\n")
    (src / "tools/research/src/research/pods/nebius/sky/jobs/config-run.yaml").write_text("run: new\n")
    run("git", "init", "-q", str(src))
    run("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "x", "--allow-empty", cwd=src)
    run("git", "add", "-A", cwd=src)
    run("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "sky", cwd=src)
    run("git", "init", "-q", "--bare", str(bare))
    run("git", "push", "-q", str(bare), "HEAD:refs/kueue-fold/infra-nebius", cwd=src)
    (sky / "jobs").mkdir(parents=True)
    (sky / "kueue.yaml").write_text("a: 1\n")
    (sky / "jobs" / "config-run.yaml").write_text("run: old\n")
    (sky / "jobs" / "config-run.yaml.bak-1").write_text("x\n")
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"commit": "abc123", "pushed_at": "2026-09-30T21:50:00Z"}))
    m = _pool_n1(monkeypatch, POOL_N1_GIT=bare, POOL_N1_REF_JSON=ref, DISPATCH_SKY=sky)
    d = m.template_drift()
    assert d["kind"] == "template-drift" and d["files"] == ["changed jobs/config-run.yaml"]
    (sky / "jobs" / "config-run.yaml").write_text("run: new\n")
    assert m.template_drift() == {}


def test_publish_pool_reports_busy_useful_holders_and_ready_gpu_h_and_stays_quiet_in_a_timed_window(tmp_path, monkeypatch):
    util, fill = tmp_path / "util", tmp_path / "fill"
    util.mkdir()
    for d in ("queue", "running", "done"):
        (fill / d).mkdir(parents=True)
    (fill / "status.txt").write_text("2026-09-30T21:00:00Z free GPUs 0/2, kept free -, waiters -, timed False, window waiting False\n")
    (fill / "running" / "work-a.sh").write_text("# fill: owner=bc-a gpus=1 max_min=20 project=pous\n")
    (fill / "running" / "pad-b.sh").write_text("# fill: owner=bc-b gpus=1 max_min=20 project=pous filler=idle-gpu\n")
    (fill / "queue" / "q1.sh").write_text("# fill: owner=bc-a gpus=1 max_min=30\n")
    (fill / "queue" / "q2.sh").write_text("# fill: owner=bc-a gpus=0 max_min=30\n")
    lease = "pid=1 run=- who=fill:{} since=x cmd=nice -n 19 taskset -c 96-127 bash /workspace/pouw/fill/running/{}"
    now = datetime.datetime(2026, 9, 30, 21, 0, 0, tzinfo=datetime.timezone.utc)
    with open(util / "2026-09-30.jsonl", "w") as f:
        for s in range(0, 3600, 12):
            t = now - datetime.timedelta(seconds=3600 - s)
            recent = s >= 3300
            f.write(json.dumps({"t": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "cpu_busy": 0.5 if recent else 0.1, "gpus": [
                {"i": 0, "util": 90, "smact": 0.5, "lease": lease.format("bc-a", "work-a")},
                {"i": 1, "util": 0 if recent else 90, "smact": 0.0 if recent else 0.4, "lease": lease.format("bc-b", "pad-b")},
            ]}) + "\n")
    monkeypatch.setenv("UTIL_OUT_DIR", str(util))
    monkeypatch.setenv("FILL_DIR", str(fill))
    spec = importlib.util.spec_from_file_location("publish_pool_under_test", NEBIUS / "publish_pool.py")
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    n2 = pp.build(now)["nodes"]["n2"]
    assert n2["gpu_busy_5m"] == 0.5 and 0.9 < n2["gpu_busy_1h"] < 1
    assert 0.5 < n2["useful_share_1h"] < 0.6
    assert n2["cpu_busy_5m"] == 0.5
    assert [(g["index"], g["holder"], g["class"], g["busy_5m"]) for g in n2["gpus"]] == [
        (0, "fill:bc-a", "pous", 1.0), (1, "fill:bc-b", "filler", 0.0)]
    assert (n2["ready_gpu_h"], n2["waiting"], n2["waiting_gpu"], n2["waiting_cpu"]) == (0.5, 2, 1, 1)
    out = tmp_path / "pool.json"
    (fill / "status.txt").write_text("2026-09-30T21:00:00Z free GPUs 0/2, kept free -, waiters -, timed True, window waiting False\n")
    monkeypatch.setattr(sys, "argv", ["publish_pool.py", "--out", str(out), "--no-push"])
    assert pp.main() == 0 and not out.exists()


def test_publish_pool_flags_an_idle_lease_and_an_unleased_gpu_process_once_and_ranks_kinds_by_idle_gpu_h(tmp_path, monkeypatch):
    util, fill, proc = tmp_path / "util", tmp_path / "fill", tmp_path / "proc"
    for d in (util, fill / "queue", fill / "running", fill / "done"):
        d.mkdir(parents=True)
    (fill / "done" / "kt-e70b-fold-pc.sh").write_text("# fill: owner=bc-a gpus=1 max_min=20 kind=pouw-kt-gpu1\n")
    (fill / "done" / "pad-x.sh").write_text("# fill: owner=bc-b gpus=1 max_min=20 filler=idle-gpu\n")
    (fill / "queue" / "q.sh").write_text("# fill: owner=bc-a gpus=1 max_min=30\n")
    for pid, cg, ppid in ((11, "0::/user.slice/gpu-lease-10.scope\n", 10), (21, "0::/user.slice/tmux.scope\n", 20),
                          (20, "0::/user.slice/tmux.scope\n", 1)):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "cgroup").write_text(cg)
        (proc / str(pid) / "stat").write_text(f"{pid} (python3) S {ppid} 0 0\n")
    now = datetime.datetime(2026, 9, 30, 21, 0, 0, tzinfo=datetime.timezone.utc)
    idle = "pid=10 run=- who=fill:bc-a since=2026-09-30T20:50:00Z scope=gpu-lease-10 cmd=bash /workspace/pouw/fill/running/kt-e70b-fold-pc"
    with open(util / "2026-09-30.jsonl", "w") as f:
        for s in range(0, 300, 12):
            t = (now - datetime.timedelta(seconds=300 - s)).strftime("%Y-%m-%dT%H:%M:%SZ")
            f.write(json.dumps({"t": t, "cpu_busy": 0.2, "gpus": [
                {"i": 0, "util": 12 if s < 24 else 3, "smact": 0.02, "lease": idle, "procs": [{"pid": 11, "mib": 900, "cmd": "python3 kt.py"}]},
                {"i": 1, "util": 60, "smact": 0.3, "lease": None, "procs": [{"pid": 21, "mib": 4000, "cmd": "python3 stray.py"}]},
            ]}) + "\n")
    usage = tmp_path / "lease-usage.jsonl"
    rec = {"schema": "gpu-lease/usage/v1", "end": "2026-09-30T20:30:00Z"}
    usage.write_text("".join(json.dumps(r) + "\n" for r in [
        {**rec, "who": "fill:bc-a", "since": "2026-09-30T20:00:00Z", "rc": 0, "cmd": "bash /workspace/pouw/fill/running/kt-e70b-fold-pc.sh",
         "gpus": [{"held_s": 3600, "busy_s": 1800}]},
        {**rec, "who": "fill:bc-b", "cmd": "bash /workspace/pouw/fill/running/pad-x.sh", "gpus": [{"held_s": 360, "busy_s": 360}]},
        {**rec, "who": "session:bc-c", "cmd": "python3 bench.py", "gpus": [{"held_s": 720, "busy_s": 720}]},
        {**rec, "who": "bc-d", "timed": True, "cmd": "python3 audit.py", "gpus": [{"held_s": 360, "busy_s": 360}]},
        {**rec, "end": "2026-09-29T20:00:00Z", "who": "fill:bc-a", "cmd": "x", "gpus": [{"held_s": 9999, "busy_s": 0}]},
    ]))
    for k, v in {"UTIL_OUT_DIR": util, "FILL_DIR": fill, "POOL_PROC": proc, "LEASE_USAGE": usage,
                 "POOL_ALERTS": tmp_path / "alerts.jsonl", "POOL_STATE": tmp_path / "state.json"}.items():
        monkeypatch.setenv(k, str(v))
    spec = importlib.util.spec_from_file_location("publish_pool_monitors_under_test", NEBIUS / "publish_pool.py")
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    n2 = pp.build(now)["nodes"]["n2"]
    assert [(m["kind"], m["gpu"]) for m in n2["monitors"]] == [("gpu-idle-in-lease", 0), ("gpu-unleased", 1)]
    assert n2["monitors"][0]["job_kind"] == "pouw-kt-gpu1" and n2["monitors"][0]["owner"] == "bc-a"
    assert n2["unlabeled_waiting"] == 1
    assert [(k["kind"], k["leased_gpu_h"], k["useful_gpu_h"], k["idle_gpu_h"], k["efficiency"]) for k in n2["kinds"]] == [
        ("pouw-kt-gpu1", 1.0, 0.5, 0.5, 0.5), ("unlabeled:pad-x", 0.1, 0.0, 0.1, 0.0), ("direct:session:bc-c", 0.2, 0.2, 0.0, 1.0),
        ("timed:bc-d", 0.1, 0.1, 0.0, 1.0)]
    d = n2["delivered_by_hour"]
    assert [r["hour"] for r in d] == ["2026-09-30T18:00:00Z", "2026-09-30T19:00:00Z", "2026-09-30T20:00:00Z", "2026-09-30T21:00:00Z"]
    assert (d[2]["leased_gpu_s"], d[2]["delivered_gpu_s"], d[2]["delivered_share"], d[2]["provisional"]) == (2400, 1800, 0.75, True)
    assert d[3]["leased_gpu_s"] == 0 and d[3]["delivered_share"] is None and d[0]["delivered_share"] is None
    pp.alert(n2["monitors"], now)
    pp.alert(n2["monitors"], now + datetime.timedelta(minutes=5))
    assert len((tmp_path / "alerts.jsonl").read_text().splitlines()) == 2
    pp.alert(n2["monitors"], now + datetime.timedelta(minutes=31))
    assert len((tmp_path / "alerts.jsonl").read_text().splitlines()) == 4


@pytest.mark.skipif(shutil.which("bash") is None, reason="n2_commit.sh is bash")
def test_n2_commit_refuses_before_touching_a_node(tmp_path):
    """n2_commit.sh's item checks run before any ssh: a bad key, a non-config-run item, a missing field or a row wider than TP2 (a
    node-2 guest has one or two GPUs) exits 2; a TP1 or TP2 config-run item passes."""
    script = NEBIUS / "n2_commit.sh"
    env = {"ENV": {"ROW": "llama32-1b__bf16__rtxpro6000__tp1__b8__i256__o32__mixed__greedy__bi-eager", "ROLE": "LLAMA32_1B",
                   "REPO": "unsloth/Llama-3.2-1B", "REVISION": "9535bd9b"}}

    def check(key: str, **item) -> int:
        p = tmp_path / "item.json"
        p.write_text(json.dumps({"template": "config-run", "tree": "/t", "env": dict(env["ENV"]), **item}))
        return subprocess.run(["bash", str(script), "check", key, str(p)], env={**os.environ, "VY_N1": "unreachable.invalid",
                              "VY_N2": "unreachable.invalid"}, capture_output=True).returncode

    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    assert check("cov-g153") == 0
    assert check("Cov_G153") == 2
    assert check("cov-g153", template="config-run-row") == 2
    assert check("cov-g153", env={**env["ENV"], "REVISION": ""}) == 2
    assert check("cov-p002", env={**env["ENV"], "ROW": env["ENV"]["ROW"].replace("__tp1__", "__tp2__")}) == 0
    assert check("cov-p002", env={**env["ENV"], "ROW": env["ENV"]["ROW"].replace("__tp1__", "__tp4__")}) == 2


@pytest.mark.skipif(shutil.which("bash") is None, reason="n2_commit.sh is bash")
def test_n2_commit_fetches_only_missing_manifests_that_hash_to_their_id(tmp_path):
    """Node 2's store has no remote, so a cited Build's manifest missing there failed cov-g217's precheck after a 19-min bootstrap:
    fetch-manifests copies it from node 1's store, keeps only bytes whose sha256 is the id, and leaves present ones alone."""
    n1, n2, bin_ = tmp_path / "n1", tmp_path / "n2", tmp_path / "bin"
    for d in (n1 / "manifests", n2 / "manifests", bin_):
        d.mkdir(parents=True)
    # node 1 is a directory here: the fake rsync maps HOST:/PATH to n1/PATH's last two parts
    (bin_ / "rsync").write_text('#!/bin/sh\nfor a; do :; done\nsrc=$(eval echo \\${$(($#-1))})\n'
                                f'cp "{n1}/manifests/${{src##*/}}" "$a"\n')
    (bin_ / "rsync").chmod(0o755)
    good, other = b'{"kind":"vllm-build/v1"}', b'{"kind":"other"}'
    art = lambda b: "art:" + hashlib.sha256(b).hexdigest()
    (n1 / "manifests" / f"{art(good)[4:]}.json").write_bytes(good)
    forged = "art:" + "0" * 64
    (n1 / "manifests" / f"{forged[4:]}.json").write_bytes(other)

    def fetch(*arts: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(NEBIUS / "n2_commit.sh"), "fetch-manifests", *arts], capture_output=True, text=True,
                              env={**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "VY_N1": "n1.invalid", "VY_N2": "n2.invalid",
                                   "VY_STORE": str(n2)})

    assert fetch(art(good), forged, "").returncode == 0
    assert (n2 / "manifests" / f"{art(good)[4:]}.json").read_bytes() == good
    assert sorted(p.name for p in (n2 / "manifests").iterdir()) == [f"{art(good)[4:]}.json"]
    (n1 / "manifests" / f"{art(good)[4:]}.json").unlink()
    assert fetch(art(good)).returncode == 0 and "fetched" not in fetch(art(good)).stdout
    assert fetch("art:abc").returncode == 2


@pytest.mark.skipif(shutil.which("bash") is None, reason="n2_commit.sh is bash")
def test_n2_commit_counts_the_gpus_deployments_gpu_could_still_admit(tmp_path):
    """With VY_N2_FIRST=0, offload moves a held Commit only when n1-free prints 0: deployments-gpu's own unused quota counts even
    while provers borrows it (Kueue reclaims it), the cohort's idle GPUs count up to the borrowing limit, and an unreadable Kueue
    prints nothing, so the Commit stays."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()

    def free(dep: tuple[int, int, int], prov: tuple[int, int, int], fail: bool = False) -> str:
        def cq(name, nom, bl, use):
            return {"metadata": {"name": name}, "spec": {"cohort": "nebius", "resourceGroups": [{"flavors": [{"resources": [
                {"name": "cpu", "nominalQuota": "24"}, {"name": "nvidia.com/gpu", "nominalQuota": str(nom), "borrowingLimit": str(bl)}]}]}]},
                "status": {"flavorsUsage": [{"resources": [{"name": "nvidia.com/gpu", "total": str(use)}]}]}}
        (tmp_path / "cq.json").write_text(json.dumps({"items": [cq("deployments-gpu", *dep), cq("provers", *prov),
                                                                 cq("backfill", 0, 0, 0)]}))
        (bin_ / "kubectl").write_text("#!/bin/sh\nexit 1\n" if fail else f'#!/bin/sh\ncat "{tmp_path}/cq.json"\n')
        (bin_ / "kubectl").chmod(0o755)
        return subprocess.run(["bash", str(NEBIUS / "n2_commit.sh"), "n1-free"], capture_output=True, text=True,
                              env={**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}"}).stdout.strip()

    assert free((6, 2, 5), (2, 1, 3)) == "1"      # 1 Oct 02:27Z: its sixth GPU is lent to provers
    assert free((6, 2, 6), (2, 1, 2)) == "0"
    assert free((6, 2, 6), (2, 1, 0)) == "2"      # provers idle: borrow up to the limit
    assert free((6, 2, 7), (2, 1, 0)) == "1"
    assert free((6, 2, 2), (2, 1, 0)) == "6"
    assert free((6, 2, 6), (2, 1, 2), fail=True) == ""


@pytest.mark.skipif(shutil.which("bash") is None, reason="n2_commit.sh is bash")
def test_n2_commit_records_the_queues_placement_and_keeps_a_refused_row_on_node_1(tmp_path):
    """submit's `place` asks `cluster submit` for node 2 with the row's research question and prints the placed.jsonl line; a
    refusal exits 3, and a queue that gives no answer prints nothing and exits 0, so it never holds a row back. The queue is a
    stand-in `cluster` package that echoes its arguments (tools/cluster's own tests cover the placement)."""
    src = tmp_path / "cluster-src"
    (src / "src" / "cluster").mkdir(parents=True)
    (src / "descriptions").mkdir()
    (src / "src" / "cluster" / "__init__.py").write_text("")
    (src / "src" / "cluster" / "__main__.py").write_text(
        "import json, sys\na = sys.argv[1:]\nv = lambda k: a[a.index(k) + 1]\n"
        "assert a[0] == 'submit' and a[1].endswith('descriptions/nebius.toml') and v('--node') == 'vy-nebius-2'\n"
        "if int(v('--gpus')) > 8:\n    print(json.dumps({'refused': 'vy-nebius-2 has 8 GPUs'}))\n    sys.exit(3)\n"
        "print(json.dumps({'node': 'vy-nebius-2', 'why': 'starts now', 'question': v('--question'), 'max_s': int(v('--max-s'))}))\n")

    def place(queue: Path, gpus: str = "1") -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(NEBIUS / "n2_commit.sh"), "place", "commit", "vllm-epoch-run-cov-gm149", gpus, "40",
                               "Does cov-gm149's Commit validate on vy-nebius-2?"], capture_output=True, text=True,
                              env={**os.environ, "VY_QUEUE_SRC": str(queue)})

    r = place(src)
    assert r.returncode == 0, r.stderr
    rec = json.loads(r.stdout)
    assert (rec["kind"], rec["key"], rec["node"], rec["executor"]) == ("commit", "vllm-epoch-run-cov-gm149", "vy-nebius-2", "fill-queue")
    assert rec["question"] == "Does cov-gm149's Commit validate on vy-nebius-2?"
    r = place(src, gpus="9")
    assert r.returncode == 3 and "refuses" in r.stderr and not r.stdout
    r = place(tmp_path / "no-cluster")
    assert r.returncode == 0 and not r.stdout and "no placement" in r.stderr


def test_publish_pool_merges_node_1s_own_file_only_while_it_is_fresh():
    spec = importlib.util.spec_from_file_location("publish_pool_merge_under_test", NEBIUS / "publish_pool.py")
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    now = datetime.datetime(2026, 9, 30, 22, 0, 0, tzinfo=datetime.timezone.utc)
    doc = lambda: {"nodes": {"n2": {"gpu_busy_5m": 0.9}}}
    n1 = {"generated_at": "2026-09-30T21:50:00Z", "nodes": {"n1": {"gpu_busy_5m": 0.05}, "n2": {"gpu_busy_5m": 0.0}}}
    merged = pp.merge(doc(), n1, now)
    assert merged["nodes"]["n2"] == {"gpu_busy_5m": 0.9}
    assert merged["nodes"]["n1"] == {"gpu_busy_5m": 0.05, "generated_at": "2026-09-30T21:50:00Z"}
    assert "n1" not in pp.merge(doc(), {**n1, "generated_at": "2026-09-30T21:40:00Z"}, now)["nodes"]
    assert pp.merge(doc(), None, now) == doc()


# ---- n1_lease: node 1's process-level GPU leases --------------------------------------------------------------------------------------

def _n1_lease(monkeypatch, tmp_path, holders=(), **env):
    """The controller over a 4-GPU table, with kubectl faked: `holders` are (job, pod, uuid) running holder Jobs."""
    monkeypatch.setenv("GPU_LEASE_DIR", str(tmp_path / "gl"))
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    (tmp_path / "gl").mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("n1_lease_under_test", NEBIUS / "n1_lease.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    calls: list[tuple] = []
    state = {"holders": list(holders), "pods": [], "nominal": 3, "workloads": []}

    def kubectl(*args, stdin=None):
        calls.append((args, stdin))
        if args[:2] == ("get", "clusterqueue"):
            if state["nominal"] is None:
                raise subprocess.CalledProcessError(1, "kubectl")
            return json.dumps({"spec": {"resourceGroups": [{"flavors": [{"resources": [
                {"name": "cpu", "nominalQuota": "64"}, {"name": "nvidia.com/gpu", "nominalQuota": str(state["nominal"])}]}]}]}})
        if args[:2] == ("get", "workloads"):
            return json.dumps({"items": state["workloads"]})
        if args[:2] == ("get", "jobs"):
            return json.dumps({"items": [{"metadata": {"name": j}, "status": {"active": 1}} for j, _, _ in state["holders"]]})
        if args[:2] == ("get", "pods"):
            return json.dumps({"items": [{"metadata": {"name": p, "uid": f"uid-{p}", "labels": {"job-name": j}},
                                          "status": {"phase": "Running"}} for j, p, _ in state["holders"] + state["pods"]]})
        if args[0] == "logs":
            return next(u for _, p, u in state["holders"] if p == args[1]) + "\n"
        return ""

    monkeypatch.setattr(m, "kubectl", kubectl)
    monkeypatch.setattr(m, "gpu_table", lambda: {"GPU-a": 0, "GPU-b": 1, "GPU-c": 2, "GPU-d": 3})
    return m, calls, state


@needs_flock
def test_n1_lease_fences_every_gpu_outside_the_pool_so_gpu_lease_takes_only_a_holders_gpu(monkeypatch, tmp_path, reap):
    m, calls, _ = _n1_lease(monkeypatch, tmp_path, holders=[("gpu-pool-1", "gpu-pool-1-x", "GPU-c")])
    s = m.Controller().tick(1_800_000_000.0)
    assert s["pool"] == 1 and [bool(g["fence_pids"]) for g in s["gpus"]] == [True, True, False, True]
    assert "who=kubernetes" in (tmp_path / "gl" / "0.owner").read_text()
    env = {**gpu_env(tmp_path, count=4), "GPU_LEASE_ALLOWED_FILE": str(tmp_path / "gl" / "pool.gpus")}
    got = subprocess.run(["bash", str(GPU_LEASE), "1", "--mem-gb", "0", "--", "sh", "-c", "echo $CUDA_VISIBLE_DEVICES"], env=env,
                         capture_output=True, text=True)
    assert got.returncode == 0 and got.stdout.strip() == "2"
    two = subprocess.run(["bash", str(GPU_LEASE), "2", "--mem-gb", "0", "--", "true"], env=env, capture_output=True, text=True)
    assert two.returncode == 75


@needs_flock
def test_n1_lease_grows_the_pool_for_a_waiter_and_shrinks_a_gpu_left_idle(monkeypatch, tmp_path, reap):
    import fcntl
    m, calls, state = _n1_lease(monkeypatch, tmp_path, VY_POOL_IDLE_S=120)
    c = m.Controller()
    w = tmp_path / "gl" / "wait.1.4242"
    w.write_text("pid=4242 n=1 run=r-x who=backend-sweep-2 since=2026-09-30T22:00:00Z\n")
    fd = os.open(w, os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_SH)
    t = 1_800_000_000.0
    s = c.tick(t)
    created = [json.loads(st) for a, st in calls if a[0] == "create"]
    assert len(created) == 1 and created[0]["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"] == {"nvidia.com/gpu": "1"}
    assert created[0]["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "provers" and s["pool"] == 0
    assert all(g["fence_pids"] for g in s["gpus"])
    state["holders"] = [(created[0]["metadata"]["name"], "hold-1", "GPU-b")]
    before = len(calls)
    s = c.tick(t + 10)
    assert s["pool"] == 1 and not s["gpus"][1]["fence_pids"] and not [a for a, _ in calls[before:] if a[0] == "create"]
    os.close(fd)
    w.unlink()
    c.tick(t + 20)
    assert not [a for a, _ in calls if a[:2] == ("delete", "job")]
    s = c.tick(t + 20 + 121)
    assert [a[2] for a, _ in calls if a[:2] == ("delete", "job")] == [created[0]["metadata"]["name"]]
    assert s["gpus"][1]["fence_pids"]


@needs_flock
def test_n1_lease_named_pool_labels_its_own_holders_and_a_blocked_pool_grants_nothing_new(monkeypatch, tmp_path, reap):
    import fcntl
    m, calls, state = _n1_lease(monkeypatch, tmp_path, holders=[("gpu-pool-circuits-1", "gpu-pool-circuits-1-x", "GPU-b")],
                                VY_POOL_NAME="circuits", VY_POOL_QUEUE="deployments-gpu")
    c = m.Controller()
    t = 1_800_000_000.0
    s = c.tick(t)
    assert [a for a, _ in calls if a[:2] == ("get", "jobs")][0][3] == "verity.dev/gpu-pool=circuits"
    assert s["pool"] == 1 and not s["gpus"][1]["fence_pids"] and s["blocked"] is None
    (tmp_path / "gl" / "blocked").write_text("pid 77 on GPU 3 without a lease (pod nd-x)\n")
    w = tmp_path / "gl" / "wait.1.4242"
    w.write_text("pid=4242 n=2 run=r-x who=vllm-epoch-run since=2026-09-30T22:00:00Z\n")
    fd = os.open(w, os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_SH)
    s = c.tick(t + 10)
    assert s["blocked"].startswith("pid 77") and all(g["fence_pids"] for g in s["gpus"])
    assert [a[2] for a, _ in calls if a[:2] == ("delete", "job")] == ["gpu-pool-circuits-1"]
    assert not [a for a, _ in calls if a[0] == "create"]
    os.close(fd)
    w.unlink()
    (tmp_path / "gl" / "blocked").unlink()
    state["holders"] = []
    fd = os.open(w, os.O_CREAT | os.O_RDONLY)
    w.write_text("pid=4242 n=1 run=r-x who=vllm-epoch-run since=2026-09-30T22:00:00Z\n")
    fcntl.flock(fd, fcntl.LOCK_SH)
    c.tick(t + 20)
    created = [json.loads(st) for a, st in calls if a[0] == "create"]
    assert len(created) == 1 and created[0]["metadata"]["name"].startswith("gpu-pool-circuits-")
    assert created[0]["metadata"]["labels"]["verity.dev/gpu-pool"] == "circuits"
    assert created[0]["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "deployments-gpu"
    os.close(fd)


def test_gpu_stray_allows_device_plugin_and_leased_gpus_and_blocks_the_pool_of_a_pod_off_its_lease(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("gpu_stray_under_test", NEBIUS / "gpu_stray.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    pool = tmp_path / "gpu-lease-circuits"
    pool.mkdir()
    monkeypatch.setattr(m, "POOL_DIRS", [pool])
    monkeypatch.setattr(m, "LOG", tmp_path / "stray.jsonl")
    # GPU UUID -> (index, minor); pids 10..13 compute, 50 holds the lease on GPU 1 for pod-lease
    monkeypatch.setattr(m, "compute_apps", lambda: [(10, "GPU-a"), (11, "GPU-b"), (12, "GPU-c"), (13, "GPU-d"), (14, "GPU-c")])
    monkeypatch.setattr(m, "gpu_table", lambda: {"GPU-a": (0, 4), "GPU-b": (1, 1), "GPU-c": (2, 2), "GPU-d": (3, 0)})
    monkeypatch.setattr(m, "leases", lambda: {1: [(pool, 50)]})
    monkeypatch.setattr(m, "dev_minors", lambda pid: {10: {4}, 13: {4, 0}}.get(pid, set(range(8))))
    monkeypatch.setattr(m, "pod_uid_of", lambda pid: {10: "u-plugin", 13: "u-plugin"}.get(pid, "u-lease"))
    monkeypatch.setattr(m, "ancestors", lambda pid: {pid})
    monkeypatch.setattr(m, "environ", lambda pid: {"NVIDIA_VISIBLE_DEVICES": "void"})
    monkeypatch.setattr(m, "alive", lambda pid: pid != 14)
    monkeypatch.setattr(m, "cmdline", lambda pid: f"python w{pid}.py")

    def pod(uid, name, gpus, mount=None):
        c = {"resources": {"limits": {"nvidia.com/gpu": str(gpus)}} if gpus else {}}
        vols = [{"name": "gpu-lease", "hostPath": {"path": str(mount)}}] if mount else []
        return {"metadata": {"uid": uid, "namespace": "default", "name": name, "labels": {"verity.dev/workstream": "w"}},
                "spec": {"containers": [c], "volumes": vols}}

    monkeypatch.setattr(m, "pods_by_uid", lambda: {"u-plugin": pod("u-plugin", "nd-plugin", 1),
                                                   "u-lease": pod("u-lease", "nd-lease", 0, pool)})
    strays = m.check(1_800_000_000.0)
    # 10: its one device-plugin GPU; 11: its pod's lease; 12: no lease on GPU 2; 13: a 1-GPU pod that sees two GPUs; 14: gone
    assert [(s["pid"], s["gpu"], s.get("blocked")) for s in strays] == [(12, 2, str(pool)), (13, 3, None)]
    assert "pid 12 (default/nd-lease) ran on GPU 2 without a lease" in (pool / "blocked").read_text()
    assert len((tmp_path / "stray.jsonl").read_text().splitlines()) == 2
    monkeypatch.setattr(m, "compute_apps", lambda: [(10, "GPU-a"), (11, "GPU-b")])
    (pool / "blocked").unlink()
    assert m.check(1_800_000_060.0) == [] and not (pool / "blocked").exists()


def test_gpu_stray_judges_only_a_process_listed_before_and_after_the_leases_are_read(monkeypatch, tmp_path):
    """Node 1, 06:58Z Oct 2: a leased engine was still listed while its context reset, and its lease ended before the locks were
    read, so the probe blocked its pool. A second listing after the locks drops it."""
    spec = importlib.util.spec_from_file_location("gpu_stray_under_test", NEBIUS / "gpu_stray.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    pool = tmp_path / "gpu-lease"
    pool.mkdir()
    monkeypatch.setattr(m, "POOL_DIRS", [pool])
    monkeypatch.setattr(m, "LOG", tmp_path / "stray.jsonl")
    # 20 released its lease on GPU 0 between the listings; 21 holds GPU 1 with no lease all along; 22 started after the first
    listings = iter([[(20, "GPU-a"), (21, "GPU-b")], [(21, "GPU-b"), (22, "GPU-c")]])
    monkeypatch.setattr(m, "compute_apps", lambda: next(listings))
    monkeypatch.setattr(m, "gpu_table", lambda: {"GPU-a": (0, 0), "GPU-b": (1, 1), "GPU-c": (2, 2)})
    monkeypatch.setattr(m, "leases", lambda: {})
    monkeypatch.setattr(m, "dev_minors", lambda pid: set(range(8)))
    monkeypatch.setattr(m, "pod_uid_of", lambda pid: "u")
    monkeypatch.setattr(m, "ancestors", lambda pid: {pid})
    monkeypatch.setattr(m, "environ", lambda pid: {})
    monkeypatch.setattr(m, "alive", lambda pid: True)
    monkeypatch.setattr(m, "cmdline", lambda pid: f"python w{pid}.py")
    monkeypatch.setattr(m, "pods_by_uid", lambda: {"u": {"metadata": {"uid": "u", "namespace": "default", "name": "nd-x"},
                                                         "spec": {"containers": [{}],
                                                                  "volumes": [{"name": "gpu-lease", "hostPath": {"path": str(pool)}}]}}})
    assert [(s["pid"], s["gpu"]) for s in m.check(1_800_000_000.0)] == [(21, 1)]
    assert "pid 21 (default/nd-x) ran on GPU 1 without a lease" in (pool / "blocked").read_text()


@needs_flock
def test_n1_lease_expires_a_lease_past_its_until_and_stops_one_outside_the_pool(monkeypatch, tmp_path, reap):
    import fcntl
    m, calls, state = _n1_lease(monkeypatch, tmp_path, holders=[("gpu-pool-1", "gpu-pool-1-x", "GPU-a")])
    monkeypatch.setattr(m, "pod_uid_of", lambda pid: "uid-job-pod")
    t = 1_800_000_000.0
    c = m.Controller()
    c.tick(t)
    lock = os.open(tmp_path / "gl" / "0.lock", os.O_WRONLY)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (tmp_path / "gl" / "0.owner").write_text(f"pid={os.getpid()} run=r-x who=backend-sweep-2 uuid=GPU-a "
                                             f"since={m.iso(t - 3600)} until={m.iso(t - 200)} cmd=bash -c x\n")
    state["pods"].append(("job-x", "job-pod", "-"))
    s = c.tick(t + 1)
    assert s["gpus"][0]["lease"]["pod"] == "job-pod" and s["gpus"][0]["lease"]["owner_alive"]
    assert ("delete", "pod", "job-pod", "--wait=false") in [a for a, _ in calls]
    assert [e["ev"] for e in s["events"]] == ["expired"]
    calls.clear()
    (tmp_path / "gl" / "0.owner").write_text(f"pid={os.getpid()} run=r-x who=backend-sweep-2 since={m.iso(t)} cmd=x\n")
    state["holders"] = []
    c.tick(t + 2)
    assert not [a for a, _ in calls if a[0] == "delete"]
    s = c.tick(t + 3)
    assert [e["ev"] for e in s["events"]] == ["lease-outside-pool"] and ("delete", "pod", "job-pod", "--wait=false") in [a for a, _ in calls]
    os.close(lock)


@needs_flock
def test_n1_lease_grows_only_within_the_queues_unborrowed_quota_and_sigterms_a_host_process_lease(monkeypatch, tmp_path, reap):
    import fcntl
    m, calls, state = _n1_lease(monkeypatch, tmp_path, VY_POOL_MAX=3)
    monkeypatch.setattr(m, "pod_uid_of", lambda pid: None)
    w = tmp_path / "gl" / "wait.1.4242"
    w.write_text("pid=4242 n=3 run=r-x who=adhoc:me since=2026-09-30T22:00:00Z\n")
    fd = os.open(w, os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_SH)
    bench = {"metadata": {"ownerReferences": [{"name": "nd-m0-bench"}]},
             "status": {"admission": {"clusterQueue": "provers"}, "conditions": [{"type": "QuotaReserved", "status": "True"}]},
             "spec": {"podSets": [{"count": 1, "template": {"spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": "2"}}}]}}}]}}
    state["workloads"] = [bench]
    t = 1_800_000_000.0
    c = m.Controller()
    c.tick(t)
    assert len([a for a, _ in calls if a[0] == "create"]) == 1          # 3 wanted, 3 nominal less the bench's 2
    calls.clear()
    state["nominal"] = None
    c2 = m.Controller()
    c2.tick(t)
    assert not [a for a, _ in calls if a[0] == "create"]                # the quota unread: no growth
    os.close(fd)
    w.unlink()
    state["nominal"], state["workloads"] = 3, []
    state["holders"] = [("gpu-pool-1", "gpu-pool-1-x", "GPU-a")]
    c3 = m.Controller()
    c3.tick(t)
    proc = subprocess.Popen(["cat"], stdin=subprocess.PIPE)          # blocks until signalled
    lock = os.open(tmp_path / "gl" / "0.lock", os.O_WRONLY)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (tmp_path / "gl" / "0.owner").write_text(f"pid={proc.pid} run=r-x who=adhoc:me since={m.iso(t - 3600)} until={m.iso(t - 200)} cmd=x\n")
    s = c3.tick(t + 1)
    assert [e["ev"] for e in s["events"]] == ["expired"] and not [a for a, _ in calls if a[:2] == ("delete", "pod")]
    assert proc.wait() == -15
    os.close(lock)


@needs_flock
def test_n1_lease_borrows_only_for_preemptible_waiters_before_its_deadline_and_sigkills_a_lease_that_outlives_sigterm(
        monkeypatch, tmp_path, reap):
    import fcntl
    t = 1_800_000_000.0
    m, calls, state = _n1_lease(monkeypatch, tmp_path, VY_POOL_MAX=3, VY_POOL_BORROW=1,
                                VY_POOL_BORROW_UNTIL=datetime.datetime.fromtimestamp(t + 60, datetime.timezone.utc)
                                .strftime("%Y-%m-%dT%H:%M:%SZ"))
    monkeypatch.setattr(m, "pod_uid_of", lambda pid: None)
    bench = {"metadata": {"ownerReferences": [{"name": "nd-m0-bench"}]},
             "status": {"admission": {"clusterQueue": "provers"}, "conditions": [{"type": "QuotaReserved", "status": "True"}]},
             "spec": {"podSets": [{"count": 1, "template": {"spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": "2"}}}]}}}]}}
    state["workloads"] = [bench]
    w = tmp_path / "gl" / "wait.1.4242"
    fd = os.open(w, os.O_RDONLY | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_SH)
    grown = lambda: len([a for a, _ in calls if a[0] == "create"])
    for line, at, want in [("pid=4242 n=3 who=adhoc:me since=x\n", t, 1),             # not preemptible: 3 nominal less 2
                           ("pid=4242 n=3 who=adhoc:me preempt=1 since=x\n", t, 2),   # preemptible: and 1 borrowed
                           ("pid=4242 n=3 who=adhoc:me preempt=1 since=x\n", t + 61, 1)]:  # past the deadline
        w.write_text(line)
        calls.clear()
        m.Controller().tick(at)
        assert grown() == want, line
    os.close(fd)
    w.unlink()
    state["workloads"], state["holders"] = [], [("gpu-pool-1", "gpu-pool-1-x", "GPU-a")]
    c = m.Controller()
    c.tick(t)
    proc = subprocess.Popen(["sh", "-c", "trap '' TERM; echo ready; exec cat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert proc.stdout.readline() == b"ready\n"   # the trap is set before the lease is signalled
    lock = os.open(tmp_path / "gl" / "0.lock", os.O_WRONLY)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (tmp_path / "gl" / "0.owner").write_text(f"pid={proc.pid} run=r-x who=adhoc:me since={m.iso(t - 3600)} until={m.iso(t - 200)} cmd=x\n")
    assert [e.get("signal") for e in c.tick(t + 1)["events"]] == ["SIGTERM"]
    assert [e.get("signal") for e in c.tick(t + 1 + m.KILL_AFTER_S)["events"]] == ["SIGKILL"]
    assert proc.wait() == -9
    os.close(lock)


@needs_flock
def test_n1_lease_holds_agent_lock_while_the_brain_answers_and_falls_back_when_it_stops(monkeypatch, tmp_path, reap):
    import fcntl
    brain = tmp_path / "brain.sh"
    brain.write_text('#!/bin/sh\n[ -e "$(dirname "$0")/down" ] && exit 1\ncat > /dev/null\n'
                     'echo \'{"grants": {"4242": [2]}, "pool_target": 1}\'\n')
    m, calls, state = _n1_lease(monkeypatch, tmp_path, holders=[("gpu-pool-1", "gpu-pool-1-x", "GPU-c")],
                                VY_LEASE_BRAIN=f"sh {brain}")
    w = tmp_path / "gl" / "wait.1.4242"
    w.write_text("pid=4242 n=1 run=r-x who=backend-sweep-2 since=2026-09-30T22:00:00Z\n")
    fd = os.open(w, os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_SH)
    c = m.Controller()
    t = 1_800_000_000.0
    s = c.tick(t)
    assert s["mode"] == "brain" and (tmp_path / "gl" / "grant.4242").read_text().strip() == "2"
    probe = os.open(tmp_path / "gl" / "agent.lock", os.O_RDONLY)
    with pytest.raises(BlockingIOError):
        fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
    (tmp_path / "down").write_text("")
    assert c.tick(t + 5)["mode"] == "brain"
    s = c.tick(t + 10)
    assert s["mode"] == "local" and [e["ev"] for e in s["events"]] == ["brain-down"]
    fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
    os.close(probe)
    os.close(fd)


def test_fill_runner_on_numa_0_terms_freezes_cpu_jobs_while_a_window_waits_and_binds_their_memory(tmp_path, monkeypatch):
    """PoUW's terms for CPU fill on 0-47 (bc-2aa33ad8): frozen from the moment a window waits, and memory kept off NUMA 0. Both are
    off unless set, so the runner behaves as before."""
    fill = tmp_path / "fill"
    for d in ("running", "logs"):
        (fill / d).mkdir(parents=True)
    monkeypatch.setenv("FILL_DIR", str(fill))

    def load(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        spec = importlib.util.spec_from_file_location("fill_runner_numa0", FILL_RUNNER)
        fr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fr)
        cmds = []
        monkeypatch.setattr(fr.shutil, "which", lambda name: None)
        monkeypatch.setattr(fr.subprocess, "Popen", lambda cmd, **kw: cmds.append(cmd) or type("P", (), {"pid": 1})())
        monkeypatch.setattr(fr, "event", lambda *a, **kw: None)
        return fr, cmds

    def started(fr, cmds, name, head):
        p = tmp_path / name
        p.write_text(f"# fill: owner=o {head}\n")
        fr.start(p, fr.header(p))
        return cmds[-1]

    fr, cmds = load()
    assert not fr.cpu_frozen(False, True) and fr.cpu_frozen(True, False)
    assert started(fr, cmds, "a.sh", "gpus=0")[0] == "nice"
    fr, cmds = load(FILL_FREEZE_ON_WAIT="1", FILL_CPU_MEMBIND="1")
    assert fr.cpu_frozen(False, True) and not fr.cpu_frozen(False, False)
    assert started(fr, cmds, "b.sh", "gpus=0")[:2] == ["numactl", "--membind=1"]
    assert started(fr, cmds, "c.sh", "gpus=0 project=verity")[0] == "nice"
    assert started(fr, cmds, "d.sh", "gpus=1")[0] == fr.GPU_LEASE
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("Node 0 MemFree:        2097152 kB\nNode 0 Inactive(file): 6291456 kB\nNode 0 AnonPages:      1048576 kB\n")
    monkeypatch.setattr(fr, "NODE_MEMINFO", str(meminfo))
    assert fr.numa0_gb() == {"numa0_reclaimable_gb": 8.0, "numa0_anon_gb": 1.0}


def test_fill_runner_runs_an_owner_in_cpu_sets_on_its_own_cpus_beside_the_shared_slots(tmp_path, monkeypatch):
    """verity-top (1 Oct 12:00 AM PDT): compute accounting's verifies run on 0-47, 4 at a time, at ionice idle, capped from a
    measured peak, without taking CPU_SET's slots; everyone else's CPU jobs stay on CPU_SET."""
    fill = tmp_path / "fill"
    for d in ("queue", "running", "logs", "done", "failed"):
        (fill / d).mkdir(parents=True)
    monkeypatch.setenv("FILL_DIR", str(fill))
    monkeypatch.setenv("GPU_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("FILL_CPU_SLOTS", "1")

    def load():
        spec = importlib.util.spec_from_file_location("fill_runner_cpusets", FILL_RUNNER)
        fr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fr)
        return fr

    fr = load()
    monkeypatch.setattr(fr, "disk_pct", lambda path: 0.0)
    started = []

    def fake_start(p, h, backfill=False):
        started.append((p.name, h.get("cpuset"), h.get("mem_gb")))
        p.unlink()
        fr.running[p.name] = {"h": h, "t0": 0.0, "pid": 0, "proc": type("P", (), {"poll": lambda self: None})(),
                              "started": "t", "paused": False, "unit": None}

    def queue_jobs():
        fr.running.clear()
        started.clear()
        for k in range(6):
            (fill / "queue" / f"ver-{k}.sh").write_text("# fill: owner=bc-e gpus=0 cpus=4 mem_gb=48\n")
        (fill / "queue" / "census.sh").write_text("# fill: owner=bc-d gpus=0 cpus=6\n")

    monkeypatch.setattr(fr, "start", fake_start)
    # the fake jobs' pid 0 would make a real stop signal this test's own process group: none is due, so none may happen
    monkeypatch.setattr(fr, "active_s", lambda j: 0.0)
    monkeypatch.setattr(fr, "stop", lambda k, why, requeue=True: pytest.fail(f"stopped {k}: {why}"))
    queue_jobs()
    fr.tick(0)
    assert started and all(c is None for _, c, _ in started)       # no cpu-sets: the shared slot and the Verity pool's lent ones
    queue_jobs()
    (fill / "cpu-sets").write_text("# owner cpus slots [mem_gb]\nbc-e 0-47 4 20  # verity-top 12:00 AM PDT\nbad line\n")
    fr.tick(0)
    assert sorted(started) == [("census.sh", None, None), *((f"ver-{k}.sh", "0-47", 20) for k in range(4))]
    started.clear()
    fr.tick(0)
    assert started == []                    # 4 of theirs run; the census holds the shared slot; theirs are never lent

    fr = load()
    cmds = []
    monkeypatch.setattr(fr.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(fr.subprocess, "Popen", lambda cmd, **kw: cmds.append(cmd) or type("P", (), {"pid": 1})())
    monkeypatch.setattr(fr, "event", lambda *a, **kw: None)
    p = fill / "queue" / "ver-9.sh"
    p.write_text("# fill: owner=bc-e gpus=0 cpus=4 mem_gb=48\n")
    fr.start(p, {**fr.header(p), "cpuset": "0-47", "mem_gb": 20})
    cmd = cmds[-1]
    assert "MemoryMax=20G" in cmd and cmd[cmd.index("--unit") + 1].startswith("fill-cpuset-")
    assert cmd[cmd.index("--") + 1:][:8] == ["ionice", "-c3", "nice", "-n", "19", "taskset", "-c", "0-47"]

    cg = Path(f"/proc/{os.getpid()}/cgroup").read_text().strip().split("::", 1)[1]
    (tmp_path / "cg" / cg.lstrip("/")).mkdir(parents=True)
    (tmp_path / "cg" / cg.lstrip("/") / "memory.peak").write_text(f"{12_340_000_000}\n")
    monkeypatch.setattr(fr, "CGROUP_ROOT", str(tmp_path / "cg"))
    assert fr.mem_peak_gb(os.getpid()) == 12.34 and fr.mem_peak_gb(2**22 + 7) is None


@pytest.mark.skipif(not shutil.which("zstd"), reason="zstd")
def test_backup_packs_nothing_while_a_timed_window_runs_or_waits(tmp_path):
    src, status = tmp_path / "pouw", tmp_path / "status.txt"
    (src / "jobs" / "a").mkdir(parents=True)
    (src / "jobs" / "a" / "x.txt").write_text("x")
    env = {**os.environ, "BACKUP_SRC": str(src), "BACKUP_LAST": str(tmp_path / "last"), "BACKUP_STATUS": str(status),
           "BACKUP_QUIET_MAX_S": "0"}
    for k, (line, rc) in enumerate((("t free GPUs 8/8, kept free -, waiters -, timed True, window waiting False", 75),
                     ("t free GPUs 0/8, kept free -, waiters -, timed False, window waiting True", 75),
                     ("t free GPUs 8/8, kept free -, waiters -, timed False, window waiting False", 0))):
        status.write_text(line + "\n")
        run = tmp_path / f"run{k}"
        run.mkdir()
        r = subprocess.run(["bash", str(NEBIUS / "backup.sh")], env={**env, "RESEARCH_RUN_DIR": str(run)}, capture_output=True, text=True)
        assert r.returncode == rc, (line, r.stderr)
        assert (run / "backup" / "tars").is_dir() and bool(list((run / "backup" / "tars").iterdir())) == (rc == 0)
