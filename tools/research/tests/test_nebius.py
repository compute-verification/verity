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
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

from research import remote
from research.pods import lease as L
from research.pods import registry
from research.store import custody as CU

NEBIUS = Path(registry.__file__).resolve().parent / "nebius"
GPU_LEASE = Path(registry.__file__).resolve().parent / "sh" / "gpu_lease.sh"
CHECKPOINTS = Path(__file__).resolve().parents[3] / "integrations" / "vllm" / "manifests" / "checkpoints.json"
HOST = "203.0.113.7"


@pytest.fixture
def reg(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "notes" / "machines.d"
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setenv("RESEARCH_MACHINES_D", str(d))
    monkeypatch.setenv("RESEARCH_NOTES", str(d.parent))
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


@pytest.fixture(autouse=True)
def _no_host_yield_lock(tmp_path, monkeypatch):
    """fill_runner pauses cpu-sets jobs while a check holds FILL_YIELD_LOCK, by default the node's slot-d lock, which the train
    check running these tests on a node may hold itself (r20261002-072246-b9fe, -073629-eb44): no test sees the host's."""
    monkeypatch.setenv("FILL_YIELD_LOCK", str(tmp_path / "check-d.lock"))


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


def test_fill_runner_runs_only_bare_and_one_per_node(tmp_path):
    """`fill_runner.py --help` fell through to the runner and started a second one on node 2 (memory accounting's soak worker,
    3 Oct 05:37Z): it adopted the running series job and launched the next on GPU 7. Only a bare `fill_runner.py` runs the
    runner, and only while no other runner holds FILL_DIR/.runner.lock."""
    fill = tmp_path / "fill"
    env = {**os.environ, "FILL_DIR": str(fill), "GPU_LEASE_DIR": str(tmp_path / "gl")}
    for argv, rc in ((["--help"], 0), (["-h"], 0), (["--once"], 1), (["status"], 1)):
        r = subprocess.run([sys.executable, str(FILL_RUNNER), *argv], env=env, capture_output=True, text=True)
        assert r.returncode == rc and "usage: fill_runner.py" in r.stdout + r.stderr, (argv, r)
    assert not fill.exists()
    fill.mkdir()
    with open(fill / ".runner.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        r = subprocess.run([sys.executable, str(FILL_RUNNER)], env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "another runner holds" in r.stderr and not (fill / "queue").exists(), r


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
def test_a_waiter_pinned_with_on_holds_back_only_its_own_gpus(tmp_path):
    """ra-host-tier (12 Oct 19:35Z): a waiting `gpu-lease N --on LIST` deferred every later request, so its waiter kept free GPUs
    outside LIST from seven lanes. A later request may take a GPU outside every older waiter's list (by index or UUID), and an
    unpinned waiter still holds back all of them."""
    env = {**gpu_env(tmp_path, count=4), "GPU_LEASE_UUID_TABLE": "GPU-a0,GPU-b1,GPU-c2,GPU-d3"}
    gl = tmp_path / "gl"
    gl.mkdir()
    show = 'echo "$CUDA_VISIBLE_DEVICES"'

    def hold(name: str, line: str) -> subprocess.Popen:
        p = subprocess.Popen(["bash", "-c", f'exec 9>"{gl / name}"; flock -x 9; echo "{line}" >&9; echo ready; read x'],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert p.stdout.readline().strip() == "ready"
        return p

    waiters = [hold("wait.0000000000000000001.1", "pid=1 n=2 on=1,GPU-c2 who=pinned")]
    try:
        r = subprocess.run(["bash", str(GPU_LEASE), "2", "--", "sh", "-c", show], env=env, capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == "0,3", r.stderr
        r = subprocess.run(["bash", str(GPU_LEASE), "1", "--on", "1,2", "--", "true"], env=env, capture_output=True, text=True)
        assert r.returncode == 75 and "who=pinned" in r.stderr
        r = subprocess.run(["bash", str(GPU_LEASE), "3", "--", "true"], env=env, capture_output=True, text=True)
        assert r.returncode == 75 and "who=pinned" in r.stderr
        waiters.append(hold("wait.0000000000000000002.2", "pid=2 n=1 who=whole"))
        r = subprocess.run(["bash", str(GPU_LEASE), "1", "--on", "0", "--", "true"], env=env, capture_output=True, text=True)
        assert r.returncode == 75 and "who=pinned" in r.stderr
    finally:
        for p in waiters:
            p.stdin.close()
            p.wait()


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
def test_gpu_lease_refuses_a_lease_that_overlaps_a_booked_window_unless_it_is_the_timed_one(tmp_path):
    """At 10:00Z with 10:40-11:25Z booked: a lease ending by 10:40 runs, a longer one or one with no --max-min is refused holding
    nothing, --timed runs, a lease inside the window is refused and one after it runs; with --wait it waits without queuing."""
    windows = tmp_path / "windows"
    windows.write_text("# start UTC, minutes\n2026-10-04T09:00Z 30 # over\nnot a line\n"
                       "2026-10-04T10:40Z 45 # compute-accounting exhaustion window; slot-d=runs\n")
    at = {"10:00": "1791108000", "10:50": "1791111000", "11:25": "1791113100"}
    env = {**gpu_env(tmp_path, count=2), "GPU_LEASE_WINDOWS": str(windows), "GPU_LEASE_POLL_S": "0.1"}
    ran = tmp_path / "ran"

    def lease(when: str, *flags: str) -> subprocess.CompletedProcess:
        ran.unlink(missing_ok=True)
        return subprocess.run(["bash", str(GPU_LEASE), "1", *flags, "--mem-gb", "0", "--", "touch", str(ran)],
                              env={**env, "GPU_LEASE_NOW": at[when]}, capture_output=True, text=True)

    for when, flags, ok in (("10:00", ["--max-min", "30"], True), ("10:00", ["--max-min", "40"], True),
                            ("10:00", ["--max-min", "45"], False), ("10:00", [], False), ("10:00", ["--timed"], True),
                            ("10:50", ["--max-min", "5"], False), ("11:25", ["--max-min", "45"], True), ("11:25", [], True)):
        r = lease(when, *flags)
        assert (r.returncode == 0) == ok == ran.exists(), (when, flags, r.stderr)
        if not ok:
            assert r.returncode == 75 and "2026-10-04T10:40Z 45 # compute-accounting exhaustion window" in r.stderr, r.stderr
            assert ("with no --max-min" in r.stderr) == (not flags)
    status = subprocess.run(["bash", str(GPU_LEASE), "status"], env=env, capture_output=True, text=True).stdout.split("\n")
    assert [ln.split()[:2] for ln in status if ln] == [["0", "free"], ["1", "free"]]
    waiter = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--wait", "--max-min", "45", "--mem-gb", "0", "--", "true"],
                              env={**env, "GPU_LEASE_NOW": at["10:00"]}, stderr=subprocess.PIPE, text=True)
    try:
        assert "waiting, unqueued, until the lease clears it" in waiter.stderr.readline()
        assert not list((tmp_path / "gl").glob("wait.*"))
    finally:
        waiter.kill()
        waiter.wait()


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


@needs_flock
def test_a_preemptible_waiter_waits_for_a_preemptible_lease_instead_of_stopping_it(tmp_path):
    """Node 1, 2026-10-03: a queue of --preemptible trace chunks stopped its own newest lease every ~20 s, each new chunk the
    oldest waiter in turn."""
    env = {**gpu_env(tmp_path, count=1), "GPU_LEASE_POLL_S": "0.1"}
    pause = [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.read()"]
    fill = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--preemptible", "--mem-gb", "0", "--", *pause], env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert fill.stdout.readline().strip() == "ready"
    nxt = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--wait", "--preemptible", "--mem-gb", "0", "--", "echo", "got"], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        queued(nxt)
        assert fill.poll() is None and not (tmp_path / "gl" / "preempted.log").exists()
    finally:
        fill.stdin.close()
    assert fill.wait() == 0
    out, _ = nxt.communicate()
    assert nxt.returncode == 0 and out.strip() == "got"


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
    what the agent's planner reads (infra/cluster's nebius2.job_of)."""
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
    assert embedded["/etc/systemd/system/user@.service.d/vy-cpuset.conf"] == (NEBIUS / "user-cpuset.conf").read_bytes()
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


def _auth_harness(tmp_path):
    fake = tmp_path / "nebius"
    fake.write_text('#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = --private-key-file-path ] && cp "$2" "$OUT"; shift; done\n')
    fake.chmod(0o755)

    def auth(**env: str) -> tuple[subprocess.CompletedProcess, str]:
        out = tmp_path / "key.out"
        out.unlink(missing_ok=True)
        r = subprocess.run(["bash", "-c", f". {NEBIUS / 'common.sh'}; nebius_auth"], capture_output=True, text=True,
                           env={"PATH": os.environ["PATH"], "NEBIUS_BIN": str(fake), "OUT": str(out), "XDG_RUNTIME_DIR": str(tmp_path),
                                "NEBIUS_SA_ID": "sa-dummy", "NEBIUS_SA_PUBLIC_KEY_ID": "pk-dummy", **env})
        return r, out.read_text() if out.exists() else ""

    return auth


def test_nebius_auth_reads_the_key_from_one_line_of_base64_and_refuses_the_multi_line_secret(tmp_path):
    auth = _auth_harness(tmp_path)
    body = "-----BEGIN PRIVATE KEY-----\ndummy line one\ndummy line two\n-----END PRIVATE KEY-----\n"

    r, key = auth(NEBIUS_SA_PRIVATE_KEY_B64=base64.b64encode(body.encode()).decode())
    assert r.returncode == 0 and key == body and r.stdout == r.stderr == ""
    r, key = auth(NEBIUS_SA_PRIVATE_KEY=body.rstrip("\n"))    # a multi-line secret leaks through `env | cut -d= -f1`
    assert r.returncode == 1 and key == "" and "NEBIUS_SA_PRIVATE_KEY (the multi-line key) is no longer read" in r.stderr
    assert "dummy line" not in r.stderr
    r, key = auth(NEBIUS_SA_PRIVATE_KEY=body, NEBIUS_SA_PRIVATE_KEY_B64=base64.b64encode(body.encode()).decode())
    assert r.returncode == 0 and key == body
    r, _ = auth(NEBIUS_SA_PRIVATE_KEY_B64="not base64!")
    assert r.returncode == 1 and "NEBIUS_SA_PRIVATE_KEY_B64 is not base64" in r.stderr
    r, _ = auth()
    assert r.returncode == 1 and "NEBIUS_SA_PRIVATE_KEY_B64 is not set" in r.stderr
    r, _ = auth(CLOUD_AGENT_INJECTED_SECRET_NAMES="NEBIUS_SA_ID,NEBIUS_SA_PRIVATE_KEY_B64")    # injected, but lost from this shell
    assert r.returncode == 1 and "the Nebius key is not in this shell though this VM injects it" in r.stderr


def test_nebius_auth_reads_a_der_key_whose_base64_fits_the_cursor_secret_cap(tmp_path):
    """An RSA-4096 key's PEM is ~3272 bytes, whose base64 (~4364) exceeds Cursor's 4096-byte secret cap, so the secret may
    hold base64 of the DER (~3132 chars) instead; nebius_auth converts it back to PEM through openssl.  A throwaway
    RSA-2048 key generated here keeps the test fast; the size arithmetic scales, and the decode path is identical."""
    auth = _auth_harness(tmp_path)
    pem = subprocess.run(["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048"],
                         capture_output=True, check=True).stdout
    der = subprocess.run(["openssl", "pkey", "-outform", "DER"], input=pem, capture_output=True, check=True).stdout
    assert not der.startswith(b"-----") and der[0] == 0x30    # the two formats are told apart by their first bytes

    r, key = auth(NEBIUS_SA_PRIVATE_KEY_B64=base64.b64encode(der).decode())
    assert r.returncode == 0, r.stderr
    assert key.startswith("-----BEGIN PRIVATE KEY-----")
    pub = lambda k: subprocess.run(["openssl", "pkey", "-pubout"], input=k, capture_output=True, check=True).stdout  # noqa: E731
    assert pub(key.encode()) == pub(pem)                      # the same key came out

    r, key = auth(NEBIUS_SA_PRIVATE_KEY_B64=base64.b64encode(b"\x30\x82 not a real DER key").decode())
    assert r.returncode == 1 and key == "" and "neither a PEM file nor a DER private key" in r.stderr


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
    (tmp_path / "fill").mkdir()
    (tmp_path / "fill" / "max-min").write_text("# glob minutes until\nsoak-* 540 2026-10-02T14:45Z  # infra, thread 1\n"
                                               "gate.sh 60 2026-10-02T14:45Z\nbroken.sh sixty 2026-10-02T14:45Z\n")
    gate, soak, other = tmp_path / "gate.sh", tmp_path / "soak-a.sh", tmp_path / "broken.sh"
    for p in (gate, soak, other):
        p.write_text("# fill: owner=t gpus=1 max_min=900 project=pous\n")
    at = datetime.datetime(2026, 10, 2, 6, 30, tzinfo=datetime.timezone.utc).timestamp()
    assert fr.header(gate, at)["max_min"] == 60
    assert fr.header(soak, at)["max_min"] == 495                        # never past the line's until: 06:30 + 495 min = 14:45
    assert fr.header(other, at)["max_min"] == 30                        # a malformed line grants nothing
    late = datetime.datetime(2026, 10, 2, 14, 44, 30, tzinfo=datetime.timezone.utc).timestamp()
    assert fr.header(soak, late)["max_min"] == 30                       # under a minute left: the exception is over
    assert fr.header(soak, late + 3600)["max_min"] == 30
    calls = []
    monkeypatch.setattr(fr.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    assert fr.scope({"unit": "fill-verity-1"}, "freeze") and calls == [["systemctl", "--user", "freeze", "fill-verity-1.scope"]]
    assert not fr.scope({"unit": None}, "freeze") and len(calls) == 1


def test_the_verity_pool_lends_idle_slots_to_pous_cpu_jobs_and_takes_them_back(tmp_path, monkeypatch):
    """With no Verity job waiting, queued pous CPU jobs start in the Verity pool's free slots (its CPUs, a scope, its memory caps);
    a Verity job that finds no slot requeues the newest lent job, and nothing more is lent while it waits."""
    monkeypatch.setenv("FILL_DIR", str(tmp_path / "fill"))
    monkeypatch.setenv("FILL_VERITY_CPU_SLOTS", "2")
    monkeypatch.setenv("FILL_VERITY_UNTIL", "9999-01-01T00:00:00+00:00")
    monkeypatch.setenv("FILL_VERITY_STOP", "9999-01-01T01:00:00+00:00")
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
    monkeypatch.setenv("LEASE_USAGE", str(tmp_path / "lease-usage.jsonl"))
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
def test_n2_commit_gives_a_long_row_the_long_max_min():
    """3 Oct: node 2's 25-min limit stopped cov-gm359's TP2 B8 1024/128 Commit (~45 min) twice, 1.7 GPU-h, before it went back to
    node 1. A row with 1024 input tokens or batch 32+ gets VY_N2_COMMIT_LONG_MAX_MIN (120); the rest keep VY_N2_COMMIT_MAX_MIN."""
    def max_min(row: str, **env: str) -> str:
        return subprocess.run(["bash", str(NEBIUS / "n2_commit.sh"), "max-min", row], capture_output=True, text=True, check=True,
                              env={**os.environ, "VY_N1": "n1.invalid", "VY_N2": "n2.invalid", "VY_N2_COMMIT_MAX_MIN": "25", **env}).stdout.strip()

    assert max_min("qwen3-14b__bf16__rtxpro6000__tp2__b8__i1024__o128__mixed__greedy__bi-eager") == "120"
    assert max_min("gemma2-2b__bf16__rtxpro6000__tp1__b32__i256__o32__mixed__greedy__bi-eager") == "120"
    assert max_min("qwen3-14b__bf16__rtxpro6000__tp2__b8__i1024__o128__g", VY_N2_COMMIT_LONG_MAX_MIN="90") == "90"
    assert max_min("phi4-14b__bf16__rtxpro6000__tp2__b8__i256__o32__mixed__greedy__bi-eager") == "25"
    assert max_min("llama32-1b__bf16__rtxpro6000__tp1__b16__i512__o2__mixed__greedy__bi-eager", VY_N2_COMMIT_MAX_MIN="") == "40"


#: n2_custody.sh's node 1, faked under N1_FS: rsync copies the run dir there, and ssh answers the round's calls as node 1 would: the
#: research tree, the Job (its verify writes .custody over node 1's copy; a run JOB_ERR's file names fails with that line instead),
#: and the cat of that .custody; DURING_JOB=RUN:PATH appends to PATH while RUN's Job runs, LOOP_DURING_JOB=RUN starts the stopgap
#: loop (STATE/push.sh, its pid in STATE/push.pid) then, TIMED_DURING_JOB=RUN starts a timed window on node 2 then, ACTIVE_JOB=RUN
#: says RUN's Job has not finished, UNREACHABLE fails every call; CLOCK_AFTER_RSYNC=T sets the clock to T once a run dir is copied.
#: Every call is a line of FAKE_LOG.
_N2_RSYNC = r'''
import os, shutil, sys
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write("rsync " + " ".join(sys.argv[1:]) + "\n")
shutil.copytree(sys.argv[-2], os.environ["N1_FS"] + sys.argv[-1].split(":", 1)[1], symlinks=True, dirs_exist_ok=True)
if os.environ.get("CLOCK_AFTER_RSYNC"):
    with open(os.environ["FAKE_CLOCK"], "w") as f:
        f.write(os.environ["CLOCK_AFTER_RSYNC"])
'''
_N2_SSH = r'''
import hashlib, json, os, re, subprocess, sys
env, args = os.environ, sys.argv[1:]
at = args.index(env["VY_N1"])
cmd, spec = " ".join(args[at + 1:]), "" if "-n" in args[:at] else sys.stdin.read()
with open(env["FAKE_LOG"], "a") as f:
    f.write(f"ssh {' '.join(args)} {spec}\n")
if env.get("UNREACHABLE"):
    sys.exit(255)
if "test -f" in cmd:
    if env.get("UNTRUSTED"):
        print(env["UNTRUSTED"])
        sys.exit(4)
    sys.exit(1 if env.get("NO_SRC") else 0)
if "{.metadata.name}" in cmd:
    name = re.search(r"get job (\S+)", cmd).group(1)
    print(name if name == "n2-custody-" + env.get("ACTIVE_JOB", "") else "", end="")
    sys.exit(0)
if m := re.search(r"sudo -n cat (\S+)", cmd):
    try:
        sys.stdout.write(open(env["N1_FS"] + m.group(1)).read())
    except FileNotFoundError:
        sys.exit(1)
    sys.exit(0)
d = re.search(r"data custody (\S+) --publish", spec).group(1)
err = json.load(open(env["JOB_ERR"])).get(os.path.basename(d)) if os.path.exists(env["JOB_ERR"]) else None
if err:
    print(err, file=sys.stderr)
    sys.exit(1)
run, _, path = env.get("DURING_JOB", "").partition(":")
if run == os.path.basename(d):
    with open(path, "a") as f:
        f.write("a line node 2 wrote after the rsync\n")
if env.get("LOOP_DURING_JOB") == os.path.basename(d):
    loop = subprocess.Popen(["bash", env["VY_CUSTODY_STATE"] + "/push.sh"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    with open(env["VY_CUSTODY_STATE"] + "/push.pid", "w") as f:
        f.write(str(loop.pid))
if env.get("TIMED_DURING_JOB") == os.path.basename(d):
    status = env["VY_CUSTODY_FILL"] + "/status.txt"
    with open(status) as f:
        line = f.read().replace("timed False", "timed True")
    with open(status, "w") as f:
        f.write(line)
copy, files = env["N1_FS"] + d, {}
for base, _, names in os.walk(copy):
    for n in names:
        rel = os.path.relpath(os.path.join(base, n), copy)
        if rel != ".custody":
            files[rel] = [os.path.getsize(f"{copy}/{rel}"), hashlib.sha256(open(f"{copy}/{rel}", "rb").read()).hexdigest()]
with open(f"{copy}/.custody", "w") as f:
    json.dump({"schema": "research/run-custody/v0.1", "preserved": True, "verified_from": 4102444800.0, "files": files}, f)
'''
SETTLED = 1_000_000_000                         # a run dir's files last changed in 2001
N2_NOW = 1_790_996_400                          # the scripts' clock (2026-10-03T03:00:00Z): `date -u +FORMAT` reads FAKE_CLOCK
_FAKE_DATE = '#!/bin/sh\nif [ "$1" = -u ] && [ $# = 2 ] && [ "${2#+}" != "$2" ]; then exec %s -u -d "@$(cat "$FAKE_CLOCK")" "$2"; fi\nexec %s "$@"\n'


def _fill_status(fill: Path, t: int, line: str = "free GPUs 0/8, kept free -, waiters -, timed False, window waiting False") -> None:
    """Node 2's fill runner's status.txt, as it writes it every 10 s, stamped `t`."""
    fill.mkdir(exist_ok=True)
    (fill / "status.txt").write_text(f"{datetime.datetime.fromtimestamp(t, datetime.UTC):%Y-%m-%dT%H:%M:%SZ} {line}\nqueued -\n")


def _n2_custody(tmp_path: Path) -> types.SimpleNamespace:
    """Node 2 for n2_custody.sh: `run` makes a run dir under its runs root, `round_` runs one round against the fake node 1
    (_N2_SSH) at `clock` (default N2_NOW) with a fresh fill status unless `status=False`, `calls` is what the rounds asked of node 1,
    `job_err` fails the named runs' Jobs with a line each."""
    root, state, bin_, n1 = tmp_path / "runs", tmp_path / "state", tmp_path / "bin", tmp_path / "n1"
    log, errs, clock_file, fill = tmp_path / "calls.log", tmp_path / "job-err.json", tmp_path / "clock", tmp_path / "fill"
    bin_.mkdir()
    for name, src in (("rsync", _N2_RSYNC), ("ssh", _N2_SSH)):
        (bin_ / name).write_text(f"#!{sys.executable}\n{src}")
        (bin_ / name).chmod(0o755)
    (bin_ / "date").write_text(_FAKE_DATE % (shutil.which("date"), shutil.which("date")))
    (bin_ / "date").chmod(0o755)

    def run(rid, status, *, settled=True, custody=None, extra=0):
        d = root / rid
        d.mkdir(parents=True)
        (d / "job.json").write_text(json.dumps({"store": {"campaign": "c1"}}))
        (d / "status.json").write_text(json.dumps(status))
        if custody is not None:
            (d / ".custody").write_text(json.dumps(custody))
        for i in range(extra):
            (d / f"row{i}.jsonl").write_text("{}\n")
        for p in d.iterdir():
            if settled:
                os.utime(p, (SETTLED, SETTLED))
        return d

    def round_(*args, clock=N2_NOW, status=True, **env):
        clock_file.write_text(str(clock))
        if status:
            _fill_status(fill, clock - 5)
        return subprocess.run(["bash", str(NEBIUS / "n2_custody.sh"), *args], capture_output=True, text=True,
                              env={**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "VY_N1": "n1.invalid", "FAKE_LOG": str(log),
                                   "N1_FS": str(n1), "JOB_ERR": str(errs), "VY_CUSTODY_ROOTS": f"{root}=/n1/runs", "FAKE_CLOCK": str(clock_file),
                                   "VY_CUSTODY_FILL": str(fill), "VY_CUSTODY_STATE": str(state), "VY_CUSTODY_SRC": "/n1/src/abc",
                                   "VY_CUSTODY_MAX_FILES": "4", **env})

    return types.SimpleNamespace(root=root, state=state, n1=n1, fill=fill, run=run, round_=round_,
                                 calls=lambda: log.read_text() if log.exists() else "",
                                 job_err=lambda by_run: errs.write_text(json.dumps(by_run)),
                                 names=lambda sub: sorted(p.name for p in (state / sub).iterdir()))


N2_CUSTODY = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("pgrep") is None or not Path("/proc/self/stat").exists(),
                                reason="n2_custody.sh is bash, and reads /proc and pgrep")


@N2_CUSTODY
def test_n2_custody_takes_every_ended_run_home_with_its_run_record_in_bounded_rounds(tmp_path):
    """Node 2's store has no remote: a night of runs there sat on the node (2-3 Oct) until a push loop took them, attempts without
    their logs, and on 3 Oct that loop's row sync mirrored 1.2M files and overran its window. Every ended run goes to node 1's Job
    with its run record, a runner killed before it recorded an end with --force, and comes back with the .custody that Job's verify
    wrote; a live one, one still settling and one its own custody preserved don't. A round pushes at most VY_CUSTODY_MAX_RUNS within
    VY_CUSTODY_PASS_S, runs that failed before going last; a run dir over the file bound is held, never mirrored, and so is one whose
    attempt node 1 holds without a run record; a failed push is never given up and fails the round until it succeeds. While the
    stopgap loop runs, while another round holds the lock (logged, so a round that never ran is seen), or while node 1 lacks the
    research tree, a round refuses."""
    n2 = _n2_custody(tmp_path)
    root, state = n2.root, n2.state
    R = {k: f"r20261003-{100000 + i}-{i:04x}" for i, k in enumerate(
        ("done", "dead", "live", "retimed", "unread", "fresh", "preserved", "fails", "big", "legacy", "new"), 1)}
    me = int(open("/proc/self/stat").read().rsplit(")", 1)[1].split()[19])
    n2.run(R["done"], {"state": "done"})
    n2.run(R["dead"], {"state": "running", "transitions": [{"pid": 2**22 + 7, "starttime_ticks": 1}]})
    n2.run(R["live"], {"state": "running", "transitions": [{"pid": os.getpid(), "starttime_ticks": me}]})
    n2.run(R["retimed"], {"state": "running", "pid": os.getpid(), "starttime_ticks": me,     # status.json after `tele set-timeout`
                          "transitions": [{"state": "running", "pid": os.getpid(), "starttime_ticks": me}, {"state": "running", "timeout_s": 60}]})
    n2.run(R["unread"], {"state": "running", "transitions": [{"state": "running", "pid": "?"}]})
    n2.run(R["fresh"], {"state": "failed"}, settled=False)
    n2.run(R["preserved"], {"state": "done"}, custody={"preserved": True})
    n2.run(R["fails"], {"state": "failed"})
    n2.run(R["big"], {"state": "done"}, extra=3)
    n2.run(R["legacy"], {"state": "done"})
    n2.job_err({R["legacy"]: f"{R['legacy']}: NO custody on the remote: the attempt names no run record"})
    rogue = n2.run("r20261003-100012-000c; touch pwned", {"state": "done"})       # not a run id: never reaches node 1's shell

    assert n2.round_("--now").returncode == 2
    dry = n2.round_("--dry-run")
    assert dry.returncode == 0 and not n2.calls() and not n2.names("pushed")
    assert f"would push {root}/{R['done']} (c1, done, 2 files" in dry.stdout and f"would push {root}/{R['dead']} (c1, dead" in dry.stdout
    assert f"would hold {root}/{R['big']} (c1, done): over 4 files" in dry.stdout
    assert not [k for k in ("live", "retimed", "unread", "fresh", "preserved") if R[k] in dry.stdout] and rogue.name not in dry.stdout

    (state / "push.sh").write_text("read x\n")
    loop = subprocess.Popen(["bash", str(state / "push.sh")], stdin=subprocess.PIPE)
    try:
        refused = n2.round_()
    finally:
        loop.stdin.close()
        loop.wait()
    assert refused.returncode == 3 and "refused: the stopgap loop" in refused.stdout and not n2.calls()
    with open(state / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = n2.round_()
    assert locked.returncode == 3 and f"refused: another round holds {state}/.lock" in (state / "log").read_text()
    assert not n2.calls()
    no_src = n2.round_(NO_SRC="1")
    assert no_src.returncode == 3 and "no research tree at n1.invalid:/n1/src/abc" in no_src.stdout and "rsync" not in n2.calls()
    writable = n2.round_(UNTRUSTED="/n1/src/abc/tools/research/src/research/store")
    assert writable.returncode == 3 and "rsync" not in n2.calls()
    assert "refused: n1.invalid:/n1/src/abc/tools/research/src/research/store could be changed by a uid other than root" in writable.stdout
    assert "find /n1/src/abc/tools/research /workspace/custody/python -path /workspace/custody/python/.lock -prune" in n2.calls()
    unreached = n2.round_(UNREACHABLE="1")
    assert unreached.returncode == 3 and "refused: node 1 not reached (`ssh n1.invalid` exited 255)" in unreached.stdout

    assert n2.round_(VY_CUSTODY_PASS_S="0").returncode == 0 and "rsync" not in n2.calls()
    bounded = n2.round_(VY_CUSTODY_MAX_RUNS="1")
    assert bounded.returncode == 0 and "round bounded (1 push(es)" in bounded.stdout and "9 run dir(s) left" in bounded.stdout
    assert n2.names("pushed") == [R["done"]]
    assert json.loads((root / R["done"] / ".custody").read_text())["preserved"] is True

    n2.job_err({R["legacy"]: f"{R['legacy']}: NO custody on the remote: the attempt names no run record", R["fails"]: "Error: boom"})
    first = n2.round_()
    assert first.returncode == 1 and f"{R['fails']} (c1, failed): push failed (1 so far): node 1's Job: Error: boom" in first.stdout
    assert n2.names("pushed") == [R["done"], R["dead"], R["preserved"]]
    assert n2.names("held") == [R["big"], R["legacy"]] and "2 run(s) held" in first.stdout
    assert "the attempt names no run record. node 1's store holds its attempt without a run record" in first.stdout
    calls = n2.calls()
    assert f"{root}/{R['done']}/ n1.invalid:/n1/runs/{R['done']}/" in calls and R["fresh"] not in calls
    assert not [k for k in ("live", "retimed", "unread") if R[k] in calls], "a live run is never published as dead"
    assert f"data custody /n1/runs/{R['dead']} --publish --keep-local --force --runner-gone" in calls
    assert f'data custody /n1/runs/{R["done"]} --publish --keep-local  & wait $!"]' in calls
    assert "PYTHONPATH=/n1/src/abc/tools/research/src" in calls and R["preserved"] not in calls and R["big"] not in calls
    assert ('"bash", "-c", "trap \'exit 143\' TERM; cd / || exit 1; HOME=/tmp PYTHONDONTWRITEBYTECODE=1 ' in calls and "uv run" not in calls
            and " /workspace/custody/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12 -m research data custody " in calls)
    assert "touch pwned" not in calls and rogue.name not in n2.names("pushed") + n2.names("held")
    keepalive = "ssh -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "
    assert all(x.startswith(keepalive) for x in calls.splitlines() if x.startswith("ssh ")), "a dropped link ends every call"
    assert "kubectl --request-timeout=30s create -f -" in calls and "kubectl --request-timeout=30s get job" in calls

    n2.run(R["new"], {"state": "done"})
    n2.job_err({})
    newer = n2.round_(VY_CUSTODY_MAX_RUNS="1")
    assert (state / "pushed" / R["new"]).exists() and not (state / "pushed" / R["fails"]).exists(), "a failing run goes last"
    assert newer.returncode == 1 and "1 run dir(s) left" in newer.stdout
    second = n2.round_()
    assert second.returncode == 1 and f"{R['fails']} (c1, failed" in second.stdout and not (state / "failures" / R["fails"]).exists()
    assert (state / "pushed" / R["fails"]).exists() and n2.calls().count(f"{R['done']}/ n1.invalid") == 1, "a pushed run is done"
    for p in (state / "held").iterdir():
        p.rename(state / "pushed" / p.name)
    assert n2.round_().returncode == 0


@N2_CUSTODY
def test_n2_custody_holds_what_no_retry_can_mend_each_with_its_reason(tmp_path):
    """A Job that fails the same way every round is held, with its line and what a person decides, and the rest are retried.
    r20261003-174723-7ea5's runner pushed its attempt and run record, then lean-audit.log changed, and the stopgap published an
    attempt without a run record on node 1: R2's write-once key refuses that, so waiving its logs is the wrong advice. An attempt
    whose run record lacks what the run dir holds now never changes either."""
    n2 = _n2_custody(tmp_path)
    legacy, conflict, stale, differs, flaky = (n2.run(f"r20261003-10000{i}-000{i}", {"state": "failed"}) for i in range(1, 6))
    lines = {legacy.name: f"{legacy.name}: NO custody on the remote: the attempt names no run record (published without custody)",
             conflict.name: (f"{conflict.name}: NO custody on the remote: the attempt names no run record (published without custody); "
                             f"push: RemoteConflict: attempts/{conflict.name}.json: remote holds 4171 bytes, local is 2655"),
             stale.name: f"{stale.name}: NO custody on the remote: 0 file(s) not in the run record, 1 changed since it was taken: x.log",
             differs.name: (f"{differs.name}: NO custody on the remote: not preserved on the remote: attempts/{differs.name}.json: the "
                            "remote holds a DIFFERENT attempt document for this run id"),
             flaky.name: "error: the server was unable to return a response in the time allotted"}
    n2.job_err(lines)
    first = n2.round_()
    assert first.returncode == 1 and n2.names("held") == [legacy.name, conflict.name, stale.name, differs.name]
    assert n2.names("failures") == [flaky.name] and "4 run(s) held" in first.stdout
    for d in (legacy, conflict, stale, differs):
        assert (n2.state / "held" / d.name).read_text() == f"node 1's Job: {lines[d.name]}\n"
    said = {d.name: next(x for x in first.stdout.splitlines() if f"{d} (c1, failed): held: " in x) for d in (legacy, conflict, stale, differs)}
    assert "Label it `custody waived`" in said[legacy.name] and "custody waived" not in said[conflict.name] + said[stale.name]
    assert f"R2 already holds another attempt for {conflict.name}" in said[conflict.name] and f"{conflict}/.custody names its" in said[conflict.name]
    assert f"on node 1, `sudo -u research vy-custody --reconcile {conflict.name} --by WHO`" in said[conflict.name]
    assert f"rm {n2.state}/held/{conflict.name} instead, and the next round brings its .custody home" in said[conflict.name]
    assert "R2 already holds another attempt" in said[differs.name] and "lacks what the run dir holds now" in said[stale.name]
    assert said[legacy.name].endswith(f"then mv {n2.state}/held/{legacy.name} {n2.state}/pushed/")
    n2.job_err({})
    again = n2.round_()
    assert again.returncode == 1 and n2.names("pushed") == [flaky.name] and "4 run(s) held" in again.stdout
    assert all(n2.calls().count(f"{d}/ n1.invalid") == 1 for d in (legacy, conflict, stale, differs)), "a held run is not retried"


@N2_CUSTODY
def test_n2_custody_leaves_a_run_to_its_runners_own_custody_until_the_runners_key_expires(tmp_path):
    """A `research run --custody-r2` runner writes nothing into the run dir between the workload's end and its .custody (one took
    11 min, past the settle time), so a round raced it. While the runner holds its key (<root>/../requests/RUN/custody/cred.json,
    which forget_credential removes) the run is its runner's; a key a dead runner left counts until it expires, one that doesn't
    read for 24 h from when it was written."""
    n2 = _n2_custody(tmp_path)
    mine, left, garbled, old = (n2.run(f"r20261003-10000{i}-000{i}", {"state": "done"}) for i in range(1, 5))
    keys = {d: tmp_path / "requests" / d.name / "custody" / "cred.json" for d in (mine, left, garbled, old)}
    for d, cred in ((mine, '{"expires_utc": "2100-01-01T00:00:00Z"}'), (left, '{"expires_utc": "2001-09-09T01:46:40Z"}'),
                    (garbled, "{"), (old, "{")):
        keys[d].parent.mkdir(parents=True)
        keys[d].write_text(cred)
    os.utime(keys[old], (SETTLED, SETTLED))
    dry = n2.round_("--dry-run")
    assert f"would push {left}" in dry.stdout and f"would push {old}" in dry.stdout
    assert mine.name not in dry.stdout and garbled.name not in dry.stdout
    assert n2.round_().returncode == 0 and n2.names("pushed") == [left.name, old.name]
    keys[mine].unlink()
    assert n2.round_().returncode == 0 and n2.names("pushed") == [mine.name, left.name, old.name]


@N2_CUSTODY
def test_n2_custody_waits_on_a_job_an_earlier_round_left_running_and_never_deletes_it(tmp_path):
    """A round whose ssh dropped leaves its Job running on node 1, still pushing node 1's copy. The next round deleted that Job, push
    and all, and rsynced under it; it waits on it instead, and the marker that Job writes comes home only over node 2's files."""
    n2 = _n2_custody(tmp_path)
    d = n2.run("r20261003-100001-0001", {"state": "done"})
    n2.job_err({d.name: "Job not done in 30 min"})
    assert n2.round_().returncode == 1 and n2.calls().count(f"delete job n2-custody-{d.name}") == 1
    n2.job_err({})
    again = n2.round_(ACTIVE_JOB=d.name)
    assert again.returncode == 0 and n2.names("pushed") == [d.name] and CU.marker_problem(d) is None
    assert f"node 1's Job n2-custody-{d.name}, from an earlier round, has not finished: waiting on it" in again.stdout
    calls = n2.calls()
    assert calls.count(f"{d}/ n1.invalid") == 1 and calls.count(f"delete job n2-custody-{d.name}") == 1 and " n1.invalid true &&" in calls


@N2_CUSTODY
def test_n2_custody_asks_before_every_push_whether_the_stopgap_loop_runs(tmp_path):
    """The stopgap loop takes no lock, and a round asked for it once, at its start: a loop started during a round's 20-min pass (tmux
    custody-push-3 restarted) pushed beside it, both deleting and creating the same Jobs. Every push asks again."""
    n2 = _n2_custody(tmp_path)
    first, second = (n2.run(f"r20261003-10000{i}-000{i}", {"state": "done"}) for i in (1, 2))
    n2.state.mkdir()
    os.mkfifo(tmp_path / "gate")
    (n2.state / "push.sh").write_text(f"read x <> {tmp_path / 'gate'}\n")
    try:
        refused = n2.round_(LOOP_DURING_JOB=first.name)
    finally:
        if (n2.state / "push.pid").exists():
            os.kill(int((n2.state / "push.pid").read_text()), signal.SIGKILL)
    assert refused.returncode == 3 and n2.names("pushed") == [first.name] and second.name not in n2.calls()
    assert f"refused: the stopgap loop {n2.state}/push.sh still runs" in refused.stdout


@N2_CUSTODY
def test_n2_custody_takes_node_1s_marker_home_only_over_the_files_node_1_verified(tmp_path):
    """Node 1's .custody vouches for the copy the rsync left there, on node 1's clock. A line appended on node 2 after the rsync and
    before the Job's verify is in no run record, yet the marker listed the file and was dated after the append, so node 2's triage
    called the run r2. It comes home only over files with the size and sha256 it lists, dated by node 2's clock. A run still writing
    any file (its sampler, a late log line) is settling, whatever its status.json says."""
    n2 = _n2_custody(tmp_path)
    ok, late, busy = (n2.run(f"r20261003-10000{i}-000{i}", {"state": "done"}, extra=1) for i in (1, 2, 3))
    (busy / "row0.jsonl").write_text("{}\n{}\n")
    first = n2.round_(DURING_JOB=f"{late.name}:{late / 'row0.jsonl'}")
    assert first.returncode == 1 and n2.names("pushed") == [ok.name] and busy.name not in n2.calls()
    assert json.loads((ok / ".custody").read_text())["verified_from"] < 4102444800.0, "dated by node 2's clock, not node 1's"
    assert CU.marker_problem(ok) is None
    assert (f"{late} (c1, done): push failed (1 so far): node 1's .custody not taken home: row0.jsonl differs from node 1's copy: "
            "changed here after the rsync") in first.stdout
    assert not (late / ".custody").exists() and not (late / ".custody.tmp").exists() and CU.marker_problem(late) == "no .custody"
    second = n2.round_()
    assert second.returncode == 0 and n2.calls().count(f"{late}/ n1.invalid") == 1, "a file of it changed: settling again"


@N2_CUSTODY
def test_n2_custody_pauses_for_node_2s_timed_windows(tmp_path):
    """Node 2's timed windows (PoUW's measurements) run with nothing else on the node, and a round's rsync and hashing ran through
    them at Nice=19 (#984's red-team finding 13). A round, and each push, starts only while the fill runner's status is under 60 s
    old and says no window runs or waits, and no booked window is on or starts within the round's bound (PASS_S + PUSH_S: 80 min);
    a status that is missing, stale or doesn't read is a window. A skipped or paused round is no failure: one log line, exit 0."""
    n2 = _n2_custody(tmp_path)
    first, second = (n2.run(f"r20261003-10000{i}-000{i}", {"state": "done"}) for i in (1, 2))
    skips = []
    for line in ("free GPUs 0/8, kept free 8, waiters -, timed True, window waiting False",
                 "free GPUs 2/8, kept free 8, waiters -, timed False, window waiting True"):
        _fill_status(n2.fill, N2_NOW - 5, line)
        skips.append(n2.round_(status=False))
    _fill_status(n2.fill, N2_NOW - 61)
    skips.append(n2.round_(status=False))
    (n2.fill / "status.txt").write_text("free GPUs 0/8, kept free -, waiters -, timed False, window waiting False\n")
    skips.append(n2.round_(status=False))
    (n2.fill / "status.txt").unlink()
    skips.append(n2.round_(status=False))
    (n2.fill / "windows").write_text("# booked windows\n2026-10-03T04:15Z 30 # a measurement, 75 min on\n")
    skips.append(n2.round_())
    skips.append(n2.round_("--dry-run"))
    (n2.fill / "windows").write_text("2026-10-03T02:45Z 30 # on since 02:45\n")
    skips.append(n2.round_())
    said = [s.stdout for s in skips]
    assert all(s.returncode == 0 and "skipped: node 2's timed window: " in s.stdout for s in skips), said
    assert not n2.calls() and not n2.names("pushed") and "would push" not in said[6]
    assert "timed True" in said[0] and "window waiting True" in said[1] and "its fill status is 61 s old" in said[2]
    assert "its fill status doesn't read (ValueError" in said[3] and "its fill status doesn't read (FileNotFoundError" in said[4]
    assert "a window booked at 2026-10-03T04:15Z for 30 min is on or starts within the round's 80 min" in said[5]
    assert "a window booked at 2026-10-03T02:45Z" in said[7]

    (n2.fill / "windows").write_text("2026-10-03T02:00Z 30 # ended\n2026-10-03T04:25Z 30 # past the round's bound\n")
    paused = n2.round_(TIMED_DURING_JOB=first.name)
    assert paused.returncode == 0 and n2.names("pushed") == [first.name] and second.name not in n2.calls()
    assert "paused after 1 push(es), for node 2's timed window: its fill status: " in paused.stdout
    assert "1 run dir(s) left for the next round" in paused.stdout
    assert n2.round_().returncode == 0 and n2.names("pushed") == [first.name, second.name]


@N2_CUSTODY
def test_n2_custody_starts_no_job_that_could_run_into_node_1s_quiet_hour(tmp_path):
    """n2_custody.sh's Jobs run on node 1, yet it started them through node 1's daily hold (12:10-13:30Z), which n1_custody.sh
    keeps, and a Job started at 12:09Z could run to 13:09Z (activeDeadlineSeconds 3600), into node 1's quiet hour. No push starts in
    the hold, or unless its rsync (20 min) and a Job of 5 min end before 12:30Z, and its Job's deadline ends it before 12:30Z."""
    n2 = _n2_custody(tmp_path)
    a, b, c = (n2.run(f"r20261003-10000{i}-000{i}", {"state": "done"}) for i in (1, 2, 3))
    at = lambda hhmm: N2_NOW + (int(hhmm[:2]) - 3) * 3600 + int(hhmm[2:]) * 60          # N2_NOW is 03:00Z
    for hhmm in ("1210", "1229", "1230", "1329", "1206"):
        held = n2.round_(clock=at(hhmm))
        assert held.returncode == 0 and "skipped: node 1's daily hold (1210-1330 UTC; no Job of it runs past 1230Z" in held.stdout, hhmm
    assert not n2.calls() and f"would push {a}" in n2.round_("--dry-run", clock=at("1215")).stdout, "a dry run starts no Job"

    assert n2.round_(clock=at("1150"), VY_CUSTODY_MAX_RUNS="1").returncode == 0 and n2.names("pushed") == [a.name]
    assert '"activeDeadlineSeconds": 2330,' in n2.calls(), "40 min to 12:30Z, less the minute and the grace"
    cut = n2.round_(clock=at("1203"), CLOCK_AFTER_RSYNC=str(at("1226")))
    assert cut.returncode == 0 and n2.names("pushed") == [a.name] and f"data custody /n1/runs/{b.name}" not in n2.calls()
    assert ("paused after 0 push(es), for node 1's daily hold (1210-1330 UTC; no Job of it runs past 1230Z, where node 1's quiet hour "
            "starts): 0s left, the copy waits on node 1: 2 run dir(s) left for the next round") in cut.stdout
    late = n2.round_(clock=at("1331"))
    assert late.returncode == 0 and n2.names("pushed") == [a.name, b.name, c.name]
    assert n2.calls().count('"activeDeadlineSeconds": 3600,') == 2
    assert n2.round_(clock=at("1215"), VY_CUSTODY_QUIET="").returncode == 0, "VY_CUSTODY_QUIET='' turns the hold off"


# a stand-in kubectl: a created n1-custody Job runs its command at once, against the fs vault KVAULT names (the Secret's R2); `get pod`
# finds the pods LIVE_PODS lists, or can't reach the API server under POD_ERR
FAKE_KUBECTL = '''
import json, os, subprocess, sys
from pathlib import Path
jobs, log, a = Path(os.environ["KSTATE"]), Path(os.environ["KLOG"]), sys.argv[1:]
with log.open("a") as f:
    f.write("kubectl " + " ".join(a) + "\\n")
if a[:2] == ["get", "pod"]:
    if os.environ.get("POD_ERR"):
        sys.exit("Unable to connect to the server: dial tcp 10.0.0.1:6443: i/o timeout")
    if a[2] not in os.environ.get("LIVE_PODS", "").split(","):
        sys.exit(f'Error from server (NotFound): pods "{a[2]}" not found')
    print(f"pod/{a[2]}")
elif a[:2] == ["delete", "job"]:
    (jobs / a[2]).unlink(missing_ok=True)
elif a[:3] == ["create", "-f", "-"]:
    job = json.load(sys.stdin)
    name, spec = job["metadata"]["name"], job["spec"]["template"]["spec"]
    with log.open("a") as f:
        f.write(f"created {name}\\n")
    rc, out = 0, ""
    if name.startswith("n1-custody-"):
        cmd = spec["containers"][0]["command"]
        if os.environ.get("FAIL_RUN") and os.environ["FAIL_RUN"] in name:
            rc, out = 1, "Error: boom\\n"
        else:
            p = subprocess.run(["bash", "-c", cmd[5].replace("HOME=/tmp", "HOME=" + os.environ["KHOME"]), *cmd[6:]], capture_output=True,
                               text=True, stdin=subprocess.DEVNULL, env={**os.environ, "RESEARCH_STORE_CONFIG": os.environ["KVAULT"]})
            rc, out = p.returncode, p.stdout + p.stderr
    (jobs / name).write_text(json.dumps({"job": job, "rc": rc, "out": out}))
elif a[:2] == ["get", "job"]:
    if not (jobs / a[2]).exists():
        sys.exit(f'Error from server (NotFound): jobs.batch "{a[2]}" not found')
    failed = json.loads((jobs / a[2]).read_text())["rc"] or (os.environ.get("FAIL_PENDING") and a[2].startswith("n1-store-push-"))
    print("Failed=True " if failed else "Complete=True ", end="")
elif a[0] == "logs":
    print("\\n".join(json.loads((jobs / a[1].split("/", 1)[1]).read_text())["out"].splitlines()[-int(a[2].split("=")[1]):]))
'''


N1_OLD, N1_FUTURE = 1_000_000_000, 4_102_444_800     # a run's records last changed in 2001 (settled, and stale for a pod), or in 2100
N1_CUSTODY = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("flock") is None or not Path("/proc/self/stat").exists(),
                                reason="n1_custody.sh is bash, and takes a flock")


def _n1_custody(tmp_path: Path) -> types.SimpleNamespace:
    """Node 1 for n1_custody.sh: its runs roots `jobs` (store uid 1000) and `research` (1001), node 2's relay copies under jobs/runs-n2,
    and the Secret's R2 as the fs vault `vault`. `run` makes a run dir from the run_v0.2 fixture (a pod's, pod-RUN, unless `host`),
    `round_` runs one round through FAKE_KUBECTL and a sudo that logs its argv (at `clock`, when given, as _n2_custody's), `created`
    lists the Jobs created so far, `job` reads one back with its outcome, `command` its container's command, `held` the holds."""
    src_root = Path(registry.__file__).resolve().parents[2]
    n = types.SimpleNamespace(jobs=tmp_path / "jobs", research=tmp_path / "research", state=tmp_path / "state", vault=tmp_path / "vault",
                              log=tmp_path / "calls.log", absent=tmp_path / "absent.toml", remote=tmp_path / "vault.toml")
    bin_, kjobs, home = tmp_path / "bin", tmp_path / "k8s", tmp_path / "home"
    for d in (n.jobs / "runs", n.research / "runs", bin_, kjobs, n.vault, home, tmp_path / "py" / "bin"):
        d.mkdir(parents=True)
    # the round runs only a tree and a Python that no other uid can change: copies, not links into the checkout
    shutil.copytree(src_root, tmp_path / "src" / "tools" / "research" / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(src_root.parent / "store.pod.toml", tmp_path / "src" / "tools" / "research" / "store.pod.toml")
    (tmp_path / "py" / "bin" / "python3").write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    (tmp_path / "py" / "bin" / "python3").chmod(0o755)
    for p in (tmp_path, tmp_path / "src", *(tmp_path / "src").rglob("*"), tmp_path / "py", *(tmp_path / "py").rglob("*")):
        p.chmod(p.stat().st_mode & ~0o022)       # tmp_path too: a subdirectory a test makes takes the node's umask (002 on node 1)
    n.src, n.py = tmp_path / "src", tmp_path / "py" / "bin" / "python3"
    n.remote.write_text(f'[remote]\ntype = "fs"\npath = "{n.vault}"\n')
    (bin_ / "kubectl").write_text(f"#!{sys.executable}\n{FAKE_KUBECTL}")
    (bin_ / "sudo").write_text(f'#!/bin/sh\necho "sudo $*" >> {n.log}\n[ "$1 $2" = "-n -u" ] || exit 1\nshift 3\nexec "$@"\n')
    (bin_ / "python3").write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    (tmp_path / "datebin").mkdir()
    (tmp_path / "datebin" / "date").write_text(_FAKE_DATE % (shutil.which("date"), shutil.which("date")))
    for f in (*bin_.iterdir(), tmp_path / "datebin" / "date"):
        f.chmod(0o755)

    def run(root, rid, state, *, boot="boot-n1", host=None, t=N1_OLD, empty=False, extra=0, runs="runs", status=None,
            stdout="the workload's stdout\n", files=None):
        d = root / runs / rid
        if empty:
            d.mkdir()
            (d / "stdout.log").write_text("")
            (d / "stderr.log").write_text("")
            job = json.loads((Path(__file__).parent / "fixtures" / "run_v0.2" / "job.json").read_text())
        else:
            shutil.copytree(Path(__file__).parent / "fixtures" / "run_v0.2", d)
            (d / "state.json").unlink()
            (d / "stdout.log").write_text(stdout)
            job = json.loads((d / "job.json").read_text())
        job["run"]["id"], job["environment"]["boot_id"], job["environment"]["hostname"] = rid, boot, host or f"pod-{rid}"
        (d / "job.json").write_text(json.dumps(job))
        (d / "status.json").write_text(json.dumps(status or {"state": state, "transitions": [{"state": state, "pid": 4243}]}))
        for i in range(extra):
            (d / f"row{i}.jsonl").write_text("{}\n")
        for rel, text in (files or {}).items():
            (d / rel).write_text(text)
        for p in d.rglob("*"):
            os.utime(p, (t, t))
        return d

    def round_(*args, clock=None, **env):
        if clock is not None:                                   # `date -u +FORMAT` reads FAKE_CLOCK
            (tmp_path / "clock").write_text(str(clock))
            env = {"FAKE_CLOCK": str(tmp_path / "clock"), **env}
        path = f"{tmp_path / 'datebin'}:{bin_}" if clock is not None else str(bin_)
        return subprocess.run(["bash", str(NEBIUS / "n1_custody.sh"), *args], capture_output=True, text=True,
                              env={**os.environ, "PATH": f"{path}:{os.environ['PATH']}", "HOME": str(home), "KSTATE": str(kjobs),
                                   "KLOG": str(n.log), "KHOME": str(home), "KVAULT": str(n.remote), "RESEARCH_STORE_CONFIG": str(n.absent),
                                   "VY_CUSTODY_QUIET": "", "VY_CUSTODY_BOOT": "boot-n1", "VY_CUSTODY_PY": str(n.py),
                                   "VY_CUSTODY_PYHOME": str(tmp_path / "py"), "VY_CUSTODY_STATE": str(n.state), "VY_CUSTODY_SRC": str(n.src),
                                   "VY_CUSTODY_ROOTS": f"{n.jobs}/runs={n.jobs}/store=1000,{n.research}/runs={n.research}/store=1001",
                                   "VY_CUSTODY_RELAY": f"{n.jobs}/runs-n2={n.jobs}/store=1000", "VY_CUSTODY_MAX_FILES": "10", **env})

    n.run, n.round_ = run, round_
    n.created = lambda: [ln.split()[1] for ln in n.log.read_text().splitlines() if ln.startswith("created ")] if n.log.exists() else []
    n.job = lambda name: json.loads((kjobs / name).read_text())
    n.command = lambda name: n.job(name)["job"]["spec"]["template"]["spec"]["containers"][0]["command"]
    n.held = lambda: sorted(p.name for p in (n.state / "held").iterdir())
    return n


@N1_CUSTODY
def test_n1_custody_takes_node_1s_own_ended_runs_to_r2_with_their_run_record_in_bounded_rounds(tmp_path, monkeypatch):
    """3 Oct: ~326 runs under node 1's runs roots had ended, a pod preempted before its runner recorded an end among them, and sat
    there without custody. A round takes each run this machine ran, once it has settled, through a Job as its store's owner, with
    the Secret's R2: `research data custody --publish --keep-local`, --force --runner-gone for a status that never became terminal
    once Kubernetes says its pod is gone. A run that never ran its workload is held (never published as evidence) until a label
    lifts it, and so is one over the bounds and one whose remote holds another attempt document (after its one Job), until
    `--reconcile`. Node 2's relay copies, live runs, settling runs and an attempt published without its run record get no Job (one
    R2 lacks is held record-less, and the pending push leaves it out, until a person rules). A
    round runs at most VY_CUSTODY_MAX_RUNS Jobs within VY_CUSTODY_PASS_S, runs that failed before going last, none in node 1's daily
    hold, and starts each store's pending push."""
    from research.store import attempt as A
    from research.store.cli import main as data_main
    from research.store.local import LocalStore, attempt_key

    node = _n1_custody(tmp_path)
    jobs, research, state, log, vault, absent = node.jobs, node.research, node.state, node.log, node.vault, node.absent
    run, round_, created, job, command, held = node.run, node.round_, node.created, node.job, node.command, node.held
    old, future = N1_OLD, N1_FUTURE

    def preserved(d):
        return json.loads((d / ".custody").read_text())["preserved"] is True

    R = {k: f"r20261003-0000{n:02d}-00{n:02d}" for n, k in enumerate(
        ("done", "dead", "empty", "n2copy", "fresh", "fails", "big", "legacy", "conflict", "live", "new", "research", "relay"), 1)}
    n1 = {k: f"n1-custody-{r}" for k, r in R.items()}
    J = {k: jobs / "runs" / r for k, r in R.items()}
    J["relay"] = jobs / "runs-n2" / R["relay"]
    run(jobs, R["done"], "done")
    run(jobs, R["dead"], "running")                       # a preempted pod: no record changed for 24 h
    run(jobs, R["empty"], "submitted", empty=True, status={"state": "submitted"})      # its runner never spawned the workload
    run(jobs, R["n2copy"], "done", boot="boot-n2")        # node 2's relay copy: n2_custody.sh's
    run(jobs, R["fresh"], "failed", t=future)
    run(jobs, R["fails"], "failed")
    run(jobs, R["big"], "done", extra=4)
    run(jobs, R["legacy"], "done")
    run(jobs, R["conflict"], "done")
    run(jobs, R["live"], "running", t=future)
    run(research, R["research"], "done")
    run(jobs, R["relay"], "done", boot="boot-n2", runs="runs-n2")
    offline = LocalStore(jobs / "store", config_path=absent)
    assert A.publish(J["legacy"], store=offline, remote_push=False, run_record=False)["published"]
    for k in ("conflict", "relay"):
        assert A.publish(J[k], store=offline, remote_push=False, run_record=True)["published"]
        other = json.loads(offline.attempt_path(R[k]).read_bytes())
        other["status"]["rc"] = 99
        (vault / attempt_key(R[k])).parent.mkdir(parents=True, exist_ok=True)
        (vault / attempt_key(R[k])).write_text(json.dumps(other, sort_keys=True, separators=(",", ":")))
    for d in (J["legacy"], J["conflict"], J["relay"]):
        for p in d.rglob("*"):
            os.utime(p, (old, old))

    assert round_("--now").returncode == 2 and round_("--reconcile", R["done"], "--by").returncode == 2
    assert round_("--reconcile", f"../{R['done']}", "--by", "x").returncode == 2
    no_src = round_(VY_CUSTODY_SRC=str(tmp_path / "nowhere"))
    assert no_src.returncode == 3 and f"refused: no research tree at {tmp_path}/nowhere" in no_src.stdout
    quiet = round_(VY_CUSTODY_QUIET="0000-2400")
    assert quiet.returncode == 0 and "node 1's daily hold" in quiet.stdout and not log.exists()
    assert round_("--reconcile", R["conflict"], "--by", "x", VY_CUSTODY_QUIET="0000-2400").returncode == 3

    dry = round_("--dry-run")
    assert dry.returncode == 0 and created() == [] and not held() and not (state / "boots").exists(), dry.stdout + dry.stderr
    assert f"would publish {J['done']} (r21-fixture, done, 8 files" in dry.stdout
    assert (f"would publish {J['dead']} (r21-fixture, running, 8 files, 5.9 KiB) with --force --runner-gone (its pod pod-{R['dead']} "
            "is NotFound)") in dry.stdout
    assert f"would hold {J['empty']}: never ran its workload (r21-fixture, status submitted" in dry.stdout
    assert f"would hold {J['big']}: over 10 files" in dry.stdout and f"would publish {research}/runs/{R['research']}" in dry.stdout
    assert "no Job: 1 not run by this machine, 1 settling (triage lists them)" in dry.stdout
    assert R["n2copy"] not in dry.stdout and R["fresh"] not in dry.stdout and R["live"] not in dry.stdout
    assert f"would start {jobs}/store's pending push (Job n1-store-push-jobs)" in dry.stdout
    assert "1 record-less attempt(s) held (triage's attempt-unpreserved)" in dry.stdout
    assert "sudo -n -u #1000 env" in log.read_text() and "sudo -n -u #1001 env" in log.read_text()

    unbounded = round_(VY_CUSTODY_PASS_S="0")
    assert unbounded.returncode == 1 and created() == ["n1-store-push-jobs", "n1-store-push-research"], unbounded.stdout
    assert "round bounded (0 Job(s) in" in unbounded.stdout and "5 run(s) left" in unbounded.stdout
    assert held() == [R["empty"], R["big"]] and "2 run(s) held" in unbounded.stdout
    assert f"research data label {R['empty']} custody never-ran --by WHO) and the hold lifts itself" in unbounded.stdout
    push = job("n1-store-push-research")["job"]
    assert push["spec"]["template"]["spec"]["securityContext"] == {"runAsUser": 1001, "runAsGroup": 1001}
    assert push["spec"]["ttlSecondsAfterFinished"] == 3600 and "research data push --pending --jobs 8" in command("n1-store-push-research")[5]
    assert command("n1-store-push-research")[6:] == ["n1-custody", str(node.src), str(node.py), f"{research}/store", f"{research}/runs",
                                                     "--skip-record-less"]

    bounded = round_(VY_CUSTODY_MAX_RUNS="1")
    assert bounded.returncode == 1 and created()[2:] == [n1["done"]] and "4 run(s) left" in bounded.stdout, bounded.stdout
    assert preserved(J["done"]) and LocalStore(jobs / "store", config_path=tmp_path / "vault.toml").get_attempt(R["done"])
    spec = job(n1["done"])["job"]["spec"]
    pod = spec["template"]["spec"]
    assert (spec["backoffLimit"], spec["activeDeadlineSeconds"], pod["nodeSelector"]) == (0, 1800, {"verity.dev/server": "vy-nebius-1"})
    assert pod["securityContext"]["runAsUser"] == 1000 and pod["containers"][0]["envFrom"] == [{"secretRef": {"name": "research-r2"}}]
    assert command(n1["done"])[:5] == ["taskset", "-c", "96-127", "bash", "-c"]
    assert command(n1["done"])[5].endswith('{ "$2" -m research data custody "$5" --publish --keep-local "${@:6}"; } & wait $!')
    assert command(n1["done"])[6:] == ["n1-custody", str(node.src), str(node.py), f"{jobs}/store", f"{jobs}/runs", str(J["done"])]
    assert (state / "boots").read_text() == "boot-n1\n"

    first = round_(FAIL_RUN=R["fails"])
    assert first.returncode == 1, first.stdout + first.stderr
    assert created()[3:] == [n1["dead"], n1["fails"], n1["conflict"], n1["research"]]
    assert command(n1["dead"])[-3:] == [str(J["dead"]), "--force", "--runner-gone"]
    assert f"kubectl get pod pod-{R['dead']} --namespace=default --output=name --request-timeout=30s" in log.read_text()
    assert preserved(J["dead"]) and preserved(research / "runs" / R["research"])
    assert job(n1["research"])["job"]["spec"]["template"]["spec"]["securityContext"]["runAsUser"] == 1001
    assert LocalStore(research / "store", config_path=tmp_path / "vault.toml").get_attempt(R["research"])
    assert f"{R['fails']} (r21-fixture, failed): Job failed (1 so far): Job {n1['fails']}: Error: boom; the next round" in first.stdout
    assert "1 record-less attempt(s) held" in first.stdout and not (vault / attempt_key(R["legacy"])).exists()
    assert f"{R['conflict']} (r21-fixture, done): held: R2 holds another attempt document" in first.stdout
    assert f"vy-custody --reconcile {R['conflict']} --by WHO" in first.stdout
    assert held() == [R["empty"], R["big"], R["conflict"]] and (state / "failures" / R["fails"]).read_text() == "1\n"
    assert not (state / "failures" / R["legacy"]).exists() and not (state / "failures" / R["conflict"]).exists()

    run(jobs, R["new"], "done")
    newer = round_(VY_CUSTODY_MAX_RUNS="1")
    assert created()[7:] == [n1["new"]], "a run that failed before goes last"
    assert newer.returncode == 1 and "1 run(s) left" in newer.stdout, newer.stdout
    second = round_()
    assert created()[8:] == [n1["fails"]] and preserved(J["fails"]) and not list((state / "failures").iterdir())
    assert second.returncode == 1 and "3 run(s) held" in second.stdout

    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(absent))
    assert data_main(["label", R["empty"], "custody", "never-ran", "--by", "test", "--store", str(jobs / "store")]) == 0
    lifted = round_()
    assert f"{R['empty']}: hold lifted" in lifted.stdout and held() == [R["big"], R["conflict"]] and len(created()) == 9, lifted.stdout

    with (state / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        busy = round_("--reconcile", R["conflict"], "--by", "test")
        assert busy.returncode == 3 and "--reconcile after it" in busy.stderr
        assert round_().returncode == 0 and len(created()) == 9, "a round never waits on another"
    assert round_("--reconcile", "r20261003-000099-0099", "--by", "test").returncode == 2
    rec = round_("--reconcile", R["conflict"], "--by", "test")
    assert rec.returncode == 0 and f"{R['conflict']}: reconciled by test" in rec.stdout, rec.stdout + rec.stderr
    assert "node 2's copy" not in rec.stdout
    assert 'reconcile-attempts "$5" --apply &&' in command(n1["conflict"])[5] and command(n1["conflict"])[-2:] == [R["conflict"], str(J["conflict"])]
    assert preserved(J["conflict"]) and list((jobs / "store" / "attempts" / ".superseded").iterdir())
    assert held() == [R["big"]]
    assert R["relay"] not in "".join(created()) and not (J["relay"] / ".custody").exists(), "node 2's copies are n2_custody.sh's"
    relay = round_("--reconcile", R["relay"], "--by", "test")
    assert relay.returncode == 0 and "(node 2's copy: remove n2_custody.sh's hold on node 2" in relay.stdout and preserved(J["relay"])

    (state / "held" / R["big"]).unlink()
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(node.remote))
    assert data_main(["push", R["legacy"], "--store", str(jobs / "store")]) == 0       # a person's ruling: on R2 as it is
    done = round_(VY_CUSTODY_MAX_FILES="20")
    assert done.returncode == 0 and created()[-1] == n1["big"] and preserved(J["big"]), done.stdout
    assert "record-less" not in done.stdout and "1 attempt-no-record" in done.stdout
    assert round_().returncode == 0 and created()[-1] == n1["big"], "nothing left: no Job"
    pending = round_(FAIL_PENDING="1")
    assert pending.returncode == 1 and f"{jobs}/store: the pending push failed (Job n1-store-push-jobs" in pending.stdout


@N1_CUSTODY
def test_n1_custody_forces_a_status_that_never_ended_only_once_its_runner_is_seen_gone(tmp_path):
    """#1004's red-team F1 and F2: a round published a run --force as dead while it ran, and an attempt is write-once, so its real
    end never reached R2. A host run's runner is gone when its pid is (a pid-less `tele set-timeout` transition is no end); a pod's
    when `kubectl get pod` says NotFound, however long its records have been quiet, as backfill_attempts.sh asks. A pod Kubernetes
    still lists waits without failing the round; a kubectl that doesn't answer publishes nothing and fails it; a hostname of 63
    characters may be a longer pod name cut short, so its NotFound proves nothing."""
    import socket

    node = _n1_custody(tmp_path)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    me = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    R = {k: f"r20261003-2000{n:02d}-0a{n:02d}" for n, k in enumerate(("retimed", "dead", "alive", "sampling", "gone", "long"), 1)}
    J = {k: node.jobs / "runs" / r for k, r in R.items()}
    host = socket.gethostname()
    node.run(node.research, R["retimed"], "running", boot=boot, host=host, status={
        "state": "running", "pid": os.getpid(), "starttime_ticks": me,
        "transitions": [{"state": "running", "pid": os.getpid(), "starttime_ticks": me},
                        {"state": "running", "timeout_s": 7200, "timeout_changed_utc": "2026-10-03T20:00:00.000Z"}]})
    node.run(node.research, R["dead"], "running", boot=boot, host=host,
             status={"state": "running", "transitions": [{"state": "running", "pid": 2**22 + 7, "starttime_ticks": 1}]})
    node.run(node.jobs, R["alive"], "running", boot=boot)
    d = node.run(node.jobs, R["sampling"], "running", boot=boot, files={"resources.jsonl": '{"kind": "sample"}\n'})
    os.utime(d / "resources.jsonl", (N1_FUTURE, N1_FUTURE))
    node.run(node.jobs, R["gone"], "running", boot=boot)
    node.run(node.jobs, R["long"], "running", boot=boot, host="p" * 63)
    live = {"LIVE_PODS": f"pod-{R['alive']},pod-{R['sampling']}", "VY_CUSTODY_BOOT": boot}

    dry = node.round_("--dry-run", **live)
    assert dry.returncode == 0 and node.created() == [], dry.stdout + dry.stderr
    assert f"would leave {J['alive']} (r21-fixture, running): its pod pod-{R['alive']} still runs" in dry.stdout
    assert f"would publish {J['gone']} (r21-fixture, running, 8 files, 5.9 KiB) with --force --runner-gone (its pod" in dry.stdout
    assert f"with --force --runner-gone (its runner's pid is gone)" in dry.stdout and "1 its runner not seen gone" in dry.stdout
    assert not [k for k in ("retimed", "sampling", "long") if R[k] in dry.stdout], dry.stdout

    unreached = node.round_(POD_ERR="1", **live)
    assert unreached.returncode == 1 and node.created() == [f"n1-custody-{R['dead']}", "n1-store-push-jobs", "n1-store-push-research"]
    assert (f"{J['gone']} (r21-fixture, running): not published: `kubectl get pod pod-{R['gone']}` failed: Unable to connect to the "
            "server: dial tcp 10.0.0.1:6443: i/o timeout; the next round asks again") in unreached.stdout
    assert node.command(f"n1-custody-{R['dead']}")[-3:] == [f"{node.research}/runs/{R['dead']}", "--force", "--runner-gone"]

    first = node.round_(**live)
    assert first.returncode == 0 and node.created()[3:] == [f"n1-custody-{R['gone']}"], first.stdout + first.stderr
    assert node.command(f"n1-custody-{R['gone']}")[-3:] == [str(J["gone"]), "--force", "--runner-gone"]
    assert "1 run(s) wait on their pod, which Kubernetes still lists" in first.stdout
    assert not [k for k in ("retimed", "alive", "sampling", "long") if f"n1-custody-{R[k]}" in node.created()]
    assert node.round_(VY_CUSTODY_BOOT=boot).returncode == 0 and node.created()[-1] == f"n1-custody-{R['alive']}", "its pod ended"


@N1_CUSTODY
def test_n1_custody_holds_as_never_ran_only_a_run_that_shows_no_sign_it_started(tmp_path):
    """#1004's red-team F3: the never-ran hold took any status that never ended with empty stdout and stderr, counting the sampler's
    readings and the events as the runner's bookkeeping, and asked a person to label the run `custody never-ran`. All six runs node 1
    would have held (overnight-sep30) were `running` with a pid, span_start and sampler_start events and 6-70 sampler readings of
    the workload's processes: they ran, were killed and printed nothing. Only a status before `running` with none of ran_evidence's
    signs of a start is held; a killed run is a failed attempt, published --force once its pod is gone."""
    node = _n1_custody(tmp_path)
    killed, spoke, idle = (f"r20261003-2000{i:02d}-0d{i:02d}" for i in (1, 2, 3))
    d = node.run(node.jobs, killed, "running", stdout="", files={
        "stderr.log": "", "sampler.log": "", "resources.jsonl": '{"kind": "sample", "root_pid": 450, "processes": [{"pid": 451}]}\n' * 55,
        "events.jsonl": '{"kind": "started"}\n{"kind": "span_start", "stage": "run", "cmd0": "python"}\n{"kind": "sampler_start"}\n'})
    shutil.rmtree(d / "out")
    for p in ("result.json", "failure.json"):
        (d / p).unlink()
    node.run(node.jobs, spoke, "submitted", status={"state": "submitted"}, files={"events.jsonl": ""})
    node.run(node.jobs, idle, "admitted", empty=True, status={"state": "admitted"})
    out = node.round_()
    assert out.returncode == 1 and node.held() == [idle], out.stdout + out.stderr
    assert f"{node.jobs}/runs/{idle}: held: never ran its workload (r21-fixture, status admitted: no pid, start event" in out.stdout
    assert {f"n1-custody-{killed}", f"n1-custody-{spoke}"} <= set(node.created())
    assert node.command(f"n1-custody-{killed}")[-2:] == ["--force", "--runner-gone"]


@N1_CUSTODY
def test_n1_custody_starts_no_job_that_could_run_into_node_1s_quiet_hour(tmp_path):
    """#1004's red-team F4: the daily hold only kept Jobs from starting at 12:10Z, so a custody Job started at 12:09 could run to
    12:39 (activeDeadlineSeconds 1800) and the pending push (7200; the research store's first one uploads 22.3 GiB) to 14:09, through
    node 1's quiet hour (12:30-13:30Z), which measurements own. Every Job's deadline and its grace (10 s) end it a minute before
    12:30Z, none starts with under 5 min left, and a round in the daily hold starts nothing."""
    at = lambda hhmm: N2_NOW + (int(hhmm[:2]) - 3) * 3600 + int(hhmm[2:]) * 60          # N2_NOW is 03:00Z
    rid = "r20261003-200000-0e01"
    deadlines = {}
    for hhmm in ("1209", "1150", "1331"):
        node = _n1_custody(tmp_path / hhmm)
        node.run(node.jobs, rid, "done")
        out = node.round_(clock=at(hhmm), VY_CUSTODY_QUIET="1210-1330")
        assert out.returncode == 0, out.stdout + out.stderr
        deadlines[hhmm] = [node.job(n)["job"]["spec"]["activeDeadlineSeconds"] for n in node.created()]
        assert node.created() == [f"n1-custody-{rid}", "n1-store-push-jobs", "n1-store-push-research"], hhmm
    assert deadlines == {"1209": [1190, 1190, 1190], "1150": [1800, 2330, 2330], "1331": [1800, 7200, 7200]}
    node = _n1_custody(tmp_path / "held")
    node.run(node.jobs, rid, "done")
    for hhmm, quiet in (("1210", "1210-1330"), ("1329", "1210-1330"), ("1226", "1228-1330")):
        out = node.round_(clock=at(hhmm), VY_CUSTODY_QUIET=quiet)
        assert out.returncode == 0 and f"node 1's daily hold ({quiet} UTC; no Job runs past 1230Z" in out.stdout, hhmm
        rec = node.round_("--reconcile", rid, "--by", "test", clock=at(hhmm), VY_CUSTODY_QUIET=quiet)
        assert rec.returncode == 3 and "--reconcile after it" in rec.stderr
    assert node.created() == [] and not node.log.exists()


def _sigterm_while_research_runs(argv: list[str], research: Path) -> int:
    """A Job's command (taskset aside) with `research`, the Python it names, a stand-in that says it runs and blocks: SIGTERM then,
    and the command's exit status. The stand-in is let go after."""
    r, w = os.pipe()
    research.write_text(f"#!{sys.executable}\nimport os\nprint('research runs', flush=True)\nos.read({r}, 1)\n")
    research.chmod(0o755)
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True, pass_fds=(r,))
    os.close(r)
    try:
        assert p.stdout.readline() == "research runs\n", "research never ran"
        p.send_signal(signal.SIGTERM)
        return p.wait()
    finally:
        os.close(w)
        p.stdout.close()
        p.wait()


@N1_CUSTODY
@N2_CUSTODY
def test_n1_custody_no_job_outlives_its_deadline_into_the_quiet_hour(tmp_path):
    """#1004's red-team recheck N3: the deadline was read to the minute (a Job created at 12:07:45Z got 1320 s, to 12:29:45), and a
    Job stopped at its deadline has its pod's grace (30 s unless set) before SIGKILL, which its main process, a PID namespace's init
    that ignores SIGTERM without a handler, used up: to 12:30:15. Each Job of n1_custody.sh and n2_custody.sh gets a deadline to the
    second that, with its grace (10 s), ends it a minute before 12:30Z, and its bash, the container's init, traps SIGTERM and exits
    while research runs."""
    at = lambda h, m, s: N2_NOW + (h - 3) * 3600 + m * 60 + s                         # N2_NOW is 03:00Z
    node = _n1_custody(tmp_path / "n1")
    node.run(node.jobs, "r20261003-200000-2301", "done")
    out = node.round_(clock=at(12, 7, 45), VY_CUSTODY_QUIET="1210-1330")
    assert out.returncode == 0 and len(node.created()) == 3, out.stdout + out.stderr
    for name in node.created():
        spec = node.job(name)["job"]["spec"]
        assert spec["activeDeadlineSeconds"] + spec["template"]["spec"]["terminationGracePeriodSeconds"] == 22 * 60 + 15 - 60, name
        cmd = node.command(name)
        assert _sigterm_while_research_runs([*cmd[3:8], str(tmp_path / "research"), *cmd[9:]], tmp_path / "research") == 143, name

    (tmp_path / "n2").mkdir()
    n2 = _n2_custody(tmp_path / "n2")
    n2.run("r20261003-100001-0001", {"state": "done"})
    assert n2.round_(clock=at(11, 50, 30), VY_CUSTODY_PY=str(tmp_path / "research")).returncode == 0
    calls = n2.calls()
    job = json.JSONDecoder().raw_decode(calls[calls.index('{"apiVersion"'):])[0]
    assert job["spec"]["activeDeadlineSeconds"] + job["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] == 39 * 60 + 30 - 60
    cmd = job["spec"]["template"]["spec"]["containers"][0]["command"]
    assert _sigterm_while_research_runs(cmd[3:], tmp_path / "research") == 143


@N1_CUSTODY
def test_n1_custody_runs_only_a_tree_and_a_python_no_other_uid_can_change(tmp_path):
    """#1004's red-team F5: the round runs its research tree on the host as research (sudo ALL) and in Jobs with the Secret
    research-r2, from /workspace/jobs/custody, which /workspace/jobs (1777) let any uid-1000 workload create first and deploy.py
    then installs into keeping that owner, with ubuntu's Python under /workspace/jobs/cache. The tree and the Python are root's under
    /workspace/custody; a round refuses when a file of either, or a directory above, could be changed by a uid other than root and
    its own (another owner, or writable by group or others; a sticky directory such as /tmp above, or uv's empty .lock, aside), and
    when its Python isn't under PYHOME."""
    node = _n1_custody(tmp_path)
    rid = "r20261003-200000-0f01"
    node.run(node.jobs, rid, "done")
    tree, custody_py = node.src / "tools" / "research", node.src / "tools" / "research" / "src" / "research" / "store" / "custody.py"
    (tmp_path / "py" / ".lock").write_text("")
    (tmp_path / "py" / ".lock").chmod(0o666)
    said = []
    for p, mode in ((tree, 0o777), (custody_py, 0o664), (node.src, 0o775), (node.py, 0o757)):
        was = p.stat().st_mode
        p.chmod(mode)
        out = node.round_()
        rec = node.round_("--reconcile", rid, "--by", "test")
        p.chmod(was)
        assert out.returncode == 3 and rec.returncode == 3 and f"refused: {p} could be changed by a uid other than root and " in out.stdout, p
        said.append(out.stdout)
    assert "on this host and in Jobs with the Secret research-r2" in said[0]
    elsewhere = node.round_(VY_CUSTODY_PY=sys.executable)
    assert elsewhere.returncode == 3 and f"refused: {sys.executable} is no interpreter under {tmp_path / 'py'}" in elsewhere.stdout
    assert node.created() == [] and "sudo" not in (node.log.read_text() if node.log.exists() else ""), "a refused round runs nothing"
    ok = node.round_()
    assert ok.returncode == 0 and node.created()[0] == f"n1-custody-{rid}", ok.stdout + ok.stderr


@N1_CUSTODY
def test_n1_custody_takes_a_run_dirs_text_as_data_never_as_a_plan_line_or_a_shells(tmp_path):
    """#1004's red-team F6: status.json's state went into the plan's `|`-lines as written, and the force list into the Job's bash -c
    unquoted, so a run dir's writer (uid 1000: any workload) could add a Job line and run a command as uid 1001 with the Secret
    research-r2. A state the runner never writes holds the run, unechoed; a campaign shows anything but [A-Za-z0-9._:@+/-] as `?`;
    a Job's script is fixed, with the run dir and the force flags as its arguments, and its JSON is written by json."""
    node = _n1_custody(tmp_path)
    pwned = tmp_path / "PWNED"
    evil, other, odd = (f"r20261003-200000-10{i:02d}" for i in (1, 2, 3))
    node.run(node.jobs, evil, "running", status={"state": f'running\njob|{node.research}/runs|{other}|;touch {pwned};|c|s|1|1"',
                                                  "transitions": [{"state": "running", "pid": 7}]})
    d = node.run(node.jobs, odd, "done")
    job = json.loads((d / "job.json").read_text())
    job["store"]["campaign"] = "x$(id)|y\nz"
    (d / "job.json").write_text(json.dumps(job))
    os.utime(d / "job.json", (N1_OLD, N1_OLD))
    dry = node.round_("--dry-run")
    assert f"would hold {node.jobs}/runs/{evil}: its status.json says a state the runner never writes (r21-fixture)" in dry.stdout
    assert f"would publish {d} (x??id??y?z, done, " in dry.stdout and other not in dry.stdout, dry.stdout
    out = node.round_()
    assert out.returncode == 1 and not pwned.exists() and node.held() == [evil], out.stdout + out.stderr
    assert "touch" not in out.stdout and node.created() == [f"n1-custody-{odd}", "n1-store-push-jobs", "n1-store-push-research"]
    assert node.command(f"n1-custody-{odd}")[6:] == ["n1-custody", str(node.src), str(node.py), f"{node.jobs}/store", f"{node.jobs}/runs", str(d)]


@N1_CUSTODY
def test_n1_custody_triages_with_no_remote_another_uid_could_name(tmp_path):
    """#1004's red-team F8: triage ran with HOME=/tmp and no RESEARCH_STORE_CONFIG, so its remote was /tmp/.research/store.toml,
    which any uid on node 1 may write first: get_attempt would cache that remote's attempt documents into the store, and the next
    Job push one to R2 as the run's write-once attempt. Triage runs with the round's own HOME and RESEARCH_STORE_CONFIG=/dev/null,
    which no uid but root can replace and which names no remote."""
    node = _n1_custody(tmp_path)
    node.run(node.jobs, "r20261003-200000-1201", "done")
    assert node.round_("--dry-run").returncode == 0
    triage = [ln for ln in node.log.read_text().splitlines() if ln.startswith("sudo ") and "--triage" in ln]
    assert len(triage) == 2 and all(f" HOME={tmp_path / 'home'} " in ln and " RESEARCH_STORE_CONFIG=/dev/null " in ln for ln in triage), triage


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="node_ops reads /proc")
def test_node_ops_reads_a_runs_pid_from_its_runners_transition_not_the_last(tmp_path, monkeypatch):
    """`tele set-timeout` appends a `running` transition without a pid, and node_ops read the pid from the last transition only, so it
    alerted dead-run on a run whose runner was alive (the reading #1004's red team found in triage, F1)."""
    from research.pods.nebius import node_ops as NO

    me = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    d = tmp_path / "r20261003-200000-1301"
    d.mkdir()
    (d / "status.json").write_text(json.dumps({"state": "running", "pid": os.getpid(), "starttime_ticks": me, "transitions": [
        {"state": "running", "pid": os.getpid(), "starttime_ticks": me, "command": ["python", "x.py"], "started_utc": "2026-10-03T20:00:00.000Z"},
        {"state": "running", "timeout_s": 7200, "timeout_changed_utc": "2026-10-03T21:00:00.000Z"}]}))
    alerts = []
    monkeypatch.setattr(NO, "RUNS", tmp_path)
    monkeypatch.setattr(NO, "alert", lambda *a, **kw: alerts.append(a))
    assert NO.runs() == ([{"run": d.name, "since": "2026-10-03T20:00:00", "cmd": "python x.py", "alive": True}], []) and alerts == []


@N1_CUSTODY
def test_n1_custody_pushes_no_record_less_attempt_that_never_left_node_1(tmp_path, monkeypatch):
    """#1004's red-team F7: a local attempt published without its run record that R2 lacks is `attempt-unpreserved`, so a round ran
    `custody --publish` on it, and each store's pending push takes every attempt R2 lacks: R2's write-once key would keep it without
    its logs for good (28 under node 1's /workspace/jobs/runs on 3 Oct). Until a person rules, a round holds them record-less: no
    Job, one line counting them, a failed round and nothing superseded; the pending push's own command skips every attempt without a
    run record. Once one is on R2 (pushed by hand, as it is) triage calls it attempt-no-record, and it is no longer held."""
    from research.store import attempt as A
    from research.store.cli import main as data_main
    from research.store.local import LocalStore, attempt_key

    node = _n1_custody(tmp_path)
    rid, ok, gone = "r20261003-200000-1101", "r20261003-200000-1102", "r20261003-200000-1103"
    d = node.run(node.jobs, rid, "done")
    node.run(node.jobs, ok, "done")
    for r in (rid, gone):                                   # `gone`'s run dir was removed: no triage lists it, the push still skips it
        assert A.publish(node.run(node.jobs, r, "done") if r == gone else d, store=LocalStore(node.jobs / "store", config_path=node.absent),
                         remote_push=False, run_record=False)["published"]
    shutil.rmtree(node.jobs / "runs" / gone)
    for p in d.rglob("*"):
        os.utime(p, (N1_OLD, N1_OLD))
    out = node.round_()
    assert out.returncode == 1 and node.created() == [f"n1-custody-{ok}", "n1-store-push-jobs", "n1-store-push-research"], out.stdout + out.stderr
    assert "1 record-less attempt(s) held (triage's attempt-unpreserved)" in out.stdout and not node.held()
    push = node.command("n1-store-push-jobs")
    assert push[-1] == "--skip-record-less"
    pushed = subprocess.run(["bash", "-c", push[5].replace("HOME=/tmp", f"HOME={tmp_path / 'home'}"), *push[6:]], capture_output=True,
                            text=True, env={**os.environ, "RESEARCH_STORE_CONFIG": str(node.remote)})
    assert pushed.returncode == 0, pushed.stdout + pushed.stderr
    assert f"2 attempt(s) without a run record not pushed (--skip-record-less): {rid} {gone}" in pushed.stderr, pushed.stderr
    assert not (node.vault / attempt_key(rid)).exists() and not (node.vault / attempt_key(gone)).exists()
    assert not (node.jobs / "store" / "attempts" / ".superseded").exists()

    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(node.remote))
    assert data_main(["push", rid, "--store", str(node.jobs / "store")]) == 0
    after = node.round_()
    assert after.returncode == 0 and "record-less" not in after.stdout and "1 attempt-no-record" in after.stdout, after.stdout
    assert len(node.created()) == 3


def test_store_push_template_pushes_no_record_less_attempt(tmp_path, monkeypatch, capsys):
    """#1004's red-team recheck N1: the store-push template's MODE=push (one ready file on node 1 starts it; it pushed 146 objects
    on 2 Oct) ran `research data push --pending` on node 1's job store without --skip-record-less, so it would have put every
    attempt n1_custody.sh holds record-less on R2's write-once keys before a person rules. Every template's pending push skips them,
    and still pushes the rest. `push --pending` itself still takes them: a local `research run` without --run-record publishes one,
    and `push --pending` is what the store says preserves it."""
    from research.store import attempt as A
    from research.store.cli import main as data_main
    from research.store.local import LocalStore, attempt_key

    lines = [ln for t in (NEBIUS / "sky" / "jobs").glob("*.yaml") for ln in t.read_text().splitlines()
             if "data push --pending" in ln and not ln.lstrip().startswith("#")]
    assert lines and all("--skip-record-less" in ln for ln in lines), lines
    line = next(ln for ln in lines if ln.strip().startswith("push)"))
    argv = shlex.split(re.search(r"research data (push [^>]*)", line).group(1).replace('"$PUSH_JOBS"', "1"))
    node = _n1_custody(tmp_path)
    less, full = "r20261003-200000-2101", "r20261003-200000-2102"
    for rid in (less, full):
        assert A.publish(node.run(node.jobs, rid, "done"), store=LocalStore(node.jobs / "store", config_path=node.absent),
                         remote_push=False, run_record=rid == full)["published"]
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(node.remote))
    monkeypatch.setenv("RESEARCH_STORE", str(node.jobs / "store"))
    assert data_main(argv) == 0
    assert f"1 attempt(s) without a run record not pushed (--skip-record-less): {less}" in capsys.readouterr().err
    assert not (node.vault / attempt_key(less)).exists() and (node.vault / attempt_key(full)).exists()


@N1_CUSTODY
def test_n1_custody_holds_a_forced_attempt_whose_run_dir_changed_after_its_one_job(tmp_path, monkeypatch):
    """#1004's red-team recheck N2: a run published --force whose dir then gained a file is forced-unverified, and a round started a
    custody Job for it every round, for good, though none can mend it: its attempt is write-once. As n2_custody.sh does, the round
    holds it after its one Job and says what a person does; a `custody waived` label lifts the hold."""
    from research.store import attempt as A
    from research.store.cli import main as data_main
    from research.store.local import LocalStore

    node = _n1_custody(tmp_path)
    rid = "r20261003-200000-2201"
    d = node.run(node.jobs, rid, "running")
    st = LocalStore(node.jobs / "store", config_path=node.remote)
    assert A.publish(d, store=st, remote_push=False, run_record=True, force=True)["published"]
    assert CU.push_run(st, rid) == [] and CU.verify(st, d)["preserved"]
    (d / "out" / "late.json").write_text('{"verdict": "pass"}\n')
    (d / "status.json").write_text(json.dumps({"state": "done", "transitions": [{"state": "running", "pid": 4243}, {"state": "done"}]}))
    rounds = []
    for _ in range(2):
        for p in d.rglob("*"):
            os.utime(p, (N1_OLD, N1_OLD))
        rounds.append(node.round_())
    out = "\n".join(o.stdout + o.stderr for o in rounds)
    assert [n for n in node.created() if n.startswith("n1-custody-")] == [f"n1-custody-{rid}"] and node.held() == [rid], out
    assert all(o.returncode == 1 for o in rounds) and "Job failed" not in out
    assert "held: its attempt's run record, taken from the run dir as it was earlier, lacks what it holds now" in rounds[0].stdout
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(node.absent))
    assert data_main(["label", rid, "custody", "waived", "--by", "test", "--store", str(node.jobs / "store")]) == 0
    lifted = node.round_()
    assert lifted.returncode == 0 and node.held() == [] and f"{rid}: hold lifted" in lifted.stdout, lifted.stdout


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
    stand-in `cluster` package that echoes its arguments (infra/cluster's own tests cover the placement)."""
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
    silent = tmp_path / "silent-cluster"     # answers nothing; a missing one would fall through to infra/cluster's editable install
    (silent / "src" / "cluster").mkdir(parents=True)
    (silent / "src" / "cluster" / "__init__.py").write_text("")
    (silent / "src" / "cluster" / "__main__.py").write_text("import sys\nsys.exit('no queue here')\n")
    r = place(silent)
    assert r.returncode == 0 and not r.stdout and "no placement" in r.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="n2_commit.sh is bash")
def test_n2_commit_keeps_a_row_whose_plans_are_keyed_to_other_code_on_node_1(tmp_path):
    """cov-gm176's Build wrote its plans from a job tree that the 14:17Z redeploy (2 Oct) replaced. Node 2's Commit ran from the new
    tree, found the plans' code key differing and derived both with GPU 6 held: two 25-min leases at 0%. submit refuses such a row
    before it touches node 2, so node 1's leased Commit derives the plans before it takes a GPU. The job tree is a stand-in whose
    source_identity hashes one file (integrations/vllm's own tests cover code_identity)."""
    row_name = "gemma2-9b__bf16__rtxpro6000__tp1__b8__i1024__o128__mixed__greedy__bi-eager"
    tree, row, bin_, log = tmp_path / "tree", tmp_path / "sweep" / row_name, tmp_path / "bin", tmp_path / "ssh.log"
    pipeline = tree / "integrations" / "vllm" / "verity_vllm" / "pipeline"
    for d in (pipeline, row / "plan", bin_):
        d.mkdir(parents=True)
    (pipeline.parent / "__init__.py").write_text("")
    (pipeline / "__init__.py").write_text("")
    (pipeline / "source_identity.py").write_text(
        "import hashlib, pathlib\nTREE = pathlib.Path(__file__).resolve().parents[4]\n"
        "def code_identity(root):\n    return {'sha256': hashlib.sha256((pathlib.Path(root) / 'code').read_bytes()).hexdigest()}\n")
    (tree / "code").write_text("after the redeploy")
    new, old = hashlib.sha256(b"after the redeploy").hexdigest(), hashlib.sha256(b"before it").hexdigest()
    (bin_ / "ssh").write_text(f'#!/bin/sh\necho "$*" >> {log}\nexit 1\n')
    (bin_ / "ssh").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "VY_JOB_PY": sys.executable, "VY_N1": "n1.invalid",
           "VY_N2": "n2.invalid", "VY_N2_SRC": str(tree)}

    def plans(commit: str | None, boundary: str | None) -> None:
        for name, code in (("plan/commit_plan.json", commit), ("call_boundary_plan.json", boundary)):
            (row / name).unlink(missing_ok=True)
            if code is not None:
                (row / name).write_text(json.dumps({"schema": "s", "key": {"program_digest": "d", "code": code}}))

    def keyed(t: Path = tree) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(NEBIUS / "n2_commit.sh"), "plan-keyed", str(t), str(row)], capture_output=True, text=True,
                              env=env)

    assert keyed().returncode == 0
    plans(new, new)
    assert keyed().returncode == 0
    plans(new, old)
    r = keyed()
    assert r.returncode == 1 and r.stderr.strip() == f"call_boundary_plan.json is keyed to code {old[:16]}, the Commit's job tree is {new[:16]}"
    plans(old, new)
    assert "plan/commit_plan.json is keyed to code" in keyed().stderr
    plans(None, None)
    (row / "plan" / "commit_plan.json").write_text("not json")
    r = keyed()
    assert r.returncode == 1 and "plan/commit_plan.json is unreadable" in r.stderr
    (tmp_path / "empty").mkdir()
    r = keyed(tmp_path / "empty")
    assert r.returncode == 1 and r.stderr.startswith("source_identity came from ") and r.stderr.strip().endswith("/empty"), r.stderr
    (tree / "code").unlink()
    r = keyed()
    assert r.returncode == 1 and "the job tree's code identity isn't computable (FileNotFoundError" in r.stderr, r.stderr
    (tree / "code").write_text("after the redeploy")

    item = tmp_path / "item.json"
    item.write_text(json.dumps({"template": "config-run", "tree": "/t", "env": {
        "ROW": row_name, "ROLE": "GEMMA2_9B", "REPO": "unsloth/gemma-2-9b", "REVISION": "9145841c", "SWEEP_DIR": str(row.parent)}}))
    (row / "manifest.json").write_text("{}")
    (row / "two-task-build-run").write_text("r20261002-120300-0000")

    def submit() -> subprocess.CompletedProcess:
        log.write_text("")
        return subprocess.run(["bash", str(NEBIUS / "n2_commit.sh"), "submit", "vllm-epoch-run-cov-gm176", str(item)],
                              capture_output=True, text=True, env=env)

    plans(old, old)
    r = submit()
    assert r.returncode == 3 and "plan/commit_plan.json is keyed to code" in r.stderr and "not queued" in r.stderr, r.stderr
    assert log.read_text() == ""
    plans(new, new)
    r = submit()
    assert r.returncode == 3 and "isn't staged on node 2" in r.stderr, r.stderr
    assert log.read_text().startswith("-n n2.invalid ")


@pytest.mark.skipif(shutil.which("bash") is None, reason="n2_commit.sh is bash")
def test_n2_commit_marks_a_commit_it_hands_back_so_offload_never_moves_it_to_node_2_again(tmp_path):
    """cov-cg04-norm-taps-2 went from node 1 to node 2 seven times (22:30-23:06Z, 1 Oct): node 2 handed it back after two stops,
    and node 1's offload moved it again 120 s later. A Commit handed back (task 1) goes into node 1's reclaimed.txt, which offload
    skips, before node 1's dispatcher takes it; a replay (task 2) doesn't, and a proof goes to neither. `dispatch_n1` runs as the
    script defines it, with ssh faked."""
    fn = re.search(r"^dispatch_n1\(\) \{.*?^\}\n", (NEBIUS / "n2_commit.sh").read_text(), re.M | re.S).group(0)
    bin_, log = tmp_path / "bin", tmp_path / "ssh.log"
    bin_.mkdir()
    (bin_ / "ssh").write_text(f'#!/bin/sh\n[ "$1" = -n ] && shift\nshift\necho "$*" >> {log}\n')
    (bin_ / "ssh").chmod(0o755)

    def hand_back(task: int, proof: str = "") -> list[str]:
        log.write_text("")
        subprocess.run(["bash", "-c", f"set -euo pipefail\n{fn}\nITEM_ARGS=(--env ROW=r)\ndispatch_n1 {task}"], check=True,
                       capture_output=True, env={**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "N1": "vy-n1", "GUEST": "/g",
                                                 "DISPATCH": "/d.py", "KUBE_DISPATCH": "KUBECONFIG=k", "TREE": "/t", "PROOF": proof,
                                                 "COMMIT_KEY": "vllm-epoch-run/cov-cg04-norm-taps-2"})
        return log.read_text().splitlines()

    mark, submit = hand_back(1)
    assert mark == "echo vllm-epoch-run/cov-cg04-norm-taps-2 >> /g/reclaimed.txt"
    assert "/d.py submit config-run vllm-epoch-run/cov-cg04-norm-taps-2 --tree /t --task 1" in submit
    [replay] = hand_back(2)
    assert "--task 2" in replay
    assert hand_back(1, proof="1") == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="n2_commit.sh is bash")
def test_a_commit_that_goes_back_to_node_1_keeps_its_lease_and_pinned_template(tmp_path):
    """A `lease: self` Commit that comes back from node 2 without its lease runs as a whole-GPU pod and holds the GPU through its CPU
    prep. Both ways back (dispatch_n1 after a stop or failure, reclaim after a long wait) take their `dispatch.py submit` flags from
    submit_args, which keeps every item field that submit takes."""
    text = (NEBIUS / "n2_commit.sh").read_text()
    body = lambda name: re.search(rf"^{name}\(\) \{{.*?^\}}\n", text, re.M | re.S).group(0)
    item = tmp_path / "item.json"
    item.write_text(json.dumps({"tree": "/t", "lease": "self", "template_sha": "87e09228d7e0", "priority": "circuits-gpu",
                                "resources": {"gpu": {"gpus": 1}}, "env": {"ROW": "gemma2-9b__bf16", "REPLAY_DEFERRED": "1"},
                                "n2": {"commit_key": "vllm-epoch-run/cov-gm148-r2"}}))
    out = subprocess.run(["bash", "-c", f"set -euo pipefail\n{body('submit_args')}\nsubmit_args {item}"], check=True, capture_output=True,
                         text=True).stdout
    words = shlex.split(out)
    pairs = dict(zip(words[::2], words[1::2]))
    assert pairs["--lease"] == "self" and pairs["--template-sha"] == "87e09228d7e0" and pairs["--priority"] == "circuits-gpu"
    assert json.loads(pairs["--resources"]) == {"gpu": {"gpus": 1}}
    assert [w for w in words if w.startswith("ROW=") or w.startswith("REPLAY_DEFERRED=")] == ["ROW=gemma2-9b__bf16", "REPLAY_DEFERRED=1"]
    assert 'ITEM_ARGS=($(submit_args "$1"))' in body("load_item") and 'args=($(submit_args "$item"))' in body("reclaim")
    assert "--priority" not in body("load_item") + body("reclaim"), "one list of the item's flags, in submit_args"


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


def test_n1_lease_setup_makes_lease_files_writable_as_root_but_never_through_a_link(monkeypatch, tmp_path):
    """The lease dir is 0777 and setup's chmod runs as root, so a job that planted a symlink to a file of root's there (or a hard
    link, or swapped a listed file for a symlink) had root make that file world-writable."""
    m, _, _ = _n1_lease(monkeypatch, tmp_path)
    gl = tmp_path / "gl"
    gl.chmod(0o777)
    secret, other = tmp_path / "secret", tmp_path / "other"
    for p in (secret, other, gl / "0.owner"):
        p.write_text("x")
        p.chmod(0o600)
    (gl / "planted").symlink_to(secret)
    os.link(other, gl / "hard")
    real, ran = subprocess.run, []
    monkeypatch.setattr(m.subprocess, "run", lambda argv, **kw: ran.append(argv) or subprocess.CompletedProcess(argv, 0))
    m.setup()
    monkeypatch.setattr(m.subprocess, "run", real)
    assert {a[-1] for a in ran} == {str(gl / "0.owner"), str(gl / "hard")}          # a symlink is never handed to root
    for argv in ran:
        assert argv[:4] == ["sudo", "-n", "python3", "-c"]
        real([sys.executable, *argv[3:]], check=True)
    assert (gl / "0.owner").stat().st_mode & 0o777 == 0o666
    assert secret.stat().st_mode & 0o777 == 0o600 and other.stat().st_mode & 0o777 == 0o600
    (gl / "0.owner").unlink()
    (gl / "0.owner").symlink_to(secret)                                                  # swapped after setup listed it
    assert real([sys.executable, "-c", m.FIX_MODE, str(gl / "0.owner")], capture_output=True).returncode != 0
    assert secret.stat().st_mode & 0o777 == 0o600


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
    assert len(created) == 1 and created[0]["spec"]["template"]["spec"]["containers"][0]["resources"] == {
        "requests": {"nvidia.com/gpu": "1"}, "limits": {"nvidia.com/gpu": "1"}}      # no CPU or memory: waiters hold the queue's
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


def test_gpu_stray_reads_only_the_gpu_variables_of_a_process(monkeypatch):
    """Daniel's ruling (7 Oct): the probe reads another process's GPU variables and nothing else of its environment, which may
    hold its secrets. Under sudo only the matching lines leave `grep`; read directly, the rest is dropped at once."""
    spec = importlib.util.spec_from_file_location("gpu_stray_under_test", NEBIUS / "gpu_stray.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    env = {"PATH": os.environ["PATH"], "CUDA_VISIBLE_DEVICES": "3", "NVIDIA_VISIBLE_DEVICES": "all", "JOB_TOKEN": "dummy"}
    p = subprocess.Popen(["cat"], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        p.stdin.write(b"up\n")
        p.stdin.flush()
        assert p.stdout.readline() == b"up\n"          # cat echoes once it has exec'd with `env`
        assert m.environ(p.pid) == {"CUDA_VISIBLE_DEVICES": "3", "NVIDIA_VISIBLE_DEVICES": "all"}
        ran = []

        def denied(self):
            raise PermissionError(str(self))

        def sudo(cmd, **kw):
            ran.append(cmd)
            return real_run(cmd[2:], **kw)              # the same grep, without sudo

        real_run = subprocess.run
        monkeypatch.setattr(m.Path, "read_bytes", denied)
        monkeypatch.setattr(m.subprocess, "run", sudo)
        assert m.environ(p.pid) == {"CUDA_VISIBLE_DEVICES": "3", "NVIDIA_VISIBLE_DEVICES": "all"}
        assert ran == [["sudo", "-n", "grep", "-zaE", "^(NVIDIA_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES)=", f"/proc/{p.pid}/environ"]]
    finally:
        p.communicate()


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


def _policy_lease(monkeypatch, tmp_path, max_: int, policy: str | None = None):
    """The controller with VY_POOL_MAX=`max_` under a policy command (`cat` of a document, or `policy`), and a waiter that wants
    3 GPUs, which 3 nominal allow; the waiter's lock fd, and `keys(**kv)`, which writes the document."""
    import fcntl
    doc = tmp_path / "policy.json"
    m, calls, _ = _n1_lease(monkeypatch, tmp_path, VY_POOL_MAX=max_, VY_POOL_POLICY=policy or f"cat {doc}")
    w = tmp_path / "gl" / "wait.1.4242"
    w.write_text("pid=4242 n=3 who=adhoc:me since=x\n")
    fd = os.open(w, os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_SH)

    def keys(pools=("provers",), **kv):
        doc.write_text(json.dumps({"schema": "cluster/policy/v1", "keys": [
            {"key": f"vy-nebius-1.lease-pool.{p}.{k.replace('_', '-')}", "value": v} for p in pools for k, v in kv.items()]}))
    return m, calls, fd, doc, keys


@needs_flock
def test_n1_lease_takes_its_pool_limits_from_the_policy_command_and_keeps_the_last_while_it_fails(monkeypatch, tmp_path, reap):
    t = 1_800_000_000.0
    m, calls, fd, doc, keys = _policy_lease(monkeypatch, tmp_path, 1)
    c = m.Controller()
    steps = [(lambda: doc.write_text("Traceback"), t, 1, "JSONDecodeError", "environment"),  # never answered: the environment's 1
             (lambda: keys(max=3, borrow=0, idle_s=600), t + 1, 1, None, "environment"),     # read again after POLICY_EVERY_S
             (lambda: None, t + m.POLICY_EVERY_S, 3, None, "policy"),            # 3 wanted, 3 nominal, max 3 by policy
             (lambda: keys(max=2), t + 2 * m.POLICY_EVERY_S, 2, None, "policy"),
             (lambda: doc.write_text("Traceback"), t + 3 * m.POLICY_EVERY_S, 2, "JSONDecodeError", "last-policy"),
             (lambda: keys(max=3, borrow_until="tomorrow"), t + 4 * m.POLICY_EVERY_S, 2, "borrow-until 'tomorrow'", "last-policy"),
             (lambda: keys(("provers", "circuits"), max=3), t + 5 * m.POLICY_EVERY_S, 2, "two keys give max", "last-policy")]
    for change, at, want, error, source in steps:
        change()
        calls.clear()
        events = [e for e in c.tick(at)["events"] if e["ev"] == "policy"]
        assert len([a for a, _ in calls if a[0] == "create"]) == want, (at - t, want)
        if error:
            assert len(events) == 1 and error in events[0]["error"] and \
                (events[0]["limits"]["max"], events[0]["fallback"]) == (want, source)
        assert (tmp_path / "gl" / "n1-lease.limits").read_text() == f"max={want} borrow=0 borrow-until= from={source}\n"
    os.close(fd)


@needs_flock
def test_n1_lease_kills_everything_a_policy_command_that_overruns_started(monkeypatch, tmp_path, reap):
    started, alive, block = tmp_path / "started", tmp_path / "alive", tmp_path / "block"
    for f in (started, alive, block):
        os.mkfifo(f)
    # the shell's child holds `alive` open for writing until it dies, so its read end's EOF is that death (a kill that misses it
    # leaves the read blocked)
    hung = f"sh -c 'exec 3>{alive}; echo $$ > {started}; cat {block}'; true"
    m, calls, fd, _, _ = _policy_lease(monkeypatch, tmp_path, 3, policy=hung)
    r = os.open(alive, os.O_RDONLY | os.O_NONBLOCK)
    os.set_blocking(r, True)
    child: list[int] = []

    def overrun(p, timeout):
        child.append(int(started.read_text()))
        raise subprocess.TimeoutExpired(p.args, timeout)
    monkeypatch.setattr(m, "communicate", overrun)
    events = [e for e in m.Controller().tick(1_800_000_000.0)["events"] if e["ev"] == "policy"]
    os.close(fd)
    assert os.read(r, 1) == b"" and child
    os.close(r)
    assert len(events) == 1 and events[0]["error"].startswith("TimeoutExpired") and events[0]["fallback"] == "environment"
    assert len([a for a, _ in calls if a[0] == "create"]) == 3


@needs_flock
def test_n1_lease_sees_a_waiter_that_a_read_of_proc_locks_misses(monkeypatch, tmp_path, reap):
    """/proc/locks comes a page per read, so on a node with many locks one read can miss a lock held throughout (seen on node 1,
    where it left a waiter unserved for a tick): a waiter is its file's lock, probed directly."""
    m, calls, fd, _, keys = _policy_lease(monkeypatch, tmp_path, 3)
    keys(max=3)
    (tmp_path / "gl" / "wait.2.4343").write_text("pid=4343 n=1 who=gone since=x\n")
    monkeypatch.setattr(m, "lock_holders", dict)
    assert [w["file"] for w in m.waiters()] == ["wait.1.4242"]
    m.Controller().tick(1_800_000_000.0)
    os.close(fd)
    assert len([a for a, _ in calls if a[0] == "create"]) == 3
    assert not m.waiters()


@needs_flock
def test_n1_lease_logs_a_limits_file_it_cant_write_and_goes_on(monkeypatch, tmp_path, reap):
    m, calls, fd, _, keys = _policy_lease(monkeypatch, tmp_path, 1)
    keys(max=3)
    (tmp_path / "gl" / "n1-lease.limits").mkdir()
    events = [e["ev"] for e in m.Controller().tick(1_800_000_000.0)["events"]]
    os.close(fd)
    assert events[:2] == ["policy", "limits-write-failed"]
    assert len([a for a, _ in calls if a[0] == "create"]) == 3


@needs_flock
@pytest.mark.parametrize("value", ["-1", -1, 1.5, float("inf"), "8x", 5, True, None])
def test_n1_lease_refuses_a_policy_max_that_isnt_a_whole_number_of_its_gpus_and_goes_on(monkeypatch, tmp_path, reap, value):
    m, calls, fd, _, keys = _policy_lease(monkeypatch, tmp_path, 3)
    keys(max=value)
    events = [e for e in m.Controller().tick(1_800_000_000.0)["events"] if e["ev"] == "policy"]
    os.close(fd)
    assert len([a for a, _ in calls if a[0] == "create"]) == 3
    assert len(events) == 1 and events[0]["limits"]["max"] == 3 and events[0]["fallback"] == "environment"
    assert f"max {value!r} isn't a whole number of GPUs from 0 to 4" in events[0]["error"]


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


def test_fill_runner_pauses_cpu_sets_jobs_on_slot_d_while_a_train_check_holds_its_lock(tmp_path, monkeypatch):
    """Compute accounting (1 Oct 8:58 AM PDT): GPU 0's verifies go back on 0-47 yielding to slot d's train checks. A check runs in
    a login session's cgroup, where nice 19 can't reach, so the runner starts none and pauses its running ones while the check
    holds slot d's lock; an owner whose CPUs miss 0-47 runs on."""
    fill = tmp_path / "fill"
    for d in ("queue", "running", "logs", "done", "failed"):
        (fill / d).mkdir(parents=True)
    lock = tmp_path / "check-d.lock"
    lock.touch()
    monkeypatch.setenv("FILL_DIR", str(fill))
    monkeypatch.setenv("GPU_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("FILL_YIELD_LOCK", str(lock))
    spec = importlib.util.spec_from_file_location("fill_runner_yield", FILL_RUNNER)
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    monkeypatch.setattr(fr, "disk_pct", lambda path: 0.0)
    monkeypatch.setattr(fr, "event", lambda *a, **kw: None)
    monkeypatch.setattr(fr, "scope", lambda j, *a: True)
    monkeypatch.setattr(fr, "active_s", lambda j: 0.0)
    monkeypatch.setattr(fr, "stop", lambda k, why, requeue=True: pytest.fail(f"stopped {k}: {why}"))
    signals = []
    monkeypatch.setattr(fr.os, "killpg", lambda pid, sig: signals.append(sig))

    def fake_start(p, h, backfill=False):
        p.unlink()
        fr.running[p.name] = {"h": h, "t0": 0.0, "pid": 0, "proc": type("P", (), {"poll": lambda self: None})(),
                              "started": "t", "paused": False, "unit": None}

    monkeypatch.setattr(fr, "start", fake_start)
    (fill / "cpu-sets").write_text("bc-e 0-47 2 40\nbc-f 96-123 1\n")
    for k in range(3):
        (fill / "queue" / f"ver-{k}.sh").write_text("# fill: owner=bc-e gpus=0 cpus=4\n")
    (fill / "queue" / "other.sh").write_text("# fill: owner=bc-f gpus=0 cpus=4\n")

    def mine():
        return sorted((k, j["paused"]) for k, j in fr.running.items() if j["h"]["owner"] == "bc-e")

    check = open(lock)
    fcntl.flock(check, fcntl.LOCK_EX)
    fr.tick(0)
    assert mine() == [] and "other.sh" in fr.running          # slot d is checking: none of 0-47's start, 96-123's does
    check.close()
    fr.tick(0)
    assert mine() == [("ver-0.sh", False), ("ver-1.sh", False)]
    check = open(lock)
    fcntl.flock(check, fcntl.LOCK_EX)
    fr.tick(0)
    assert mine() == [("ver-0.sh", True), ("ver-1.sh", True)] and not fr.running["other.sh"]["paused"]
    assert signals == [signal.SIGSTOP, signal.SIGSTOP]
    check.close()
    fr.tick(0)
    assert mine() == [("ver-0.sh", False), ("ver-1.sh", False)] and signals[2:] == [signal.SIGCONT, signal.SIGCONT]


def test_fill_runner_fence_command_caps_a_fence_keeps_one_per_owner_and_logs_it(tmp_path, monkeypatch, capsys):
    """Memory accounting's honest-latency fence on node 2 (2 Oct) took five round trips with infra: keep-free, a respawned pane for
    fill's CPUs, a max-min line and a revert timer. `fill_runner.py fence` writes one self-expiring line within the caps instead
    (1 GPU, at most 8 cores on its NUMA node, at most 4 h, one per owner, none on another's GPU or cores) and logs each change."""
    fill, log = tmp_path / "fill", tmp_path / "quota-changes.log"
    fill.mkdir()
    monkeypatch.setenv("FILL_DIR", str(fill))
    monkeypatch.setenv("FILL_QUOTA_LOG", str(log))
    spec = importlib.util.spec_from_file_location("fill_runner_fence", FILL_RUNNER)
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    monkeypatch.setattr(fr, "gpu_count", lambda: 8)
    monkeypatch.setattr(fr, "gpu_cpus", lambda i: set(range(96)) if i < 4 else set(range(96, 192)))
    t = fr.utc("2026-10-02T20:00Z")
    base = {"--gpu": "7", "--cpus": "116-123", "--until": "2026-10-02T23:00Z", "--why": "honest latency, thread 1"}

    def fence(owner, change=None):
        args = {**base, **(change or {})}
        return fr.fence([owner, *(x for k, v in args.items() if v is not None for x in (k, v))], t=t)

    for change in ({"--cpus": "112-120"}, {"--cpus": "90-97"}, {"--cpus": "x"}, {"--until": "2026-10-03T00:01Z"},
                   {"--until": "2026-10-02T19:59Z"}, {"--gpu": "8"}, {"--jobs": "meas-*"}, {"--jobs": "meas-*", "--max-min": "241"},
                   {"--why": None}):
        assert fence("ma", change) == 2, change
    (fill / "keep-free").write_text("7  # pous\n")
    assert fence("ma") == 2
    (fill / "keep-free").unlink()
    assert not (fill / "fences").exists() and not log.exists()

    (fill / "windows").write_text("2026-10-02T21:00Z 30 # a booked window\n")
    assert fence("ma", {"--jobs": "meas-*", "--max-min": "180"}) == 0
    assert "the window booked at 21:00Z" in capsys.readouterr().out
    assert (fill / "fences").read_text() == "ma 7 116-123 2026-10-02T23:00:00Z meas-* 180  # honest latency, thread 1\n"
    (f,) = fr.fences(t)
    assert (f["gpu"], f["cpus"], f["jobs"], f["minutes"]) == (7, "116-123", "meas-*", 180)
    assert fence("pm") == 2 and fence("pm", {"--gpu": "6", "--cpus": "120-121"}) == 2
    assert fence("pm", {"--gpu": "6", "--cpus": "124,125", "--until": "2026-10-02T21:00Z"}) == 0
    assert fence("ma", {"--gpu": "5", "--cpus": "100-101"}) == 0
    assert sorted((f["owner"], f["gpu"]) for f in fr.fences(t)) == [("ma", 5), ("pm", 6)]
    assert [f["owner"] for f in fr.fences(fr.utc("2026-10-02T21:00Z"))] == ["ma"]
    assert fr.fence(["ma", "--release"], t=t) == 0 and fr.fence(["ma", "--release"], t=t) == 2
    assert [f["owner"] for f in fr.fences(t)] == ["pm"]
    notes = [line.split(" ", 1)[1] for line in log.read_text().splitlines()]
    assert notes == ["ma (fill_runner.py fence): fence ma 7 116-123 2026-10-02T23:00:00Z meas-* 180  # honest latency, thread 1",
                     "pm (fill_runner.py fence): fence pm 6 124-125 2026-10-02T21:00:00Z  # honest latency, thread 1",
                     "ma (fill_runner.py fence): fence (replaces its last) ma 5 100-101 2026-10-02T23:00:00Z  # honest latency, "
                     "thread 1",
                     "ma (fill_runner.py fence): fence released, was ma 5 100-101 2026-10-02T23:00:00Z"]


def test_fill_runner_keeps_fill_off_a_fence_and_runs_the_fences_jobs_there(tmp_path, monkeypatch):
    """A fence takes its GPU at once: fill stops its own job there, starts no other job there or on the fence's cores, keeps the
    GPU in "kept free" (so a waiter pinned to it holds no fill back, and node 2's spill keeps off it), and moves running jobs off
    the cores once. A job its glob names runs on that GPU alone, on those cores, with the fence's max_min, and no Commit stops
    it for a GPU the Commit can't have."""
    gl, fill = tmp_path / "gl", tmp_path / "fill"
    for d in ("queue", "running", "logs"):
        (fill / d).mkdir(parents=True)
    gl.mkdir()
    monkeypatch.setenv("GPU_LEASE_DIR", str(gl))
    monkeypatch.setenv("FILL_DIR", str(fill))
    spec = importlib.util.spec_from_file_location("fill_runner_fenced", FILL_RUNNER)
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    (fill / "fences").write_text("ma 1 116-123 2099-01-01T00:00Z meas-* 90  # honest latency\n")
    held: dict[int, str] = {}
    started, stopped, repins = [], [], []
    monkeypatch.setattr(fr, "reap", lambda: None)
    monkeypatch.setattr(fr, "verity_pool", lambda queued: None)
    monkeypatch.setattr(fr, "lease", lambda i: held.get(i))
    monkeypatch.setattr(fr, "waiters", lambda: [])
    monkeypatch.setattr(fr, "repin", lambda f, fs: repins.append(f["owner"]))
    monkeypatch.setattr(fr, "start", lambda p, h, backfill=False: started.append((p.name, h["on"], h["max_min"])) or p.unlink())
    monkeypatch.setattr(fr, "stop", lambda k, why, requeue=True: stopped.append((k, why)) or fr.running[k].update(stopping=True,
                                                                                                                  why=why))

    def run(name, gpu):
        held[gpu] = f"pid={10 + gpu} run=- who=fill:o since=2026-10-02T20:00:00Z preempt=1 cmd=x"
        fr.running[name] = {"pid": 10 + gpu, "t0": 0, "paused": False, "started": "-",
                            "h": {"gpus": 1, "project": "pous", "owner": "o", "max_min": 10**9}}

    def done(name, gpu):
        fr.running.pop(name)
        held.pop(gpu)

    run("pous-a.sh", 0)
    run("pous-old.sh", 1)
    for name, head in (("pous-b.sh", "gpus=1 max_min=90"), ("meas-1.sh", "gpus=1 max_min=90")):
        (fill / "queue" / name).write_text(f"# fill: owner=ma {head}\n")
    fr.tick(2)
    assert stopped == [("pous-old.sh", "GPU 1 is fenced for ma")] and started == [] and repins == ["ma"]
    status = (fill / "status.txt").read_text().splitlines()
    assert "kept free [1]" in status[0] and status[-1] == "  fence ma 1 116-123 2099-01-01T00:00Z meas-* 90"
    done("pous-old.sh", 1)
    fr.tick(2)
    assert started == [("meas-1.sh", "1", 90)] and repins == ["ma"]
    run("meas-1.sh", 1)
    (fill / "queue" / "verity-commit-c.sh").write_text("# fill: owner=c gpus=1 project=verity\n")
    fr.tick(2)
    assert stopped[1:] == [("pous-a.sh", "a Commit reclaims its GPU")]
    done("pous-a.sh", 0)
    fr.tick(2)
    assert started[1:] == [("verity-commit-c.sh", "0", 25)]
    assert fr.header(fill / "queue" / "pous-b.sh")["max_min"] == 30          # the fence's minutes are its jobs' alone

    spec.loader.exec_module(fr)                                             # the real start
    cmds = []
    monkeypatch.setattr(fr.shutil, "which", lambda name: None)
    monkeypatch.setattr(fr.subprocess, "Popen", lambda cmd, **kw: cmds.append(cmd) or type("P", (), {"pid": 1})())
    monkeypatch.setattr(fr, "event", lambda *a, **kw: None)
    for name, head in (("meas-2.sh", "gpus=1"), ("pous-c.sh", "gpus=1"), ("cpu.sh", "gpus=0")):
        p = fill / "queue" / name
        p.write_text(f"# fill: owner=ma {head}\n")
        fr.start(p, fr.header(p))
        assert cmds[-1][cmds[-1].index("taskset") + 2] == ("116-123" if name == "meas-2.sh" else "96-115,124-127"), name


def test_fill_runner_moves_running_fill_jobs_off_a_new_fences_cores(tmp_path, monkeypatch):
    """A fill job already running when a fence starts would keep the fence's cores for up to its max_min: the runner moves each
    thread of every running fill job off them, except the fence's own jobs and a thread with no other core to go to."""
    cpus = os.sched_getaffinity(0)
    if len(cpus) < 2:
        pytest.skip("needs two CPUs")
    monkeypatch.setenv("FILL_DIR", str(tmp_path / "fill"))
    spec = importlib.util.spec_from_file_location("fill_runner_repin", FILL_RUNNER)
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    events = []
    monkeypatch.setattr(fr, "event", lambda kind, job, **kw: events.append((kind, kw["threads_moved"])))
    pause = [sys.executable, "-c", "import signal; print('ready', flush=True); signal.pause()"]
    procs = {k: subprocess.Popen(pause, stdout=subprocess.PIPE, text=True, start_new_session=True)
             for k in ("pous-a.sh", "meas-1.sh", "pous-pinned.sh")}
    try:
        for p in procs.values():
            assert p.stdout.readline().strip() == "ready"
        off = max(cpus)
        os.sched_setaffinity(procs["pous-pinned.sh"].pid, {off})
        for k, p in procs.items():
            fr.running[k] = {"pid": p.pid, "unit": None, "h": {"gpus": 1}}
        f = {"owner": "ma", "gpu": 0, "cpus": str(off), "jobs": "meas-*"}
        fr.repin(f, [f])
        assert os.sched_getaffinity(procs["pous-a.sh"].pid) == cpus - {off}
        assert os.sched_getaffinity(procs["meas-1.sh"].pid) == cpus
        assert os.sched_getaffinity(procs["pous-pinned.sh"].pid) == {off}
        assert events == [("fence", 1)]
    finally:
        for p in procs.values():
            stop_group(p)


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
