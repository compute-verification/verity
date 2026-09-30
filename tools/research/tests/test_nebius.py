"""The Nebius server (pods/nebius/): an ssh machine `research run --on` reaches by the host its registry file names, the lease loop
stopping a non-RunPod machine through its own command, gpu-lease sharing the GPUs, launch.sh's rendered cloud-init and VM spec,
and the weight list's pins.  No cloud: nothing here calls Nebius; the ssh probe is stubbed."""
from __future__ import annotations

import base64
import gzip
import importlib.util
import json
import os
import re
import shutil
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
                              stdout=subprocess.PIPE, text=True)
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
                                  env={**env, "GPU_LEASE_POLL_S": "3600"}, stderr=subprocess.PIPE, text=True)
        line = waiter.stderr.readline()
        assert line.startswith("gpu-lease: waiting for 1 GPU(s), queued as wait."), line
        assert fr.waiters() == [1]
    finally:
        for p in (waiter, holder):
            if p is not None:
                p.terminate()
                p.wait()


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
