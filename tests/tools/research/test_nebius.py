"""The Nebius server (pods/nebius/): an ssh machine `research run --on` reaches by the host its registry file names, the lease loop
stopping a non-RunPod machine through its own command, which sends its stop without the null device, gpu-lease sharing the GPUs,
launch.sh's rendered cloud-init and VM spec, and the weight list's pins.  No cloud: nothing here calls Nebius; the ssh probe and the
CLI are stubbed."""
from __future__ import annotations

import base64
import datetime
import fcntl
import gzip
import importlib.util
import json
import os
import re
import shutil
import signal
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
from research.store import private

NEBIUS = Path(registry.__file__).resolve().parent / "nebius"
GPU_LEASE = Path(registry.__file__).resolve().parent / "sh" / "gpu_lease.sh"
CHECKPOINTS = Path(__file__).resolve().parents[3] / "adapters" / "vllm" / "manifests" / "checkpoints.json"
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


def test_register_records_the_owning_lane_and_a_reregistration_keeps_it_and_the_deadline(reg):
    """The lane whose budget line pays for a node stops it (lane contract v2.19 §3c), and fleet.py's daily report rings it: launch.sh
    re-registers with --replace and no --lane, which must not hand the machine back to infra, nor drop the sampled deadline."""
    p, _ = registry.register_ssh("vy-glm53-eu", HOST, "verity", user="research", lane="circuits", probe=reachable)
    assert tomllib.loads(p.read_text())["lane"] == "circuits"
    p.write_text(p.read_text() + 'deadline = "2026-10-23T21:00:00Z"\n')
    assert registry.register_ssh("vy-glm53-eu", HOST, "verity", user="research", probe=reachable)[1] == "unchanged"
    assert registry.register_ssh("vy-glm53-eu", HOST, "verity", user="research", rate=43.2, replace=True, probe=reachable)[1] == "updated"
    doc = tomllib.loads(p.read_text())
    assert (doc["lane"], doc["deadline"], doc["rate_usd_per_hour"]) == ("circuits", "2026-10-23T21:00:00Z", 43.2)
    assert tomllib.loads(registry.register_ssh("vy-glm53-eu", HOST, "verity", user="research", rate=43.2, lane="kernels",
                                               probe=reachable)[0].read_text())["lane"] == "kernels"
    with pytest.raises(remote.Refused, match="lane '@circuits' is not a lane's handle"):
        registry.register_ssh("vy-glm53-eu2", HOST, "verity", lane="@circuits", probe=reachable)
    assert "lane" not in tomllib.loads(registry.register_ssh("vy-nebius-1", "198.51.100.9", "verity", probe=reachable)[0].read_text())


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


# ---- the stop command ----------------------------------------------------------------------------------------------------------------------

VM = "computeinstance-e00fake"


def stop_vm(tmp_path: Path, *args: str, can_read: bool = True) -> tuple[subprocess.CompletedProcess, list[str]]:
    """vm_stop.sh on a machine whose null device cannot be opened (every `/dev/null` in it names a path whose parent is a regular
    file), beside a metadata service that hands out a token and a `nebius` that prints as the CLI does; and the CLI's calls."""
    bin_dir, tmp, calls = tmp_path / "bin", tmp_path / "tmp", tmp_path / "calls"
    for d in (bin_dir, tmp):
        d.mkdir(exist_ok=True)
    (tmp_path / "file").write_text("")
    script = tmp_path / "vy-nebius-stop"
    script.write_text((NEBIUS / "vm_stop.sh").read_text().replace("/dev/null", str(tmp_path / "file" / "null")))
    (bin_dir / "curl").write_text("#!/bin/sh\nprintf token-from-metadata\n")
    (bin_dir / "nebius").write_text(f'#!/bin/sh\necho "$*" >> {calls}\necho "what the CLI prints"\n'
                                    f'case "$*" in *"instance get"*) exit {0 if can_read else 1};; esac\n')
    for shim in bin_dir.iterdir():
        shim.chmod(0o755)
    calls.unlink(missing_ok=True)
    r = subprocess.run(["sh", str(script), *args], env={"PATH": f"{bin_dir}:/usr/bin:/bin", "TMPDIR": str(tmp)},
                       capture_output=True, text=True)
    assert list(tmp.iterdir()) == [], "the token's directory is removed however the script ends"
    return r, calls.read_text().splitlines() if calls.exists() else []


def test_the_stop_command_sends_the_stop_though_the_null_device_cannot_be_opened(tmp_path):
    """Catches a redirect to the null device before the stop: under `set -e` it ends the script with the stop never sent, and the
    machine runs on past its lease."""
    r, calls = stop_vm(tmp_path, VM)
    assert (r.returncode, r.stdout) == (0, f"TERMINATED {VM} (nebius compute instance stop)\n"), r.stderr
    profile, stop = calls
    assert re.search(rf" --token-file {re.escape(str(tmp_path / 'tmp'))}/[^/ ]+/token --skip-auth$", profile), "the token goes by file"
    assert stop.endswith(f"compute instance stop --id {VM} --async") and "token-from-metadata" not in profile + stop


def test_the_stop_commands_probe_says_whether_the_account_can_read_the_vm_though_the_null_device_cannot_be_opened(tmp_path):
    r, calls = stop_vm(tmp_path, "--probe", VM)
    assert (r.returncode, r.stdout) == (0, f"the attached service account can read {VM}\n"), r.stderr
    assert calls[-1].endswith(f"compute instance get --id {VM}") and not any("instance stop" in c for c in calls)
    r, calls = stop_vm(tmp_path, "--probe", VM, can_read=False)
    assert r.returncode == 4 and r.stdout.startswith(f"the attached service account cannot read {VM}") and len(calls) == 2


# ---- gpu-lease -----------------------------------------------------------------------------------------------------------------------------

needs_flock = pytest.mark.skipif(shutil.which("flock") is None or shutil.which("bash") is None, reason="gpu-lease needs bash and flock(1)")


def gpu_env(tmp_path: Path, count: int = 3) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "RESEARCH_RUN_DIR"}
    return {**env, "GPU_LEASE_DIR": str(tmp_path / "gl"), "GPU_LEASE_COUNT": str(count), "RESEARCH_RUN_ID": "r-test",
            "GPU_LEASE_USAGE_LOG": str(tmp_path / "lease-usage.jsonl"), "GPU_LEASE_UTIL_LOG": str(tmp_path / "util-*.jsonl"),
            "GPU_LEASE_PREEMPT_GRACE_S": "0", "GPU_LEASE_PREEMPT_GRACE_MAX_S": "0"}


@needs_flock
def test_gpu_lease_reports_each_gpus_held_and_busy_time_from_the_sampler_log_and_keeps_cmds_status(tmp_path):
    # started long ago, so the lease's held time is far past its 30 s of readings and the limit at held time doesn't bite
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
def test_gpu_lease_limits_busy_and_sampled_time_to_the_held_time(tmp_path):
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
    """fill_runner pauses cpu-sets jobs while a check holds FILL_YIELD_LOCK, by default the node's slot-d lock, which the candidate's
    check running these tests on a node may hold itself (r20261002-072246-b9fe, -073629-eb44): no test sees the host's."""
    monkeypatch.setenv("FILL_YIELD_LOCK", str(tmp_path / "check-d.lock"))


@needs_flock
def test_fill_runner_counts_a_queued_waiter_but_not_a_lease_that_holds_its_gpus_with_wait_in_its_argv(tmp_path, monkeypatch):
    """A `gpu-lease 1 --wait` that holds its GPU keeps `--wait` in its argv. Counting it as a waiter would freeze fill across node 2
    for as long as any such lease ran; only a request queued in gpu-lease's own wait files is a waiter."""
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
    """`fill_runner.py --help` must not fall through to the runner: a second runner on node 2 adopts the running series job and
    launches the next beside it. Only a bare `fill_runner.py` runs the runner, and only while no other runner holds
    FILL_DIR/.runner.lock."""
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
        assert p.poll() is not None, "a group is gone only once its leader is reaped"
    p.wait()


def test_fill_runner_does_not_count_a_waiter_pinned_inside_keep_free(tmp_path, monkeypatch):
    """A `gpu-lease 1 --on 7 --wait` request, on node 2's kept-free GPU 7, must not stop fill from starting on the free GPUs: fill
    never takes a kept-free GPU, so a waiter pinned inside keep-free holds none back.
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
    """While the agent decides, fill_runner stops no GPU fill for a waiter, since the agent evicts; a waiting or running window
    still stops it (the agent's eviction signals gpu-lease's pid only, which leaves a `research run` workload on the GPU), and
    once agent.lock is free the next tick is the runner's own rules."""
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
    """Node 2's order: circuits' Commits, then pous, then proofs. A queued Commit with no free GPU stops the
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
    for i in range(3):      # an ended lease's line says so, and still names the run that held the GPU last
        line = (tmp_path / "gl" / f"{i}.owner").read_text()
        assert re.match(r"released=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ pid=\d+ run=r-test ", line), line


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


def smi_env(tmp_path: Path, count: int = 2, script: str = 'cat "$(dirname "$0")/apps.csv"') -> dict[str, str]:
    """gpu-lease asking a fake nvidia-smi on PATH, which answers its compute apps from bin/apps.csv (or runs `script`)."""
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    smi = bin_ / "nvidia-smi"
    smi.write_text(f"#!/bin/sh\n{script}\n")
    smi.chmod(0o755)
    (bin_ / "apps.csv").write_text("")
    return {**gpu_env(tmp_path, count=count), "PATH": f"{bin_}:{os.environ['PATH']}", "GPU_LEASE_SMI": "nvidia-smi",
            "GPU_LEASE_UUID_TABLE": ",".join(f"GPU-{k}" for k in "ab"[:count]), "GPU_LEASE_WHO": "bc-test"}


def _paused() -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, "-c", "import signal; print('ready', flush=True); signal.pause()"], stdout=subprocess.PIPE,
                         text=True)
    assert p.stdout.readline().strip() == "ready"
    return p


@needs_flock
def test_gpu_lease_never_grants_a_gpu_whose_memory_a_process_under_no_lease_holds(tmp_path):
    """A cancelled run's vLLM TP workers, which hold no lock, can keep most of each GPU's memory, and a GPU handed to the next
    waiter then fails it for lack of memory. A process over 1 GiB on a free GPU that no
    lease holder owns keeps that GPU from being granted; it is logged, kept in <i>.foreign and shown by `status`; a small one
    doesn't count."""
    env = smi_env(tmp_path)
    stray = _paused()
    try:
        (tmp_path / "bin" / "apps.csv").write_text(f"{stray.pid}, 94000 MiB, GPU-a\n{os.getpid()}, 512 MiB, GPU-b\n")
        r = subprocess.run(["bash", str(GPU_LEASE), "1", "--mem-gb", "0", "--", "sh", "-c", "echo $CUDA_VISIBLE_DEVICES"], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == "1", r.stderr
        assert f"not granting a GPU a foreign process holds memory on: 0 pid={stray.pid} mem=94000MiB user=" in r.stderr
        log = (tmp_path / "gl" / "lease.log").read_text()
        assert f"foreign: 0 pid={stray.pid} mem=94000MiB" in log and "who=bc-test" in log and " cmd=" in log
        assert (tmp_path / "gl" / "0.foreign").read_text().startswith(f"pid={stray.pid} mem=94000MiB")
        assert not (tmp_path / "gl" / "1.foreign").exists()
        for argv in (["2"], ["1", "--on", "0"]):
            r = subprocess.run(["bash", str(GPU_LEASE), *argv, "--mem-gb", "0", "--", "true"], env=env, capture_output=True, text=True)
            assert r.returncode == 75, (argv, r.stderr)
        status = subprocess.run(["bash", str(GPU_LEASE), "status"], env=env, capture_output=True, text=True).stdout.splitlines()
        assert [ln.split()[:3] for ln in status if ln.startswith("foreign ")] == [["foreign", "0", f"pid={stray.pid}"]]
    finally:
        stray.kill()
        stray.wait()
    status = subprocess.run(["bash", str(GPU_LEASE), "status"], env=env, capture_output=True, text=True).stdout
    assert "foreign" not in status                                                      # gone with its process
    r = subprocess.run(["bash", str(GPU_LEASE), "1", "--on", "0", "--mem-gb", "0", "--", "true"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and not (tmp_path / "gl" / "0.foreign").exists()


@needs_flock
def test_a_waiter_waits_out_foreign_memory_and_takes_the_gpu_once_its_process_is_gone(tmp_path):
    env = {**smi_env(tmp_path, count=1), "GPU_LEASE_POLL_S": "0.1"}
    stray = _paused()
    (tmp_path / "bin" / "apps.csv").write_text(f"{stray.pid}, 2048 MiB, GPU-a\n")
    waiter = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--wait", "--mem-gb", "0", "--", "sh", "-c", "echo got=$CUDA_VISIBLE_DEVICES"],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert "not granting a GPU a foreign process holds memory on" in waiter.stderr.readline()
        assert "waiting for 1 GPU(s)" in waiter.stderr.readline()
    finally:
        stray.kill()
        stray.wait()
    out, _ = waiter.communicate()
    assert waiter.returncode == 0 and out.strip() == "got=0"
    assert (tmp_path / "gl" / "lease.log").read_text().count("foreign:") == 1          # logged once, not on every poll


@needs_flock
def test_a_lease_holders_own_processes_and_the_requests_own_ancestors_are_not_foreign(tmp_path):
    """gpu_adopt.py's rule: a process with a lock owner among its ancestors is that lease's, wherever it runs; and the requester
    itself (a run that already uses a GPU and asks gpu-lease for one) is not foreign to its own request."""
    env = smi_env(tmp_path)
    apps = tmp_path / "bin" / "apps.csv"
    holder = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--on", "1", "--mem-gb", "0", "--", sys.executable, "-c",
                               "import os, signal; print(os.getpid(), flush=True); signal.pause()"],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        worker = int(holder.stdout.readline())
        apps.write_text(f"{worker}, 90000 MiB, GPU-a\n")
        r = subprocess.run(["bash", str(GPU_LEASE), "1", "--on", "0", "--mem-gb", "0", "--", "true"], env=env, capture_output=True,
                           text=True)
        assert r.returncode == 0 and "foreign" not in r.stderr, r.stderr
    finally:
        holder.terminate()
        holder.wait()
    r = subprocess.run(["sh", "-c", f'echo "$$, 90000 MiB, GPU-a" > {apps}; bash {GPU_LEASE} 1 --on 0 --mem-gb 0 -- true; echo rc=$?'],
                       env=env, capture_output=True, text=True)
    assert "rc=0" in r.stdout and "foreign" not in r.stderr, r.stderr


@needs_flock
def test_gpu_lease_grants_as_before_and_logs_unverified_when_nvidia_smi_fails_or_never_answers(tmp_path):
    """A wedged nvidia-smi (a GPU fallen off the bus) must never hold every grant up. The never-answering fake blocks on a FIFO
    nobody opens, so the 1 s limit is always reached: the outcome doesn't depend on the clock."""
    fifo = tmp_path / "never"
    os.mkfifo(fifo)
    for script, extra in (("exit 9", {}), (f'cat "{fifo}"', {"GPU_LEASE_SMI_TIMEOUT_S": "1"})):
        env = {**smi_env(tmp_path, count=1, script=script), **extra}
        r = subprocess.run(["bash", str(GPU_LEASE), "1", "--mem-gb", "0", "--", "sh", "-c", "echo $CUDA_VISIBLE_DEVICES"], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == "0", (script, r.stderr)
        assert "unverified: nvidia-smi gave no answer" in r.stderr and "GPUs 0 go to pid" in r.stderr
    assert (tmp_path / "gl" / "lease.log").read_text().count("unverified:") == 2


@needs_flock
def test_a_waiter_pinned_with_on_holds_back_only_its_own_gpus(tmp_path):
    """A waiting `gpu-lease N --on LIST` must not defer every later request, keeping free GPUs outside LIST from everyone. A
    later request may take a GPU outside every older waiter's list (by index or UUID), and an
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
    """A queue of --preemptible trace chunks must not stop its own newest lease every few seconds, each new chunk the oldest
    waiter in turn."""
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


ROOT = NEBIUS.parents[5]
GPU_REGISTRY = ROOT / "tools" / "config" / "verity_config" / "registry" / "gpus.toml"


def test_dry_run_renders_the_cloud_init_and_vm_spec_without_credentials():
    pub = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIdummy research@test"
    r = launch("--dry-run", NEBIUS_SSH_PUBKEY=pub)
    assert r.returncode == 0, r.stderr
    ud, spec = r.stdout.split("# instance create request (ids are placeholders)\n")
    assert ud.splitlines()[1] == "#cloud-config" and f"      - {pub}" in ud and "PRIVATE KEY" not in r.stdout
    assert len(ud.split("\n", 1)[1].rstrip("\n")) <= 32768, "Nebius refuses cloud-init over 32768 characters"
    embedded = {m[0]: gzip.decompress(base64.b64decode(m[1]))
                for m in re.findall(r"  - path: (\S+)\n    permissions: '\d+'\n    encoding: gz\+b64\n    content: (\S+)", ud)}

    def code(script: bytes) -> list[str]:
        return [ln for i, ln in enumerate(script.decode().split("\n")) if i == 0 or not re.match(r"\s*#", ln)]

    for path, src in [("/usr/local/sbin/vy-setup", NEBIUS / "vm_setup.sh"), ("/usr/local/sbin/vy-nebius-stop", NEBIUS / "vm_stop.sh"),
                      ("/usr/local/share/vy/lease.sh", L.LEASE_SH), ("/usr/local/bin/gpu-lease", GPU_LEASE)]:
        assert code(embedded[path]) == code(src.read_bytes()), path
        assert subprocess.run(["bash", "-n"], input=embedded[path], capture_output=True).returncode == 0, path
    for path, name in [("/etc/systemd/system/user@.service.d/vy-cpuset.conf", "user-cpuset.conf"),
                       ("/etc/apt/apt.conf.d/52vy-trust", "apt-security-updates.conf"),
                       ("/etc/needrestart/conf.d/50-vy.conf", "needrestart-list-only.conf")]:
        assert embedded[path] == (NEBIUS / name).read_bytes(), path
    # the first boot writes no unattended-upgrades policy of its own: the one cloud-init wrote is the one deploy.toml installs
    assert "apt.conf.d" not in "\n".join(code((NEBIUS / "vm_setup.sh").read_bytes()))
    assert b"VY_USER=research\n" in embedded["/etc/vy/vy.env"]
    s = json.loads(spec)["spec"]
    assert s["resources"] == {"platform": "gpu-rtx6000-a", "preset": "8gpu-192vcpu-1744gb"}
    nic = s["network_interfaces"][0]
    assert nic["public_ip_address"] == {"static": True} and nic["security_groups"] and s["secondary_disks"][0]["device_id"] == "data"
    assert "# data disk vy-nebius-1-data: attached as it is if it exists, else created (5022 GiB" in ud


def test_a_platform_that_locks_clocks_gets_both_from_its_card_file_and_another_gets_neither(tmp_path):
    """The memory lock is the clock.memory of the card file the GPU registry's entry for the platform names; a locking
    platform is refused before anything is created when the registry refuses an entry naming no card."""
    (entry,) = [g for g in tomllib.loads(GPU_REGISTRY.read_text())["gpu"] if "gpu-rtx6000" in g["nebius_platforms"]]
    clock = tomllib.loads((ROOT / "calibration" / "hardware" / "cards" / f"{entry['card']}.toml").read_text())["clock"]

    def settings(**env: str) -> list[str]:
        r = launch("--dry-run", NEBIUS_SSH_PUBKEY="ssh-ed25519 dummy", **env)
        assert r.returncode == 0, r.stderr
        b64 = re.search(r"  - path: /etc/vy/vy.env\n    permissions: '\d+'\n    encoding: gz\+b64\n    content: (\S+)", r.stdout)[1]
        return [ln for ln in gzip.decompress(base64.b64decode(b64)).decode().splitlines() if "CLOCK" in ln]

    gpu = int(re.search(r"^GPU_LOCK_MHZ=(\d+)$", (NEBIUS / "launch.sh").read_text(), re.M)[1])
    assert gpu <= clock["sm"]["value"]
    assert settings() == ["VY_LOCK_CLOCKS=1", f"VY_GPU_CLOCK_MHZ={gpu}", f"VY_MEM_CLOCK_MHZ={clock['memory']['value']}"]
    assert settings(NEBIUS_PLATFORM_PREFIX="gpu-h100-sxm") == ["VY_LOCK_CLOCKS=0"]
    copy = tmp_path / NEBIUS.relative_to(ROOT)
    copy.mkdir(parents=True)
    for name in ("launch.sh", "common.sh"):
        shutil.copy(NEBIUS / name, copy / name)
    package = GPU_REGISTRY.parents[1]
    shutil.copytree(package, tmp_path / package.relative_to(ROOT), ignore=shutil.ignore_patterns("__pycache__"))
    registry = tmp_path / GPU_REGISTRY.relative_to(ROOT)
    registry.write_text(re.sub(r"^card = .*\n", "", GPU_REGISTRY.read_text(), flags=re.M))
    base = {k: v for k, v in os.environ.items() if not k.startswith(("NEBIUS_", "RUNPOD_"))}
    r = subprocess.run(["bash", str(copy / "launch.sh"), "--dry-run"], env={**base, "NEBIUS_SSH_PUBKEY": "ssh-ed25519 dummy"},
                       capture_output=True, text=True)
    assert r.returncode == 2 and f"GPU {entry['id']} names no card" in r.stderr and not r.stdout, r.stderr


def test_the_first_boot_refuses_to_lock_with_a_clock_unset_before_writing_anything(tmp_path):
    """vm_setup.sh has no clock of its own: told to lock with either clock unset or empty it stops, naming both, before its
    first write; untold, it locks nothing and writes no clock unit."""
    text = (NEBIUS / "vm_setup.sh").read_text()
    head, rest = text.split("\nfi\n", 1)
    assert "cat >" not in head and "systemctl" not in head
    assert re.search(r'^\[ "\$LOCK_CLOCKS" = 0 \] \|\| cat > /etc/systemd/system/vy-clocks\.service', rest, re.M)
    env_file = tmp_path / "vy.env"
    prefix = tmp_path / "head.sh"
    prefix.write_text(head.replace("/etc/vy/vy.env", str(env_file)) + "\nfi\necho passed\n")
    for settings, refused in [("VY_LOCK_CLOCKS=1\nVY_GPU_CLOCK_MHZ=2100\n", True),
                              ("VY_LOCK_CLOCKS=1\nVY_GPU_CLOCK_MHZ=\nVY_MEM_CLOCK_MHZ=12481\n", True),
                              ("VY_LOCK_CLOCKS=1\nVY_GPU_CLOCK_MHZ=2100\nVY_MEM_CLOCK_MHZ=12481\n", False),
                              ("VY_USER=research\n", False), ("VY_LOCK_CLOCKS=0\n", False)]:
        env_file.write_text(settings)
        r = subprocess.run(["bash", str(prefix)], capture_output=True, text=True)
        assert (r.returncode, r.stdout) == ((2, "") if refused else (0, "passed\n")), settings
        assert not refused or ("VY_GPU_CLOCK_MHZ" in r.stderr and "VY_MEM_CLOCK_MHZ" in r.stderr), settings


def test_the_first_boot_writes_and_applies_the_user_namespace_setting_on_the_nodes_deploy_gives_it():
    """A node deploy.toml gives /etc/sysctl.d/61-vy-userns.conf has it from its first boot, from the same source, so a check
    or a training job sandboxed there doesn't fail before the first install; a node it doesn't name gets none."""
    manifest = tomllib.loads((NEBIUS / "deploy.toml").read_text())
    entry = next(f for f in manifest["file"] if f["path"] == "/etc/sysctl.d/61-vy-userns.conf")
    src = (NEBIUS / "../../../../../.." / entry["src"]).read_bytes()
    for name in ("vy-nebius-cpu-14", "vy-nebius-2", "vy-nebius-1", "vy-auditor-1"):
        r = launch("--dry-run", NEBIUS_SSH_PUBKEY="ssh-ed25519 dummy", NEBIUS_VM_NAME=name)
        assert r.returncode == 0, r.stderr
        ud = r.stdout.split("# instance create request (ids are placeholders)\n")[0]
        assert len(ud.split("\n", 1)[1].rstrip("\n")) <= 32768, name
        embedded = {m[0]: gzip.decompress(base64.b64decode(m[1]))
                    for m in re.findall(r"  - path: (\S+)\n    permissions: '0644'\n    encoding: gz\+b64\n    content: (\S+)", ud)}
        assert embedded.get(entry["path"]) == (src if name in entry["nodes"] else None), name
    assert name not in entry["nodes"] and "vy-nebius-cpu-14" in entry["nodes"]
    assert "sysctl -q -e -p /etc/sysctl.d/61-vy-userns.conf" in (NEBIUS / "vm_setup.sh").read_text()


def test_the_first_boot_keeps_bash_logout_from_setting_a_login_shells_exit_status(tmp_path):
    """Ubuntu's stock ~/.bash_logout ends in `[ -x /usr/bin/clear_console ] && ...`, which fails with no console, so a job run as
    `bash -lc` under set -e exited 1 whatever it did. vm_setup.sh's loop over skel's and every user's file, run here on copies,
    appends `|| true` to that line once and leaves any other file as it is."""
    loop = re.search(r"^for f in [^\n]*\.bash_logout; do\n.*?^done\n", (NEBIUS / "vm_setup.sh").read_text(), re.S | re.M)
    assert loop and "/etc/skel/.bash_logout" in loop[0] and "/home/*/.bash_logout" in loop[0]
    stock = ("# ~/.bash_logout: executed by bash(1) when login shell exits.\n\n"
             "if [ \"$SHLVL\" = 1 ]; then\n    [ -x /usr/bin/clear_console ] && /usr/bin/clear_console -q\nfi\n")
    files = {"stock": stock, "other": "echo bye\n", "mended": stock.replace(" -q\n", " -q || true\n")}
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    script = re.sub(r"^for f in [^;]*;", 'for f in "$@";', loop[0], flags=re.M)

    def run() -> dict[str, str]:
        r = subprocess.run(["bash", "-euc", script, "_", *(str(tmp_path / n) for n in files), str(tmp_path / "absent")],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return {n: (tmp_path / n).read_text() for n in files}

    first = run()
    assert first == {**files, "stock": files["mended"]}
    assert run() == first
    r = subprocess.run(["bash", "-c", f"set -e; SHLVL=1; . {tmp_path / 'stock'}"], capture_output=True, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, "with no console the mended file's last status is 0"


def test_a_vm_can_boot_on_another_ones_kept_data_disk():
    """n1's replacement may boot on vy-nebius-1-data: launch.sh and teardown.sh look the data disk up by this name."""
    r = launch("--dry-run", NEBIUS_SSH_PUBKEY="ssh-ed25519 dummy", NEBIUS_VM_NAME="vy-nebius-cpu-4", NEBIUS_PLATFORM_PREFIX="cpu-d3",
               NEBIUS_PRESET="192vcpu-768gb", NEBIUS_DATA_DISK="vy-nebius-1-data")
    assert r.returncode == 0, r.stderr
    assert "# data disk vy-nebius-1-data: attached as it is if it exists" in r.stdout
    assert json.loads(r.stdout.split("(ids are placeholders)\n")[1])["metadata"]["name"] == "vy-nebius-cpu-4"
    for script in ("launch.sh", "teardown.sh"):
        assert 'disk_id "$DATA_NAME"' in (NEBIUS / script).read_text() and '"$VM_NAME-data"' not in (NEBIUS / script).read_text()


def test_a_data_disk_of_a_type_that_comes_in_93_gib_units_is_refused_at_another_size_before_anything_is_created():
    """Nebius refuses such a data disk only after launch.sh has created the boot disk, which then stays behind, billed."""
    cpu = {"NEBIUS_SSH_PUBKEY": "ssh-ed25519 dummy", "NEBIUS_VM_NAME": "vy-nebius-cpu-14", "NEBIUS_PLATFORM_PREFIX": "cpu-d3",
           "NEBIUS_PRESET": "192vcpu-768gb"}
    for kind in ("network_ssd_io_m3", "network_ssd_non_replicated"):
        r = launch("--dry-run", NEBIUS_DATA_TYPE=kind, NEBIUS_DATA_GIB="1280", **cpu)
        assert r.returncode == 1 and f"not a multiple of 93, the unit {kind} disks come in: 1209 is the next size down" in r.stderr
        r = launch("--dry-run", NEBIUS_DATA_TYPE=kind, NEBIUS_DATA_GIB="1209", **cpu)
        assert r.returncode == 0 and f"else created (1209 GiB {kind})" in r.stdout, r.stderr
    r = launch("--dry-run", NEBIUS_DATA_TYPE="network_ssd", NEBIUS_DATA_GIB="1280", **cpu)
    assert r.returncode == 0 and "else created (1280 GiB network_ssd)" in r.stdout, r.stderr


def test_resolve_project_refuses_to_go_on_without_the_projects_id(tmp_path):
    """No file holds a project's id: with NEBIUS_PROJECT_ID unset, launch.sh's and teardown.sh's resolve_project ends naming
    it before any Nebius call; set, it is the project; set empty, the project is looked up by name, as before."""
    script = (f". {NEBIUS / 'common.sh'}; nb() {{ echo \"$*\" >> \"$CALLS\"; echo '{{\"items\": []}}'; }}; "
              "resolve_project && echo \"project $PROJECT_ID\"")
    calls = tmp_path / "calls"
    held = {k: v for k, v in os.environ.items() if not k.startswith("NEBIUS_")}

    def resolve(**env: str) -> subprocess.CompletedProcess:
        calls.unlink(missing_ok=True)
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                              env={**held, "CALLS": str(calls), "XDG_RUNTIME_DIR": str(tmp_path), **env})

    r = resolve()
    assert r.returncode == 1 and r.stderr.startswith("nebius: NEBIUS_PROJECT_ID is not set: the id of the project") and not calls.exists()
    r = resolve(NEBIUS_PROJECT_ID="project-e00example")
    assert (r.returncode, r.stdout) == (0, "project project-e00example\n"), r.stderr
    assert calls.read_text() == "compute instance list --parent-id project-e00example\n"
    r = resolve(NEBIUS_PROJECT_ID="", NEBIUS_TENANT_ID="tenant-e00example")
    assert r.returncode == 1 and "no project named vy-nebius in tenant tenant-e00example" in r.stderr
    assert calls.read_text() == "iam v2 project list --parent-id tenant-e00example --all\n"


def test_launch_refuses_to_spend_without_yes_and_a_lease():
    r = launch("--max-hours", "4")
    assert r.returncode == 1 and "pass --yes" in r.stderr
    r = launch("--yes")
    assert r.returncode == 1 and "--max-hours H" in r.stderr
    r = launch("--yes", "--max-hours", "4", NEBIUS_SSH_PUBKEY="ssh-ed25519 dummy", NEBIUS_BIN="/bin/true")
    assert r.returncode == 1 and "NEBIUS_SA_ID is not set" in r.stderr


def _wait_state(tmp_path, want: str, states: list[str]) -> subprocess.CompletedProcess:
    """`wait_state ID WANT` against a VM whose successive `instance get`s read ``states`` (the last one repeats)."""
    seq = tmp_path / "states"
    seq.write_text("\n".join(states) + "\n")
    script = (f". {NEBIUS / 'common.sh'}; "
              'nb() { s=$(head -1 "$SEQ"); [ "$(wc -l < "$SEQ")" -gt 1 ] && sed -i 1d "$SEQ"; '
              'printf \'{"status": {"state": "%s"}}\\n\' "$s"; }; '
              f"wait_state computeinstance-x {want} 20")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          env={**os.environ, "SEQ": str(seq), "XDG_RUNTIME_DIR": str(tmp_path), "NEBIUS_POLL_S": "0"})


def test_wait_state_ends_when_a_refused_start_goes_back_to_stopped(tmp_path):
    r = _wait_state(tmp_path, "RUNNING", ["STOPPED", "STARTING", "STARTING", "STOPPED"])
    assert r.returncode == 1 and "went back to STOPPED: Nebius refused its start" in r.stderr
    assert (tmp_path / "states").read_text() == "STOPPED\n", "it read past the fallback"
    r = _wait_state(tmp_path, "RUNNING", ["STOPPED", "STARTING", "RUNNING"])
    assert r.returncode == 0, r.stderr
    r = _wait_state(tmp_path, "RUNNING", ["STOPPED"])                    # never started: it waits out its tries
    assert r.returncode == 1 and "is STOPPED, not RUNNING, after waiting" in r.stderr
    r = _wait_state(tmp_path, "STOPPED", ["STARTING", "STOPPED"])
    assert r.returncode == 0, r.stderr


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
    # adjacent literals, so no secret scanner (tests/test_repository.py among them) reads the dummy PEM as a credential
    body = "-----BEGIN PRIVATE" " KEY-----\ndummy line one\ndummy line two\n-----END PRIVATE" " KEY-----\n"

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


def test_nebius_auth_reads_a_der_key_whose_base64_fits_the_cursor_secret_limit(tmp_path):
    """An RSA-4096 key's PEM is ~3272 bytes, whose base64 (~4364) exceeds Cursor's 4096-byte secret limit, so the secret may
    hold base64 of the DER (~3132 chars) instead; nebius_auth converts it back to PEM through openssl.  A throwaway
    RSA-2048 key generated here keeps the test fast; the size arithmetic scales, and the decode path is identical."""
    auth = _auth_harness(tmp_path)
    pem = subprocess.run(["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048"],
                         capture_output=True, check=True).stdout
    der = subprocess.run(["openssl", "pkey", "-outform", "DER"], input=pem, capture_output=True, check=True).stdout
    assert not der.startswith(b"-----") and der[0] == 0x30    # the two formats are told apart by their first bytes

    r, key = auth(NEBIUS_SA_PRIVATE_KEY_B64=base64.b64encode(der).decode())
    assert r.returncode == 0, r.stderr
    assert key.startswith("-----BEGIN PRIVATE" " KEY-----")
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
    split.write_text("none\n")
    with pytest.raises(remote.Refused, match="Kueue's"):
        remote._runner_env({**req, "gpus": 1}, tmp_path / "runs", None, {})


def test_a_direct_run_takes_gpus_only_through_the_lease_on_a_gpu_lease_host(tmp_path, monkeypatch):
    """On a GPU VM whose GPUs gpu-lease hands out (/etc/vy/gpu-lease-host), a direct run sees no GPU whatever its --env says,
    so a TP8 job can't open CUDA beside the lease; `--gpus N` runs the workload under `gpu-lease N --wait` (a FCFS wait, capped at
    its --mem-gb). A machine without gpu-lease refuses --gpus; a queued job's GPUs stay the queue's."""
    host = tmp_path / "gpu-lease-host"
    monkeypatch.setenv("GPU_LEASE_HOST_FILE", str(host))
    req = {"env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}, "command": ["python", "serve.py"]}
    with pytest.raises(remote.Refused, match="aren't handed out by gpu-lease"):
        remote._runner_env({**req, "gpus": 8}, tmp_path / "runs", None, {})
    assert remote._runner_env(req, tmp_path / "runs", None, {})["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    host.write_text("")
    rec: dict = {}
    assert remote._runner_env(req, tmp_path / "runs", None, rec)["CUDA_VISIBLE_DEVICES"] == ""
    assert rec["gpus"].startswith("none: gpu-lease hands out every GPU here") and remote.gpu_lease_wrap(req) == []
    rec = {}
    assert remote._runner_env({**req, "gpus": 8}, tmp_path / "runs", None, rec)["CUDA_VISIBLE_DEVICES"] == ""
    assert rec["gpus"].startswith("8 through `gpu-lease 8 --wait`")
    assert remote.gpu_lease_wrap({**req, "gpus": 8}) == ["gpu-lease", "8", "--wait", "--mem-gb", "0", "--"]
    assert remote.gpu_lease_wrap({**req, "gpus": 1, "mem_gb": 120.5}) == ["gpu-lease", "1", "--wait", "--mem-gb", "120", "--"]
    queued = {**req, "gpus": 8, remote.QUEUE_WRAP: ["systemd-run"]}
    rec = {}
    assert remote._runner_env(queued, tmp_path / "runs", None, rec)["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "gpus" not in rec and remote.gpu_lease_wrap(queued) == []


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


def test_fill_runner_gives_a_tp2_commit_two_gpus_on_one_numa_node_and_commits_a_longer_limit(tmp_path, monkeypatch):
    """With node 2 first, a Commit guest may take 2 GPUs (a TP2 row), from one NUMA node when it
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
    at = datetime.datetime(2026, 10, 2, 6, 30, tzinfo=datetime.UTC).timestamp()
    assert fr.header(gate, at)["max_min"] == 60
    assert fr.header(soak, at)["max_min"] == 495                        # never past the line's until: 06:30 + 495 min = 14:45
    assert fr.header(other, at)["max_min"] == 30                        # a malformed line grants nothing
    late = datetime.datetime(2026, 10, 2, 14, 44, 30, tzinfo=datetime.UTC).timestamp()
    assert fr.header(soak, late)["max_min"] == 30                       # under a minute left: the exception is over
    assert fr.header(soak, late + 3600)["max_min"] == 30
    calls = []
    monkeypatch.setattr(fr.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    assert fr.scope({"unit": "fill-verity-1"}, "freeze") and calls == [["systemctl", "--user", "freeze", "fill-verity-1.scope"]]
    assert not fr.scope({"unit": None}, "freeze") and len(calls) == 1


def test_the_verity_pool_lends_idle_slots_to_pous_cpu_jobs_and_takes_them_back(tmp_path, monkeypatch):
    """With no Verity job waiting, queued pous CPU jobs start in the Verity pool's free slots (its CPUs, a scope, its memory limits);
    a Verity job that finds no slot requeues the newest lent job, and nothing more is lent while it waits."""
    monkeypatch.setenv("FILL_DIR", str(tmp_path / "fill"))
    monkeypatch.setenv("FILL_VERITY_CPU_SLOTS", "2")
    monkeypatch.setenv("FILL_VERITY_UNTIL", "9999-01-01T00:00:00+00:00")
    monkeypatch.setenv("FILL_VERITY_STOP", "9999-01-01T01:00:00+00:00")
    spec = importlib.util.spec_from_file_location("fill_runner_lend", FILL_RUNNER)
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    monkeypatch.setattr(fr, "disk_pct", lambda path: 0.0)
    monkeypatch.setattr(fr, "claim", lambda name, k: ("48-55", None))   # cores: test_nebius_fill_timing's pool tests
    started, stopped = [], []
    monkeypatch.setattr(fr, "start", lambda p, h, backfill=False, claim_fd=None: started.append((p.name, h)) or fr.running.__setitem__(
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
    """A job that finishes while a restarted runner has adopted it goes to done/, not back to the queue, where its rerun would
    delete its result; exit 0 is done even when a stop raced it; and a job whose exit the runner can't learn
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
    """Node 2's store has no remote, and a push loop takes attempts without their logs, while a row sync mirrors millions of files
    and overruns its window. Every ended run goes to node 1's Job
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
    assert f"find /n1/src/abc/tools/research /n1/src/abc/{private.TREE_NAMES} /workspace/custody/python -path /workspace/custody/python/.lock -prune" in n2.calls()
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
    """Node 2's timed windows (PoUW's measurements) run with nothing else on the node, and a round's rsync and hashing must not run
    through them, even at Nice=19. A round, and each push, starts only while the fill runner's status is under 60 s
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
    def at(hhmm):  # N2_NOW is 03:00Z
        return N2_NOW + (int(hhmm[:2]) - 3) * 3600 + int(hhmm[2:]) * 60
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


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="node_ops reads /proc")
def test_node_ops_reads_a_runs_pid_from_its_runners_transition_not_the_last(tmp_path, monkeypatch):
    """`tele set-timeout` appends a `running` transition without a pid, so reading the pid from the last transition only would
    alert dead-run on a run whose runner is alive."""
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


def test_gpu_adopt_holds_the_lease_of_a_gpu_used_outside_it_until_the_process_exits(monkeypatch, tmp_path):
    """On a GPU VM a process whose GPU no lease of its ancestors covers gets that GPU's lease in its run's name (so gpu-lease
    hands it to nobody else) until it exits; a GPU another lease holds is a collision, unless that lease names the process's own run
    (a holder that handed its locks over a socket); each is logged once."""
    spec = importlib.util.spec_from_file_location("gpu_adopt_under_test", NEBIUS / "gpu_adopt.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    leases, runs = tmp_path / "gpu-lease", tmp_path / "runs"
    leases.mkdir()
    (runs / "r20261018-000000-abcd").mkdir(parents=True)
    (runs / "r20261018-000000-abcd" / "job.json").write_text(json.dumps({"ownership": {"lane": "circuits"}}))
    monkeypatch.setattr(m, "LEASE_DIR", leases)
    monkeypatch.setattr(m, "RUNS", runs)
    monkeypatch.setattr(m, "LOG", tmp_path / "adopt.jsonl")
    held = []
    for i, owner in ((0, os.getpid()), (2, 1)):          # GPU 0 leased by this test (an ancestor of its children), GPU 2 by pid 1
        fd = os.open(leases / f"{i}.lock", os.O_WRONLY | os.O_CREAT, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX)
        (leases / f"{i}.owner").write_text(f"pid={owner} run=r-x who=t uuid=GPU-{i} since=2026-10-18T00:00:00Z cmd=x\n")
        held.append(fd)
    block = [sys.executable, "-c", "import sys; print(flush=True); sys.stdin.read()"]       # lives until killed
    fd = os.open(leases / "3.lock", os.O_WRONLY | os.O_CREAT, 0o666)          # GPU 3 leased by pid 1 in run r20261018-000000-beef
    fcntl.flock(fd, fcntl.LOCK_EX)
    (leases / "3.owner").write_text("pid=1 run=r20261018-000000-beef who=t uuid=GPU-3 since=2026-10-18T00:00:00Z cmd=x\n")
    held.append(fd)
    leased, stray, other, handed = (subprocess.Popen(block + extra, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                                    for extra in ([], ["research", "run", "--id", "r20261018-000000-abcd"], [],
                                                  ["research", "run", "--id", "r20261018-000000-beef"]))
    for p in (leased, stray, other, handed):
        p.stdout.readline()                                  # it has exec'd: /proc/PID/cmdline is its own, not empty
    try:
        monkeypatch.setattr(m, "compute_apps", lambda: [(leased.pid, "GPU-a"), (stray.pid, "GPU-b"), (other.pid, "GPU-c"),
                                                        (handed.pid, "GPU-d")])
        monkeypatch.setattr(m, "gpu_index", lambda: {"GPU-a": 0, "GPU-b": 1, "GPU-c": 2, "GPU-d": 3})
        a = m.Adopter()
        monkeypatch.setattr(m, "SPLIT", tmp_path / "direct-gpus")
        (tmp_path / "direct-gpus").write_text("none\n")
        assert a.check(1_800_000_000.0) == [], "a cluster node's GPUs are Kueue's"
        (tmp_path / "direct-gpus").unlink()
        found = a.check(1_800_000_000.0)
        assert [(r["kind"], r["gpu"], r["pid"]) for r in found] == [("adopted", 1, stray.pid), ("collision", 2, other.pid)]
        assert found[0]["run"] == "r20261018-000000-abcd" and found[0]["lane"] == "circuits" and found[1]["lease_owner"] == 1
        owner = (leases / "1.owner").read_text()
        assert f"pid={os.getpid()} run=r20261018-000000-abcd who=adopted:circuits uuid=GPU-b" in owner
        assert m.lease_owner(1) == os.getpid()                 # what gpu-lease sees: GPU 1 is held
        a.check(1_800_000_030.0)
        assert len((tmp_path / "adopt.jsonl").read_text().splitlines()) == 2, "each finding is logged once"
        stray.kill()
        stray.wait()
        monkeypatch.setattr(m, "compute_apps", lambda: [(leased.pid, "GPU-a")])
        assert a.check(1_800_000_060.0) == [] and m.lease_owner(1) is None
    finally:
        for p in (leased, stray, other, handed):
            p.kill()
            p.wait()
        for fd in held:
            os.close(fd)


def test_fill_runner_on_numa_0_terms_freezes_cpu_jobs_while_a_window_waits_and_binds_their_memory(tmp_path, monkeypatch):
    """PoUW's terms for CPU fill on 0-47: frozen from the moment a window waits, and memory kept off NUMA 0. Both are
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
    """An owner in cpu-sets (compute accounting's verifies) runs on 0-47, 4 at a time, at ionice idle, capped from a
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
    """GPU 0's verifies run on 0-47 yielding to slot d's candidates' checks. A check runs in
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


def test_fill_runner_fence_command_limits_a_fence_keeps_one_per_owner_and_logs_it(tmp_path, monkeypatch, capsys):
    """A fence on node 2 (an honest-latency measurement's) would otherwise take four steps by hand: keep-free, a respawned pane for
    fill's CPUs, a max-min line and a revert timer. `fill_runner.py fence` writes one self-expiring line within the limits instead
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


def test_a_run_on_a_stopped_nebius_vm_starts_it_and_waits_for_its_ssh(monkeypatch):
    """The H200s stop after 2 h idle with their disks kept, and `research run --on` brings one back: a machine whose registry
    entry names its nebius_instance_id and whose ssh doesn't answer is started through the API, then waited for."""
    answers, ran = iter([1, 1, 1, 0, 0]), []
    monkeypatch.setattr(remote, "ssh_run", lambda ssh, argv, timeout=120: ran.append(argv) or subprocess.CompletedProcess(argv, next(answers)))
    calls, t = [], [0.0]

    def nebius(*args):
        calls.append(args[2])
        return {"status": {"state": "STOPPED"}} if args[2] == "get" else {}

    def sleep(s):
        t[0] += s

    m = {"name": "vy-glm53-eu", "nebius_instance_id": "computeinstance-x"}
    assert remote.wake(m, ["ssh"], nebius=nebius, sleep=sleep, clock=lambda: t[0]) == "started from STOPPED"
    assert calls == ["get", "start"] and t[0] == 3 * remote.WAKE_POLL_S
    assert ran[-1] == ["sh", "-c", remote.TRUST_UPGRADE], "a woken VM upgrades its trust store before the run (INF-314)"
    assert remote.wake({"name": "vy-nebius-2"}, ["ssh"], nebius=nebius) is None, "no instance id: nothing to wake"
    monkeypatch.setattr(remote, "ssh_run", lambda ssh, argv, timeout=120: subprocess.CompletedProcess(argv, 1))
    with pytest.raises(remote.Refused, match="didn't answer in 15 min"):
        remote.wake(m, ["ssh"], nebius=nebius, sleep=sleep, clock=lambda: t[0])

    def no_creds(*args):
        raise RuntimeError("no NEBIUS_SA_ID")

    with pytest.raises(remote.Refused, match="ask @infra to start computeinstance-x"):
        remote.wake(m, ["ssh"], nebius=no_creds)

    def plain_start(*args):   # the CLI's --async start prints its operation's id, which fleet.nb can't parse
        if args[2] == "start":
            raise json.JSONDecodeError("Expecting value", "computeoperation-x", 0)
        return nebius(*args)

    answers = iter([1, 0, 0])
    monkeypatch.setattr(remote, "ssh_run", lambda ssh, argv, timeout=120: subprocess.CompletedProcess(argv, next(answers)))
    assert remote.wake(m, ["ssh"], nebius=plain_start, sleep=sleep, clock=lambda: t[0]) == "started from STOPPED"


def test_a_run_names_the_agent_that_sent_it_a_worker_its_own_conversation():
    """OOM and lost-check alerts name a run's sender from its request; workers never set RESEARCH_AGENT_URL, and a worker
    that inherits its coordinator's would be reported as the coordinator. Every agent's shell carries its own conversation id."""
    assert remote.sender({"CURSOR_CONVERSATION_ID": "bc-worker", "RESEARCH_AGENT_URL": "https://cursor.com/agents/bc-coord"}) == \
        "https://cursor.com/agents/bc-worker"
    assert remote.sender({"RESEARCH_AGENT_URL": "https://cursor.com/agents/bc-coord"}) == "https://cursor.com/agents/bc-coord"
    assert remote.sender({}) is None
