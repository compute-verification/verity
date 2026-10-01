"""A leased Commit (`pipeline.gpu_lease`): only the Commit process holds the GPU, from right before its engine to its exit, under the
lease tool; the row and every other child see no CUDA device; an unleased row is unchanged."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from verity_vllm.pipeline import gpu_lease as GL
from verity_vllm.pipeline import row_driver as RD


def fake_lease(tmp_path: Path, gpu: str = "5") -> str:
    """A stand-in for gpu-lease: records its arguments, locks <gpu>.lock on a descriptor CMD inherits, sets CUDA_VISIBLE_DEVICES and
    runs CMD as its child with its stdin, as gpu-lease does."""
    p = tmp_path / "gpu-lease"
    p.write_text(f'#!/usr/bin/env bash\necho "$*" > "{tmp_path}/lease.args"\nwhile [ "$1" != -- ]; do shift; done; shift\n'
                 f'exec 7>>"{tmp_path}/{gpu}.lock"; flock -x 7\n'
                 f'export CUDA_VISIBLE_DEVICES={gpu} CUDA_DEVICE_ORDER=PCI_BUS_ID GPU_LEASE_UUID=GPU-{gpu}\n"$@" 0<&0 &\nwait $!\n')
    p.chmod(0o755)
    return str(p)


def commit(script: str) -> list[str]:
    """A Commit stand-in: `script` after the CLI's environment and `GL.acquire()`, with `before` the devices it saw first."""
    return [sys.executable, "-c", "import json, os, sys\nfrom verity_vllm.pipeline import cli, gpu_lease as GL\n"
            "before = os.environ['CUDA_VISIBLE_DEVICES']\ncli.process_settings(os.environ)\nGL.acquire()\n" + script]


def released(path: Path) -> bool:
    """Waits for the lock on `path` to be free (the holder ends on the Commit's exit)."""
    return subprocess.run(["flock", str(path), "true"]).returncode == 0


def test_only_an_item_with_a_lease_tool_is_leased_and_its_devices_are_hidden_outside_the_lease():
    assert not GL.leased({}) and not GL.leased({"GPU_LEASE": ""}) and GL.leased({"GPU_LEASE": "/x/gpu-lease"})
    env = {"CUDA_VISIBLE_DEVICES": "0"}
    GL.hide_devices(env)
    assert env == {"CUDA_VISIBLE_DEVICES": ""}
    env = {"GPU_LEASE": "/x/gpu-lease", "CUDA_VISIBLE_DEVICES": ""}
    GL.acquire(environ=env)                            # not started by a leased row's Commit stage: nothing taken
    assert env == {"GPU_LEASE": "/x/gpu-lease", "CUDA_VISIBLE_DEVICES": ""}


def test_the_commit_requests_its_tasks_gpus_for_at_most_its_timeout():
    env = {"GPU_LEASE": "/l/gpu-lease", GL.STARTED: "/row/gpu_lease.started", GL.MAX_MIN: str(GL.max_min(2400))}
    argv = GL.holder_argv(env, 7)
    assert argv[:9] == ["/l/gpu-lease", "1", "--wait", "--max-min", "40", "--mem-gb", "0", "--", sys.executable]
    assert argv[-2:] == ["/row/gpu_lease.started", "7"]
    assert GL.max_min(61) == 2 and GL.max_min(1) == 1
    assert GL.holder_argv({**env, "GPU_LEASE_N": "2"}, 7)[:2] == ["/l/gpu-lease", "2"]
    assert GL.STARTED != GL.TOOL_START and GL.tool_env({**env, GL.TOOL_START: "/row/x"}) == env    # the tool's own start override


def test_the_commit_takes_the_lease_itself_holds_its_lock_and_sees_only_the_leased_gpu(tmp_path):
    env = {"PATH": os.environ["PATH"], "GPU_LEASE": fake_lease(tmp_path)}
    GL.hide_devices(env)
    seen = tmp_path / "seen"
    cmd = commit(f"fds = [f for f in os.listdir('/proc/self/fd') if os.path.realpath('/proc/self/fd/' + f) == {str(tmp_path / '5.lock')!r}]\n"
                 f"json.dump(dict(before=before, after=os.environ['CUDA_VISIBLE_DEVICES'], uuid=os.environ['GPU_LEASE_UUID'], locks=len(fds)), "
                 f"open({str(seen)!r}, 'w'))\n")
    log = tmp_path / "commit.log"
    timeouts = []

    def run_tl(to: int, argv: list[str]) -> int:
        timeouts.append(to)
        with open(log, "a") as f:
            return subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT).returncode

    rc, rec = GL.run(str(tmp_path), str(log), cmd, 2400, env, run_tl, clock=lambda: 0.0)
    assert rc == 0, log.read_text()
    assert json.loads(seen.read_text()) == {"before": "", "after": "5", "uuid": "GPU-5", "locks": 1}
    assert env == {"PATH": os.environ["PATH"], "GPU_LEASE": env["GPU_LEASE"], "CUDA_VISIBLE_DEVICES": ""}
    assert timeouts == [4800] and (tmp_path / "lease.args").read_text().startswith(f"1 --wait --max-min 40 --mem-gb 0 -- {sys.executable} -c ")
    stamp = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"
    assert re.fullmatch(stamp, rec["started"]) and re.fullmatch(stamp, rec["requested"]) and rec["held_min"] is not None
    assert json.loads((tmp_path / GL.RECORD).read_text()) == rec and rec["max_min"] == 40 and rec["rc"] == 0
    assert released(tmp_path / "5.lock")


def test_the_lease_tool_ending_first_ends_the_commit(tmp_path):
    env = {"PATH": os.environ["PATH"], "GPU_LEASE": fake_lease(tmp_path), GL.STARTED: str(tmp_path / "gpu_lease.started"), GL.MAX_MIN: "1"}
    GL.hide_devices(env)
    r = subprocess.run(commit("GL._HELD[0][0].kill()\nimport threading\nthreading.Event().wait()\n"), env=env, capture_output=True, text=True)
    assert r.returncode == -9 and "GPU lease ended under the Commit" in r.stderr, r.stderr


def test_a_commit_that_never_got_its_gpu_records_no_start(tmp_path):
    rc, rec = GL.run(str(tmp_path), str(tmp_path / "commit.log"), ["true"], 60, {"GPU_LEASE": "/l/gpu-lease"},
                     lambda to, argv: 75, clock=lambda: 0.0)
    assert rc == 75 and rec["started"] is None and rec["requested"] is None and rec["held_min"] is None and rec["waited_min"] is None


class _Ticks(threading.Event):
    """An event whose wait returns `n` times without being set (one heartbeat each), then as if set."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def wait(self, timeout: float | None = None) -> bool:
        self.n -= 1
        return self.n < 0


def test_a_queued_commit_writes_a_heartbeat_to_its_log_until_the_gpu_is_held(tmp_path):
    log, started, requested = tmp_path / "commit.log", tmp_path / "gpu_lease.started", tmp_path / "gpu_lease.requested"
    GL._heartbeat(str(log), str(started), _Ticks(3), 60)
    assert not log.exists()                            # the Commit's CPU work before its request is not a wait
    requested.write_text("2026-10-01T16:00:00Z\n")
    GL._heartbeat(str(log), str(started), _Ticks(3), 60)
    assert log.read_text().count("waiting for a GPU lease") == 3
    started.write_text("2026-10-01T16:00:00Z\n")
    GL._heartbeat(str(log), str(started), _Ticks(3), 60)
    assert log.read_text().count("waiting for a GPU lease") == 3


def test_the_cuda_jit_cache_persists_beside_the_trees_triton_cache(tmp_path):
    env = {"TRITON_CACHE_DIR": str(tmp_path / "cache" / "triton" / "tree1")}
    RD.jit_cache(env)
    assert env["CUDA_CACHE_PATH"] == str(tmp_path / "cache" / "cuda" / "tree1") and Path(env["CUDA_CACHE_PATH"]).is_dir()
    assert env["CUDA_CACHE_MAXSIZE"] == str(4 << 30)
    kept = {"TRITON_CACHE_DIR": "/t/triton/x", "CUDA_CACHE_PATH": "/mine"}
    RD.jit_cache(kept)
    assert kept == {"TRITON_CACHE_DIR": "/t/triton/x", "CUDA_CACHE_PATH": "/mine"}
    none: dict[str, str] = {}
    RD.jit_cache(none)
    assert none == {}
