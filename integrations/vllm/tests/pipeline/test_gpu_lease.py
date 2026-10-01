"""A leased Commit (`pipeline.gpu_lease`): only the Commit process holds the GPU, under the lease tool; the row and every other child
see no CUDA device; an unleased row is unchanged."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from pathlib import Path

from verity_vllm.pipeline import gpu_lease as GL


def fake_lease(tmp_path: Path, gpu: str = "5") -> str:
    """A stand-in for gpu-lease: records its arguments, sets CUDA_VISIBLE_DEVICES and runs CMD."""
    p = tmp_path / "gpu-lease"
    p.write_text(f'#!/usr/bin/env bash\necho "$*" > "{tmp_path}/lease.args"\nwhile [ "$1" != -- ]; do shift; done; shift\n'
                 f'export CUDA_VISIBLE_DEVICES={gpu}\nexec "$@"\n')
    p.chmod(0o755)
    return str(p)


def test_only_an_item_with_a_lease_tool_is_leased_and_its_devices_are_hidden_outside_the_lease():
    assert not GL.leased({}) and not GL.leased({"GPU_LEASE": ""}) and GL.leased({"GPU_LEASE": "/x/gpu-lease"})
    env = {"CUDA_VISIBLE_DEVICES": "0"}
    GL.hide_devices(env)
    assert env == {"CUDA_VISIBLE_DEVICES": ""}


def test_the_commit_runs_under_one_leased_gpu_for_at_most_its_timeout():
    argv = GL.wrap(["python", "-m", "commit"], 2400, "/row/gpu_lease.started", {"GPU_LEASE": "/l/gpu-lease"})
    assert argv[:9] == ["/l/gpu-lease", "1", "--wait", "--max-min", "40", "--mem-gb", "0", "--", "bash"]
    assert argv[-4:] == ["/row/gpu_lease.started", "python", "-m", "commit"]
    assert GL.max_min(61) == 2 and GL.max_min(1) == 1


def test_the_commit_sees_only_the_leased_gpu_and_the_lease_is_recorded(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "GPU_LEASE": fake_lease(tmp_path)}
    GL.hide_devices(env)
    seen = tmp_path / "seen"
    cmd = [sys.executable, "-c", f"import os; open({str(seen)!r}, 'w').write(os.environ['CUDA_VISIBLE_DEVICES'])"]
    log = tmp_path / "commit.log"
    timeouts = []

    def run_tl(to: int, argv: list[str]) -> int:
        timeouts.append(to)
        with open(log, "a") as f:
            return subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT).returncode

    rc, rec = GL.run(str(tmp_path), str(log), cmd, 2400, env, run_tl, clock=lambda: 0.0)
    assert rc == 0 and seen.read_text() == "5" and env["CUDA_VISIBLE_DEVICES"] == ""
    assert timeouts == [4800] and (tmp_path / "lease.args").read_text().startswith("1 --wait --max-min 40 --mem-gb 0 -- bash -c ")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", rec["started"]) and rec["held_min"] is not None
    assert json.loads((tmp_path / GL.RECORD).read_text()) == rec and rec["max_min"] == 40 and rec["rc"] == 0


def test_a_commit_that_never_got_its_gpu_records_no_start(tmp_path):
    rc, rec = GL.run(str(tmp_path), str(tmp_path / "commit.log"), ["true"], 60, {"GPU_LEASE": "/l/gpu-lease"},
                     lambda to, argv: 75, clock=lambda: 0.0)
    assert rc == 75 and rec["started"] is None and rec["held_min"] is None and rec["waited_min"] is None


class _Ticks(threading.Event):
    """An event whose wait returns `n` times without being set (one heartbeat each), then as if set."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def wait(self, timeout: float | None = None) -> bool:
        self.n -= 1
        return self.n < 0


def test_a_queued_commit_writes_a_heartbeat_to_its_log_until_the_gpu_is_held(tmp_path):
    log, started = tmp_path / "commit.log", tmp_path / "gpu_lease.started"
    GL._heartbeat(str(log), str(started), _Ticks(3), 60)
    assert log.read_text().count("waiting for a GPU lease") == 3
    started.write_text("2026-10-01T16:00:00Z\n")
    GL._heartbeat(str(log), str(started), _Ticks(3), 60)
    assert log.read_text().count("waiting for a GPU lease") == 3
