"""`lease.py`: `node.sh` starts its engines, one a GPU CUDA_VISIBLE_DEVICES names, only when those are exactly the GPUs
its gpu-lease holds. Gate run 7's lease of all 8 GPUs passes. Gate run 7's first attempt (8 engines asked under a lease
of GPUs 1-7, GPU 0 held by another run) is refused before any engine starts, as are a count off GPU_LEASE_UUID's and a
GPU named twice."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import lease  # noqa: E402

ME, OTHER = 4242, 777
ALL = ",".join(map(str, range(8)))
UUIDS = ",".join(f"GPU-{i:04x}" for i in range(8))


def line(i: int, pid: int | str) -> str:
    return (f"{i} held 0m pid={pid} run=r20261023-123709-339d who=research uuid=GPU-{i:04x} "
            f"since=2026-10-23T12:37:43Z until=2026-10-23T14:07:43Z mem=256G preempt=1 scope=gpu-lease-{pid}")


def status(holders: dict[int, int]) -> str:
    """`gpu-lease status` with GPU i held by pid holders[i], the rest free, and another run waiting."""
    return "\n".join([*(line(i, holders[i]) if i in holders else f"{i} free" for i in range(8)),
                      "waiting 3m pid=9 run=r-next who=proofs since=2026-10-23T12:40:00Z"]) + "\n"


FIRST_ATTEMPT = status({0: OTHER, **{i: ME for i in range(1, 8)}})
WHOLE_NODE = status({i: ME for i in range(8)})


@pytest.mark.parametrize("engines, env, st, why", [
    (8, {"CUDA_VISIBLE_DEVICES": ALL, "GPU_LEASE_UUID": UUIDS}, WHOLE_NODE, None),
    (8, {"CUDA_VISIBLE_DEVICES": UUIDS}, WHOLE_NODE, None),
    (8, {"CUDA_VISIBLE_DEVICES": ALL}, None, None),
    (8, {"CUDA_VISIBLE_DEVICES": ALL}, FIRST_ATTEMPT,
     "gpu-lease status shows this lease holding GPUs 1,2,3,4,5,6,7, not the 8 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' "
     "names: a GPU another process held never joined it"),
    (8, {"CUDA_VISIBLE_DEVICES": ALL, "GPU_LEASE_UUID": UUIDS.split(",", 1)[1]}, None,
     "the lease holds 7 GPUs (GPU_LEASE_UUID), not the 8 CUDA_VISIBLE_DEVICES names"),
    (8, {"CUDA_VISIBLE_DEVICES": "1,2,3,4,5,6,7"}, FIRST_ATTEMPT,
     "8 engines, but CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7' names 7 GPUs"),
    (8, {"CUDA_VISIBLE_DEVICES": "1,1,2,3,4,5,6,7"}, None, "CUDA_VISIBLE_DEVICES='1,1,2,3,4,5,6,7' names a GPU twice"),
    (8, {"CUDA_VISIBLE_DEVICES": ALL}, status({}),
     "gpu-lease status shows this lease holding GPUs none, not the 8 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' names: a "
     "GPU another process held never joined it"),
], ids=["whole-node", "by-uuid", "no-gpu-lease", "gpu0-never-joined", "uuid-count", "fewer-visible", "twice",
        "held-by-none"])
def test_the_engines_start_only_on_exactly_the_gpus_the_lease_holds(engines, env, st, why):
    assert lease.refusal(engines, env, st, {ME, 1}) == why


def test_node_sh_refuses_before_any_engine_starts_when_a_gpu_never_joined_the_lease(tmp_path):
    """node.sh with gpu-lease status as on gate run 7's first attempt (GPU 0 held by another run, 1-7 by this one)
    and CUDA_VISIBLE_DEVICES naming all 8: it exits 3 with the reason before nvidia-smi, a build or an engine runs.
    The same lease.py passes once the lease holds all 8."""
    bin_, o, b = tmp_path / "bin", tmp_path / "out", tmp_path / "b"
    for d in (bin_, o, b):
        d.mkdir()

    def stub(holders: dict[int, int | str]) -> None:
        """gpu-lease whose `status` shows GPU i held by holders[i]; $PPID is lease.py, its caller."""
        lines = "".join(f'echo "{line(i, p)}"\n' for i, p in holders.items())
        (bin_ / "gpu-lease").write_text(f'#!/bin/sh\n[ "$1" = status ] || exit 2\n{lines}')
        (bin_ / "gpu-lease").chmod(0o755)

    env = {**os.environ, "PATH": f"{bin_}{os.pathsep}{os.environ['PATH']}", "CUDA_VISIBLE_DEVICES": ALL,
           "O": str(o), "B": str(b), "RS": str(HERE), "PY": sys.executable, "VPY": "/nonexistent", "PORT": "7951",
           "PLAN": "honest"}
    env.pop("GPU_LEASE_UUID", None)
    stub({0: OTHER, **{i: "$PPID" for i in range(1, 8)}})
    p = subprocess.run(["bash", str(HERE / "node.sh")], env=env, capture_output=True, text=True, cwd=tmp_path)
    assert p.returncode == 3, p.stderr[-2000:]
    assert p.stdout == ("lease: refused before any engine started: gpu-lease status shows this lease holding GPUs "
                        "1,2,3,4,5,6,7, not the 8 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' names: a GPU another process "
                        "held never joined it\n")
    assert sorted(o.iterdir()) == [] and sorted(b.iterdir()) == []
    stub({i: "$PPID" for i in range(8)})
    p = subprocess.run([sys.executable, str(HERE / "lease.py"), "8"], env=env, capture_output=True, text=True)
    assert (p.returncode, p.stdout) == (0, "lease: 8 engines, one on each GPU the lease holds (0,1,2,3,4,5,6,7)\n")
