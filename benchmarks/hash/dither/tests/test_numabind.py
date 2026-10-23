"""numabind.py: the node a leased GPU's run binds to, within the run's own CPUs, and each way it runs unbound instead."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DITHER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DITHER))
import numabind  # noqa: E402

LEASE = {"GPU_LEASE_UUID": "GPU-4,GPU-5", "CUDA_VISIBLE_DEVICES": "4,5"}
RUN_CPUS = set(range(120))


def fake_sys(root: Path, numa_node: str = "1") -> Path:
    dev = root / "bus/pci/devices/0000:ef:00.0"
    dev.mkdir(parents=True)
    (dev / "numa_node").write_text(numa_node + "\n")
    for node, cpus in ((0, "0-95"), (1, "96-191")):
        d = root / f"devices/system/node/node{node}"
        d.mkdir(parents=True)
        (d / "cpulist").write_text(cpus + "\n")
    return root


def plan(root: Path, env=LEASE, affinity=RUN_CPUS, numactl="/usr/bin/numactl", check=lambda prefix: None, bus_id=None,
         numa_node="1"):
    asked = []

    def smi(gpu):
        asked.append(gpu)
        return bus_id(gpu) if bus_id else "00000000:EF:00.0"
    prefix, rec = numabind.plan(env, affinity, bus_id=smi, sys_root=fake_sys(root, numa_node), numactl=numactl, check=check)
    return prefix, rec, asked


def test_a_run_binds_to_its_gpus_node_on_the_cpus_it_was_given_there(tmp_path):
    prefix, rec, asked = plan(tmp_path)
    assert asked == ["GPU-4"]
    assert prefix == ["/usr/bin/numactl", "--physcpubind=96-119", "--membind=1"]
    assert rec == {"bound": True, "node": 1, "affinity": "0-119", "gpu": "GPU-4", "pci": "0000:ef:00.0", "node_cpus": "96-191",
                   "cpus": "96-119", "membind": 1, "numactl": "/usr/bin/numactl --physcpubind=96-119 --membind=1"}
    assert numabind.line(rec) == ("numa: bound to node 1 (GPU GPU-4 at 0000:ef:00.0): CPUs 96-119 of this run's 0-119, "
                                  "memory node 1")


def test_without_a_lease_uuid_the_gpu_is_cuda_visible_devices_first(tmp_path):
    assert plan(tmp_path, env={"CUDA_VISIBLE_DEVICES": "4,5"})[2] == ["4"]


def gone(gpu):
    raise FileNotFoundError("nvidia-smi")


@pytest.mark.parametrize("case, reason, node", [
    ({"env": {**LEASE, "DITHER_NUMA": "0"}}, "DITHER_NUMA=0", None),
    ({"env": {}}, "no leased GPU", None),
    ({"bus_id": gone}, "no PCI address for GPU GPU-4", None),
    ({"numa_node": "-1"}, "0000:ef:00.0 has no NUMA node", None),
    ({"affinity": set(range(96))}, "none of node 1's CPUs is among this run's", 1),
    ({"numactl": None}, "no numactl on PATH", 1),
    ({"check": lambda prefix: "set_mempolicy: Invalid argument"}, "numactl refused --physcpubind=96-119 --membind=1", 1),
])
def test_a_run_that_cannot_bind_runs_unbound_and_says_why(tmp_path, case, reason, node):
    prefix, rec, _ = plan(tmp_path, **case)
    assert prefix == [] and rec["bound"] is False and rec["node"] == node
    assert rec["reason"].startswith(reason)
    assert numabind.line(rec).startswith(f"numa: unbound ({reason}")


def test_an_unbound_line_names_the_node_it_would_have_bound_to(tmp_path):
    assert numabind.line(plan(tmp_path, numactl=None)[1]) == ("numa: unbound (no numactl on PATH; GPU GPU-4 at 0000:ef:00.0, "
                                                              "which is on node 1)")


def test_cpu_lists_and_bus_ids_read_as_the_kernel_writes_them():
    assert numabind.cpus_of("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}
    assert numabind.cpu_list({11, 0, 1, 2, 3, 8, 10}) == "0-3,8,10-11"
    assert numabind.sysfs_bdf("00000000:E5:00.0") == "0000:e5:00.0"


def test_the_command_runs_with_the_record_in_its_environment_and_its_exit_code(tmp_path):
    rec = tmp_path / "out/numa.json"
    show = "import json, os; print(json.loads(os.environ['DITHER_NUMA_RECORD'])['reason']); raise SystemExit(7)"
    r = subprocess.run([sys.executable, str(DITHER / "numabind.py"), str(rec), sys.executable, "-c", show],
                       env={**os.environ, "DITHER_NUMA": "0"}, capture_output=True, text=True)
    assert r.returncode == 7
    assert r.stdout.splitlines() == ["numa: unbound (DITHER_NUMA=0)", "DITHER_NUMA=0"]
    assert json.loads(rec.read_text())["bound"] is False
