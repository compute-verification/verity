#!/usr/bin/env python3
"""Run a timed command on the leased GPU's NUMA node: `numabind.py RECORD CMD [ARG ...]`, inside the lease.

The GPU is the lease's first (GPU_LEASE_UUID's, else CUDA_VISIBLE_DEVICES's), its node its PCI address's
/sys/bus/pci/devices/<bdf>/numa_node, and CMD runs under `numactl --physcpubind=CPUS --membind=NODE`, CPUS the node's CPUs
among this process's own: a bound run never takes CPUs its run wasn't given. A served step's host side (the forward's launches,
the engine's scheduling) is timed with its GPU work, so a row depends on where its threads and memory sit, and every live row
is taken bound unless DITHER_NUMA=0.

CMD always runs: with DITHER_NUMA=0, or no numactl, GPU, node or shared CPU, or a bind numactl refuses, it runs unbound.
Either way RECORD (JSON) and one `numa:` line on stdout say how it ran and why, and CMD's environment carries the record as
DITHER_NUMA_RECORD (sbench.py's summary keeps it as "numa").
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def cpus_of(text: str) -> set[int]:
    """A kernel CPU list (`0-95,192`) as a set."""
    out: set[int] = set()
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def cpu_list(cpus: set[int]) -> str:
    """A set of CPUs as a kernel CPU list."""
    runs: list[list[int]] = []
    for c in sorted(cpus):
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def sysfs_bdf(bus_id: str) -> str:
    """nvidia-smi's `00000000:E5:00.0` as sysfs names the device, `0000:e5:00.0`."""
    domain, rest = bus_id.strip().split(":", 1)
    return f"{int(domain, 16):04x}:{rest.lower()}"


def smi_bus_id(gpu: str) -> str:
    return subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=pci.bus_id", "--format=csv,noheader"], capture_output=True,
                          text=True, check=True, timeout=60).stdout.split()[0]


def probe(prefix: list[str]) -> str | None:
    """None when numactl accepts the bind, else its complaint."""
    r = subprocess.run([*prefix, "true"], capture_output=True, text=True)
    return None if r.returncode == 0 else (r.stderr.strip() or f"rc {r.returncode}")


def plan(env, affinity: set[int], *, bus_id=smi_bus_id, sys_root: Path = Path("/sys"), numactl: str | None = None,
         check=probe) -> tuple[list[str], dict]:
    """The prefix CMD runs under (empty: unbound) and the record of how and why."""
    rec: dict = {"bound": False, "node": None, "affinity": cpu_list(affinity)}
    if env.get("DITHER_NUMA", "1") == "0":
        return [], {**rec, "reason": "DITHER_NUMA=0"}
    gpu = (env.get("GPU_LEASE_UUID") or env.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    if not gpu:
        return [], {**rec, "reason": "no leased GPU (GPU_LEASE_UUID, CUDA_VISIBLE_DEVICES)"}
    rec["gpu"] = gpu
    try:
        rec["pci"] = bdf = sysfs_bdf(bus_id(gpu))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as e:
        return [], {**rec, "reason": f"no PCI address for GPU {gpu}: {e}"}
    try:
        node = int((sys_root / "bus/pci/devices" / bdf / "numa_node").read_text())
    except (OSError, ValueError) as e:
        return [], {**rec, "reason": f"no numa_node for {bdf}: {e}"}
    if node < 0:
        return [], {**rec, "reason": f"{bdf} has no NUMA node (numa_node {node})"}
    rec["node"] = node
    try:
        node_cpus = cpus_of((sys_root / f"devices/system/node/node{node}/cpulist").read_text())
    except (OSError, ValueError) as e:
        return [], {**rec, "reason": f"no CPU list for node {node}: {e}"}
    rec["node_cpus"] = cpu_list(node_cpus)
    if not (cpus := node_cpus & affinity):
        return [], {**rec, "reason": f"none of node {node}'s CPUs is among this run's"}
    if numactl is None:
        return [], {**rec, "reason": "no numactl on PATH"}
    prefix = [numactl, f"--physcpubind={cpu_list(cpus)}", f"--membind={node}"]
    if (err := check(prefix)) is not None:
        return [], {**rec, "reason": f"numactl refused {' '.join(prefix[1:])}: {err}"}
    return prefix, {**rec, "bound": True, "cpus": cpu_list(cpus), "membind": node, "numactl": " ".join(prefix)}


def line(rec: dict) -> str:
    gpu = f"GPU {rec['gpu']}" + (f" at {rec['pci']}" if "pci" in rec else "") if "gpu" in rec else ""
    if rec["bound"]:
        return (f"numa: bound to node {rec['node']} ({gpu}): CPUs {rec['cpus']} of this run's {rec['affinity']}, "
                f"memory node {rec['membind']}")
    on = f", which is on node {rec['node']}" if rec["node"] is not None else ""
    return f"numa: unbound ({rec['reason']}{'; ' + gpu if gpu else ''}{on})"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: numabind.py RECORD CMD [ARG ...]", file=sys.stderr)
        return 2
    record, cmd = Path(argv[0]), argv[1:]
    prefix, rec = plan(os.environ, os.sched_getaffinity(0), numactl=shutil.which("numactl"))
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(rec, indent=1) + "\n")
    print(line(rec), flush=True)
    try:
        os.execvpe((prefix or cmd)[0], prefix + cmd, {**os.environ, "DITHER_NUMA_RECORD": json.dumps(rec)})
    except OSError as e:
        print(f"numabind: {cmd[0]}: {e}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
