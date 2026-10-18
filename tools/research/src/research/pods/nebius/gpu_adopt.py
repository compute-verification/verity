#!/usr/bin/env python3
"""A GPU VM's out-of-lease probe (INF-55): a process on a GPU that no lease gave it holds that GPU's lease until it exits.

    gpu_adopt.py loop [--every 30]   as the research user, unit vy-gpu-adopt (vm_setup.sh installs it on a VM with GPUs)
    gpu_adopt.py check               one pass, printed as JSON (it adopts nothing that outlives the pass)

gpu-lease hands out this VM's GPUs (one flock'd lock per GPU in $GPU_LEASE_DIR, default /run/gpu-lease). A compute process
(`nvidia-smi --query-compute-apps`) on GPU i is allowed when a lease on i has an owner pid that is the process or one of its
ancestors. Any other is out of lease: a job that opened CUDA without gpu-lease (a `research run --on` from a tree before the
runner hid the GPUs, or a shell). If i's lock is free, this probe takes it with gpu-lease's owner line naming the process's run (its
runner's `--id`) and `who=adopted:<lane>`, so gpu-lease hands i to nobody else, and drops it when the process exits. If a lease
holds i already, two jobs share the GPU (`collision`). Each finding is a line in $GPU_ADOPT_LOG (default
/workspace/research/gpu-adopt.jsonl), which the duty loop reads; adopting is announced once per process and GPU.
Only a process listed both before and after the leases are read is judged, as gpu_stray.py does on node 1. No process's environment
is read. On a cluster node ($GPU_ADOPT_SPLIT_FILE, default /etc/vy/direct-gpus, present) Kueue's pods hold GPUs without gpu-lease and
gpu_stray.py judges them, so this probe judges nothing there.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LEASE_DIR = Path(os.environ.get("GPU_LEASE_DIR", "/run/gpu-lease"))
LOG = Path(os.environ.get("GPU_ADOPT_LOG", "/workspace/research/gpu-adopt.jsonl"))
RUNS = Path(os.environ.get("GPU_ADOPT_RUNS", "/workspace/research/runs"))
SPLIT = Path(os.environ.get("GPU_ADOPT_SPLIT_FILE", "/etc/vy/direct-gpus"))
RUN_ID = re.compile(r"(?:^|\s)--id\s+(r\d{8}-\d{6}-[0-9a-f]{4})(?:\s|$)")


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_apps() -> list[tuple[int, str]]:
    """(pid, GPU UUID) of every compute process."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], capture_output=True,
                         text=True, check=True).stdout
    apps = []
    for line in out.splitlines():
        pid, _, uuid = (s.strip() for s in line.partition(","))
        if pid.isdigit():
            apps.append((int(pid), uuid))
    return apps


def gpu_index() -> dict[str, int]:
    """UUID -> nvidia-smi index (gpu-lease's lock name; it runs with CUDA_DEVICE_ORDER=PCI_BUS_ID)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True,
                         check=True).stdout
    return {u.strip(): int(i) for i, _, u in (ln.partition(",") for ln in out.splitlines()) if i.strip().isdigit()}


def parent(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def ancestors(pid: int) -> list[int]:
    """The process and its ancestors, nearest first."""
    out: list[int] = []
    while pid and pid > 1 and pid not in out:
        out.append(pid)
        pid = parent(pid) or 0
    return out


def cmdline(pid: int) -> str:
    try:
        return " ".join(Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")).strip()
    except OSError:
        return ""


def comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


def run_of(chain: list[int]) -> str | None:
    """The run id of the nearest `research run ... --id RUN` runner among the process's ancestors."""
    for p in chain:
        if "research" in (c := cmdline(p)) and (m := RUN_ID.search(c)):
            return m.group(1)
    return None


def lane_of(run: str | None) -> str | None:
    try:
        return (json.loads((RUNS / run / "job.json").read_text()).get("ownership") or {}).get("lane") if run else None
    except (OSError, ValueError):
        return None


def lease_owner(i: int) -> int | None:
    """The owner pid of a lease held on GPU i, else None (the lock is free)."""
    lock = LEASE_DIR / f"{i}.lock"
    try:
        fd = os.open(lock, os.O_RDONLY)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return None
    except BlockingIOError:
        try:
            m = re.search(r"\bpid=(\d+)", (LEASE_DIR / f"{i}.owner").read_text())
        except OSError:
            return -1
        return int(m.group(1)) if m else -1
    finally:
        os.close(fd)


class Adopter:
    def __init__(self) -> None:
        self.held: dict[tuple[int, int], int] = {}      # (pid, GPU index) -> the lock's fd this process holds for it
        self.told: set[tuple[int, int, str]] = set()

    def release_dead(self) -> None:
        for key, fd in list(self.held.items()):
            if not Path(f"/proc/{key[0]}").exists():
                os.close(fd)
                del self.held[key]

    def adopt(self, i: int, uuid: str, pid: int, run: str | None, lane: str | None, t: float) -> bool:
        lock = LEASE_DIR / f"{i}.lock"
        fd = os.open(lock, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        (LEASE_DIR / f"{i}.owner").write_text(
            f"pid={os.getpid()} run={run or '-'} who=adopted:{lane or 'unknown'} uuid={uuid} since={iso(t)} mem=0G "
            f"cmd=gpu_adopt.py: pid {pid} ({comm(pid)}) used GPU {i} without a lease; held until it exits\n")
        self.held[(pid, i)] = fd
        return True

    def check(self, t: float) -> list[dict]:
        self.release_dead()
        if SPLIT.exists():
            return []
        apps = compute_apps()
        if not apps:
            return []
        index = gpu_index()
        owners = {i: lease_owner(i) for i in set(index.values())}
        again = set(compute_apps())
        found = []
        for pid, uuid in apps:
            i = index.get(uuid)
            if i is None or (pid, uuid) not in again or (pid, i) in self.held:
                continue
            chain = ancestors(pid)
            owner = owners.get(i)
            if owner is not None and owner in chain:
                continue
            if not Path(f"/proc/{pid}").exists():
                continue
            run = run_of(chain)
            lane = lane_of(run)
            kind = "adopted" if owner is None and self.adopt(i, uuid, pid, run, lane, t) else "collision"
            rec = {"t": iso(t), "kind": kind, "gpu": i, "uuid": uuid, "pid": pid, "comm": comm(pid), "run": run, "lane": lane,
                   **({"lease_owner": owner} if kind == "collision" else {})}
            found.append(rec)
            if (pid, i, kind) not in self.told:
                self.told.add((pid, i, kind))
                with open(LOG, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        return found


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    lp = sub.add_parser("loop")
    lp.add_argument("--every", type=float, default=30)
    ns = ap.parse_args(argv)
    a = Adopter()
    if ns.cmd == "check":
        print(json.dumps(a.check(time.time())))
        return 0
    while True:
        try:
            for rec in a.check(time.time()):
                print(json.dumps(rec), flush=True)
        except (subprocess.CalledProcessError, OSError, ValueError) as e:
            print(f"{iso(time.time())} check failed: {e}", file=sys.stderr, flush=True)
        time.sleep(ns.every)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
