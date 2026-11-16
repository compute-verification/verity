#!/usr/bin/env python3
"""vy-nebius-1's stray-GPU probe: a process on a GPU that neither Kubernetes nor a lease gave it blocks its lease pool.

    gpu_stray.py loop [--every 60]   as research on node 1 (unit vy-gpu-stray); reads only the GPU variables of other users' /proc/PID/environ, with sudo -n
    gpu_stray.py check               one pass, printed as JSON

A leased pod sees every GPU (`NVIDIA_VISIBLE_DEVICES=all`) and keeps to its lease by honouring CUDA_VISIBLE_DEVICES, so this
probe checks that it does. A compute process (`nvidia-smi --query-compute-apps`) on GPU U, index i, is allowed when
  - its pod requests k GPUs and its container's /dev holds exactly k GPU device nodes, U's among them: the device plugin gave
    it U (its NVIDIA_VISIBLE_DEVICES is `void`; the devices come by CDI, so the variable names none); or
  - a lease (a held lock that isn't n1_lease.py's fence) on lock i in any pool dir of `$GPU_POOL_DIRS` has an owner pid in the
    process's pod (the cgroup's pod uid), or that is one of the process's ancestors.
Only a process listed both before and after the leases are read is judged. Any other is a stray. Each is a line in `$GPU_STRAY_LOG`, and when its pod mounts a pool dir (its `gpu-lease` hostPath volume)
that dir's `blocked` file is written, which stops the pool's new grants (n1_lease.py). The block stays until a person removes
the file, once the stray is understood; a stray still running then blocks the pool again on the next pass.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

POOL_DIRS = [Path(p) for p in os.environ.get("GPU_POOL_DIRS", "/run/gpu-lease,/run/gpu-lease-circuits").split(",") if p]
LOG = Path(os.environ.get("GPU_STRAY_LOG", "/workspace/verity-guest/gpu-stray.jsonl"))
KUBECTL = os.environ.get("VY_KUBECTL", "sudo -n k3s kubectl").split()
FENCE_CMD = "n1-lease-fence"


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_apps() -> list[tuple[int, str]]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], capture_output=True,
                         text=True, check=True).stdout
    apps = []
    for line in out.splitlines():
        pid, _, uuid = (s.strip() for s in line.partition(","))
        if pid.isdigit():
            apps.append((int(pid), uuid))
    return apps


def gpu_table() -> dict[str, tuple[int, int]]:
    out = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, check=True).stdout
    table, uuid, index = {}, None, -1
    for line in out.splitlines():
        if line.startswith("GPU "):
            index, uuid = index + 1, None
        k, _, v = (x.strip() for x in line.partition(":"))
        if k == "GPU UUID":
            uuid = v
        elif k == "Minor Number" and uuid:
            table[uuid] = (index, int(v))
    return table


def dev_minors(pid: int) -> set[int]:
    r = subprocess.run(["sudo", "-n", "ls", f"/proc/{pid}/root/dev"], capture_output=True, text=True)
    return {int(m.group(1)) for m in (re.fullmatch(r"nvidia(\d+)", n) for n in r.stdout.split()) if m}


GPU_VARS = ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def environ(pid: int) -> dict[str, str]:
    path = f"/proc/{pid}/environ"
    pattern = "^(" + "|".join(GPU_VARS) + ")="
    try:
        raw = Path(path).read_bytes()
    except PermissionError:
        raw = subprocess.run(["sudo", "-n", "grep", "-zaE", pattern, path], capture_output=True).stdout
    except OSError as e:
        if alive(pid):
            print(f"gpu_stray: pid {pid}'s environment unread, so its GPU variables are unknown ({type(e).__name__}: {e})",
                  file=sys.stderr, flush=True)
        return {}
    pairs = (kv.split("=", 1) for kv in raw.decode(errors="replace").split("\0") if "=" in kv)
    return {k: v for k, v in pairs if k in GPU_VARS}


def alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def cmdline(pid: int) -> str:
    try:
        return " ".join(Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0"))[:200]
    except OSError:
        return ""


def pod_uid_of(pid: int) -> str | None:
    try:
        m = re.search(r"pod([0-9a-f_-]{36})", Path(f"/proc/{pid}/cgroup").read_text())
    except OSError as e:
        if alive(pid):
            print(f"gpu_stray: pid {pid}'s cgroup unread, so its pod is unknown ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
        return None
    return m.group(1).replace("_", "-") if m else None


def ancestors(pid: int) -> set[int]:
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError) as e:
            if alive(pid):
                print(f"gpu_stray: pid {pid}'s parent unread, so its ancestors stop there ({type(e).__name__}: {e})", file=sys.stderr,
                      flush=True)
            break
    return seen


def leases() -> dict[int, list[tuple[Path, int]]]:
    locked = set()
    for line in Path("/proc/locks").read_text().splitlines():
        f = line.split()
        if len(f) >= 6 and f[1] == "FLOCK":
            locked.add(int(f[5].split(":")[-1]))
    out: dict[int, list[tuple[Path, int]]] = {}
    for d in POOL_DIRS:
        for lock in d.glob("*.lock"):
            if not lock.stem.isdigit() or lock.stat().st_ino not in locked:
                continue
            try:
                owner = (d / f"{lock.stem}.owner").read_text()
            except OSError as e:
                print(f"gpu_stray: {d / f'{lock.stem}.owner'} unread, so its held lock is skipped ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
                continue
            m = re.search(r"\bpid=(\d+)", owner)
            if m and FENCE_CMD not in owner:
                out.setdefault(int(lock.stem), []).append((d, int(m.group(1))))
    return out


def requested_gpus(pod: dict) -> int:
    return sum(int(((c.get("resources") or {}).get("limits") or {}).get("nvidia.com/gpu", 0) or 0)
               for c in pod["spec"].get("containers") or [])


def pods_by_uid() -> dict[str, dict]:
    out = subprocess.run([*KUBECTL, "get", "pods", "-A", "-o", "json"], capture_output=True, text=True, check=True).stdout
    return {p["metadata"]["uid"]: p for p in json.loads(out)["items"]}


def check(t: float) -> list[dict]:
    apps = compute_apps()
    if not apps:
        return []
    table = gpu_table()
    held = leases()
    again = set(compute_apps())
    apps = [a for a in apps if a in again]
    pods = pods_by_uid()
    strays = []
    for pid, uuid in apps:
        i, minor = table.get(uuid, (None, None))
        uid, up = pod_uid_of(pid), ancestors(pid)
        k = requested_gpus(pods[uid]) if uid in pods else 0
        if k:
            devs = dev_minors(pid)
            if len(devs) == k and minor in devs:
                continue
        if any((uid and pod_uid_of(owner) == uid) or owner in up for _, owner in held.get(i, [])):
            continue
        if not alive(pid):
            continue
        strays.append({"t": iso(t), "pid": pid, "gpu": i, "uuid": uuid, "pod_uid": uid, "requested": k,
                       "visible": environ(pid).get("NVIDIA_VISIBLE_DEVICES", "")[:80],
                       "cmd": cmdline(pid)})
    if strays:
        for s in strays:
            pod = pods.get(s["pod_uid"] or "")
            if not pod:
                continue
            s["pod"] = f"{pod['metadata']['namespace']}/{pod['metadata']['name']}"
            s["workstream"] = (pod["metadata"].get("labels") or {}).get("verity.dev/workstream")
            pool = next((v["hostPath"]["path"] for v in pod["spec"].get("volumes") or []
                         if v.get("name") == "gpu-lease" and v.get("hostPath")), None)
            if pool and Path(pool) in POOL_DIRS:
                s["blocked"] = pool
                blocked = Path(pool) / "blocked"
                if not blocked.exists():
                    blocked.write_text(f"{s['t']} pid {s['pid']} ({s['pod']}) ran on GPU {s['gpu']} without a lease; "
                                       f"remove this file once that's cleared (gpu_stray.py)\n")
                    os.chmod(blocked, 0o666)
    for s in strays:
        with open(LOG, "a") as f:
            f.write(json.dumps(s) + "\n")
    return strays


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    lp = sub.add_parser("loop")
    lp.add_argument("--every", type=float, default=60)
    ns = ap.parse_args(argv)
    if ns.cmd == "check":
        print(json.dumps(check(time.time())))
        return 0
    while True:
        try:
            for s in check(time.time()):
                print(json.dumps(s), flush=True)
        except (subprocess.CalledProcessError, OSError, ValueError) as e:
            print(f"{iso(time.time())} check failed: {e}", file=sys.stderr, flush=True)
        time.sleep(ns.every)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
