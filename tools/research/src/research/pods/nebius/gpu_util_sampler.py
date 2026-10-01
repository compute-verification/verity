#!/usr/bin/env python3
"""Per-GPU utilisation sampler for a shared GPU server (stdlib only).

Every INTERVAL seconds it appends one JSON line to OUT_DIR/<UTC date>.jsonl:
  t, cpu_busy (fraction of all cores since the last sample), load1, mem_used_gib,
  gpus: [{i, uuid, util, mem_util, mem_mib, power_w, sm_mhz, mem_mhz, temp_c,
          smact, tensor (DCGM, fraction), lease (gpu-lease owner or null), procs: [{pid, mib, cmd}]}],
  lease_waiters: [argv of gpu-lease processes still waiting for GPUs]

nvidia-smi's utilization.gpu is "a kernel was resident"; DCGM's SM activity (1002) and tensor activity (1004) say how much
of the chip that kernel used, so both are kept.  Nothing here needs a GPU lease: it only reads counters.

Timed windows are left alone: while one gpu-lease holder has every GPU, the sample makes no NVML or DCGM query (the DCGM
watch is stopped) and records only the leases, with "timed": true.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

INTERVAL = float(os.environ.get("UTIL_INTERVAL_S", "10"))
OUT_DIR = Path(os.environ.get("UTIL_OUT_DIR", "/workspace/pouw/infra/util"))
LEASE_DIR = Path(os.environ.get("GPU_LEASE_DIR", "/run/gpu-lease"))

QUERY = "index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu"


def num(s: str):
    s = s.strip()
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return None


class Dcgm(threading.Thread):
    """Streams `dcgmi dmon` and keeps the latest SM and tensor activity per GPU index."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.latest: dict[int, tuple[float | None, float | None]] = {}
        self.running = threading.Event()
        self.running.set()
        self.proc: subprocess.Popen | None = None

    def pause(self) -> None:
        self.running.clear()
        self.latest = {}
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def resume(self) -> None:
        self.running.set()

    def run(self) -> None:
        while True:
            self.running.wait()
            try:
                self.proc = subprocess.Popen(["dcgmi", "dmon", "-e", "1002,1004", "-d", str(int(INTERVAL * 1000))],
                                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                for line in self.proc.stdout:
                    f = line.split()
                    if len(f) >= 4 and f[0] == "GPU" and f[1].isdigit() and self.running.is_set():
                        self.latest[int(f[1])] = (num(f[2]), num(f[3]))
            except Exception as e:  # noqa: BLE001
                print(f"dcgmi: {e}", file=sys.stderr)
            time.sleep(5)


def cpu_times() -> tuple[int, int]:
    f = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    idle = f[3] + f[4]
    return sum(f), idle


def lease_owner(i: int) -> str | None:
    lock, owner = LEASE_DIR / f"{i}.lock", LEASE_DIR / f"{i}.owner"
    try:
        r = subprocess.run(["flock", "-n", "-s", str(lock), "true"], capture_output=True, timeout=5)
    except Exception:  # noqa: BLE001
        return None
    if r.returncode == 0:
        return None
    try:
        return owner.read_text().strip() or "held"
    except OSError:
        return "held"


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()[:200]
    except OSError:
        return "?"


def lease_waiters() -> list[str]:
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        s = [a.decode(errors="replace") for a in argv if a]
        if len(s) >= 3 and s[1].endswith("gpu-lease") and "--wait" in s[:6]:
            out.append(" ".join(s[1:8]))
    return out


def wait_files() -> list[str]:
    out = []
    for f in sorted(LEASE_DIR.glob("wait.*")):
        try:
            if subprocess.run(["flock", "-n", "-s", str(f), "true"], capture_output=True, timeout=5).returncode != 0:
                out.append(f.read_text().strip())
        except Exception:  # noqa: BLE001
            pass
    return out


def base(prev_cpu: tuple[int, int]) -> tuple[dict, tuple[int, int]]:
    cur = cpu_times()
    dt, di = cur[0] - prev_cpu[0], cur[1] - prev_cpu[1]
    mem = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()[:5]}
    return {"t": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cpu_busy": round(1 - di / dt, 4) if dt > 0 else None,
            "load1": float(Path("/proc/loadavg").read_text().split()[0]),
            "mem_used_gib": round((mem["MemTotal"] - mem["MemAvailable"]) / 2**20, 1),
            "lease_waiters": lease_waiters(), "wait_files": wait_files()}, cur


def timed_window(leases: list[str | None]) -> bool:
    if not leases or not all(leases):
        return False
    pids = {l.split()[0] for l in leases}
    return len(pids) == 1 and next(iter(pids)).startswith("pid=")


def sample(dcgm: Dcgm, prev_cpu: tuple[int, int], n_gpus: int) -> tuple[dict, tuple[int, int]]:
    leases = [lease_owner(i) for i in range(n_gpus)]
    if timed_window(leases):
        dcgm.pause()
        rec, cur = base(prev_cpu)
        rec["timed"] = True
        rec["gpus"] = [{"i": i, "lease": leases[i]} for i in range(n_gpus)]
        return rec, cur
    dcgm.resume()
    r = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=30).stdout
    procs: dict[str, list] = {}
    for line in apps.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 3 and f[1].isdigit():
            procs.setdefault(f[0], []).append({"pid": int(f[1]), "mib": num(f[2]), "cmd": cmdline(int(f[1]))})
    gpus = []
    for line in r.stdout.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 9:
            continue
        i = int(f[0])
        smact, tensor = dcgm.latest.get(i, (None, None))
        gpus.append({"i": i, "uuid": f[1], "util": num(f[2]), "mem_util": num(f[3]), "mem_mib": num(f[4]),
                     "power_w": num(f[5]), "sm_mhz": num(f[6]), "mem_mhz": num(f[7]), "temp_c": num(f[8]),
                     "smact": smact, "tensor": tensor, "lease": leases[i] if i < len(leases) else lease_owner(i),
                     "procs": procs.get(f[1], [])})
    rec, cur = base(prev_cpu)
    rec["gpus"] = gpus
    return rec, cur


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_gpus = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
                  if l.startswith("GPU ")])
    dcgm = Dcgm()
    dcgm.start()
    prev = cpu_times()
    time.sleep(min(INTERVAL, 2))
    while True:
        t0 = time.monotonic()
        try:
            rec, prev = sample(dcgm, prev, n_gpus)
            with open(OUT_DIR / f"{rec['t'][:10]}.jsonl", "a") as fh:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"sample failed: {e}", file=sys.stderr, flush=True)
        time.sleep(max(0.5, INTERVAL - (time.monotonic() - t0)))


if __name__ == "__main__":
    main()
