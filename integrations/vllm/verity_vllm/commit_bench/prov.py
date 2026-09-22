"""Provenance + timers for the CPERF measurements (host CPU, GPU, driver, venv, thread count, cgroup quota)."""

from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}>"


def provenance() -> dict[str, Any]:
    lscpu = _run(["lscpu"])
    cpu = next((l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith("Model name")), platform.processor())
    flags = next((l.split(":", 1)[1] for l in lscpu.splitlines() if l.startswith("Flags")), "")
    out: dict[str, Any] = {
        "host": platform.node(), "cpu": cpu, "sha_ni": " sha_ni " in f" {flags} ", "avx2": " avx2 " in f" {flags} ", "nproc": os.cpu_count(),
        "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip() if Path("/sys/fs/cgroup/cpu.max").exists() else None,
        "gpu": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]),
        "python": sys.version.split()[0], "openssl": None, "blake3": None, "torch": None, "threads": threading.active_count(),
        "load_avg_1m": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
        "snapshot_commit": None, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import ssl
        out["openssl"] = ssl.OPENSSL_VERSION
    except Exception:  # noqa: BLE001
        pass
    try:
        import blake3
        out["blake3"] = blake3.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        exp = json.loads(Path("EXPORT.json").read_text())
        out["snapshot_commit"] = exp.get("commit")
        out["snapshot_dirty_bytes"] = exp.get("dirty_patch_bytes")
    except Exception:  # noqa: BLE001
        pass
    return out


class Timer:
    """wall, process CPU time and peak-RSS delta around a block; ``reps`` medians via :func:`timed`."""

    def __enter__(self) -> "Timer":
        self.rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.cpu0 = time.process_time()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.wall = time.perf_counter() - self.t0
        self.cpu = time.process_time() - self.cpu0
        self.rss_delta_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - self.rss0


def timed(fn, reps: int = 3, **extra: Any) -> dict[str, Any]:
    """Run ``fn`` ``reps`` times; report first / median wall, median cpu, and whatever ``fn`` returns last."""
    walls, cpus, rss = [], [], []
    ret = None
    for _ in range(reps):
        with Timer() as t:
            ret = fn()
        walls.append(t.wall)
        cpus.append(t.cpu)
        rss.append(t.rss_delta_kb)
    d = {"first_s": walls[0], "median_s": statistics.median(walls[1:]) if reps > 1 else walls[0], "all_s": walls,
         "cpu_median_s": statistics.median(cpus[1:]) if reps > 1 else cpus[0], "peak_rss_delta_kb_max": max(rss)}
    d.update(extra)
    if isinstance(ret, dict):
        d.update(ret)
    return d


class GpuSampler:
    def __init__(self, interval_s: float = 0.1) -> None:
        self.samples: list[dict] = []
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            out = _run(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,power.draw", "--format=csv,noheader,nounits"]).split(",")
            try:
                self.samples.append({"util": float(out[0]), "mem_util": float(out[1]), "power_w": float(out[2])})
            except (ValueError, IndexError):
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join(timeout=5)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"samples": 0}
        return {"samples": len(self.samples), "util_mean": statistics.mean(s["util"] for s in self.samples), "util_max": max(s["util"] for s in self.samples),
                "mem_util_mean": statistics.mean(s["mem_util"] for s in self.samples), "power_mean_w": statistics.mean(s["power_w"] for s in self.samples)}
