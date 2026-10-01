"""A leased Commit: a dispatch item marked ``"lease": "self"`` runs in a pod with no GPU of its own and ``GPU_LEASE`` naming the
node's lease tool (``gpu-lease``), and only its GPU phase holds a GPU.

Before the lease and after its release the row's processes see no CUDA device (``CUDA_VISIBLE_DEVICES`` empty), so nothing outside
the lease can open a context on any GPU of the node. The GPU phase (engine build, run, Commit and its checks: the Commit process)
runs as ``"$GPU_LEASE" 1 --wait --max-min M --mem-gb 0 -- CMD``, which sets ``CUDA_VISIBLE_DEVICES`` to the leased GPU and ends
CMD after M minutes, M the Commit's timeout. Prechecks, manifest and weights of record come before it; the verdict, the row's
records and the research run's publish after. Without ``GPU_LEASE`` nothing changes.

``gpu_lease.json`` beside the row records the lease: requested, started (written inside the lease, before CMD) and released (when
CMD's lease tool exits), and the minutes waited and held."""
from __future__ import annotations

import calendar
import json
import math
import os
import threading
import time
from typing import Any, Callable, MutableMapping

RECORD = "gpu_lease.json"
HEARTBEAT_S = 60


def leased(environ: dict[str, str]) -> bool:
    return bool(environ.get("GPU_LEASE"))


def hide_devices(env: MutableMapping[str, str]) -> None:
    """No CUDA device outside the lease: torch and every child see none, so none can open a context on an unleased GPU."""
    env["CUDA_VISIBLE_DEVICES"] = ""


def max_min(timeout_s: int) -> int:
    return max(1, math.ceil(int(timeout_s) / 60))


def wrap(cmd: list[str], timeout_s: int, started: str, environ: dict[str, str]) -> list[str]:
    """``cmd`` under one leased GPU for at most the Commit's timeout; ``started`` is written (UTC) once the GPU is held."""
    return [environ["GPU_LEASE"], "1", "--wait", "--max-min", str(max_min(timeout_s)), "--mem-gb", "0", "--",
            "bash", "-c", 'date -u +%Y-%m-%dT%H:%M:%SZ > "$0"; exec "$@"', started, *cmd]


def _utc(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _heartbeat(log: str, started: str, stop: threading.Event, every_s: float) -> None:
    """While the lease is waited for, a line in the Commit's log now and then, so a stall watchdog on the log doesn't stop a queued
    Commit."""
    while not stop.wait(every_s):
        if os.path.exists(started):
            return
        with open(log, "a") as f:
            f.write(f"{_utc(time.time())} waiting for a GPU lease\n")


def run(row_dir: str, log: str, cmd: list[str], timeout_s: int, environ: dict[str, str], run_tl: Callable[[int, list[str]], int],
        *, clock: Callable[[], float] = time.time, every_s: float = HEARTBEAT_S) -> tuple[int, dict[str, Any]]:
    """Run the GPU phase leased -> (its rc, the lease record). ``run_tl(timeout, argv)`` runs argv as the row runs a stage; it may
    wait for the GPU as long as the Commit may run, and the lease tool ends the Commit at its timeout."""
    started = os.path.join(row_dir, "gpu_lease.started")
    if os.path.exists(started):
        os.remove(started)
    t0 = clock()
    stop = threading.Event()
    beat = threading.Thread(target=_heartbeat, args=(log, started, stop, every_s), daemon=True)
    beat.start()
    try:
        rc = run_tl(2 * int(timeout_s), wrap(cmd, timeout_s, started, environ))
    finally:
        stop.set()
        beat.join()
    t1 = clock()
    rec: dict[str, Any] = {"schema": "verity-vllm/gpu-lease/v1", "tool": environ["GPU_LEASE"], "max_min": max_min(timeout_s),
                           "requested": _utc(t0), "released": _utc(t1), "rc": rc, "started": None, "waited_min": None, "held_min": None}
    if os.path.exists(started):
        s = open(started).read().strip()
        ts = calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))
        rec.update(started=s, waited_min=round(max(0.0, ts - t0) / 60, 2), held_min=round(max(0.0, t1 - ts) / 60, 2))
    with open(os.path.join(row_dir, RECORD), "w") as f:
        json.dump(rec, f, indent=1)
    return rc, rec
