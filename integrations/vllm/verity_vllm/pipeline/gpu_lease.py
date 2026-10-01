"""A leased Commit: a dispatch item marked ``"lease": "self"`` runs in a pod with no GPU of its own and ``GPU_LEASE`` naming the
node's lease tool (``gpu-lease``), and only its GPU phase holds a GPU.

The row's processes see no CUDA device (``CUDA_VISIBLE_DEVICES`` empty), so nothing outside the lease can open a context on any GPU
of the node. The Commit process starts without a GPU too: its imports, the manifest, the producer facts and the boundary plan run on
the CPU, and only right before the engine (its target reads the device, then the build) does it take the lease (``acquire``):
``"$GPU_LEASE" N --wait --max-min M --mem-gb 0 -- holder``, a child of the Commit that hands it the lease's lock descriptors and the
leased ``CUDA_VISIBLE_DEVICES``. The GPUs stay
locked until the Commit process exits, the only point vLLM's in-process engine lets go of its CUDA context. The lease tool ending the
holder (M minutes, the Commit's timeout) ends the Commit. Prechecks, manifest and weights of record come before the Commit process;
the verdict, the row's records and the research run's publish after it. Without ``GPU_LEASE`` nothing changes.

``gpu_lease.json`` beside the row records the lease: requested and started (written by the Commit and the holder) and released (when
the Commit process exits), and the minutes waited and held."""
from __future__ import annotations

import atexit
import calendar
import ctypes
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, MutableMapping

RECORD = "gpu_lease.json"
HEARTBEAT_S = 60
STARTED = "VERITY_LEASE_STARTED"   # set by the row for its Commit: the marker the holder writes; the Commit takes its own lease
MAX_MIN = "VERITY_LEASE_MAX_MIN"
TOOL_START = "GPU_LEASE_STARTED"   # the lease tool's own start-time override (its tests): never passed to it

# the holder: start marker, the lease's lock descriptors to the Commit (socket argv[2]), the leased devices on stdout, then wait for
# the Commit to exit (EOF on stdin)
_HOLDER = r"""
import json, os, re, socket, sys, time
open(sys.argv[1], "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
fds = []
for f in os.listdir("/proc/self/fd"):
    try:
        if re.fullmatch(r"\d+\.lock", os.path.basename(os.readlink(f"/proc/self/fd/{f}"))):
            fds.append(int(f))
    except OSError:
        pass
socket.send_fds(socket.socket(fileno=int(sys.argv[2])), [b"L"], fds)
print(json.dumps({k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "GPU_LEASE_UUID") if k in os.environ}), flush=True)
sys.stdin.buffer.read()
"""
_HELD: list[Any] = []
_EXITING = threading.Event()
_ENVIRON: list[MutableMapping[str, str]] = []


def use_environ(environ: MutableMapping[str, str]) -> None:
    """The process environment ``acquire`` reads the lease from and sets the leased devices in (the CLI's, once)."""
    _ENVIRON[:] = [environ]


def leased(environ: MutableMapping[str, str]) -> bool:
    return bool(environ.get("GPU_LEASE"))


def hide_devices(env: MutableMapping[str, str]) -> None:
    """No CUDA device outside the lease: torch and every child see none, so none can open a context on an unleased GPU."""
    env["CUDA_VISIBLE_DEVICES"] = ""


def max_min(timeout_s: int) -> int:
    return max(1, math.ceil(int(timeout_s) / 60))


def _utc(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _requested(started: str) -> str:
    return started[: -len("started")] + "requested"


def holder_argv(environ: MutableMapping[str, str], sock: int) -> list[str]:
    """The lease request: the task's GPU count (``GPU_LEASE_N``, which dispatch sets; else one) for at most the Commit's timeout."""
    return [environ["GPU_LEASE"], environ.get("GPU_LEASE_N") or "1", "--wait", "--max-min", environ.get(MAX_MIN) or "60", "--mem-gb", "0",
            "--", sys.executable, "-c", _HOLDER, environ[STARTED], str(sock)]


def cuda_initialised() -> bool:
    """Whether this process has initialised the CUDA driver (``cuDeviceGetCount`` answers only after ``cuInit``): one initialised
    before the lease keeps the empty device list it saw."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False
    return cuda.cuDeviceGetCount(ctypes.byref(ctypes.c_int())) == 0


def tool_env(environ: MutableMapping[str, str]) -> dict[str, str]:
    """The lease tool's environment: the Commit's, less the tool's start-time override (a stale one dates the lease to 1970 and
    the pool ends it as expired)."""
    return {k: v for k, v in environ.items() if k != TOOL_START}


def _end_with(holder: subprocess.Popen) -> None:
    holder.wait()
    if not _EXITING.is_set():
        print(f"{_utc(time.time())} GPU lease ended under the Commit (holder rc {holder.returncode}): the Commit ends now", file=sys.stderr,
              flush=True)
        os.kill(os.getpid(), signal.SIGKILL)


def acquire(tl: Any = None, environ: MutableMapping[str, str] | None = None) -> None:
    """Take the leased Commit's GPUs now (``VERITY_LEASE_STARTED`` set in ``environ``, default the CLI's; else nothing): wait for the
    lease, then see only its devices. Nothing before this call may initialise CUDA (the process sees no device until it returns). The
    lease's locks are this process's from here to its exit; the holder ending first (the lease tool's --max-min) kills this process."""
    environ = environ if environ is not None else (_ENVIRON[0] if _ENVIRON else None)
    if environ is None or not environ.get(STARTED) or not leased(environ):
        return
    if cuda_initialised():
        raise SystemExit("GPU lease: CUDA was initialised before the lease (with no device visible); nothing may touch CUDA before "
                         "gpu_lease.acquire, so no lease is taken")
    with open(_requested(environ[STARTED]), "w") as f:
        f.write(_utc(time.time()) + "\n")
    ours, theirs = socket.socketpair()
    if tl is not None:
        tl.start("gpu_lease.wait")
    holder = subprocess.Popen(holder_argv(environ, theirs.fileno()), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              pass_fds=(theirs.fileno(),), env=tool_env(environ))
    theirs.close()
    line = holder.stdout.readline() if holder.stdout else b""
    if not line:
        raise SystemExit(f"GPU lease: {environ['GPU_LEASE']} ended (rc {holder.wait()}) without a GPU")
    _, fds, _, _ = socket.recv_fds(ours, 1, 64)
    want = int(environ.get("GPU_LEASE_N") or "1")
    if len(fds) < want:
        holder.kill()
        raise SystemExit(f"GPU lease: the holder passed {len(fds)} lock descriptor(s) for {want} GPU(s); the Commit would not hold its lease")
    environ.update(json.loads(line))
    _HELD.append((holder, ours, fds))
    atexit.register(_EXITING.set)        # registered after torch's and vLLM's, so it runs before their teardown
    threading.Thread(target=_end_with, args=(holder,), daemon=True).start()
    if tl is not None:
        tl.end("gpu_lease.wait", devices=environ.get("CUDA_VISIBLE_DEVICES"), locks=len(fds))


def _heartbeat(log: str, started: str, stop: threading.Event, every_s: float) -> None:
    """While the Commit waits for its lease, a line in its log now and then, so a stall watchdog on the log doesn't stop a queued
    Commit."""
    while not stop.wait(every_s):
        if os.path.exists(started):
            return
        if os.path.exists(_requested(started)):
            with open(log, "a") as f:
                f.write(f"{_utc(time.time())} waiting for a GPU lease\n")


def _stamp(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    return calendar.timegm(time.strptime(open(path).read().strip(), "%Y-%m-%dT%H:%M:%SZ"))


def run(row_dir: str, log: str, cmd: list[str], timeout_s: int, environ: dict[str, str], run_tl: Callable[[int, list[str]], int],
        *, clock: Callable[[], float] = time.time, every_s: float = HEARTBEAT_S) -> tuple[int, dict[str, Any]]:
    """Run the leased Commit ``cmd`` -> (its rc, the lease record). ``run_tl(timeout, argv)`` runs argv as the row runs a stage; it may
    wait for the GPU as long as the Commit may run, and the lease tool ends the Commit at its timeout."""
    started = os.path.join(row_dir, "gpu_lease.started")
    for p in (started, _requested(started)):
        if os.path.exists(p):
            os.remove(p)
    t0 = clock()
    stop = threading.Event()
    beat = threading.Thread(target=_heartbeat, args=(log, started, stop, every_s), daemon=True)
    beat.start()
    environ.update({STARTED: started, MAX_MIN: str(max_min(timeout_s))})
    try:
        rc = run_tl(2 * int(timeout_s), cmd)
    finally:
        environ.pop(STARTED, None)
        environ.pop(MAX_MIN, None)
        stop.set()
        beat.join()
    t1 = clock()
    rec: dict[str, Any] = {"schema": "verity-vllm/gpu-lease/v1", "tool": environ["GPU_LEASE"], "max_min": max_min(timeout_s),
                           "process_started": _utc(t0), "requested": None, "released": _utc(t1), "rc": rc, "started": None,
                           "waited_min": None, "held_min": None}
    tr, ts = _stamp(_requested(started)), _stamp(started)
    if tr is not None:
        rec["requested"] = _utc(tr)
    if ts is not None:
        rec.update(started=_utc(ts), waited_min=round(max(0.0, ts - (tr if tr is not None else t0)) / 60, 2),
                   held_min=round(max(0.0, t1 - ts) / 60, 2))
    with open(os.path.join(row_dir, RECORD), "w") as f:
        json.dump(rec, f, indent=1)
    return rc, rec
