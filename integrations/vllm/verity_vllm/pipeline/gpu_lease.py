"""A leased Commit: a dispatch item marked ``"lease": "self"`` runs in a pod with no GPU of its own and ``GPU_LEASE`` naming the
node's lease tool (``gpu-lease``), and only its GPU phase holds a GPU.

The row's processes see no CUDA device (``CUDA_VISIBLE_DEVICES`` empty), so nothing outside the lease can open a context on any GPU
of the node. The Commit process starts without a GPU too: its imports, the manifest, the producer facts and the boundary plan run on
the CPU, and only right before the engine (its target reads the device, then the build) does it take the lease (``acquire``):
``"$GPU_LEASE" N --wait --max-min M --mem-gb 0 -- holder``, a child of the Commit that hands it the lease's lock descriptors and the
leased ``CUDA_VISIBLE_DEVICES``. The lease tool ending the holder (M minutes, the Commit's timeout) ends the Commit. The holder ends
once the Commit lets go and nvidia-smi lists nothing on the leased GPUs, so a Commit's exit, whose context the driver frees seconds
after the process has closed the lease's descriptors, stays inside its lease.

A leased Commit's weights registration hashes on its leased GPU (``VERITY_WEIGHTS_HASH=device`` unless the item sets it: the same
leaves, tree and root as the host path, ``commit.committer.leafhash``), where the host path holds the GPU idle while it copies and
hashes every tensor (79 to 91 s for a 7B or 8B model in cov-gm186, gm177 and gm178, against 0.1 s on the device).

After its last run the Commit gives the GPU back (``release``): the engine shut down, the device's CUDA context destroyed with every
allocation in it (as at the process's exit), nvidia-smi showing nothing left on the leased GPU (the node's stray probe reads the
same list), then the lease's locks and holder let go. What follows (binding and manifest coverage, the replay bundle, the weights
pin, the openings after release, the closure) reads host memory only. From there nothing in the process may touch CUDA:
``torch.cuda.is_available()`` is False, a context that reappears ends the process at once, and the process ends with its command
(``run_main``), before the interpreter's teardown could open one. A Commit that still needs the device after its last run (one
with ``--serving-rows``, whose window the C2 site commits from the engine's weights), or whose context did not go, keeps the
lease to its exit and says why (``kept_reason``). Prechecks, manifest and weights of record come before
the Commit process; the verdict, the row's records and the research run's publish after it. Without ``GPU_LEASE`` nothing changes.
A grouped Commit (`commit_group`: the Commits of several rows of one engine in one process) holds one lease: its first row's
``acquire`` takes it, its later rows keep it and the devices it set (``LEASED``), and only its last row's last run releases it.
A TP Commit (`pipeline.tp.lease`) takes all its GPUs with one ``acquire`` and gives them back together after its last run's
finalize, its ranks off their devices first (``give_back`` is either release, over the engine in this process or over the ranks).
A TP Match's capture and its control (`row_tp.TpRow.match`) each take all their GPUs the same way, right before their engine, and
hold them to their exit (their record in ``match/lease/<stage>/``); the Match's check, its served-shape derives and its fold Match see
no device and hold no lease.
A single-GPU Match (`row_stages.SingleRow.match`) runs as its two phases: the GPU phase (``match --phase gpu``: the capture, the
control and the rest of `match.GPU_STAGES`) takes the lease the same way and holds it to its exit (its record in ``match/lease/``),
and the CPU phase over that record runs after it with no device and no lease.

``gpu_lease.json`` beside the row records the lease: requested and started (written by the Commit and the holder), released and the
stage it was released after (``released_at``: the Commit's last run, or ``process.end`` with ``kept``, the reason), and the minutes
waited and held."""
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
import traceback
from typing import Any, Callable, MutableMapping

RECORD = "gpu_lease.json"
MATCH_LEG_KEPT = "a Match stage holds its GPUs to its exit: its engine is its last work"
HEARTBEAT_S = 60
WATCH_S = 0.2                      # after the release, how often the Commit checks that no CUDA context reappeared
STARTED = "VERITY_LEASE_STARTED"   # set by the row for its Commit: the marker the holder writes; the Commit takes its own lease
MAX_MIN = "VERITY_LEASE_MAX_MIN"
TOOL_START = "GPU_LEASE_STARTED"   # the lease tool's own start-time override (its tests): never passed to it
LEASED = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "GPU_LEASE_UUID")   # what `acquire` sets from the holder: the leased devices
WEIGHTS_HASH = "VERITY_WEIGHTS_HASH"   # commit's --weights-hash; a leased Commit's default is device

EMPTY_LOOKS, EMPTY_EVERY_S = 120, 0.5   # how long the holder outlives the Commit's exit at most, waiting for its context to go
CLEAR_LOOKS, CLEAR_EVERY_S = 240, 5.0   # how long a new lease waits at most for another process to leave its GPUs

# the holder: start marker, the lease's lock descriptors to the Commit (socket argv[2]), the leased devices on stdout, then wait for
# the Commit to let go (EOF on stdin: its release, or its exit). A Commit's exit closes the lease's descriptors before the driver has
# freed its context, which nvidia-smi lists for seconds after: the holder ends only once nothing is on the leased GPUs (argv[3] looks
# at most, argv[4] seconds apart; one that nvidia-smi can't answer counts as occupied), so the lease outlives the context.
_HOLDER = r"""
import json, os, re, socket, subprocess, sys, time
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
uuids, looks, t0, k = {u for u in os.environ.get("GPU_LEASE_UUID", "").split(",") if u}, int(sys.argv[3]), time.monotonic(), 0
while k < looks:
    try:
        on = set(subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], capture_output=True, text=True,
                                timeout=10, check=True).stdout.split())
    except (OSError, subprocess.SubprocessError):
        on = uuids
    if not on & uuids:
        break
    k += 1
    time.sleep(float(sys.argv[4]))
if k:
    print(f"gpu-lease holder: {'nvidia-smi still lists a compute process on' if k == looks else 'nothing is left on'} the leased GPU "
          f"{time.monotonic() - t0:.1f} s after the Commit let go ({k} look(s) occupied); the lease ends", file=sys.stderr, flush=True)
"""
_HELD: list[Any] = []
_EXITING = threading.Event()
_RELEASED = threading.Event()
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


def _released(started: str) -> str:
    return started[: -len("started")] + "released"


def holder_argv(environ: MutableMapping[str, str], sock: int) -> list[str]:
    """The lease request: the task's GPU count (``GPU_LEASE_N``, which dispatch sets; else one) for at most the Commit's timeout."""
    return [environ["GPU_LEASE"], environ.get("GPU_LEASE_N") or "1", "--wait", "--max-min", environ.get(MAX_MIN) or "60", "--mem-gb", "0",
            "--", sys.executable, "-c", _HOLDER, environ[STARTED], str(sock), str(EMPTY_LOOKS), str(EMPTY_EVERY_S)]


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
    if not (_EXITING.is_set() or _RELEASED.is_set()):
        print(f"{_utc(time.time())} GPU lease ended under the Commit (holder rc {holder.returncode}): the Commit ends now", file=sys.stderr,
              flush=True)
        os.kill(os.getpid(), signal.SIGKILL)


def _foreign(leased_uuids: list[str], wait: Callable[[float], Any], looks: int) -> tuple[int, list[str]]:
    """Waits while nvidia-smi lists a compute process on the newly leased GPUs: another process's, since this one has no context yet
    (a cancelled Commit's engine workers outlive its lease, and its next waiter's engine found the GPUs full). Returns the looks that
    found one and the GPUs still occupied after ``looks`` more; none when nvidia-smi can't say, and the engine's own check stands."""
    k, on = 0, on_gpus(leased_uuids)
    while on and k < looks:
        k += 1
        wait(CLEAR_EVERY_S)
        on = on_gpus(leased_uuids)
    return k, on or []


def acquire(tl: Any = None, environ: MutableMapping[str, str] | None = None, wait: Callable[[float], Any] = time.sleep) -> None:
    """Take the leased Commit's GPUs now (``VERITY_LEASE_STARTED`` set in ``environ``, default the CLI's; else nothing): wait for the
    lease and for any other process to leave its GPUs (`_foreign`), then see only its devices. Nothing before this call may initialise
    CUDA (the process sees no device until it returns). The lease's locks are this process's from here to its ``release`` or its exit;
    the holder ending first (the lease tool's --max-min) kills this process. A process that holds its lease already (a grouped
    Commit's later rows, `commit_group`) keeps it."""
    environ = environ if environ is not None else (_ENVIRON[0] if _ENVIRON else None)
    if environ is None or not environ.get(STARTED) or not leased(environ):
        return
    if _HELD:
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
    k, left = _foreign(uuids(environ), wait, CLEAR_LOOKS)
    if left:
        holder.kill()
        raise SystemExit(f"GPU lease: nvidia-smi still lists another process on the leased GPU(s) {left} after {k} look(s) "
                         f"{CLEAR_EVERY_S:g} s apart (a cancelled Commit's engine workers outlive its lease): no engine can start there")
    if k:
        print(f"{_utc(time.time())} GPU lease: another process left the leased GPU(s) after {k} look(s)", file=sys.stderr, flush=True)
    _HELD.append((holder, ours, fds, environ[STARTED], environ))
    atexit.register(_EXITING.set)        # registered after torch's and vLLM's, so it runs before their teardown
    threading.Thread(target=_end_with, args=(holder,), daemon=True).start()
    if tl is not None:
        tl.end("gpu_lease.wait", devices=environ.get("CUDA_VISIBLE_DEVICES"), locks=len(fds))


def _kept_by(args: Any, com: Any) -> str | None:
    retain, exclude = getattr(com, "retain", None), getattr(com, "retain_exclude", None)
    replays = getattr(args, "replay_openings", "auto") == "always" or retain != "host" or bool(exclude)
    mode = getattr(getattr(com, "pool", None), "mode", None)
    on_device = com._retained_device_bytes() if hasattr(com, "_retained_device_bytes") else 0
    return (f"the openings after release replay the run on the engine (retain {retain}, excluded {sorted(exclude or [])}, replay openings "
            f"{getattr(args, 'replay_openings', 'auto')})" if getattr(args, "open_after_release", False) and replays else
            f"the retained bytes are in torch's pinned host memory ({mode}), which the CUDA context's end frees"
            if mode is not None and not str(mode).startswith("exact-registered") else
            f"{on_device >> 20} MiB of committed bytes or tree levels are on the device" if on_device else None)


def kept_reason(args: Any, coms: list[Any], eng: Any, environ: MutableMapping[str, str], *, bundle: bool) -> str | None:
    """Why the Commit keeps its GPU to its exit after its last run, or None when nothing after that run reads the device: the C2
    replay deferred to its bundle, whose write ``replay_bundle.start`` began (``bundle``) after reading its device-resident tensors
    (otherwise `defer` reads them at the C2 site, or the replay runs in this process), and for every committer the
    openings after release opened from host and the committed bytes and tree levels in host memory that outlives the CUDA context
    (the pool's registered shared mappings; torch's pinned blocks die with it); one GPU, and vLLM's engine core in this process (its
    shutdown frees it)."""
    core = getattr(getattr(getattr(eng, "llm", None), "llm_engine", None), "engine_core", None)
    gpus = [d for d in environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    return ("the C2 replay runs in this process, on the engine's weights (no --replay-deferred)" if not getattr(args, "replay_deferred", False) else
            "the replay bundle reads the model's buffers at the C2 site (--replay-bundle-overlap 0)" if not getattr(args, "replay_bundle_overlap", 1) else
            "the replay bundle is written at the C2 site or the C2 replay runs in this process (replay_bundle.start began no write: a slim "
            "bundle, a live-only replay, a retention other than host, or a store over the cap)" if not bundle else
            "--serving-rows commits its window at the C2 site, from the engine's weights (c2_replay.serving_rows)"
            if getattr(args, "serving_rows", None) else
            "--tensor-digests reads the retained bytes after the run" if getattr(args, "tensor_digests", False) else
            next((why for why in (_kept_by(args, c) for c in coms) if why), None) or
            ("vLLM's engine core is not in this process" if getattr(core, "engine_core", None) is None else
             f"a lease of {len(gpus)} GPUs" if len(gpus) != 1 else None))


class _Driver:
    """The CUDA driver (libcuda) on this process's devices: which primary contexts are active, and their reset (the context and every
    allocation in it destroyed, as at the process's exit). Neither creates a context."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        n = ctypes.c_int()
        self._ok(self.lib.cuDeviceGetCount(ctypes.byref(n)), "cuDeviceGetCount")
        self.devs = []
        for i in range(n.value):
            d = ctypes.c_int()
            self._ok(self.lib.cuDeviceGet(ctypes.byref(d), i), "cuDeviceGet")
            self.devs.append(d)

    @staticmethod
    def _ok(rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{what}: CUDA driver error {rc}")

    def active(self) -> list[int]:
        out = []
        for i, d in enumerate(self.devs):
            flags, on = ctypes.c_uint(), ctypes.c_int()
            if self.lib.cuDevicePrimaryCtxGetState(d, ctypes.byref(flags), ctypes.byref(on)) != 0 or on.value:
                out.append(i)
        return out

    def reset(self) -> None:
        for d in self.devs:
            self._ok(self.lib.cuDevicePrimaryCtxReset_v2(d), "cuDevicePrimaryCtxReset")


DRIVER: Callable[[], Any] = _Driver


def on_gpus(uuids: list[str]) -> list[str] | None:
    """The leased GPUs (UUIDs) any compute process is on, from nvidia-smi's compute apps as the node's stray probe reads them; None
    when nvidia-smi can't say. Nothing but this process may be on a leased GPU, so after its reset the answer must be none."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    on = {ln.strip() for ln in out.splitlines()}
    return [u for u in uuids if u in on]


def uuids(environ: MutableMapping[str, str]) -> list[str]:
    """The leased GPUs' UUIDs (``GPU_LEASE_UUID``, which the holder set)."""
    return [u for u in environ.get("GPU_LEASE_UUID", "").split(",") if u]


def _gone(drv: Any, leased_uuids: list[str], wait: Callable[[float], Any], tries: int = 10) -> str | None:
    """None once the device has no context and nvidia-smi lists nothing on the leased GPUs, else why not (after ``tries`` looks)."""
    for k in range(tries):
        left, on = drv.active(), on_gpus(leased_uuids)
        if not left and on == []:
            return None
        if k + 1 < tries:
            wait(0.5)
    return (f"a CUDA context is still active on device(s) {left} after its reset" if left else
            "nvidia-smi can't list the compute processes, so the release can't be checked" if on is None else
            f"nvidia-smi still lists a compute process on {on} after the reset")


def _shut_down(eng: Any) -> str | None:
    """vLLM's in-process engine teardown (model runner, scheduler, distributed groups, cached device memory), before the context's
    reset frees the rest; returns its error, if any."""
    import gc

    import torch

    try:
        torch.cuda.synchronize()
        eng.llm.llm_engine.engine_core.shutdown()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 -- the reset frees the device either way; the error goes on the record
        return repr(e)[:300]
    return None


def _read(p: str) -> dict[str, Any]:
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def _note(started: str, **fields: Any) -> None:
    p = _released(started)
    doc = _read(p)
    with open(p + ".tmp", "w") as f:
        json.dump(doc | fields, f)
    os.replace(p + ".tmp", p)


def _no_device() -> Any:
    import torch

    from verity_vllm.engine import hooks

    return hooks.patch(torch.cuda, "is_available", lambda: False)


def give_back(tl: Any, holders: Any, *, after: str, wait: Callable[[float], Any]) -> bool:
    """A leased Commit's GPUs back, at any world size, once the run of stage ``after`` was its last device work. ``holders`` is what
    holds them besides this process's own context: the engine in this process (`_InProcess`) or a TP Commit's ranks
    (`pipeline.tp.lease`). Each step below is theirs; the order, the record and the lease's two ends are this function's:

    1. ``holders.kept()``: why they still need their devices, which keeps the lease to the exit (`keep`);
    2. ``holders.off()``: they leave their devices (``shutdown_s``), or say why they could not;
    3. ``holders.reset()``: their own contexts reset, or why one stays; then this process's (``holders.driver()``), and nvidia-smi
       must list nothing on the leased GPUs (`_gone`), all in ``reset_s``. An error among ``holders.fails`` here keeps the lease
       with ``holders.failed`` and the error;
    4. the lease let go (`let_go`) with ``holders.done()``, or kept naming why; ``holders.error()`` (their teardown's error) and
       ``holders.record()`` (their own fields) go on the record either way."""
    environ = _HELD[0][4]
    tl.start("gpu_lease.release", after=after)
    why = holders.kept()
    if why is not None:
        return keep(tl, why, after=after)
    drv, t0 = None, time.perf_counter()
    why = holders.off()
    t1 = time.perf_counter()
    if why is None:
        try:
            why = holders.reset()
            if why is None:
                drv = holders.driver()
                drv.reset()
                why = _gone(drv, uuids(environ), wait)
        except holders.fails as e:
            why = f"{holders.failed}: {e!r}"[:300]
    spans = {"shutdown_s": round(t1 - t0, 3), "reset_s": round(time.perf_counter() - t1, 3)}
    err, fields = holders.error(), holders.record()
    if why is not None:
        return keep(tl, why, after=after, shutdown_error=err, **fields)
    let_go(tl, drv, after=after, shutdown_error=err, spans=spans | fields, done=holders.done())
    return True


class _InProcess:
    """A one-GPU Commit's holders (`give_back`): vLLM's engine in this process and the committers on it."""

    failed = "the CUDA context could not be reset"
    fails: tuple[type[BaseException], ...] = (OSError, RuntimeError)

    def __init__(self, eng: Any, coms: list[Any], args: Any, environ: MutableMapping[str, str], bundle: bool) -> None:
        self.eng, self.coms, self.args, self.environ, self.bundle = eng, coms, args, environ, bundle
        self.err: str | None = None

    def kept(self) -> str | None:
        return kept_reason(self.args, self.coms, self.eng, self.environ, bundle=self.bundle)

    def off(self) -> None:
        for c in self.coms:             # a thread that used the device ends while its context lives: its exit after the reset opens one
            getattr(c, "_stop_worker", lambda: None)()
        self.err = _shut_down(self.eng)

    def reset(self) -> None:
        """The engine's context is this process's, which `give_back` resets."""

    def driver(self) -> Any:
        return DRIVER()

    def error(self) -> str | None:
        return self.err

    def record(self) -> dict[str, Any]:
        return {}

    def done(self) -> str:
        return f"engine shut down{' (' + self.err + ')' if self.err else ''}, CUDA context reset"


def release(tl: Any, eng: Any, coms: list[Any], args: Any, *, last: bool, after: str, bundle: bool,
            wait: Callable[[float], Any] = time.sleep) -> bool:
    """Give a leased Commit's GPU back once ``last`` (the run of stage ``after`` was its last device work): the engine shut down,
    the device's context reset, nothing left on the leased GPU, then the lease's locks closed, the holder told (EOF) and waited for,
    and ``gpu_lease.released`` written with the time and ``after`` (`give_back`). Kept to the exit (``kept_reason``, or a context
    that did not go) the file names why instead. ``bundle``: ``replay_bundle.start`` began the bundle's write. Returns whether the
    GPU was released."""
    if not _HELD or not last:
        return False
    return give_back(tl, _InProcess(eng, coms, args, _HELD[0][4], bundle), after=after, wait=wait)


def keep(tl: Any, why: str, *, after: str, shutdown_error: str | None = None, **fields: Any) -> bool:
    """The lease kept to the Commit's exit, ``gpu_lease.released`` saying why (``kept``); returns False (not released)."""
    _note(_HELD[0][3], kept=why, after=after, shutdown_error=shutdown_error, **fields)
    tl.end("gpu_lease.release", released=False, why=why)
    print(f"[delta] GPU LEASE kept to the Commit's exit: {why}", flush=True)
    return False


def let_go(tl: Any, drv: Any, *, after: str, shutdown_error: str | None, spans: dict[str, Any], done: str) -> None:
    """Once nothing of the Commit is on the leased GPUs (``done`` says how): every lease lock closed at once, the holder told (EOF)
    and waited for, ``gpu_lease.released`` written with the time and ``after``; from here no CUDA in this process (``_watch``)."""
    holder, ours, fds, started, environ = _HELD[0]
    _RELEASED.set()
    _no_device()
    t2 = time.perf_counter()
    for fd in fds:
        os.close(fd)
    ours.close()
    if holder.stdin:
        holder.stdin.close()
    try:
        holder.wait(timeout=60)
    except subprocess.TimeoutExpired:
        holder.kill()
        holder.wait()
    _HELD.clear()
    t = time.time()
    spans["holder_s"] = round(time.perf_counter() - t2, 3)
    _note(started, released=_utc(t), t=round(t, 3), released_at=after, shutdown_error=shutdown_error, **spans)
    threading.Thread(target=_watch, args=(drv, started), daemon=True).start()
    tl.end("gpu_lease.release", released=True, after=after, shutdown_error=shutdown_error, **spans)
    print(f"[delta] GPU LEASE released after {after}: {done}, nothing on {environ.get('GPU_LEASE_UUID')}; the checks that follow read "
          "host memory", flush=True)


def _watch(drv: Any, started: str, wait: Callable[[float], Any] = time.sleep, every_s: float = WATCH_S) -> None:
    """After the release, a CUDA context that reappears (a CUDA call in this process) is on a GPU the pool may have granted to
    another holder: the Commit ends at once, named in its log and in ``gpu_lease.released``."""
    import faulthandler

    while True:
        wait(every_s)
        on = drv.active()
        if on:
            names = sorted(t.name for t in threading.enumerate())
            print(f"{_utc(time.time())} GPU lease: a CUDA context reappeared on device(s) {on} after the lease was released: the Commit "
                  f"ends now (threads {names}; their stacks follow)", file=sys.stderr, flush=True)
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            _note(started, context_reappeared=_utc(time.time()), threads=names)
            os.kill(os.getpid(), signal.SIGKILL)
            return


def run_main(main: Callable[..., Any], *a: Any) -> Any:
    """``main(*a)``, the process's command. Once the process has released its lease it ends here, with ``main``'s exit status
    (``os._exit``), so nothing in the interpreter's or the libraries' teardown (torch's allocator and handles, vLLM's atexit,
    CUDA's static destructors) opens a context on the GPU it gave back."""
    try:
        code = main(*a)
    except SystemExit as e:
        if not _RELEASED.is_set():
            raise
        code = e.code
    except BaseException:
        if not _RELEASED.is_set():
            raise
        traceback.print_exc()
        code = 1
    if not _RELEASED.is_set():
        return code
    if code is not None and not isinstance(code, int):
        print(code, file=sys.stderr)
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code or 0)


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
        *, clock: Callable[[], float] = time.time, every_s: float = HEARTBEAT_S, kept: str | None = None) -> tuple[int, dict[str, Any]]:
    """Run the leased Commit ``cmd`` -> (its rc, the lease record). ``run_tl(timeout, argv)`` runs argv as the row runs a stage; it may
    wait for the GPU as long as the Commit may run, and the lease tool ends the Commit at its timeout.  ``kept``: the record's reason
    when the process never released (one that holds its lease to its exit by design, a TP Match's capture); else the Commit's."""
    started = os.path.join(row_dir, "gpu_lease.started")
    for p in (started, _requested(started), _released(started)):
        if os.path.exists(p):
            os.remove(p)
    t0 = clock()
    stop = threading.Event()
    beat = threading.Thread(target=_heartbeat, args=(log, started, stop, every_s), daemon=True)
    beat.start()
    environ.update({STARTED: started, MAX_MIN: str(max_min(timeout_s))})
    hash_ours = WEIGHTS_HASH not in environ
    if hash_ours:
        environ[WEIGHTS_HASH] = "device"
    try:
        rc = run_tl(2 * int(timeout_s), cmd)
    finally:
        environ.pop(STARTED, None)
        environ.pop(MAX_MIN, None)
        if hash_ours:
            environ.pop(WEIGHTS_HASH, None)
        stop.set()
        beat.join()
    t1 = clock()
    rel = _read(_released(started))
    t_rel = float(rel["t"]) if rel.get("released") else t1
    rec: dict[str, Any] = {"schema": "verity-vllm/gpu-lease/v1", "tool": environ["GPU_LEASE"], "max_min": max_min(timeout_s),
                           "process_started": _utc(t0), "requested": None, "released": _utc(t_rel),
                           "released_at": rel.get("released_at") or "process.end", "process_ended": _utc(t1),
                           "kept": rel.get("kept") or (None if rel.get("released") else kept or "the Commit ended before its release (its last "
                                                       "run was not an instrumented one, or it ended before that run's records)"),
                           "rc": rc, "started": None, "waited_min": None, "held_min": None}
    rec.update({k: rel[k] for k in ("shutdown_s", "reset_s", "holder_s", "shutdown_error", "context_reappeared", "threads", "ranks") if rel.get(k) is not None})
    tr, ts = _stamp(_requested(started)), _stamp(started)
    if tr is not None:
        rec["requested"] = _utc(tr)
    if ts is not None:
        rec.update(started=_utc(ts), waited_min=round(max(0.0, ts - (tr if tr is not None else t0)) / 60, 2),
                   held_min=round(max(0.0, t_rel - ts) / 60, 2))
    with open(os.path.join(row_dir, RECORD), "w") as f:
        json.dump(rec, f, indent=1)
    return rc, rec
