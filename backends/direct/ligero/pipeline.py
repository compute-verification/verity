"""Asynchronous plumbing for the B-Ligero prover (lane hp2-host): the prover is a generator of *stages* separated by
the points where the host must wait for the device (the Merkle root; the test messages; the openings) or for a host
computation that runs off the main thread (the SHAKE-256 challenge expansion, the statement digest).  Running one
generator to completion (``run_to_completion``) is the plain prover; ``prove_many`` interleaves several sub-batches
on their own CUDA streams so that the host work of sub-batch i (the ~2.9 MB sequential SHAKE-256 squeeze, ~5-8 ms
of CPU; the blake2b statement digest; Python launch overhead) hides under the device work of sub-batch i + 1.

The protocol is untouched: every sub-batch is an independent session (own coins, own transcript), each message is
still computed only after the coin it must follow has been opened, and the bytes are those of the sequential
prover (``bitexact`` harness of lane hostphase / hp2-host).

Waitables a stage may ``yield``: a ``torch.cuda.Event`` (recorded on the stage's stream), a ``concurrent.futures.Future``,
or ``None`` (a cooperative yield point).  ``Slot``: per-pipeline-position resources -- a CUDA stream and reusable
pinned host buffers for the challenge bytes, the transcript messages and the openings (one D2H copy each).

SHAKE-256 is squeezed by OpenSSL's ``EVP_DigestFinalXOF`` through ``ctypes`` (the same library CPython's ``hashlib``
wraps, so the stream is byte-identical) because a ``ctypes`` call releases the GIL and can run on a worker thread
while the main thread launches kernels; ``hashlib``'s XOF ``digest()`` holds the GIL.  Without ``libcrypto`` the
squeeze falls back to ``hashlib`` on the calling thread (correct, not overlapped).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

from .phases import rescale as _rescale

__all__ = ["AnyOf", "Slot", "POOL", "prove_many", "run_to_completion", "shake_into", "slot_for", "stream_context"]


# ---------------------------------------------------------------------------------------------------------------------
# SHAKE-256 squeeze off the main thread
# ---------------------------------------------------------------------------------------------------------------------

_LIB = None
_LIB_TRIED = False
_LIB_LOCK = threading.Lock()


def _libcrypto():
    global _LIB, _LIB_TRIED
    if _LIB_TRIED:
        return _LIB
    with _LIB_LOCK:
        if _LIB_TRIED:
            return _LIB
        lib = None
        for name in ("libcrypto.so.3", "libcrypto.so", "libcrypto.3.dylib", "libcrypto.dylib", ctypes.util.find_library("crypto")):
            if not name:
                continue
            try:
                cand = ctypes.CDLL(name)
                for fn in ("EVP_MD_CTX_new", "EVP_shake256", "EVP_DigestInit_ex", "EVP_DigestUpdate", "EVP_DigestFinalXOF", "EVP_MD_CTX_free"):
                    getattr(cand, fn)
            except (OSError, AttributeError):
                continue
            cand.EVP_MD_CTX_new.restype = ctypes.c_void_p
            cand.EVP_shake256.restype = ctypes.c_void_p
            cand.EVP_DigestInit_ex.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            cand.EVP_DigestInit_ex.restype = ctypes.c_int
            cand.EVP_DigestUpdate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            cand.EVP_DigestUpdate.restype = ctypes.c_int
            cand.EVP_DigestFinalXOF.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            cand.EVP_DigestFinalXOF.restype = ctypes.c_int
            cand.EVP_MD_CTX_free.argtypes = [ctypes.c_void_p]
            cand.EVP_MD_CTX_free.restype = None
            # self-check against hashlib once: the two must be the same stream
            probe = np.empty(4096, dtype=np.uint8)
            if _xof_libcrypto(cand, b"ligero-b/pipeline-selftest", probe) and bytes(probe) == hashlib.shake_256(b"ligero-b/pipeline-selftest").digest(4096):
                lib = cand
                break
        _LIB = lib
        _LIB_TRIED = True
        return _LIB


_TLS = threading.local()


def _xof_libcrypto(lib, seed: bytes, out: np.ndarray) -> bool:
    """One EVP context per thread (created once): three ctypes calls per squeeze -- each call releases and re-acquires
    the GIL, and a re-acquisition can wait a full switch interval behind a busy main thread (5 ms by default; see
    ``prove_many``), so fewer calls = less GIL ping-pong."""
    ctx = getattr(_TLS, "ctx", None)
    if ctx is None:
        ctx = lib.EVP_MD_CTX_new()
        if not ctx:
            return False
        _TLS.ctx = ctx
    ok = lib.EVP_DigestInit_ex(ctx, lib.EVP_shake256(), None) == 1
    ok = ok and lib.EVP_DigestUpdate(ctx, seed, len(seed)) == 1
    ok = ok and lib.EVP_DigestFinalXOF(ctx, out.ctypes.data, out.nbytes) == 1
    return ok


def shake_into(seed: bytes, out: np.ndarray) -> np.ndarray:
    """``out[:] = SHAKE-256(seed)`` (``out``: contiguous uint8), identical to ``hashlib.shake_256(seed).digest(len(out))``.
    libcrypto through ctypes when available (GIL released for the squeeze), else hashlib."""
    assert out.dtype == np.uint8 and out.flags.c_contiguous
    lib = _libcrypto()
    if lib is not None and _xof_libcrypto(lib, seed, out):
        return out
    out[:] = np.frombuffer(hashlib.shake_256(seed).digest(out.nbytes), dtype=np.uint8)
    return out


def shake_overlapped() -> bool:
    """Whether the squeeze runs GIL-free (libcrypto found and self-checked)."""
    return _libcrypto() is not None


POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ligero-host")


class HotWorker:
    """One thread for the challenge squeeze.  A pool thread that sleeps between requests runs the squeeze on a cold core
    (measured on the EPYC pod: 11-14 ms for a 2.85 MB squeeze that takes 7.4 ms on a hot core, i.e. the ramp of the
    frequency governor / caches); while a pipelined pass is active (``active(True)``) this thread keeps its core busy
    between requests with GIL-free blake2b bursts (~0.3 ms each, so a request waits <= 0.3 ms), and sleeps otherwise.
    Costs one core during a pass."""

    _WARM = bytes(300_000)

    def __init__(self):
        import queue
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._active = False
        self._thread = threading.Thread(target=self._run, name="ligero-squeeze", daemon=True)
        self._thread.start()

    def active(self, on: bool) -> None:
        self._active = on
        if not on:
            self._q.put(None)                    # wake the loop so it notices and goes back to sleeping gets

    def submit(self, fn, *args) -> Future:
        fut: Future = Future()
        self._q.put((fut, fn, args))
        return fut

    def _run(self):
        import queue
        while True:
            if self._active:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    h = hashlib.blake2b()
                    h.update(self._WARM)         # GIL released for the hashing: the core stays hot, the main thread runs
                    continue
            else:
                item = self._q.get()
            if item is None:
                continue
            fut, fn, args = item
            try:
                fut.set_result(fn(*args))
            except BaseException as e:  # noqa: BLE001
                fut.set_exception(e)


SQUEEZE = HotWorker()


def squeeze(seed: bytes, out: np.ndarray, inline: bool) -> Future:
    """The challenge squeeze as a Future: inline (sequential prover: the main thread would only block on it, and it is
    the hot thread) or on the hot worker (pipelined prover: the main thread keeps launching)."""
    if inline:
        fut: Future = Future()
        try:
            fut.set_result(shake_into(seed, out))
        except BaseException as e:  # noqa: BLE001
            fut.set_exception(e)
        return fut
    return SQUEEZE.submit(shake_into, seed, out)


# ---------------------------------------------------------------------------------------------------------------------
# Streams and pinned buffers per pipeline position
# ---------------------------------------------------------------------------------------------------------------------

class Slot:
    """One pipeline position: a CUDA stream and reusable pinned host buffers (``pinned(name, nbytes)`` returns a
    uint8 pinned tensor of at least ``nbytes``, grown on demand; a buffer is reused by the next sub-batch in the same
    slot, so consumers copy out of it before the stage that refills it runs)."""

    def __init__(self, index: int, device):
        self.index = index
        self.device = torch.device(device) if device is not None else None
        self.cuda = self.device is not None and self.device.type == "cuda"
        self.stream = torch.cuda.Stream(device=self.device) if self.cuda else None
        self._pinned: dict[str, torch.Tensor] = {}

    def pinned(self, name: str, nbytes: int) -> torch.Tensor:
        buf = self._pinned.get(name)
        if buf is None or buf.numel() < nbytes:
            buf = torch.empty(max(nbytes, 1), dtype=torch.uint8, pin_memory=self.cuda)
            self._pinned[name] = buf
        return buf

    def __repr__(self):
        return f"Slot({self.index}, {self.device})"


_SLOTS: dict = {}
_DEFAULT_SLOT: dict = {}


def slot_for(device, index: int = 0) -> Slot:
    key = (str(device), index)
    s = _SLOTS.get(key)
    if s is None:
        s = Slot(index, device)
        _SLOTS[key] = s
    return s


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def stream_context(slot: Slot | None):
    """``torch.cuda.stream(slot.stream)`` for a CUDA slot, a no-op context otherwise (CPU, or the caller's stream)."""
    if slot is not None and slot.cuda and slot.stream is not None:
        return torch.cuda.stream(slot.stream)
    return _NullContext()


def cupy_stream_context():
    """cupy launches (encoder, Merkle, mask sampler, fused tests) follow torch's CURRENT stream inside this context."""
    import warnings

    import cupy as cp
    ts = torch.cuda.current_stream()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return cp.cuda.ExternalStream(ts.cuda_stream)


# ---------------------------------------------------------------------------------------------------------------------
# Running stage generators
# ---------------------------------------------------------------------------------------------------------------------

class AnyOf:
    """A waitable ready when any of ``parts`` is (a job that drives two sub-provers -- relchain's paired G/H -- waits on
    whichever advances first)."""
    __slots__ = ("parts",)

    def __init__(self, parts):
        self.parts = list(parts)


def _is_ready(w) -> bool:
    if w is None:
        return True
    if isinstance(w, Future):
        return w.done()
    if isinstance(w, AnyOf):
        return any(_is_ready(p) for p in w.parts)
    return bool(w.query())          # torch.cuda.Event


def _is_net(w) -> bool:
    """A waitable that stands for a reply from a LIVE VERIFIER (``live.LiveCoins``: a Future with ``net = True``) --
    time the driver spends idle with only such waitables pending is network wait, not prover time."""
    if isinstance(w, AnyOf):
        return all(_is_net(p) for p in w.parts)
    return isinstance(w, Future) and bool(getattr(w, "net", False))


WAIT_POLL = 0.0001   # seconds between polls while the driver waits on a mix of device events and network replies


def _wait_any(active: list) -> list:
    """Block until at least one active job's waitable is ready (polling: a CUDA event and a socket reply cannot be waited
    on together any other way); returns the ready jobs."""
    while True:
        ready = [j for j in active if _is_ready(j.waitable)]
        if ready:
            return ready
        time.sleep(WAIT_POLL)


def _wait(w) -> None:
    if w is None:
        return
    if isinstance(w, Future):
        w.result()
    elif isinstance(w, AnyOf):
        while not _is_ready(w):
            time.sleep(WAIT_POLL)
    else:
        w.synchronize()


def run_to_completion(gen, slot: Slot | None = None):
    """Drive one stage generator to its return value on the calling thread (the plain, sequential prover)."""
    try:
        with stream_context(slot):
            w = next(gen)
        while True:
            _wait(w)
            with stream_context(slot):
                w = gen.send(None)
    except StopIteration as stop:
        return stop.value


class _Job:
    __slots__ = ("index", "gen", "waitable", "slot", "started", "done", "value", "stage")

    def __init__(self, index, gen, slot):
        self.index, self.gen, self.slot = index, gen, slot
        self.waitable, self.started, self.done, self.value = None, False, False, None
        self.stage = 0

    def step(self):
        try:
            with stream_context(self.slot):
                self.waitable = self.gen.send(None) if self.started else next(self.gen)
                self.started = True
        except StopIteration as stop:
            self.started, self.done, self.value, self.waitable = True, True, stop.value, None
        self.stage += 1


def prove_many(starters, device, depth: int = 2, timings_key: str = "total") -> tuple[list, float]:
    """Run ``len(starters)`` stage generators with at most ``depth`` in flight, each on its own slot (stream + pinned
    buffers), interleaving them: a job advances when its waitable is ready; when none is, the scheduler blocks on the
    oldest.  ``starters[i](slot)`` must return the generator of sub-batch i (it is called when the job STARTS, so
    per-sub-batch randomness -- the verifier's coins, the mask keys -- is drawn in sub-batch order, as sequentially).
    Returns ``(values, wall_seconds)``; each value's ``timings`` (when it has one) is rescaled so that the per-value
    ``timings_key`` entries sum to the wall: the phases become proportional shares of the overlapped wall (the
    un-rescaled event-measured phases are kept under ``timings["measured_<phase>"]``)."""
    import sys
    depth = max(1, int(depth))
    n = len(starters)
    results = [None] * n
    active: list[_Job] = []
    # the worker threads (SHAKE squeeze, statement digest) re-acquire the GIL a few times per sub-batch while the main
    # thread launches kernels: with the default 5 ms switch interval each re-acquisition can wait ~5 ms (measured: a
    # 7.4 ms squeeze took 17.6 ms next to a busy main thread); 0.5 ms bounds the wait without measurable launch cost
    prev_switch = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_INTERVAL)
    SQUEEZE.active(depth > 1)
    free_slots = [slot_for(device, i) for i in range(depth)]
    nxt = 0
    busy = waited = 0.0                                         # main-thread time launching vs blocked on a waitable
    net_idle = 0.0                                              # ... of which: every job waiting on a live verifier
    stage_busy: dict = {}                                       # main-thread time per stage index (diagnostics)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    while nxt < n or active:
        # start jobs while slots are free (their first stage launches device work immediately)
        while nxt < n and free_slots:
            slot = free_slots.pop(0)
            job = _Job(nxt, None, slot)
            tb = time.perf_counter()
            with stream_context(slot):
                job.gen = starters[nxt](slot)
            ts_ = time.perf_counter()
            job.step()
            dt = time.perf_counter() - tb
            busy += dt
            stage_busy[0] = stage_busy.get(0, 0.0) + dt
            stage_busy[-1] = stage_busy.get(-1, 0.0) + (ts_ - tb)      # of which: the starter (coins, marshal lookup)
            nxt += 1
            if job.done:
                results[job.index] = job.value
                free_slots.append(slot)
            else:
                active.append(job)
        if not active:
            continue
        ready = [j for j in active if _is_ready(j.waitable)]
        if not ready:
            tw = time.perf_counter()
            if any(_is_net(j.waitable) for j in active):
                # a live verifier's reply is among the pending waitables: wait for whichever comes first (device event
                # or reply); when EVERY job waits on the verifier the interval is network wait (nothing to prove)
                net_only = all(_is_net(j.waitable) for j in active)
                ready = _wait_any(active)
                dt = time.perf_counter() - tw
                if net_only:
                    net_idle += dt
            else:
                oldest = active[0]
                _wait(oldest.waitable)
                dt = time.perf_counter() - tw
                ready = [oldest]
            waited += dt
        for job in ready:
            tb = time.perf_counter()
            job.step()
            dt = time.perf_counter() - tb
            busy += dt
            stage_busy[job.stage - 1] = stage_busy.get(job.stage - 1, 0.0) + dt
            if job.done:
                results[job.index] = job.value
                active.remove(job)
                free_slots.append(job.slot)
                free_slots.sort(key=lambda s: s.index)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0
    sys.setswitchinterval(prev_switch)
    SQUEEZE.active(False)
    # with a live verifier: the prover's own time is the wall minus the intervals where every sub-batch in flight was
    # waiting for a reply (nothing to launch, nothing on the device); that idle time is each sub-batch's ``net_wait``
    # share, so total + hints_host + net_wait over the sub-batches = the wall (the runners' t.total_live)
    _rescale(results, wall - net_idle, timings_key)
    if net_idle > 0.0:
        for r in results:
            tm = getattr(r, "timings", None)
            if isinstance(tm, dict):
                tm["net_wait"] = tm.get("net_wait", 0.0) + net_idle / max(1, n)
    LAST_STATS.update({"n": n, "depth": depth, "wall": wall, "host_busy": busy, "host_waited": waited, "net_idle": net_idle,
                       "stage_busy_ms": {k: round(v * 1e3, 2) for k, v in sorted(stage_busy.items())}})
    return results, wall


SWITCH_INTERVAL = 0.0005   # GIL switch interval during a pipelined pass (seconds)

#: the last ``prove_many`` pass: wall, main-thread busy (launching / host work) and blocked time -- ``host_busy / wall``
#: near 1 means the pass is bound by the main thread's launch + host work, not by the device.
LAST_STATS: dict = {}
