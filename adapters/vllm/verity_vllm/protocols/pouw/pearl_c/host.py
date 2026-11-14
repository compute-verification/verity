"""The Pearl-C device path's host helpers (`pouw_pearl_c_device`): the epoch of the challenger's salt coin, held in this
process or a live session's (`epoch_of`), device byte buffers and their uploads, the current stream's
handle, each call's keys by native SHA-512 (`_Keys`, byte for byte `call_keys`), the pinned upload ring (`_Uploads`), the keys
kept on the device by call (`_KeyTable`), the streams a run owns (`_Lane`) and the driver calls a lane graph's replay makes on
their handles (`_Driver`), and a retained pass's files (`_write`, `write_behind`, `write_manifest`) and its
completion behind its step (`behind`).
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from verity.commitments import RangeIndexedDomain
from verity.commitments.merkle import domain_message
from verity.protocols.pouw.verifier import salt as PS
from verity.protocols.pouw.verifier import window as Wd
from verity.protocols.pouw.verifier.audit import (
    HM96,
    Epoch,
    Workload,
    activation_binding,
    hm96_kernel_constants,
    rows_domain,
    salt_of,
    weights_root,
)
from verity.protocols.pouw.verifier.coins import CoinCommitment, CoinOpening
from verity.protocols.pouw.verifier.protocol import DomainError
from verity.protocols.spot.device import challenger as Ch
from verity.protocols.spot.device.challenger import LiveCommitment, LiveOpening

#: the pinned ring's key slots (each call's dom_a ‖ unit, 168 bytes)
UPLOAD_SLOTS = 1024
#: torch's stream priority for each of a schedule's priorities (torch maps values past its range to the nearest valid one)
PRIORITY = {"least": 0, "caller": 0, "greatest": -(1 << 10)}
#: the rows of the call the native keys are held to `call_keys` at, once (any row count of the domain does)
CHECK_ROWS = 16
#: the threads that write a retained pass's call files behind its calls (`write_behind`)
WRITE_THREADS = 8
#: the pinned host bytes a device's copies to its retained files land in (`_Arena`): this many rings of this many bytes
RING_BYTES = 1 << 30
ARENA_RINGS = 4
_writers: tuple[int, ThreadPoolExecutor] | None = None     # the pool and the process it was made in
_finisher: tuple[int, ThreadPoolExecutor] | None = None    # the thread `behind` runs on, and its process
_copiers: dict[tuple[int, int | None], Any] = {}           # each (process, device)'s `_Copies`


def torch_uint8() -> Any:
    import torch
    return torch.uint8


def _zeros(n: int, device: Any) -> Any:
    import torch
    return torch.zeros(max(n, 16), dtype=torch.uint8, device=device)


def _put(t: Any, data: bytes) -> None:
    """`data` into device bytes `t`, on the current stream (the pinned staging copy is torch's to keep alive)."""
    import torch
    host = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    t[:len(data)].copy_(host.pin_memory() if t.is_cuda else host, non_blocking=t.is_cuda)


def _constants(scheme: Any, device: Any, salt_key: bytes | None) -> Any:
    """The pipeline's constant keys on the device: the prover's hm96 salt key and, from byte 256,
    `audit.hm96_kernel_constants`."""
    if salt_key is None or len(salt_key) != 32:
        raise ValueError(f"{scheme.name}: -h3's trees take the prover's 32-byte hm96 salt key")
    consts = hm96_kernel_constants()
    t = _zeros(256 + len(consts), device)
    _put(t, salt_key.ljust(256, b"\0") + consts)
    return t


def _constant_keys(t: Any) -> dict[str, int]:
    p = t.data_ptr()
    return {"salt_key": p, "hm": p + 256}


#: the salt coin's commitment and opening a device run takes: the challenger's held in this process, or a live session's
type SaltCoins = CoinCommitment | LiveCommitment
type SaltOpening = CoinOpening | LiveOpening


def epoch_of(workload: Workload, roots: dict[int, bytes], coins: SaltCoins, opening: SaltOpening,
             pass_index: int | None = None) -> Epoch:
    """The epoch whose salt is the challenger's salt coin `opening`, opened against `coins` after the weights' root: a
    `CoinOpening` of a `CoinCommitment` through `Epoch.start`, or a live session's `LiveOpening` of its `LiveCommitment`
    (the auditor host's, `pouw_h1_window.open_session`) under the same rules: every weight committed and only those, the
    commitment a Pearl-C salt session's, its salt coin alone (`window.salt_epoch_reason`), and the opening that coin,
    opening against it after the salt epoch's root (`challenger.live_reason`), `salt.epoch_open_root` of the weights' root
    W and `pass_index`, the epoch's number k over W, an integer in [0, 2^64).  Either is refused with `DomainError`; a
    live salt is `audit.salt_of`'s over W, as H1's and ncp-v2's are."""
    live = isinstance(coins, LiveCommitment), isinstance(opening, LiveOpening)
    if not any(live):
        return Epoch.start(workload, roots, coins, opening)
    if not all(live):
        raise DomainError("a live session's salt coin opens against that session's live coin commitment, and nothing else")
    if set(roots) != set(workload.weights()):
        raise DomainError("every weight of the workload is committed, and only those")
    root = weights_root(roots)
    why = Wd.salt_epoch_reason(coins) or Ch.live_reason(coins, opening, Wd.SALT, 0, PS.epoch_open_root(root, pass_index))
    if why is not None:
        raise DomainError(f"the salt's {why}")
    return Epoch(workload, dict(roots), salt_of(coins, opening.coin, root))


class _TorchStream:
    """torch's Stream object for the current stream, looked up again only when the raw handle changes: `Event.record()`
    without a stream calls `torch.cuda.current_stream`, about 15 µs."""

    def __init__(self):
        self.raw, self.stream = None, None

    def __call__(self, raw: int) -> Any:
        if raw != self.raw:
            import torch
            self.raw, self.stream = raw, torch.cuda.current_stream()
        return self.stream


def _stream(device: Any) -> int:
    """The current stream's handle: torch's raw accessor, not `torch.cuda.current_stream`, which costs about 15 µs a call."""
    import torch
    if device.type != "cuda":
        return 0
    return torch._C._cuda_getCurrentRawStream(device.index if device.index is not None else torch.cuda.current_device())


def _bytes(t: Any) -> bytes:
    return t.cpu().numpy().tobytes()


def call_keys(fixture: Any, salt: bytes, root_b: bytes, index: int, m16: int, hash: str = HM96) -> bytes:
    """A call's dom_a ‖ unit, as `fixture.keys_of` derives them for matmul `index` of m₁₆ rows, under tree hash `hash` (-h3's
    is `HM96`)."""
    return rows_domain(activation_binding(salt, index), index, m16, hash).domain_id + fixture.unit_block(salt, root_b, index)


class _Keys:
    """`call_keys` natively: dom_a is `merkle.domain_message` hashed by hashlib's SHA-512.  The native path is held to
    `call_keys` once at construction, so a tree hash whose domain id isn't that digest of that message fails there.
    A call's index is unique in its epoch (`pouw_pass.call_index`), so no call's keys are another's and none are kept.
    ``native`` set to None takes `call_keys` itself (the reference path)."""

    def __init__(self, fixture: Any, salt: bytes, hash: str = HM96):
        self.native, self.fixture, self.salt, self.hash = hashlib.sha512, fixture, salt, hash
        self.positions: dict[int, bytes] = {}
        root = bytes(64)
        if self(root, 0, CHECK_ROWS) != call_keys(fixture, salt, root, 0, CHECK_ROWS, hash):
            raise ValueError(f"native keys don't frame tree hash {hash!r} as CommitmentDomain does")

    def __call__(self, root_b: bytes, index: int, m16: int) -> bytes:
        if self.native is None:
            return call_keys(self.fixture, self.salt, root_b, index, m16, self.hash)
        pos = self.positions.get(m16)
        if pos is None:
            pos = self.positions[m16] = RangeIndexedDomain(m16).identity_sha512()
        msg = domain_message(activation_binding(self.salt, index), index, pos, m16, "sha512")
        return self.native(msg).digest() + self.salt + root_b + index.to_bytes(8, "little")


def _event() -> Any:
    """A CUDA event (no timing), created now, so that its handle (``cuda_event``) is valid for `_Driver`'s calls before torch
    first records it."""
    import torch
    e = torch.cuda.Event()
    e.record()
    return e


class _Driver:
    """The CUDA driver calls a lane graph's replay makes on raw handles (`pouw_pearl_c_lanes`): graph launches, event records
    and waits, and the slot's copy, each a few µs fewer than through torch, which looks up the device and the current stream
    around each.  `cu` is the kernel's driver library (``run.Dev.cu``); a call that fails raises with the driver's code."""

    SIGS: ClassVar[dict[str, tuple[str, ...]]] = {
        "cuGraphLaunch": ("vp", "vp"), "cuEventRecord": ("vp", "vp"), "cuStreamWaitEvent": ("vp", "vp", "u"),
        "cuMemcpyHtoDAsync_v2": ("u64", "vp", "sz", "vp")}

    def __init__(self, cu: Any):
        types = {"vp": ctypes.c_void_p, "u": ctypes.c_uint, "u64": ctypes.c_uint64, "sz": ctypes.c_size_t}
        self.f = {}
        for name, args in self.SIGS.items():
            f = getattr(cu, name)
            f.argtypes, f.restype = [types[a] for a in args], ctypes.c_int
            self.f[name] = f

    def _ck(self, name: str, *args: Any) -> None:
        rc = self.f[name](*args)
        if rc:
            raise RuntimeError(f"{name} failed: CUDA driver error {rc}")

    def launch(self, graph_exec: int, stream: int) -> None:
        self._ck("cuGraphLaunch", graph_exec, stream)

    def record(self, event: Any, stream: int) -> None:
        self._ck("cuEventRecord", event.cuda_event, stream)

    def wait(self, stream: int, event: Any) -> None:
        self._ck("cuStreamWaitEvent", stream, event.cuda_event, 0)

    def copy_in(self, dst: int, src: int, n: int, stream: int) -> None:
        self._ck("cuMemcpyHtoDAsync_v2", dst, src, n, stream)


class _Lane:
    """A stream the run owns, at a schedule's priority, and its handle; off the GPU a stand-in handle (``standin``) for the
    recording device's log.  ``priority`` is the one the device gave it.  Its events hand work over to it: from the lane
    before it (``event``; for A's tree, the main lane's per-row keying) and, on a side lane, from the other side lane
    where that ran the call's A tree, at its end (``tree_event``)."""
    stream: Any                 # torch's stream on the GPU; None off it

    def __init__(self, device: Any, priority: str, standin: int):
        import torch
        if device.type == "cuda":
            self.stream = torch.cuda.Stream(device=device, priority=PRIORITY[priority])
            self.raw, self.priority = self.stream.cuda_stream, self.stream.priority
            self.event, self.tree_event = _event(), _event()
        else:
            self.stream, self.raw, self.priority = None, standin, PRIORITY[priority]
            self.event = self.tree_event = None


class _Uploads:
    """Small host-to-device copies through a pinned ring allocated once: each is one copy on the current stream, and a slot is
    written again only once its last copy is done (its event)."""

    def __init__(self, device: Any, slots: int = UPLOAD_SLOTS, width: int = 256):
        import torch
        self.cuda, self.slots, self.width, self.i = device.type == "cuda", slots, width, 0
        self.host = torch.zeros((slots, width), dtype=torch.uint8, pin_memory=self.cuda)
        self.view, self.host_ptr = self.host.numpy(), self.host.data_ptr()
        self.events: list[Any] = [None] * slots

    def _slot(self, data: bytes) -> int:
        """The next slot, once its last copy is done, holding `data`."""
        i = self.i
        self.i = (i + 1) % self.slots
        if self.events[i] is not None:
            self.events[i].synchronize()
        self.view[i, :len(data)] = np.frombuffer(data, np.uint8)
        return i

    def put(self, dst: Any, data: bytes, stream: Any = None) -> None:
        """`data` into `dst` on `stream` (torch's current stream when None)."""
        import torch
        i, n = self._slot(data), len(data)
        dst[:n].copy_(self.host[i, :n], non_blocking=self.cuda)
        if self.cuda:
            if self.events[i] is None:
                self.events[i] = torch.cuda.Event()
            self.events[i].record(stream)

    def put_raw(self, dst: int, data: bytes, stream: int, driver: _Driver) -> None:
        """`data` into device address `dst` on raw stream `stream`, by `driver`'s calls."""
        i = self._slot(data)
        driver.copy_in(dst, self.host_ptr + i * self.width, len(data), stream)
        if self.events[i] is None:
            self.events[i] = _event()
        driver.record(self.events[i], stream)


def memmove(dst: int, src: int, n: int, raw: int) -> None:
    """The run's device-to-device copy (`PearlCDeviceRun.copy`) off the GPU: host memory, no stream."""
    ctypes.memmove(dst, src, n)


def _entry(path: Path, n: int) -> dict:
    nested = path.parent.name in ("calls", "weights", "y")
    return {"path": f"{path.parent.name}/{path.name}" if nested else path.name, "bytes": n}


def _write(path: Path, data: bytes, digest: bool = True) -> dict:
    """A transcript file.  Row files carry no digest: the committed roots bind them, and the verifier checks every opened row
    against them."""
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        f.write(data)
    out = _entry(path, len(data))
    if digest:
        out["sha256"] = hashlib.sha256(data).hexdigest()
    return out


class _Ring:
    """Pinned host bytes handed out in order and given back in any order: a region is free again once it and every region
    handed out before it are given back."""

    def __init__(self, n: int):
        import torch
        self.buf, self.n, self.head, self.live = torch.empty(n, dtype=torch.uint8, pin_memory=True), n, 0, deque()

    def take(self, size: int) -> tuple[list, Any] | None:
        if not self.live:
            self.head = 0
        tail = self.live[0][0] if self.live else 0
        if self.live and self.head <= tail:                   # wrapped: free from head to tail
            if tail - self.head <= size:
                return None
        elif self.n - self.head < size:
            if tail <= size:
                return None
            self.live.append([self.head, self.n, True])       # the end past head, skipped
            self.head = 0
        region = [self.head, self.head + size, False]
        self.live.append(region)
        self.head += size
        return region, self.buf[region[0]:region[1]]

    def give(self, region: list) -> None:
        region[2] = True
        while self.live and self.live[0][2]:
            self.live.popleft()


class _Arena:
    """The pinned host bytes a device tensor's copy lands in on its way to its file: `ARENA_RINGS` rings of `RING_BYTES`,
    made at the first take; a copy past half a ring gets pinned memory of its own.  A take waits for a give where no ring
    has room."""

    def __init__(self, ring_bytes: int, rings: int):
        self.ring_bytes, self.count, self.rings, self.cv = ring_bytes, rings, [], threading.Condition()

    def take(self, size: int) -> tuple[Callable[[], None], Any]:
        import torch
        if size > self.ring_bytes // 2:
            return (lambda: None), torch.empty(size, dtype=torch.uint8, pin_memory=True)
        size = -(-size // 512) * 512
        with self.cv:
            if not self.rings:
                self.rings = [_Ring(self.ring_bytes) for _ in range(self.count)]
            while True:
                for ring in self.rings:
                    got = ring.take(size)
                    if got is not None:
                        return (lambda: self._give(ring, got[0])), got[1]
                self.cv.wait()

    def _give(self, ring: _Ring, region: list) -> None:
        with self.cv:
            ring.give(region)
            self.cv.notify_all()


class _Copies:
    """A device's copies to the host behind its current stream: a stream of their own, which waits for the current stream's
    work so far, and the arena they land in."""

    def __init__(self, device: Any):
        import torch
        self.stream, self.arena = torch.cuda.Stream(device=device), _Arena(RING_BYTES, ARENA_RINGS)

    def start(self, srcs: list[Any], fresh: bool = False, here: bool = False) -> tuple[list[Any], list[Callable[[], None]], Any]:
        """Each source's bytes as they are now on the current stream, copied to host memory behind it: (the host tensors,
        each one's give-back, the event after the copies).  `fresh`: nothing writes the sources after this, so they are
        read as they are, not copied on the device first.  `here`: the copies go on the current stream itself, in its
        order (a stream already behind the caller's, `PassEnds._retained`'s)."""
        import torch
        current = torch.cuda.current_stream(srcs[0].device)
        kept = srcs if fresh else [s.clone() for s in srcs]
        if here:
            return self._copied(kept, current, False)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            return self._copied(kept, self.stream, True)

    def _copied(self, kept: list[Any], stream: Any, held: bool) -> tuple[list[Any], list[Callable[[], None]], Any]:
        import torch
        hosts, gives = [], []
        for src in kept:
            give, host = self.arena.take(src.numel())
            host = host[:src.numel()]
            host.copy_(src, non_blocking=True)
            if held:
                src.record_stream(stream)
            hosts.append(host)
            gives.append(give)
        done = torch.cuda.Event()
        done.record(stream)
        return hosts, gives, done


def _copies(device: Any) -> _Copies:
    key = (os.getpid(), device.index)
    if key not in _copiers:
        _copiers[key] = _Copies(device)
    return _copiers[key]


def write_behind(files: tuple[tuple, ...], pending: list[Future], fresh: bool = False, here: bool = False) -> list[dict]:
    """`_write`'s entries for tensors' bytes, each file (path, tensor, digest[, then, *also]), with the files written
    behind the caller: the bytes are copied as they are now on the current stream (a device's to the host on a stream of its
    own, through `_Copies`' arena, with no wait on the caller's), and one of `WRITE_THREADS` threads writes the files in
    order, its future appended to `pending`, which the pass's end joins (`joined`) before its manifest names them; a
    digest's ``sha256`` lands in its entry with its write, and a file's ``then`` is called with its bytes and those of its
    `also` tensors (uint8 host tensors, copied with it) once they are written.  `fresh` and `here`: `_Copies.start`'s."""
    import torch
    files = [(path, t, digest, (more or (None,))[0], tuple(more[1:])) for path, t, digest, *more in files]
    srcs = [s.contiguous().reshape(-1).view(torch.uint8) for _, t, _, _, also in files for s in (t, *also)]
    if srcs and srcs[0].is_cuda:
        hosts, gives, done = _copies(srcs[0].device).start(srcs, fresh, here)
    else:
        hosts, gives, done = [s.clone() for s in srcs], [], None
    global _writers
    if _writers is None or _writers[0] != os.getpid():    # a forked child has none of its parent's threads
        _writers = (os.getpid(), ThreadPoolExecutor(WRITE_THREADS, thread_name_prefix="pearl-c-retain"))
    entries, batch, at = [], [], 0
    for path, _, digest, then, also in files:
        k = 1 + len(also)
        entry = _entry(path, hosts[at].numel())
        batch.append((path, hosts[at:at + k], digest, entry, then))
        entries.append(entry)
        at += k
    if batch:
        pending.append(_writers[1].submit(_landed, batch, done, gives))
    return entries


def _landed(batch: list[tuple], done: Any, gives: list[Callable[[], None]]) -> None:
    try:
        if done is not None:
            done.synchronize()
        for path, hosts, digest, entry, then in batch:
            _write_host(path, hosts[0], digest, entry)
            if then is not None:
                then(*hosts)
    finally:
        for give in gives:
            give()


def _write_host(path: Path, host: Any, digest: bool, entry: dict) -> None:
    data = memoryview(host.numpy())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        left = data
        while left:
            left = left[os.write(fd, left):]
    finally:
        os.close(fd)
    if digest:
        entry["sha256"] = hashlib.sha256(data).hexdigest()


def behind(fn: Callable[[], None]) -> Future:
    """`fn` on the thread that ends retained passes behind their steps, one at a time in the order they were given."""
    global _finisher
    if _finisher is None or _finisher[0] != os.getpid():
        _finisher = (os.getpid(), ThreadPoolExecutor(1, thread_name_prefix="pearl-c-retain-end"))
    return _finisher[1].submit(fn)


def joined(pending: list[Future]) -> None:
    """`write_behind`'s writes in `pending` done, and `pending` cleared; the first failed write's error is raised."""
    try:
        for f in pending:
            f.result()
    finally:
        pending.clear()


def _linked(path: Path, kept: dict[str, tuple[Path, dict]], data: Callable[[], bytes], digest: bool = True) -> dict:
    """`_write`'s file for bytes that don't change within an install (a weight's rows and tree): a hard link to the copy an
    earlier retained pass wrote (`kept`, by file name), so every pass's directory holds what it opens without writing it
    again; written (and kept) where there is none or the link fails (another file system, the copy deleted)."""
    first = kept.get(path.name)
    if first is not None and first[0] != path:
        try:
            path.unlink(missing_ok=True)
            os.link(first[0], path)
            return dict(first[1])
        except OSError:
            pass
    out = _write(path, data(), digest)
    kept[path.name] = (path, out)
    return out


def write_manifest(d: Path, head: dict, weights: dict[str, dict], matmuls: list[dict]) -> Path:
    """The pass's manifest, which `verify_run.py` reads: `head` (the scheme and its arguments, the epoch's coin commitment
    and its salt coin's opening, the salt, the live session that gave it if one did (`pouw_pearl_c_manifest.session_head`),
    the records root, the commitment, the leaves and the tile tree), every weight (n, k, root, excluded B rows, its rows and
    tree levels) and every matmul in order (index, m, k, n, weight, real rows, |K|, excluded A rows, A's root, rows and tree
    levels).  Trees are the device's layout: level 0 padded to a power of two, then each level above it, 64 bytes a node; the
    head also holds the prover's hm96 salt key.  It appears whole or not at all."""
    path, part = d / "manifest.json", d / "manifest.json.part"
    _write(part, (json.dumps({**head, "weights": weights, "matmuls": matmuls}, indent=1) + "\n").encode(), digest=False)
    os.replace(part, path)
    return path
