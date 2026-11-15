"""The prover's side of the streamed wipe on a GPU: the fill kept in device memory as it arrives, and each timed spot read
answered from it (`StoreProver`), so the wipe covers the memory the fill occupied (the tier `gpu-hbm`).

`Cuda(ordinal)` is one GPU through the CUDA driver API (`libcuda.so.1` through ctypes: no toolkit and no build on the
node): one allocation for the whole fill, a pinned staging buffer the connection is read into, a synchronous copy to the
device a piece, and a synchronous 64-byte copy back a read. `HostMemory()` is its stand-in on a CPU node, the same calls
on host memory (tier `host-dram`). Neither derives a label: the prover never holds the labels' key, so what it keeps is a
copy of what arrived.

`Disk(targets)` is a server's disks (tier `disk`): a fill file of its own in each target directory, or a block device
written raw, every write and read `O_DIRECT`, the tier declaring the whole disks under the targets. `Joint(parts)` lays
one fill over several stores in order, so that one session covers every tier: a tier wiped in a session of its own
leaves every other tier in that session's `R` (`Pous.Guarantees.StreamedWipeMeetsDisk`).

`Cuda.info()`'s `total_bytes`, the device's size the wipe declares, is the larger of CUDA's total (`cuDeviceTotalMem`,
`cuda_total_bytes`) and NVML's for the GPU at the PCI bus id CUDA reports (`nvml_total`, `nvml_total_bytes`), and
`total_source` says which. NVML's counts the memory the driver reserves, which CUDA's leaves out (615 MiB on an H200):
the driver is the prover's, so that memory is too. Without NVML it is CUDA's, `nvml_error` says why, and the wipe's
record names the reserve uncounted (`wipe_period.uncertified`).

`StoreProver(params, memory)` allocates the fill, `gate()`s the store before the fill (a distinct block written at each of
`gate_indices`, the first, middle and last labels and both sides of every power-of-two byte boundary from 32 MiB up, then
each read back; a mismatch raises, so a wrapped or truncated address shows before the fill, not as a rejected session),
keeps each frame (`receive_into`, read from the connection straight into staging), `sync()`s the store once the fill is
in (a disk's writes on the media) before the fill is acknowledged, answers each read (`answer`), and `record()`s its
timings: the fill from its first frame to its last, the copies' share, the sync, and each read's device latency.
With `foreign=n` it is the cheat `wipe_period.Keeping` is on host memory: the first `n` labels' places keep prior content
of its own, so a wipe of it must reject once `n` labels exceed the published `foreign_bits` (`wipe_period.certify`).
"""

from __future__ import annotations

import collections
import ctypes
import errno
import hashlib
import math
import os
import queue
import socket
import stat
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from . import streamed_wipe as sw

LABEL = sw.LABEL_BYTES
STAGING = 1 << 20
"""Bytes of pinned staging: one fill frame of `wipe_party.CHUNK` labels."""


def percentiles(xs: Sequence[int]) -> dict[str, int]:
    """`{"n", "p50_ns", "p99_ns", "max_ns", "mean_ns"}` of the readings `xs` (nearest rank)."""
    s = sorted(xs)
    if not s:
        return {"n": 0}

    def rank(q: float) -> int:
        return s[max(1, math.ceil(q * len(s))) - 1]

    return {"n": len(s), "p50_ns": rank(0.5), "p99_ns": rank(0.99), "max_ns": s[-1], "mean_ns": round(sum(s) / len(s))}


def gbps(n: int, ns: int) -> float | None:
    """`n` bytes over `ns` nanoseconds, in GB/s."""
    return round(n / ns, 3) if ns > 0 else None


class Memory(Protocol):
    tier: str

    def info(self) -> dict[str, Any]: ...
    def mem_info(self) -> tuple[int, int]: ...
    def alloc(self, n: int) -> int: ...
    def free(self, addr: int) -> None: ...
    def pinned(self, n: int) -> tuple[int, memoryview]: ...
    def free_pinned(self, addr: int) -> None: ...
    def htod(self, dst: int, src: int, n: int) -> None: ...
    def dtoh(self, dst: int, src: int, n: int) -> None: ...
    def close(self) -> None: ...


class HostMemory:
    """The device's stand-in on a CPU node: the same calls on host memory. With `whole`, the host's DRAM as a tier of
    its own, declared at all its physical memory (`info`'s `total_bytes`), so that what the fill doesn't reach counts in
    `R`."""

    tier = "host-dram"

    def __init__(self, whole: bool = False) -> None:
        self.whole = whole
        self._held: dict[int, ctypes.Array[ctypes.c_char]] = {}

    def info(self) -> dict[str, Any]:
        out = {"backend": "host", "host": socket.gethostname(), "tier": self.tier}
        return out | {"total_bytes": self.mem_info()[1]} if self.whole else out

    def mem_info(self) -> tuple[int, int]:
        """`(free, total)` bytes."""
        page = os.sysconf("SC_PAGE_SIZE")
        return os.sysconf("SC_AVPHYS_PAGES") * page, os.sysconf("SC_PHYS_PAGES") * page

    def alloc(self, n: int) -> int:
        buf = (ctypes.c_char * n)()
        self._held[ctypes.addressof(buf)] = buf
        return ctypes.addressof(buf)

    def free(self, addr: int) -> None:
        self._held.pop(addr, None)

    def pinned(self, n: int) -> tuple[int, memoryview]:
        addr = self.alloc(n)
        return addr, memoryview(self._held[addr]).cast("B")

    free_pinned = free

    def htod(self, dst: int, src: int, n: int) -> None:
        ctypes.memmove(dst, src, n)

    def dtoh(self, dst: int, src: int, n: int) -> None:
        ctypes.memmove(dst, src, n)

    def close(self) -> None:
        self._held.clear()


_I, _S, _U, _P = ctypes.c_int, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_void_p
_SIGNATURES = {
    "cuInit": (ctypes.c_uint,),
    "cuDriverGetVersion": (ctypes.POINTER(_I),),
    "cuDeviceGet": (ctypes.POINTER(_I), _I),
    "cuDeviceGetName": (ctypes.c_char_p, _I, _I),
    "cuDeviceGetUuid": (ctypes.c_char_p, _I),
    "cuDeviceGetAttribute": (ctypes.POINTER(_I), _I, _I),
    "cuDeviceGetPCIBusId": (ctypes.c_char_p, _I, _I),
    "cuDeviceTotalMem_v2": (ctypes.POINTER(_S), _I),
    "cuDevicePrimaryCtxRetain": (ctypes.POINTER(_P), _I),
    "cuDevicePrimaryCtxRelease_v2": (_I,),
    "cuCtxSetCurrent": (_P,),
    "cuMemGetInfo_v2": (ctypes.POINTER(_S), ctypes.POINTER(_S)),
    "cuMemAlloc_v2": (ctypes.POINTER(_U), _S),
    "cuMemFree_v2": (_U,),
    "cuMemHostAlloc": (ctypes.POINTER(_P), _S, ctypes.c_uint),
    "cuMemFreeHost": (_P,),
    "cuMemcpyHtoD_v2": (_U, _P, _S),
    "cuMemcpyDtoH_v2": (_P, _U, _S),
    "cuGetErrorName": (_I, ctypes.POINTER(ctypes.c_char_p)),
}
COMPUTE_CAPABILITY = (75, 76)
"""`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` and `_MINOR`."""
NVML = "libnvidia-ml.so.1"
load: Callable[[str], Any] = ctypes.CDLL
"""Opens a driver library by name, raising `OSError` when the host hasn't it."""


class NvmlMemory(ctypes.Structure):
    """`nvmlMemory_t`, in bytes."""

    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


_NVML_SIGNATURES = {
    "nvmlInit_v2": ((), _I),
    "nvmlShutdown": ((), _I),
    "nvmlErrorString": ((_I,), ctypes.c_char_p),
    "nvmlDeviceGetHandleByPciBusId_v2": ((ctypes.c_char_p, ctypes.POINTER(_P)), _I),
    "nvmlDeviceGetMemoryInfo": ((_P, ctypes.POINTER(NvmlMemory)), _I),
}


def nvml_total(pci_bus_id: str, library: str = NVML) -> int:
    """NVML's total memory of the GPU at `pci_bus_id` (`nvmlDeviceGetMemoryInfo`), the driver's reserve included. A
    host without NVML, or an NVML without that GPU, raises `RuntimeError`."""
    try:
        lib = load(library)
        for name, (args, res) in _NVML_SIGNATURES.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, res
    except (OSError, AttributeError) as e:
        raise RuntimeError(f"no NVML on this host ({library}): {e}") from e

    def call(name: str, *args: Any) -> None:
        if r := getattr(lib, name)(*args):
            raise RuntimeError(f"{name} failed for the GPU at {pci_bus_id}: "
                               f"{(lib.nvmlErrorString(r) or b'NVML_ERROR').decode()} ({r})")

    call("nvmlInit_v2")
    try:
        handle, mem = _P(), NvmlMemory()
        call("nvmlDeviceGetHandleByPciBusId_v2", pci_bus_id.encode(), ctypes.byref(handle))
        call("nvmlDeviceGetMemoryInfo", handle, ctypes.byref(mem))
    finally:
        lib.nvmlShutdown()
    return mem.total


class Cuda:
    """GPU `ordinal` (of those `CUDA_VISIBLE_DEVICES` shows) through the CUDA driver API, its primary context current
    on the calling thread, sized with `nvml` as well (module docstring). Every copy is synchronous: a pinned buffer's
    copy returns once the bytes have landed."""

    tier = "gpu-hbm"

    def __init__(self, ordinal: int = 0, library: str = "libcuda.so.1", nvml: str = NVML) -> None:
        try:
            self.lib = load(library)
        except OSError as e:
            raise RuntimeError(f"no CUDA driver on this host ({library}): {e}") from e
        for name, args in _SIGNATURES.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, _I
        self.ordinal, self.nvml = ordinal, nvml
        self._call("cuInit", 0)
        dev = _I()
        self._call("cuDeviceGet", ctypes.byref(dev), ordinal)
        self.dev = dev.value
        ctx = _P()
        self._call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), self.dev)
        self._call("cuCtxSetCurrent", ctx)

    def _call(self, name: str, *args: Any) -> None:
        r = getattr(self.lib, name)(*args)
        if r:
            s = ctypes.c_char_p()
            self.lib.cuGetErrorName(r, ctypes.byref(s))
            raise RuntimeError(f"{name} failed: {(s.value or b'CUDA_ERROR').decode()} ({r})")

    def info(self) -> dict[str, Any]:
        name, pci, uuid = ctypes.create_string_buffer(256), ctypes.create_string_buffer(64), ctypes.create_string_buffer(16)
        self._call("cuDeviceGetName", name, 256, self.dev)
        self._call("cuDeviceGetPCIBusId", pci, 64, self.dev)
        self._call("cuDeviceGetUuid", uuid, self.dev)
        cc = []
        for attr in COMPUTE_CAPABILITY:
            v = _I()
            self._call("cuDeviceGetAttribute", ctypes.byref(v), attr, self.dev)
            cc.append(v.value)
        total, driver = _S(), _I()
        self._call("cuDeviceTotalMem_v2", ctypes.byref(total), self.dev)
        self._call("cuDriverGetVersion", ctypes.byref(driver))
        u, bus = uuid.raw.hex(), pci.value.decode()
        try:
            nvml, why = nvml_total(bus, self.nvml), None
        except RuntimeError as e:
            nvml, why = None, str(e)
        out = {"backend": "cuda", "compute_capability": f"{cc[0]}.{cc[1]}", "cuda_total_bytes": total.value,
               "device": name.value.decode(), "driver": driver.value, "host": socket.gethostname(),
               "nvml_total_bytes": nvml, "ordinal": self.ordinal, "pci_bus_id": bus, "tier": self.tier,
               "total_bytes": max(total.value, nvml or 0),
               "total_source": "nvml" if nvml is not None and nvml >= total.value else "cuda",
               "uuid": f"GPU-{u[:8]}-{u[8:12]}-{u[12:16]}-{u[16:20]}-{u[20:]}"}
        return out if why is None else out | {"nvml_error": why}

    def mem_info(self) -> tuple[int, int]:
        """`(free, total)` bytes of the device."""
        free, total = _S(), _S()
        self._call("cuMemGetInfo_v2", ctypes.byref(free), ctypes.byref(total))
        return free.value, total.value

    def alloc(self, n: int) -> int:
        p = _U()
        self._call("cuMemAlloc_v2", ctypes.byref(p), n)
        return p.value

    def free(self, addr: int) -> None:
        self._call("cuMemFree_v2", addr)

    def pinned(self, n: int) -> tuple[int, memoryview]:
        p = _P()
        self._call("cuMemHostAlloc", ctypes.byref(p), n, 0)
        return p.value, memoryview((ctypes.c_char * n).from_address(p.value)).cast("B")

    def free_pinned(self, addr: int) -> None:
        self._call("cuMemFreeHost", addr)

    def htod(self, dst: int, src: int, n: int) -> None:
        self._call("cuMemcpyHtoD_v2", dst, src, n)

    def dtoh(self, dst: int, src: int, n: int) -> None:
        self._call("cuMemcpyDtoH_v2", dst, src, n)

    def close(self) -> None:
        self._call("cuDevicePrimaryCtxRelease_v2", self.dev)


BLOCK = 4096
"""The alignment of every disk write and read: `O_DIRECT` takes whole logical blocks, at most 4 KiB on any drive."""
DEPTH = 16
"""Disk writes of up to `STAGING` bytes in flight at once."""
DISK_RESERVE = 10 << 30
"""The free space a fill file leaves its filesystem."""
FILL_FILE = "verity-wipe-fill-{pid}-{i}.bin"
VIRTUAL_DRIVERS = ("virtio_blk", "xen_blkfront", "vmw_pvscsi")
"""Drivers of a virtual disk, whose backing store is the hypervisor's."""


def whole_disks(dev: int) -> dict[str, int]:
    """The whole disks under the block device `dev` (an `st_dev` or `st_rdev`), by name with their size in bytes: a
    partition's disk, and each member's of a device-mapper or md device (its `slaves`); `{}` for a filesystem on no block
    device (tmpfs, overlay)."""
    out: dict[str, int] = {}

    def walk(p: Path) -> None:
        members = sorted((p / "slaves").iterdir()) if (p / "slaves").is_dir() else []
        for m in members:
            walk(m.resolve())
        if not members:
            disk = p.parent if (p / "partition").exists() else p
            out[disk.name] = int((disk / "size").read_text()) * 512

    p = Path(f"/sys/dev/block/{os.major(dev)}:{os.minor(dev)}")
    if p.exists():
        walk(p.resolve())
    return out


def driver(disk: str) -> str | None:
    """The kernel driver of the whole disk `disk` (`nvme`, `virtio_blk`, `sd`), or None when sysfs doesn't say."""
    p = Path(f"/sys/block/{disk}/device/driver")
    if p.exists():
        return p.resolve().name
    p = Path(f"/sys/block/{disk}/device/device/driver")
    return p.resolve().name if p.exists() else None


class _Aligned:
    """Host buffers aligned to `BLOCK`, as `O_DIRECT` needs, kept by address."""

    def __init__(self) -> None:
        self._held: dict[int, ctypes.Array[ctypes.c_char]] = {}

    def get(self, n: int) -> int:
        raw = (ctypes.c_char * (n + BLOCK))()
        addr = -(-ctypes.addressof(raw) // BLOCK) * BLOCK
        self._held[addr] = raw
        return addr

    def put(self, addr: int) -> None:
        self._held.pop(addr, None)


def _view(addr: int, n: int) -> memoryview:
    return memoryview((ctypes.c_char * n).from_address(addr)).cast("B")


class Disk:
    """A server's disks as the store, tier `disk`. Each target is a directory, where the fill takes a file of its own
    (created for the session with `O_EXCL`, never an existing file, allocated whole up front, removed at `close`), or a
    block device written raw (opened `O_EXCL`, so a device that is mounted or held is refused). The fill's room is each
    directory's free space less `reserve_bytes` (the reserved blocks counted only for root) and each device whole.

    Every write and read is `O_DIRECT` from a `BLOCK`-aligned buffer, so the labels reach the drive and the page cache
    keeps no copy of them; `direct=None` falls back to buffered I/O where the filesystem refuses `O_DIRECT` (tmpfs), as
    `info` records. A write copies its bytes out of the caller's buffer and is in flight `depth` at a time on a thread
    each; one that isn't whole blocks is read, patched and written back once the writes before it land. `sync` waits for
    every write and `fdatasync`s each target, and `StoreProver.sync` calls it before the fill is acknowledged.

    The tier declares the whole disks under its targets (`whole_disks`), every partition and filesystem on them, plus
    `spare_bytes`, the drives' spare area and cache from their datasheets, which no write through the block interface
    reaches: what the fill doesn't overwrite counts in `R`. A target on no block device declares its filesystem."""

    tier = "disk"

    def __init__(self, targets: Sequence[str | os.PathLike[str]], *, reserve_bytes: int = DISK_RESERVE,
                 spare_bytes: int = 0, direct: bool | None = True, depth: int = DEPTH) -> None:
        if not targets:
            raise ValueError("a disk store has at least one target, a directory or a block device")
        self.reserve, self.spare, self.direct, self.depth = reserve_bytes, spare_bytes, direct, depth
        self.targets: list[dict[str, Any]] = []
        self.disks: dict[str, int] = {}
        filesystems: dict[int, int] = {}
        self._fds: list[int] = []
        self._made: list[Path] = []
        seen: set[int] = set()
        try:
            for t in map(Path, targets):
                st = os.stat(t)
                if (st.st_rdev if stat.S_ISBLK(st.st_mode) else st.st_dev) in seen:
                    raise ValueError(f"disk target {t} is on a device another target is on: one target a filesystem")
                seen.add(st.st_rdev if stat.S_ISBLK(st.st_mode) else st.st_dev)
                if stat.S_ISBLK(st.st_mode):
                    fd = os.open(t, os.O_RDWR | os.O_EXCL | self._direct_flag(direct))
                    self._fds.append(fd)
                    room = os.lseek(fd, 0, os.SEEK_END) // BLOCK * BLOCK
                    self.targets.append({"path": str(t), "kind": "raw", "room": room, "fd": fd})
                    under = whole_disks(st.st_rdev)
                elif stat.S_ISDIR(st.st_mode):
                    fs = os.statvfs(t)
                    free = (fs.f_bfree if os.geteuid() == 0 else fs.f_bavail) * fs.f_frsize
                    room = max(0, free - reserve_bytes) // BLOCK * BLOCK
                    self.targets.append({"path": str(t), "kind": "file", "room": room, "fd": None})
                    under = whole_disks(st.st_dev)
                    if not under:
                        filesystems[st.st_dev] = fs.f_blocks * fs.f_frsize
                else:
                    raise ValueError(f"disk target {t} is neither a directory nor a block device")
                self.disks |= under
        except BaseException:
            self._close_fds()
            raise
        self.total = sum(self.disks.values()) + sum(filesystems.values()) + spare_bytes
        self._buffers = _Aligned()
        self._free: queue.SimpleQueue[int] = queue.SimpleQueue()
        for _ in range(depth):
            self._free.put(self._buffers.get(STAGING))
        self._scratch = self._buffers.get(STAGING + BLOCK)
        self._pool = ThreadPoolExecutor(depth, thread_name_prefix="wipe-disk")
        self._inflight: collections.deque[Future[None]] = collections.deque()
        self._spans: list[tuple[int, int, int]] = []

    @staticmethod
    def _direct_flag(direct: bool | None) -> int:
        return getattr(os, "O_DIRECT", 0) if direct is not False else 0

    def info(self) -> dict[str, Any]:
        return {"backend": "disk", "host": socket.gethostname(), "tier": self.tier, "direct": self.direct,
                "targets": [{k: t[k] for k in ("path", "kind", "room", "file", "bytes") if k in t} for t in self.targets],
                "disks": {d: {"bytes": b, "driver": driver(d)} for d, b in sorted(self.disks.items())},
                "spare_bytes": self.spare, "reserve_bytes": self.reserve, "total_bytes": self.total}

    def mem_info(self) -> tuple[int, int]:
        """`(room, total)`: the bytes the fill can take, and the tier's declared bytes."""
        return sum(t["room"] for t in self.targets), self.total

    def _open_file(self, t: dict[str, Any], i: int) -> int:
        path = Path(t["path"]) / FILL_FILE.format(pid=os.getpid(), i=i)
        flags, direct = os.O_RDWR | os.O_CREAT | os.O_EXCL, self._direct_flag(self.direct)
        try:
            fd = os.open(path, flags | direct, 0o600)
        except OSError as e:
            if e.errno != errno.EINVAL or not direct:
                raise
            if self.direct is not None:
                raise RuntimeError(f"{t['path']} refuses O_DIRECT ({e}); direct=None falls back to buffered I/O") from e
            fd, self.direct = os.open(path, flags, 0o600), False
        self._made.append(path)
        t["file"] = str(path)
        return fd

    def alloc(self, n: int) -> int:
        """Lays the fill's `n` bytes over the targets in order, each target's share whole blocks: a fill file is created
        and allocated whole, so a full filesystem refuses it here rather than in the fill. Returns 0, the store's first
        byte."""
        if self._spans:
            raise RuntimeError("a disk store keeps one fill")
        if n <= 0 or n % BLOCK:
            raise ValueError(f"a disk fill is whole {BLOCK}-byte blocks, not {n} bytes")
        if n > (room := self.mem_info()[0]):
            raise RuntimeError(f"the disk targets have room for {room} bytes, less than the fill's {n}")
        at = 0
        for i, t in enumerate(self.targets):
            share = min(n - at, t["room"])
            if share <= 0:
                break
            if t["kind"] == "file":
                t["fd"] = fd = self._open_file(t, i)
                self._fds.append(fd)
                os.posix_fallocate(fd, 0, share)
            t["bytes"] = share
            self._spans.append((at, share, t["fd"]))
            at += share
        if self.direct is None:
            self.direct = True
        return 0

    def free(self, addr: int) -> None:
        pass

    def pinned(self, n: int) -> tuple[int, memoryview]:
        addr = self._buffers.get(n)
        return addr, _view(addr, n)

    def free_pinned(self, addr: int) -> None:
        self._buffers.put(addr)

    def _pieces(self, addr: int, n: int) -> list[tuple[int, int, int, int]]:
        """`(fd, offset in the target, offset in the caller's buffer, length)` for the store's bytes `[addr, addr + n)`."""
        out, end = [], addr + n
        for at, size, fd in self._spans:
            lo, hi = max(addr, at), min(end, at + size)
            if lo < hi:
                out.append((fd, lo - at, lo - addr, hi - lo))
        if sum(p[3] for p in out) != n:
            raise ValueError(f"{n} bytes at {addr} are outside the disk fill")
        return out

    def _prune(self, limit: int) -> None:
        while self._inflight and (len(self._inflight) > limit or self._inflight[0].done()):
            self._inflight.popleft().result()

    def _drain(self) -> None:
        self._prune(0)

    def _write(self, fd: int, buf: int, n: int, off: int) -> None:
        try:
            view, done = _view(buf, n), 0
            while done < n:
                done += os.pwrite(fd, view[done:], off + done)
        finally:
            self._free.put(buf)

    def _patch(self, fd: int, off: int, src: int | None, dst: int | None, n: int) -> None:
        """Bytes `[off, off + n)` of a target through whole blocks: read, then copied out to `dst`, or patched from
        `src` and written back."""
        self._drain()
        done = 0
        while done < n:
            lo = (off + done) // BLOCK * BLOCK
            m = min(n - done, STAGING - (off + done - lo))
            span = -(-(off + done + m - lo) // BLOCK) * BLOCK
            view, got = _view(self._scratch, span), 0
            while got < span:
                r = os.preadv(fd, [view[got:]], lo + got)
                if not r:
                    raise RuntimeError(f"a disk target ended at byte {lo + got}, inside the fill")
                got += r
            if src is None:
                ctypes.memmove(dst + done, self._scratch + off + done - lo, m)
            else:
                ctypes.memmove(self._scratch + off + done - lo, src + done, m)
                put = 0
                while put < span:
                    put += os.pwrite(fd, view[put:], lo + put)
            done += m

    def htod(self, dst: int, src: int, n: int) -> None:
        for fd, off, at, m in self._pieces(dst, n):
            if off % BLOCK or m % BLOCK:
                self._patch(fd, off, src + at, None, m)
                continue
            for k in range(0, m, STAGING):
                piece = min(STAGING, m - k)
                buf = self._free.get()
                ctypes.memmove(buf, src + at + k, piece)
                self._inflight.append(self._pool.submit(self._write, fd, buf, piece, off + k))
                self._prune(4 * self.depth)

    def dtoh(self, dst: int, src: int, n: int) -> None:
        for fd, off, at, m in self._pieces(src, n):
            self._patch(fd, off, None, dst + at, m)

    def sync(self) -> None:
        """Every write landed and each target `fdatasync`ed: the fill on the media."""
        self._drain()
        for _, _, fd in self._spans:
            os.fdatasync(fd)

    def _close_fds(self) -> None:
        for fd in self._fds:
            os.close(fd)
        self._fds = []

    def close(self) -> None:
        """Waits for the writes, closes the targets and removes the fill files this store made."""
        try:
            self._drain()
        finally:
            self._pool.shutdown(wait=True)
            self._close_fds()
            for p in self._made:
                p.unlink(missing_ok=True)
            self._made = []


class Joint:
    """Stores in one address space with the fill laid over them in order, `parts` each a store and the bytes of the fill
    it keeps: one session's labels over every tier, so that what any tier keeps of the prior content counts against the
    one bound. A tier wiped in a session of its own leaves every other tier in that session's `R`, and two sessions don't
    compose. Staging and reads go through the first part that isn't a `Disk` (a GPU's pinned memory) when there is
    one."""

    def __init__(self, parts: Sequence[tuple[Memory, int]]) -> None:
        if not parts or any(b <= 0 for _, b in parts):
            raise ValueError("a joint store has parts, each keeping some of the fill")
        self.parts = list(parts)
        self.tier = "+".join(m.tier for m, _ in self.parts)
        self.bases: list[int] = []
        self._host = next((m for m, _ in self.parts if not isinstance(m, Disk)), self.parts[0][0])

    def declared(self) -> tuple[tuple[str, int], ...]:
        """Each part's tier and declared bytes: its `info()`'s `total_bytes`, else the fill it keeps."""
        return tuple((m.tier, m.info().get("total_bytes") or b) for m, b in self.parts)

    def info(self) -> dict[str, Any]:
        return {"backend": "joint", "host": socket.gethostname(), "tier": self.tier,
                "parts": [m.info() | {"bytes": b} for m, b in self.parts],
                "total_bytes": sum(b for _, b in self.declared())}

    def mem_info(self) -> tuple[int, int]:
        got = [m.mem_info() for m, _ in self.parts]
        return sum(f for f, _ in got), sum(t for _, t in got)

    def alloc(self, n: int) -> int:
        if n != sum(b for _, b in self.parts):
            raise ValueError(f"the joint store keeps {sum(b for _, b in self.parts)} bytes of fill, not {n}")
        self.bases = [m.alloc(b) for m, b in self.parts]
        return 0

    def _pieces(self, addr: int, n: int) -> list[tuple[Memory, int, int, int]]:
        """`(part, its address, offset in the caller's buffer, length)` for the store's bytes `[addr, addr + n)`."""
        out, at, end = [], 0, addr + n
        for (m, b), base in zip(self.parts, self.bases):
            lo, hi = max(addr, at), min(end, at + b)
            if lo < hi:
                out.append((m, base + lo - at, lo - addr, hi - lo))
            at += b
        if sum(p[3] for p in out) != n:
            raise ValueError(f"{n} bytes at {addr} are outside the joint fill")
        return out

    def free(self, addr: int) -> None:
        for (m, _), base in zip(self.parts, self.bases):
            m.free(base)

    def pinned(self, n: int) -> tuple[int, memoryview]:
        return self._host.pinned(n)

    def free_pinned(self, addr: int) -> None:
        self._host.free_pinned(addr)

    def htod(self, dst: int, src: int, n: int) -> None:
        for m, a, at, k in self._pieces(dst, n):
            m.htod(a, src + at, k)

    def dtoh(self, dst: int, src: int, n: int) -> None:
        for m, a, at, k in self._pieces(src, n):
            m.dtoh(dst + at, a, k)

    def sync(self) -> None:
        for m, _ in self.parts:
            if (s := getattr(m, "sync", None)) is not None:
                s()

    def close(self) -> None:
        for m, _ in self.parts:
            m.close()


def gate_indices(labels: int) -> list[int]:
    """The labels `StoreProver.gate` writes: the first, the middle and the last, and the labels on both sides of each
    power-of-two byte boundary from 32 MiB up (2^31 and 2^32 among them, where a 32-bit offset wraps)."""
    out = {0, labels // 2, labels - 1}
    for b in range(25, 64):
        i = (1 << b) // LABEL
        if i >= labels:
            break
        out |= {i - 1, i}
    return sorted(out)


class StoreProver:
    """The honest server keeping the fill of `params` in `memory` and answering from it, its timings on `clock`; with
    `foreign` labels, the server that keeps its own prior content in the first `foreign` labels' places instead."""

    def __init__(self, params: sw.WipeParams, memory: Memory, *, staging: int = STAGING,
                 clock: Callable[[], int] = time.perf_counter_ns, foreign: int = 0) -> None:
        if not 0 <= foreign <= params.labels:
            raise ValueError(f"foreign is {foreign} labels, outside 0 to the fill's {params.labels}")
        self.params, self.memory, self.clock, self.foreign = params, memory, clock, foreign
        self.size = params.labels * LABEL
        self.base = memory.alloc(self.size)
        self.stage_addr, self.stage = memory.pinned(staging)
        self.read_addr, self.read = memory.pinned(LABEL)
        self.prior_addr: int | None = None
        if foreign:
            self.prior_addr, self.prior = memory.pinned(staging)
            for j in range(staging // LABEL):
                self.prior[LABEL * j:LABEL * (j + 1)] = hashlib.sha512(b"prior content %d" % j).digest()
        self.gated: dict[str, Any] | None = None
        self.frames = self.filled = self.copy_ns = 0
        self.sync_ns: int | None = None
        self.first_ns: int | None = None
        self.last_ns: int | None = None
        self.reads: list[int] = []

    def gate(self) -> dict[str, Any]:
        """Every `gate_indices` label written with a block of its own, then each read back; a mismatch raises."""
        idx = gate_indices(self.params.labels)
        block = {i: hashlib.sha512(b"verity-pous/wipe-store/gate/%d" % i).digest() for i in idx}
        for i in idx:
            self.stage[:LABEL] = block[i]
            self.memory.htod(self.base + LABEL * i, self.stage_addr, LABEL)
        bad = []
        for i in idx:
            self.memory.dtoh(self.read_addr, self.base + LABEL * i, LABEL)
            if bytes(self.read[:LABEL]) != block[i]:
                bad.append(i)
        if bad:
            raise RuntimeError(f"the store doesn't keep labels {bad[:8]} of {len(idx)} gated: refused before the fill")
        self.gated = {"labels": len(idx), "highest": idx[-1], "passed": True}
        return self.gated

    def _keep(self, start: int, n: int, take: Callable[[memoryview], None]) -> None:
        if start < 0 or LABEL * start + n > self.size:
            raise ValueError(f"a frame of {n} bytes at label {start} is outside the fill of {self.params.labels}")
        t = self.clock()
        if self.first_ns is None:
            self.first_ns = t
        done = 0
        while done < n:
            m = min(n - done, len(self.stage))
            take(self.stage[:m])
            dst = self.base + LABEL * start + done
            prior = min(m, max(0, LABEL * self.foreign - LABEL * start - done))
            c = self.clock()
            if prior:
                self.memory.htod(dst, self.prior_addr, prior)
            if prior < m:
                self.memory.htod(dst + prior, self.stage_addr + prior, m - prior)
            self.copy_ns += self.clock() - c
            done += m
        self.frames, self.filled, self.last_ns = self.frames + 1, self.filled + n, self.clock()

    def receive(self, start: int, data: bytes) -> None:
        view, at = memoryview(data), 0

        def take(stage: memoryview) -> None:
            nonlocal at
            stage[:] = view[at:at + len(stage)]
            at += len(stage)

        self._keep(start, len(data), take)

    def receive_into(self, start: int, count: int, f: BinaryIO) -> None:
        """A fill frame's `count` labels from `f` straight into staging, then to the store."""

        def take(stage: memoryview) -> None:
            got = 0
            while got < len(stage):
                r = f.readinto(stage[got:])
                if not r:
                    raise ConnectionError(f"the challenger closed in a fill frame at label {start}")
                got += r

        self._keep(start, LABEL * count, take)

    def sync(self) -> None:
        """The whole fill in the store (a disk's writes landed and `fdatasync`ed), before the fill is acknowledged, so
        that no read is answered from a write still in flight or from the host's page cache."""
        t = self.clock()
        if (s := getattr(self.memory, "sync", None)) is not None:
            s()
        self.sync_ns = self.clock() - t

    def answer(self, indices: Sequence[int]) -> bytes:
        out = bytearray()
        for i in indices:
            if not 0 <= i < self.params.labels:
                raise ValueError(f"label {i} is outside the fill of {self.params.labels}")
            t = self.clock()
            self.memory.dtoh(self.read_addr, self.base + LABEL * i, LABEL)
            self.reads.append(self.clock() - t)
            out += self.read[:LABEL]
        return bytes(out)

    def record(self) -> dict[str, Any]:
        """The store, its gate, the fill (bytes, frames, from the first frame to the last, the copies' share, the sync
        after it) and each read's device latency, on this prover's clock."""
        span = 0 if self.first_ns is None or self.last_ns is None else self.last_ns - self.first_ns
        out = {"store": self.memory.info() | {"bytes": self.size}, "gate": self.gated,
               "fill": {"bytes": self.filled, "frames": self.frames, "ns": span, "GBps": gbps(self.filled, span),
                        "copy_ns": self.copy_ns, "copy_GBps": gbps(self.filled, self.copy_ns),
                        "sync_ns": self.sync_ns},
               "reads": percentiles(self.reads)}
        return out | {"foreign": {"labels": self.foreign, "bytes": LABEL * self.foreign}} if self.foreign else out

    def close(self) -> None:
        self.stage.release()
        self.read.release()
        self.memory.free_pinned(self.stage_addr)
        self.memory.free_pinned(self.read_addr)
        if self.prior_addr is not None:
            self.prior.release()
            self.memory.free_pinned(self.prior_addr)
        self.memory.free(self.base)


__all__ = ["BLOCK", "COMPUTE_CAPABILITY", "Cuda", "DEPTH", "DISK_RESERVE", "Disk", "HostMemory", "Joint", "Memory",
           "NVML", "NvmlMemory", "STAGING", "StoreProver", "VIRTUAL_DRIVERS", "driver", "gate_indices", "gbps", "load",
           "nvml_total", "percentiles", "whole_disks"]
