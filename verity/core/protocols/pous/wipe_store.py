"""The prover's side of the streamed wipe on a GPU: the fill kept in device memory as it arrives, and each timed spot read
answered from it (`StoreProver`), so the wipe covers the memory the fill occupied (the tier `gpu-hbm`).

`Cuda(ordinal)` is one GPU through the CUDA driver API (`libcuda.so.1` through ctypes: no toolkit and no build on the
node): one allocation for the whole fill, a pinned staging buffer the connection is read into, a synchronous copy to the
device a piece, and a synchronous 64-byte copy back a read. `HostMemory()` is its stand-in on a CPU node, the same calls
on host memory (tier `host-dram`). Neither derives a label: the prover never holds the labels' key, so what it keeps is a
copy of what arrived.

`StoreProver(params, memory)` allocates the fill, `gate()`s the store before the fill (a distinct block written at each of
`gate_indices`, the first, middle and last labels and both sides of every power-of-two byte boundary from 32 MiB up, then
each read back; a mismatch raises, so a wrapped or truncated address shows before the fill, not as a rejected session),
keeps each frame (`receive_into`, read from the connection straight into staging), answers each read (`answer`), and
`record()`s its timings: the fill from its first frame to its last, the copies' share, and each read's device latency.
With `foreign=n` it is the cheat `wipe_period.Keeping` is on host memory: the first `n` labels' places keep prior content
of its own, so a wipe of it must reject once `n` labels exceed `f·M`.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import socket
import time
from collections.abc import Callable, Sequence
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
    """The device's stand-in on a CPU node: the same calls on host memory."""

    tier = "host-dram"

    def __init__(self) -> None:
        self._held: dict[int, ctypes.Array[ctypes.c_char]] = {}

    def info(self) -> dict[str, Any]:
        return {"backend": "host", "host": socket.gethostname(), "tier": self.tier}

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


class Cuda:
    """GPU `ordinal` (of those `CUDA_VISIBLE_DEVICES` shows) through the CUDA driver API, its primary context current
    on the calling thread. Every copy is synchronous: a pinned buffer's copy returns once the bytes have landed."""

    tier = "gpu-hbm"

    def __init__(self, ordinal: int = 0, library: str = "libcuda.so.1") -> None:
        try:
            self.lib = ctypes.CDLL(library)
        except OSError as e:
            raise RuntimeError(f"no CUDA driver on this host ({library}): {e}") from e
        for name, args in _SIGNATURES.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, _I
        self.ordinal = ordinal
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
        u = uuid.raw.hex()
        return {"backend": "cuda", "compute_capability": f"{cc[0]}.{cc[1]}", "device": name.value.decode(),
                "driver": driver.value, "host": socket.gethostname(), "ordinal": self.ordinal,
                "pci_bus_id": pci.value.decode(), "tier": self.tier, "total_bytes": total.value,
                "uuid": f"GPU-{u[:8]}-{u[8:12]}-{u[12:16]}-{u[16:20]}-{u[20:]}"}

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
        """The store, its gate, the fill (bytes, frames, from the first frame to the last, the copies' share) and each
        read's device latency, on this prover's clock."""
        span = 0 if self.first_ns is None or self.last_ns is None else self.last_ns - self.first_ns
        out = {"store": self.memory.info() | {"bytes": self.size}, "gate": self.gated,
               "fill": {"bytes": self.filled, "frames": self.frames, "ns": span, "GBps": gbps(self.filled, span),
                        "copy_ns": self.copy_ns, "copy_GBps": gbps(self.filled, self.copy_ns)},
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


__all__ = ["COMPUTE_CAPABILITY", "Cuda", "HostMemory", "Memory", "STAGING", "StoreProver", "gate_indices", "gbps",
           "percentiles"]
