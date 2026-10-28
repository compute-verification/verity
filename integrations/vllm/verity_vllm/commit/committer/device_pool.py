"""Device-held values (serve-openai's ``--retain device``): a Commit window's committed bytes stay in a bounded device buffer
(`DevicePool`) until the window's draw-gated release, which keeps on host only the leaves some subprotocol drew (`gather`).

The windowed collector's hit steps copy each window of their stream into the pool instead of the pinned host pool, so the step's
retained block (`host_retention._HostRanges` / `_HostWindows`, marked ``on_device``) holds device views, read like the host
ones (a slice is a device tensor; the served record's readback copies it to host). The pool is one slab, carved first fit and
owned by window (the committer view the step committed into): a take that doesn't fit waits for an earlier window's release,
and that wait is the worker's, so the serving thread's next slot wait (in its step) carries it; a take that the open window's own
bytes leave no room for is refused by name, since no release can come before that window closes.

`gather` is the release's: given the drawn leaves of a step it copies exactly their bytes to host as a `DrawnRanges` and the
step's device views are dropped; a read of any other leaf of the step is refused by name (`DrawnRanges.dropped`), never answered
from a recomputation. The step's tree levels and salt key stay, so a drawn leaf's opening path is whole.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from verity_vllm.commit.committer.host_retention import _HostRanges
from verity_vllm.commit.native_ranges import DROPPED


class DevicePoolFull(RuntimeError):
    """A take the open window's own bytes leave no room for: no release can come before the window closes."""


class DevicePool:
    """A slab of ``capacity`` bytes on ``device``, carved first fit, each piece owned by a window key until `free(key)`."""

    def __init__(self, capacity: int, device: Any = "cuda", align: int = 512) -> None:
        if capacity <= 0:
            raise ValueError(f"device pool: capacity {capacity} bytes")
        self.align = int(align)
        self.capacity = int(capacity) // self.align * self.align
        self.slab = torch.empty(self.capacity, dtype=torch.uint8, device=device)
        self._free: list[tuple[int, int]] = [(0, self.capacity)]          # (offset, bytes), sorted by offset, coalesced
        self._owned: dict[Any, list[tuple[int, int]]] = {}
        self._cv = threading.Condition()
        self.waited_s = 0.0
        self.waits = 0
        self.peak = 0
        self.failure: str | None = None

    @property
    def held(self) -> int:
        return sum(n for parts in self._owned.values() for _o, n in parts)

    def held_by(self, key: Any) -> int:
        return sum(n for _o, n in self._owned.get(key, ()))

    def _fit(self, n: int) -> int | None:
        for i, (off, size) in enumerate(self._free):
            if size >= n:
                self._free[i] = (off + n, size - n)
                if size == n:
                    del self._free[i]
                return off
        return None

    def take(self, nbytes: int, key: Any) -> tuple[Any, float]:
        """A device view of ``nbytes`` owned by ``key``, and the seconds waited for an earlier window's release."""
        n = -(-int(nbytes) // self.align) * self.align
        t0 = time.perf_counter()
        waited = False
        with self._cv:
            while True:
                if self.failure:
                    raise RuntimeError(f"device pool: {self.failure}")
                off = self._fit(n)
                if off is not None:
                    break
                if self.capacity - self.held_by(key) < n or not any(k != key for k in self._owned):
                    raise DevicePoolFull(f"device pool of {self.capacity >> 20} MiB: window holds {self.held_by(key) >> 20} MiB and needs "
                                         f"{n >> 20} MiB more, which no earlier window's release can free (the window must close first): "
                                         "raise --retain-device-gb or lower --window-gb")
                waited = True
                self._cv.wait()
            self._owned.setdefault(key, []).append((off, n))
            self.peak = max(self.peak, self.held)
            dt = time.perf_counter() - t0 if waited else 0.0
            if waited:
                self.waits += 1
                self.waited_s += dt
        return self.slab[off:off + int(nbytes)], dt

    def free(self, key: Any) -> int:
        """Every piece ``key`` owns back to the slab; the bytes."""
        with self._cv:
            parts = self._owned.pop(key, [])
            merged = sorted(self._free + parts)
            self._free = []
            for off, size in merged:
                if self._free and self._free[-1][0] + self._free[-1][1] == off:
                    self._free[-1] = (self._free[-1][0], self._free[-1][1] + size)
                else:
                    self._free.append((off, size))
            self._cv.notify_all()
        return sum(n for _o, n in parts)

    def fail(self, why: str) -> None:
        """Wake every waiting take with ``why`` (the run failed: no release is coming)."""
        with self._cv:
            self.failure = why
            self._cv.notify_all()

    def record(self) -> dict[str, Any]:
        return {"capacity_bytes": self.capacity, "held_bytes": self.held, "peak_bytes": self.peak, "windows_held": len(self._owned),
                "waits": self.waits, "waited_s": round(self.waited_s, 6)}


class DrawnRanges(_HostRanges):
    """A released step's retained bytes: exactly its drawn leaves' (``drawn``, leaf indices), on host. Any other byte was dropped at
    the release, and a read of it is refused by name (`DROPPED`), never routed to a replay."""

    dropped = True

    def __init__(self, ranges: list[tuple[int, int, Any]], used: int, drawn: Iterable[int], chunk: int) -> None:
        super().__init__(ranges, used, chunk=chunk)
        self.drawn = frozenset(int(i) for i in drawn)

    def __getitem__(self, sl: slice) -> Any:
        lo, hi = int(sl.start or 0), int(min(sl.stop if sl.stop is not None else self.n, self.n))
        if hi > lo and self._cover(lo, hi) is None:
            raise KeyError(f"bytes [{lo}, {hi}) of the step stream: {DROPPED}")
        return super().__getitem__(sl)


def leaf_ranges(leaves: Iterable[int], chunk: int, padded: int) -> list[tuple[int, int]]:
    """The byte ranges of ``leaves`` (leaf indices of ``chunk`` bytes) inside a step of ``padded`` bytes, merged where adjacent."""
    out: list[tuple[int, int]] = []
    for i in sorted({int(i) for i in leaves}):
        lo, hi = i * chunk, min((i + 1) * chunk, padded)
        if not 0 <= lo < padded:
            raise IndexError(f"leaf {i} lies outside the step's {padded // chunk} leaves")
        if out and out[-1][1] == lo:
            out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return out


def pool_key(com: Any) -> Any:
    """The window a committer, or its window view, holds pool pieces for (`window_views.WindowViews` gives each view its own)."""
    return com.__dict__.setdefault("_device_pool_key", object())


def release(com: Any, drawn: Mapping[int, Iterable[int]]) -> dict[str, Any]:
    """`NativeCollectCommitter.release_drawn` over ``com`` (a window view, or a committer whose steps are one window): by step, its
    drawn leaves `gather`ed, the rest dropped, its pool pieces freed; the record of what was kept and dropped."""
    if getattr(com, "released", None) is not None:
        raise RuntimeError("release_drawn: this window was released already")
    waited = com.wait_absorbed() if callable(getattr(com, "wait_absorbed", None)) else 0.0
    if stray := sorted(set(drawn) - set(com._gpu_blocks)):
        raise KeyError(f"release_drawn: drawn leaves name steps {stray[:8]} this window did not commit")
    t0 = time.perf_counter()
    kept = held = n_leaves = 0
    for s, (blk, used) in sorted(com._gpu_blocks.items()):
        leaves = sorted({int(i) for i in drawn.get(s, ())})
        n = int(com._done[s].n_leaves)
        if bad := [i for i in leaves if not 0 <= i < n]:
            raise IndexError(f"release_drawn: step {s} has {n} leaves, not {bad[:8]}")
        held += 0 if blk is None else int(blk.retained_bytes if hasattr(blk, "retained_bytes") else blk.numel())
        out = gather(blk, used, leaves, int(com.chunk))
        com._gpu_blocks[s] = (out, used)
        kept, n_leaves = kept + out.retained_bytes, n_leaves + len(leaves)
    pool = getattr(com, "device_pool", None)
    freed = pool.free(pool_key(com)) if pool is not None else 0
    com.released = {"steps": len(com._gpu_blocks), "drawn_leaves": n_leaves, "kept_bytes": kept, "dropped_bytes": held - kept,
                    "device_freed_bytes": freed, "absorb_wait_s": round(waited, 6), "gather_s": round(time.perf_counter() - t0, 6)}
    return com.released


def gather(blk: Any, used: int, leaves: Iterable[int], chunk: int) -> DrawnRanges:
    """The drawn ``leaves`` of a step whose retained block is ``blk`` (any of the committer's retainers), copied to host."""
    leaves = list(leaves)
    padded = -(-int(used) // chunk) * chunk
    pin = torch.cuda.is_available()
    ranges = []
    for lo, hi in leaf_ranges(leaves, chunk, padded):
        holds = getattr(blk, "holds", None)
        if blk is None or (holds is not None and not holds(lo, hi)):
            raise KeyError(f"bytes [{lo}, {hi}) of a drawn leaf were not retained: the release can't keep them")
        hv = torch.empty(hi - lo, dtype=torch.uint8, pin_memory=pin)
        hv.copy_(blk[lo:hi])
        ranges.append((lo, hi, hv))
    return DrawnRanges(ranges, used, leaves, chunk)
