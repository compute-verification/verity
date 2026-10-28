"""The committer's pinned host pool (`PinnedPool`): the page-locked blocks the D2H copies land in and the openings read.

Exact mode (the default; `VERITY_PINNED_EXACT=0` for torch's allocator) maps shared anonymous memory of the page-rounded size and
registers it with `cudaHostRegister`, so the process RSS is the pool's accounting and `release()` returns it to the OS (`_pinned_block`).
A fresh block's pages are faulted in by `populate_threads` helper threads before the caller registers it, and `reserve` makes a burst
of blocks together.  torch is imported where a block is made: the offline core builds a pool and never allocates from it.
`lend` takes a run's blocks out of the pool for a reader on another thread (a served window's seal) and `restore` brings them back.
"""

from __future__ import annotations

import ctypes
import errno
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

BLOCK_BYTES = 64 << 20
PINNED_EXACT = os.environ.get("VERITY_PINNED_EXACT", "1") != "0"
_PAGE = 4096
# Growing the exact pool is page faulting: every page of a fresh shared mapping is allocated and zeroed when it is first touched, which
# `cudaHostRegister` does one page at a time on the calling thread.  On the leased node (a VM: first touch also faults on the host) that
# ran 0.4-2.9 s/GiB, so the instrumented warm-up of a leased Commit grew its ~105 GiB pool in 40-270 s with the GPU idle (266 s of
# 271).  A pool's `populate_threads` helper threads fault the pages in first (MADV_POPULATE_WRITE, in pieces); the registration then pins
# pages that are present.  Faults on one mapping (one shmem inode) scale less over threads than faults on separate mappings: 8 threads on
# the node's 32-CPU dispatch slice, one 8 GiB mapping 0.15-0.2 s/GiB, 32 mappings 0.085-0.105, one thread 0.5.  0 = no population.
POPULATE_THREADS = 8
_POPULATE_PIECE = 32 << 20
_MADV_POPULATE_WRITE = 23                   # Linux >= 5.14; EINVAL before
_POPULATE: dict[str, Any] = {"ok": True, "pools": {}}
os.register_at_fork(after_in_child=lambda: _POPULATE["pools"].clear())   # a forked child has none of the parent's helper threads


def _page_round(nbytes: int) -> int:
    return (int(nbytes) + _PAGE - 1) // _PAGE * _PAGE


def _populate(ranges: list[tuple[int, int]], threads: int, times: dict[str, float] | None = None) -> bool:
    """Write-fault every page of the page-aligned (address, bytes) ranges, split in `_POPULATE_PIECE`s over `threads` helper threads.
    The helpers make no CUDA call: a thread that used the device and outlives the lease's context reset opens a context again on a GPU
    the lease gave back (`gpu_lease.release`), so only the caller registers.  False when nothing was populated (disabled, or the kernel
    has no MADV_POPULATE_WRITE): the registration then faults the pages in itself."""
    if threads <= 0 or not _POPULATE["ok"] or not ranges:
        return False
    t0 = time.perf_counter()
    if "madvise" not in _POPULATE:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        _POPULATE["madvise"] = libc.madvise
    madvise = _POPULATE["madvise"]

    def one(job: tuple[int, int]) -> int:
        return 0 if madvise(job[0], job[1], _MADV_POPULATE_WRITE) == 0 else ctypes.get_errno() or -1

    jobs = [(a + o, min(_POPULATE_PIECE, n - o)) for a, n in ranges for o in range(0, n, _POPULATE_PIECE)]
    if threads == 1 or len(jobs) == 1:
        errs = [one(j) for j in jobs]
    else:
        pools = _POPULATE["pools"]
        if threads not in pools:
            pools[threads] = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="pinned-populate")
        errs = list(pools[threads].map(one, jobs))
    if errno.EINVAL in errs:
        _POPULATE["ok"] = False
    if times is not None:
        times["populate_s"] = times.get("populate_s", 0.0) + time.perf_counter() - t0
    return not any(errs)


def _shared_map(nbytes: int) -> tuple[Any, Any]:
    """A fresh MAP_SHARED|MAP_ANONYMOUS mapping of the page-rounded size and its uint8 view (see `_pinned_block` for why shared)."""
    import mmap

    import torch

    mm = mmap.mmap(-1, _page_round(nbytes), flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)
    return torch.frombuffer(mm, dtype=torch.uint8), mm


def _register(blk: Any, mm: Any, times: dict[str, float] | None = None) -> None:
    import torch

    t0 = time.perf_counter()
    rc = torch.cuda.cudart().cudaHostRegister(blk.data_ptr(), blk.numel(), 0)
    if int(rc) != 0:
        mm.close()
        raise RuntimeError(f"cudaHostRegister({blk.numel()} B) failed: {rc}")
    if times is not None:
        times["register_s"] = times.get("register_s", 0.0) + time.perf_counter() - t0


def _pinned_block(nbytes: int, exact: bool, threads: int, times: dict[str, float] | None = None) -> tuple[Any, Any, str]:
    """One page-locked host block of `nbytes`.  Returns (block view, its owner tensor, mode).

    `torch.empty(n, pin_memory=True)` goes through torch's CachingHostAllocator, which rounds EVERY request up to
    the next power of two and never returns a freed block to the OS (torch 2.13.0+cu129 on cp-h100, measured: 3 GiB -> RSS +4 GiB,
    5 GiB + 7 B -> +8 GiB, 1 GiB + 1 B -> +2 GiB; `del` -> +0).  Pinned pages are committed at allocation, so the process RSS and the
    cgroup (as shmem) take the ROUNDED size at once while `bytes_total` counts the requested one: on the B8 FP8 workload that gap was
    21,152 MiB of run-phase RSS outside any term of the host bound.
    exact=True: a MAP_SHARED|MAP_ANONYMOUS mmap of the page-rounded size + `cudaHostRegister` on it -> page-locked for the D2H copies
    (measured 29 GB/s, = torch's pinned blocks), RSS == the pool's accounting, `cudaHostUnregister` + munmap on `release()` returns it to
    the OS.
    The block MUST be a SHARED mapping, not private anonymous memory: Linux >= 5.9 copies DMA-pinned PRIVATE pages eagerly at fork()
    (copy_present_pte -> page_needs_cow_for_dma), so every child the C2 sampled replay forks (its evaluator Pool) would get its own copy
    of the whole frozen pool at fork time -- on a 59,350 MiB pool, cgroup anon 135 -> 239 GB (the cap) in a minute and 20 OOM-killed
    children in three (measured on a 2 GiB block: private-registered fork -> cgroup anon +2,653 MiB; shared-registered -> +336, the
    child's RssShmem 2,048 = the shared pages).  Shared mappings are never copy-on-write, so children read the committed store in place,
    as on torch's (also MAP_SHARED) blocks; the cgroup accounts them as shmem, like torch's."""
    import torch

    if exact and torch.cuda.is_available():
        blk, mm = _shared_map(nbytes)
        _populate([(blk.data_ptr(), blk.numel())], threads, times)
        _register(blk, mm, times)
        return blk, mm, "exact-registered-shared"
    return (b := torch.empty(int(nbytes), dtype=torch.uint8, pin_memory=True)), b, "torch-pinned-pow2"


EXACT_MODES = ("exact-registered-shared", "exact-registered")      # the second, the private-anonymous variant, is no longer allocated: fork() copies its pages


class PinnedPool:
    """Pinned host blocks; `take(n)` returns a uint8 tensor view of n bytes (a tensor never straddles blocks).  Blocks are
    retained for the run (openings read them) and returned by `reset()`; the pool grows only when the free list is empty
    (the warm-up instrumented pass sizes it, so steady-state runs allocate nothing).  `bytes_total` is the process RSS the pool
    holds when `mode == "exact-registered-shared"` (the default; env VERITY_PINNED_EXACT=0 for torch's allocator).
    `populate_threads` (default `POPULATE_THREADS`; the commit command's --pinned-populate-threads) fault a fresh exact block's pages in."""

    def __init__(self, block_bytes: int = BLOCK_BYTES, exact: bool | None = None, populate_threads: int | None = None) -> None:
        self.block_bytes = block_bytes
        self.exact = PINNED_EXACT if exact is None else bool(exact)
        self.populate_threads = POPULATE_THREADS if populate_threads is None else int(populate_threads)
        self.mode: str | None = None
        self.free: list[Any] = []
        self.all: list[Any] = []
        self._owners: list[Any] = []          # the registered allocations behind the blocks (unregistered by release())
        self.cur: Any | None = None
        self.off = 0
        self.in_use: list[Any] = []
        self.allocs = 0
        self.alloc_s = 0.0
        self.times: dict[str, float] = {}     # populate_s, register_s: the two parts of alloc_s for exact blocks
        self._ready: list[Any] = []           # blocks `reserve` made, in the order the takes after it allocate them
        self.reserved_bytes = 0
        self.reserve_unused = 0               # reserved blocks no take asked for (folded into the free list by `reset`): 0 when exact
        self._returned: deque = deque()       # lent blocks back from their reader (`restore`, any thread), free from the next `reset`

    def fresh(self) -> PinnedPool:
        """An empty pool with this one's settings (block size, mode, helper threads)."""
        return type(self)(self.block_bytes, self.exact, self.populate_threads)

    def _alloc(self, nbytes: int) -> Any:
        want = max(int(nbytes), self.block_bytes)
        if self._ready and self._ready[0].numel() == _page_round(want):
            return self._ready.pop(0)
        blk, owner, mode = _pinned_block(want, self.exact, self.populate_threads, self.times)
        if self.mode is None:
            self.mode = mode
        self.all.append(blk)
        self._owners.append(owner)
        self.allocs += 1
        return blk

    def reserve(self, sizes: list[int]) -> int:
        """Make now, faulted in together, the blocks that `take(n)` for each n of `sizes` in turn would allocate one at a time (the
        same first fit over the free list, the same `cur`/`off`), so those takes find them ready: a burst of pool growth -- the warm-up's
        priming of a step's levels and retained windows -- costs its page faults over every helper thread and mapping at once, not
        block by block.  Exact mode only (else a no-op).  Returns the bytes made."""
        import torch

        if not (self.exact and torch.cuda.is_available()) or self.populate_threads <= 0:
            return 0
        free = [int(b.numel()) for b in self.free]
        cur = None if self.cur is None else int(self.cur.numel())
        off, new = self.off, []
        for n in map(int, sizes):
            if cur is None or off + n > cur:
                i = next((k for k, f in enumerate(free) if f >= n), -1)
                if i >= 0:
                    cur = free.pop(i)
                else:
                    cur = _page_round(max(n, self.block_bytes))
                    new.append(cur)
                off = 0
            off = (off + n + 255) // 256 * 256
        if not new:
            return 0
        t0 = time.perf_counter()
        maps = [_shared_map(n) for n in new]
        _populate([(b.data_ptr(), b.numel()) for b, _mm in maps], self.populate_threads, self.times)
        for b, mm in maps:
            _register(b, mm, self.times)
            self.all.append(b)
            self._owners.append(mm)
            self._ready.append(b)
        self.mode = self.mode or "exact-registered-shared"
        self.allocs += len(new)
        self.alloc_s += time.perf_counter() - t0
        self.reserved_bytes += sum(new)
        return sum(new)

    def release(self) -> int:
        """Unregister and drop every block (the pool's memory goes back to the OS in exact mode; torch's cache keeps it otherwise).
        Returns the bytes released.  The pool is empty and reusable afterwards."""
        n = self.bytes_total
        if self.mode in EXACT_MODES:
            import torch

            rt = torch.cuda.cudart()
            for b in self.all:
                rt.cudaHostUnregister(b.data_ptr())
        owners = self._owners
        self.free, self.all, self._owners, self.in_use, self.cur, self.off, self.mode = [], [], [], [], None, 0, None
        self._ready = []
        self._returned.clear()
        self.deferred_unmaps = 0
        import sys

        for o in owners:
            if hasattr(o, "close"):
                # torch.frombuffer holds ONE reference to the mmap object from the tensor's storage (every view shares it) and no buffer
                # export -- `close()` would succeed under a live view and leave it dangling.  Refs here: `owners`, the loop variable, the
                # getrefcount argument = 3; more means a view handed out by take() is alive: leave the object to it (munmap happens in
                # the mmap's dealloc when the last view dies) and say so
                if sys.getrefcount(o) <= 3:
                    o.close()                     # munmap now: the shared pages go back to the OS
                else:
                    self.deferred_unmaps += 1
        return n

    def __del__(self) -> None:  # pragma: no cover -- ordering: unregister BEFORE the owners are freed
        try:
            if self.mode in EXACT_MODES:
                self.release()
        except Exception:  # noqa: BLE001
            pass

    def _fresh(self, nbytes: int) -> Any:
        t0 = time.perf_counter()
        for i, b in enumerate(self.free):
            if b.numel() >= nbytes:
                blk = self.free.pop(i)
                break
        else:
            blk = self._alloc(nbytes)
        self.alloc_s += time.perf_counter() - t0
        self.in_use.append(blk)
        return blk

    def take(self, nbytes: int) -> Any:
        if self.cur is None or self.off + nbytes > self.cur.numel():
            self.cur = self._fresh(nbytes)
            self.off = 0
        v = self.cur[self.off:self.off + nbytes]
        self.off = (self.off + nbytes + 255) // 256 * 256
        return v

    def reset(self) -> None:
        self.free.extend(self.in_use)
        self.reserve_unused += len(self._ready)
        self.free.extend(self._ready)
        while self._returned:
            self.free.extend(self._returned.popleft())
        self._ready = []
        self.in_use = []
        self.cur = None
        self.off = 0

    def lend(self) -> list[Any]:
        """The blocks this run took, out of the pool until `restore` brings them back: the retained values of a closed run stay
        intact while the next run takes blocks of its own, so the pool grows by what is out.  `release` unmaps lent blocks too:
        restore them first."""
        lent, self.in_use, self.cur, self.off = self.in_use, [], None, 0
        return lent

    def restore(self, blocks: list[Any]) -> None:
        """`lend`'s blocks back, from any thread: free for the takes after the next `reset`."""
        self._returned.append(list(blocks))

    def take_block(self, nbytes: int) -> Any:
        """A whole block for a SHORT-LIVED buffer (returned with `give_back`, unlike `take` slices which live until `reset`)."""
        for i, b in enumerate(self.free):
            if b.numel() >= nbytes:
                return self.free.pop(i)
        t0 = time.perf_counter()
        blk = self._alloc(nbytes)
        self.alloc_s += time.perf_counter() - t0
        return blk

    def give_back(self, blk: Any) -> None:
        self.free.append(blk)

    @property
    def bytes_total(self) -> int:
        return sum(int(b.numel()) for b in self.all)

    @property
    def bytes_in_use(self) -> int:
        return sum(int(b.numel()) for b in self.in_use)
