"""The host committer's staging blocks on the device, one for each step in flight (`_DeviceArena`)."""

from __future__ import annotations

import threading
import time
from typing import Any

from verity_vllm import settings
from verity_vllm.commit.committer.optional_torch import torch


class _DeviceArena:
    """Device (GPU) staging blocks, one per in-flight step.  A hook copies every acquired tensor D2D into the step's block ON THE
    COMPUTE STREAM (stream-ordered => the snapshot is taken before any later kernel overwrites the tensor in place, e.g. the fused
    add+RMSNorm residual), so the worker can move / hash the packed step stream from a SIDE stream while the engine runs on.
    Blocks are allocated once (warm-up) and recycled when the worker releases them."""

    def __init__(self, block_bytes: int, n_blocks: int = 2, extra_blocks: list[int] | None = None) -> None:
        self.block_bytes = block_bytes
        self.free: list[Any] = []
        self.all: list[Any] = []
        self.allocs = 0
        self.alloc_s = 0.0
        self._n_init = n_blocks
        self._extra = list(extra_blocks or [])       # additional preallocated blocks of other sizes (one prefill-sized block)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self.waits = 0                                # backpressure: how often / how long the serving thread waited for a block
        self.wait_s = 0.0

    def prealloc(self) -> None:
        for _ in range(self._n_init):
            self.free.append(self._alloc(self.block_bytes))
        for nb in self._extra:
            self.free.append(self._alloc(nb))

    def _alloc(self, nbytes: int) -> Any:
        t0 = time.perf_counter()
        blk = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        self.all.append(blk)
        self.allocs += 1
        self.alloc_s += time.perf_counter() - t0
        return blk

    # Backpressure: the ring never grows past its preallocation while a fitting block is merely BUSY; the serving thread waits for
    # the worker to release one (the engine stalls instead of the collector OOMing).  Growth only when NO block of the ring can ever fit (a
    # step larger than every block), with VERITY_ARENA_GROW=1, or `wait=False`: the worker, the only releaser, must not wait on itself.
    GROW = bool(settings.VERITY_ARENA_GROW)
    WAIT_TIMEOUT_S = settings.VERITY_ARENA_WAIT_S

    def _best_free(self, min_bytes: int) -> int:
        best = -1
        for i, b in enumerate(self.free):
            if b.numel() >= min_bytes and (best < 0 or b.numel() < self.free[best].numel()):
                best = i
        return best

    def take(self, min_bytes: int = 0, *, wait: bool = True) -> Any:
        # best fit: the SMALLEST free block that holds the step (decode steps must not consume the prefill-sized block)
        with self._cv:
            best = self._best_free(min_bytes)
            if best >= 0:
                return self.free.pop(best)
            if wait and not self.GROW and any(b.numel() >= min_bytes for b in self.all):
                t0 = time.perf_counter()
                while best < 0:
                    if not self._cv.wait(timeout=1.0) and time.perf_counter() - t0 > self.WAIT_TIMEOUT_S:
                        raise RuntimeError(f"native collector arena: waited {self.WAIT_TIMEOUT_S:.0f}s for a free block >= {min_bytes} B "
                                           f"(ring {[int(b.numel()) >> 20 for b in self.all]} MiB, all busy): the worker is not releasing blocks")
                    best = self._best_free(min_bytes)
                self.waits += 1
                self.wait_s += time.perf_counter() - t0
                return self.free.pop(best)
        return self._alloc(max(self.block_bytes, min_bytes))

    def release(self, blk: Any) -> None:
        with self._cv:
            self.free.append(blk)
            self._cv.notify_all()

    @property
    def bytes_total(self) -> int:
        return sum(int(b.numel()) for b in self.all)
