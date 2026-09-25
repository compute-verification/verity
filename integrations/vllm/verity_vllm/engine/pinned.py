"""Pinned host memory for the capture's asynchronous D2H copies, and the storage byte range a tensor view spans."""

from __future__ import annotations

import torch


class PinnedArena:
    """Bump allocator over one pinned host buffer (reset per step); overflow falls back to fresh pinned tensors."""

    def __init__(self, nbytes: int) -> None:
        self.buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        self.off = 0
        self.overflow = 0

    def reset(self) -> None:
        self.off = 0

    def take(self, nbytes: int) -> torch.Tensor:
        aligned = (self.off + 255) // 256 * 256
        if aligned + nbytes > self.buf.numel():
            self.overflow += 1
            return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        v = self.buf[aligned:aligned + nbytes]
        self.off = aligned + nbytes
        return v


def spanned_bytes(t: torch.Tensor) -> tuple[int, int]:
    """(byte offset in the storage, byte length) of the storage range a view can touch."""
    es = t.element_size()
    if t.numel() == 0:
        return int(t.storage_offset()) * es, 0
    last = sum((s - 1) * st for s, st in zip(t.shape, t.stride()))
    return int(t.storage_offset()) * es, (int(last) + 1) * es


def storage_bytes_view(t: torch.Tensor, byte_off: int, nbytes: int) -> torch.Tensor:
    """A contiguous uint8 view over `[byte_off, byte_off + nbytes)` of `t`'s storage (no device allocation)."""
    v = torch.empty(0, dtype=torch.uint8, device=t.device)
    v.set_(t.untyped_storage(), byte_off, (nbytes,), (1,))
    return v
