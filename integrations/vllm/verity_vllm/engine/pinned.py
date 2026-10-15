"""Pinned host memory for the capture's asynchronous D2H copies, the storage byte range a tensor view spans, and whether a
CPU tensor's memory is really on the device."""

from __future__ import annotations

import ctypes
from typing import Any

import torch

_CUDA_MEMORY_TYPE_DEVICE = 2     # cudaMemoryTypeDevice


class _PointerAttributes(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("device", ctypes.c_int), ("devicePointer", ctypes.c_void_p),
                ("hostPointer", ctypes.c_void_p)]


_cudart: Any = None


def _runtime() -> Any:
    """The CUDA runtime torch loaded (by soname, so no second copy), None when there is none."""
    global _cudart
    if _cudart is None:
        _cudart = False
        for name in ("libcudart.so.12", "libcudart.so"):
            try:
                _cudart = ctypes.CDLL(name)
                break
            except OSError:
                continue
    return _cudart or None


def device_backed(t: torch.Tensor) -> bool:
    """True when ``t`` says it is on the CPU but its storage is device memory: a tensor C++ code wrapped with
    ``torch::from_blob`` and no device (DeepGEMM's paged MQA logits, VLLM-8).  Reading its bytes on the host faults.  False
    for a device tensor, without CUDA, or when the runtime can't say."""
    if t.device.type != "cpu" or not torch.cuda.is_available():
        return False
    lib = _runtime()
    if lib is None:
        return False
    attrs = _PointerAttributes()
    if lib.cudaPointerGetAttributes(ctypes.byref(attrs), ctypes.c_void_p(t.untyped_storage().data_ptr())) != 0:
        lib.cudaGetLastError()
        return False
    return attrs.type == _CUDA_MEMORY_TYPE_DEVICE


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
