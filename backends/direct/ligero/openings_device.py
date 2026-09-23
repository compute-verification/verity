"""Device-side openings for the B-Ligero prover (lane hp2-host): the Merkle root, the opened columns and the
authentication paths leave the device as ONE pinned, non-blocking copy each instead of one synchronous ``.cpu()`` /
``cp.asnumpy`` per object (the root; the (M, t) column block; the (t, depth, 8) sibling block).  Same bytes as
``merkle.commit_device`` / ``merkle.open_device``: leaf-level-first sibling per level, ``lvl[(c >> k) ^ 1]``, 32 bytes
each as 8 little-endian uint32 words.  The hash kernels themselves (``backends/shared/hash_gpu``) are untouched; this
module only reads the tree levels they produce.
"""
from __future__ import annotations

import numpy as np
import torch

from . import merkle

__all__ = ["commit_device_async", "gather_paths_device", "paths_from_raw"]


def _as_torch_i32(cupy_arr, device) -> torch.Tensor:
    """Zero-copy view of a cupy uint32 array as a torch int32 tensor on ``device`` (CUDA array interface)."""
    import cupy as cp
    return torch.as_tensor(cupy_arr.view(cp.int32), device=device)


def commit_device_async(U_int32: torch.Tensor):
    """``merkle.commit_device`` without the host read of the root: returns ``(root_dev (8,) int32, tree)``; the caller
    copies ``root_dev`` out (32 bytes: 8 LE uint32 words == ``merkle._gpu_merkle.digest_bytes(tree.root)``)."""
    tree = merkle._gpu_merkle.commit(U_int32, "blake3")
    return _as_torch_i32(tree.root, U_int32.device), tree


def gather_paths_device(tree, cols_t: torch.Tensor) -> torch.Tensor:
    """(t, depth, 8) int32 siblings, leaf level first: ``levels[k][(c >> k) ^ 1]`` -- on the device, on the current
    torch stream (the tree levels are read through the CUDA array interface; no cupy launch)."""
    levels = tree.levels[:-1]
    t = cols_t.shape[0]
    if not levels or t == 0:
        return torch.empty((t, 0, 8), dtype=torch.int32, device=cols_t.device)
    sib = [_as_torch_i32(lvl, cols_t.device).index_select(0, (cols_t >> k) ^ 1) for k, lvl in enumerate(levels)]
    return torch.stack(sib, dim=1)


def paths_from_raw(raw: bytes, t: int, depth: int) -> list[list[bytes]]:
    """The ``Proof.paths`` list-of-lists from the (t, depth, 8) LE-uint32 block's bytes (``merkle.open_device`` layout)."""
    return [[raw[(c * depth + k) * 32:(c * depth + k + 1) * 32] for k in range(depth)] for c in range(t)]


def opened_from_pinned(buf_i32: torch.Tensor, M: int, t: int) -> np.ndarray:
    """(M, t) uint32 C-contiguous copy out of a pinned int32 view (elements < p < 2^31: the int32 and uint32 bit patterns agree)."""
    return buf_i32.numpy().view(np.uint32).reshape(M, t).copy()
