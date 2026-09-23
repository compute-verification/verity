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


_OFFS: dict = {}


def _level_offsets(sizes: tuple, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(offsets (depth,), shifts (depth,)) of the concatenated tree levels (cached per level-size tuple)."""
    key = (sizes, str(device))
    hit = _OFFS.get(key)
    if hit is None:
        offs, at = [], 0
        for sz in sizes:
            offs.append(at)
            at += sz
        hit = (torch.tensor(offs, dtype=torch.int64, device=device),
               torch.arange(len(sizes), dtype=torch.int64, device=device))
        _OFFS[key] = hit
    return hit


def gather_paths_device(tree, cols_t: torch.Tensor) -> torch.Tensor:
    """(t, depth, 8) int32 siblings, leaf level first: ``levels[k][(c >> k) ^ 1]`` -- on the device, on the current
    torch stream (the tree levels are read through the CUDA array interface; no cupy launch).  The levels are
    concatenated once (2n x 32 bytes) and gathered by ONE index_select with per-level offsets instead of one
    index_select + stack per level (was 2 x depth launches)."""
    levels = tree.levels[:-1]
    t = cols_t.shape[0]
    if not levels or t == 0:
        return torch.empty((t, 0, 8), dtype=torch.int32, device=cols_t.device)
    dev = cols_t.device
    flat = torch.cat([_as_torch_i32(lvl, dev) for lvl in levels])                  # (sum_k N_k, 8)
    offs, shifts = _level_offsets(tuple(int(lvl.shape[0]) for lvl in levels), dev)
    idx = offs[None, :] + ((cols_t[:, None] >> shifts[None, :]) ^ 1)              # (t, depth)
    return flat.index_select(0, idx.reshape(-1)).view(t, len(levels), 8)


def paths_from_raw(raw: bytes, t: int, depth: int) -> list[list[bytes]]:
    """The ``Proof.paths`` list-of-lists from the (t, depth, 8) LE-uint32 block's bytes (``merkle.open_device`` layout).
    numpy's void dtype keeps every byte (``S32`` would strip trailing NULs) and ``tolist`` builds the t x depth
    ``bytes`` objects in C (was t x depth Python slices, ~1 ms per sub-batch)."""
    if t == 0 or depth == 0:
        return [[] for _ in range(t)]
    return np.frombuffer(raw, dtype=np.dtype(("V", 32))).reshape(t, depth).tolist()


def opened_from_pinned(buf_i32: torch.Tensor, M: int, t: int) -> np.ndarray:
    """(M, t) uint32 C-contiguous copy out of a pinned int32 view (elements < p < 2^31: the int32 and uint32 bit patterns agree)."""
    return buf_i32.numpy().view(np.uint32).reshape(M, t).copy()
