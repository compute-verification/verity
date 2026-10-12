"""Device-side openings for the B-Ligero prover (lane hp2-host): the Merkle root, the opened columns and the
authentication paths leave the device as ONE pinned, non-blocking copy each instead of one synchronous ``.cpu()`` /
``cp.asnumpy`` per object (the root; the (M, t) column block; the (t, depth, 8) sibling block).  Same bytes as
``merkle.commit_device`` / ``merkle.open_device``: leaf-level-first sibling per level, ``lvl[(c >> k) ^ 1]``, 32 bytes
each as 8 little-endian uint32 words.  The hash kernels themselves (``verity/ml/kernels/hash_gpu``) are untouched; this
module only reads the tree levels they produce.
"""
from __future__ import annotations

import numpy as np
import torch

from . import merkle

__all__ = ["commit_device_async", "gather_paths_device", "paths_from_raw"]


def _as_torch_i32(arr, device) -> torch.Tensor:
    """Zero-copy view of a cupy uint32 array as a torch int32 tensor on ``device`` (CUDA array interface); a torch
    tensor (a ``StaticMerkle`` level) passes through."""
    if isinstance(arr, torch.Tensor):
        return arr
    import cupy as cp
    return torch.as_tensor(arr.view(cp.int32), device=device)


class StaticMerkle:
    """Graph-capturable Blake3 Merkle commit of a fixed-shape ``(R, N)`` int32 matrix (lane hp2-host).  The same two
    kernels and the same launch geometry as ``hash_gpu.merkle.commit(..., "blake3")`` (``blake3_chunks`` over the
    columns in 256-word chunks, ``blake3_parents`` to fold the chunk digests, then ``blake3_chunks`` on 16-word
    pairs per tree level), but every buffer is allocated once with torch, so ``run`` is ~21 launches on the current
    stream with no allocation -- capturable into the caller's CUDA graph (``hash_gpu`` allocates with cupy per call).
    ``levels`` (leaf level first, root level last) and ``root`` have the ``hash_gpu.merkle.Tree`` meaning; the levels
    are views into ONE ``flat`` buffer, so ``gather_paths_device`` needs no concatenation.  Digests are identical."""

    def __init__(self, R: int, N: int, device):
        from . import merkle as _mk                                     # puts kernels on sys.path
        assert _mk.gpu_available(), "StaticMerkle needs the GPU blake3"
        from hash_gpu import blake3_cuda as b3
        self.b3, self.R, self.N, self.device = b3, int(R), int(N), device
        CW = b3.CHUNK_WORDS
        nchunks = max(1, (self.R + CW - 1) // CW)
        i32 = dict(dtype=torch.int32, device=device)
        # leaves: chunk chaining values (nchunks, N, 8) folded pairwise to (1, N, 8)
        self.cvs = torch.empty((nchunks, self.N, 8), **i32)
        self.folds: list = []                                           # (src, count, dst) per parents launch
        cur, count = self.cvs, nchunks
        while count > 1:
            nout = (count + 1) // 2
            nxt = torch.empty((nout, self.N, 8), **i32)
            self.folds.append((cur, count, nxt))
            cur, count = nxt, nout
        # tree levels in one flat buffer: N + N/2 + ... + 1 rows of 8 words
        sizes, c = [], self.N
        while True:
            sizes.append(c)
            if c == 1:
                break
            c = (c + 1) // 2
        self.flat = torch.empty((sum(sizes), 8), **i32)
        self.levels, at = [], 0
        for sz in sizes:
            self.levels.append(self.flat[at:at + sz])
            at += sz
        self.leaf_src = cur                                             # (1, N, 8): copied into levels[0]
        self.root = self.levels[-1][0]                                  # (8,) int32, LE words
        self._cp = None

    def _views(self):
        """cupy views of the torch buffers (once; zero-copy through the CUDA array interface)."""
        if self._cp is None:
            import cupy as cp
            v = lambda t: cp.asarray(t).view(cp.uint32)  # noqa: E731
            self._cp = dict(cvs=v(self.cvs), folds=[(v(a), c, v(b)) for a, c, b in self.folds],
                            leaf_src=v(self.leaf_src), levels=[v(lv) for lv in self.levels])
        return self._cp

    def run(self, U_int32: torch.Tensor) -> None:
        """Commit ``U_int32`` ((R, N) int32, contiguous) on the current stream (call inside the cupy stream context)."""
        import cupy as cp
        assert U_int32.shape == (self.R, self.N) and U_int32.dtype == torch.int32 and U_int32.is_contiguous()
        b3, N, R = self.b3, self.N, self.R
        cv = self._views()
        mod = b3.module()
        chunks, parents = mod.get_function("blake3_chunks"), mod.get_function("blake3_parents")
        base = cp.asarray(U_int32).view(cp.uint32)
        nchunks = self.cvs.shape[0]
        block = 128
        chunks(((N + block - 1) // block, nchunks), (block,),
               (base, np.int64(1), np.int64(b3.CHUNK_WORDS * N), np.int64(N), np.int32(R), np.int32(N),
                np.int32(nchunks), cv["cvs"]))
        for src, count, dst in cv["folds"]:                            # hash_strided passes its block (128) on
            total = dst.shape[0] * N
            parents(((total + block - 1) // block,), (block,), (src, np.int32(N), np.int32(count), dst))
        self.levels[0].copy_(self.leaf_src.view(N, 8))
        for k in range(1, len(self.levels)):
            prev, nxt = self.levels[k - 1], self.levels[k]
            count = prev.shape[0]
            npairs = count // 2
            out = cv["levels"][k][:npairs]
            chunks(((npairs + block - 1) // block, 1), (block,),
                   (cv["levels"][k - 1].reshape(-1), np.int64(16), np.int64(0), np.int64(1), np.int32(16),
                    np.int32(npairs), np.int32(1), out))
            if count % 2:
                nxt[npairs].copy_(prev[count - 1])


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
    if isinstance(tree, StaticMerkle):
        flat = tree.flat                                                            # the levels are its prefix
    else:
        flat = torch.cat([_as_torch_i32(lvl, dev) for lvl in levels])              # (sum_k N_k, 8)
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
