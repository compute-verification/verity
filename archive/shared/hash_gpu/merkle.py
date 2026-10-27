"""Merkle commitment of an ``(R, N)`` matrix of BabyBear elements: one leaf per column, tree over the ``N`` leaves.

``commit(matrix, hash)`` -> :class:`Tree`; ``hash`` in ``HASHES``.  The matrix is a row-major device buffer of
``uint32`` / ``int32`` (cupy or torch; Poseidon2 accepts int64 too).  Digests are ``(n, 8)`` uint32 (SHA-256:
big-endian words; Blake3: little-endian words; Poseidon2: 8 field elements).  Trees promote an unpaired rightmost
node unchanged, so ``N`` need not be a power of two.  ``Tree.open(columns)`` returns authentication paths.

Leaf functions (see ``reference.py`` for the CPU definitions they are tested against):
  sha256            SHA-256(column bytes), one thread per column (sequential over the column)
  sha256-chunked    SHA-256 of 1 KiB sub-chunks in parallel, binary SHA-256 tree over the chunk digests
  blake3            Blake3(column bytes), the spec's chunk tree in parallel (equals the ``blake3`` package)
  poseidon2-16/24   compression tree over 8-element blocks (``poseidon2_torch`` / ``poseidon2_cuda``)
Node functions: SHA-256(left || right), Blake3(left || right), Poseidon2 compress(left, right).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import cupy as cp
except ImportError:            # pragma: no cover
    cp = None

HASHES = ("sha256", "sha256-chunked", "blake3", "poseidon2-16", "poseidon2-24")
CHUNK_ROWS = 256


@dataclass
class Tree:
    hash: str
    rows: int
    cols: int
    levels: list = field(default_factory=list)      # levels[0] = leaves (N, 8) ... levels[-1] = (1, 8)

    @property
    def root(self):
        return self.levels[-1][0]

    @property
    def depth(self) -> int:
        return len(self.levels) - 1

    def open(self, columns) -> list[list]:
        """Authentication paths (sibling digests, leaf level first) for each opened column.  A promoted (unpaired)
        node has no sibling at that level: ``None`` is recorded and the verifier carries the digest up."""
        paths = []
        for c in columns:
            path, idx = [], int(c)
            for lvl in self.levels[:-1]:
                sib = idx ^ 1
                path.append(_row(lvl, sib) if sib < lvl.shape[0] else None)
                idx >>= 1
            paths.append(path)
        return paths

    def opening_bytes(self, n_columns: int, element_bytes: int = 4) -> int:
        """Bytes of ``n_columns`` openings: the columns themselves plus their paths (no de-duplication)."""
        return n_columns * (self.rows * element_bytes + 32 * self.depth)


def _row(level, i):
    return level[i]


def _as_cupy_u32(m):
    if cp is None:
        raise RuntimeError("cupy is required for the CUDA hashes")
    a = cp.asarray(m)
    if a.dtype != cp.uint32:
        a = a.view(cp.uint32)
    if a.ndim != 2 or not a.flags.c_contiguous:
        raise ValueError("matrix must be a contiguous (R, N) buffer")
    return a


def leaves(matrix, hash: str, **kw):
    """Leaf digests ``(N, 8)`` for the columns of ``matrix``."""
    if hash.startswith("poseidon2"):
        return _poseidon2_leaves(matrix, hash, **kw)
    m = _as_cupy_u32(matrix)
    R, N = m.shape
    if hash == "sha256":
        from . import sha256_cuda as sha

        return sha.hash_strided(m, 1, 0, N, R, N, 1, bswap=True)
    if hash == "sha256-chunked":
        return _sha256_chunked_leaves(m)
    if hash == "blake3":
        from . import blake3_cuda as b3

        return b3.hash_strided(m, 1, CHUNK_ROWS * N, N, R, N, workspace=kw.get("workspace"))
    raise ValueError(hash)


def _sha256_chunked_leaves(m):
    from . import sha256_cuda as sha

    R, N = m.shape
    full, tail = divmod(R, CHUNK_ROWS)
    nchunks = full + (1 if tail else 0)
    if nchunks == 1:
        return sha.hash_strided(m, 1, 0, N, R, N, 1, bswap=True)
    # chunk digests in (N, nchunks, 8) layout (column-major so a column's pairs are contiguous 16-word messages)
    level = cp.empty((N, nchunks, 8), dtype=cp.uint32)
    if full:
        if tail == 0:
            sha.hash_strided(m, 1, CHUNK_ROWS * N, N, CHUNK_ROWS, N, full, bswap=True, out=level.reshape(-1, 8), out_xy=True)
        else:
            tmp = sha.hash_strided(m, 1, CHUNK_ROWS * N, N, CHUNK_ROWS, N, full, bswap=True, out_xy=True)
            level[:, :full] = tmp.reshape(N, full, 8)
    if tail:
        level[:, full] = sha.hash_strided(m[full * CHUNK_ROWS:], 1, 0, N, tail, N, 1, bswap=True)
    count = nchunks
    while count > 1:
        npairs, nout = count // 2, (count + 1) // 2
        nxt = cp.empty((N, nout, 8), dtype=cp.uint32)
        if count % 2 == 0:
            sha.hash_strided(level.reshape(-1), count * 8, 16, 1, 16, N, npairs, bswap=False, out=nxt.reshape(-1, 8), out_xy=True)
        else:
            tmp = sha.hash_strided(level.reshape(-1), count * 8, 16, 1, 16, N, npairs, bswap=False, out_xy=True)
            nxt[:, :npairs] = tmp.reshape(N, npairs, 8)
            nxt[:, nout - 1] = level[:, count - 1]
        level, count = nxt, nout
    return level.reshape(N, 8)


def poseidon2_engine(hash: str, backend: str = "cuda", mds: str = "blocks"):
    """``backend`` ``cuda`` (fused kernel, the fast path) or ``torch`` (elementwise; ``mds`` selects its MDS route)."""
    from .poseidon2_ref import instance

    width = int(hash.split("-")[1])
    if backend == "cuda":
        from .poseidon2_cuda import Poseidon2Cuda

        return Poseidon2Cuda(instance(width))
    from .poseidon2_torch import Poseidon2Torch

    return Poseidon2Torch(instance(width), mds=mds)


def _poseidon2_leaves(matrix, hash: str, **kw):
    engine = kw.get("engine") or poseidon2_engine(hash, kw.get("backend", "cuda"), kw.get("mds", "blocks"))
    if type(engine).__name__ == "Poseidon2Cuda":
        return engine.leaf_tree(matrix)
    import torch

    m = matrix if isinstance(matrix, torch.Tensor) else torch.as_tensor(matrix, device="cuda")
    if m.dtype == torch.int32:
        m = m.to(torch.int64) & 0xFFFFFFFF
    return engine.leaf_tree(m)


def node_level(level, hash: str, **kw):
    """One tree level: ``(count, 8)`` -> ``((count + 1) // 2, 8)``."""
    if hash.startswith("poseidon2"):
        return kw["engine"].merkle_level(level)
    count = level.shape[0]
    npairs = count // 2
    if hash in ("sha256", "sha256-chunked"):
        from . import sha256_cuda as sha

        out = sha.hash_strided(level.reshape(-1), 16, 0, 1, 16, npairs, 1, bswap=False)
    elif hash == "blake3":
        from . import blake3_cuda as b3

        out = b3.hash_strided(level.reshape(-1), 16, 0, 1, 16, npairs)
    else:
        raise ValueError(hash)
    if count % 2:
        out = cp.concatenate((out, level[-1:]), axis=0)
    return out


def commit(matrix, hash: str = "blake3", **kw) -> Tree:
    """Commit the columns of ``matrix`` (``(R, N)`` row-major device buffer) with ``hash``."""
    if hash not in HASHES:
        raise ValueError(f"hash must be one of {HASHES}")
    if hash.startswith("poseidon2") and "engine" not in kw:
        kw["engine"] = poseidon2_engine(hash, kw.pop("backend", "cuda"), kw.pop("mds", "blocks"))
    R, N = matrix.shape
    t = Tree(hash, int(R), int(N))
    lvl = leaves(matrix, hash, **kw)
    t.levels.append(lvl)
    while lvl.shape[0] > 1:
        lvl = node_level(lvl, hash, **kw)
        t.levels.append(lvl)
    return t


def digest_bytes(d, hash: str) -> bytes:
    """One ``(8,)`` digest -> its 32-byte serialisation (BE words for SHA-256, LE for Blake3 and Poseidon2)."""
    a = np.asarray(d.get() if hasattr(d, "get") else (d.cpu().numpy() if hasattr(d, "cpu") else d))
    return a.astype(">u4" if hash.startswith("sha256") else "<u4").tobytes()
