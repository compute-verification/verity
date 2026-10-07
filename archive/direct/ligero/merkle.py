"""Merkle commitment to the columns of the codeword matrix U (m x n): leaf c = H(column c as m little-endian uint32),
internal = H(left || right).  Host hashing: Blake3 (``blake3`` package) when importable, else SHA-256 (hashlib) --
both PLACEHOLDERS for the GPU Merkle of ``verity/ml/kernels/hash_gpu``; the hash is named in the proof header.
"""
from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import blake3 as _blake3

    def _h(data: bytes) -> bytes:
        return _blake3.blake3(data).digest()

    HASH_NAME = "blake3 (host, placeholder)"
except ImportError:                                   # pragma: no cover
    def _h(data: bytes) -> bytes:
        return hashlib.sha256(data).digest()

    HASH_NAME = "sha256 (host, placeholder)"

__all__ = ["HASH_NAME", "commit", "commit_device", "gpu_available", "hash_bytes", "open_columns", "open_device", "verify_path"]

# --- GPU Merkle (verity/ml/kernels/hash_gpu, the hash-gpu lane): Blake3 leaves and nodes on the device, bit-exact with
# the host Blake3 above, so ``verify_path`` is unchanged. -----------------------------------------------------------
try:
    import sys as _sys
    from pathlib import Path as _Path

    _shared = str(_Path(__file__).resolve().parents[3] / "verity" / "ml" / "kernels")
    if _shared not in _sys.path:
        _sys.path.insert(0, _shared)
    import cupy as _cp  # noqa: F401
    from hash_gpu import merkle as _gpu_merkle

    GPU_HASH_NAME = "blake3 (GPU, verity/ml/kernels/hash_gpu)"
except Exception as _e:  # noqa: BLE001 -- no cupy / no hash_gpu: the host path stays
    _gpu_merkle = None
    GPU_HASH_NAME = None
    # Loud, not silent (dev-4090 11:23Z: a tree shipped without kernels ran every gate green on a ~35x slower
    # host Merkle).  LIGERO_GPU_STRICT=1 turns the fallback into an error.
    import os as _os
    import warnings as _warnings

    try:
        import cupy as _cp_probe  # noqa: F401  -- cupy present => a GPU host whose tree lacks verity/ml/kernels/hash_gpu
        _has_cupy = True
    except Exception:  # noqa: BLE001
        _has_cupy = False
    if _has_cupy or _os.environ.get("LIGERO_GPU_STRICT"):
        if _os.environ.get("LIGERO_GPU_STRICT"):
            raise
        _warnings.warn(f"merkle: GPU Merkle unavailable on a cupy host ({type(_e).__name__}: {_e}); falling back to the HOST "
                       f"Blake3 Merkle (~35x slower). Ship kernels, or set LIGERO_GPU_STRICT=1 to fail instead.",
                       RuntimeWarning, stacklevel=1)


try:
    from . import commit_gpu as _commit_gpu           # the protocol-specialised kernels (lane enc-hopper)
except Exception as _e:  # noqa: BLE001
    _commit_gpu = None
    if _gpu_merkle is not None:   # cupy is there, so this is a real breakage of the fused path, not a CPU-only host
        import warnings as _warnings

        _warnings.warn(f"merkle: commit_gpu unavailable ({type(_e).__name__}: {_e}); using hash_gpu level by level (2.5x slower)",
                       RuntimeWarning, stacklevel=1)

USE_FUSED_COMMIT = True      # commit_gpu (3 launches, 2.5x the shared kernels on the H100); False = hash_gpu level by level


def gpu_available() -> bool:
    return _gpu_merkle is not None and HASH_NAME.startswith("blake3")


def commit_device(U_int32, stream=None):
    """``U_int32``: (m, n) contiguous int32 torch tensor on the device -> (root bytes, Tree).  ``stream``: the torch
    stream to launch on (default: the current one)."""
    n = int(U_int32.shape[1])
    if USE_FUSED_COMMIT and _commit_gpu is not None and n & (n - 1) == 0 and U_int32.shape[0] <= 4096:
        tree = _commit_gpu.commit(U_int32, stream=stream)
    else:
        tree = _gpu_merkle.commit(U_int32, "blake3")
    return _gpu_merkle.digest_bytes(tree.root, "blake3"), tree


def open_device(tree, cols: list[int]) -> list[list[bytes]]:
    """Authentication paths as bytes (leaf level first), from the device tree."""
    import cupy as cp

    levels = tree.levels[:-1]                                     # (N_k, 8) uint32 each, LE words
    if not levels or not cols:
        return [[] for _ in cols]
    # the siblings gathered on the device -- (t, depth, 8) -- and copied once (was: every level of the tree copied to
    # the host and t x depth small astype/tobytes calls; lane hostphase).  Same bytes: lvl[(c >> k) ^ 1] per level.
    idx = cp.asarray(np.asarray([int(c) for c in cols], dtype=np.int64))
    sib = cp.stack([lvl[(idx >> k) ^ 1] for k, lvl in enumerate(levels)], axis=1)
    raw = cp.asnumpy(sib).astype("<u4", copy=False).tobytes()
    depth = len(levels)
    return [[raw[(c * depth + k) * 32:(c * depth + k + 1) * 32] for k in range(depth)] for c in range(len(cols))]


def hash_bytes(data: bytes) -> bytes:
    return _h(data)


def leaves_of(U_cols: np.ndarray, threads: int | None = None) -> list[bytes]:
    """U_cols: (n, m) uint32 (column-major view of U) -> n leaf digests."""
    n = U_cols.shape[0]
    raw = np.ascontiguousarray(U_cols.astype(np.uint32))
    threads = threads or min(32, (os.cpu_count() or 4))
    if n < 256 or threads <= 1:
        return [_h(raw[i].tobytes()) for i in range(n)]
    chunks = np.array_split(np.arange(n), threads)

    def work(idx):
        return [_h(raw[i].tobytes()) for i in idx]

    with ThreadPoolExecutor(threads) as ex:
        parts = list(ex.map(work, chunks))
    return [d for part in parts for d in part]


def build_tree(leaves: list[bytes]) -> list[list[bytes]]:
    levels = [leaves]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        levels.append([_h(cur[i] + cur[i + 1]) for i in range(0, len(cur), 2)])
    return levels


def commit(U_cols: np.ndarray) -> tuple[bytes, list[list[bytes]]]:
    """Root and the tree (all levels, leaves first)."""
    levels = build_tree(leaves_of(U_cols))
    return levels[-1][0], levels


def open_columns(levels: list[list[bytes]], cols: list[int]) -> list[list[bytes]]:
    """Authentication path (sibling per level) for each column."""
    paths = []
    for c in cols:
        path = []
        i = c
        for lvl in levels[:-1]:
            path.append(lvl[i ^ 1])
            i >>= 1
        paths.append(path)
    return paths


def verify_path(root: bytes, col: int, column: np.ndarray, path: list[bytes]) -> bool:
    h = _h(np.ascontiguousarray(column.astype(np.uint32)).tobytes())
    i = col
    for sib in path:
        h = _h(h + sib) if i % 2 == 0 else _h(sib + h)
        i >>= 1
    return h == root
