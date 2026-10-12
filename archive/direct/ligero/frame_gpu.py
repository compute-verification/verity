"""The committer's frame-v3 trees on the GPU: ``verity/ml/kernels/hash_gpu/frame_v3.py`` behind the tree builders of
``auth.py`` (word leaves) and ``hashauth.py`` (row leaves, output words), byte-identical to their host paths (the
``frame_gpu_test.py`` pins).  ``LIGERO_COMMIT_GPU=0`` keeps the host paths; ``LIGERO_GPU_STRICT=1`` makes a missing
GPU path on a cupy host an error rather than a silent host fallback.

Row leaves: keyed BLAKE3 (``leaf/blake3.py``) -- the rows' bytes on the device, one thread per chunk, the root fold per
row -- whose 32-byte roots are the leaf values, and the scheme's published digests (header + chunk CV limbs) from the
same chunk CVs.  :func:`blake3_chain` is the prover's carried state of every column from the same kernel (the
``row_sponges`` fast path of ``hashchain.row_chain``).
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import numpy as np

ENABLED = os.environ.get("LIGERO_COMMIT_GPU", "1") != "0"
ROW_SCHEMES = ("blake3", "blake3-xob")  # leaf schemes whose row digests run here (one leaf, two circuits)

try:
    _shared = str(Path(__file__).resolve().parents[3] / "verity" / "ml" / "kernels")
    if _shared not in sys.path:
        sys.path.insert(0, _shared)
    import cupy as cp
    from hash_gpu import frame_v3 as fv3

    _ERR = None
except Exception as _e:  # noqa: BLE001 -- no cupy / no hash_gpu: the host paths stay
    cp = fv3 = None
    _ERR = _e


def available() -> bool:
    if not ENABLED:
        return False
    if fv3 is None:
        if os.environ.get("LIGERO_GPU_STRICT") and not isinstance(_ERR, ImportError):
            raise RuntimeError(f"frame_gpu: the GPU committer is unavailable ({_ERR})")
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


@contextlib.contextmanager
def _torch_stream():
    """cupy launches on torch's current stream (the rows come from torch)."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            yield
    else:
        yield


def _dev(x):
    """A device array (torch tensor or cupy) or a host array as a cupy array."""
    return cp.asarray(x)


def row_bytes(rows, word_bits: int):
    """(N, n_words) integer words (torch / cupy on the device, or numpy) -> (N, n_words * word_bits / 8) uint8 cupy: the
    row's little-endian words (``leaf/blake3.row_bytes``)."""
    r = _dev(rows)
    if word_bits == 8:
        if r.dtype in (cp.uint8, cp.int8):
            return cp.ascontiguousarray(r).view(cp.uint8)
        if bool(((r < 0) | (r >= 256)).any()):
            raise ValueError("word out of range")
        return r.astype(cp.uint8)
    if word_bits == 16:
        if r.dtype in (cp.uint16, cp.int16):
            return cp.ascontiguousarray(r).view(cp.uint8).reshape(r.shape[0], -1)
        if bool(((r < 0) | (r >= 1 << 16)).any()):
            raise ValueError("word out of range")
        return r.astype(cp.uint16).view(cp.uint8).reshape(r.shape[0], -1)
    raise ValueError(f"word_bits {word_bits}: rows are bytes or little-endian u16 words")


def word_tree(name: str, binding: bytes, words, *, nbytes: int, schema: str):
    """The ``auth.py`` / ``hashauth.py`` word tree (one ``nbytes``-byte big-endian word per leaf, rank = position = index)
    built on the device; ``words`` a host or device integer array."""
    from .auth import LIMITS, OWNER, TreeRef, WordTree, domain_for
    with _torch_stream():
        w = _dev(words).ravel()
        count = int(w.shape[0])
        LIMITS.enforce("max_positions", count)
        if nbytes == 2:
            w, vmode = (w.view(cp.uint16) if w.dtype == cp.int16 else w.astype(cp.uint16, copy=False)), fv3.VALUE_U16
        elif nbytes == 4:
            w, vmode = w.astype(cp.uint32, copy=False), fv3.VALUE_U32
        else:
            raise ValueError("word leaves are 2 or 4 bytes")
        domain = domain_for(binding, OWNER[name], count)
        t = fv3.build_tree(domain.domain_id, count, schema, cp.ascontiguousarray(w), vmode=vmode)
        return WordTree.on_device(TreeRef(name, bytes(binding), OWNER[name], count, t.root), t)


def _limbs(x):
    """(..., 8) u32 words -> (..., 16) 16-bit limbs, low first (``leaf/blake3.cvs_to_elems``)."""
    x = x.astype(cp.int64)
    return cp.stack([x & 0xFFFF, x >> 16], -1).reshape(*x.shape[:-1], 16)


def blake3_rows(rows, *, word_bits: int, role: int, block_out: bool = False):
    """Keyed BLAKE3 of every row on the device: ``(cvs, roots[, block_out])`` (``hash_gpu.frame_v3.blake3_keyed_rows``)
    and the row bytes."""
    from .leaf import blake3 as b3
    rb = row_bytes(rows, word_bits)
    return fv3.blake3_keyed_rows(rb, b3.key_for(role), block_out=block_out), rb


def blake3_digests(cvs, role: int, ncm: int | None = None):
    """(N, n_chunks, 8) chunk CVs -> (N, 1 + 16 ncm) int64 cupy: the scheme's published digests (``cvs_to_elems``; ``ncm`` the
    frame's chunk slots, default the registry instance's)."""
    from .leaf import blake3 as b3
    global _DIGESTS
    ncm = b3.N_CHUNKS_MAX if ncm is None else ncm
    n, nch = int(cvs.shape[0]), int(cvs.shape[1])
    if nch > ncm:
        raise ValueError(f"{nch} chunks exceed the frame's {ncm}")
    if _DIGESTS is None:
        _DIGESTS = cp.ElementwiseKernel(
            "raw uint32 cvs, int32 nch, int32 width, int64 header", "int64 out",
            "long long r = i / width; int e = (int)(i % width), k = e - 1;"
            "out = e == 0 ? header : (k / 16 >= nch ? 0ll"
            " : (long long)((cvs[(r * nch + k / 16) * 8 + (k % 16) / 2] >> (16 * (k & 1))) & 0xFFFFu));",
            "blake3_digest_elems")
    width = b3.digest_elems_of(ncm)
    out = cp.empty((n, width), dtype=cp.int64)
    _DIGESTS(cp.ascontiguousarray(cvs), np.int32(nch), np.int32(width), np.int64(b3.header(nch, role, ncm)), out)
    return out


_DIGESTS = None


def row_tree(name: str, binding: bytes, rows, *, word_bits: int, role: int, scheme, expect=None):
    """The ``hashauth.build_row_tree`` of a keyed-BLAKE3 row scheme on the device: (tree, digests (count, DIGEST_ELEMS)
    int64 host) from ``rows`` (N, n_words) words (device or host).  ``expect``: digests (host or device) that must be
    these."""
    from .auth import LIMITS, OWNER, TreeRef, WordTree, domain_for
    if scheme.name not in ROW_SCHEMES:
        raise ValueError(f"no GPU row digest for leaf scheme {scheme.name!r}")
    with _torch_stream():
        (cvs, roots), _ = blake3_rows(rows, word_bits=word_bits, role=role)
        count = int(cvs.shape[0])
        LIMITS.enforce("max_positions", count)
        domain = domain_for(binding, OWNER[name], count)
        t = fv3.build_tree(domain.domain_id, count, scheme.schema, roots.view(cp.uint8).reshape(count, 32), vlen=32)
        dg = blake3_digests(cvs, role, getattr(scheme, "n_chunks_max", None))
        if expect is not None:
            e = _dev(expect)
            if e.shape != dg.shape or not bool((e.astype(cp.int64, copy=False) == dg).all()):
                raise ValueError(f"leaf {scheme.name}: the chain digests are not the committed rows' digests")
        dg = dg.get()
    return WordTree.on_device(TreeRef(name, bytes(binding), OWNER[name], count, t.root), t), dg


def blake3_chain(words, *, word_bits: int, role: int, row_words: int, n: int, steps: int, ncm: int | None = None):
    """``Blake3Leaf.row_sponges``: ``words`` (n * steps, k) int64 CUDA torch, row-major -> ``(carry (n, steps,
    carry_elems), digests (n, digest_elems))`` int64 torch = ``carry_values`` / ``native`` of the rows (``ncm``: the frame's
    chunk slots)."""
    import torch
    from .leaf import blake3 as b3
    ncm = b3.N_CHUNKS_MAX if ncm is None else ncm
    ce, hm = b3.carry_elems_of(ncm), b3.hold_msg_of(ncm)
    dev = words.device
    rows = words.reshape(n, row_words)
    with _torch_stream():
        (cvs, _roots, bo), rb = blake3_rows(rows, word_bits=word_bits, role=role, block_out=True)
        nb = int(bo.shape[1])
        half = steps == 2 * nb
        if not half and nb % steps:
            raise ValueError("whole (or half) blocks per column")
        nblk = 1 if half else nb // steps
        car = cp.zeros((n, steps, ce), dtype=cp.int64)
        cols = cp.arange(2 if half else 1, steps)
        j = cols // 2 - 1 if half else cols * nblk - 1          # the last block of the previous column
        if cols.size:
            car[:, cols, 0:16] = _limbs(bo[:, j, :])
        col_of = (lambda jj: 2 * jj + 1) if half else (lambda jj: jj // nblk)
        nch = int(cvs.shape[1])
        for c in range(nch - 1):
            end = col_of(min(16 * (c + 1), nb) - 1)             # the column that ends chunk c
            car[:, end + 1:, 16 * (c + 1):16 * (c + 2)] = _limbs(cvs[:, c, :])[:, None, :]
        if half:                                               # the msg hold: the previous column's 16 limbs
            prev = rb.reshape(n, steps, 16, 2).astype(cp.int64)
            car[:, 1:, hm:hm + 16] = prev[:, :-1, :, 0] | (prev[:, :-1, :, 1] << 8)
        car[:, :, ce - 1] = cp.arange(steps)[None, :]
        dg = blake3_digests(cvs, role, ncm)
        return torch.as_tensor(car, device=dev), torch.as_tensor(dg, device=dev)
