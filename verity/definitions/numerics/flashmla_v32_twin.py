"""numpy twin of FlashMLA's sparse FP8 decode at its V32 instance (6bc49418: `flash_fwd_splitkv_mla_fp8_sparse_kernel<
KernelTemplate<V32, 64>>` and its combine, on the H200), over `flashmla_twin`'s parts.

The instance differs from the one `flashmla_twin` was written for in what it reads and how it is called, not in the block, the
split's end or the combine: a cache row is 512 E4M3 bytes, four f32 scales (one per 128 elements, each rounded to bf16 before it
multiplies) and 64 bf16 words; a row is valid when its index is not -1 and nothing else; the key is 576 wide and the value its
first 512 words; there is no sink (the kernel runs its sink arm on -inf) and no second cache.  `kv_rows` is the staged row,
`decode` a token's heads at its SPLITS.
"""
from __future__ import annotations

import numpy as np

from verity.definitions.kernels import tc as CK
from verity.definitions.numerics import derived_rows as DR
from verity.definitions.numerics import flashmla_twin as T

U32 = np.uint32


def kv_rows(nope, scale, rope, valid, QT: int) -> np.ndarray:
    """The staged rows: (N, DN) e4m3 bytes, (N, DN / QT) f32 scale words, (N, DR) bf16 words, (N,) valid -> (N, DN + DR) bf16.

    An invalid row's elements are +0 (its bytes and its scales are zeroed before the multiply).  Its last DR words are whatever
    row the masked index addresses; the score they enter is replaced by -inf and the value product never reads them, so no
    output word depends on them and they are 0 here.
    """
    rope = np.asarray(rope, dtype=np.uint16)
    x = DR.f2fp_bf16(CK.e4m3_to_f32_batch(np.asarray(nope))[0])
    s = np.repeat(DR.f2fp_bf16(np.asarray(scale, dtype=U32)), QT, axis=-1)
    el = DR.f2fp_bf16(T.fmul(T._bf16(x), T._bf16(s)))
    rows = np.concatenate([el, rope], axis=-1)
    return np.where(np.asarray(valid, dtype=bool)[..., None], rows, np.uint16(0)).astype(np.uint16)


def row_valid(idx) -> np.ndarray:
    return np.asarray(idx, dtype=np.int64).astype(U32) != U32(0xFFFFFFFF)


def decode(q, nope, scale, rope, idx, *, SPLITS, QT: int, BN: int, DV: int, SCALE: float) -> np.ndarray:
    """q (NH, DK) bf16 and the token's N rows in index order (nope (N, DN), scale (N, DN / QT) f32 words, rope (N, DR), idx (N,))
    -> (NH, DV) bf16.  One split writes bf16 itself; more leave partials the combine weighs."""
    sp = tuple(SPLITS)
    valid = row_valid(idx)
    k = kv_rows(nope, scale, rope, valid, QT)
    q, sc = np.asarray(q, dtype=np.uint16), T.scale_log2(SCALE)
    sink = np.full(q.shape[0], T.F_NINF, dtype=U32)
    if len(sp) == 2:
        return T.split(q, k, valid, sc, BN, sink)[:, :DV]
    parts = [T.split(q, k[a * BN:b * BN], valid[a * BN:b * BN], sc, BN) for a, b in zip(sp, sp[1:])]
    return T.combine(np.stack([p[1] for p in parts], axis=1), np.stack([p[0][:, :DV] for p in parts], axis=1), sink)
