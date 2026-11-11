"""Numpy twins of `registry.fa3_split`'s heads, vectorised over rows that share the key count: `AttentionHeadSplit_v1`
(`head_split_rows`), `AttentionHeadLse_v1` (`head_lse_rows`) and `AttentionHeadSplitLse_v1` (`head_split_lse_rows`).  The blocks are
`derived_rows.attn_block` (`AttnBlock_v4`'s steps), each split's end `sparse_mla_split_twins.split_end_rows` and the combine
`combine_rows` / `combine_lse_rows`.  The capture's fast path (`tests/properties/glm53_fa3_prefill_capture_gpu.py`), checked word
for word against the Definitions in `tests/program/test_fa3_split.py`.  A row whose dot reads a non-finite operand is flagged
(`attn_block`'s ``nonfinite``) and never claimed exact."""
from __future__ import annotations

import numpy as np

from verity.definitions.numerics import derived_rows as DR
from verity.definitions.numerics import sparse_mla_split_twins as TW
from verity.definitions.registry import fa3_split as F


def _ends(q, k, v, *, T: int, BN: int, PER: int, masked_from: int, dot16, inv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each split of `row_splits(T, BN, PER)` ended: (lse (R, NSPLIT), partial (R, NSPLIT, D), nonfinite (R,))."""
    q = np.asarray(q, dtype=np.uint16)
    R, D = q.shape
    k, v = (np.broadcast_to(np.asarray(a, dtype=np.uint16), (R, T, D)) if np.ndim(a) == 2 else np.asarray(a, dtype=np.uint16)
            for a in (k, v))
    if k.shape[1:] != (T, D) or v.shape != k.shape:
        raise ValueError(f"head_split_rows: k {k.shape}, v {v.shape}, T={T} rows of {D}")
    nf = np.zeros(R, dtype=bool)
    lses, parts = [], []
    for b0, b1 in F.row_splits(T, BN, PER):
        m = l = o = None
        for b in range(b1 - 1, b0 - 1, -1):
            lo, hi = b * BN, min(T, (b + 1) * BN)
            kb, vb = (np.zeros((R, BN, D), dtype=np.uint16) for _ in range(2))
            kb[:, :hi - lo], vb[:, :hi - lo] = k[:, lo:hi], v[:, lo:hi]
            first = b == b1 - 1
            m, l, o, f = DR.attn_block(q, kb, vb, m, l, o, D=D, NVIS=hi - lo, FIRST=first, dot16=dot16,
                                       check=first or b >= masked_from)
            nf |= f
        part, lse = TW.split_end_rows(m, l, o, scale=D ** -0.5, inv=inv)
        parts.append(part)
        lses.append(lse)
    return np.stack(lses, axis=1), np.stack(parts, axis=1), nf


def head_split_rows(q, k, v, *, T: int, BN: int, PER: int, masked_from: int, dot16=DR.tc_dot16_hopper,
                    inv=DR.fa3_inv_sum) -> tuple[np.ndarray, np.ndarray]:
    """`AttentionHeadSplit_v1{T, D, BN, DOT, INV, MASKED_FROM, PER}`: q (R, D), k and v (R, T, D) or (T, D) bf16 words ->
    (out (R, D) bf16 words, nonfinite (R,))."""
    lses, parts, nf = _ends(q, k, v, T=T, BN=BN, PER=PER, masked_from=masked_from, dot16=dot16, inv=inv)
    return TW.combine_rows(lses, parts, inv=inv), nf


def head_split_lse_rows(q, k, v, *, T: int, BN: int, PER: int, masked_from: int, dot16=DR.tc_dot16_hopper,
                        inv=DR.fa3_inv_sum) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`AttentionHeadSplitLse_v1{T, D, BN, DOT, INV, MASKED_FROM, PER}`: `head_split_rows`' operands -> (out (R, D) bf16 words,
    lse (R,) f32 words, nonfinite (R,))."""
    lses, parts, nf = _ends(q, k, v, T=T, BN=BN, PER=PER, masked_from=masked_from, dot16=dot16, inv=inv)
    out, lse = TW.combine_lse_rows(lses, parts, inv=inv)
    return out, lse, nf


def head_lse_rows(q, k, v, *, T: int, BN: int, masked_from: int, dot16=DR.tc_dot16_hopper,
                  inv=DR.fa3_inv_sum) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`AttentionHeadLse_v1{T, D, BN, DOT, INV, MASKED_FROM}`: the unsplit head, one split of all ceil(T / BN) blocks ended, its
    partial to bf16 -> (out (R, D) bf16 words, lse (R,) f32 words, nonfinite (R,))."""
    lses, parts, nf = _ends(q, k, v, T=T, BN=BN, PER=-(-T // BN), masked_from=masked_from, dot16=dot16, inv=inv)
    return DR.f2fp_bf16(parts[:, 0]), lses[:, 0], nf


__all__ = ["head_lse_rows", "head_split_lse_rows", "head_split_rows"]
