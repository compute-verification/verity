"""Numpy twins of `registry.sparse_mla_split`'s Definitions, vectorised over rows: `MufuLg2Ftz_v1` (`mufu_lg2_words`, with
`undetermined_words`), `SparseMlaSplitEnd_v1` (`split_end_rows`), `SparseMlaCombine_v1` (`combine_rows`),
`SparseMlaSplitPart_v1` (`part_rows`, one split's partial and lse) and `SparseMlaHeadSplit_v1` (`head_split_rows`, which at NS = 1
is `SparseMlaHead_v1`).  The blocks are `derived_rows.attn_block` over
q = [q_rope | q_nope], k = [k_pe | kv_c] and v = [kv_c | 0] (`AttnBlock_v4`'s steps, which `SparseMlaBlock_v1` restates), the step
`tc_dot16_hopper` and the inverse `fa3_inv_sum`; the captures' fast path, checked word for word against the Definitions in
`tests/program/test_sparse_mla_split.py`.  A row whose dot reads a non-finite operand is flagged and never claimed exact
(`attn_block`'s ``nonfinite``)."""
from __future__ import annotations

import numpy as np

from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.registry import sparse_mla as M
from verity.ml.definitions.registry import sparse_mla_split as W

U32 = np.uint32


def mufu_lg2_words(x, c0: str = "C0") -> np.ndarray:
    """`MufuLg2Ftz_v1` on f32 words (``c0`` "C0_max" for the other end of the measured range)."""
    f = W.lg2_fit(c0)
    x = np.asarray(x, dtype=U32)
    E, m = ((x >> 23) & 0xFF).astype(np.int64), (x & 0x7FFFFF).astype(np.int64)
    g, u = m >> f.nb, m & ((1 << f.nb) - 1)
    c0w, c1, c2 = (np.asarray(c, np.int64)[g] for c in (f.C0, f.C1, f.C2))
    v = np.where(m == 0, 0, (c0w + c1 * u + ((c2 * f.square(u)) << f.square_shift)) >> f.shift)
    neg = E < 127
    Mw = ((((E + 1) & 0x7F) << 36) | v) ^ np.where(neg, (1 << 43) - 1, 0)
    n = np.frexp(Mw.astype(np.float64))[1].astype(np.int64)        # bit_length, exact below 2^53
    mant = np.where(n >= 24, Mw >> np.maximum(n - 24, 0), Mw << np.maximum(24 - n, 0)) & 0x7FFFFF
    out = np.where(Mw == 0, 0, (neg.astype(np.int64) << 31) | ((n + 90) << 23) | mant)
    out = np.where(E == 255, np.where(m != 0, W.QNAN, W.INF), out)
    out = np.where((x >> 31).astype(bool) & (E != 0), W.QNAN, out)
    return np.where(E == 0, W.NEG_INF, out).astype(U32)


def undetermined_words(x, words=None) -> np.ndarray:
    """`sparse_mla_split.mufu_lg2_undetermined` on f32 words; ``words`` is `mufu_lg2_words(x)` when the caller has it."""
    x = np.asarray(x, dtype=U32)
    E, m = (x >> 23) & 0xFF, x & 0x7FFFFF
    normal = ((x >> 31) == 0) & (E != 0) & (E != 255)
    words = mufu_lg2_words(x) if words is None else np.asarray(words, dtype=U32)
    return normal & (((m == 0) & (E < 127)) | (words != mufu_lg2_words(x, "C0_max")))


def nez(x) -> np.ndarray:
    """`F32NeZeroFtz_v1`."""
    x = np.asarray(x, dtype=U32)
    return ((x & 0x7F800000) != 0) & ~(((x & 0x7F800000) == 0x7F800000) & ((x & 0x7FFFFF) != 0))


def gtz(x) -> np.ndarray:
    """`F32GtZeroFtz_v1`."""
    return nez(x) & ((np.asarray(x, dtype=U32) >> 31) == 0)


def split_end_rows(m, l, o, *, scale: float, inv=DR.fa3_inv_sum, scale_bits: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`SparseMlaSplitEnd_v1{DV, INV, SCALE}`: m (R,), l (R, 4), o (R, DV) f32 words -> (o_partial (R, DV), lse (R,)); the running
    max's multiplier ``scale_bits`` (`sparse_mla_split.split_end`'s, default `lse_scale_bits(scale)`)."""
    m, l, o = (np.asarray(a, dtype=U32) for a in (m, l, o))
    tot = DR.add_ftz(DR.add_ftz(l[:, 0], l[:, 2]), DR.add_ftz(l[:, 1], l[:, 3]))
    part = DR.mul_ftz(o, inv(tot)[:, None])
    c = np.full(tot.shape, W.lse_scale_bits(scale) if scale_bits is None else scale_bits, dtype=U32)
    lse = DR.fma_ftz(m, c, DR.mul_ftz(mufu_lg2_words(tot), np.full(tot.shape, W.LN2, dtype=U32)))
    return part, np.where(nez(tot), lse, U32(W.NEG_INF)).astype(U32)


def _lane0(op, v: np.ndarray) -> np.ndarray:
    """Lane 0 of the warp butterfly of ``op`` over v (R, LANES): offsets LANES/2 .. 1, ``op(own, partner)``."""
    off = v.shape[1] // 2
    while off:
        v = op(v[:, :off], v[:, off:2 * off])
        off //= 2
    return v[:, 0]


def combine_rows(lse, o, *, inv=DR.fa3_inv_sum) -> np.ndarray:
    """`SparseMlaCombine_v1{NSPLIT, DV, INV}`: lse (R, NSPLIT), o (R, NSPLIT, DV) f32 words -> (R, DV) bf16 words."""
    lse, o = np.asarray(lse, dtype=U32), np.asarray(o, dtype=U32)
    R, NSPLIT = lse.shape
    if not 1 <= NSPLIT <= W.LANES:
        raise ValueError(f"combine_rows: NSPLIT={NSPLIT} splits, the warp holds 1 to {W.LANES}")
    lanes = np.concatenate([lse, np.full((R, W.LANES - NSPLIT), W.NEG_INF, dtype=U32)], axis=1)
    cur = DR.guard_neg_inf_zero(_lane0(DR.max_fmnmx, lanes))
    log2e = np.full(lanes.shape, W.LOG2E, dtype=U32)
    scale = DR.mufu_ex2_ftz(DR.mul_ftz(DR.sub_ftz(lanes, cur[:, None]), log2e))
    inv_sum = inv(_lane0(DR.add_ftz, DR.add_ftz(np.zeros_like(scale), scale)))
    acc = np.zeros((R, o.shape[2]), dtype=U32)
    for s in range(NSPLIT):
        w = DR.mul_ftz(scale[:, s], inv_sum)
        acc = np.where(gtz(w)[:, None], DR.fma_ftz(np.broadcast_to(w[:, None], acc.shape), o[:, s], acc), acc)
    return DR.f2fp_bf16(acc)


def _operands(what: str, q_nope, q_rope, kv, T: int, DR_: int, DV: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The blocks' operands of a head over T keys: q = [q_rope | q_nope], k = [k_pe | kv_c] and v = [kv_c | 0], each (R, ..)."""
    q_nope, q_rope, kv = (np.asarray(a, dtype=np.uint16) for a in (q_nope, q_rope, kv))
    R, KW = q_nope.shape[0], DR_ + DV
    if kv.ndim == 2:
        kv = np.broadcast_to(kv, (R,) + kv.shape)
    if kv.shape[1:] != (T, KW):
        raise ValueError(f"{what}: kv {kv.shape[1:]}, T={T} rows of {KW}")
    return (np.concatenate([q_rope, q_nope], axis=1), np.concatenate([kv[:, :, DV:], kv[:, :, :DV]], axis=2),
            np.concatenate([kv[:, :, :DV], np.zeros((R, T, DR_), dtype=np.uint16)], axis=2))


def _split(q, kb, vb, b0: int, b1: int, *, T: int, BN: int, DV: int, sbits: int, dot16) -> tuple[tuple, np.ndarray]:
    """One split: blocks [b0, b1) last to first (its first visited one FIRST, the keys' last block zero-padded) -> its end's
    (m, l, o[:, :DV]) and the rows whose dot read a non-finite operand."""
    R, KW = q.shape
    nf = np.zeros(R, dtype=bool)
    m = l = o = None
    for b in range(b1 - 1, b0 - 1, -1):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        kblk, vblk = (np.zeros((R, BN, KW), dtype=np.uint16) for _ in range(2))
        kblk[:, :hi - lo], vblk[:, :hi - lo] = kb[:, lo:hi], vb[:, lo:hi]
        first = b == b1 - 1
        m, l, o, f = DR.attn_block(q, kblk, vblk, m, l, o, D=KW, NVIS=hi - lo, FIRST=first, dot16=dot16, check=first,
                                   scale_bits=sbits)
        nf |= f
    return (m, l, o[:, :DV]), nf


def part_rows(q_nope, q_rope, kv, *, T: int, DR_: int = 64, DV: int = 512, BN: int = W.BLOCK_N, scale: float,
              dot16=DR.tc_dot16_hopper, inv=DR.fa3_inv_sum) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`SparseMlaSplitPart_v1{T, DR, DV, BN, DOT, INV, SCALE}`: q_nope (R, DV), q_rope (R, DR), kv (R, T, DV + DR) or (T, DV + DR)
    bf16 words -> (o_partial (R, DV) f32 words, lse (R,) f32 words, nonfinite (R,)): one split over all its T keys."""
    if T < 1:
        raise ValueError(f"part_rows: T={T} keys")
    q, kb, vb = _operands("part_rows", q_nope, q_rope, kv, T, DR_, DV)
    (m, l, o), nf = _split(q, kb, vb, 0, -(-T // BN), T=T, BN=BN, DV=DV, sbits=M.scale_log2_bits(scale), dot16=dot16)
    part, lse = split_end_rows(m, l, o, scale=scale, inv=inv)
    return part, lse, nf


def head_split_rows(q_nope, q_rope, kv, *, T: int, NS: int, DR_: int = 64, DV: int = 512, BN: int = W.BLOCK_N, scale: float,
                    dot16=DR.tc_dot16_hopper, inv=DR.fa3_inv_sum) -> tuple[np.ndarray, np.ndarray]:
    """`SparseMlaHeadSplit_v1{T, NS, DR, DV, BN, DOT, INV, SCALE}`: q_nope (R, DV), q_rope (R, DR), kv (R, T, DV + DR) or (T, DV + DR)
    bf16 words -> (out (R, DV) bf16 words, nonfinite (R,)).  NS = 1 is `SparseMlaHead_v1`."""
    W.check_split_statics(T, NS)
    q, kb, vb = _operands("head_split_rows", q_nope, q_rope, kv, T, DR_, DV)
    sbits = M.scale_log2_bits(scale)
    nf = np.zeros(q.shape[0], dtype=bool)
    ends = []
    for b0, b1 in W.split_blocks(T, NS, BN):
        e, f = _split(q, kb, vb, b0, b1, T=T, BN=BN, DV=DV, sbits=sbits, dot16=dot16)
        ends.append(e)
        nf |= f
    if NS == 1:
        m, l, o = ends[0]
        tot = DR.add_ftz(DR.add_ftz(l[:, 0], l[:, 2]), DR.add_ftz(l[:, 1], l[:, 3]))
        return DR.f2fp_bf16(DR.mul_ftz(o, inv(tot)[:, None])), nf
    parts = [split_end_rows(m, l, o, scale=scale, inv=inv) for m, l, o in ends]
    return combine_rows(np.stack([p[1] for p in parts], axis=1), np.stack([p[0] for p in parts], axis=1), inv=inv), nf
