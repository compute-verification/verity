"""Numpy twins of `registry.fa2_split`'s Definitions, vectorised over rows: `Fa2SplitEnd_v1` (`sparse_mla_split_twins.split_end_rows`
at FA2's multiplier), `Fa2Combine_v1` (`combine_rows`), `Fa2AttentionHeadSplit_v1` (`head_split_rows`, at NS = 1
`AttentionHead_v5`) and `Fa2AttentionSplit_v1` (`attention_rows`).  The blocks are `derived_rows.attn_block` with `rescale_only`
(`AttnBlock_v5`'s steps); the Match's fast path, checked word for word against the Definitions and their Boolean versions in
`tests/program/test_fa2_split.py`.  A row whose dot reads a non-finite operand is flagged and never claimed exact (`attn_block`'s
``nonfinite``)."""
from __future__ import annotations

import numpy as np

from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics import sparse_mla_split_twins as SMT
from verity.ml.definitions.registry import fa2_split as W

U32 = np.uint32


def _full(shape, word: int) -> np.ndarray:
    return np.full(shape, word, dtype=U32)


def combine_rows(lse, o) -> np.ndarray:
    """`Fa2Combine_v1{NS, D}`: lse (R, NS), o (R, NS, D) f32 words -> (R, D) bf16 words."""
    lse, o = np.asarray(lse, dtype=U32), np.asarray(o, dtype=U32)
    (R, NS), D = lse.shape, o.shape[2]
    if not 2 <= NS <= W.MAX_SPLITS:
        raise ValueError(f"combine_rows: NS={NS} splits, the combine runs 2 to {W.MAX_SPLITS}")
    lanes, L = W.combine_geometry(NS, D)
    held = np.concatenate([lse, _full((R, L * lanes - NS), W.NEG_INF)], axis=1).reshape(R, L, lanes)
    mx = held[:, 0]
    for i in range(1, L):
        mx = DR.max_fmnmx(mx, held[:, i])
    cur = DR.guard_neg_inf_zero(SMT._lane0(DR.max_fmnmx, mx))
    scaled = DR.mufu_ex2_ftz(DR.mul_ftz(DR.sub_ftz(held, cur[:, None, None]), _full(held.shape, W.LOG2E)))
    tot = scaled[:, 0]
    for i in range(1, L):
        tot = DR.add_ftz(tot, scaled[:, i])
    tot = SMT._lane0(DR.add_ftz, tot)
    logsum = np.where(SMT.nez(tot), DR.fma_ftz(SMT.mufu_lg2_words(tot), _full(R, W.LN2), cur), U32(W.INF)).astype(U32)
    acc = np.zeros((R, D), dtype=U32)
    for s in range(NS):
        w = DR.mufu_ex2_ftz(DR.mul_ftz(DR.sub_ftz(lse[:, s], logsum), _full(R, W.LOG2E)))
        acc = DR.fma_ftz(np.broadcast_to(w[:, None], acc.shape), o[:, s], acc)
    return DR.f2fp_bf16(acc)


def head_split_rows(q, k, v, *, T: int, NS: int, BN: int, masked_from: int, dot16=DR.tc_dot16_hopper,
                    inv=DR.fa2_inv_sum) -> tuple[np.ndarray, np.ndarray]:
    """`Fa2AttentionHeadSplit_v1{T, NS, D, BN, DOT, INV, MASKED_FROM}` over rows: q (R, D), k, v (R, T, D) bf16 words -> (out (R, D)
    bf16 words, nonfinite (R,)).  NS = 1 is `AttentionHead_v5`."""
    q, k, v = (np.asarray(a, dtype=np.uint16) for a in (q, k, v))
    R, D = q.shape
    W.check_split_statics(T, NS, D)
    if NS == 1:
        st = DR.attention_head_states(q, k, v, BN, dot16=dot16, inv=inv, masked_from=masked_from, rescale_only=True)
        return st["out"], st["nonfinite"]
    nf = np.zeros(R, dtype=bool)
    lses, parts = [], []
    for b0, b1 in W.split_ranges(T, NS, BN):
        if b0 >= b1:
            lses.append(_full(R, W.NEG_INF))
            parts.append(np.zeros((R, D), dtype=U32))
            continue
        m = l = o = None
        for b in range(b1 - 1, b0 - 1, -1):
            lo, hi = b * BN, min(T, (b + 1) * BN)
            kb, vb = (np.zeros((R, BN, D), dtype=np.uint16) for _ in range(2))
            kb[:, :hi - lo], vb[:, :hi - lo] = k[:, lo:hi], v[:, lo:hi]
            first = b == b1 - 1
            m, l, o, f = DR.attn_block(q, kb, vb, m, l, o, D=D, NVIS=hi - lo, FIRST=first, dot16=dot16,
                                       check=first or b >= masked_from, rescale_only=True)
            nf |= f
        part, lse = SMT.split_end_rows(m, l, o, scale=D ** -0.5, inv=inv, scale_bits=W.scale_bits(D ** -0.5))
        parts.append(part)
        lses.append(lse)
    return combine_rows(np.stack(lses, axis=1), np.stack(parts, axis=1)), nf


def attention_rows(q, kc, vc, *, NS: int, BN: int, masked_from: int, dot_prim: str = "HopperBF16WgmmaDot16_v1",
                   inv_prim: str = "Fa2InvSum_v1") -> tuple[np.ndarray, np.ndarray]:
    """`Fa2AttentionSplit_v1` for one query token: q (NH, D), kc / vc (T, KVH, D) bf16 words -> (out (NH, D) bf16 words,
    nonfinite (NH,)); head h reads KV head h // (NH / KVH)."""
    q, kc, vc = (np.asarray(a, dtype=np.uint16) for a in (q, kc, vc))
    NH, KVH = q.shape[0], kc.shape[1]
    kvh = np.arange(NH) // (NH // KVH)
    k, v = (np.ascontiguousarray(np.transpose(a, (1, 0, 2))[kvh]) for a in (kc, vc))
    return head_split_rows(q, k, v, T=kc.shape[0], NS=NS, BN=BN, masked_from=masked_from,
                           dot16=DR.TC_DOT16_BY_PRIM[dot_prim], inv=DR.INV_SUM_BY_PRIM[inv_prim])
