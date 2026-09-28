"""FA2's per-iteration `Check_inf` as a static: the attention chain `AttnBlock_v5` / `AttentionHead_v5` / `Attention_v5`.

vllm-flash-attn 506341a `flash_fwd_kernel.h::compute_attn_1rowblock_splitkv` runs a tile's first `n_masking_steps` key blocks (from
`n_block_max - 1` down) with `softmax_rescale_o<..., Check_inf = Is_causal || Is_local>` and the rest with `Check_inf = Is_local`
(false): `n_masking_steps` is `ceil(kBlockM / kBlockN) + 1` in a causal launch (varlen, so not `Is_even_MN`) and 1 in a non-causal one
(`varlen_fwd` makes a launch with `max_seqlen_q == 1` non-causal).  With `Check_inf` false the rescale reads the running max as it is,
`exp2f((m_prev - m) * scale_log2)`; `softmax.h::scale_apply_exp2` maps a running max of -inf to 0 in every block.  So, unlike FA3
(`fa3_check_inf`, whose unguarded iterations read the running max as it is in both places), only the rescale goes unguarded.
`AttnBlock_v2` guards both on every block; on a row whose running max is still -inf at an unguarded block the two differ (the kernel's
rescale is exp2(NaN), the guarded block's 0).

`AttentionHead_v5` takes the lowest key block the kernel guards as the static `MASKED_FROM` (`attention_vocab.fa2_masked_from`): block b runs
`AttnBlock_v5{..., CHECK = b >= MASKED_FROM}`.  The row's first block is always at or above it.  `targets.attention_spec` binds `Attention_v5`
on an FA2 target whose family registers `fa2_construction = "check-inf-per-iteration"`; the other FA2 targets keep their accepted chains.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Affine, Coll as _Coll, Explicit, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.gemm import DotBf16 as DotBf16V2
from verity.ml.circuits.ops.prims import ZERO32, const
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.b1 import ZERO16, _lane, _scale_log2_bits
from verity_vllm.program.registry.prims import A16, BF16, F32


@composite("AttnBlock", 5, ["D", "NVIS", "FIRST", "BN", "DOT", "CHECK"],
           lambda S: (((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
                        ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)))),
                      Tuple(F32, Array(4, F32), Array(S.D, F32))),
           conformance="not-yet-established")
def AttnBlockV5(B, S, q, k, v, m_prev, l_prev, o_prev):
    """`AttnBlock_v2` with FA2's per-iteration `Check_inf` as the static `CHECK`: with it the rescale reads the guarded running max
    (`GuardNegInfZero`), as in v2; without it (an unmasked iteration) the rescale reads the running max as it is.  `max * scale` reads the
    guarded running max in every block.  `FIRST` needs `CHECK` (the first iteration runs `Check_inf`)."""
    return _check_inf_block(B, S, q, k, v, m_prev, l_prev, o_prev, None)


def _check_inf_block(B, S, q, k, v, m_prev, l_prev, o_prev, scale_log2):
    D, NVIS, FIRST, BN, DOT, CHECK = S.D, S.NVIS, S.FIRST, S.BN, S.DOT, S.CHECK
    if FIRST and not CHECK:
        raise ValueError("AttnBlock_v5: the first iteration runs Check_inf (FIRST needs CHECK)")
    dot = bind(DotBf16V2, K=D, DOT=DOT)
    s = [B.call(dot, q, k[c]) for c in range(NVIS)]  # A: S = Q K^T, chained k-steps, acc 0 per block
    bmax = s[0]
    for c in range(1, NVIS):
        bmax = B.call(P.F32Max, bmax, s[c])  # B1: row max
    m_new = bmax if FIRST else B.call(P.F32Max, m_prev, bmax)
    m_use = B.call(P.GuardNegInfZero, m_new)
    if scale_log2 is None:
        scale_log2 = B.call(const(32, _scale_log2_bits(D)))
    if not FIRST:  # B2: rescale factor
        scores_scale = B.call(P.MufuEx2Ftz, B.call(P.F32MulFtz, B.call(P.F32SubFtz, m_prev, m_use if CHECK else m_new), scale_log2))
    max_scaled = B.call(P.F32MulFtz, m_use, scale_log2)
    p = [B.call(P.MufuEx2Ftz, B.call(P.F32FmaSubFtz, s[c], scale_log2, max_scaled)) for c in range(NVIS)]  # B3
    # B4: per-lane sequential sums; the block's first add is contracted with the rescale
    lanes = []
    for ln in range(4):
        cols = [c for c in range(NVIS) if _lane(c) == ln]
        if FIRST:
            acc = p[cols[0]] if cols else B.call(ZERO32)
        else:
            acc = B.call(P.F32FmaFtz, scores_scale, l_prev[ln], p[cols[0]] if cols else B.call(ZERO32))
        for c in cols[1:]:
            acc = B.call(P.F32AddFtz, acc, p[c])
        lanes.append(acc)
    # B5 + C: P -> bf16, O = (O * scale) + P V in BN/16 ascending k-steps of 16 keys
    pb = [B.call(P.F2fpBf16, x) for x in p]
    zero16 = B.call(ZERO16) if NVIS < BN else None
    o = [B.call(ZERO32) for _ in range(D)] if FIRST else [B.call(P.F32MulFtz, o_prev[d], scores_scale) for d in range(D)]
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break  # all-masked k-step: P = 0 exactly, no accumulator change
        prow = _Coll(A16, concat([(pb[k0 + j] if k0 + j < NVIS else zero16).refs for j in range(16)]))
        vblk = v[k0:k0 + 16]
        vt_refs = strided_view(vblk, 0, 16, 1, D, 16 * D) if isinstance(vblk.refs, Affine) else \
            Explicit(tuple(vblk.refs[j * D + d] for d in range(D) for j in range(16)))
        vt = _Coll(Array(D, A16), vt_refs)
        o = list(B.batch(DOT, array_of(o), prow, vt, axes=(0, None, 0)))
    return tuple_of(m_new, array_of(lanes), array_of(o))


@composite("AttentionHead", 5, ["T", "D", "BN", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
                      Array(S.D, BF16)),
           conformance="not-yet-established")
def AttentionHeadV5(B, S, q, k, v):
    """`AttentionHead_v2` over `AttnBlock_v5`: causal attention of one query at position T-1 over keys 0..T-1, ceil(T/BN) blocks visited
    last -> first, block b with `CHECK = b >= MASKED_FROM` (FA2's masking iterations; `attention_vocab.fa2_masked_from`); epilogue
    `sum = (l0 + l2) + (l1 + l3)`, `inv = INV(sum)`, `out = bf16(o * inv)`."""
    return _check_inf_head(B, S, q, k, v, AttnBlockV5, ())


def _check_inf_head(B, S, q, k, v, block, extra):
    T, D, BN, DOT, INV, MF = S.T, S.D, S.BN, S.DOT, S.INV, S.MASKED_FROM
    nb = (T + BN - 1) // BN
    if not 0 <= MF <= nb - 1:
        raise ValueError(f"AttentionHead_v5: MASKED_FROM {MF} outside blocks 0..{nb - 1} (the first block is always a Check_inf iteration)")
    m = l = o = None
    zero_row = None
    for b in range(nb - 1, -1, -1):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb, vb = k[lo:hi], v[lo:hi]
        if nvis < BN:  # pad the block's K/V rows with zeros (Clear_OOB_MN); masked columns are never read
            if zero_row is None:
                zero_row = _Coll(Array(D, BF16), concat([B.call(ZERO16).refs] * D))
            pad = [zero_row.refs] * (BN - nvis)
            kb = _Coll(Array(BN, Array(D, BF16)), concat([kb.refs] + pad))
            vb = _Coll(Array(BN, Array(D, BF16)), concat([vb.refs] + pad))
        first = b == nb - 1
        blk = bind(block, D=D, NVIS=nvis, FIRST=first, BN=BN, DOT=DOT, CHECK=b >= MF)
        if first:
            z = B.call(ZERO32)
            st = B.call(blk, q, kb, vb, z, _Coll(Array(4, F32), concat([z.refs] * 4)), _Coll(Array(D, F32), concat([z.refs] * D)), *extra)
        else:
            st = B.call(blk, q, kb, vb, m, l, o, *extra)
        m, l, o = st[0], st[1], st[2]
    total = B.call(P.F32AddFtz, B.call(P.F32AddFtz, l[0], l[2]), B.call(P.F32AddFtz, l[1], l[3]))
    inv = B.call(INV, total)
    y = B.batch(P.F32MulFtz, o, inv, axes=(0, None))
    return B.batch(P.F2fpBf16, y, axes=(0,))


@composite("Attention", 5, ["T", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="not-yet-established")
def AttentionV5(B, S, q, kc, vc):
    """`Attention_v2` over `AttentionHead_v5{T, D, BN, DOT, INV, MASKED_FROM}`: GQA, head h reads KV head h // (NH/KVH); every head of the
    row shares its tile, so one MASKED_FROM."""
    return _check_inf_attention(B, S, q, kc, vc, AttentionHeadV5, ())


def _check_inf_attention(B, S, q, kc, vc, head, extra):
    T, NH, KVH, D = S.T, S.NH, S.KVH, S.D
    G = NH // KVH
    head = bind(head, T=T, D=D, BN=S.BN, DOT=S.DOT, INV=S.INV, MASKED_FROM=S.MASKED_FROM)
    outs = []
    for g in range(KVH):
        kg = _Coll(Array(T, Array(D, BF16)), strided_view(kc, g * D, D, KVH * D, 1, T * D))
        vg = _Coll(Array(T, Array(D, BF16)), strided_view(vc, g * D, D, KVH * D, 1, T * D))
        outs.append(B.batch(head, q[g * G:(g + 1) * G], kg, vg, *extra, axes=(0, None, None) + (None,) * len(extra)))
    return _Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


# The v5 chain with the softmax scale an input (`scale_log2`, as `value_inputs.AttentionV10` reads it for the v3 chain).


@composite("AttnBlock", 11, ["D", "NVIS", "FIRST", "BN", "DOT", "CHECK"],
           lambda S: (((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
                        ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)), ("scale_log2", F32))),
                      Tuple(F32, Array(4, F32), Array(S.D, F32))),
           conformance="not-yet-established: AttnBlock_v5's body with the scale an input")
def AttnBlockV11(B, S, q, k, v, m_prev, l_prev, o_prev, scale_log2):
    """`AttnBlock_v5{D, NVIS, FIRST, BN, DOT, CHECK}` with `scale_log2` an input."""
    return _check_inf_block(B, S, q, k, v, m_prev, l_prev, o_prev, scale_log2)


@composite("AttentionHead", 11, ["T", "D", "BN", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16))),
                       ("scale_log2", F32)), Array(S.D, BF16)),
           conformance="not-yet-established: AttentionHead_v5's body over AttnBlock_v11")
def AttentionHeadV11(B, S, q, k, v, scale_log2):
    """`AttentionHead_v5{T, D, BN, DOT, INV, MASKED_FROM}` over `AttnBlock_v11`, every block reading the one `scale_log2`."""
    return _check_inf_head(B, S, q, k, v, AttnBlockV11, (scale_log2,))


@composite("Attention", 11, ["T", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))), ("scale_log2", F32)),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="not-yet-established: Attention_v5's body over AttentionHead_v11")
def AttentionV11(B, S, q, kc, vc, scale_log2):
    """`Attention_v5{T, NH, KVH, D, BN, DOT, INV, MASKED_FROM}` over `AttentionHead_v11`, every head reading the one `scale_log2`."""
    return _check_inf_attention(B, S, q, kc, vc, AttentionHeadV11, (scale_log2,))


FA2_CHECK_INF_COMPOSITES = [AttnBlockV5, AttentionHeadV5, AttentionV5, AttnBlockV11, AttentionHeadV11, AttentionV11]

__all__ = ["AttnBlockV5", "AttentionHeadV5", "AttentionV5", "AttnBlockV11", "AttentionHeadV11", "AttentionV11", "FA2_CHECK_INF_COMPOSITES"]
