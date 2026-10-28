"""FA3's per-iteration `Check_inf` as a static: the attention chain `AttnBlock_v4` / `AttentionHead_v4` / `Attention_v4`.

FlashAttention-3's mainloop (vllm-flash-attn 506341a1 `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`, `mma()`, IntraWGOverlap) runs a tile's key
blocks in three kinds of iteration: the first (`n_block_max - 1`: `max_get_scale` / `online_softmax` with `Check_inf = true`), the
causal-masked ones (`n_block >= n_block_min_causal_local_mask`: `fwd_step(..., check_inf = true)`) and the unmasked ones below
(`fwd_step(..., check_inf = false)`).  With `Check_inf` false, `hopper/softmax.h` uses the running max as it is -- the rescale reads
`exp2f((m_prev - m) * scale_log2)` and `scale_apply_exp2` reads `max_scaled = m * scale_log2` -- where the `Check_inf` iterations first map a
running max of -inf to 0.  `AttnBlock_v2` guards on every block; on a row whose running max is still -inf at an unmasked block the two differ
(the kernel's `max_scaled` is -inf and its P and rescale NaN, the guarded block's 0, 0 and 0).

`AttentionHead_v4` takes the tile's `n_block_min_causal_local_mask` as the static `MASKED_FROM` (the lowest key block the kernel guards;
`attention_vocab.fa3_masked_from`): block b runs `AttnBlock_v4{..., CHECK = b >= MASKED_FROM}`.  The row's first block (`ceil(T/BN) - 1`) is always at or
above it.  `targets.attention_spec` binds `Attention_v4` under `TargetProfile.fa3_construction = "check-inf-per-iteration"` (the default);
`"guard-every-block"` binds `Attention_v2`.  New versions because the identity changed; v2 stays registered and untouched (registry version
discipline).

`AttentionHead_v20` and `Attention_v20` are v4 with every step a Call, as `dead_free`'s v19 is FA2's v18, so a head's key blocks can be
units of their own (Llama-3.2-1B's head on sm_90 is a 64-dim row over kBlockN 192, up to 6 blocks at the window class of
record's 1,087 keys).  v4's first iteration, `AttnBlock_v4{FIRST=True, CHECK=True}`, is FA2's word for word (both guard the running max
before `max * scale`, and a first block has no rescale), so it is `AttnBlockFirst_v1`, which pads only what it reads; v4's epilogue is
`AttnHeadEnd_v1`'s.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Affine, Coll as _Coll, Explicit, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.gemm import DotBf16 as DotBf16V2
from verity.ml.circuits.ops.prims import ZERO32, const
from verity.ml.definitions.registry import fa2_check_inf as F2, prims as P
from verity.ml.definitions.registry.b1 import ZERO16, _lane, scale_log2_bits
from verity.ml.definitions.registry.dead_free import AttnBlockFirstV1, AttnHeadEndV1
from verity.ml.definitions.registry.prims import A16, BF16, F32


@composite("AttnBlock", 4, ["D", "NVIS", "FIRST", "BN", "DOT", "CHECK"],
           lambda S: (((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
                        ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)))),
                      Tuple(F32, Array(4, F32), Array(S.D, F32))),
           conformance="not-yet-established")
def AttnBlockV4(B, S, q, k, v, m_prev, l_prev, o_prev, scale_log2=None):
    """`AttnBlock_v2` with FA3's per-iteration `Check_inf` as the static `CHECK`: with it the running max is guarded
    (`GuardNegInfZero`) before the rescale and `max * scale`, as in v2; without it (an unmasked iteration) both read the running max as
    it is.  `FIRST` needs `CHECK` (the first iteration runs `Check_inf`).  `scale_log2`: `AttnBlock_v13`'s input."""
    D, NVIS, FIRST, BN, DOT, CHECK = S.D, S.NVIS, S.FIRST, S.BN, S.DOT, S.CHECK
    if FIRST and not CHECK:
        raise ValueError("AttnBlock_v4: the first iteration runs Check_inf (FIRST needs CHECK)")
    dot = bind(DotBf16V2, K=D, DOT=DOT)
    s = [B.call(dot, q, k[c]) for c in range(NVIS)]  # A: S = Q K^T, chained k-steps, acc 0 per block
    bmax = s[0]
    for c in range(1, NVIS):
        bmax = B.call(P.F32Max, bmax, s[c])  # B1: row max
    m_new = bmax if FIRST else B.call(P.F32Max, m_prev, bmax)
    m_use = B.call(P.GuardNegInfZero, m_new) if CHECK else m_new
    scale_log2 = B.call(const(32, scale_log2_bits(D))) if scale_log2 is None else scale_log2
    if not FIRST:  # B2: rescale factor
        scores_scale = B.call(P.MufuEx2Ftz, B.call(P.F32MulFtz, B.call(P.F32SubFtz, m_prev, m_use), scale_log2))
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


@composite("AttentionHead", 4, ["T", "D", "BN", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
                      Array(S.D, BF16)),
           conformance="not-yet-established")
def AttentionHeadV4(B, S, q, k, v, *extra, block=AttnBlockV4):
    """`AttentionHead_v2` over `AttnBlock_v4`: causal attention of one query at position T-1 over keys 0..T-1, ceil(T/BN) blocks visited
    last -> first, block b with `CHECK = b >= MASKED_FROM` (the FA3 tile's causal-masked iterations; `attention_vocab.fa3_masked_from`); epilogue
    `sum = (l0 + l2) + (l1 + l3)`, `inv = INV(sum)`, `out = bf16(o * inv)`.  `extra` / `block`: `AttentionHead_v13`'s."""
    T, D, BN, DOT, INV, MF = S.T, S.D, S.BN, S.DOT, S.INV, S.MASKED_FROM
    nb = (T + BN - 1) // BN
    if not 0 <= MF <= nb - 1:
        raise ValueError(f"AttentionHead_v4: MASKED_FROM {MF} outside blocks 0..{nb - 1} (the first block is always a Check_inf iteration)")
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


@composite("Attention", 4, ["T", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="not-yet-established")
def AttentionV4(B, S, q, kc, vc, *extra, head=AttentionHeadV4):
    """`Attention_v2` over `AttentionHead_v4{T, D, BN, DOT, INV, MASKED_FROM}`: GQA, head h reads KV head h // (NH/KVH); every head of the
    row shares its tile, so one MASKED_FROM.  `extra` / `head`: `Attention_v13`'s."""
    T, NH, KVH, D = S.T, S.NH, S.KVH, S.D
    G = NH // KVH
    head = bind(head, T=T, D=D, BN=S.BN, DOT=S.DOT, INV=S.INV, MASKED_FROM=S.MASKED_FROM)
    outs = []
    for g in range(KVH):
        kg = _Coll(Array(T, Array(D, BF16)), strided_view(kc, g * D, D, KVH * D, 1, T * D))
        vg = _Coll(Array(T, Array(D, BF16)), strided_view(vc, g * D, D, KVH * D, 1, T * D))
        outs.append(B.batch(head, q[g * G:(g + 1) * G], kg, vg, *extra, axes=(0, None, None) + (None,) * len(extra)))
    return _Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


@composite("AttentionHead", 20, ["T", "D", "BN", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
                      Array(S.D, BF16)),
           conformance="not-yet-established: AttentionHead_v4's function")
def AttentionHeadV20(B, S, q, k, v):
    """`AttentionHead_v4{T, D, BN, DOT, INV, MASKED_FROM}` with every step a Call, so each key block can be a unit of its own: the
    first visited block `AttnBlockFirst_v1{D, NVIS, BN, DOT}` (keys (nb - 1) BN ... T - 1), block b below it
    `AttnBlock_v4{D, BN, FIRST=False, BN, DOT, CHECK = b >= MASKED_FROM}`, and the epilogue `AttnHeadEnd_v1{D, INV}`."""
    T, D, BN, DOT, MF = S.T, S.D, S.BN, S.DOT, S.MASKED_FROM
    nb = (T + BN - 1) // BN
    if not 0 <= MF <= nb - 1:
        raise ValueError(f"AttentionHead_v20: MASKED_FROM {MF} outside blocks 0..{nb - 1} "
                         "(the first block is always a Check_inf iteration)")
    lo = (nb - 1) * BN
    st = B.call(bind(AttnBlockFirstV1, D=D, NVIS=T - lo, BN=BN, DOT=DOT), q, k[lo:T], v[lo:T])
    for b in range(nb - 2, -1, -1):
        block = bind(AttnBlockV4, D=D, NVIS=BN, FIRST=False, BN=BN, DOT=DOT, CHECK=b >= MF)
        st = B.call(block, q, k[b * BN:(b + 1) * BN], v[b * BN:(b + 1) * BN], st[0], st[1], st[2])
    return B.call(bind(AttnHeadEndV1, D=D, INV=S.INV), st[1], st[2])


@composite("Attention", 20, ["T", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="not-yet-established: Attention_v4's function")
def AttentionV20(B, S, q, kc, vc):
    """`Attention_v4{T, NH, KVH, D, BN, DOT, INV, MASKED_FROM}` over `AttentionHead_v20`."""
    return F2._check_inf_attention(B, S, q, kc, vc, AttentionHeadV20, ())


FA3_CHECK_INF_COMPOSITES = [AttnBlockV4, AttentionHeadV4, AttentionV4, AttentionHeadV20, AttentionV20]

__all__ = ["AttnBlockV4", "AttentionHeadV4", "AttentionV4", "AttentionHeadV20", "AttentionV20", "FA3_CHECK_INF_COMPOSITES"]
