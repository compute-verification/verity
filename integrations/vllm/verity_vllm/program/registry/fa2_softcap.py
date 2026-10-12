"""FA2 with logit soft-capping on a target: `AttnBlockSoftcap_v2` / `AttentionHeadSoftcap_v2` / `AttentionSoftcap_v2`, the chain
`Attention_v5` (`fa2_check_inf`) with the one stage `Is_softcap` adds.

vllm-flash-attn 506341a `flash_fwd_kernel.h::compute_attn_1rowblock_splitkv<..., Is_softcap=true>` (vLLM + Gemma-2
`attn_logit_softcapping = 50`), per key block:

    flash::gemm(acc_s, Q, K)                                         A   S = Q K^T (the target's k16 step, `DOT`)
    if (Is_softcap) flash::apply_softcap(acc_s, params.softcap)      A'  s = tanh.approx.f32(s * params.softcap)  (cutlass::fast_tanh)
    mask.apply_mask<Is_causal, Is_even_MN>(acc_s, ...)                   masked columns -> -inf (never read here; P = 0 exactly)
    softmax_rescale_o<Is_first, Check_inf>(acc_s, acc_o, params.scale_softmax_log2)   B1..B5, C as `AttnBlock_v5{..., CHECK}`

with `flash_api.cpp::set_params_fprop`: `params.softcap = softmax_scale / softcap` (`dense.softcap_mul_bits`) and
`params.scale_softmax_log2 = softcap * M_LOG2E` (`dense.softcap_scale_log2_bits`).  The softcap runs before the row max, so the
`Check_inf` placement (`MASKED_FROM`, `attention_vocab.fa2_masked_from`) and every stage after A' are `AttnBlock_v5`'s word for word;
`AttnBlockSoftcap_v1` (dense) is this body with every block guarded and the Ampere step.  `MufuTanh_v1` is the sm_89 table: on another
target its words are the target's only where the capture's tanh probe measured them equal.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Affine, Coll as _Coll, Explicit, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.gemm import DotBf16 as DotBf16V2
from verity.ml.circuits.ops.prims import ZERO32, const
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.b1 import ZERO16, _lane
from verity_vllm.program.registry.dense import softcap_mul_bits, softcap_scale_log2_bits
from verity_vllm.program.registry.prims import A16, BF16, F32


@composite("AttnBlockSoftcap", 2, ["D", "NVIS", "FIRST", "BN", "CAP", "DOT", "CHECK"],
           lambda S: (((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
                        ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)))),
                      Tuple(F32, Array(4, F32), Array(S.D, F32))),
           conformance="not-yet-established")
def AttnBlockSoftcapV2(B, S, q, k, v, m_prev, l_prev, o_prev, softcap_mul=None, scale_log2=None):
    """`AttnBlock_v5{D, NVIS, FIRST, BN, DOT, CHECK}` with FA2's stage A' in front of the online softmax: `s'[c] = MufuTanh(F32MulFtz(s[c],
    softmax_scale / CAP))`, exp2 constant `CAP * log2e`.  `softcap_mul` / `scale_log2`: `AttnBlockSoftcap_v5`'s inputs (no `CAP`)."""
    D, NVIS, FIRST, BN, CAP, DOT, CHECK = S.D, S.NVIS, S.FIRST, S.BN, getattr(S, "CAP", None), S.DOT, S.CHECK
    if FIRST and not CHECK:
        raise ValueError("AttnBlockSoftcap_v2: the first iteration runs Check_inf (FIRST needs CHECK)")
    if softcap_mul is None and not (isinstance(CAP, (int, float)) and CAP > 0):
        raise ValueError(f"AttnBlockSoftcap_v2: CAP={CAP!r} must be a positive softcap (FA2 takes the Is_softcap path iff softcap > 0)")
    dot = bind(DotBf16V2, K=D, DOT=DOT)
    softcap_mul = B.call(const(32, softcap_mul_bits(D, CAP))) if softcap_mul is None else softcap_mul
    s = [B.call(P.MufuTanh, B.call(P.F32MulFtz, B.call(dot, q, k[c]), softcap_mul)) for c in range(NVIS)]  # A, A'
    bmax = s[0]
    for c in range(1, NVIS):
        bmax = B.call(P.F32Max, bmax, s[c])  # B1: row max (of the capped scores)
    m_new = bmax if FIRST else B.call(P.F32Max, m_prev, bmax)
    m_use = B.call(P.GuardNegInfZero, m_new)
    scale_log2 = B.call(const(32, softcap_scale_log2_bits(CAP))) if scale_log2 is None else scale_log2
    if not FIRST:  # B2: rescale factor
        scores_scale = B.call(P.MufuEx2Ftz, B.call(P.F32MulFtz, B.call(P.F32SubFtz, m_prev, m_use if CHECK else m_new), scale_log2))
    max_scaled = B.call(P.F32MulFtz, m_use, scale_log2)
    p = [B.call(P.MufuEx2Ftz, B.call(P.F32FmaSubFtz, s[c], scale_log2, max_scaled)) for c in range(NVIS)]  # B3
    lanes = []
    for ln in range(4):  # B4: per-lane sequential sums; the block's first add is contracted with the rescale
        cols = [c for c in range(NVIS) if _lane(c) == ln]
        if FIRST:
            acc = p[cols[0]] if cols else B.call(ZERO32)
        else:
            acc = B.call(P.F32FmaFtz, scores_scale, l_prev[ln], p[cols[0]] if cols else B.call(ZERO32))
        for c in cols[1:]:
            acc = B.call(P.F32AddFtz, acc, p[c])
        lanes.append(acc)
    pb = [B.call(P.F2fpBf16, x) for x in p]  # B5 + C: P -> bf16, O = (O * scale) + P V in BN/16 ascending k-steps of 16 keys
    zero16 = B.call(ZERO16) if NVIS < BN else None
    o = [B.call(ZERO32) for _ in range(D)] if FIRST else [B.call(P.F32MulFtz, o_prev[d], scores_scale) for d in range(D)]
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break
        prow = _Coll(A16, concat([(pb[k0 + j] if k0 + j < NVIS else zero16).refs for j in range(16)]))
        vblk = v[k0:k0 + 16]
        vt_refs = strided_view(vblk, 0, 16, 1, D, 16 * D) if isinstance(vblk.refs, Affine) else \
            Explicit(tuple(vblk.refs[j * D + d] for d in range(D) for j in range(16)))
        vt = _Coll(Array(D, A16), vt_refs)
        o = list(B.batch(DOT, array_of(o), prow, vt, axes=(0, None, 0)))
    return tuple_of(m_new, array_of(lanes), array_of(o))


@composite("AttentionHeadSoftcap", 2, ["T", "D", "BN", "CAP", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
                      Array(S.D, BF16)),
           conformance="not-yet-established")
def AttentionHeadSoftcapV2(B, S, q, k, v, *extra, block=AttnBlockSoftcapV2):
    """`AttentionHead_v5`'s body over `AttnBlockSoftcap_v2`: ceil(T/BN) blocks visited last -> first, block b with `CHECK = b >= MASKED_FROM`;
    epilogue `sum = (l0 + l2) + (l1 + l3)`, `inv = INV(sum)`, `out = bf16(o * inv)`.  `extra` / `block`: `AttentionHeadSoftcap_v5`'s."""
    T, D, BN, DOT, INV, MF = S.T, S.D, S.BN, S.DOT, S.INV, S.MASKED_FROM
    cap = {} if extra else {"CAP": S.CAP}
    nb = (T + BN - 1) // BN
    if not 0 <= MF <= nb - 1:
        raise ValueError(f"AttentionHeadSoftcap_v2: MASKED_FROM {MF} outside blocks 0..{nb - 1} (the first block is always a Check_inf iteration)")
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
        blk = bind(block, D=D, NVIS=nvis, FIRST=first, BN=BN, DOT=DOT, CHECK=b >= MF, **cap)
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


@composite("AttentionSoftcap", 2, ["T", "NH", "KVH", "D", "BN", "CAP", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="not-yet-established")
def AttentionSoftcapV2(B, S, q, kc, vc, *extra, head=AttentionHeadSoftcapV2):
    """`Attention_v5` over `AttentionHeadSoftcap_v2{T, D, BN, CAP, DOT, INV, MASKED_FROM}`: GQA, head h reads KV head h // (NH/KVH); every
    head of the row shares its tile, so one MASKED_FROM.  `extra` / `head`: `AttentionSoftcap_v5`'s."""
    T, NH, KVH, D = S.T, S.NH, S.KVH, S.D
    G = NH // KVH
    head = bind(head, T=T, D=D, BN=S.BN, DOT=S.DOT, INV=S.INV, MASKED_FROM=S.MASKED_FROM, **({} if extra else {"CAP": S.CAP}))
    outs = []
    for g in range(KVH):
        kg = _Coll(Array(T, Array(D, BF16)), strided_view(kc, g * D, D, KVH * D, 1, T * D))
        vg = _Coll(Array(T, Array(D, BF16)), strided_view(vc, g * D, D, KVH * D, 1, T * D))
        outs.append(B.batch(head, q[g * G:(g + 1) * G], kg, vg, *extra, axes=(0, None, None) + (None,) * len(extra)))
    return _Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


FA2_SOFTCAP_COMPOSITES = [AttnBlockSoftcapV2, AttentionHeadSoftcapV2, AttentionSoftcapV2]

__all__ = ["AttnBlockSoftcapV2", "AttentionHeadSoftcapV2", "AttentionSoftcapV2", "FA2_SOFTCAP_COMPOSITES"]
