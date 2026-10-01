"""FA2 attention on bits: `AttnBlock_v6` / `AttentionHead_v6` / `Attention_v6`, `AttnBlock_v3` / `AttentionHead_v3` / `Attention_v3`'s
bodies (`b1`) line for line over Boolean Definitions, so their word views are the v3 chain at the same statics.

Every word is its bits (`verity.ir.boolean.word_type`): a bf16 word `Array<16, Value<1>>`, an f32 word `Array<32, Value<1>>`. The
callees are the Boolean versions of v3's: `DotBf16_v3{K=D, DOT=AmpereBF16TcDot16_v3}` (`DotBf16_v1{K}`), `AmpereBF16TcDot16_v3`,
`F2fpBf16_v2`, the FTZ f32 operations, `F32Max_v2`, `GuardNegInfZero_v2` and `Fa2InvSum_v2` (`verity.ml.boolean.attention`) and
`MufuEx2Ftz_v2` (`verity.ml.boolean.mufu`). v3's word constants (zero, `_scale_log2_bits(D)`) are written with the constant bits.
v4 and v5 are the FA3 / FA2 per-iteration `Check_inf` chains (`fa3_check_inf`, `fa2_check_inf`).
"""
from __future__ import annotations

from verity.ir import boolean as BL
from verity.ir.defs import bind, composite
from verity.ir.refs import Coll, Explicit, array_of, concat, strided_view, tuple_of
from verity.ir.types import Array, Tuple
from verity.ml.boolean import attention as BA
from verity.ml.boolean import gemm as BG
from verity.ml.boolean import mufu as BM
from verity_vllm.program.registry.b1 import AttentionHeadV3, AttentionV3, AttnBlockV3, _lane, _scale_log2_bits

BF16 = Array(16, BL.BIT)
F32 = Array(32, BL.BIT)
A16 = Array(16, BF16)


class _Bits:
    """The constant bits of one body, each called once, and words written with them."""

    def __init__(self, B):
        self.B, self.refs = B, {}

    def word(self, value: int, width: int) -> Coll:
        bits = []
        for i in range(width):
            b = (value >> i) & 1
            if b not in self.refs:
                self.refs[b] = self.B.call(BL.constant(b)).refs[0]
            bits.append(self.refs[b])
        return Coll(Array(width, BL.BIT), Explicit(tuple(bits)))


def _transpose(vblk: Coll, D: int) -> Coll:
    """`vblk` (`Array<16, Array<D, bf16>>`) as `Array<D, Array<16, bf16>>`: column d over the 16 keys."""
    return Coll(Array(D, A16), concat([strided_view(vblk, 16 * d, 16, 16 * D, 1, 256) for d in range(D)]))


@composite("AttnBlock", 6, ["D", "NVIS", "FIRST", "BN"],
           lambda S: (((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
                        ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)))),
                      Tuple(F32, Array(4, F32), Array(S.D, F32))),
           conformance="exact-model-tested")
def AttnBlockV6(B, S, q, k, v, m_prev, l_prev, o_prev):
    """`AttnBlock_v3`'s body on bits: one BN-key block of FA2's online softmax for one query row, NVIS visible columns."""
    D, NVIS, FIRST, BN = S.D, S.NVIS, S.FIRST, S.BN
    if BN % 16 or not 1 <= NVIS <= BN:
        raise ValueError(f"AttnBlock_v6: BN={BN} must be a multiple of 16 and 1 <= NVIS={NVIS} <= BN")
    c = _Bits(B)
    dot = bind(BG.DotBf16, K=D, DOT=BG.AmpereBF16TcDot16)
    s = [B.call(dot, q, k[j]) for j in range(NVIS)]
    bmax = s[0]
    for j in range(1, NVIS):
        bmax = B.call(BA.F32Max, bmax, s[j])
    m_new = bmax if FIRST else B.call(BA.F32Max, m_prev, bmax)
    m_use = B.call(BA.GuardNegInfZero, m_new)
    scale_log2 = c.word(_scale_log2_bits(D), 32)
    if not FIRST:
        scores_scale = B.call(BM.MufuEx2Ftz, B.call(BA.F32MulFtz, B.call(BA.F32SubFtz, m_prev, m_use), scale_log2))
    max_scaled = B.call(BA.F32MulFtz, m_use, scale_log2)
    p = [B.call(BM.MufuEx2Ftz, B.call(BA.F32FmaSubFtz, s[j], scale_log2, max_scaled)) for j in range(NVIS)]
    lanes: list[Coll] = []
    for ln in range(4):
        cols = [j for j in range(NVIS) if _lane(j) == ln]
        first = p[cols[0]] if cols else c.word(0, 32)
        acc = first if FIRST else B.call(BA.F32FmaFtz, scores_scale, l_prev[ln], first)
        for j in cols[1:]:
            acc = B.call(BA.F32AddFtz, acc, p[j])
        lanes.append(acc)
    pb = [B.call(BG.F2fpBf16, x) for x in p]
    zero16 = c.word(0, 16) if NVIS < BN else None
    o = [c.word(0, 32) for _ in range(D)] if FIRST else [B.call(BA.F32MulFtz, o_prev[d], scores_scale) for d in range(D)]
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break
        prow = Coll(A16, concat([(pb[k0 + j] if k0 + j < NVIS else zero16).refs for j in range(16)]))
        o = list(B.batch(BG.AmpereBF16TcDot16, array_of(o), prow, _transpose(v[k0:k0 + 16], D), axes=(0, None, 0)))
    return tuple_of(m_new, array_of(lanes), array_of(o))


@composite("AttentionHead", 6, ["T", "D", "BN"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
                      Array(S.D, BF16)),
           conformance="exact-model-tested")
def AttentionHeadV6(B, S, q, k, v):
    """`AttentionHead_v3`'s body on bits over `AttnBlock_v6`: one query at position T-1 over keys 0..T-1, ceil(T / BN) blocks
    visited last -> first, then `bf16(o * Fa2InvSum((l0 + l2) + (l1 + l3)))`."""
    T, D, BN = S.T, S.D, S.BN
    c = _Bits(B)
    nb = (T + BN - 1) // BN
    m = l = o = None
    zero_row = None
    for b in range(nb - 1, -1, -1):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb, vb = k[lo:hi], v[lo:hi]
        if nvis < BN:
            if zero_row is None:
                zero_row = Coll(Array(D, BF16), concat([c.word(0, 16).refs] * D))
            pad = [zero_row.refs] * (BN - nvis)
            kb = Coll(Array(BN, Array(D, BF16)), concat([kb.refs] + pad))
            vb = Coll(Array(BN, Array(D, BF16)), concat([vb.refs] + pad))
        first = b == nb - 1
        blk = bind(AttnBlockV6, D=D, NVIS=nvis, FIRST=first, BN=BN)
        if first:
            z = c.word(0, 32)
            st = B.call(blk, q, kb, vb, z, Coll(Array(4, F32), concat([z.refs] * 4)), Coll(Array(D, F32), concat([z.refs] * D)))
        else:
            st = B.call(blk, q, kb, vb, m, l, o)
        m, l, o = st[0], st[1], st[2]
    total = B.call(BA.F32AddFtz, B.call(BA.F32AddFtz, l[0], l[2]), B.call(BA.F32AddFtz, l[1], l[3]))
    inv = B.call(BA.Fa2InvSum, total)
    y = B.batch(BA.F32MulFtz, o, inv, axes=(0, None))
    return B.batch(BG.F2fpBf16, y, axes=(0,))


@composite("Attention", 6, ["T", "NH", "KVH", "D", "BN"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="exact-model-tested")
def AttentionV6(B, S, q, kc, vc):
    """`Attention_v3`'s body on bits over `AttentionHead_v6` (GQA: head h reads KV head h // (NH/KVH); caches `[T][KVH][D]`)."""
    T, NH, KVH, D, BN = S.T, S.NH, S.KVH, S.D, S.BN
    G = NH // KVH
    W = 16 * D
    outs = []
    for g in range(KVH):
        kg = Coll(Array(T, Array(D, BF16)), strided_view(kc, g * W, W, KVH * W, 1, T * W))
        vg = Coll(Array(T, Array(D, BF16)), strided_view(vc, g * W, W, KVH * W, 1, T * W))
        outs.append(B.batch(bind(AttentionHeadV6, T=T, D=D, BN=BN), q[g * G:(g + 1) * G], kg, vg, axes=(0, None, None)))
    return Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


AttnBlockV6.word = lambda b: bind(AttnBlockV3, **b)  # type: ignore[attr-defined]
AttentionHeadV6.word = lambda b: bind(AttentionHeadV3, **b)  # type: ignore[attr-defined]
AttentionV6.word = lambda b: bind(AttentionV3, **b)  # type: ignore[attr-defined]
