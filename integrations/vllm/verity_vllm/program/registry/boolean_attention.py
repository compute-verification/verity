"""Flash attention on bits: `AttnBlock` / `AttentionHead` / `Attention` v6 (v3's statics), v7 (v2's), v8 (v5's) and v9 (v4's),
the v3 and v2 bodies (`b1`) and the v5 and v4 bodies (`fa2_check_inf`, `fa3_check_inf`) line for line over Boolean
Definitions, so their word views are those chains at the same statics.

Every word is its bits (`verity.ml.circuits.boolean.word_type`): a bf16 word `Array<16, Value<1>>`, an f32 word `Array<32, Value<1>>`. The
callees are the Boolean versions of the word chains': `DotBf16_v3{K=D, DOT}` and `DOT` itself (`verity.ml.circuits.ops.boolean.gemm`;
v6's is `AmpereBF16TcDot16_v3`), `F2fpBf16_v2`, the FTZ f32 operations, `F32Max_v2`, `GuardNegInfZero_v2` and the inverse sum
(v6's `Fa2InvSum_v2`; v7's, v8's and v9's `INV`: `Fa2InvSum_v2` or `Fa3InvSum_v2`, FA3's, here) and `MufuEx2Ftz_v2`
(`verity.ml.circuits.ops.boolean.mufu`). The word constants (zero, `_scale_log2_bits(D)`) are written with the constant bits. v4 and v5 are
the FA3 / FA2 per-iteration `Check_inf` chains (`fa3_check_inf`, `fa2_check_inf`); v8 is v5's, whose unguarded blocks
(`CHECK` false, below `MASKED_FROM`) rescale by the running max as it is; v9 is v4's, whose unguarded blocks read it as it
is for `max * scale` too.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from collections.abc import Callable

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, Explicit, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.boolean import attention as BA
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import mufu as BM
from verity.ml.circuits.boolean.trace import definition, word_of
from verity_vllm.program.registry import hopper
from verity_vllm.program.registry.b1 import (AttentionHeadV2, AttentionHeadV3, AttentionV2, AttentionV3, AttnBlockV2, AttnBlockV3,
                                             _lane, _scale_log2_bits)
from verity_vllm.program.registry.fa2_check_inf import AttentionHeadV5, AttentionV5, AttnBlockV5
from verity_vllm.program.registry.fa3_check_inf import AttentionHeadV4, AttentionV4, AttnBlockV4

BF16 = Array(16, BL.BIT)
F32 = Array(32, BL.BIT)
A16 = Array(16, BF16)

Fa3InvSum = definition("Fa3InvSum", 2, BA.fa3_inv_sum, (("sum", F32),), F32, word=hopper.Fa3InvSum,
                       doc="FA3 Softmax::finalize on bits: Fa3InvSum_v1, (sum == 0 || sum != sum) ? 0 : MUFU.RCP(sum)")


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


def _block_type(S):
    return (((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
              ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)))),
            Tuple(F32, Array(4, F32), Array(S.D, F32)))


def _head_type(S):
    return ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))), Array(S.D, BF16))


def _attention_type(S):
    return ((("q", Array(S.NH, Array(S.D, BF16))), ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
             ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))), Array(S.NH, Array(S.D, BF16)))


def _block(B, name: str, D: int, NVIS: int, FIRST: bool, BN: int, DOT, q, k, v, m_prev, l_prev, o_prev, check: bool = True,
           guard_max: bool = True):
    """One BN-key block of FA2's online softmax for one query row, NVIS visible columns (`AttnBlock_v2` / `_v3`'s body); without
    ``check`` (`AttnBlock_v5`'s unmasked iteration) the rescale reads the running max as it is, not guarded, and without
    ``guard_max`` (`AttnBlock_v4`'s) so does `max * scale`."""
    if BN % 16 or not 1 <= NVIS <= BN:
        raise ValueError(f"{name}: BN={BN} must be a multiple of 16 and 1 <= NVIS={NVIS} <= BN")
    if FIRST and not check:
        raise ValueError(f"{name}: the first iteration runs Check_inf (FIRST needs CHECK)")
    c = _Bits(B)
    dot = bind(BG.DotBf16, K=D, DOT=DOT)
    s = [B.call(dot, q, k[j]) for j in range(NVIS)]
    bmax = s[0]
    for j in range(1, NVIS):
        bmax = B.call(BA.F32Max, bmax, s[j])
    m_new = bmax if FIRST else B.call(BA.F32Max, m_prev, bmax)
    m_use = B.call(BA.GuardNegInfZero, m_new) if check or guard_max else m_new
    scale_log2 = c.word(_scale_log2_bits(D), 32)
    if not FIRST:
        scores_scale = B.call(BM.MufuEx2Ftz, B.call(BA.F32MulFtz, B.call(BA.F32SubFtz, m_prev, m_use if check else m_new), scale_log2))
    max_scaled = B.call(BA.F32MulFtz, m_use if guard_max else m_new, scale_log2)
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
        o = list(B.batch(DOT, array_of(o), prow, _transpose(v[k0:k0 + 16], D), axes=(0, None, 0)))
    return tuple_of(m_new, array_of(lanes), array_of(o))


def _head(B, T: int, D: int, BN: int, block_of, INV, q, k, v):
    """Causal attention of one query at position T-1 over keys 0..T-1: ceil(T / BN) blocks (``block_of(b, nvis, first)``) visited
    last -> first, then `bf16(o * INV((l0 + l2) + (l1 + l3)))` (`AttentionHead_v2` / `_v3` / `_v5`'s body)."""
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
        blk = block_of(b, nvis, first)
        if first:
            z = c.word(0, 32)
            st = B.call(blk, q, kb, vb, z, Coll(Array(4, F32), concat([z.refs] * 4)), Coll(Array(D, F32), concat([z.refs] * D)))
        else:
            st = B.call(blk, q, kb, vb, m, l, o)
        m, l, o = st[0], st[1], st[2]
    total = B.call(BA.F32AddFtz, B.call(BA.F32AddFtz, l[0], l[2]), B.call(BA.F32AddFtz, l[1], l[3]))
    inv = B.call(INV, total)
    y = B.batch(BA.F32MulFtz, o, inv, axes=(0, None))
    return B.batch(BG.F2fpBf16, y, axes=(0,))


def _attention(B, T: int, NH: int, KVH: int, D: int, head, q, kc, vc):
    """GQA over ``head``: head h reads KV head h // (NH/KVH); caches `[T][KVH][D]` (`Attention_v2` / `_v3`'s body)."""
    G = NH // KVH
    W = 16 * D
    outs = []
    for g in range(KVH):
        kg = Coll(Array(T, Array(D, BF16)), strided_view(kc, g * W, W, KVH * W, 1, T * W))
        vg = Coll(Array(T, Array(D, BF16)), strided_view(vc, g * W, W, KVH * W, 1, T * W))
        outs.append(B.batch(head, q[g * G:(g + 1) * G], kg, vg, axes=(0, None, None)))
    return Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


def _on_bits(name: str, version: int, statics: list[str], signature: Callable, body: Callable,
             word: CompositeDefinition) -> CompositeDefinition:
    """The Boolean composite ``name_v{version}`` with body ``body``, documented by its docstring, whose word view
    (`trace.word_of`) is ``word`` at its bindings, the Boolean ``DOT`` and ``INV`` read as their word views."""
    d = CompositeDefinition(name, version, tuple(statics), signature, body, doc=body.__doc__ or "",
                            conformance="exact-model-tested")
    d.word = lambda b: bind(word, **{k: (word_of(x) if k in ("DOT", "INV") else x) for k, x in b.items()})  # type: ignore[attr-defined]
    return d


def _fa2_block(B, S, q, k, v, m_prev, l_prev, o_prev):
    """`AttnBlock_v3`'s body on bits: one BN-key block of FA2's online softmax for one query row, NVIS visible columns."""
    return _block(B, "AttnBlock_v6", S.D, S.NVIS, S.FIRST, S.BN, BG.AmpereBF16TcDot16, q, k, v, m_prev, l_prev, o_prev)


AttnBlockV6 = _on_bits("AttnBlock", 6, ["D", "NVIS", "FIRST", "BN"], _block_type, _fa2_block, AttnBlockV3)


def _fa2_head(B, S, q, k, v):
    """`AttentionHead_v3`'s body on bits over `AttnBlock_v6`, epilogue `Fa2InvSum_v2`."""
    return _head(B, S.T, S.D, S.BN, lambda b, nvis, first: bind(AttnBlockV6, D=S.D, NVIS=nvis, FIRST=first, BN=S.BN), BA.Fa2InvSum,
                 q, k, v)


AttentionHeadV6 = _on_bits("AttentionHead", 6, ["T", "D", "BN"], _head_type, _fa2_head, AttentionHeadV3)


def _fa2_attention(B, S, q, kc, vc):
    """`Attention_v3`'s body on bits over `AttentionHead_v6`."""
    return _attention(B, S.T, S.NH, S.KVH, S.D, bind(AttentionHeadV6, T=S.T, D=S.D, BN=S.BN), q, kc, vc)


AttentionV6 = _on_bits("Attention", 6, ["T", "NH", "KVH", "D", "BN"], _attention_type, _fa2_attention, AttentionV3)


def _dot_block(B, S, q, k, v, m_prev, l_prev, o_prev):
    """`AttnBlock_v2`'s body on bits: `AttnBlock_v6` with the Boolean k16 step `DOT` a static (FA2 on mma.sync, FA3 on wgmma)."""
    return _block(B, "AttnBlock_v7", S.D, S.NVIS, S.FIRST, S.BN, S.DOT, q, k, v, m_prev, l_prev, o_prev)


AttnBlockV7 = _on_bits("AttnBlock", 7, ["D", "NVIS", "FIRST", "BN", "DOT"], _block_type, _dot_block, AttnBlockV2)


def _dot_head(B, S, q, k, v):
    """`AttentionHead_v2`'s body on bits over `AttnBlock_v7{..., DOT}`, epilogue `INV` (`Fa2InvSum_v2` or `Fa3InvSum_v2`)."""
    return _head(B, S.T, S.D, S.BN, lambda b, nvis, first: bind(AttnBlockV7, D=S.D, NVIS=nvis, FIRST=first, BN=S.BN, DOT=S.DOT),
                 S.INV, q, k, v)


AttentionHeadV7 = _on_bits("AttentionHead", 7, ["T", "D", "BN", "DOT", "INV"], _head_type, _dot_head, AttentionHeadV2)


def _dot_attention(B, S, q, kc, vc):
    """`Attention_v2`'s body on bits over `AttentionHead_v7`."""
    return _attention(B, S.T, S.NH, S.KVH, S.D, bind(AttentionHeadV7, T=S.T, D=S.D, BN=S.BN, DOT=S.DOT, INV=S.INV), q, kc, vc)


AttentionV7 = _on_bits("Attention", 7, ["T", "NH", "KVH", "D", "BN", "DOT", "INV"], _attention_type, _dot_attention, AttentionV2)


def _checked_block(B, S, q, k, v, m_prev, l_prev, o_prev):
    """`AttnBlock_v5`'s body on bits: `AttnBlock_v7` with FA2's per-iteration `Check_inf` as the static `CHECK`."""
    return _block(B, "AttnBlock_v8", S.D, S.NVIS, S.FIRST, S.BN, S.DOT, q, k, v, m_prev, l_prev, o_prev, check=S.CHECK)


AttnBlockV8 = _on_bits("AttnBlock", 8, ["D", "NVIS", "FIRST", "BN", "DOT", "CHECK"], _block_type, _checked_block, AttnBlockV5)


def _checked_head(B, S, q, k, v):
    """`AttentionHead_v5`'s body on bits over `AttnBlock_v8`: block b runs with `CHECK = b >= MASKED_FROM`."""
    nb = (S.T + S.BN - 1) // S.BN
    if not 0 <= S.MASKED_FROM <= nb - 1:
        raise ValueError(f"AttentionHead_v8: MASKED_FROM {S.MASKED_FROM} outside blocks 0..{nb - 1} "
                         "(the first block is always a Check_inf iteration)")
    return _head(B, S.T, S.D, S.BN, lambda b, nvis, first: bind(AttnBlockV8, D=S.D, NVIS=nvis, FIRST=first, BN=S.BN, DOT=S.DOT,
                                                                CHECK=b >= S.MASKED_FROM), S.INV, q, k, v)


AttentionHeadV8 = _on_bits("AttentionHead", 8, ["T", "D", "BN", "DOT", "INV", "MASKED_FROM"], _head_type, _checked_head,
                           AttentionHeadV5)


def _checked_attention(B, S, q, kc, vc):
    """`Attention_v5`'s body on bits over `AttentionHead_v8`; every head of the row shares its tile, so one MASKED_FROM."""
    return _attention(B, S.T, S.NH, S.KVH, S.D, bind(AttentionHeadV8, T=S.T, D=S.D, BN=S.BN, DOT=S.DOT, INV=S.INV,
                                                     MASKED_FROM=S.MASKED_FROM), q, kc, vc)


AttentionV8 = _on_bits("Attention", 8, ["T", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"], _attention_type,
                       _checked_attention, AttentionV5)


def _fa3_checked_block(B, S, q, k, v, m_prev, l_prev, o_prev):
    """`AttnBlock_v4`'s body on bits: FA3's per-iteration `Check_inf` as the static `CHECK`; without it neither the rescale nor
    `max * scale` reads the guarded running max."""
    return _block(B, "AttnBlock_v9", S.D, S.NVIS, S.FIRST, S.BN, S.DOT, q, k, v, m_prev, l_prev, o_prev, check=S.CHECK,
                  guard_max=S.CHECK)


AttnBlockV9 = _on_bits("AttnBlock", 9, ["D", "NVIS", "FIRST", "BN", "DOT", "CHECK"], _block_type, _fa3_checked_block, AttnBlockV4)


def _fa3_checked_head(B, S, q, k, v):
    """`AttentionHead_v4`'s body on bits over `AttnBlock_v9`: block b runs with `CHECK = b >= MASKED_FROM`."""
    nb = (S.T + S.BN - 1) // S.BN
    if not 0 <= S.MASKED_FROM <= nb - 1:
        raise ValueError(f"AttentionHead_v9: MASKED_FROM {S.MASKED_FROM} outside blocks 0..{nb - 1} "
                         "(the first block is always a Check_inf iteration)")
    return _head(B, S.T, S.D, S.BN, lambda b, nvis, first: bind(AttnBlockV9, D=S.D, NVIS=nvis, FIRST=first, BN=S.BN, DOT=S.DOT,
                                                                CHECK=b >= S.MASKED_FROM), S.INV, q, k, v)


AttentionHeadV9 = _on_bits("AttentionHead", 9, ["T", "D", "BN", "DOT", "INV", "MASKED_FROM"], _head_type, _fa3_checked_head,
                           AttentionHeadV4)


def _fa3_checked_attention(B, S, q, kc, vc):
    """`Attention_v4`'s body on bits over `AttentionHead_v9`; every head of the row shares its tile, so one MASKED_FROM."""
    return _attention(B, S.T, S.NH, S.KVH, S.D, bind(AttentionHeadV9, T=S.T, D=S.D, BN=S.BN, DOT=S.DOT, INV=S.INV,
                                                     MASKED_FROM=S.MASKED_FROM), q, kc, vc)


AttentionV9 = _on_bits("Attention", 9, ["T", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"], _attention_type,
                       _fa3_checked_attention, AttentionV4)


#: the FA2 (Ampere mma.sync) and FA3 (Hopper wgmma) bindings of v7's DOT and INV
ATTENTION_FA2: dict = {"DOT": BG.AmpereBF16TcDot16, "INV": BA.Fa2InvSum}
ATTENTION_FA3: dict = {"DOT": BG.HopperBF16WgmmaDot16, "INV": Fa3InvSum}
