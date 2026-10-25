"""FA2 with logit soft-capping on bits: `AttnBlockSoftcap_v3` / `AttentionHeadSoftcap_v3` / `AttentionSoftcap_v3`, `fa2_softcap`'s
v2 bodies line for line over Boolean Definitions, so their word views are the v2 chain at the same statics.

Every word is its bits (`verity.ml.circuits.boolean.word_type`): a bf16 word `Array<16, Value<1>>`, an f32 word `Array<32, Value<1>>`.  The
v2 chain's callees on bits: ``DOT`` and ``INV`` are bound to the Boolean versions of the word chain's (`HopperBF16WgmmaDot16_v2`,
`Fa2InvSum_v2`), the dot is `DotBf16_v3{K=D, DOT}`, the cast `F2fpBf16_v2`, the f32 operations `verity.ml.circuits.ops.boolean.attention`'s.
The two MUFU operations the word chain calls, stage A' `MufuTanh_v1` and B2/B3's `MufuEx2Ftz_v1`, are the statics ``TANH`` and
``EX2``; ``word_statics`` names their word Definitions, so the switch binds each to its Boolean version.  The word constants (zero,
`softcap_mul_bits(D, CAP)`, `softcap_scale_log2_bits(CAP)`) are written with the constant bits.

`AttnBlockSoftcap_v6` / `AttentionHeadSoftcap_v6` / `AttentionSoftcap_v6` are v3 with those two words inputs and no ``CAP``
(a `registered_values` program registers them); their word views are `value_inputs`' v5 chain.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.boolean import attention as BA
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.definitions.registry import fa2_softcap as FS
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.b1 import _lane
from verity.ml.definitions.registry.boolean_attention import _Bits, _transpose
from verity.ml.definitions.registry.dense import softcap_mul_bits, softcap_scale_log2_bits
from verity.ml.definitions.registry import value_inputs as VI

__all__ = ["AttentionHeadSoftcap", "AttentionHeadSoftcapIn", "AttentionSoftcap", "AttentionSoftcapIn", "AttnBlockSoftcap",
           "AttnBlockSoftcapIn", "WORD_STATICS"]

BF16 = Array(16, BL.BIT)
F32 = Array(32, BL.BIT)
A16 = Array(16, BF16)

#: the statics the word chain has no counterpart for, as the word Definitions they compute
WORD_STATICS = {"TANH": P.MufuTanh, "EX2": P.MufuEx2Ftz}


def _word(word, b: dict):
    """``word`` (a v2 composite) at the bindings ``b`` without ``TANH`` / ``EX2``, ``DOT`` and ``INV`` as their word Definitions."""
    return bind(word, **{k: word_of(v) if k in ("DOT", "INV") else v for k, v in b.items() if k not in WORD_STATICS})


def _block(B, S, q, k, v, m_prev, l_prev, o_prev, softcap_mul: Coll | None = None, scale_log2: Coll | None = None):
    """`AttnBlockSoftcap_v2`'s body on bits: one BN-key block of FA2's online softmax for one query row, NVIS visible columns, with
    stage A' `s'[c] = TANH(F32MulFtz(s[c], softmax_scale / CAP))` in front and exp2 constant `CAP * log2e`.  With the two words
    inputs (`AttnBlockSoftcap_v5`'s ``softcap_mul`` and ``scale_log2``) there is no ``CAP``."""
    D, NVIS, FIRST, BN, DOT, CHECK, TANH, EX2 = S.D, S.NVIS, S.FIRST, S.BN, S.DOT, S.CHECK, S.TANH, S.EX2
    name = "AttnBlockSoftcap_v3" if softcap_mul is None else "AttnBlockSoftcap_v6"
    if BN % 16 or not 1 <= NVIS <= BN:
        raise ValueError(f"{name}: BN={BN} must be a multiple of 16 and 1 <= NVIS={NVIS} <= BN")
    if FIRST and not CHECK:
        raise ValueError(f"{name}: the first iteration runs Check_inf (FIRST needs CHECK)")
    c = _Bits(B)
    dot = bind(BG.DotBf16, K=D, DOT=DOT)
    if softcap_mul is None:
        CAP = S.CAP
        if not (isinstance(CAP, (int, float)) and CAP > 0):
            raise ValueError(f"{name}: CAP={CAP!r} must be a positive softcap")
        softcap_mul = c.word(softcap_mul_bits(D, CAP), 32)
    s = [B.call(TANH, B.call(BA.F32MulFtz, B.call(dot, q, k[j]), softcap_mul)) for j in range(NVIS)]
    bmax = s[0]
    for j in range(1, NVIS):
        bmax = B.call(BA.F32Max, bmax, s[j])
    m_new = bmax if FIRST else B.call(BA.F32Max, m_prev, bmax)
    m_use = B.call(BA.GuardNegInfZero, m_new)
    if scale_log2 is None:
        scale_log2 = c.word(softcap_scale_log2_bits(S.CAP), 32)
    if not FIRST:
        scores_scale = B.call(EX2, B.call(BA.F32MulFtz, B.call(BA.F32SubFtz, m_prev, m_use if CHECK else m_new), scale_log2))
    max_scaled = B.call(BA.F32MulFtz, m_use, scale_log2)
    p = [B.call(EX2, B.call(BA.F32FmaSubFtz, s[j], scale_log2, max_scaled)) for j in range(NVIS)]
    lanes: list[Coll] = []
    for ln in range(4):
        cols = [j for j in range(NVIS) if _lane(j) == ln]
        first = p[cols[0]] if cols else c.zeros(F32)
        acc = first if FIRST else B.call(BA.F32FmaFtz, scores_scale, l_prev[ln], first)
        for j in cols[1:]:
            acc = B.call(BA.F32AddFtz, acc, p[j])
        lanes.append(acc)
    pb = [B.call(BG.F2fpBf16, x) for x in p]
    zero16 = c.zeros(BF16) if NVIS < BN else None
    o = [c.zeros(F32) for _ in range(D)] if FIRST else [B.call(BA.F32MulFtz, o_prev[d], scores_scale) for d in range(D)]
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break
        prow = Coll(A16, concat([(pb[k0 + j] if k0 + j < NVIS else zero16).refs for j in range(16)]))
        o = list(B.batch(DOT, array_of(o), prow, _transpose(v[k0:k0 + 16], D), axes=(0, None, 0)))
    return tuple_of(m_new, array_of(lanes), array_of(o))


AttnBlockSoftcap = CompositeDefinition(
    "AttnBlockSoftcap", 3, ("D", "NVIS", "FIRST", "BN", "CAP", "DOT", "CHECK", "TANH", "EX2"),
    lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
                ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32))),
               Tuple(F32, Array(4, F32), Array(S.D, F32))),
    _block, doc=_block.__doc__, conformance="exact-model-tested")
AttnBlockSoftcap.word = lambda b: _word(FS.AttnBlockSoftcapV2, b)  # type: ignore[attr-defined]
AttnBlockSoftcap.word_statics = WORD_STATICS  # type: ignore[attr-defined]


def _head(B, S, q, k, v, *extra):
    """`AttentionHeadSoftcap_v2`'s body on bits over `AttnBlockSoftcap_v3`: ceil(T/BN) blocks visited last -> first, block b with
    `CHECK = b >= MASKED_FROM`, then `bf16(o * INV((l0 + l2) + (l1 + l3)))`.  With ``extra`` (the two words, inputs) over
    `AttnBlockSoftcap_v6`, every block reading them."""
    T, D, BN, MF = S.T, S.D, S.BN, S.MASKED_FROM
    nb = (T + BN - 1) // BN
    if not 0 <= MF <= nb - 1:
        raise ValueError(f"AttentionHeadSoftcap_v{6 if extra else 3}: MASKED_FROM {MF} outside blocks 0..{nb - 1}")
    c = _Bits(B)
    m = l = o = None
    for b in range(nb - 1, -1, -1):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb, vb = k[lo:hi], v[lo:hi]
        if nvis < BN:
            pad = c.zeros(Array(BN - nvis, Array(D, BF16))).refs
            kb = Coll(Array(BN, Array(D, BF16)), concat([kb.refs, pad]))
            vb = Coll(Array(BN, Array(D, BF16)), concat([vb.refs, pad]))
        first = b == nb - 1
        if extra:
            blk = bind(AttnBlockSoftcapIn, D=D, NVIS=nvis, FIRST=first, BN=BN, DOT=S.DOT, CHECK=b >= MF, TANH=S.TANH, EX2=S.EX2)
        else:
            blk = bind(AttnBlockSoftcap, D=D, NVIS=nvis, FIRST=first, BN=BN, CAP=S.CAP, DOT=S.DOT, CHECK=b >= MF, TANH=S.TANH,
                       EX2=S.EX2)
        if first:
            st = B.call(blk, q, kb, vb, c.zeros(F32), c.zeros(Array(4, F32)), c.zeros(Array(D, F32)), *extra)
        else:
            st = B.call(blk, q, kb, vb, m, l, o, *extra)
        m, l, o = st[0], st[1], st[2]
    total = B.call(BA.F32AddFtz, B.call(BA.F32AddFtz, l[0], l[2]), B.call(BA.F32AddFtz, l[1], l[3]))
    inv = B.call(S.INV, total)
    y = B.batch(BA.F32MulFtz, o, inv, axes=(0, None))
    return B.batch(BG.F2fpBf16, y, axes=(0,))


AttentionHeadSoftcap = CompositeDefinition(
    "AttentionHeadSoftcap", 3, ("T", "D", "BN", "CAP", "DOT", "INV", "MASKED_FROM", "TANH", "EX2"),
    lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
               Array(S.D, BF16)),
    _head, doc=_head.__doc__, conformance="exact-model-tested")
AttentionHeadSoftcap.word = lambda b: _word(FS.AttentionHeadSoftcapV2, b)  # type: ignore[attr-defined]
AttentionHeadSoftcap.word_statics = WORD_STATICS  # type: ignore[attr-defined]


def _attention(B, S, q, kc, vc, *extra):
    """`AttentionSoftcap_v2`'s body on bits over `AttentionHeadSoftcap_v3` (GQA: head h reads KV head h // (NH/KVH); caches
    `[T][KVH][D]`; one MASKED_FROM for the row).  With ``extra`` (the two words, inputs) over `AttentionHeadSoftcap_v6`."""
    T, NH, KVH, D = S.T, S.NH, S.KVH, S.D
    G = NH // KVH
    W = 16 * D
    if extra:
        head = bind(AttentionHeadSoftcapIn, T=T, D=D, BN=S.BN, DOT=S.DOT, INV=S.INV, MASKED_FROM=S.MASKED_FROM, TANH=S.TANH,
                    EX2=S.EX2)
    else:
        head = bind(AttentionHeadSoftcap, T=T, D=D, BN=S.BN, CAP=S.CAP, DOT=S.DOT, INV=S.INV, MASKED_FROM=S.MASKED_FROM,
                    TANH=S.TANH, EX2=S.EX2)
    outs = []
    for g in range(KVH):
        kg = Coll(Array(T, Array(D, BF16)), strided_view(kc, g * W, W, KVH * W, 1, T * W))
        vg = Coll(Array(T, Array(D, BF16)), strided_view(vc, g * W, W, KVH * W, 1, T * W))
        outs.append(B.batch(head, q[g * G:(g + 1) * G], kg, vg, *extra, axes=(0, None, None) + (None,) * len(extra)))
    return Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


AttentionSoftcap = CompositeDefinition(
    "AttentionSoftcap", 3, ("T", "NH", "KVH", "D", "BN", "CAP", "DOT", "INV", "MASKED_FROM", "TANH", "EX2"),
    lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
               Array(S.NH, Array(S.D, BF16))),
    _attention, doc=_attention.__doc__, conformance="exact-model-tested")
AttentionSoftcap.word = lambda b: _word(FS.AttentionSoftcapV2, b)  # type: ignore[attr-defined]
AttentionSoftcap.word_statics = WORD_STATICS  # type: ignore[attr-defined]


def _words_in(type_of):
    """``type_of``'s signature with the two words `softcap_mul` and `scale_log2` last parameters (`value_inputs`' v5 chain's)."""
    def sig(S):
        params, ret = type_of(S)
        return ((*params, ("softcap_mul", F32), ("scale_log2", F32)), ret)
    return sig


AttnBlockSoftcapIn = CompositeDefinition(
    "AttnBlockSoftcap", 6, ("D", "NVIS", "FIRST", "BN", "DOT", "CHECK", "TANH", "EX2"),
    _words_in(lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
                          ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32))),
                         Tuple(F32, Array(4, F32), Array(S.D, F32)))),
    _block, doc="`AttnBlockSoftcap_v5`'s body on bits: `AttnBlockSoftcap_v3` with `softcap_mul` and `scale_log2` inputs",
    conformance="exact-model-tested")
AttnBlockSoftcapIn.word = lambda b: _word(VI.AttnBlockSoftcapV5, b)  # type: ignore[attr-defined]
AttnBlockSoftcapIn.word_statics = WORD_STATICS  # type: ignore[attr-defined]

AttentionHeadSoftcapIn = CompositeDefinition(
    "AttentionHeadSoftcap", 6, ("T", "D", "BN", "DOT", "INV", "MASKED_FROM", "TANH", "EX2"),
    _words_in(lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
                         Array(S.D, BF16))),
    _head, doc="`AttentionHeadSoftcap_v5`'s body on bits over `AttnBlockSoftcap_v6`, every block reading the two words",
    conformance="exact-model-tested")
AttentionHeadSoftcapIn.word = lambda b: _word(VI.AttentionHeadSoftcapV5, b)  # type: ignore[attr-defined]
AttentionHeadSoftcapIn.word_statics = WORD_STATICS  # type: ignore[attr-defined]

AttentionSoftcapIn = CompositeDefinition(
    "AttentionSoftcap", 6, ("T", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM", "TANH", "EX2"),
    _words_in(lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                          ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                          ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
                         Array(S.NH, Array(S.D, BF16)))),
    _attention, doc="`AttentionSoftcap_v5`'s body on bits over `AttentionHeadSoftcap_v6`, every head reading the two words",
    conformance="exact-model-tested")
AttentionSoftcapIn.word = lambda b: _word(VI.AttentionSoftcapV5, b)  # type: ignore[attr-defined]
AttentionSoftcapIn.word_statics = WORD_STATICS  # type: ignore[attr-defined]
