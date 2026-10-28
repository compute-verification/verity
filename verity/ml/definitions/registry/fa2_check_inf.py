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

A key block cut into proof units: `AttnBlockSoftmax_v1{D, NVIS, FIRST, DOT, CHECK, PK, LAST}` is `AttnBlock_v5`'s steps up to P (and,
past the first block, O's rescale), its P in groups of PK keys, and `AttnBlockPv_v1{D, N, VS, DOT, FIRST}` one group's P·V k-steps, its
V as `AttnBlockFirst_v2`'s slices of VS columns.  Softmax, then its groups in order, each reading the last one's O, is `AttnBlock_v5`
(and at the first block `AttnBlockFirst_v1`) word for word; at LAST, the head's last key block, whose state only the epilogue reads
(the lane sums and O, never the running max), the step returns no running max, which it still computes for P and the rescale.  A block
of 192 keys over a 64-dim head, a key and a value slice a proof unit's row, is three units of at most 201 rows at PK 96 and VS 32, where
`AttnBlockFirst_v2{VS=32}` alone is 580 rows, past a proof block's 256.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Affine, Coll as _Coll, Explicit, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.gemm import DotBf16 as DotBf16V2
from verity.ml.circuits.ops.prims import ZERO32, const
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.b1 import ZERO16, _lane, scale_log2_bits
from verity.ml.definitions.registry.prims import A16, BF16, F32


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


def _check_inf_block(B, S, q, k, v, m_prev, l_prev, o_prev, scale_log2, *, read_pads_only=False):
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
        scale_log2 = B.call(const(32, scale_log2_bits(D)))
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
    zero16 = B.call(ZERO16) if (NVIS % 16 if read_pads_only else NVIS < BN) else None
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
    last -> first, block b with `CHECK = b >= MASKED_FROM` (FA2's masking iterations; `attention_vocab.fa2_masked_from`), the first
    visited block's steps (`AttnBlock_v5{FIRST=True}`'s) in this body; epilogue `sum = (l0 + l2) + (l1 + l3)`, `inv = INV(sum)`,
    `out = bf16(o * inv)`."""
    return _check_inf_head(B, S, q, k, v, AttnBlockV5, ())


def _check_inf_head(B, S, q, k, v, block, extra, *, read_pads_only=False):
    """`read_pads_only`: pad the first visited block's V rows only to the k-steps it reads, and its K rows not at all (it reads no key
    past `nvis`), so that no zero padding goes unread when `nvis` is a multiple of 16 below `BN`."""
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
        if nvis < BN and read_pads_only:
            if nvis % 16:  # the last k-step's V rows past nvis meet P's zero words
                zero_row = _Coll(Array(D, BF16), concat([B.call(ZERO16).refs] * D))
                vb = _Coll(Array(nvis + -nvis % 16, Array(D, BF16)), concat([vb.refs] + [zero_row.refs] * (-nvis % 16)))
        elif nvis < BN:  # pad the block's K/V rows with zeros (Clear_OOB_MN); masked columns are never read
            if zero_row is None:
                zero_row = _Coll(Array(D, BF16), concat([B.call(ZERO16).refs] * D))
            pad = [zero_row.refs] * (BN - nvis)
            kb = _Coll(Array(BN, Array(D, BF16)), concat([kb.refs] + pad))
            vb = _Coll(Array(BN, Array(D, BF16)), concat([vb.refs] + pad))
        if b == nb - 1:  # a Call of AttnBlock{FIRST=True} would pass it a running state it ignores, so its steps are written here
            first = SimpleNamespace(D=D, NVIS=nvis, FIRST=True, BN=BN, DOT=DOT, CHECK=True)
            st = _check_inf_block(B, first, q, kb, vb, None, None, None, extra[0] if extra else None, read_pads_only=read_pads_only)
        else:
            st = B.call(bind(block, D=D, NVIS=nvis, FIRST=False, BN=BN, DOT=DOT, CHECK=b >= MF), q, kb, vb, m, l, o, *extra)
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


# A key block as its softmax step and its P·V k-steps in groups of PK keys (module docstring).


def pv_groups(NVIS: int, PK: int) -> list[tuple[int, int]]:
    """The key ranges [lo, hi) of a block of NVIS keys in groups of PK, the last the rest."""
    return [(lo, min(NVIS, lo + PK)) for lo in range(0, NVIS, PK)]


def _softmax_refuse(b: dict) -> str | None:
    pk, nvis = b.get("PK"), b.get("NVIS")
    if pk is not None and not (type(pk) is int and pk >= 16 and pk % 16 == 0):
        return f"PK must be whole k-steps of 16 keys (got PK={pk!r})"
    if nvis is not None and not (type(nvis) is int and nvis >= 1):
        return f"NVIS must be at least 1 key (got NVIS={nvis!r})"
    if b.get("FIRST") is True and b.get("CHECK") is False:
        return "the first iteration runs Check_inf (FIRST needs CHECK)"
    return None


def _softmax_type(S):
    params = [("q", Array(S.D, BF16)), ("k", Array(S.NVIS, Array(S.D, BF16)))]
    if not S.FIRST:
        params += [("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32))]
    m = () if S.LAST else (F32,)
    o = () if S.FIRST else (Array(S.D, F32),)
    return tuple(params), Tuple(*m, Array(4, F32), *o, *(Array(hi - lo, BF16) for lo, hi in pv_groups(S.NVIS, S.PK)))


def softmax_steps(B, S, q, k, state, *, guard_max_scaled_only):
    """A key block's steps up to P and O's rescale, `_check_inf_block`'s in its order; ``guard_max_scaled_only``: FA2's guard
    (`AttnBlock_v5`, the unguarded rescale reads the running max as it is), else FA3's (`AttnBlock_v4`, both read it)."""
    D, NVIS, FIRST, DOT, CHECK = S.D, S.NVIS, S.FIRST, S.DOT, S.CHECK
    m_prev, l_prev, o_prev = (None, None, None) if FIRST else state
    dot = bind(DotBf16V2, K=D, DOT=DOT)
    s = [B.call(dot, q, k[c]) for c in range(NVIS)]
    bmax = s[0]
    for c in range(1, NVIS):
        bmax = B.call(P.F32Max, bmax, s[c])
    m_new = bmax if FIRST else B.call(P.F32Max, m_prev, bmax)
    m_use = B.call(P.GuardNegInfZero, m_new) if CHECK or guard_max_scaled_only else m_new
    scale_log2 = B.call(const(32, scale_log2_bits(D)))
    if not FIRST:
        scores_scale = B.call(P.MufuEx2Ftz, B.call(P.F32MulFtz, B.call(P.F32SubFtz, m_prev, m_use if CHECK else m_new), scale_log2))
    max_scaled = B.call(P.F32MulFtz, m_use, scale_log2)
    p = [B.call(P.MufuEx2Ftz, B.call(P.F32FmaSubFtz, s[c], scale_log2, max_scaled)) for c in range(NVIS)]
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
    pb = [B.call(P.F2fpBf16, x) for x in p]
    o = [] if FIRST else [array_of([B.call(P.F32MulFtz, o_prev[d], scores_scale) for d in range(D)])]
    m = [] if S.LAST else [m_new]
    return tuple_of(*m, array_of(lanes), *o, *(array_of(pb[lo:hi]) for lo, hi in pv_groups(NVIS, S.PK)))


@composite("AttnBlockSoftmax", 1, ["D", "NVIS", "FIRST", "DOT", "CHECK", "PK", "LAST"], _softmax_type,
           conformance="not-yet-established: AttnBlock_v5's steps up to P", refuse=_softmax_refuse)
def AttnBlockSoftmaxV1(B, S, q, k, *state):
    """`AttnBlock_v5{D, NVIS, FIRST, BN, DOT, CHECK}`'s steps up to P over its NVIS keys: S = Q K^T, the row max and the running
    max (FA2's `Check_inf`: CHECK guards the rescale, `max * scale` is guarded in every block), P, the lane sums, and when not
    FIRST O's rescale `o_prev * scale`.  Returns (m unless LAST, l, o unless FIRST, P_0 ... P_{G-1}), P_g the bf16 P of keys
    [g PK, min(NVIS, (g + 1) PK)), each the `AttnBlockPv_v1` group that reads it.  LAST is the head's last key block (block 0),
    whose running max no later step reads."""
    return softmax_steps(B, S, q, k, state, guard_max_scaled_only=True)


def _pv_refuse(b: dict) -> str | None:
    D, vs, n = b.get("D"), b.get("VS"), b.get("N")
    if vs is not None and not (type(vs) is int and vs >= 1 and (D is None or D % vs == 0)):
        return f"VS must be a positive divisor of D (got D={D!r}, VS={vs!r})"
    if n is not None and not (type(n) is int and n >= 1):
        return f"N must be at least 1 key (got N={n!r})"
    return None


def _pv_type(S):
    params = [("p", Array(S.N, BF16)), ("v", Array(S.N * S.D // S.VS, Array(S.VS, BF16)))]
    return tuple(params + ([] if S.FIRST else [("o", Array(S.D, F32))])), Array(S.D, F32)


@composite("AttnBlockPv", 1, ["D", "N", "VS", "DOT", "FIRST"], _pv_type,
           conformance="not-yet-established: AttnBlock_v5's P·V k-steps over N keys", refuse=_pv_refuse)
def AttnBlockPvV1(B, S, p, v, *o_in):
    """``o + P V`` over N keys in ascending k-steps of 16 (`AttnBlock_v5`'s C over a group of its keys), from +0 when FIRST; v's
    element ``t D/VS + s`` is key t's value columns [s VS, (s+1) VS) (`AttnBlockFirst_v2`'s slices).  When N % 16 (the first
    visited block's last group) P is padded with +0 words and V with +0 rows to the last k-step, as `AttnBlockFirst_v1` pads."""
    D, N, DOT = S.D, S.N, S.DOT
    v = _Coll(Array(N, Array(D, BF16)), v.refs)
    if N % 16:
        zero_row = _Coll(Array(D, BF16), concat([B.call(ZERO16).refs] * D))
        v = _Coll(Array(N + -N % 16, Array(D, BF16)), concat([v.refs] + [zero_row.refs] * (-N % 16)))
    zero16 = B.call(ZERO16) if N % 16 else None
    o = array_of([B.call(ZERO32) for _ in range(D)]) if S.FIRST else o_in[0]
    for k0 in range(0, N, 16):
        prow = _Coll(A16, concat([(p[k0 + j] if k0 + j < N else zero16).refs for j in range(16)]))
        vblk = v[k0:k0 + 16]
        vt_refs = strided_view(vblk, 0, 16, 1, D, 16 * D) if isinstance(vblk.refs, Affine) else \
            Explicit(tuple(vblk.refs[j * D + d] for d in range(D) for j in range(16)))
        o = B.batch(DOT, o, prow, _Coll(Array(D, A16), vt_refs), axes=(0, None, 0))
    return o


FA2_CHECK_INF_COMPOSITES = [AttnBlockV5, AttentionHeadV5, AttentionV5, AttnBlockV11, AttentionHeadV11, AttentionV11]

__all__ = ["AttnBlockPvV1", "AttnBlockSoftmaxV1", "AttnBlockV5", "AttentionHeadV5", "AttentionV5", "AttnBlockV11", "AttentionHeadV11",
           "AttentionV11", "FA2_CHECK_INF_COMPOSITES", "pv_groups", "softmax_steps"]
