"""FA2's per-iteration `Check_inf` head with each key block cut into units of at most 256 rows at any key count.

`AttentionHead_v19` Calls a whole key block, `AttnBlock_v5` (or `AttnBlockFirst_v1`), which reads a row a key and a row
or more a key's value: past a few dozen keys a unit holds more than a Flock proof block's 256 rows.  A block's online softmax
reads all its scores (its max is over every key), and its P·V is a chain of 16-key k-steps carrying O's FP32 accumulator,
so `AttentionHead_v21` Calls, for each block, its softmax step `AttnSoftmaxStep_v1` (q, the block's keys and the running
max and lane sums in; the new max, lane sums, rescale factor and P as bf16, in chunks of PK keys, out) and then, for each
chunk, `AttnPV_v1` (its P chunk, its keys' values and O in; O out), the block's first chunk also rescaling O, the head's
first chunk starting O at zero.  Its gates are v19's, each in a unit of its own; `AttnHeadEnd_v1` ends it.
"""
from __future__ import annotations

from verity.circuits.defs import bind, composite
from verity.circuits.ops.prims import ZERO32
from verity.circuits.refs import Coll as _Coll
from verity.circuits.refs import array_of, concat, tuple_of
from verity.circuits.types import Array, Tuple

from verity_vllm.definitions.registry import fa2_check_inf as F2
from verity_vllm.definitions.registry import prims as P
from verity_vllm.definitions.registry.b1 import ZERO16
from verity_vllm.definitions.registry.dead_free import AttnHeadEndV1
from verity_vllm.definitions.registry.prims import BF16, F32


def _chunks(nvis: int, pk: int) -> list[tuple[int, int]]:
    """A block of ``nvis`` keys' P chunks of ``pk`` keys, the last short."""
    return [(a, min(a + pk, nvis)) for a in range(0, nvis, pk)]


def _pk_refuse(pk) -> str | None:
    if not (type(pk) is int and pk >= 16 and pk % 16 == 0):
        return f"PK must be a positive multiple of 16, a P chunk whole k-steps (got PK={pk!r})"
    return None


def _vs_refuse(D, vs) -> str | None:
    if not (type(vs) is int and vs >= 1 and (D is None or D % vs == 0)):
        return f"VS must be a positive divisor of D (got D={D!r}, VS={vs!r})"
    return None


def _softmax_refuse(b: dict) -> str | None:
    if b.get("FIRST") and not b.get("CHECK"):
        return "the first iteration runs Check_inf (FIRST needs CHECK)"
    if (n := b.get("NVIS")) is not None and not (type(n) is int and n >= 1):
        return f"NVIS must be at least 1 (got {n!r})"
    return _pk_refuse(b["PK"]) if "PK" in b else None


def _softmax_signature(S):
    state = () if S.FIRST else (("m_prev", F32), ("l_prev", Array(4, F32)))
    out = (F32, Array(4, F32)) + (() if S.FIRST else (F32,)) + tuple(Array(b - a, BF16) for a, b in _chunks(S.NVIS, S.PK))
    return (("q", Array(S.D, BF16)), ("k", Array(S.NVIS, Array(S.D, BF16)))) + state, Tuple(*out)


@composite("AttnSoftmaxStep", 1, ["D", "NVIS", "FIRST", "CHECK", "DOT", "PK"], _softmax_signature,
           conformance="not-yet-established: AttnBlock_v5's steps A to B5", refuse=_softmax_refuse)
def AttnSoftmaxStepV1(B, S, q, k, *state):
    """`AttnBlock_v5{D, NVIS, FIRST, BN, DOT, CHECK}`'s steps A to B5 on its NVIS keys: the scores, the block's max, the online
    softmax and P as bf16.  Returns the running max, the four lane sums, the factor O is rescaled by (not at FIRST, whose O
    starts at zero) and P in chunks of PK keys, each an output row its `AttnPV_v1` reads."""
    m_prev, l_prev = state if state else (None, None)
    m_new, lanes, scores_scale, pb = F2._softmax_step(B, S, q, k, m_prev, l_prev, None)
    scale = () if S.FIRST else (scores_scale,)
    return tuple_of(m_new, array_of(lanes), *scale, *(array_of(pb[a:b]) for a, b in _chunks(S.NVIS, S.PK)))


def _pv_refuse(b: dict) -> str | None:
    if b.get("FIRST") and b.get("RESCALE"):
        return "an accumulator that starts at zero is not rescaled (FIRST and RESCALE exclude each other)"
    if (n := b.get("NV")) is not None and not (type(n) is int and n >= 1):
        return f"NV must be at least 1 (got {n!r})"
    return _vs_refuse(b.get("D"), b["VS"]) if "VS" in b else None


def _pv_signature(S):
    o = () if S.FIRST else (("o", Array(S.D, F32)),)
    scale = (("scale", F32),) if S.RESCALE else ()
    return o + scale + (("p", Array(S.NV, BF16)), ("v", Array(S.NV * S.D // S.VS, Array(S.VS, BF16)))), Array(S.D, F32)


@composite("AttnPV", 1, ["D", "NV", "FIRST", "RESCALE", "DOT", "VS"], _pv_signature,
           conformance="not-yet-established: AttnBlock_v5's step C on a chunk of its keys", refuse=_pv_refuse)
def AttnPVV1(B, S, *args):
    """`AttnBlock_v5`'s step C on NV of a block's keys: O += P V in ⌈NV / 16⌉ ascending k-steps of 16 keys, from O = 0 at
    FIRST (the head's first visited block's first chunk), from O·scale at RESCALE (a later block's first chunk), else from O
    as it is.  v's element ``t·D/VS + s`` is key t's columns [s·VS, (s+1)·VS); at NV not a multiple of 16 (the first visited
    block's last chunk) P's words past NV are zero and so are V's rows to the k-step's end."""
    D, NV = S.D, S.NV
    o_in = None if S.FIRST else args[0]
    scale = args[1] if S.RESCALE else None
    p, v = args[-2], args[-1]
    v = _Coll(Array(NV, Array(D, BF16)), v.refs)
    zero16 = None
    if NV % 16:
        zero_row = _Coll(Array(D, BF16), concat([B.call(ZERO16).refs] * D))
        v = _Coll(Array(NV + -NV % 16, Array(D, BF16)), concat([v.refs] + [zero_row.refs] * (-NV % 16)))
        zero16 = B.call(ZERO16)
    if S.FIRST:
        o = [B.call(ZERO32) for _ in range(D)]
    elif S.RESCALE:
        o = [B.call(P.F32MulFtz, o_in[d], scale) for d in range(D)]
    else:
        o = [o_in[d] for d in range(D)]
    return array_of(F2._pv_steps(B, D, S.DOT, [p[j] for j in range(NV)], v, o, zero16, -(-NV // 16)))


def _head_refuse(b: dict) -> str | None:
    if (bn := b.get("BN")) is not None and not (type(bn) is int and bn >= 16 and bn % 16 == 0):
        return f"BN must be a positive multiple of 16 (got {bn!r})"
    if "PK" in b and (why := _pk_refuse(b["PK"])):
        return why
    return _vs_refuse(b.get("D"), b["VS"]) if "VS" in b else None


@composite("AttentionHead", 21, ["T", "D", "BN", "DOT", "INV", "MASKED_FROM", "PK", "VS"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
                      Array(S.D, BF16)),
           conformance="not-yet-established: AttentionHead_v19's function", refuse=_head_refuse)
def AttentionHeadV21(B, S, q, k, v):
    """`AttentionHead_v19{T, D, BN, DOT, INV, MASKED_FROM}` with each key block a softmax step and its P·V chunks: keys
    (nb - 1)·BN ... T - 1 first, nb = ⌈T / BN⌉, then each block b below it with `CHECK = b >= MASKED_FROM`; a block's
    `AttnSoftmaxStep_v1{D, NVIS, FIRST, CHECK, DOT, PK}` then its `AttnPV_v1{D, NV, FIRST, RESCALE, DOT, VS}` for each chunk
    of PK keys in ascending order, and `AttnHeadEnd_v1{D, INV}` on the last block's lane sums and the last chunk's O."""
    T, D, BN, DOT, MF, PK, VS = S.T, S.D, S.BN, S.DOT, S.MASKED_FROM, S.PK, S.VS
    nb = (T + BN - 1) // BN
    if not 0 <= MF <= nb - 1:
        raise ValueError(f"AttentionHead_v21: MASKED_FROM {MF} outside blocks 0..{nb - 1} "
                         "(the first block is always a Check_inf iteration)")
    m = l = o = None
    for b in range(nb - 1, -1, -1):
        lo, hi, first = b * BN, min(T, (b + 1) * BN), b == nb - 1
        step = bind(AttnSoftmaxStepV1, D=D, NVIS=hi - lo, FIRST=first, CHECK=first or b >= MF, DOT=DOT, PK=PK)
        st = B.call(step, q, k[lo:hi], *(() if first else (m, l)))
        m, l, at = st[0], st[1], 2 if first else 3
        for j, (a, e) in enumerate(_chunks(hi - lo, PK)):
            pv = bind(AttnPVV1, D=D, NV=e - a, FIRST=first and j == 0, RESCALE=not first and j == 0, DOT=DOT, VS=VS)
            vs = _Coll(Array((e - a) * D // VS, Array(VS, BF16)), v[lo + a:lo + e].refs)
            o = B.call(pv, *(() if first and j == 0 else (o,)), *((st[2],) if not first and j == 0 else ()), st[at + j], vs)
    return B.call(bind(AttnHeadEndV1, D=D, INV=S.INV), l, o)


@composite("Attention", 21, ["T", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM", "PK", "VS"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="not-yet-established: Attention_v19's function", refuse=_head_refuse)
def AttentionV21(B, S, q, kc, vc):
    """`Attention_v19{T, NH, KVH, D, BN, DOT, INV, MASKED_FROM}` over `AttentionHead_v21`."""
    return F2._check_inf_attention(B, S, q, kc, vc, AttentionHeadV21, (), ("PK", "VS"))


__all__ = ["AttentionHeadV21", "AttentionV21", "AttnPVV1", "AttnSoftmaxStepV1"]
