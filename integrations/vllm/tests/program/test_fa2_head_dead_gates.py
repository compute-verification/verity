"""`AttentionHead_v5` / `_v11` write their first block's steps in their own body: a Call of `AttnBlock{FIRST=True}` was passed a
zero running state it ignores (one dead `Const32` per head).

Each head is held word for word to its old form, rebuilt here, on edge and random inputs, and has no dead gate
(`liveness.dead_gates`, transitive: circuit-check's count), where the old form had one.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.circuits.liveness import dead_gates
from verity.ml.circuits.ops.prims import ZERO32
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.ml.definitions.registry import b1, fa2_check_inf as F2, hopper, prims as P
from verity.ml.definitions.registry.prims import BF16, F32

RNG = np.random.default_rng(20261024)
#: bf16 words at the edges: ±0, ±inf, NaNs, the largest finite, the smallest subnormal, ±1, one ulp above 1
BF16_EDGES = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F7F, 0xFF7F, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x3F81]


def _old(name: str, of, body) -> CompositeDefinition:
    return CompositeDefinition(name, 1, of.statics, of.signature, body, register=False)


def _dead(fn) -> int:
    return dead_gates(Program(fn), transitive=True).computed


def _bf16(x) -> list[int]:
    return [int(w) for w in (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16)]


def _f32(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


def _rows(n: int, count: int) -> list[list[int]]:
    """Rows of n bf16 words: constant rows of each edge word, edge words mixed, and random words and normals."""
    rows = [[w] * n for w in BF16_EDGES]
    rows += [[int(w) for w in RNG.choice(BF16_EDGES, n)] for _ in range(count)]
    rows += [[int(w) for w in RNG.integers(0, 1 << 16, n)] for _ in range(count)]
    rows += [_bf16(RNG.normal(0, 10.0 ** RNG.integers(-3, 4), n)) for _ in range(count)]
    return rows


def _same(new, old, *args) -> None:
    assert evaluate(new, *args) == evaluate(old, *args), args


# --- Attention_v5 / v11: the first block's steps in the head ---------------------------------------------------------------


def _old_head(B, S, q, k, v, *extra, block):
    """`_check_inf_head` before: the first block a Call of `AttnBlock{FIRST=True}` on a zero running state."""
    T, D, BN, DOT, MF = S.T, S.D, S.BN, S.DOT, S.MASKED_FROM
    nb = (T + BN - 1) // BN
    m = l = o = None
    zero_row = None
    for b in range(nb - 1, -1, -1):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb, vb = k[lo:hi], v[lo:hi]
        if nvis < BN:
            if zero_row is None:
                zero_row = Coll(Array(D, BF16), concat([B.call(b1.ZERO16).refs] * D))
            pad = [zero_row.refs] * (BN - nvis)
            kb = Coll(Array(BN, Array(D, BF16)), concat([kb.refs] + pad))
            vb = Coll(Array(BN, Array(D, BF16)), concat([vb.refs] + pad))
        first = b == nb - 1
        blk = bind(block, D=D, NVIS=nvis, FIRST=first, BN=BN, DOT=DOT, CHECK=b >= MF)
        if first:
            z = B.call(ZERO32)
            st = B.call(blk, q, kb, vb, z, Coll(Array(4, F32), concat([z.refs] * 4)), Coll(Array(D, F32), concat([z.refs] * D)), *extra)
        else:
            st = B.call(blk, q, kb, vb, m, l, o, *extra)
        m, l, o = st[0], st[1], st[2]
    total = B.call(P.F32AddFtz, B.call(P.F32AddFtz, l[0], l[2]), B.call(P.F32AddFtz, l[1], l[3]))
    y = B.batch(P.F32MulFtz, o, B.call(S.INV, total), axes=(0, None))
    return B.batch(P.F2fpBf16, y, axes=(0,))


OLD_HEAD_V5 = _old("AttentionHeadZeroState", F2.AttentionHeadV5, lambda B, S, q, k, v: _old_head(B, S, q, k, v, block=F2.AttnBlockV5))
OLD_HEAD_V11 = _old("AttentionHeadZeroStateScale", F2.AttentionHeadV11,
                    lambda B, S, q, k, v, s: _old_head(B, S, q, k, v, s, block=F2.AttnBlockV11))
FA2 = dict(D=16, BN=16, DOT=hopper.HopperBF16WgmmaDot16, INV=P.Fa2InvSum)


@pytest.mark.parametrize("T,MF", [(1, 0), (16, 0), (17, 1), (17, 0), (40, 2), (40, 1)])
def test_attention_head_v5_and_v11_equal_the_zero_state_form(T, MF):
    new5, old5 = bind(F2.AttentionHeadV5, T=T, MASKED_FROM=MF, **FA2), bind(OLD_HEAD_V5, T=T, MASKED_FROM=MF, **FA2)
    new11, old11 = bind(F2.AttentionHeadV11, T=T, MASKED_FROM=MF, **FA2), bind(OLD_HEAD_V11, T=T, MASKED_FROM=MF, **FA2)
    assert (_dead(new5), _dead(old5), _dead(new11), _dead(old11)) == (0, 1, 0, 1)
    D = FA2["D"]
    cases = [(row[:D], row[:T * D], row[-T * D:]) for row in _rows(2 * T * D, 2)]
    cases += [(_bf16(RNG.normal(0, 1, D)), _bf16(RNG.normal(0, 1, T * D)), _bf16(RNG.normal(0, 1, T * D))) for _ in range(4)]
    k_neg_inf = [0xFF80] * (T * D)                      # every score -inf: the running max stays -inf through the blocks
    cases.append((_bf16(RNG.normal(0, 1, D)), k_neg_inf, _bf16(RNG.normal(0, 1, T * D))))
    for q, k, v in cases:
        _same(new5, old5, q, k, v)
        _same(new11, old11, q, k, v, _f32(0.25 * 1.4426950408889634))


def test_attention_v5_and_v11_have_no_dead_gate():
    for fn in (F2.AttentionV5, F2.AttentionV11):
        assert _dead(bind(fn, T=40, NH=4, KVH=2, MASKED_FROM=1, **FA2)) == 0
