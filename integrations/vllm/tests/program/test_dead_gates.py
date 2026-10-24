"""The template-internal dead gates removed, each Definition computing the same function as before.

* `AttentionHead_v5` / `_v11` write their first block's steps in their own body: a Call of `AttnBlock{FIRST=True}` was passed a
  zero running state it ignores (one dead `Const32` per head).
* `RMSNormTriton_v1` (and `RMSNormTriton_v3`, `MeanTriton_v1`) reduce each warp by lane 0's adds alone (`warp_reduce`): the
  butterfly's other lanes added in every round, 129 dead adds a warp.
* `TokenSelect_v1`, `TopPMask_v1` and `GumbelSelectF32_v1` / `GumbelSelectF32GivenGreedy_v1` stop their scan one lane early and
  take the last lane's index or store alone: the scan's final best and i + 1 were read by nothing.

Each is held word for word to its old form, rebuilt here, on edge and random inputs, and has no dead gate (`liveness.dead_gates`,
transitive: circuit-check's count), where the old form had the counts named.
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.circuits.liveness import dead_gates
from verity.ml.circuits.ops.fp32 import F32AddFtz
from verity.ml.circuits.ops.prims import ZERO32, const
from verity.ml.circuits.ops.scalar import BitNot, F32Eq, SelectF32
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array
from verity_vllm.program.registry import b1, dense, fa2_check_inf as F2, hopper, prims as P, sampling as SA, value_inputs as VI
from verity_vllm.program.registry.prims import BF16, F32

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


# --- RMSNormTriton: lane 0's adds ------------------------------------------------------------------------------------------


def _all_lanes(B, vals, masks, add):
    """The butterfly before: every lane adds in every round."""
    n, cur = len(vals), list(vals)
    for m in masks:
        cur = [B.call(add, cur[i], cur[i ^ m]) if (i ^ m) < n else cur[i] for i in range(n)]
    return cur[0]


def _with_all_lanes(module, body):
    def run(B, S, *args):
        with mock.patch.object(module, "lane0_butterfly", _all_lanes):
            return body(B, S, *args)
    return run


OLD_RMSNORM = _old("RMSNormTritonAllLanes", b1.RMSNormTriton, _with_all_lanes(b1, b1.RMSNormTriton.body))
OLD_RMSNORM_V3 = _old("RMSNormTritonV3AllLanes", VI.RMSNormTritonV3, _with_all_lanes(b1, VI.RMSNormTritonV3.body))
OLD_MEAN = _old("MeanTritonAllLanes", dense.MeanTriton, _with_all_lanes(dense, dense.MeanTriton.body))


@pytest.mark.parametrize("N,warps", [(16, 4), (64, 4), (1024, 4), (2048, 8)])
def test_rmsnorm_triton_equals_the_all_lanes_form(N, warps):
    new, old = bind(b1.RMSNormTriton, N=N, EPS=1e-5), bind(OLD_RMSNORM, N=N, EPS=1e-5)
    assert (_dead(new), _dead(old)) == (0, 129 * warps)
    rows = _rows(N, 3)
    tiny = [0x0001 + (j % 7) for j in range(N)]          # subnormal squares: the sum underflows to 0 and eps alone is left
    for x, w in zip(rows + [tiny], rows[3:] + rows[:4]):
        _same(new, old, x, w)


def test_rmsnorm_triton_v3_and_mean_triton_equal_the_all_lanes_form():
    N = 64
    new, old = bind(VI.RMSNormTritonV3, N=N), bind(OLD_RMSNORM_V3, N=N)
    newm, oldm = bind(dense.MeanTriton, N=N), bind(OLD_MEAN, N=N)
    assert (_dead(new), _dead(old), _dead(newm), _dead(oldm)) == (0, 516, 0, 516)
    rows = _rows(N, 2)
    for x, w in zip(rows, rows[1:] + rows[:1]):
        for eps in (_f32(1e-5), _f32(1e-6), 0, 0x7F800000):
            _same(new, old, x, w, eps)
        xf = [int(u) for u in (np.asarray(x, dtype=np.uint32) << 16)]
        _same(newm, oldm, xf)


# --- the selects' last scan iteration --------------------------------------------------------------------------------------


def _old_token_select(B, S, logits):
    init = tuple_of(logits[0], B.call(const(32, 0)), B.call(const(32, 1)))
    return B.scan(b1.ArgmaxStep, init, xs=(logits[1:S.V],))[0][1]


def _old_topp_mask(B, S, logits, top_p, splits):
    keep = B.call(SA.topp_mask_word(S.V), logits, top_p, splits)
    return B.scan(bind(SA.TopPMaskStep, V=S.V), B.call(const(32, 0)), xs=(logits,), shared=(keep,))[1]


def _old_first_max(B, S, logits, seed, pos, noisy):
    key = B.call(SA.GumbelStreamKey, seed, pos)
    zero = B.call(const(32, 0))
    y0 = B.call(SelectF32, noisy, B.call(F32AddFtz, logits[0], B.call(SA.GumbelNoiseLane, key, zero)), logits[0])
    init = tuple_of(y0, zero, B.call(const(32, 1)))
    return B.scan(SA.GumbelSelectStep, init, xs=(logits[1:S.V],), shared=(key, noisy))[0][1]


OLD_TOKEN_SELECT = _old("TokenSelectFullScan", b1.TokenSelect, _old_token_select)
OLD_TOPP_MASK = _old("TopPMaskFullScan", SA.TopPMask, _old_topp_mask)
OLD_SELECT_F32 = _old("GumbelSelectF32FullScan", SA.GumbelSelectF32, lambda B, S, logits, seed, pos, temp: _old_first_max(
    B, S, logits, seed, pos, B.call(BitNot, B.call(F32Eq, temp, B.call(const(32, 0))))))
OLD_GIVEN_GREEDY = _old("GumbelSelectF32GivenGreedyFullScan", SA.GumbelSelectF32GivenGreedy,
                        lambda B, S, logits, seed, pos, greedy: _old_first_max(B, S, logits, seed, pos, B.call(BitNot, greedy)))


def _old_topp_shared(B, S, logits, top_p, seed, pos, temp, splits):
    greedy = B.call(F32Eq, temp, B.call(const(32, 0)))
    scaled = B.call(bind(SA.TemperatureScaleGivenGreedy, V=S.V), logits, temp, greedy)
    masked = B.call(bind(OLD_TOPP_MASK, V=S.V), scaled, top_p, splits)
    return B.call(bind(OLD_GIVEN_GREEDY, V=S.V), masked, seed, pos, greedy)


OLD_TOPP_SHARED = _old("GumbelTopPTokenSelectSharedGreedyFullScan", SA.GumbelTopPTokenSelectSharedGreedy, _old_topp_shared)
#: at V = 1 the old selects scan no lane, and a zero-length scan's carry reads other words (`TokenSelect_v1{V=1}` gave the logit),
#: so V = 1 is held to what it means instead (`test_one_logit_selects_index_0`)
VOCABS = [2, 3, 16, 257]


def _logit_rows(V: int) -> list[list[int]]:
    """`_rows` and rows whose maximum is first, last, tied at both ends, or behind a NaN."""
    rows = _rows(V, 4)
    for at in ({0}, {V - 1}, {0, V - 1}):
        rows.append([0x3F80 if j in at else 0xBF80 for j in range(V)])
    rows.append([0x7FC0] + [0x3F80] * (V - 1))
    return rows


@pytest.mark.parametrize("V", VOCABS)
def test_token_select_equals_the_full_scan(V):
    new, old = bind(b1.TokenSelect, V=V), bind(OLD_TOKEN_SELECT, V=V)
    assert (_dead(new), _dead(old)) == (0, 3)
    for row in _logit_rows(V):
        _same(new, old, row)


def test_one_logit_selects_index_0():
    assert _dead(bind(b1.TokenSelect, V=1)) == 0
    for (w,) in _logit_rows(1):
        seed, pos = int(RNG.integers(0, 1 << 63)), int(RNG.integers(0, 1 << 20))
        assert evaluate(bind(b1.TokenSelect, V=1), [w]) == (0,)
        for temp in TEMPS:
            assert evaluate(bind(SA.GumbelSelectF32, V=1), [w << 16], seed, pos, temp) == (0,)
            assert evaluate(bind(SA.GumbelTopPTokenSelectSharedGreedy, V=1), [w], TOP_P[1], seed, pos, temp, 32) == (0,)


TOP_P = [_f32(p) for p in (1.0, 0.95, 0.5, 0.0)]
TEMPS = [_f32(t) for t in (0.0, -0.0, 1.0, 0.8)]
SPLITS = [1, 2, 32, 3]


def _f32_rows(V: int) -> list[list[int]]:
    """f32 rows: each bf16 logit row widened, and random normals."""
    rows = [[int(w) << 16 for w in r] for r in _logit_rows(V)]
    return rows + [[_f32(x) for x in RNG.normal(0, 3, V)] for _ in range(4)]


@pytest.mark.parametrize("V", [1, *VOCABS])
def test_topp_mask_equals_the_full_scan(V):
    new, old = bind(SA.TopPMask, V=V), bind(OLD_TOPP_MASK, V=V)
    assert (_dead(new), _dead(old)) == (0, 2)
    for i, row in enumerate(_f32_rows(V)):
        _same(new, old, row, TOP_P[i % 4], SPLITS[(i // 4) % 4])


@pytest.mark.parametrize("V", VOCABS)
def test_gumbel_selects_equal_the_full_scan(V):
    new, old = bind(SA.GumbelSelectF32, V=V), bind(OLD_SELECT_F32, V=V)
    newg, oldg = bind(SA.GumbelSelectF32GivenGreedy, V=V), bind(OLD_GIVEN_GREEDY, V=V)
    assert (_dead(new), _dead(newg), _dead(old), _dead(oldg)) == (0, 0, 3, 3)
    for i, row in enumerate(_f32_rows(V)):
        seed, pos = int(RNG.integers(0, 1 << 63)), int(RNG.integers(0, 1 << 20))
        temp = TEMPS[i % 4]
        _same(new, old, row, seed, pos, temp)
        _same(newg, oldg, row, seed, pos, int((temp & 0x7FFFFFFF) == 0))


@pytest.mark.parametrize("V", VOCABS)
def test_gumbel_top_p_shared_greedy_equals_the_full_scans(V):
    new, old = bind(SA.GumbelTopPTokenSelectSharedGreedy, V=V), bind(OLD_TOPP_SHARED, V=V)
    assert (_dead(new), _dead(old)) == (0, 5)
    for i, row in enumerate(_logit_rows(V)):
        seed, pos = int(RNG.integers(0, 1 << 63)), int(RNG.integers(0, 1 << 20))
        _same(new, old, row, TOP_P[i % 4], seed, pos, TEMPS[(i // 4) % 4], SPLITS[(i // 16) % 4])
