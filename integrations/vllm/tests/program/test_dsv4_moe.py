"""DeepSeek-V4's routers and clamped SwiGLU as word Definitions (`registry.dsv4_moe`): MUFU.LG2's interpolant at words measured
on the RTX PRO 6000, each Definition at its edges, and the statics checked.  The kernels' numpy twin, its agreement with these
Programs on many rows and the interpolant on all 2^23 mantissas are `tests/properties/test_dsv4_moe_capture.py`'s."""
from __future__ import annotations

import math

import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.program import Program
from verity_vllm.program.registry import act
from verity_vllm.program.registry import dsv4_moe as D
from verity_vllm.program.registry import prims as P

NEG_INF, PINF, QNAN, GPU_NAN = 0xFF800000, 0x7F800000, 0x7FC00000, 0x7FFFFFFF

#: `lg2.approx.ftz.f32(1.m)` as `LG2_MEASURED`'s run read it, at segment boundaries and both ends
MEASURED = {0x1: 0x3495F600, 0x2: 0x34F24A00, 0x3: 0x35274F00, 0x1FFFF: 0x3CB73C76, 0x20000: 0x3CB73CD1, 0x20001: 0x3CB73D2C,
            0xA0000: 0x3DDE4224, 0x3FFFFF: 0x3F15C019, 0x400000: 0x3F15C01B, 0x7E3039: 0x3F7D6088, 0x7FFFFE: 0x3F7FFFFE,
            0x7FFFFF: 0x3F7FFFFF}


def _w(x: float) -> int:
    return P.bits_of(x)


def _f(w: int) -> float:
    return float(P.f32_of(w))


def test_lg2_is_the_measured_unit_where_pinned():
    for m, y in MEASURED.items():
        assert D.MufuLg2Mantissa.evaluate(0x3F800000 | m) == y, hex(m)
    assert D.MufuLg2Mantissa.evaluate(0x3F800000) == 0


def test_lg2_reads_only_the_mantissa_and_stays_near_log2():
    for m in [*MEASURED, 0x155555, 0x2AAAAA, 0x6DB6DB]:
        y = D.MufuLg2Mantissa.evaluate(0x3F800000 | m)
        assert all(D.MufuLg2Mantissa.evaluate(e << 23 | s << 31 | m) == y for e in (0, 1, 0x80, 0xFF) for s in (0, 1))
        assert abs(_f(y) - math.log2(1 + m / 2 ** 23)) < 2 ** -21


def _weight(x: int) -> int:
    return evaluate(D.Dsv4HashWeight, [x])[0]


def test_the_hash_weight_at_its_edges():
    sqrt_ln2 = _w(math.sqrt(math.log(2)))
    assert _weight(0) == _weight(0x80000000) == _weight(1) == _weight(0x80000001) == sqrt_ln2
    assert _weight(_w(25.0)) == _w(5.0) and _weight(PINF) == PINF
    assert _weight(NEG_INF) == _weight(_w(-200.0)) == _weight(QNAN) == _weight(0xFFC00001) == 0
    for x in (-30.0, -8.0, -1.0, -0.25, 0.5, 1.0, 3.0, 10.0, 16.0, 40.0, 1e30):
        want = max(x, 0.0) + math.log1p(math.exp(-abs(x)))
        assert abs(_f(_weight(_w(x))) ** 2 - want) <= 3e-7 + 2e-7 * want, x


def _hash(logits: list[int], ids: list[int], E: int) -> list[int]:
    return list(evaluate(bind(D.MoeRouterHash, E=E, SCALE=2.5), logits + ids))


def test_the_hash_router_reads_the_table_ids_and_renormalises():
    logits = [_w(v) for v in (0.5, -1.0, 2.0, 0.0, -3.0, 1.5, 4.0, -0.5)]
    ids = [6, 2, 2, 0, 7, 3]
    out = _hash(logits, ids, 8)
    w = [_weight(logits[i]) for i in ids]
    total = P.F32Add.evaluate(P.F32Add.evaluate(P.F32Add.evaluate(w[0], w[4]), w[2]),
                              P.F32Add.evaluate(P.F32Add.evaluate(w[1], w[5]), w[3]))
    assert out == [P.F32Div.evaluate(P.F32Mul.evaluate(v, _w(2.5)), total) for v in w]
    assert out[1] == out[2] and abs(sum(map(_f, out)) - 2.5) < 1e-5


def test_the_hash_router_with_no_weight_or_an_infinite_one():
    assert _hash([NEG_INF] * 8, [0, 1, 2, 3, 4, 5], 8) == [0] * 6
    assert _hash([QNAN] * 8, [7, 6, 5, 4, 3, 2], 8) == [0] * 6
    assert _hash([PINF] + [0] * 7, [1, 0, 2, 3, 4, 5], 8) == [0, GPU_NAN, 0, 0, 0, 0]


def _topk(logits: list[int], bias: list[int]) -> tuple[list[int], list[int]]:
    out = list(evaluate(bind(D.MoeRouterSqrtSoftplus, E=len(logits), SCALE=2.5), logits + bias))
    return out[:D.TOPK], out[D.TOPK:]


def _score(x: int) -> int:
    return evaluate(D.Dsv4SqrtSoftplus, [x])[0]


def test_the_sqrtsoftplus_score_at_its_edges():
    assert _score(NEG_INF) == _score(_w(-200.0)) == 0 and _score(PINF) == PINF
    assert _score(QNAN) == _score(0xFFC00001) == GPU_NAN
    for x in (-60.0, -30.0, -8.0, -1.0, 0.0, 0.5, 1.0, 3.0, 10.0, 19.5, 20.0, 20.5, 25.0, 40.0):
        want = math.log1p(math.exp(x)) if x <= 20 else x
        assert abs(_f(_score(_w(x))) ** 2 - want) <= 2e-7 + 2e-6 * want, x


def test_the_triton_router_selects_by_score_plus_bias_and_weights_by_score():
    x = [_w(v) for v in (0.5, 2.0, -1.0, 2.0, 0.0, 1.0, -2.0, 3.0)]
    b = [0] * 8
    b[6] = _w(10.0)
    w, ids = _topk(x, b)
    assert ids == [6, 7, 1, 3, 5, 0]
    s = [_score(x[i]) for i in ids]
    assert w[2] == w[3] and abs(sum(map(_f, w)) - 2.5) < 1e-5
    assert all(abs(_f(a) / _f(c) - _f(w[0]) / _f(s[0])) < 1e-6 for a, c in zip(w, s))


def test_the_triton_router_picks_again_when_fewer_than_six_are_above_minus_inf():
    x = [_w(v) for v in (0.5, 2.0, -1.0, 2.0, 0.0, 1.0, -2.0, 3.0)]
    b = [NEG_INF, 0, NEG_INF, 0, NEG_INF, 0, NEG_INF, 0]
    _, ids = _topk(x, b)
    assert ids == [7, 1, 3, 5, 0, 0]


def test_a_nan_logit_ranks_at_minus_1e30_and_its_weight_is_the_gpu_nan():
    x = [QNAN] + [_w(1.0)] * 4 + [NEG_INF] * 3
    b = [0] * 5 + [NEG_INF] * 3
    w, ids = _topk(x, b)
    assert ids == [1, 2, 3, 4, 0, 0]
    one = _score(_w(1.0))
    assert w == [P.F32Mul.evaluate(one, _w(2.5))] * 4 + [GPU_NAN, GPU_NAN]


@pytest.mark.parametrize("E", [1, 5])
def test_the_triton_router_needs_six_experts(E):
    with pytest.raises(ValueError):
        Program(bind(D.MoeRouterSqrtSoftplus, E=E, SCALE=2.5))


def test_the_clamp_limit_is_a_positive_bf16_value():
    assert D.limit_bits(10.0) == 0x41200000 and D.limit_bits(7.0) == 0x40E00000
    for bad in (10.1, 0.0, -10.0, math.inf, math.nan, 2.0 ** -140):
        with pytest.raises(ValueError):
            D.limit_bits(bad)


def _bf(x: float) -> int:
    return P.bits_of(x) >> 16


def test_the_clamped_swiglu_clamps_the_gate_above_and_the_up_both_ways():
    one = bind(D.SiluMulClampBf16, LIMIT=10.0)
    ten, mten = _bf(10.0), _bf(-10.0)
    for g, u, gc, uc in ((12.0, 3.0, 10.0, 3.0), (3.0, 30.0, 3.0, 10.0), (3.0, -30.0, 3.0, -10.0), (-50.0, 2.0, -50.0, 2.0),
                         (math.inf, -math.inf, 10.0, -10.0)):
        assert evaluate(one, [_bf(g), _bf(u)])[0] == act.SiluMulBf16.evaluate(_bf(gc), _bf(uc)), (g, u)
    assert evaluate(one, [0x7FC0, 0x7FC0])[0] == act.SiluMulBf16.evaluate(ten, ten)
    assert evaluate(one, [0xFF80, 0x8000])[0] == act.SiluMulBf16.evaluate(0xFF80, 0x8000)
    assert mten == 0xC120


def test_the_clamped_swiglu_row_pairs_gate_and_up():
    g, u = [_bf(v) for v in (0.5, 11.0, -3.0, 2.0)], [_bf(v) for v in (1.0, -12.0, 4.0, 10.5)]
    want = [evaluate(bind(D.SiluMulClampBf16, LIMIT=10.0), [a, c])[0] for a, c in zip(g, u)]
    assert list(evaluate(bind(D.SiluMulClamp, I=4, LIMIT=10.0), g + u)) == want
