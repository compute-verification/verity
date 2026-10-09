"""GLM-5.3's router (`registry.moe.MoeRouterSigmoidBias_v1`, DeepSeek-V3's `noaux_tc` at one group) against a model of vLLM's
`single_group_topk_{block,warp}_kernel<float, float, int, SIGMOID, MaxE, 8>` written the way the CUDA source runs it: per expert
`preprocess_score` with the sigmoid's SASS (`moe_twins.nv_sigmoid_bits`: FSETP.GE, FSEL and the LOP3 sign OR as they are), the 64-bit keys
`(TwiddleIn(score) << 32) | (65535 - expert)` sorted per lane and `reduceTopK`'s rounds over 32 lanes (the winner's list shifted, an
`InvalidScore` filler at its end), then `write_outputs` with the 32-lane butterfly over every lane.  The Program states the same
function with each value computed once: the selection's order as comparisons, the sum's +0 lanes dropped, the sign as a select."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import codec
from verity.ml.circuits.defs import bind
from verity.ml.circuits.program import Program
from verity_vllm.program.registry import moe
from verity_vllm.program.kernels import moe_twins
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry import ref_prims as R

NEG_INF, PINF, QNAN = 0xFF800000, 0x7F800000, 0x7FC00000
TIERS = ((128, 8), (256, 8), (384, 8), (512, 8), (512, 22), (768, 16), (896, 16), (1024, 16))   # SigmoidBiasTiers


def _finite(w: int) -> bool:
    return (w >> 23) & 0xFF != 0xFF


def _twiddle(w: int) -> int:
    return ~w & 0xFFFFFFFF if w >> 31 else w | 0x80000000


def preprocess(x: int, b: int) -> tuple[int, int]:
    """`preprocess_score<float, float, SIGMOID>`: (unbiased, selection)."""
    if not (_finite(x) and _finite(b)):
        return 0, NEG_INF
    sig = moe_twins.nv_sigmoid_bits(x)
    biased = P.F32Add.evaluate(b, sig)
    return sig, 0 if P.f32_of(biased) == 0 else biased


def kernel(x: list[int], b: list[int], topk: int, scale: float, launch: str) -> tuple[list[int], list[int]]:
    """One token row of `single_group_topk_{launch}_kernel` at the tier `invoke` picks for (E, topk), renormalize=True."""
    E = len(x)
    max_e = next(e for e, k in TIERS if e >= E and k >= topk)
    pre = [preprocess(x[i], b[i]) for i in range(E)]
    lanes = []
    for lane in range(32):
        keys = []
        for index in range((max_e + 31) // 32):
            e = index * 32 + lane
            score = pre[e][1] if e < E else NEG_INF
            keys.append((_twiddle(score) << 32) | (0xFFFF & (65535 - e)))
        lanes.append((sorted(keys, reverse=True), [index * 32 + lane for index in range(len(keys))]))
    top, packed = [], None
    for kk in range(topk):
        for keys, idx in lanes:
            if kk > 0 and packed == keys[0]:
                keys[:] = keys[1:] + [(_twiddle(NEG_INF) << 32) | (0xFFFF & (65535 - idx[-1]))]
        packed = max(keys[0] for keys, _ in lanes)
        top.append(packed)
    lane_score = [(_twiddle_out(k >> 32), 65535 - (k & 0xFFFF)) for k in top]
    unbiased = []
    for lane in range(32):
        if lane >= topk:
            unbiased.append(0)
            continue
        s, e = lane_score[lane]
        u = pre[e][0] if launch == "block" else R.F32Sub.evaluate(s, b[e])
        unbiased.append(u if s != NEG_INF else 0)
    v = list(unbiased)
    for mask in (16, 8, 4, 2, 1):
        v = [P.F32Add.evaluate(v[t], v[t ^ mask]) for t in range(32)]
    total = v[0]
    weights = []
    for lane in range(topk):
        if all(s == NEG_INF for s, _ in lane_score):
            weights.append(P.F32Div.evaluate(P.bits_of(1.0), P.bits_of(float(topk))))
        elif lane_score[lane][0] != NEG_INF:
            q = P.F32Div.evaluate(P.bits_of(np.float32(scale)), P.F32Add.evaluate(total, P.bits_of(np.float32(1e-20))))
            weights.append(P.F32Mul.evaluate(unbiased[lane], q))
        else:
            weights.append(0)
    return weights, [e for _, e in lane_score]


def _twiddle_out(t: int) -> int:
    return t & 0x7FFFFFFF if t >> 31 else ~t & 0xFFFFFFFF


def _f32(a) -> list[int]:
    return np.asarray(a, np.float32).view(np.uint32).astype(np.int64).tolist()


def rows(E: int, n: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Router rows that reach every branch: ties in the score and in score + bias, a bias that decides, zero and -0 sums, the
    sigmoid's three ranges and its saturation, non-finite logits and biases (fewer finite experts than slots, none at all)."""
    rng = np.random.default_rng(seed)
    out = [([0] * E, [0] * E), ([NEG_INF] * E, [0] * E), ([QNAN] * E, _f32(rng.standard_normal(E))),
           (_f32(rng.standard_normal(E)), [PINF] * E), (_f32([-30.0] * E), [0] * E), (_f32([30.0] * E), _f32([-1.0] * E)),
           (_f32(rng.standard_normal(E)), _f32([-1.0] * E))]
    few = [QNAN] * E
    few[E - 1], few[E // 2] = P.bits_of(-3.0), P.bits_of(0.5)
    out += [(few, [0] * E), ([0] * E, [PINF] * (E - 1) + [0]), (few, [NEG_INF] * (E // 2) + [0] * (E - E // 2))]
    for r in range(n):
        kind = r % 6
        if kind == 0:
            x, b = rng.standard_normal(E) * 4, rng.standard_normal(E) * 1e-2
        elif kind == 1:
            x, b = np.repeat(rng.standard_normal(E // 2 + 1), 2)[:E], np.zeros(E)
        elif kind == 2:
            x, b = rng.choice(np.array([-1.2, 0.6, 1.2000000476837158, 18.0, 18.03, -40.0]), E), rng.choice([0.0, -0.5, 0.25], E)
        elif kind == 3:
            x = rng.standard_normal(E) * 1e-3
            b = rng.standard_normal(E) * 1e-7
        elif kind == 4:
            x, b = rng.standard_normal(E) * 3, -0.5 - np.float32(0.5) * np.ones(E)
        else:
            x, b = rng.uniform(-25, 25, E), rng.standard_normal(E) * 1e3
        xw, bw = _f32(x), _f32(b)
        for i in rng.choice(E, int(rng.integers(0, E // 2 + 2)) if r % 5 == 0 else int(rng.integers(0, 3)), replace=False):
            if rng.integers(2):
                xw[int(i)] = int(rng.choice([QNAN, PINF, NEG_INF, 0x80000000, 0x00000001, 0xFF7FFFFF, 0x7F7FFFFF]))
            else:
                bw[int(i)] = int(rng.choice([QNAN, PINF, NEG_INF, 0x80000000, 0x7F7FFFFF, 0xFF7FFFFF]))
        out.append((xw, bw))
    return out


SPECIAL = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, PINF, NEG_INF, QNAN, 0xFFC00000, 0x7FFFFFFF, 0x7F7FFFFF, 0xFF7FFFFF,
           0x3F99999A, 0x3F999999, 0x41902CB4, 0x41902CB3, 0xBF99999A, 0xBF999999, 0xC1902CB4, 0xC1902CB3]


def test_nv_sigmoid_is_the_sass():
    rng = np.random.default_rng(5)
    words = SPECIAL + [int(w) for w in rng.integers(0, 1 << 32, 3000, dtype=np.int64)]
    words += _f32(np.concatenate([rng.uniform(-20, 20, 3000), rng.standard_normal(1000) * 1e-3]))
    for w in words:
        assert evaluate(moe.NvSigmoid, [w]) == (moe_twins.nv_sigmoid_bits(w),), hex(w)


def test_nv_sigmoid_saturates_and_is_symmetric_where_the_sass_is():
    assert moe_twins.nv_sigmoid_bits(P.bits_of(18.03)) == P.bits_of(1.0) and moe_twins.nv_sigmoid_bits(P.bits_of(-18.03)) == 0
    assert moe_twins.nv_sigmoid_bits(0) == P.bits_of(0.5) == moe_twins.nv_sigmoid_bits(0x80000000)
    assert moe_twins.nv_sigmoid_bits(QNAN) == QNAN


@pytest.mark.parametrize("launch", ["block", "warp"])
@pytest.mark.parametrize("E,TOPK,n", [(8, 2, 60), (16, 8, 40), (256, 8, 4)])
def test_the_program_is_the_kernel(launch, E, TOPK, n):
    d = bind(moe.MoeRouterSigmoidBias, E=E, TOPK=TOPK, SCALE=2.5, KERNEL=launch)
    for x, b in rows(E, n, seed=E * 31 + TOPK):
        w, ids = kernel(x, b, TOPK, 2.5, launch)
        assert list(evaluate(d, x + b)) == w + ids, (x, b)


def test_ties_go_to_the_lower_expert_and_the_bias_only_selects():
    x, b = [0] * 16, [0] * 16
    x[9] = x[3] = P.bits_of(2.0)
    b[12] = P.bits_of(5.0)
    out = list(evaluate(bind(moe.MoeRouterSigmoidBias, E=16, TOPK=4, SCALE=2.5, KERNEL="block"), x + b))
    assert out[4:] == [12, 3, 9, 0]
    w = [P.f32_of(v) for v in out[:4]]
    assert w[1] == w[2] and abs(sum(w) - 2.5) < 1e-6 and abs(w[0] / w[3] - 1.0) < 1e-6


def test_no_finite_expert_spreads_the_weight():
    out = list(evaluate(bind(moe.MoeRouterSigmoidBias, E=16, TOPK=8, SCALE=2.5, KERNEL="warp"), [QNAN] * 16 + [0] * 16))
    assert out == [P.bits_of(0.125)] * 8 + list(range(8))


@pytest.mark.parametrize("bad", [{"KERNEL": "fast"}, {"TOPK": 33}])
def test_statics_are_checked(bad):
    st = {"E": 64, "TOPK": 8, "SCALE": 2.5, "KERNEL": "block"} | bad
    with pytest.raises(ValueError):
        Program(bind(moe.MoeRouterSigmoidBias, **st))


#: the served word Programs (256 experts, 8 slots, x2.5), by launch: `codec.program_digest` of each standalone Program
SERVED = {"block": "59895b10a0a99cfe2276a8be5ef4afddebb606630ef7781e13425d937e841ea3e9d95bb9f32f70fd0d02b0e1d949b7414020bf79d6a089caa7f83886eb704f7d",
          "warp": "9ff429aef6778d86029b3b14c66ef8011bb4183105ebf321ee55704a568f625392527c39025907e352819d6081e18c456bb728bf4c0e2c03a734d405a8a75216"}


@pytest.mark.parametrize("launch", sorted(SERVED))
def test_the_served_router_is_pinned(launch):
    prog, _ = standalone_call(bind(moe.MoeRouterSigmoidBias, E=256, TOPK=8, SCALE=2.5, KERNEL=launch))
    assert codec.program_digest(codec.encode_program(prog)) == SERVED[launch]
