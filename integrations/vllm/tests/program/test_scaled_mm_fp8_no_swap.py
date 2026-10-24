"""`ScaledMmFp8NoSwap_v1{K,N,BIAS,DOT}`: the per-tensor FP8 linear as `_C.cutlass_scaled_mm` computes it WITHOUT swap_ab (ScaleA = the
activation scale), on the target's e4m3 step -- bound on cc 12.0 -- and the red team's counterexamples to the swapped order that
`ScaledMmFp8_v1` (cc 9.0's batch-invariant dispatch) computes."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.prims import BlackwellE4m3QmmaDot32, HopperE4m3QgmmaDot32
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.registry import fp8
from verity_vllm.program.registry import prims as P
from verity_vllm.program.frontend import targets as TG
from verity_vllm.query import word as W

K, N = 32, 4
DOT = BlackwellE4m3QmmaDot32

#: red-team-vllm-semantics, finding `fp8-per-tensor-scale-order`, run r20260930-081238-476e (`fp8_order_repro.json`): the operand
#: sets its 4,000,000-draw search found on which the two scale orders round to different words.  (xq, wq as 32 e4m3 bytes in hex;
#: sx, sw, the chain's accumulator; the unswapped word bf16(sx*(sw*acc)); the swapped word bf16(sw*(sx*acc))).
RED_TEAM = [
    ("1575" + "00" * 30, "5811" + "00" * 30, 0x3C0447DB, 0x3C1F9608, 0x41020000, 0x3A28, 0x3A27),
    ("2a33" + "00" * 30, "060e" + "00" * 30, 0x3A776DEF, 0x3C87AC2F, 0x3CB80000, 0x34BD, 0x34BC),
    ("3473" + "00" * 30, "2111" + "00" * 30, 0x3C8EABA0, 0x3A80068E, 0x40C96000, 0x38E1, 0x38E0),
    ("1442" + "00" * 30, "0b07" + "00" * 30, 0x3C3D982F, 0x3C0D3421, 0x3D102000, 0x366B, 0x366C),
]


def _word(defn, *args) -> int:
    return int(np.asarray(evaluate(defn, *args)).reshape(-1)[0])


def _bytes(h: str) -> list[int]:
    return list(bytes.fromhex(h))


@pytest.mark.parametrize("case", range(len(RED_TEAM)))
def test_the_red_teams_counterexamples_take_the_unswapped_word(case):
    xq, wq, sx, sw, acc, unswapped, swapped = RED_TEAM[case]
    xq, wq = _bytes(xq), _bytes(wq)
    assert _word(bind(fp8.DotE4m3V2, K=K, DOT=DOT), xq, wq) == acc
    assert _word(bind(fp8.ScaledMmFp8NoSwapCoordinate, K=K, BIAS=False, DOT=DOT), xq, sx, wq, sw) == unswapped
    assert _word(bind(fp8.ScaledMmFp8Coordinate, K=K, BIAS=False), xq, sx, wq, sw) == swapped != unswapped
    row = evaluate(bind(fp8.ScaledMmFp8NoSwap, K=K, N=1, BIAS=False, DOT=DOT), xq, sx, wq, sw)   # w [1, K], flat
    assert [int(v) for v in np.asarray(row).reshape(-1)] == [unswapped]


def test_each_word_is_the_step_chain_then_the_weight_scale_then_the_activation_scale():
    """With a bias the second multiply is one FMA with the widened bias (ScaledEpilogueBias): bf16(fma(sx, sw * acc, f32(b)))."""
    rng = np.random.default_rng(12)
    chain = bind(fp8.DotE4m3V2, K=K, DOT=DOT)
    for bias in (False, True):
        spec = bind(fp8.ScaledMmFp8NoSwap, K=K, N=N, BIAS=bias, DOT=DOT)
        for _ in range(24):
            xq = rng.integers(0, 0x7F, K).tolist()
            w = [rng.integers(0, 0x7F, K).tolist() for _ in range(N)]
            sx, sw = (int(np.float32(rng.uniform(2.0 ** -12, 2.0 ** -2)).view(np.uint32)) for _ in range(2))
            b = rng.integers(0x3E00, 0x4040, N).tolist()
            got = [int(v) for v in np.asarray(evaluate(spec, xq, sx, sum(w, []), sw, *((b,) if bias else ()))).reshape(-1)]
            want = []
            for j in range(N):
                t = P.F32Mul.evaluate(sw, _word(chain, xq, w[j]))
                y = P.F32Fma.evaluate(sx, t, P.Bf16ToF32.evaluate(b[j])) if bias else P.F32Mul.evaluate(sx, t)
                want.append(P.F2fpBf16.evaluate(y))
            assert got == want, (bias, xq[:2])


def test_the_chain_is_the_cc_12_step_not_hoppers():
    """The 26-bit adder keeps a product that Hopper's 14-bit FP8 adder truncates against the accumulator: 1 * 1 in the first step,
    then 2^-9 * 2^-6 = 2^-15 in the second."""
    xq = [0x38] + [0x00] * 31 + [0x01] + [0x00] * 31
    wq = [0x38] + [0x00] * 31 + [0x08] + [0x00] * 31
    assert _word(bind(fp8.DotE4m3V2, K=64, DOT=DOT), xq, wq) == 0x3F800100
    assert _word(bind(fp8.DotE4m3V2, K=64, DOT=HopperE4m3QgmmaDot32), xq, wq) == 0x3F800000 == _word(bind(fp8.DotE4m3, K=64), xq, wq)


def test_bound_on_cc_12_only_with_its_own_step():
    sm, hopper, ada = (TargetProfile(compute_capability=cc) for cc in ((12, 0), (9, 0), (8, 9)))
    for bias in (False, True):
        spec = TG.scaled_mm_fp8_spec(sm, 1536, 2048, bias)
        assert spec.id == bind(fp8.ScaledMmFp8NoSwap, K=1536, N=2048, BIAS=bias, DOT=DOT).id
        assert "BlackwellE4m3QmmaDot32_v1" in spec.id and not spec.id.startswith("ScaledMmFp8_v1")
    for tp in (hopper, ada):
        with pytest.raises(TG.UnregisteredTarget, match="per-tensor FP8 linear"):
            TG.scaled_mm_fp8_spec(tp, 1536, 2048, False)


def test_q_word_cuts_one_unit_per_output_word_with_no_committed_interior():
    rule = W.unit_rule(bind(fp8.ScaledMmFp8NoSwap, K=64, N=N, BIAS=True, DOT=DOT))
    assert rule["units_per_call"] == N and rule["committed_interior_words"] == 0 and rule["max_out_bits"] == 16, rule
