"""The NVFP4 linear on cc 12.0 (`registry.nvfp4`): the activation quantizer `cvt_fp16_to_fp4` block by block, the block-scaled GEMM
over core's pinned step with the `alpha` epilogue, the device layouts the Commit opens, and where it is bound."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.prims import BlackwellNvf4OmmaDot64, F32ToE4m3Sat
from verity.ml.circuits.silicon.models import BLACKWELL_SM120_NVF4
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.registry import nvfp4 as N4
from verity_vllm.program.kernels import nvfp4_twins
from verity_vllm.program.registry import prims as P
from verity_vllm.program.frontend import targets as TG
from verity_vllm.query import word as W

DOT = BlackwellNvf4OmmaDot64
ONE = 0x3F800000


def f32(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def bf16(v: float) -> int:
    return f32(v) >> 16


def quant(block, g=ONE):
    r = evaluate(N4.Nvfp4QuantBlock, [bf16(v) if isinstance(v, float) else v for v in block], g)
    return list(r[:16]), int(r[16])


def test_a_block_at_global_scale_one_takes_its_max_to_six_and_rounds_every_element_to_nearest_even():
    """vecMax 6 -> SF = e4m3(6 * rcp(6)) = 1.0 (0x38), outputScale 1: each element is its own E2M1 rounding (ties to even)."""
    vals = [0.1, -0.2, 0.3, 6.0, 1.0, -3.0, 0.5, 0.25, 2.5, -1.25, 0.0, -0.0, 4.0, 5.0, 1.75, 0.75]
    codes, sf = quant(vals)
    assert sf == 0x38
    assert codes == [0, 8, 1, 7, 2, 13, 1, 0, 4, 10, 0, 8, 6, 6, 4, 2]


def test_the_scale_is_the_blocks_max_over_six_times_the_global_scale_in_e4m3():
    x = [float(v) for v in np.linspace(-3.0, 3.0, 16)]
    for g in (1.0, 2688.0 / 3.5, 3.7e5):
        codes, sf = quant(x, f32(g))
        m = max(abs(v) for v in x)
        want = P.F32Mul.evaluate(f32(g), P.F32Mul.evaluate(f32(m), P.MufuRcpFtz.evaluate(f32(6.0))))
        assert sf == F32ToE4m3Sat.evaluate(want), g


def test_a_zero_block_has_scale_zero_and_keeps_its_signed_zeros():
    codes, sf = quant([0.0, -0.0] * 8)
    assert sf == 0 and codes == [0, 8] * 8


def test_an_infinite_element_saturates_the_scale_and_its_code():
    codes, sf = quant([float("inf")] + [1.0] * 14 + [float("-inf")])
    assert sf == 0x7E and codes[0] == 7 and codes[15] == 15


def test_the_row_quantizer_is_its_blocks():
    rng = np.random.default_rng(4)
    x = [bf16(float(v)) for v in rng.standard_normal(64) * 3]
    q = evaluate(bind(N4.Nvfp4ActQuant, K=64), x, f32(448.0))
    codes, sfs = list(q[:64]), list(q[64:])
    for b in range(4):
        c, s = quant(x[16 * b:16 * b + 16], f32(448.0))
        assert codes[16 * b:16 * b + 16] == c and sfs[b] == s


def test_the_gemm_is_the_step_chain_then_alpha_then_bf16():
    rng = np.random.default_rng(6)
    K = 256
    for _ in range(12):
        xq, wq = rng.integers(0, 16, K).tolist(), rng.integers(0, 16, K).tolist()
        xs, ws = rng.integers(0x20, 0x48, K // 16).tolist(), rng.integers(0x20, 0x48, K // 16).tolist()
        alpha = f32(float(np.float32(rng.uniform(1e-5, 1e-1))))
        acc = 0
        for s in range(K // 64):
            acc = BLACKWELL_SM120_NVF4.step_scaled(acc, xq[64 * s:64 * s + 64], wq[64 * s:64 * s + 64], xs[4 * s:4 * s + 4], ws[4 * s:4 * s + 4])
        assert int(np.asarray(evaluate(bind(N4.DotNvfp4, K=K, DOT=DOT), xq, xs, wq, ws)).reshape(-1)[0]) == acc
        y = evaluate(bind(N4.ScaledMmNvfp4Coordinate, K=K, DOT=DOT), xq, xs, wq, ws, alpha)
        assert int(np.asarray(y).reshape(-1)[0]) == P.F2fpBf16.evaluate(P.F32Mul.evaluate(alpha, acc))


def test_the_device_layouts():
    assert nvfp4_twins.unpack_codes([0x21, 0xF0]) == [1, 2, 0, 15]
    K = 4096                                                 # 64 k-tiles of 4 blocks
    assert nvfp4_twins.sf_offset(0, 0, K) == 0 and nvfp4_twins.sf_offset(0, 3, K) == 3 and nvfp4_twins.sf_offset(0, 4, K) == 512
    assert nvfp4_twins.sf_offset(1, 0, K) == 16 and nvfp4_twins.sf_offset(32, 0, K) == 4 and nvfp4_twins.sf_offset(128, 0, K) == 64 * 512
    offs = {nvfp4_twins.sf_offset(r, b, K) for r in range(256) for b in range(K // 16)}
    assert len(offs) == 256 * K // 16 and max(offs) < 256 * K // 16      # a permutation of the padded tile


def test_bound_on_cc_12_only():
    sm, hopper, ada = (TargetProfile(compute_capability=cc) for cc in ((12, 0), (9, 0), (8, 9)))
    assert TG.nvfp4_gemm_spec(sm, 4096, 6144).id == bind(N4.ScaledMmNvfp4, K=4096, N=6144, DOT=DOT).id
    assert TG.nvfp4_quant_spec(sm, 4096).id == bind(N4.Nvfp4ActQuant, K=4096).id
    for tp in (hopper, ada):
        with pytest.raises(TG.UnregisteredTarget, match="NVFP4 linear"):
            TG.nvfp4_gemm_spec(tp, 4096, 6144)
        with pytest.raises(TG.UnregisteredTarget, match="NVFP4 linear"):
            TG.nvfp4_quant_spec(tp, 4096)
    with pytest.raises(TG.UnregisteredTarget, match="multiple of 64"):
        TG.nvfp4_gemm_spec(sm, 4096 + 32, 6144)


def test_q_word_cuts_one_unit_per_output_word_with_no_committed_interior():
    rule = W.unit_rule(bind(N4.ScaledMmNvfp4, K=128, N=4, DOT=DOT))
    assert rule["units_per_call"] == 4 and rule["committed_interior_words"] == 0 and rule["max_out_bits"] == 16, rule
