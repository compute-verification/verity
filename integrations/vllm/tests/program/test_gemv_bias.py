"""`GemvBiasF32_v1{K,N,V,T}`: cuBLAS gemvx at M = 1 with a bias (strided FFMA chains, a halving tree, the bias in FP32), its row, its
(K, N) -> (V, T) table and where it is bound (cc 12.0 only)."""
from __future__ import annotations

import numpy as np
import pytest

from verity.evaluation import evaluate
from verity.ir.defs import bind
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.kernels import rows as RK  # noqa: F401  (imports gemv_rows, which registers the row)
from verity_vllm.program.kernels import gemv_rows as GR
from verity_vllm.program.kernels.kernel_registry import Declined
from verity_vllm.program.registry import gemm_bias, gemv_bias
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry import targets as TG
from verity_vllm.program.registry.gemm_targets import GEMV_BIAS_TABLES
from verity_vllm.query import word as W



def _words(rng, n, scale=1.0):
    return ((rng.standard_normal(n) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tolist()


def _reference(x, w, b, K, V, T):
    """The structure written out: thread t's chunks c = t, t + T, ... in ascending k, FFMA from +0; a halving tree; + b; RNE."""
    widen = P.Bf16ToF32.evaluate
    parts = []
    for t in range(T):
        acc = 0
        for c in range(t, K // V, T):
            for k in range(c * V, c * V + V):
                acc = P.F32Fma.evaluate(widen(x[k]), widen(w[k]), acc)
        parts.append(acc)
    while len(parts) > 1:
        half = len(parts) // 2
        parts = [P.F32Add.evaluate(parts[t], parts[t + half]) for t in range(half)]
    return P.F2fpBf16.evaluate(P.F32Add.evaluate(parts[0], widen(b)))


@pytest.mark.parametrize("K, V, T", [(64, 1, 8), (64, 8, 8), (96, 1, 32), (48, 3, 16)])
def test_each_word_is_the_strided_chains_then_the_halving_tree_then_the_bias(K, V, T):
    rng = np.random.default_rng(K + V + T)
    N = 3
    spec = bind(gemv_bias.GemvBiasF32, K=K, N=N, V=V, T=T)
    for _ in range(4):
        x, w, b = _words(rng, K), _words(rng, N * K, 0.02), _words(rng, N, 0.02)
        got = list(evaluate(spec, x, w, b))
        assert got == [_reference(x, w[j * K:(j + 1) * K], b[j], K, V, T) for j in range(N)], (K, V, T)


def test_the_row_is_the_definition_and_declines_non_finite_words():
    rng = np.random.default_rng(5)
    K, N = 64, 4
    for V, T in ((1, 8), (8, 8), (1, 16)):
        spec = bind(gemv_bias.GemvBiasF32, K=K, N=N, V=V, T=T)
        for scale in (1.0, 2.0 ** -40, 2.0 ** 40):
            x, w, b = _words(rng, K, scale), _words(rng, N * K, scale), _words(rng, N)
            want = np.array(evaluate(spec, x, w, b), dtype=np.uint16)
            got = GR.gemv_bias_f32(np.array(x, np.uint16), np.array(w, np.uint16), np.array(b, np.uint16), K=K, N=N, V=V, T=T)
            assert got.tolist() == want.tolist(), (V, T, scale)
    x = np.array([0x7F80] + [0x3F80] * (K - 1), np.uint16)
    with pytest.raises(Declined):
        GR.gemv_bias_f32(x, np.full(N * K, 0x3F80, np.uint16), np.zeros(N, np.uint16), K=K, N=N, V=1, T=8)


def test_the_gemv_word_is_not_the_f32_epilogue_chain():
    """+2^25 at k = 0, -2^25 at k = 16 and +1 at k = 1: with T = 16, k = 0 and 16 share a thread and cancel before the 1 joins
    (gemv: 1), while the ascending k16 chain absorbs the 1 into 2^25 at its first step (0)."""
    K = 32
    x, w, b = [0x3F80] * K, [0x0000] * K, [0x0000]
    w[0], w[16], w[1] = 0x4C00, 0xCC00, 0x3F80
    gemv = bind(gemv_bias.GemvBiasF32, K=K, N=1, V=1, T=16)
    chain = bind(gemm_bias.GemmBiasF32Epilogue, K=K, N=1, DOT=TG.DOTS["HopperBF16WgmmaDot16"])
    assert list(evaluate(gemv, x, w, b)) == [0x3F80] and list(evaluate(chain, x, w, b)) == [0x0000]


def test_the_table_entries_are_well_formed():
    for key, table in GEMV_BIAS_TABLES.items():
        for (K, N), (V, T) in table.items():
            assert K % V == 0 and T & (T - 1) == 0 and K // V >= T, (key, K, N, V, T)


def test_bound_on_cc_12_from_the_table_and_refused_elsewhere():
    sm120 = TargetProfile(compute_capability=(12, 0))
    assert TG.gemv_bias_spec(sm120, 1536, 2048).id == bind(gemv_bias.GemvBiasF32, K=1536, N=2048, V=1, T=16).id
    assert TG.gemv_bias_spec(sm120, 768, 2304).id == bind(gemv_bias.GemvBiasF32, K=768, N=2304, V=48, T=16).id
    with pytest.raises(TG.UnregisteredTarget, match="unsupported:gemv-table"):
        TG.gemv_bias_spec(sm120, 1024, 1024)
    for cc in ((9, 0), (8, 9)):
        with pytest.raises(TG.UnregisteredTarget, match="no gemv table"):
            TG.gemv_bias_spec(TargetProfile(compute_capability=cc), 1536, 2048)


def test_q_word_cuts_one_unit_per_output_word_with_no_committed_interior():
    rule = W.unit_rule(bind(gemv_bias.GemvBiasF32, K=64, N=5, V=1, T=8))
    assert rule["units_per_call"] == 5 and rule["committed_interior_words"] == 0 and rule["max_out_bits"] == 16, rule
