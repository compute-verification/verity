"""`GemmBiasF32Epilogue_v1{K,N,DOT}`: a bf16 linear with a bias as cuBLASLt computes it off cc 8.x -- the ascending k16 chain, the bias
added in the FP32 epilogue, one rounding to bf16 -- and where it is bound (cc 12.0), where it is not (cc 8.x keeps `GemmBias_v1`; cc 9.0
is refused until a biased linear is recorded there)."""
from __future__ import annotations

import numpy as np
import pytest

from verity.evaluation import evaluate
from verity.ir.defs import bind
from verity.ml import gemm as core_gemm
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.kernels import rows as RK
from verity_vllm.program.kernels.kernel_registry import Declined
from verity_vllm.program.registry import gemm_bias
from verity_vllm.program.registry import gemm_targets as GT
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry import targets as TG
from verity_vllm.query import word as W

K, N = 32, 8
DOTS = {"hopper": TG.DOTS["HopperBF16WgmmaDot16"], "ampere": TG.DOTS["AmpereBF16TcDot16"]}
EDGES = (0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0x3F80, 0xBF80)


def _vectors(n_random: int, seed: int):
    rng = np.random.default_rng(seed)
    out = [([e] * K, [e] * (N * K), [e] * N) for e in EDGES]
    for _ in range(n_random):
        out.append(tuple(rng.choice(EDGES, n).tolist() for n in (K, N * K, N)))
        out.append(tuple(((rng.integers(0x3E00, 0x4040, n)) | (rng.integers(0, 2, n) << 15)).tolist() for n in (K, N * K, N)))
    return out


def _finite(*words) -> bool:
    return all((int(v) & 0x7F80) != 0x7F80 for ws in words for v in ws)


@pytest.mark.parametrize("arch", sorted(DOTS))
def test_each_word_is_the_chain_then_one_f32_add_of_the_bias(arch):
    dot = DOTS[arch]
    spec, chain = bind(gemm_bias.GemmBiasF32Epilogue, K=K, N=N, DOT=dot), bind(core_gemm.DotBf16, K=K, DOT=dot)
    for x, w, b in _vectors(12, 5):
        got = list(evaluate(spec, x, w, b))
        accs = [np.asarray(evaluate(chain, x, w[j * K:(j + 1) * K])).reshape(-1)[0] for j in range(N)]
        want = [P.F2fpBf16.evaluate(P.F32Add.evaluate(int(accs[j]), P.Bf16ToF32.evaluate(b[j]))) for j in range(N)]
        assert got == want, (arch, x[:2], b[:2])


@pytest.mark.parametrize("arch", sorted(DOTS))
def test_the_row_kernel_is_the_definition_and_declines_non_finite_lanes(arch):
    dot = DOTS[arch]
    spec = bind(gemm_bias.GemmBiasF32Epilogue, K=K, N=N, DOT=dot)
    statics = {"K": K, "N": N, "DOT": dot}
    checked = 0
    for x, w, b in _vectors(24, 11):
        want = np.array(evaluate(spec, x, w, b), dtype=np.uint16)
        args = (np.array(x, np.uint16), np.array(w, np.uint16), np.array(b, np.uint16))
        if _finite(x, w, b, want.tolist()):
            assert RK.ROWS["GemmBiasF32Epilogue_v1"].fn(statics, args)["0"].tolist() == want.tolist()
            checked += 1
        elif not _finite(x, w):
            with pytest.raises(Declined):
                RK.ROWS["GemmBiasF32Epilogue_v1"].fn(statics, args)
    assert checked >= 24


def test_the_f32_epilogue_is_not_gemm_bias_v1():
    """Two roundings (GemmBias_v1: the word, then a bf16 add) against one (the FP32 epilogue): different words on ordinary vectors."""
    rng = np.random.default_rng(3)
    old, new = bind(gemm_bias.GemmBias, K=K, N=N), bind(gemm_bias.GemmBiasF32Epilogue, K=K, N=N, DOT=P.AmpereBF16TcDot16)
    differ = 0
    for _ in range(16):
        x, w, b = (((rng.integers(0x3E00, 0x4040, n)) | (rng.integers(0, 2, n) << 15)).tolist() for n in (K, N * K, N))
        differ += sum(p != q for p, q in zip(evaluate(old, x, w, b), evaluate(new, x, w, b)))
    assert differ > 0


def test_bound_on_cc_12_only_where_the_linear_is_cublaslt():
    sm120, ada, hopper = (TargetProfile(compute_capability=cc) for cc in ((12, 0), (8, 9), (9, 0)))
    spec = TG.gemm_bias_spec(sm120, 2048, 3072)
    assert spec.id == bind(gemm_bias.GemmBiasF32Epilogue, K=2048, N=3072, DOT=TG.tensor_core_dot(sm120)).id
    assert spec.id.startswith("GemmBiasF32Epilogue_v1{") and "HopperBF16WgmmaDot16_v1" in spec.id
    assert TG.gemm_bias_spec(ada, 2048, 3072).id == bind(gemm_bias.GemmBias, K=2048, N=3072).id
    with pytest.raises(TG.UnregisteredTarget, match="with a bias on hopper"):
        TG.gemm_bias_spec(hopper, 2048, 3072)
    assert {key for key, gt in GT.REGISTERED.items() if gt.bias_epilogue == "f32"} == {"blackwell_consumer"}


def test_q_word_cuts_one_unit_per_output_word_with_no_committed_interior():
    rule = W.unit_rule(bind(gemm_bias.GemmBiasF32Epilogue, K=K, N=N, DOT=P.AmpereBF16TcDot16))
    assert rule["units_per_call"] == N and rule["committed_interior_words"] == 0 and rule["max_out_bits"] == 16, rule
