"""The non-batch-invariant CUDA RMSNorm word/Boolean Definitions and numpy twins."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops.boolean import mufu as BM
from verity_vllm.program.kernels import twins as TW
from verity_vllm.program.frontend import targets as TG
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.registry import boolean_norms as BN
from verity_vllm.program.registry import cuda_rmsnorm as C


@pytest.mark.parametrize(
    "M,N,pure,fused",
    [
        (1, 6144, 768, 1024), (42, 6144, 768, 1024), (43, 6144, 768, 1024), (64, 6144, 768, 1024),
        (256, 6144, 256, 256), (2048, 6144, 256, 256),
        (1, 7168, 896, 1024), (42, 7168, 896, 1024), (43, 7168, 896, 1024), (64, 7168, 896, 1024),
        (256, 7168, 256, 256), (2048, 7168, 256, 256),
    ],
)
def test_launch_blocks_are_the_pinned_vllm_policy(M: int, N: int, pure: int, fused: int) -> None:
    assert C.launch_block(M, N, fused=False) == pure
    assert C.launch_block(M, N, fused=True) == fused


def test_launch_policy_refuses_shapes_outside_the_width_8_path() -> None:
    with pytest.raises(ValueError, match="positive"):
        C.launch_block(0, 6144, fused=False)
    with pytest.raises(ValueError, match="multiple of 8"):
        C.launch_block(1, 6145, fused=True)


@pytest.mark.parametrize("M,N,pure,fused", [(1, 6144, 768, 1024), (256, 6144, 256, 256), (1, 7168, 896, 1024)])
def test_target_selectors_keep_batch_invariant_definitions_and_bind_default_launch_blocks(
    M: int, N: int, pure: int, fused: int
) -> None:
    batch_invariant = TargetProfile.default()
    default = batch_invariant.with_overrides(batch_invariant=False)

    assert TG.rmsnorm_spec(batch_invariant, N, 1e-5, M).base_id == "RMSNormTriton_v1"
    assert TG.rmsnorm_fused_spec(batch_invariant, N, 1e-5, M=M).base_id == "RMSNormFusedCuda_v2"

    pure_spec = TG.rmsnorm_spec(default, N, 1e-5, M)
    fused_spec = TG.rmsnorm_fused_spec(default, N, 1e-5, M=M)
    assert pure_spec.base_id == "RMSNormCudaVector8_v1" and pure_spec.bindings["BLOCK"] == pure
    assert fused_spec.base_id == "RMSNormFusedCudaVector8_v1" and fused_spec.bindings["BLOCK"] == fused


@pytest.mark.parametrize("cls", [TW.RMSNormCudaVector8Twin, TW.RMSNormFusedCudaVector8Twin])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_numpy_twins_match_word_definitions_on_random_and_special_words(cls, eps: float) -> None:
    rng = np.random.default_rng(0xCADA + (cls is TW.RMSNormFusedCudaVector8Twin))
    for kind in ("gauss", "wide", "special", "extreme", "subnormal", "boundary"):
        rep = TW.check_instances(cls, lambda _rng: {"N": 72, "EPS": eps, "BLOCK": 4}, rng, 1, (kind,))
        assert rep["status"] == "pass" and not rep["mismatches"], rep


@pytest.mark.parametrize("fused,block", [(False, 2), (True, 16)])
def test_boolean_v2_has_the_word_v1_view_and_matches_it(fused: bool, block: int) -> None:
    N, EPS = 16, 1e-5
    rng = np.random.default_rng(44 + fused)
    x = TW.rand_bf16(rng, 4 * N, "special").reshape(4, N).astype(np.uint64)
    w = TW.rand_bf16(rng, 4 * N, "special").reshape(4, N).astype(np.uint64)
    if fused:
        r = TW.rand_bf16(rng, 4 * N, "special").reshape(4, N).astype(np.uint64)
        target = bind(BN.RMSNormFusedCudaVector8, N=N, EPS=EPS, BLOCK=block, RSQRT=bind(BM.RsqrtApprox))
        got = np.asarray(BS.evaluate_words(target, x, r, w), np.uint64).reshape(4, -1)
        word = bind(C.RMSNormFusedCudaVector8, N=N, EPS=EPS, BLOCK=block)
        expect = TW.RMSNormFusedCudaVector8Twin(word)([x[0], r[0], w[0]])
    else:
        target = bind(BN.RMSNormCudaVector8, N=N, EPS=EPS, BLOCK=block, RSQRT=bind(BM.RsqrtApprox))
        got = np.asarray(BS.evaluate_words(target, x, w), np.uint64).reshape(4, -1)
        word = bind(C.RMSNormCudaVector8, N=N, EPS=EPS, BLOCK=block)
        expect = TW.RMSNormCudaVector8Twin(word)([x[0], w[0]])
    assert TR.word_of(target).id == word.id
    assert np.array_equal(got[0], expect)
