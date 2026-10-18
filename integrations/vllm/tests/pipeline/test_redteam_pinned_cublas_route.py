"""Red-team repro (circuits W6, claim 1): a served cuBLAS route that departs from the pin passes a TP Commit, digest equality intact.

A switched row's AOT Program equals its traced Build because both read the same pin: the Build's binding
(`program/frontend/rules/vllm_bindings/bf16_gemm.py`), the fold (`observe/fold/patterns/cublas.py`) and the predictor
(`predict/dsa.py` on cursor/aot-frontier-21f3) each take the split count of a bf16 GEMM from `bf16_gemm_check.default_cublas_route`,
a table of what cuBLAS chose on the capture box.  The serving process's cuBLAS decides its own route, and two environment variables
and two torch flags it inherits from whoever starts it move that route.  The capture of record (`glm53_bf16_gemm_capture`,
art:5fcc5a49..., H200 run r20261016-234133-bbc8) measured it: GLM-5.3's MoE gate at M 17 runs `nvjet_tss_64x8_64x16_2x4_h_bz_splitK_TNT`
plus `cublasLt::splitKreduce_kernel` (split 4) at vLLM's default, and `nvjet_tss_64x8_64x16_2x4_h_bz_TNT` (no split, word-exact against
`GemmF32Out_v1`) under CUBLAS_WORKSPACE_CONFIG=:16:8, CUBLASLT_WORKSPACE_SIZE=1 and the reduced-precision flags off.

GLM-5.3 B1 TP8's Commit of record (r20261016-213336-6c35, the row's record art:440154ed...) commits 153,600 rows of
`CublasSplitKGemmF32Out_v1` (the prefill router) and 48,216 of `CublasSplitKGemm_v1` that its replay lists `unreplayable`, "no registered
evaluator"; its oracle compares only the TP partials, against the Match run the same prover served under the same switches; the
engine facts the replay measures (`engine_facts_of_record`) are recorded and read by no gate; `records.check_capture` accepts any
holding capture of the property, whatever stack it ran on.  So the switched process commits router logits that differ from the
Program's Definition on nearly every word and its Commit passes.

No engine, no torch: the route is a table, the arithmetic numpy models, the gates pure functions of the record."""
from __future__ import annotations

import numpy as np
import pytest

from verity_vllm.pipeline.tp.c2 import c2_and_pin_gates
from verity_vllm.program.registry import bf16_gemm_check as BC

#: vLLM's batch-invariant cuBLAS configuration on cc 9.0 (determinism/batch_invariant.py), the capture's `split_k_off` variant
SWITCH_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":16:8", "CUBLASLT_WORKSPACE_SIZE": "1"}
ENGINE_OF_RECORD = {**SWITCH_ENV, "VLLM_BATCH_INVARIANT": "0", "vllm_batch_invariant_mode": False,
                    "torch_allow_bf16_reduced_precision_reduction": False, "torch_preferred_blas_library": "Cublaslt",
                    "compute_capability": [9, 0]}
#: GLM-5.3's MoE gate (`GateLinear`, K 6144, N 256, f32 out) at the first M past ll_bf16, where cuBLAS takes it
ROUTER = ("router", 17, 6144, 256)
#: r20261016-213336-6c35's replay population (rows over 8 ranks)
POPULATION = {"DeepGemmFp8Linear_v1": 841728, "DsaRMSNorm_v1": 358176, "FlashInferFp8Linear_v1": 101928, "Gemm_v2": 256, "MoeSum_v1": 172200,
              "SiluMul_v2": 1556688, "TokenSelect_v1": 256}
#: its unreplayable bf16 GEMMs, each bound at a pinned route (rows over 8 ranks)
PINNED_ROUTE = {"CublasSplitKGemmF32Out_v1": 153600, "CublasSplitKGemm_v1": 48216, "LlBf16DotprodGemm_v1": 18600, "BatchedGemm_v1": 358176}
#: its match oracle on rank 0: every TP partial compared and equal
ORACLE = {"verdict": "PASS", "steps_compared": 32, "tensors": 5056, "equal": 5056, "mismatch_n": 0, "missing_n": 0, "not_in_oracle_n": 0,
          "by_member": {"partial": {"tensors": 5056, "equal": 5056}}}


def _bf16(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _gates(engine: dict) -> tuple[bool | None, str | None]:
    checks = {k: {"n": 0, "first": []} for k in ("vus_outside_query", "identities_without_rows", "identities_unreachable_by_alias", "rows_disagree")}
    pop = {"vus": sum(POPULATION.values()) // 8, "by_family": dict(POPULATION), "scope": "Q_module_v1(P)", "query_checks": checks | {"ok": True}}
    sr = {"result": True, "grade": "complete", "population": pop, "engine_of_record": dict(engine),
          "sample": {"picked": 3985, "evaluated": 3985, "equal": 3985, "mismatch_n": 0},
          "unreplayable": [{"family": f, "n": n, "why": "no registered evaluator", "ranks": list(range(8))} for f, n in PINNED_ROUTE.items()]}
    f = {"sampled_replay": {"sampled_replay": sr, "boundary_linkage": {"result": True, "linked": 32, "pairs": 32}},
         "weights_pin": {"result": True, "compared": 1700}}
    return c2_and_pin_gates({"match_oracle": dict(ORACLE)}, f, True, None, rank=0, replay_on=True, required=True, bundle="bundle_pair0_rank0")


def test_the_bound_route_is_the_pin_whatever_the_process_configures(monkeypatch):
    for k, v in SWITCH_ENV.items():
        monkeypatch.setenv(k, v)
    assert BC.default_cublas_route(*ROUTER) == ("splitk", {"SPLIT_K": 4})
    assert BC.default_cublas_route("linear", 16, 6144, 160) == ("splitk", {"SPLIT_K": 4})


def test_the_split_k_off_route_departs_from_the_bound_definition():
    """The router logits the switched process commits (the serial chain, `GemmF32Out_v1`) against the Program's split-K-4 route."""
    _, M, K, N = ROUTER
    rng = np.random.default_rng(0)
    x, w = _bf16(rng.standard_normal((M, K))), _bf16(rng.standard_normal((N, K)) * 0.02)
    served = BC.f32out_rows(x, w)
    bound = BC.cublas_splitk_rows(x, w, BC.default_cublas_route(*ROUTER)[1]["SPLIT_K"], out="f32")
    differ = served != bound
    assert differ.any(axis=1).all() and differ.mean() > 0.9


def test_the_honest_rank_passes():
    assert _gates({"VLLM_BATCH_INVARIANT": "0", "CUBLAS_WORKSPACE_CONFIG": None, "CUBLASLT_WORKSPACE_SIZE": None}) == (True, None)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="claim 1 BROKEN: the TP verdict passes a rank served under another cuBLAS configuration than the route pin's, "
                          "its pinned-route GEMMs unevaluated; fix: record the serving process's cuBLAS version and configuration in each "
                          "rank's header and refuse a Commit whose facts are not the pin's capture's, or evaluate CublasSplitKGemm*_v1 in the "
                          "sampled replay")
def test_the_tp_verdict_fails_a_rank_served_off_the_route_pin():
    ok, why = _gates(ENGINE_OF_RECORD)
    assert ok is False, why
    assert "CUBLAS" in (why or "") or "CublasSplitKGemm" in (why or "")
