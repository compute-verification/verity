"""A served cuBLAS route that departs from the pin must not pass a TP Commit, digest equality intact.

A switched workload's AOT Program equals its traced Build because both read the same pin: the Build's binding
(`program/frontend/rules/vllm_bindings/bf16_gemm.py`), the fold (`observe/fold/patterns/cublas.py`) and the predictor
(`predict/dsa.py`) each take the split count of a bf16 GEMM from `bf16_gemm_check.default_cublas_route`,
a table of what cuBLAS chose on the capture box.  The serving process's cuBLAS decides its own route, and two environment variables
and two torch flags it inherits from whoever starts it move that route.  The capture of record (`glm53_bf16_gemm_capture`,
on an H200) measures it: GLM-5.3's MoE gate at M 17 runs `nvjet_tss_64x8_64x16_2x4_h_bz_splitK_TNT`
plus `cublasLt::splitKreduce_kernel` (split 4) at vLLM's default, and `nvjet_tss_64x8_64x16_2x4_h_bz_TNT` (no split, word-exact against
`GemmF32Out_v1`) under CUBLAS_WORKSPACE_CONFIG=:16:8, CUBLASLT_WORKSPACE_SIZE=1 and the reduced-precision flags off.

GLM-5.3 B1 TP8's Commit of record commits 153,600 rows of
`CublasSplitKGemmF32Out_v1` (the prefill router) and 48,216 of `CublasSplitKGemm_v1` that its replay lists `unreplayable`, "no registered
evaluator"; its oracle compares only the TP partials, against the Match run the same prover served under the same switches; the
engine facts the replay measures (`engine_facts_of_record`) are recorded and read by no gate; `records.check_capture` accepts any
holding capture of the property, whatever stack it ran on.  So a switched process would commit router logits that differ from the
Program's Definition on nearly every word and pass.

Each rank records what decides its cuBLAS routes (`engine_of_record.cublas`: the device, torch, the mapped cuBLAS, the
reduced-precision flags, the preferred backend, the workspace variables), and the TP verdict refuses a rank whose Program binds a
family only the pin binds unless those are the pin's scan's (`check.cublas_route_pin`, `bf16_gemm_check.ROUTE_CONDITIONS`), an
unrecorded fact by name too; and the replay evaluates those families (`kernels.cublas_gemm_rows`).

No engine, no torch: the route is a table, the arithmetic numpy models, the gates pure functions of the record."""
from __future__ import annotations

import numpy as np
import pytest

from verity_vllm.check.cublas_route_pin import route_pin_gate
from verity_vllm.pipeline.tp.c2 import c2_and_pin_gates, replay_gates
from verity.ml.definitions.registry import bf16_gemm_check as BC

#: vLLM's batch-invariant cuBLAS configuration on cc 9.0 (determinism/batch_invariant.py), the capture's `split_k_off` variant
SWITCH_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":16:8", "CUBLASLT_WORKSPACE_SIZE": "1"}
ENGINE_OF_RECORD = {**SWITCH_ENV, "VLLM_BATCH_INVARIANT": "0", "vllm_batch_invariant_mode": False,
                    "torch_allow_bf16_reduced_precision_reduction": False, "torch_preferred_blas_library": "Cublaslt",
                    "compute_capability": [9, 0]}
#: GLM-5.3's MoE gate (`GateLinear`, K 6144, N 256, f32 out) at the first M past ll_bf16, where cuBLAS takes it
ROUTER = ("router", 17, 6144, 256)
#: that Commit's replay population (rows over 8 ranks)
POPULATION = {"DeepGemmFp8Linear_v1": 841728, "DsaRMSNorm_v1": 358176, "FlashInferFp8Linear_v1": 101928, "Gemm_v2": 256, "MoeSum_v1": 172200,
              "SiluMul_v2": 1556688, "TokenSelect_v1": 256}
#: its bf16 GEMMs bound at a pinned route (rows over 8 ranks): `unreplayable` on that record, evaluated now (`kernels.cublas_gemm_rows`,
#: `glm53_rows`, `batched_gemm_rows`), so the per-rank no-evaluator rule passes a rank whose replay counts them
PINNED_ROUTE = {"CublasSplitKGemmF32Out_v1": 153600, "CublasSplitKGemm_v1": 48216, "LlBf16DotprodGemm_v1": 18600, "BatchedGemm_v1": 358176}
#: its match oracle on rank 0: every TP partial compared and equal
ORACLE = {"verdict": "PASS", "steps_compared": 32, "tensors": 5056, "equal": 5056, "mismatch_n": 0, "missing_n": 0, "not_in_oracle_n": 0,
          "by_member": {"partial": {"tensors": 5056, "equal": 5056}}}


def _bf16(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _gates(engine: dict) -> tuple[bool | None, str | None]:
    checks = {k: {"n": 0, "first": []} for k in ("vus_outside_query", "identities_without_rows", "identities_unreachable_by_alias", "rows_disagree")}
    by_family = POPULATION | PINNED_ROUTE
    pop = {"vus": sum(by_family.values()) // 8, "by_family": by_family, "scope": "Q_module_v1(P)", "query_checks": checks | {"ok": True}}
    sr = {"result": True, "grade": "complete", "population": pop, "engine_of_record": dict(engine),
          "sample": {"picked": 3985, "evaluated": 3985, "equal": 3985, "mismatch_n": 0}, "unreplayable": []}
    f = {"sampled_replay": {"sampled_replay": sr, "boundary_linkage": {"result": True, "linked": 32, "pairs": 32}},
         "weights_pin": {"result": True, "compared": 1700}}
    return c2_and_pin_gates({"match_oracle": dict(ORACLE)}, f, True, None, rank=0, replay_on=True, required=True, bundle="bundle_pair0_rank0",
                            xrank_families=None)


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


#: the facts that Commit's ranks recorded: the workspace variables, no cuBLAS facts
FACTS_OF_RECORD = {"VLLM_BATCH_INVARIANT": "0", "CUBLAS_WORKSPACE_CONFIG": None, "CUBLASLT_WORKSPACE_SIZE": None}
#: an honest rank's: the pin's scan's conditions, as `engine_facts_of_record` measures them
HONEST = {**FACTS_OF_RECORD, "cublas": {**BC.ROUTE_CONDITIONS, "cublaslt_version": 120901, "cublas_library": "/venv/nvidia/cublas/lib/libcublas.so.12"}}
#: the switched process's, measured the same way (the capture's `switch`, read back)
SWITCHED = {**HONEST, **SWITCH_ENV, "cublas": {**HONEST["cublas"], "env": dict(SWITCH_ENV), "preferred_blas_library": "_BlasBackend.Cublaslt",
                                               "allow_fp16_reduced_precision_reduction": "(False, False)",
                                               "allow_bf16_reduced_precision_reduction": "(False, False)"}}


def test_the_honest_rank_passes():
    assert _gates(HONEST) == (True, None)


def test_the_tp_verdict_fails_a_rank_served_off_the_route_pin():
    ok, why = _gates(ENGINE_OF_RECORD)
    assert ok is False, why
    assert "CUBLAS" in (why or "") or "CublasSplitKGemm" in (why or "")


def test_the_switched_rank_is_refused_naming_every_departure():
    ok, why = _gates(SWITCHED)
    assert ok is False
    assert why.startswith("cuBLAS route pin (rank 0): CublasSplitKGemmF32Out_v1 x153600, CublasSplitKGemm_v1 x48216 bound at the routes of "
                          f"{BC.CUBLAS_ROUTES['run_record']}, served otherwise: ")
    assert why.split("served otherwise: ")[1].split("; ") == [
        "allow_fp16_reduced_precision_reduction '(False, False)' (pin: 'True')",
        "allow_bf16_reduced_precision_reduction '(False, False)' (pin: 'True')",
        "preferred_blas_library '_BlasBackend.Cublaslt' (pin: '_BlasBackend.Cublas')",
        "CUBLAS_WORKSPACE_CONFIG ':16:8' (pin: None)", "CUBLASLT_WORKSPACE_SIZE '1' (pin: None)"]


@pytest.mark.parametrize("fact,served", [("cublas_version", 130101), ("multiprocessors", 114), ("device", "NVIDIA H100 80GB HBM3"),
                                         ("torch", "2.12.0+cu128"), ("compute_capability", [10, 0])])
def test_another_library_device_or_torch_is_a_departure(fact, served):
    ok, why = _gates({**HONEST, "cublas": {**HONEST["cublas"], fact: served}})
    assert ok is False and why.endswith(f"served otherwise: {fact} {served!r} (pin: {BC.ROUTE_CONDITIONS[fact]!r})")


def test_a_fact_the_rank_could_not_read_or_did_not_record_is_refused_by_name():
    unread = {**HONEST, "cublas": {**HONEST["cublas"], "cublas_version": "unavailable: LookupError: 0 libcublas.so* mapped []"}}
    assert _gates(unread)[1].endswith("served otherwise: cublas_version 'unavailable: LookupError: 0 libcublas.so* mapped []' (pin: 120901)")
    no_env = {**HONEST, "cublas": {k: v for k, v in HONEST["cublas"].items() if k != "env"}}
    assert _gates(no_env)[1].endswith("CUBLAS_WORKSPACE_CONFIG not recorded (pin: None); CUBLASLT_WORKSPACE_SIZE not recorded (pin: None)")


def test_the_commit_of_record_is_now_refused_for_its_unrecorded_cublas_facts():
    """GLM-5.3 B1 TP8's Commit of record recorded the workspace variables alone: under the gate it no longer passes."""
    ok, why = _gates(FACTS_OF_RECORD)
    assert ok is False and "no cuBLAS facts recorded (engine_of_record.cublas" in why


def test_a_rank_that_binds_no_pinned_route_family_is_not_gated():
    sr = {"population": {"by_family": {"Gemm_v2": 256, "SiluMul_v2": 4}}, "unreplayable": [{"family": "BatchedGemm_v1", "n": 3}],
          "engine_of_record": FACTS_OF_RECORD}
    assert route_pin_gate(sr, rank=1) == (None, None)
    block, why = route_pin_gate(dict(sr, population={"by_family": {"GemmF32Out_v1": 2}}), rank=1)
    assert block["families"] == {"GemmF32Out_v1": 2} and block["ok"] is False and why.startswith("cuBLAS route pin (rank 1): GemmF32Out_v1 x2")


def test_the_gate_reads_the_deferred_replays_facts_too():
    """`commit_replay.fold` decides a deferred rank by `replay_gates` over its CPU replay, whose `engine_of_record` is the bundle's
    (the Commit engine's, measured in its process)."""
    sr = {"result": True, "grade": "complete", "engine_of_record": SWITCHED, "population": {"by_family": {"CublasSplitKGemmF32Out_v1": 9}}}
    cblock: dict = {}
    ok, why = replay_gates(cblock, sr, {"result": True}, True, None, rank=3, replay_on=True, required=False, xrank_families=None)
    assert ok is False and why.startswith("cuBLAS route pin (rank 3): CublasSplitKGemmF32Out_v1 x9")
    assert cblock["cublas_route_pin"]["departures"] and cblock["cublas_route_pin"]["pin"] == BC.CUBLAS_ROUTES["run_record"]
