"""`predict.step_needs`: what `dsa` calls for one engine step, counted without building the Program, each refusal in its place."""
from __future__ import annotations

from collections import Counter

import pytest

from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.step_needs import StepNeeds, workload_phi
from tests.program.predict.test_dsa_long_rows import phi  # noqa: F401  (the fixture)

CLASS_G_ROW = "glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy__eager"


@pytest.fixture(scope="module")
def S():
    return StepNeeds(workload_phi(CLASS_G_ROW, "GLM53"))


def _gaps(need):
    return {(n.route, n.id.split("{")[0], n.gap.split(":")[0]) for n in need if n.gap}


def test_class_g_steps_need_no_refusal(S):
    """Eight greedy decodes at short contexts (class G's step) and one request's 256-row chunk need only Definitions."""
    assert not _gaps(S.step(8, 8, [], [100 + i for i in range(8)], 8, {"greedy": 8}, True))
    assert not _gaps(S.step(256, 256, [(256, 1024)], [], 1, {"greedy": 1}, False))


def test_production_sizes_refuse_by_name(S):
    """Past class G's limits each refusal `dsa` would raise is its own Need, and the step's other calls are still counted: a row
    past INDEX_TOPK keys, under class G's engine environment, which lacks the canonical top-k, refuses its top-k in place of its
    order and its check but keeps its logits and every layer's routed rows, the 8,192-token launch has no captured
    all-reduce (vLLM's NCCL route) or indexer cuBLAS route, a 2,000-row prefill takes the dense-MHA backend and one past
    INDEX_TOPK keys the sparse-MQA one, and the unseeded and top-k requests have no select (a seeded request in the batch keeps
    FlashInfer's sampler off)."""
    need = S.step(8192, 8192, [(2000, 0), (4000, 16000)], [5000] + [100] * 191, 194,
                  {"greedy": 180, "seeded": 4, "random": 10, "top_k": 2}, True)
    assert _gaps(need) == {
        ("dsa-topk", "kernel-topk-order", "kernel top-k order"), ("tp-allreduce", "allreduce-pynccl_comm", "no capture"),
        ("cublas-bf16-gemm", "cublas-linear", "no capture"), ("router-gemm", "cublas-router", "no capture"),
        ("fa3-prefill-attention", "prefill-call", "dense-MHA prefill"), ("dsa-sparse-prefill", "prefill-call", "sparse-MQA prefill"),
        ("token-select", "select-random", "unseeded sampling"), ("token-select", "select-top_k", "top-k sampling")}
    ids = Counter({n.id: c for n, c in need.items()})
    assert ids["DsaMqaLogits_v3{N=5000,H=32,D=128}"] == 21 and ids["kernel-topk-order{TOPK=2048,KEYS=5000}"] == 2 * 21
    assert ids["DsaWaksmanRows_v1{N=5000,K=2048,W=576,E=16}"] == 78
    assert ids["prefill-call{ROWS=2000,KEYS=2000}"] == 78 * 2000
    assert ids["MoeRouterSigmoidBias_v1{E=256,TOPK=8,SCALE=2.5,KERNEL='warp'}"] == 75 * 8192


def test_flashinfer_samples_only_unseeded_filtered_steps(S):
    """vLLM's FlashInfer sampler takes a step's unseeded requests only when one filters by top-k or top-p and none is seeded; each
    unseeded request's select is refused once, and a seeded one's is the stochastic select's Definition."""
    def refused(sampling, on=True):
        return {n.id.split("{")[0]: c for n, c in S.step(8, 8, [], [100] * 8, 8, sampling, on).items() if n.gap}

    assert refused({"greedy": 2, "random": 6, "top_p": 6}) == {"flashinfer-sampler": 6}
    assert refused({"greedy": 2, "random": 6, "top_p": 6}, on=False) == {"select-random": 6}
    assert refused({"seeded": 1, "random": 5, "top_p": 6}) == {"select-random": 5}
    assert refused({"random": 8}) == {"select-random": 8}
    assert refused({"seeded": 8, "top_p": 8}) == {}


def _decoders_calls(phi, M: int, pos0: int) -> Counter[str]:
    """Each Definition id `dsa`'s own Program for one engine step of M rows at pos0 (`build_program(..., "step")`) calls, Consts
    left out, with how often."""
    from verity_vllm.program.predict import dsa
    from verity_vllm.program.predict.build import build_program
    seen: Counter[str] = Counter()
    orig = dsa._Model.call

    def call(self, fn, *refs):
        seen[fn.id] += 1
        return orig(self, fn, *refs)

    dsa._Model.call = call
    try:
        build_program(phi, "step", M=M, pos0=pos0, with_desc=False)
    finally:
        dsa._Model.call = orig
    return Counter({k: v for k, v in seen.items() if not k.startswith("Const")})


def test_rows_past_index_topk_count_the_decoders_calls(phi):
    """GLM-5.3 cut to 4 layers and an INDEX_TOPK of 4 (`test_dsa_long_rows`), under the canonical top-k: a step of 6 rows, whose
    last two read 5 and 6 keys, calls DsaWaksmanOrder and DsaTopKLongRow at each full indexer layer and DsaWaksmanRows at every
    layer exactly as often as `step` counts. Without the canonical top-k their top-k is one refusal in place of the two."""
    need = StepNeeds(phi).step(6, 6, [(6, 0)], [], 1, {"greedy": 1}, False)
    assert not _gaps(need)
    assert Counter({n.id: c for n, c in need.items()}) == _decoders_calls(phi, 6, 0)
    ids = Counter({n.id: c for n, c in need.items()})
    assert ids["DsaTopKLongRow_v1{N=6,TOPK=4}"] == 3 and ids["DsaWaksmanRows_v1{N=6,K=4,W=576,E=16}"] == 4
    plain = Phi.from_json({**phi.to_json(), "engine": {**phi.engine, "env": {"VLLM_BATCH_INVARIANT": "0"}}})
    refused = {n.id: c for n, c in StepNeeds(plain).step(6, 6, [(6, 0)], [], 1, {"greedy": 1}, False).items() if n.gap}
    assert refused == {"kernel-topk-order{TOPK=4,KEYS=5}": 6, "kernel-topk-order{TOPK=4,KEYS=6}": 6}


@pytest.mark.slow
@pytest.mark.parametrize("M", [8, 64])
def test_a_step_counts_the_decoders_calls(M, S):
    """`dsa`'s own Program for one engine step of M rows (`build_program(..., "step")`: one call of M rows with no history, the
    last row sampled greedily) calls each Definition exactly as often as `step` counts for that step: the FlashInfer linears and
    the fused all-reduce + norm at 8, DeepGEMM, the multimem all-reduce and the fused norm at 64. This is what holds `step_needs`
    to `dsa` at the served checkpoint; no cheaper test compares the two. About 20 s, nearly all of it the decoder's build."""
    need = S.step(M, M, [(M, 0)], [], 1, {"greedy": 1}, False)
    assert not _gaps(need)
    assert Counter({n.id: c for n, c in need.items()}) == _decoders_calls(S.phi, M, 8)
