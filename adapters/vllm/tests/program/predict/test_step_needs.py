"""`predict.step_needs`: what `dsa` calls for one engine step, counted without building the Program, each refusal in its place."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from verity_vllm.config import row_file
from verity_vllm.program.predict import step_needs
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.step_needs import StepNeeds, workload_phi

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


def test_a_row_that_pins_no_revision_has_no_snapshot_dir_to_write(tmp_path, monkeypatch):
    """`workload_phi` writes the pinned config.json where `score.phi_of` looks for it, under the row's revision: a row whose
    engine block pins none names no dir, which is `os.path.join`'s TypeError."""
    w = json.loads(row_file(CLASS_G_ROW).read_text())
    (bare := tmp_path / "row.json").write_text(json.dumps({**w, "engine": {k: v for k, v in w["engine"].items() if k != "revision"}}))
    monkeypatch.setattr(step_needs, "row_file", lambda row: bare)
    monkeypatch.setattr(step_needs.tempfile, "mkdtemp", lambda prefix: str(tmp_path))
    with pytest.raises(TypeError, match="NoneType"):
        workload_phi(CLASS_G_ROW, "GLM53")


def test_a_device_named_by_no_str_names_no_tuned_file(S):
    """The tuned fused-MoE file a step's expert GEMMs take their tile from is named by φ's `device_name`: one that is no str
    has no name to put in it, an AttributeError, and is no missing file's refusal."""
    doc = S.phi.to_json()
    numbered = StepNeeds(Phi.from_json({**doc, "target": {**doc["target"], "device_name": 200}}))
    with pytest.raises(AttributeError, match="'int' object has no attribute 'replace'"):
        numbered.step(8, 8, [], [100 + i for i in range(8)], 8, {"greedy": 8}, True)


def test_production_sizes_refuse_by_name(S):
    """Past class G's limits each refusal `dsa` would raise is its own Need, and the step's other calls are still counted: a row
    past INDEX_TOPK keys refuses its top-k (the radix selection) but keeps its logits, the 8,192-token launch has no captured
    all-reduce (vLLM's NCCL route), a 2,000-row prefill takes the dense-MHA backend and one past INDEX_TOPK keys the sparse-MQA
    one, and the unseeded and top-k requests have no select (a seeded request in the batch keeps FlashInfer's sampler off)."""
    need = S.step(8192, 8192, [(2000, 0), (4000, 16000)], [5000] + [100] * 191, 194,
                  {"greedy": 180, "seeded": 4, "random": 10, "top_k": 2}, True)
    assert _gaps(need) == {
        ("dsa-topk", "radix-topk", "radix top-k"), ("tp-allreduce", "allreduce-pynccl_comm", "no capture"),
        ("fa3-prefill-attention", "prefill-call", "dense-MHA prefill"), ("dsa-sparse-prefill", "prefill-call", "sparse-MQA prefill"),
        ("token-select", "select-random", "unseeded sampling"), ("token-select", "select-top_k", "top-k sampling")}
    ids = Counter({n.id: c for n, c in need.items()})
    assert ids["DsaMqaLogits_v3{N=5000,H=32,D=128}"] == 21 and ids["radix-topk{TOPK=2048,KEYS=5000}"] == 21
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


@pytest.mark.slow
@pytest.mark.parametrize("M", [8, 64])
def test_a_step_counts_the_decoders_calls(M, S):
    """`dsa`'s own Program for one engine step of M rows (`build_program(..., "step")`: one call of M rows with no history, the
    last row sampled greedily) calls each Definition exactly as often as `step` counts for that step: the FlashInfer linears and
    the fused all-reduce + norm at 8, DeepGEMM, the multimem all-reduce and the fused norm at 64. This is what holds `step_needs`
    to `dsa`; no cheaper test compares the two. About 20 s, nearly all of it the decoder's build."""
    from verity_vllm.program.predict import dsa
    from verity_vllm.program.predict.build import build_program
    seen: Counter[str] = Counter()
    orig = dsa._Model.call

    def call(self, fn, *refs):
        seen[fn.id] += 1
        return orig(self, fn, *refs)

    dsa._Model.call = call
    try:
        build_program(S.phi, "step", M=M, pos0=8)
    finally:
        dsa._Model.call = orig
    need = S.step(M, M, [(M, 0)], [], 1, {"greedy": 1}, False)
    assert not _gaps(need)
    assert Counter({n.id: c for n, c in need.items()}) == Counter({k: v for k, v in seen.items() if not k.startswith("Const")})
