"""The policy on a rank Program shaped like DeepSeek-V4-Pro's (`query.required.Population`):

* a row-parallel block-FP8 linear that reduces in its own body (`wo_b`: `DeepGemmFp8Linear_v1` then the all-reduce, both
  under the linear's module) has its partial and every other port of the producer required, the partial spelt `part_rank<r>` and the
  others by their ports (`required.member_name`);
* UNRETURNED: a Call no Call reads, the Program does not return and no policy requires (the indexer's projection on its short path)
  is dead, so the word check skips it, while a dead Call the Build recorded a return slot for stays required."""
from __future__ import annotations

from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
from verity_vllm.query.required import Q_WORD_ID, Population
from verity_vllm.query.units import ValueRef as R

N, K = 4, 4
WO_B = "model.layers.0.attn.wo_b"


def _program() -> tuple[ProgramView, dict[str, int]]:
    calls: list[CallView] = []

    def call(definition: str, name: str, reads, ports=(("out", N, 16),)) -> int:
        calls.append(CallView(len(calls), definition, definition.split("{")[0], "call", name,
                              tuple(Port(p, 0, n, n, w) for p, n, w in ports), tuple(reads)))
        return len(calls) - 1

    w = lambda path: R(-1, path)
    x = R(-2, "x")
    ids = {"linear": call(f"DeepGemmFp8Linear_v1{{K={K},N={N}}}", f"{WO_B}/dynamic_flashinfer_deepgemm_blockscale_gemm[0]",
                          [x, w(f"{WO_B}.weight"), w(f"{WO_B}.weight_scale_inv")], (("0", K, 8), ("1", 1, 32), ("2", N, 16)))}
    ids["all_reduce"] = call("AllReduceMultimem_v1", f"{WO_B}/all_reduce_1[0]", [R(ids["linear"], "2"), R(-3, "peers_0")])
    ids["dead"] = call(f"CublasSplitKGemm_v1{{K={K},N={N}}}", "model.layers.0.attn.indexer.weights_proj/linear[0]",
                       [x, w("model.layers.0.attn.indexer.weights_proj.weight")])
    ids["recorded"] = call("RMSNorm_v1", "model.layers.0.attn/state/kv[0]", [x])
    ids["logits"] = call(f"Gemm_v2{{K={K},N={N}}}", "logits_processor/linear[0]", [R(ids["all_reduce"], "out"), w("lm_head.weight")])
    ids["argmax"] = call("ArgmaxTokenSelect_v1", "argmax[0]", [R(ids["logits"], "out")], (("out", 1, 32),))
    weights = ("weights", "{...}", tuple(Port(p, N * i, N * (i + 1), N, 16) for i, p in enumerate(
        [f"{WO_B}.weight", f"{WO_B}.weight_scale_inv", "model.layers.0.attn.indexer.weights_proj.weight", "lm_head.weight"])))
    inputs = (InputView(-1, *weights), InputView(-2, "x", f"Array<{K},Value<16>>", (Port("x", 0, K, K, 16),)),
              InputView(-3, "peers_0", f"Array<1,Array<{N},Value<16>>>", (Port("peers_0", 0, N, N, 16),)))
    return ProgramView(tuple(calls), inputs, (R(ids["argmax"], "out"),), "synthetic", "synthetic"), ids


def _population() -> tuple[Population, dict[str, int]]:
    P, ids = _program()
    return Population(P, Correspondence(P), tp={"world": 2, "rank": 0}, query=Q_WORD_ID), ids


def test_a_partial_s_producer_has_every_port_required():
    pop, ids = _population()
    assert {R(ids["linear"], p) for p in ("0", "1", "2")} <= pop.required.values
    assert [m.member for m in pop.members_of(ids["linear"])] == ["0", "1", "part_rank0"]


def test_a_dead_call_is_unreturned_unless_a_policy_requires_it():
    pop, ids = _population()
    assert ids["dead"] in pop.unreturned and R(ids["dead"], "out") not in pop.required.values
    assert ids["recorded"] not in pop.unreturned and R(ids["recorded"], "out") in pop.required.values
    assert not pop.unreturned & {ids["linear"], ids["all_reduce"], ids["logits"], ids["argmax"]}
