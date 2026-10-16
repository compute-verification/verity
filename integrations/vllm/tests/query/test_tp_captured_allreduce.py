"""GLM-5.3's world-8 all-reduces are the captured routes (`AllReduceMultimem_v1`, the fused `AllReduceRmsNormMnnvl_v1`), not
`AllReduce_v2`.  The population did not know them as collectives, so their received `peers_<k>` inputs counted as the prompt's token
ids: the manifest of record required `input_token_ids` at every decoder layer (2,496 identities per rank) and at the model, and no
all-reduce had its rank-local partial or a peer binding.  Each is a TP-04 collective: the peers are received partials, bound to each
peer's own `part_rank<peer>`, and the fused route's rank-local partial is its `parts` operand, not the residual it also reads."""
from __future__ import annotations

from tests.query.test_module_body import W, _call
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.compose import merge_ranks
from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
from verity_vllm.query.required import Population, request_manifest
from verity_vllm.query.units import ValueRef

WORLD, N = 8, 4
EMBED_W, O_W = f"{W}.embed_tokens.weight", f"{W}.layers.0.self_attn.o_proj.weight"
NORM_W, LM_W = f"{W}.layers.0.post_attention_layernorm.weight", f"{W}.lm_head.weight"
FUSED = f"AllReduceRmsNormMnnvl_v1{{WORLD=8,N={N},SHOT=1,CLUSTER=8,EPS=1e-05,FMA=True}}"


def rank_program(rank: int) -> ProgramView:
    """Rank `rank`'s Program: the embedding shard all-reduced (multimem), `o_proj`'s partial through the fused all-reduce + residual add
    + RMSNorm of `post_attention_layernorm` (the decoder layer calls it, so it sits in the layer's body), the logits and a sampling event."""
    weights = InputView(-1, "weights", "{...}", tuple(Port(n, N * i, N, N, 16) for i, n in enumerate((EMBED_W, O_W, NORM_W, LM_W))))
    ids = InputView(-2, "ids", "Value<32>", (Port("ids", 0, 1, 1, 32),))
    peers = tuple(InputView(-3 - k, f"peers_{k}", f"Array<1,Array<{N},Value<16>>>", (Port(f"peers_{k}", 0, N, N, 16),))
                  for k in range(2 * (WORLD - 1)))
    R = ValueRef

    def parts(own: ValueRef, k: int) -> list[ValueRef]:
        received = [R(p.id, p.name) for p in peers[k * (WORLD - 1): (k + 1) * (WORLD - 1)]]
        return received[:rank] + [own] + received[rank:]

    calls = (
        _call(0, "EmbeddingShard_v1", f"{W}.embed_tokens/embedding_shard[0]/shard", [R(-2, "ids"), R(-1, EMBED_W)]),
        _call(1, "AllReduceMultimem_v1", f"{W}.embed_tokens/all_reduce/all_reduce", parts(R(0, "out"), 0)),
        _call(2, "Gemm_v1", f"{W}.layers.0.self_attn.o_proj/linear[0]", [R(1, "out"), R(-1, O_W)]),
        CallView(3, FUSED, "AllReduceRmsNormMnnvl_v1", "call", f"{W}.layers.0.fused_allreduce_rms_norm/fused_allreduce_rms_norm[0]/fused_all_reduce_rmsnorm",
                 (Port("0", 0, N, N, 16), Port("1", 0, N, N, 16)), tuple(parts(R(2, "out"), 1) + [R(1, "out"), R(-1, NORM_W)])),
        _call(4, "Gemm_v1", f"{W}.lm_head/linear[0]", [R(3, "0"), R(-1, LM_W)]),
        _call(5, "ArgmaxTokenSelect_v1", "argmax[0]", [R(4, "out")], n=1),
    )
    return ProgramView(calls, (weights, ids, *peers), (ValueRef(5, "out"),), "synthetic", "synthetic")


def _manifest(rank: int) -> dict:
    P = rank_program(rank)
    res = {"program": {"digest": f"d{rank}", "function": "f"}, "target": {"declared": {"compute_capability": [9, 0]}},
           "tp": {"world": WORLD, "rank": rank}}
    return request_manifest(P, Correspondence(P), res, request_id="r0", rank=rank, ranks=WORLD).manifest


def test_the_captured_all_reduces_are_collectives_whose_peers_are_received_partials():
    P = rank_program(3)
    pop = Population(P, Correspondence(P), tp={"world": WORLD, "rank": 3})
    assert set(pop.collectives) == {1, 3} and not pop.tp_faults
    assert pop.peer_inputs == {ValueRef(p.id, p.name) for p in P.params if p.name.startswith("peers_")}
    assert pop.partial_of == {ValueRef(0, "out"): 1}, "the embedding shard is the multimem all-reduce's partial, in its own body"
    assert pop.operand[3] == ValueRef(2, "out") and pop.carries == {3}, "the fused route's partial is o_proj's row, not its residual"


def test_no_decoder_layer_requires_token_ids_and_every_all_reduce_binds_its_peers():
    ms = [_manifest(rank) for rank in range(WORLD)]
    for rank, m in enumerate(ms):
        assert m["complete"], m["unmodelled"]
        ids = [(x["op_path"], x["rows"]) for x in m["identities"] if x["family"] == "input_token_ids"]
        assert ids == [(f"{W}.embed_tokens", 1)], ids
        partials = {(x["op_path"], x["output_member"], x["element_range"][1]) for x in m["identities"] if x["family"] == "tp_rank_partials"}
        assert partials == {(f"{W}.embed_tokens", f"part_rank{rank}", N), (f"{W}.layers.0", f"part_rank{rank}", N)}, partials
    g = merge_ranks(ms)
    b = g["tp_peer_binding"]
    assert g["complete"] and b["n_unbound"] == 0 and b["collective_outputs"] == 3 * WORLD, b
    assert b["bound"] == b["collective_outputs"] * (WORLD - 1), b
