"""GLM-5.3's world-8 all-reduces are the captured routes (`AllReduceMultimem_v1`, the fused `AllReduceRmsNormMnnvl_v1`), not
`AllReduce_v2`.  The population did not know them as collectives, so their received `peers_<k>` inputs counted as the prompt's token
ids: the manifest of record required `input_token_ids` at every decoder layer (2,496 identities per rank) and at the model, and no
all-reduce had its rank-local partial or a peer binding.  Each is a rank-policy collective: the peers are received partials, bound to each
peer's own `part_rank<peer>`, and the fused route's rank-local partial is its `parts` operand, not the residual it also reads.
At M > 1 tokens the routes are flat (`AllReduceRule`: one Call over the M rows, each rank's part the M row Values it reshapes), and the
partial source commits the whole [M, N] input once per call as `<site>/part_rank<r>`: the rank-local partial is one identity over the M
rows, as wide as the flat output, so each of the W-1 peer parts binds to it."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import pytest

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


def rank_program(rank: int, M: int = 1) -> ProgramView:
    """Rank `rank`'s Program at `M` tokens: the embedding shard all-reduced (multimem), `o_proj`'s partial through the fused all-reduce +
    residual add + RMSNorm of `post_attention_layernorm` (the decoder layer calls it, so it sits in the layer's body), the logits and a
    sampling event.  Each all-reduce is flat: one Call reads the M own row Values and the peers' [M, N] parts."""
    weights = InputView(-1, "weights", "{...}", tuple(Port(n, N * i, N, N, 16) for i, n in enumerate((EMBED_W, O_W, NORM_W, LM_W))))
    ids = InputView(-2, "ids", "Value<32>", (Port("ids", 0, 1, 1, 32),))
    peers = tuple(InputView(-3 - k, f"peers_{k}", f"Array<1,Array<{M * N},Value<16>>>", (Port(f"peers_{k}", 0, M * N, M * N, 16),))
                  for k in range(2 * (WORLD - 1)))
    R = ValueRef
    emb, o = range(M), range(M + 1, 2 * M + 1)
    fused, lm = 2 * M + 1, 2 * M + 2

    def parts(own: list[ValueRef], k: int) -> list[ValueRef]:
        received = [R(p.id, p.name) for p in peers[k * (WORLD - 1): (k + 1) * (WORLD - 1)]]
        return received[:rank] + own + received[rank:]

    calls = (
        *(_call(i, "EmbeddingShard_v1", f"{W}.embed_tokens/embedding_shard[{i}]/shard", [R(-2, "ids"), R(-1, EMBED_W)]) for i in emb),
        _call(M, "AllReduceMultimem_v1", f"{W}.embed_tokens/all_reduce/all_reduce", parts([R(i, "out") for i in emb], 0), n=M * N),
        *(_call(c, "Gemm_v1", f"{W}.layers.0.self_attn.o_proj/linear[{i}]", [R(M, "out"), R(-1, O_W)]) for i, c in enumerate(o)),
        CallView(fused, FUSED, "AllReduceRmsNormMnnvl_v1", "call",
                 f"{W}.layers.0.fused_allreduce_rms_norm/fused_allreduce_rms_norm[0]/fused_all_reduce_rmsnorm",
                 (Port("0", 0, M * N, M * N, 16), Port("1", 0, M * N, M * N, 16)),
                 tuple(parts([R(c, "out") for c in o], 1) + [R(M, "out"), R(-1, NORM_W)])),
        _call(lm, "Gemm_v1", f"{W}.lm_head/linear[0]", [R(fused, "0"), R(-1, LM_W)]),
        _call(lm + 1, "ArgmaxTokenSelect_v1", "argmax[0]", [R(lm, "out")], n=1),
    )
    return ProgramView(calls, (weights, ids, *peers), (ValueRef(lm + 1, "out"),), "synthetic", "synthetic")


def _manifest(rank: int, M: int = 1) -> dict:
    P = rank_program(rank, M)
    res = {"program": {"digest": f"d{rank}", "function": "f"}, "target": {"declared": {"compute_capability": [9, 0]}},
           "tp": {"world": WORLD, "rank": rank}}
    return request_manifest(P, Correspondence(P), res, request_id="r0", rank=rank, ranks=WORLD).manifest


@pytest.mark.parametrize("M", [1, 4])
def test_the_captured_all_reduces_are_collectives_whose_peers_are_received_partials(M):
    P = rank_program(3, M)
    pop = Population(P, Correspondence(P), tp={"world": WORLD, "rank": 3})
    assert set(pop.collectives) == {M, 2 * M + 1} and not pop.tp_faults, pop.tp_faults
    assert pop.peer_inputs == {ValueRef(p.id, p.name) for p in P.params if p.name.startswith("peers_")}
    assert pop.partial_of == {ValueRef(i, "out"): M for i in range(M)}, "the embedding rows are the multimem all-reduce's partial, in its own body"
    assert pop.operand[2 * M + 1] == ValueRef(M + 1, "out") and pop.carries == {2 * M + 1}, "the fused route's partial is o_proj's rows, not its residual"


@pytest.mark.parametrize("M", [1, 4])
def test_no_decoder_layer_requires_token_ids_and_every_all_reduce_binds_its_peers(M):
    ms = [_manifest(rank, M) for rank in range(WORLD)]
    for rank, m in enumerate(ms):
        assert m["complete"], m["unmodelled"]
        ids = [(x["op_path"], x["rows"]) for x in m["identities"] if x["family"] == "input_token_ids"]
        assert ids == [(f"{W}.embed_tokens", M)], ids
        partials = sorted((x["op_path"], x["output_member"], x["element_range"]) for x in m["identities"] if x["family"] == "tp_rank_partials")
        assert partials == [(f"{W}.embed_tokens", f"part_rank{rank}", [0, M * N]),
                            (f"{W}.layers.0.post_attention_layernorm", f"part_rank{rank}", [0, M * N])], partials
    g = merge_ranks(ms)
    b = g["tp_peer_binding"]
    assert g["complete"] and b["n_unbound"] == 0 and b["collective_outputs"] == 3 * WORLD, b
    assert b["bound"] == b["collective_outputs"] * (WORLD - 1), b
