"""A MoE block's Values inside its body are committed as its `moe_block_stream` planes and are no module member, under Q_word as under
Q_module_body.  S1 kept them out of `interior` (so no `call_boundaries`) but left them required, so each became a member of the body's `out`:
the router's, the expert Gemms', the act and down rows' Values all claimed `<experts>/out`, the block's real output (MoeSum's, or on a rank
Program its AllReduce's) was dropped as a "second producer", and the manifest was incomplete: on a TP2 qwen3-30b-a3b workload every
rank-0 partial of the dropped all-reduce outputs was left without its peer's collective output."""
from __future__ import annotations

import json

import pytest

from tests.query.test_module_body import W, _call
from verity_vllm.query.units import ValueRef
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.compose import merge_ranks
from verity_vllm.query.program_view import InputView, Port, ProgramView
from verity_vllm.query.required import QUERIES, request_manifest

EMBED_W, W13, W2, LM_W = f"{W}.embed.weight", f"{W}.layers.0.mlp.experts.w13_weight", f"{W}.layers.0.mlp.experts.w2_weight", f"{W}.lm_head.weight"
EXPERTS = f"{W}.layers.0.mlp.experts"


def tp_moe_program() -> ProgramView:
    """A world-2 rank Program at toy size with a MoE block in one body (`...mlp.experts`): router, expert Gemm, act, down Gemm and the sum,
    then the rank's partial all-reduced with the received peer partial in the same body."""
    weights = InputView(-1, "weights", "{...}", tuple(Port(n, 8 * i, 8, 8, 16) for i, n in enumerate((EMBED_W, W13, W2, LM_W))))
    ids = InputView(-2, "ids", "Value<32>", (Port("ids", 0, 1, 1, 32),))
    peers = InputView(-3, "peers_0", "Array<1,Array<4,Value<16>>>", (Port("peers_0", 0, 4, 4, 16),))
    R = ValueRef
    calls = (
        _call(0, "Gather_v1", f"{W}.embed/embedding/out[0]", [R(-2, "ids"), R(-1, EMBED_W)]),
        _call(1, "MoeRouterTopK_v1", f"{EXPERTS}/topk_softmax[0]", [R(0, "out")]),
        _call(2, "MoeExpertGemm_v1", f"{EXPERTS}/expert_gemm[0]", [R(0, "out"), R(1, "out"), R(-1, W13)], n=8),
        _call(3, "SiluMul_v1", f"{EXPERTS}/silu_and_mul[0]", [R(2, "out")]),
        _call(4, "MoeExpertGemmW_v1", f"{EXPERTS}/expert_gemm_w[0]", [R(3, "out"), R(1, "out"), R(-1, W2)]),
        _call(5, "MoeSum_v1", f"{EXPERTS}/moe_sum[0]", [R(4, "out")]),
        _call(6, "AllReduce2_v1", f"{EXPERTS}/all_reduce[0]/all_reduce", [R(5, "out"), R(-3, "peers_0")]),
        _call(7, "Add_v1", f"{W}/add[0]", [R(0, "out"), R(6, "out")]),
        _call(8, "Gemm_v1", f"{W}.lm_head/linear[0]", [R(7, "out"), R(-1, LM_W)]),
        _call(9, "ArgmaxTokenSelect_v1", "argmax[0]", [R(8, "out")], n=1),
    )
    return ProgramView(calls, (weights, ids, peers), (ValueRef(9, "out"),), "synthetic", "synthetic")


def _manifest(rank: int, query: str) -> dict:
    P = tp_moe_program()
    res = {"program": {"digest": f"d{rank}", "function": "f"}, "target": {"declared": {"compute_capability": [8, 9]}}, "tp": {"world": 2, "rank": rank}}
    return request_manifest(P, Correspondence(P), res, request_id="r0", rank=rank, ranks=2, query=query).manifest


@pytest.mark.parametrize("query", QUERIES)
def test_a_rank_programs_moe_block_names_its_all_reduce_and_its_partial_and_the_merge_binds_every_peer(query):
    ms = [_manifest(rank, query) for rank in (0, 1)]
    for rank, m in enumerate(ms):
        assert m["complete"] and not any("two producers" in k for k in m["unmodelled"]), m["unmodelled"]
        rows = {(x["family"], x["output_member"]): x for x in m["identities"] if x["op_path"] == EXPERTS}
        assert set(rows) == {("instance_outputs", "out"), ("tp_rank_partials", f"part_rank{rank}")}, sorted(rows)
        assert rows[("instance_outputs", "out")]["spec"] == "AllReduce2_v1" and rows[("tp_rank_partials", f"part_rank{rank}")]["spec"] == "MoeSum_v1"
        assert not [x for x in m["identities"] if x["family"] == "call_boundaries"], "the block's rows are committed as its planes"
    g = merge_ranks(ms)
    assert g["complete"] and g["tp_peer_binding"]["n_unbound"] == 0 and g["tp_peer_binding"]["bound"] == 2, g["tp_peer_binding"]


def test_an_incomplete_manifest_is_a_failed_manifest_step(tmp_path, monkeypatch, capsys):
    """The Commit refuses an incomplete manifest, so `manifest build` / `build-global` exit 4 on one (the file is still written)."""
    from verity_vllm.pipeline import cli
    from verity_vllm.pipeline import manifest as M
    bad = {"complete": False, "unmodelled": {"AllReduce2_v1 under m.experts (out: two producers MoeExpertGemm_v1 8 / AllReduce2_v1 4)": 1},
           "tp_peer_binding": {"n_unbound": 2}, "populations": {"identities": 0}, "manifest_digest": "m" * 64, "identities": []}
    assert M.incomplete_rc(dict(bad, complete=True)) == 0 and M.incomplete_rc(bad) == 4
    assert "manifest INCOMPLETE" in capsys.readouterr().err
    monkeypatch.setattr(M, "build", lambda *a, **k: bad)
    out = tmp_path / "manifest.json"
    assert M.main(cli.parse("manifest", ["build", "--program", "p", "--out", str(out)], {})) == 4 and json.loads(out.read_text())["complete"] is False
