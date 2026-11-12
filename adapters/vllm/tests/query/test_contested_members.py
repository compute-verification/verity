"""Two producers never share an identity (`query.required.Population.contested`): on a layer shaped like DeepSeek-V4-Pro's -- the
attention module's compressor kv_score Gemm and wo_a einsum, both unhinted and so both its member `out`, beside the indexer's kv_score
Gemm left dead on the short path, and the layer's two mHC post-pres, both `post_pre.<port>` -- each fx node's Values are spelt under
`<module>/<fx node>`, the manifest is complete, the dead one is listed unreturned under its own node, and the binder binds a declared
port's occurrences to them in Program order."""
from __future__ import annotations

from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.manifest.bind import Binder, Port as Declared
from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
from verity_vllm.query.required import Q_WORD_ID, Population, request_manifest
from verity_vllm.query.units import ValueRef as R

N, K = 4, 4
L0, ATTN = "model.layers.0", "model.layers.0.attn"
RES = {"program": {"digest": "d", "function": "f"}, "target": {"declared": {"compute_capability": [9, 0]}}}
WEIGHTS = (f"{L0}.hc_attn_fn", f"{ATTN}.compressor.fused_wkv_wgate.weight", f"{ATTN}.indexer.compressor.fused_wkv_wgate.weight",
           f"{ATTN}.wo_a.weight", f"{ATTN}.wo_b.weight", f"{L0}.hc_ffn_fn", f"{L0}.mlp.weight", "lm_head.weight")


def _program() -> tuple[ProgramView, dict[str, int]]:
    calls: list[CallView] = []

    def call(definition: str, name: str, reads, ports=(("out", N, 16),)) -> int:
        calls.append(CallView(len(calls), definition, definition.split("{")[0], "call", name,
                              tuple(Port(p, 0, n, n, w) for p, n, w in ports), tuple(reads)))
        return len(calls) - 1

    w = lambda path: R(-1, path)
    mhc = (("0", N, 16), ("1", 1, 32))
    ids = {"pre_attn": call("MhcPostPreTf32_v1", f"{L0}/mhc_post_pre/post_pre[0]", [R(-2, "x"), w(f"{L0}.hc_attn_fn")], mhc)}
    ids["kv_score"] = call(f"CublasSplitKGemmF32Out_v1{{K={K},N={N}}}", f"{ATTN}/mm[0]",
                           [R(ids["pre_attn"], "0"), w(f"{ATTN}.compressor.fused_wkv_wgate.weight")], (("out", N, 32),))
    ids["index_kv"] = call(f"CublasSplitKGemmF32Out_v1{{K={K},N={N}}}", f"{ATTN}/mm_1[0]",
                           [R(ids["pre_attn"], "0"), w(f"{ATTN}.indexer.compressor.fused_wkv_wgate.weight")], (("out", N, 32),))
    ids["compress"] = call("Dsv4CompressorScore_v1", f"{ATTN}.compressor/compress[0]", [R(ids["kv_score"], "out")])
    ids["einsum"] = call(f"ScaledMmFp8BlockGrouped_v1{{K={K},N={N}}}", f"{ATTN}/fp8_einsum[0]", [R(ids["compress"], "out"), w(f"{ATTN}.wo_a.weight")])
    ids["wo_b"] = call(f"Gemm_v1{{K={K},N={N}}}", f"{ATTN}.wo_b/linear[0]", [R(ids["einsum"], "out"), w(f"{ATTN}.wo_b.weight")])
    ids["pre_ffn"] = call("MhcPostPreTf32_v1", f"{L0}/mhc_post_pre_1/post_pre[0]",
                          [R(ids["wo_b"], "out"), R(ids["pre_attn"], "1"), w(f"{L0}.hc_ffn_fn")], mhc)
    ids["mlp"] = call(f"Gemm_v1{{K={K},N={N}}}", f"{L0}.mlp/linear[0]", [R(ids["pre_ffn"], "0"), w(f"{L0}.mlp.weight")])
    ids["logits"] = call(f"Gemm_v2{{K={K},N={N}}}", "logits_processor/linear[0]", [R(ids["mlp"], "out"), R(ids["pre_ffn"], "1"), w("lm_head.weight")])
    ids["argmax"] = call("ArgmaxTokenSelect_v1", "argmax[0]", [R(ids["logits"], "out")], (("out", 1, 32),))
    weights = ("weights", "{...}", tuple(Port(p, N * i, N * (i + 1), N, 16) for i, p in enumerate(WEIGHTS)))
    inputs = (InputView(-1, *weights), InputView(-2, "x", f"Array<{K},Value<16>>", (Port("x", 0, K, K, 16),)))
    return ProgramView(tuple(calls), inputs, (R(ids["argmax"], "out"),), "synthetic", "synthetic"), ids


def _rows(rows) -> set[tuple[str, str, str]]:
    return {(r["op_path"], r["output_member"], r["spec"].split("{")[0]) for r in rows if r["family"] in ("instance_outputs", "call_boundaries")}


def test_each_producer_of_a_shared_member_is_spelt_at_its_fx_node():
    P, ids = _program()
    pop = Population(P, Correspondence(P), query=Q_WORD_ID)
    assert pop.unreturned == {ids["index_kv"]}
    assert pop.contested == {ids[k] for k in ("pre_attn", "pre_ffn", "kv_score", "index_kv", "einsum")}
    assert pop.program_path(ids["einsum"]) == f"{ATTN}/fp8_einsum" and pop.program_path(ids["wo_b"]) == f"{ATTN}.wo_b"


def test_the_manifest_is_complete_with_one_identity_per_producer_and_the_dead_one_listed_at_its_node():
    P, ids = _program()
    man = request_manifest(P, Correspondence(P), RES, request_id="r0").manifest
    assert man["complete"] and not man["unmodelled"]
    keys = [(r["step"], r["op_path"], r["output_member"], r["invocation"]) for r in man["identities"] if r["family"] != "weights"]
    assert len(keys) == len(set(keys))
    assert {(f"{L0}/mhc_post_pre", "post_pre.0", "MhcPostPreTf32_v1"), (f"{L0}/mhc_post_pre_1", "post_pre.0", "MhcPostPreTf32_v1"),
            (f"{ATTN}/mm", "out", "CublasSplitKGemmF32Out_v1"), (f"{ATTN}/fp8_einsum", "out", "ScaledMmFp8BlockGrouped_v1"),
            (f"{ATTN}.wo_b", "out", "Gemm_v1"), (f"{ATTN}.compressor", "out", "Dsv4CompressorScore_v1")} <= _rows(man["identities"])
    assert not {(op, m) for op, m, _ in _rows(man["identities"])} & {(ATTN, "out"), (L0, "post_pre.0")}
    assert _rows(man["unreturned"]) == {(f"{ATTN}/mm_1", "out", "CublasSplitKGemmF32Out_v1")}


def test_a_declared_port_binds_the_producers_of_a_shared_member_in_program_order():
    P, _ = _program()
    man = request_manifest(P, Correspondence(P), RES, request_id="r0").manifest
    mhc, einsum = Declared(frozenset({"MhcPostPreTf32_v1"}), "0", call_boundary=True), Declared(frozenset({"ScaledMmFp8BlockGrouped_v1"}), "out", call_boundary=True)
    b = Binder(man["identities"], ports={"post_pre_0": mhc, "o_einsum": einsum})
    assert b.bind(0, [f"{L0}/post_pre_0", f"{ATTN}/o_einsum", f"{L0}/post_pre_0"]) == [
        (f"{L0}/mhc_post_pre", "post_pre.0", 0, "port"), (f"{ATTN}/fp8_einsum", "out", 0, "port"), (f"{L0}/mhc_post_pre_1", "post_pre.0", 0, "port")]
