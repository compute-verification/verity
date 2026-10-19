"""The attention hidden family a request manifest requires follows the FA version its attention layers ran; a resolution that did not
record it falls back to the declared capability's FA version (`TargetProfile.effective_flash_attn_version`): FA3 only on cc 9.x."""
from __future__ import annotations

import pytest

from verity_vllm.query.required import _profile


def _res(cc=None, fa=None):
    res = {"attention_impls_observed": {"model.layers.0.self_attn.attn": {} if fa is None else {"vllm_flash_attn_version": fa}}}
    if cc is not None:
        res["target"] = {"declared": {"compute_capability": list(cc)}}
    return res


@pytest.mark.parametrize("cc, want", [(None, "ampere"), ((8, 9), "ampere"), ((8, 0), "ampere"), ((9, 0), "hopper"), ((12, 0), "ampere")])
def test_unobserved_version_falls_back_to_the_capabilitys_fa_version(cc, want):
    assert _profile(_res(cc), None)[0] == want


def test_the_observed_version_decides():
    assert _profile(_res((9, 0), fa=2), None)[0] == "ampere"
    assert _profile(_res((12, 0), fa=3), None)[0] == "hopper"
    assert _profile(_res((12, 0)), "hopper")[0] == "hopper"


def _attention_program(attention: str):
    """One GLM-5.3-shaped attention layer (the DSA indexer's families and Call names, `vllm_dsa`) over a one-token prefill and a
    one-token decode step; `attention` is the layer's attention Call: `SparseMla_v1` (MLA + DSA) or `Attention_v3` (a FlashAttention
    launch, the FA3 twin)."""
    from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
    from verity_vllm.query.units import ValueRef as R

    W, A = "model.model", "model.model.layers.0.self_attn"
    weights = InputView(-1, "weights", "{...}", (Port(f"{W}.embed.weight", 0, 8, 8, 16), Port(f"{W}.lm_head.weight", 8, 16, 8, 16)))
    ids = InputView(-2, "ids", "Value<32>", (Port("ids", 0, 1, 1, 32),))
    calls: list = []

    def add(spec, name, reads, n=4, width=16):
        calls.append(CallView(len(calls), spec, spec.split("{")[0].split("[")[0], "call", name, (Port("out", 0, n, n, width),), tuple(reads)))
        return len(calls) - 1

    tok = R(-2, "ids")
    for t in range(2):
        e = add("Gather_v1{N=4}", f"{W}.embed/embedding/out[0]", [tok, R(-1, f"{W}.embed.weight")])
        qc = add("RMSNormTriton_v1{N=4,EPS=1e-06}", f"{A}/dsa_norm_rope/q_c[0]", [R(e, "out")])
        kv = add("RMSNormTriton_v1{N=4,EPS=1e-06}", f"{A}/dsa_norm_rope/kv_c[0]", [R(e, "out")])
        if attention == "SparseMla_v1":
            ik = add("DsaIndexerKRow_v1{D=4,ROPE=2,EPS=1e-06}", f"{A}/dsa_norm_rope/index_k[0]", [R(e, "out")])
            iq = add("DsaIndexerQ_v1{H=2,D=4,ROPE=2}", f"{A}/dsa_q/index_q[0]", [R(qc, "out")])
            add(f"DsaMqaLogits_v3{{N={t + 1},H=2,D=4}}", f"{A}/dsa_indexer_attn/logits[t={t}]", [R(iq, "out"), R(ik, "out")], n=t + 1, width=32)
            ln = add(f"Const32[0x{t + 1:08x}]_v1", f"{A}/dsa_indexer_attn/len[t={t}]", [], n=1, width=32)
            add("DsaTopKShortRow_v1{TOPK=4}", f"{A}/dsa_indexer_attn/topk[t={t}]", [R(ln, "out")], width=32)
            at = add(f"SparseMla_v1{{T={t + 1},NH=2}}", f"{A}/dsa_indexer_attn[t={t}]", [R(qc, "out"), R(kv, "out")])
        else:
            at = add(f"Attention_v3{{T={t + 1},NH=2,KVH=2,D=2}}", f"{A}.attn/unified_attention/out[0]", [R(qc, "out"), R(kv, "out")])
        lm = add("Gemm_v1{N=8}", f"{W}.lm_head/linear[0]", [R(at, "out"), R(-1, f"{W}.lm_head.weight")], n=8)
        tok = R(add("ArgmaxTokenSelect_v1{N=1}", "argmax[0]", [R(lm, "out")], n=1), "out")
    return ProgramView(tuple(calls), (weights, ids), (tok,), "synthetic", "synthetic")


def _hopper_manifest(attention: str, impl: dict):
    from verity_vllm.correspondence.reader_for_query import Correspondence
    from verity_vllm.query.required import request_manifest

    P = _attention_program(attention)
    res = {"program": {"digest": "d0", "function": "f"}, "target": {"declared": {"compute_capability": [9, 0]}},
           "attention_impls_observed": {"model.layers.0.self_attn.attn": impl}}
    return request_manifest(P, Correspondence(P), res, request_id="r0").manifest


def test_mla_sparse_attention_requires_no_hidden_family_on_hopper():
    """GLM-5.3 on H200: MLA + the DSA indexer launch no FlashAttention `varlen_fwd`, so the manifest has no hidden rows and its header
    requires no hidden family, though the profile is hopper."""
    man = _hopper_manifest("SparseMla_v1", {"kind": "mla", "impl": "vllm.v1.attention.backends.mla.flashattn_mla_sparse.FlashAttnMLASparseImpl",
                                            "num_heads": 2})
    q = man["query"]
    assert q["profile"] == "hopper" and q["attention_hidden_family"] is None
    assert not {"fa2_hidden_m1_stream", "fa3_hidden_m1_stream"} & set(q["required_families"])
    assert "hidden = none" in q["query_text"]
    assert not [r for r in man["identities"] if r["family"].endswith("_hidden_m1_stream")]
    assert man["populations"]["attention_layers"] == 0


def test_flash_attention_launch_keeps_the_fa3_hidden_family():
    man = _hopper_manifest("Attention_v3", {"vllm_flash_attn_version": 3})
    q = man["query"]
    assert q["attention_hidden_family"] == "fa3_hidden_m1_stream" and "fa3_hidden_m1_stream" in q["required_families"]
    assert "hidden = fa3_hidden_m1_stream" in q["query_text"]
    assert [r["op_path"] for r in man["identities"] if r["family"] == "fa3_hidden_m1_stream"] == ["fa3.m1/L0", "fa3.m1/L0"]
