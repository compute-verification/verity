"""A request manifest's profile follows the FA version its attention layers ran; a resolution that did not record it falls back to the
declared capability's FA version (`TargetProfile.effective_flash_attn_version`): FA3 only on cc 9.x.  Under either profile the manifest
requires no attention interior, so a row's committed bytes are the Values its Calls exchange."""
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


HIDDEN = {"fa2_hidden_m1_stream", "fa3_hidden_m1_stream"}


@pytest.mark.parametrize("attention, impl", [
    ("SparseMla_v1", {"kind": "mla", "impl": "vllm.v1.attention.backends.mla.flashattn_mla_sparse.FlashAttnMLASparseImpl", "num_heads": 2}),
    ("Attention_v3", {"vllm_flash_attn_version": 3}),
])
def test_no_attention_requires_its_interior(attention, impl):
    """Neither GLM-5.3's MLA + DSA indexer nor a FlashAttention launch puts an attention interior in the manifest: its header names
    no hidden family and no row is one."""
    man = _hopper_manifest(attention, impl)
    q = man["query"]
    assert q["profile"] == "hopper" and "attention_hidden_family" not in q and "hidden =" not in q["query_text"]
    assert not HIDDEN & set(q["required_families"]) and not HIDDEN & {r["family"] for r in man["identities"]}


BYTES = {"bf16": 2, "i32": 4, "i64": 8}


def _dense_program(layers: int, LP: int, decode: int, NH: int, KVH: int, D: int):
    """A dense attention stack over an LP-token prefill and `decode` one-token steps, one Call per query row as a request Program
    records them: each row's embedding, then per layer a norm and its FlashAttention launch, then the LM head and the argmax."""
    from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
    from verity_vllm.query.units import ValueRef as R

    H, W = NH * D, "model.model"
    weights = InputView(-1, "weights", "{...}", (Port(f"{W}.embed.weight", 0, 8, 8, 16), Port(f"{W}.lm_head.weight", 8, 16, 8, 16)))
    ids = InputView(-2, "ids", "Value<32>", (Port("ids", 0, 1, 1, 32),))
    calls: list = []

    def add(spec, name, reads, n):
        calls.append(CallView(len(calls), spec, spec.split("{")[0], "call", name, (Port("out", 0, n, n, 16),), tuple(reads)))
        return len(calls) - 1

    tok, T = R(-2, "ids"), 0
    for s in range(decode + 1):
        rows = LP if s == 0 else 1
        x = [add(f"Gather_v1{{N={H}}}", f"{W}.embed/embedding/out[{i}]", [tok, R(-1, f"{W}.embed.weight")], H) for i in range(rows)]
        for k in range(layers):
            A = f"{W}.layers.{k}.self_attn"
            n = [add(f"RMSNormTriton_v1{{N={H},EPS=1e-06}}", f"{A}/norm/out[{i}]", [R(x[i], "out")], H) for i in range(rows)]
            x = [add(f"Attention_v3{{T={T + i + 1},NH={NH},KVH={KVH},D={D}}}", f"{A}.attn/unified_attention/out[{i}]", [R(n[i], "out")], H)
                 for i in range(rows)]
        T += rows
        lm = add("Gemm_v1{N=8}", f"{W}.lm_head/linear[0]", [R(x[-1], "out"), R(-1, f"{W}.lm_head.weight")], 8)
        tok = R(add("ArgmaxTokenSelect_v1{N=1}", "argmax[0]", [R(lm, "out")], 1), "out")
    return ProgramView(tuple(calls), (weights, ids), (tok,), "synthetic", "synthetic")


@pytest.mark.parametrize("cc, fa", [((8, 9), 2), ((9, 0), 3)])
def test_a_flash_attention_rows_committed_bytes_are_the_values_its_calls_exchange(cc, fa):
    """Llama-3.2-1B's attention geometry (16 layers, 32 query heads over 8 KV heads of 64) at B1 1024/128: the Commit acquires the
    Values the Calls exchange, 155,721,240 bytes of them, and nothing beside.  The FlashAttention interior streams the manifest named
    until they went were 7,510,949,888 bytes more under FA2 and 7,454,326,784 under FA3, about 48 times the Values themselves."""
    from verity_vllm.correspondence.reader_for_query import Correspondence
    from verity_vllm.query.required import request_manifest

    L, LP, decode, NH, KVH, D = 16, 1024, 128, 32, 8, 64
    P = _dense_program(L, LP, decode, NH, KVH, D)
    res = {"program": {"digest": "d0", "function": "f"}, "target": {"declared": {"compute_capability": list(cc)}},
           "attention_impls_observed": {"model.model.layers.0.self_attn.attn": {"vllm_flash_attn_version": fa}}}
    man = request_manifest(P, Correspondence(P), res, request_id="r0").manifest
    committed: dict[str, int] = {}
    for r in man["identities"]:
        lo, hi = r["element_range"]
        if hi is not None:
            committed[r["family"]] = committed.get(r["family"], 0) + (hi - lo) * BYTES[r["dtype"]]
    rows, steps = LP + decode, decode + 1
    assert committed == {"input_token_ids": 4 * rows, "instance_outputs": 2 * (rows * (1 + 2 * L) * NH * D + 8 * steps),
                         "sampled_token_ids": 8 * steps}
    assert sum(committed.values()) == 155_721_240
