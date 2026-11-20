"""Sampled exact VU replay (C2 of record) -- positive on a tiny honest Program, targeted negatives named."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from verity.circuits.defs import bind
from verity.circuits.evaluation.reference import evaluate

from verity_vllm.check.replay import index as IX
from verity_vllm.check.replay import opening as OP
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.definitions.numerics import rows as RK
from verity_vllm.definitions.registry import b1
from verity_vllm.query.program_view import from_instances
from verity_vllm.query.required import Q_MODULE_BODY_ID, request_manifest

EPS = '{"f64":"0x1.4f8b588e368f1p-17"}'
V, H, INTER, LP, T = 8, 16, 16, 2, 2          # vocab, hidden, intermediate, prompt len, decode steps (steps = LP-prefill + T decode = 1 + T)
IMPL = ("model.layers.0.input_layernorm",)  # an observed implementation path: the Program's wrapper prefix is `model.`


@pytest.fixture(scope="module", autouse=True)
def _under_module_body():
    """These worlds exercise the replay's composition of interior rows (a pre-bias Gemm, the root's embedding scale): Values interior to a
    module body, which only `Q_module_body_v1` has.  Under the query of record every Call boundary is committed and nothing is composed."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IX, "DEFAULT_QUERY", Q_MODULE_BODY_ID)
        yield

def v2_manifest(inst: dict, request_ids=("r0",), *, digest: str = "t" * 64) -> dict:
    """The required-value manifest of record (`verity_vllm.query`) of `inst` for each request id: one request manifest, or (several requests of
    the same Program, arriving together at engine step 0) their global composition, as `query.cli build-global` writes it."""
    from verity_vllm.query.compose import compose_global
    res = {"program": {"digest": digest, "function": "Derived[vLLMRequest[Test]]"}, "wrapper": "request", "LP": LP, "T": T,
           "target": {"declared": {"compute_capability": [8, 9]}}}
    P = from_instances(inst)
    parts = {rid: request_manifest(P, Correspondence(P, implementation_paths=IMPL), res, request_id=rid, query=Q_MODULE_BODY_ID).manifest
             for rid in request_ids}
    if len(parts) == 1:
        return next(iter(parts.values()))
    reqs = [{"request_id": rid, "workload_request_id": rid, "index": k, "prompt_len": LP, "max_tokens": len(parts[rid]["steps"]), "arrive_step": 0}
            for k, rid in enumerate(request_ids)]
    return compose_global({"fixture": "test", "requests": reqs}, parts, reqs, workload_digest=None)


def _bf16(rng, n, scale=1.0):
    return (np.asarray(rng.standard_normal(n) * scale, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


NH, KVH, DH, MAXPOS = 3, 1, 16, 16      # rope=True: attention heads / kv heads / head dim (one k16 step) / rotary cache rows of the qkv + RoPE rows
ABN = 16                                 # attn=True: the attention rows' key block (one k16 P.V step; T <= LP + T = 4 -> one partial block)


def _program(bias: bool = False, rope: bool = False, attn: bool = False, scale: bool = False, registered: bool = False):
    """One request Program: per token row Embedding -> RMSNormTriton -> Gemm(gate_up) -> SiluMul -> Gemm(down) -> RMSNormFused(norm) ;
    per step Gemm(logits, tied embed) -> TokenSelect.  Names / args in the Build's instance-sequence schema.
    `bias`: the gate_up linear is BIASED (Qwen2.5 qkv_proj shape): Gemm -> BiasAdd_v1 under the one module, the module returns post-bias.
    `rope`: the token row also runs Gemm(qkv) -> RoPE_v1 query[k] (member `query`, NHEADS=NH) / key[k] (member `key`,
    NHEADS=KVH) exactly as the real Programs name them (`rotary_emb/rotary_embedding_{s}/query[k]`), arg 1 = ONE row of the `weights`
    field `...rotary_emb.cos_sin_cache` at the token's position (the Program's slice names the position); the rotated heads are leaves
    here (the real consumer is the state-linked Attention row).
    `attn`: (implies rope) the token row ends in `Attention_v3{T=pos+1,NH,KVH,D,BN}` exactly as the real Programs name
    it (`self_attn.attn/unified_attention_with_output_{s}[t=k]`): arg 0 = the query RoPE member, arg 1 = the T committed KEY members of
    positions 0..pos (earlier steps' rows by ref), arg 2 = the T V slices of those positions' qkv rows; the o_proj Gemm reads its output (a leaf).
    `scale`: the ROOT module scales the embedding in its own forward body -- `Bf16MulScalarTensor_v1{N=H}` at the
    UNROOTED path `model/mul_{s}[k]` (arg 0 = the embedding row, arg 1 = the one-word `weights` field `model.normalizer`, a buffer) -- and the
    first norm AND the final norm's residual read the SCALED row: an interior family (no member by the vocabulary) whose output crosses a
    module boundary (the v2 query requires it as the root module's `out`).
    `registered`: `TargetProfile.registered_values` -- the norms are `RMSNormTriton_v3` / `RMSNormFusedCuda_v4` and attention
    `Attention_v10`, each reading its eps / scale as the last arg, a field of the registered `values` parameter (index 1; weights move to 2)."""
    rope = rope or attn
    fields = [("model.model.embed_tokens.weight", V * H, f"Array<{V},Array<{H},Value<16>>>"),
              ("model.model.layers.0.input_layernorm.weight", H, f"Array<{H},Value<16>>"),
              ("model.model.layers.0.mlp.gate_up_proj.weight", 2 * INTER * H, f"Array<{2 * INTER},Array<{H},Value<16>>>"),
              ("model.model.layers.0.mlp.down_proj.weight", H * INTER, f"Array<{H},Array<{INTER},Value<16>>>"),
              ("model.model.norm.weight", H, f"Array<{H},Value<16>>")]
    if bias:
        fields.insert(3, ("model.model.layers.0.mlp.gate_up_proj.bias", 2 * INTER, f"Array<{2 * INTER},Value<16>>"))
    if scale:
        fields.append(("model.normalizer", 1, "Value<16>"))
    QKV = (NH + 2 * KVH) * DH
    if rope:
        fields[2:2] = [("model.model.layers.0.self_attn.qkv_proj.weight", QKV * H, f"Array<{QKV},Array<{H},Value<16>>>"),
                       ("model.model.layers.0.self_attn.rotary_emb.cos_sin_cache", MAXPOS * DH, f"Array<{MAXPOS},Array<{DH},Value<16>>>")]
    if attn:
        fields.append(("model.model.layers.0.self_attn.o_proj.weight", H * NH * DH, f"Array<{H},Array<{NH * DH},Value<16>>>"))
    off, woff = 0, {}
    for n, c, _t in fields:
        woff[n] = (off, off + c)
        off += c
    wtype = "{" + ", ".join(f"{n}: {t}" for n, _c, t in fields) + "}"
    rows, i = [], 0

    def add(spec, name, args):
        nonlocal i
        rows.append({"i": i, "spec": spec, "form": "call", "name": name, "args": args})
        i += 1
        return i - 1

    wp = 2 if registered else 1
    W = lambda n: [["p", wp, woff[n][0], woff[n][1]]]  # noqa: E731
    vfields = [("eps", int(np.float32(float.fromhex(json.loads(EPS)["f64"])).view(np.uint32)))]
    if attn:
        from verity_vllm.definitions.registry import b1
        vfields.append((f"scale_log2_d{DH}", b1.scale_log2_bits(DH)))
    VAL = {n: [["p", 1, k, k + 1]] for k, (n, _b) in enumerate(vfields)}
    norm_t, norm_f = (f"RMSNormTriton_v3{{N={H}}}", f"RMSNormFusedCuda_v4{{N={H}}}") if registered else \
        (f"RMSNormTriton_v1{{N={H},EPS={EPS}}}", f"RMSNormFusedCuda_v2{{N={H},EPS={EPS}}}")
    eps_arg = [VAL["eps"]] if registered else []
    sel_prev = None                                            # the row that selected the previous step's token
    kv_rows: list[tuple[int, int]] = []                        # attn: (key RoPE row, qkv row) per position, in position order
    for s in range(1 + T):
        toks = list(range(LP)) if s == 0 else [0]
        norm_rows = []
        for k in toks:
            ids = [["p", 0, k, k + 1]] if s == 0 else [["n", sel_prev, 0, 1]]
            e = add(f"Embedding_v1{{V={V},H={H}}}", f"model.model.embed_tokens/embedding_{s}[{k}]", [ids, W("model.model.embed_tokens.weight")])
            if scale:                                                  # the root module's own scaling of the embedding: read by the child norms
                e = add(f"Bf16MulScalarTensor_v1{{N={H}}}", f"model/mul_{s}[{k}]", [[["n", e, 0, H]], W("model.normalizer")])
            ln = add(norm_t, f"model.model.layers.0.input_layernorm/triton_launch_{s}[{k}]",
                     [[["n", e, 0, H]], W("model.model.layers.0.input_layernorm.weight"), *eps_arg])
            if rope:
                pos = k if s == 0 else LP + s - 1                              # the token's position = its index in the sequence
                cs_lo = woff["model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"][0] + pos * DH
                CS = [["p", wp, cs_lo, cs_lo + DH]]
                qkv = add(f"Gemm_v1{{K={H},N={QKV}}}", f"model.model.layers.0.self_attn.qkv_proj/triton_launch_{s}q[{k}]",
                          [[["n", ln, 0, H]], W("model.model.layers.0.self_attn.qkv_proj.weight")])
                qr = add(f"RoPE_v1{{NHEADS={NH},D={DH}}}", f"model.model.layers.0.self_attn.rotary_emb/rotary_embedding_{s}/query[{k}]",
                         [[["n", qkv, 0, NH * DH]], CS])
                kr = add(f"RoPE_v1{{NHEADS={KVH},D={DH}}}", f"model.model.layers.0.self_attn.rotary_emb/rotary_embedding_{s}/key[{k}]",
                         [[["n", qkv, NH * DH, (NH + KVH) * DH]], CS])
                if attn:
                    kv_rows.append((kr, qkv))
                    at = add(f"Attention_v{10 if registered else 3}{{T={pos + 1},NH={NH},KVH={KVH},D={DH},BN={ABN}}}",
                             f"model.model.layers.0.self_attn.attn/unified_attention_with_output_{s}[t={k}]",
                             [[["n", qr, 0, NH * DH]], [["n", kr_, 0, KVH * DH] for kr_, _q in kv_rows],
                              [["n", q_, (NH + KVH) * DH, QKV] for _k, q_ in kv_rows], *([VAL[f"scale_log2_d{DH}"]] if registered else [])])
                    add(f"Gemm_v1{{K={NH * DH},N={H}}}", f"model.model.layers.0.self_attn.o_proj/triton_launch_{s}o[{k}]",
                        [[["n", at, 0, NH * DH]], W("model.model.layers.0.self_attn.o_proj.weight")])
            gu = add(f"Gemm_v1{{K={H},N={2 * INTER}}}", f"model.model.layers.0.mlp.gate_up_proj/triton_launch_{s}a[{k}]",
                     [[["n", ln, 0, H]], W("model.model.layers.0.mlp.gate_up_proj.weight")])
            if bias:                                                   # the epilogue row: consumes the Gemm's (interior) output + the bias slice
                gu = add(f"BiasAdd_v1{{N={2 * INTER}}}", f"model.model.layers.0.mlp.gate_up_proj/bias_add_{s}[{k}]",
                         [[["n", gu, 0, 2 * INTER]], W("model.model.layers.0.mlp.gate_up_proj.bias")])
            act = add(f"SiluMul_v1{{I={INTER}}}", f"model.model.layers.0.mlp.act_fn/silu_and_mul_{s}[{k}]", [[["n", gu, 0, 2 * INTER]]])
            dn = add(f"Gemm_v1{{K={INTER},N={H}}}", f"model.model.layers.0.mlp.down_proj/triton_launch_{s}b[{k}]",
                     [[["n", act, 0, INTER]], W("model.model.layers.0.mlp.down_proj.weight")])
            nm = add(norm_f, f"model.model.norm/fused_add_rms_norm_{s}[{k}]",
                     [[["n", dn, 0, H]], [["n", e, 0, H]], W("model.model.norm.weight"), *eps_arg])     # residual = the embedding (one layer)
            norm_rows.append(nm)
        lg = add(f"Gemm_v1{{K={H},N={V}}}", f"model.logits_processor/triton_launch_{s}c[0]", [[["n", norm_rows[-1], 0, H]], W("model.model.embed_tokens.weight")])
        sel_prev = add(f"TokenSelect_v1{{V={V}}}", f"argmax_{s}[0]", [[["n", lg, 0, V]]])
    params = [["prompt", f"Array<{LP},Value<32>>"], ["weights", wtype]]
    inst: dict[str, Any] = {"schema": "cb-a/instance-sequence/v1", "digest": "t" * 64, "params": params, "rows": rows}
    if registered:
        params.insert(1, ["values", "{" + ", ".join(f"{n}: Value<32>" for n, _b in vfields) + "}"])
        inst["supplied"] = {"registration": {"param": 1, "name": "values", "fields": [{"name": n, "width": 32, "bits": f"0x{b:x}", "offset": k}
                                                                                         for k, (n, b) in enumerate(vfields)]}, "run": []}
    return inst, woff


def test_struct_parser_declaration_order_and_counts():
    fs = IX.parse_struct("{a: Array<3,Array<2,Value<16>>>, b: Array<4,Value<16>>, c: Value<16>}")
    assert [(n, c, s) for n, c, s, _b in fs] == [("a", 6, [3, 2]), ("b", 4, [4]), ("c", 1, [])]


def _bnd(n_req):
    return {"context": {"ok": True}, "coverage": {"ok": True, "expected_n": 1, "bound_n": 1, "missing_n": 0}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": n_req}}


def test_token_select_first_max_tie_rule():
    logits: np.ndarray = np.asarray([0x3F80, 0x4000, 0x4000, 0x3F00], dtype=np.uint16)   # 1.0, 2.0, 2.0, 0.5 -> index 1
    assert RK.token_select_first_max(logits) == 1

    nan = [[0x7FC0, 0x3F80], [0x3F80, 0x7FC0, 0x4000], [0xFF80, 0x7FC0]]   # NaN never wins a comparison, so a NaN logits[0] is the token
    assert [RK.token_select_first_max(np.asarray(r, dtype=np.uint16)) for r in nan] == [0, 2, 0]
    assert [evaluate(bind(b1.TokenSelect, V=len(r)), r)[0] for r in nan] == [0, 2, 0]


# ------------------------------------------------------------------------------------------------
# the Hopper GEMM `Gemm_v2{K,N,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}` (targets.gemm_spec on a cc 9.0 step; the spec text of every Hopper row
# of record, e.g. qwen25-15b H100 B1 1024/128) has a registered evaluator: the same VU as Gemm_v1 (one token row x[K] @ W[N,K]^T
# -> [N]) under the wgmma k16 step (one group of 16, 26-bit adder, floor -133).  Positive (honest Hopper world replays exact; the biased
# linear composes), negatives (a flipped committed input byte -> mismatch named; Ampere-committed bytes under a Hopper Program -> the two
# adders are told apart on an adversarial lane; an unregistered / absent DOT -> unresolved BY NAME, never another chain).

HOPPER_DOT = 'DOT={"fn":"HopperBF16WgmmaDot16_v1"}'


def _hopper_program(inst: dict, dot: str = HOPPER_DOT) -> dict:
    """Every Gemm_v1{K,N} row -> Gemm_v2{K,N,<dot>} (the statics-lifted GEMM a non-(Ampere, FA2) target binds)."""
    out = json.loads(json.dumps(inst))
    for r in out["rows"]:
        if r["spec"].startswith("Gemm_v1{"):
            r["spec"] = "Gemm_v2{" + r["spec"][len("Gemm_v1{"):-1] + "," + dot + "}"
    return out


def test_model_weights_provider_reads_buffers_as_well_as_parameters_the_set_the_weights_root_pins():
    """`model_weights_provider` is the replay's registered-input reader on the pods; the rotary table
    `...rotary_emb.cos_sin_cache` is a torch BUFFER (vLLM: `register_buffer(..., persistent=False)`, one shared RotaryEmbedding instance
    named once at layer 0).  A parameters-only reader returned None for it -> every RoPE_v1 row 'registered weight field ... not available
    to the replay' (300/300 strata on the smollm2 canary).  The reader's set must equal the set `register_weights` pins:
    `named_parameters()` + `named_buffers()`; the wrapper prefix rule is unchanged; an unknown field is still None (unresolved by name)."""
    torch: Any = pytest.importorskip("torch")

    class Rot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cos_sin_cache", torch.arange(0, 8, dtype=torch.float32).to(torch.bfloat16), persistent=False)

    class Attn(torch.nn.Module):
        def __init__(self, rot):
            super().__init__()
            self.rotary_emb = rot
            self.qkv_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            rot = Rot()
            self.layers = torch.nn.ModuleList([Attn(rot), Attn(rot)])       # ONE shared rotary instance, as in vLLM

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Model()

    m = Wrapper()
    read = OP.model_weights_provider(m, "model.")
    # the buffer, under the qualname `named_buffers()` gives it (layer 0; the Program names the shared table once)
    cs = read("model.model.layers.0.rotary_emb.cos_sin_cache", 0, 8)
    assert cs is not None and cs.dtype == np.uint16 and cs.shape == (8,)
    want = m.model.layers[0].rotary_emb.cos_sin_cache.view(torch.int16).numpy().view(np.uint16)
    assert np.array_equal(cs, want)
    assert read("model.model.layers.0.rotary_emb.cos_sin_cache", 2, 5).tolist() == want[2:5].tolist()      # a slice of the field
    # a parameter still resolves, in its own width, through the same prefix rule
    w = read("model.model.layers.1.qkv_proj.weight", 0, 16)
    assert w is not None and w.dtype == np.uint16 and w.size == 16
    assert np.array_equal(w, m.model.layers[1].qkv_proj.weight.detach().contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1))
    # what the weights root pins == what the reader can serve (names), and a field in neither is None
    pinned = {n for n, _ in list(m.named_parameters()) + list(m.named_buffers())}
    assert "model.layers.0.rotary_emb.cos_sin_cache" in pinned and "model.layers.1.rotary_emb.cos_sin_cache" not in pinned
    for n in pinned:
        assert read("model." + n, 0, 1) is not None
    assert read("model.model.layers.1.rotary_emb.cos_sin_cache", 0, 8) is None                              # not a pinned qualname: unresolved, never inferred
    assert read("model.model.layers.0.rotary_emb.nothing", 0, 1) is None
    assert read("model.model.layers.0.rotary_emb.cos_sin_cache", 0, 9) is None                              # past the field's end


def test_model_weights_provider_serves_an_int32_table_as_its_u32_words():
    """DeepSeek-V4's hash-routed layers register `ffn.gate.tid2eid` as an int32 Parameter [V, 6]; the in-place reader serves its words
    as the deferred replay does (`cpu_replay.WORDS`), so a `GatherI32x{V}` row reads the table either way, and an int64 one stays None."""
    torch: Any = pytest.importorskip("torch")

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tid2eid = torch.nn.Parameter(torch.tensor([[3, -1, 7], [2, 0, 5]], dtype=torch.int32), requires_grad=False)
            self.wide = torch.nn.Parameter(torch.tensor([1, 2], dtype=torch.int64), requires_grad=False)

    read = OP.model_weights_provider(Gate(), "")
    got = read("tid2eid", 0, 6)
    assert got.dtype == np.uint32 and got.tolist() == [3, 0xFFFFFFFF, 7, 2, 0, 5]
    assert read("wide", 0, 2) is None


# --- RoPE_v1: an evaluator, not a NOT_YET label -----------------------------------------------------------------


# --- Attention_v3 / Attention_v2 row relation: state-linked resolution over the committed K/V history ------------------


# ------------------------------------------------------------------------------------------------------------- fused-MoE planes
E_MOE, TOPK, VPT = 16, 2, 8               # experts, routed slots, values per router thread (E/VPT = 2 threads -> a real xor butterfly)


def _moe_program():
    """One request Program with an OLMoE-shaped fused-MoE layer per token row: Embedding -> RMSNormTriton -> gate Gemm (router logits) ->
    MoeRouterTopK -> TOPK x [MoeExpertGemm (w13) -> SiluMul under `.experts` -> MoeExpertGemmW (w2, routed weight)] -> MoeSum -> fused norm
    fed by the sum (residual = the embedding); per step logits Gemm -> TokenSelect.  Row order, names and args are the real derive_step
    shape (OLMoE B1 256/32 Program): pair-major p = t*TOPK + k, `[p]` / `[t]` name indices, the up row reads router id slot k, the down row
    reads act row p + id slot k + weight slot k, the sum row reads the TOPK down rows in slot order."""
    fields = [("model.model.embed_tokens.weight", V * H, f"Array<{V},Array<{H},Value<16>>>"),
              ("model.model.layers.0.input_layernorm.weight", H, f"Array<{H},Value<16>>"),
              ("model.model.layers.0.mlp.gate.weight", E_MOE * H, f"Array<{E_MOE},Array<{H},Value<16>>>"),
              ("model.model.layers.0.mlp.experts.routed_experts.w13_weight", E_MOE * 2 * INTER * H, f"Array<{E_MOE},Array<{2 * INTER},Array<{H},Value<16>>>>"),
              ("model.model.layers.0.mlp.experts.routed_experts.w2_weight", E_MOE * H * INTER, f"Array<{E_MOE},Array<{H},Array<{INTER},Value<16>>>>"),
              ("model.model.norm.weight", H, f"Array<{H},Value<16>>")]
    off, woff = 0, {}
    for n, c, _t in fields:
        woff[n] = (off, off + c)
        off += c
    wtype = "{" + ", ".join(f"{n}: {t}" for n, _c, t in fields) + "}"
    rows, i = [], 0

    def add(spec, name, args):
        nonlocal i
        rows.append({"i": i, "spec": spec, "form": "call", "name": name, "args": args})
        i += 1
        return i - 1

    W = lambda n: [["p", 1, woff[n][0], woff[n][1]]]  # noqa: E731
    X = "model.model.layers.0.mlp.experts"
    sel_prev = None                                            # the row that selected the previous step's token
    for s in range(1 + T):
        toks = list(range(LP)) if s == 0 else [0]
        norm_rows = []
        for t in toks:
            ids = [["p", 0, t, t + 1]] if s == 0 else [["n", sel_prev, 0, 1]]
            e = add(f"Embedding_v1{{V={V},H={H}}}", f"model.model.embed_tokens/embedding_{s}[{t}]", [ids, W("model.model.embed_tokens.weight")])
            ln = add(f"RMSNormTriton_v1{{N={H},EPS={EPS}}}", f"model.model.layers.0.input_layernorm/triton_launch_{s}[{t}]",
                     [[["n", e, 0, H]], W("model.model.layers.0.input_layernorm.weight")])
            lg = add(f"Gemm_v1{{K={H},N={E_MOE}}}", f"model.model.layers.0.mlp.gate/triton_launch_{s}g[{t}]", [[["n", ln, 0, H]], W("model.model.layers.0.mlp.gate.weight")])
            rt = add(f"MoeRouterTopK_v1{{E={E_MOE},TOPK={TOPK},VPT={VPT}}}", f"{X}/topk_softmax_{s}[{t}]", [[["n", lg, 0, E_MOE]]])
            ups = [add(f"MoeExpertGemm_v1{{E={E_MOE},K={H},N={2 * INTER}}}", f"{X}/triton_launch_{s}u[{t * TOPK + k}]",
                       [[["n", ln, 0, H]], [["n", rt, TOPK + k, TOPK + k + 1]], W("model.model.layers.0.mlp.experts.routed_experts.w13_weight")]) for k in range(TOPK)]
            acts = [add(f"SiluMul_v1{{I={INTER}}}", f"{X}/silu_and_mul_{s}[{t * TOPK + k}]", [[["n", ups[k], 0, 2 * INTER]]]) for k in range(TOPK)]
            dns = [add(f"MoeExpertGemmW_v1{{E={E_MOE},K={INTER},N={H}}}", f"{X}/triton_launch_{s}d[{t * TOPK + k}]",
                       [[["n", acts[k], 0, INTER]], [["n", rt, TOPK + k, TOPK + k + 1]], [["n", rt, k, k + 1]], W("model.model.layers.0.mlp.experts.routed_experts.w2_weight")])
                   for k in range(TOPK)]
            sm = add(f"MoeSum_v1{{TOPK={TOPK},H={H}}}", f"{X}/moe_sum_{s}[{t}]", [[["n", d, 0, H] for d in dns]])
            nm = add(f"RMSNormFusedCuda_v2{{N={H},EPS={EPS}}}", f"model.model.norm/fused_add_rms_norm_{s}[{t}]",
                     [[["n", sm, 0, H]], [["n", e, 0, H]], W("model.model.norm.weight")])
            norm_rows.append(nm)
        lgt = add(f"Gemm_v1{{K={H},N={V}}}", f"model.logits_processor/triton_launch_{s}c[0]", [[["n", norm_rows[-1], 0, H]], W("model.model.embed_tokens.weight")])
        sel_prev = add(f"TokenSelect_v1{{V={V}}}", f"argmax_{s}[0]", [[["n", lgt, 0, V]]])
    return {"schema": "cb-a/instance-sequence/v1", "digest": "m" * 64, "params": [["prompt", f"Array<{LP},Value<32>>"], ["weights", wtype]], "rows": rows}, woff


MOE_FAMS = {"MoeRouterTopK_v1", "MoeExpertGemm_v1", "MoeExpertGemmW_v1", "MoeSum_v1"}


def _plane_word(store, entries, rid, step, plane, elem, flip=0x0080, op_path="moe/L0"):
    """Flip one committed word of request `rid`'s plane slice (request-local element `elem`) in the packed step tensor."""
    e = next(e for e in entries if e["identity"]["request_id"] == rid and e["identity"]["step"] == step and e["identity"]["op_path"] == op_path and e["identity"]["output_member"] == plane)
    isz = 4 if e["encoding"]["dtype"] in ("f32", "i32") else 2
    key = (step, e["name"])
    b = bytearray(store[key])
    o = (int(e["element_offset"]) + elem) * isz
    b[o] ^= (flip & 0xFF) if isz == 2 else 0x01
    if isz == 2:
        b[o + 1] ^= (flip >> 8) & 0xFF
    store[key] = bytes(b)


def _alias_up_to_down(store, entries, rid, step):
    """Replay the pre-FLUSH-03 defect on one request's slice: the committed `up` words of (rid, step) become the `down` words of the same
    layer (the collector read the staged up view after the block's w2 launch had overwritten the shared workspace).  Widths differ in this
    fixture (2I vs H): the common prefix is what the aliased base pointer would have produced."""
    def _slice(plane):
        e = next(e for e in entries if e["identity"]["request_id"] == rid and e["identity"]["step"] == step and e["identity"]["op_path"] == "moe/L0" and e["identity"]["output_member"] == plane)
        o = int(e["element_offset"]) * 2
        n = int(e["identity"]["element_range"][1]) * 2
        return e, o, n
    eu, ou, nu = _slice("up")
    ed, od, nd = _slice("down")
    n = min(nu, nd)
    down = store[(step, ed["name"])][od:od + n]
    b = bytearray(store[(step, eu["name"])])
    b[ou:ou + n] = down
    store[(step, eu["name"])] = bytes(b)


def _r0_rows_cover_r1(planes):
    """The binding a B=2 step got before binding.py refused a MoE plane whose rows are not a multiple of the step's tokens: the committed
    plane carries r0's rows only, and r1's entry is that whole plane at offset 0 (request-local range [0, r0's words])."""
    def mutate(store, entries):
        for plane in planes:
            r0, r1 = (next(e for e in entries if e["identity"]["request_id"] == rid and e["identity"]["step"] == 0 and e["identity"]["op_path"] == "moe/L0"
                           and e["identity"]["output_member"] == plane) for rid in ("r0", "r1"))
            n0 = int(r0["identity"]["element_range"][1])
            store[(0, r0["name"])] = store[(0, r0["name"])][:n0 * (4 if r0["encoding"]["dtype"] in ("f32", "i32") else 2)]
            r1["identity"]["element_range"], r1["element_offset"] = [0, n0], 0
    return mutate


def test_tensor_core_evaluators_name_the_measured_total_family_of_their_k16_step():
    """The evaluator provenance a record carries names the total family (`silicon_total.TOTAL_FAMILY_VERSION` /
    `HOPPER_WGMMA_FAMILY_VERSION`, restated by `registry.tc_total`) of the k16 step it replays through, so two Definitions of one
    name with different non-finite content are told apart on the record itself.  The strings are literals (EVALUATORS must import
    without verity.circuits); this test holds them equal to the constants the step is evaluated with."""
    from verity.circuits.silicon import (
        total as TC,  # the step's definition of record; the record literal keeps its historical `registry.tc_total` prefix
    )

    ampere = f"registry.tc_total {TC.AMPERE_BF16_DOT16_TOTAL.name} total family v{TC.TOTAL_FAMILY_VERSION}"
    hopper = f"registry.tc_total {TC.HOPPER_BF16_WGMMA_K16.name} family v{TC.HOPPER_WGMMA_FAMILY_VERSION}"
    assert TC.TOTAL_FAMILY_VERSION == "3" and TC.HOPPER_WGMMA_FAMILY_VERSION == "1"
    for name in ("Gemm_v1", "Attention_v3"):
        assert ampere in RK.EVALUATORS[name] and "hopper" not in RK.EVALUATORS[name].split("k16 step = ")[1], name
    for name in ("Gemm_v2", "Attention_v2"):
        assert hopper in RK.EVALUATORS[name] and ampere.split(" ", 1)[1] in RK.EVALUATORS[name], name


