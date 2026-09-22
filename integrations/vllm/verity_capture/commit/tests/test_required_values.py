"""R16 reqval: RequiredValues(P, Q) -- the lazy traversal (`required_values.Traversal`) against (a) the expanded reference
(`reference_required_values`: materialize every candidate, then filter) and (b) the R14/R15 inventory (`required_manifest.build_manifest`,
the manifest of record) on the differential fixtures BRIEF §7A names: tiny dense, repeated references, shared inputs, multiple outputs,
empty arrays, nested calls (Workload components), MoE, TP.  Plus the fail-closed tests: a family with no rule cannot vanish from both the
expectation and the collector (UnmodelledFamily -> complete:false -> coverage refuses; the dataflow audit names a boundary-crossing family
the vocabulary treats as interior).  No torch, no pod."""
from __future__ import annotations

import copy
import gzip
import json

import pytest

from verity_capture.commit import required_manifest as RM
from verity_capture.commit import required_values as RV
from verity_capture.commit.tests.test_required_manifest import EPS, _layouts_from, _program
from verity_capture.commit.tests.test_required_manifest_tp import _tp_program

# ---------------------------------------------------------------------------------------------------------------------------------
# a dataflow-wired fixture builder: every row carries `args` = operand runs ["n", producer, lo, hi] | ["p", param, lo, hi]
# ---------------------------------------------------------------------------------------------------------------------------------


class DF:
    """Build a request Program instance sequence with real dataflow (args), in the derive_step vocabulary."""

    def __init__(self, *, LP: int = 3, T: int = 1, H: int = 32, V: int = 16, NH: int = 2, KVH: int = 1, D: int = 16, I: int = 64):
        self.LP, self.T, self.H, self.V, self.NH, self.KVH, self.D, self.I = LP, T, H, V, NH, KVH, D, I
        self.rows: list[dict] = []
        self.weights = {"model.model.embed_tokens.weight": f"Array<{V},Array<{H},Value<16>>>"}
        self.extra_params: list[list[str]] = []                                  # user inputs after `prompt`, before `weights` (derive order)

    def param(self, name: str, typ: str) -> int:
        """Declare a Program input (`peers_<k>` of a rank Program, `seed`, ...); returns its param index (prompt = 0, weights last)."""
        self.extra_params.append([name, typ])
        return len(self.extra_params)

    def emit(self, spec: str, name: str, inputs=(), params=()) -> int:
        """inputs: producer row indices (each read whole); params: (param index, lo, hi)."""
        args = [[["n", int(j), 0, 1]] for j in inputs] + [[["p", int(p), int(lo), int(hi)]] for p, lo, hi in params]
        self.rows.append({"i": len(self.rows), "spec": spec, "form": "call", "name": name, "args": args})
        return len(self.rows) - 1

    def weight(self, name: str, shape: str) -> None:
        self.weights[name] = shape

    def program(self, *, digest: str = "p" * 64, layers: int = 2, attn_obs: dict | None = None, tp: dict | None = None) -> dict:
        w = "{" + ", ".join(f"{k}: {v}" for k, v in self.weights.items()) + "}"
        inst = {"schema": "cb-a/instance-sequence/v1", "digest": "d" * 64, "params": [["prompt", f"Array<{self.LP},Value<32>>"], *self.extra_params, ["weights", w]],
                "rows": self.rows}
        obs = attn_obs if attn_obs is not None else {f"model.layers.{L}.self_attn.attn": {"vllm_flash_attn_version": 2, "num_heads": self.NH, "num_kv_heads": self.KVH, "head_size": self.D}
                                                    for L in range(layers)}
        res = {"program": {"digest": digest, "function": "Derived[vLLMRequest[Test]]"}, "wrapper": "request", "LP": self.LP, "T": self.T,
               "model_pin": {"model": "test"}, "target": {"declared": {"compute_capability": [8, 9]}}, "attention_impls_observed": obs}
        if tp:
            res["tp"] = tp
        return {"instances": inst, "result": res, "artifact": {"program_digest": digest, "identity": "i" * 64}, "dir": "<synthetic-df>"}


def dense_df(*, LP: int = 3, T: int = 1, layers: int = 2, extra_reader: bool = False, moe_layers: tuple[int, ...] = (), M_topk: int = 2, E: int = 4,
             padded_moe: str | None = None) -> dict:
    """A tiny dense (or partly MoE) request Program with wired dataflow: prefill of LP rows + T decode steps, each closed by TokenSelect.
    `extra_reader`: a second module (`model.model.layers.0.self_attn.probe`) reads layer 0's normed rows too (a repeated reference).
    `moe_layers`: layers whose MLP is a fused MoE block (router Gemm under `mlp.gate`, block under `mlp.experts`)."""
    b = DF(LP=LP, T=T)
    H, V, NH, KVH, D, I = b.H, b.V, b.NH, b.KVH, b.D, b.I
    for L in range(layers):
        b.weight(f"model.model.layers.{L}.self_attn.qkv_proj.weight", f"Array<{(NH + 2 * KVH) * D},Array<{H},Value<16>>>")
        b.weight(f"model.model.layers.{L}.self_attn.o_proj.weight", f"Array<{H},Array<{NH * D},Value<16>>>")
        if L in moe_layers:
            b.weight(f"model.model.layers.{L}.mlp.gate.weight", f"Array<{E},Array<{H},Value<16>>>")
            b.weight(f"model.model.layers.{L}.mlp.experts.w13_weight", f"Array<{E},Array<{2 * I},Array<{H},Value<16>>>>")
            b.weight(f"model.model.layers.{L}.mlp.experts.w2_weight", f"Array<{E},Array<{H},Array<{I},Value<16>>>>")
        else:
            b.weight(f"model.model.layers.{L}.mlp.gate_up_proj.weight", f"Array<{2 * I},Array<{H},Value<16>>>")
            b.weight(f"model.model.layers.{L}.mlp.down_proj.weight", f"Array<{H},Array<{I},Value<16>>>")
    b.weight("model.lm_head.weight", f"Array<{V},Array<{H},Value<16>>>")
    wp = 1
    keys: dict[int, list[int]] = {L: [] for L in range(layers)}                  # RoPE key rows per layer (read by every later attention row)
    pos = 0
    last_select: int | None = None
    for s in range(T + 1):
        n = LP if s == 0 else 1
        sfx = "" if s == 0 else f"_{s}"
        # the prefill embeds the prompt ids (a parameter); a decode step embeds the id the previous step SAMPLED (the selector's output)
        emb = [b.emit(f"Embedding_v1{{V={V},H={H}}}", f"model.model.embed_tokens/embedding{sfx}[{r}]", inputs=([] if s == 0 else [last_select]),
                      params=([(0, pos + r, pos + r + 1)] if s == 0 else []) + [(wp, 0, V * H)]) for r in range(n)]
        resid = emb
        normed = emb
        for L in range(layers):
            if L == 0:
                normed = [b.emit(f"RMSNormTriton_v1{{N={H},{EPS}}}", f"model.model.layers.{L}.input_layernorm/triton_launch{sfx}[{r}]", inputs=[resid[r]]) for r in range(n)]
            else:
                fused = [b.emit(f"RMSNormFusedCuda_v2{{N={H},{EPS}}}", f"model.model.layers.{L}.input_layernorm/fused_add_rms_norm{sfx}[{r}]", inputs=[mlp_out[r], resid[r]]) for r in range(n)]
                normed, resid = fused, fused
            if extra_reader and L == 0:
                for r in range(n):
                    b.emit(f"Gemm_v1{{K={H},N={H}}}", f"model.model.layers.{L}.self_attn.probe/triton_launch_p{sfx}[{r}]", inputs=[normed[r]])
            qkv = [b.emit(f"Gemm_v1{{K={H},N={(NH + 2 * KVH) * D}}}", f"model.model.layers.{L}.self_attn.qkv_proj/triton_launch_1{sfx}[{r}]", inputs=[normed[r]]) for r in range(n)]
            q = [b.emit(f"RoPE_v1{{NHEADS={NH},D={D}}}", f"model.model.layers.{L}.self_attn.rotary_emb/rotary_embedding{sfx}/query[{r}]", inputs=[qkv[r]]) for r in range(n)]
            k = [b.emit(f"RoPE_v1{{NHEADS={KVH},D={D}}}", f"model.model.layers.{L}.self_attn.rotary_emb/rotary_embedding{sfx}/key[{r}]", inputs=[qkv[r]]) for r in range(n)]
            keys[L] += k
            att = [b.emit(f"Attention_v3{{T={pos + r + 1},NH={NH},KVH={KVH},D={D},BN=128}}", f"model.model.layers.{L}.self_attn.attn/attention{sfx}_{r}[0]",
                          inputs=[q[r]] + keys[L][: pos + r + 1] + [qkv[r]]) for r in range(n)]
            o = [b.emit(f"Gemm_v1{{K={NH * D},N={H}}}", f"model.model.layers.{L}.self_attn.o_proj/triton_launch_2{sfx}[{r}]", inputs=[att[r]]) for r in range(n)]
            post = [b.emit(f"RMSNormFusedCuda_v2{{N={H},{EPS}}}", f"model.model.layers.{L}.post_attention_layernorm/fused_add_rms_norm_1{sfx}[{r}]", inputs=[o[r], resid[r]]) for r in range(n)]
            resid = post
            if L in moe_layers and padded_moe:
                # [MOE-PAD-SPEC route A (3)] the PADDED construction: the router logits Gemm, then ONE fixed-slot block call per token row
                # (`MoeFfnFixedL_v1{E,TOPK,VPT,H,I}(x, logits, ..) -> y`; `padded_moe` = the family emitted: the unlifted twin or its hand lift)
                gate = [b.emit(f"Gemm_v1{{K={H},N={E}}}", f"model.model.layers.{L}.mlp.gate/triton_launch_g{sfx}[{r}]", inputs=[post[r]]) for r in range(n)]
                mlp_out = [b.emit(f"{padded_moe}{{E={E},TOPK={M_topk},VPT={E // M_topk},H={H},I={I}}}", f"model.model.layers.{L}.mlp.experts/moe_ffn_fixed{sfx}[{r}]",
                                  inputs=[post[r], gate[r]]) for r in range(n)]
            elif L in moe_layers:
                gate = [b.emit(f"Gemm_v1{{K={H},N={E}}}", f"model.model.layers.{L}.mlp.gate/triton_launch_g{sfx}[{r}]", inputs=[post[r]]) for r in range(n)]
                mlp_out = []
                for r in range(n):
                    rt = b.emit(f"MoeRouterTopK_v1{{E={E},TOPK={M_topk}}}", f"model.model.layers.{L}.mlp.experts/router{sfx}[{r}]", inputs=[gate[r]])
                    downs = []
                    for kk in range(M_topk):
                        up = b.emit(f"MoeExpertGemm_v1{{K={H},N={2 * I}}}", f"model.model.layers.{L}.mlp.experts/expert_up{sfx}_{kk}[{r}]", inputs=[post[r], rt])
                        act = b.emit(f"SiluMul_v1{{I={I}}}", f"model.model.layers.{L}.mlp.experts/silu{sfx}_{kk}[{r}]", inputs=[up])
                        downs.append(b.emit(f"MoeExpertGemmW_v1{{K={I},N={H}}}", f"model.model.layers.{L}.mlp.experts/expert_down{sfx}_{kk}[{r}]", inputs=[act]))
                    mlp_out.append(b.emit(f"MoeSum_v1{{H={H},TOPK={M_topk}}}", f"model.model.layers.{L}.mlp.experts/moe_sum{sfx}[{r}]", inputs=downs + [rt]))
            else:
                gu = [b.emit(f"Gemm_v1{{K={H},N={2 * I}}}", f"model.model.layers.{L}.mlp.gate_up_proj/triton_launch_3{sfx}[{r}]", inputs=[post[r]]) for r in range(n)]
                act = [b.emit(f"SiluMul_v1{{I={I}}}", f"model.model.layers.{L}.mlp.act_fn/silu_and_mul{sfx}[{r}]", inputs=[gu[r]]) for r in range(n)]
                mlp_out = [b.emit(f"Gemm_v1{{K={I},N={H}}}", f"model.model.layers.{L}.mlp.down_proj/triton_launch_4{sfx}[{r}]", inputs=[act[r]]) for r in range(n)]
        fin = [b.emit(f"RMSNormFusedCuda_v2{{N={H},{EPS}}}", f"model.model.norm/fused_add_rms_norm_9{sfx}[{r}]", inputs=[mlp_out[r], resid[r]]) for r in range(n)]
        logits = b.emit(f"Gemm_v1{{K={H},N={V}}}", f"model.logits_processor/triton_launch_9{sfx}[0]", inputs=[fin[-1]])
        last_select = b.emit(f"TokenSelect_v1{{V={V}}}", f"argmax{sfx}[0]", inputs=[logits])
        pos += n
    return b.program(layers=layers)


def rows_of(ids: list[RV.Identity]) -> list[dict]:
    return sorted((x.row() for x in ids), key=RM._row_key)


def traversal_rows(prog: dict, q: RV.Query | None = None) -> tuple[list[dict], RV.Traversal]:
    t = RV.Traversal(copy.deepcopy(prog), q or RV.Query())
    return rows_of(list(t)), t


@pytest.fixture(autouse=True)
def _inventory_is_the_old_side(monkeypatch):
    """every `RM.build_manifest` here is the OLD side of a differential: the engine flag must not turn it into the traversal"""
    monkeypatch.delenv(RM.ENGINE_ENV, raising=False)


def _family_keys(um: dict) -> set[str]:
    """The (family under module) part of an unmodelled key -- the message texts of the order-dependent D2 variants differ."""
    return {k.split(" (")[0] for k in um}


def assert_three_way(prog: dict, *, q: RV.Query | None = None, workload: dict | None = None, **kw) -> tuple[list[dict], RV.Traversal]:
    """traversal == reference == the R14/R15 inventory (identities as manifest rows, digest, completeness, unmodelled families)."""
    q = q or RV.Query()
    tr, t = traversal_rows(prog, q)
    ref, summ = RV.reference_required_values(copy.deepcopy(prog), q)
    ref_rows = [x.row() for x in ref]
    assert tr == ref_rows, (len(tr), len(ref_rows), [r for r in tr if r not in ref_rows][:2], [r for r in ref_rows if r not in tr][:2])
    assert t.complete == summ["complete"] and _family_keys(t.unmodelled) == _family_keys(summ["unmodelled"]) and t.aliases == summ["aliases"] and t.steps == summ["steps"]
    old = RM.build_manifest(copy.deepcopy(prog), workload, request_id=q.request_id, rank=q.rank, ranks=q.ranks, profile=q.profile, shared_rope=q.shared_rope)
    if workload is None:
        old_arrive = 0
    else:
        old_arrive = old["workload"]["arrive_step"]
    assert old_arrive == q.arrive_step
    assert old["identities"] == tr, [r for r in old["identities"] if r not in tr][:2]
    assert RM.manifest_digest(tr) == old["manifest_digest"]
    assert old["complete"] == t.complete and old["unmodelled"] == t.unmodelled and old["op_path_aliases"] == t.aliases
    new = RV.manifest_from_traversal(copy.deepcopy(prog), workload, request_id=q.request_id, rank=q.rank, ranks=q.ranks, profile=q.profile, shared_rope=q.shared_rope)
    eq = RV.manifest_equal(old, new)
    assert eq["equal"], eq["why"]
    return tr, t


# ---------------------------------------------------------------------------------------------------------------------------------
# (2) differential fixtures
# ---------------------------------------------------------------------------------------------------------------------------------


def test_tiny_dense_traversal_equals_reference_and_inventory():
    prog = dense_df()
    tr, t = assert_three_way(prog)
    fam = t.populations()["by_family"]
    assert fam["sampled_token_ids"] == 2 and fam["input_token_ids"] == 2 and fam["fa2_hidden_m1_stream"] == 4 and fam["weights"] == 10
    assert t.complete and t.boundary_audit["available"] and t.boundary_audit["unnamed_crossing"] == {} and t.boundary_audit["named_interior"] == {}
    aud = t.boundary_audit["families"]
    assert set(aud["Gemm_v1"]) == {"boundary"} and aud["Gemm_v1"]["boundary"]["crossing"] == aud["Gemm_v1"]["boundary"]["rows"]
    assert aud["TokenSelect_v1"]["sampler"]["crossing"] == 1                        # the last sampled id is the Program's output: no reader
    # the R14 synthetic fixture (no args) agrees too; the audit is honestly unavailable there
    tr2, t2 = assert_three_way(_program())
    assert t2.boundary_audit["available"] is False and t2.boundary_audit["unnamed_crossing"] == {}


def test_repeated_references_one_identity_per_value():
    """(a) one normed value read by two modules -> one identity of the producer (the reader is its own module output); (b) the shared
    RotaryEmbedding called from every layer -> ONE canonical op_path with invocation = layer order + aliases."""
    prog = dense_df(extra_reader=True, layers=3)
    tr, t = assert_three_way(prog)
    s0 = [r for r in tr if r["step"] == 0]
    norm0 = [r for r in s0 if r["op_path"] == "model.layers.0.input_layernorm"]
    assert len(norm0) == 1 and norm0[0]["rows"] == 3 and norm0[0]["element_range"] == [0, 3 * 32]
    probe = [r for r in s0 if r["op_path"] == "model.layers.0.self_attn.probe"]
    assert len(probe) == 1 and probe[0]["output_member"] == "out" and probe[0]["rows"] == 3       # a non-linear leaf: the module's single tensor
    rope = sorted((r["invocation"], r["output_member"]) for r in s0 if r["op_path"] == "model.layers.0.self_attn.rotary_emb")
    assert rope == [(0, "0"), (0, "1"), (1, "0"), (1, "1"), (2, "0"), (2, "1")]
    assert t.aliases == {"model.layers.1.self_attn.rotary_emb": ["model.layers.0.self_attn.rotary_emb", 1], "model.layers.2.self_attn.rotary_emb": ["model.layers.0.self_attn.rotary_emb", 2]}
    assert not any(r["op_path"].endswith("layers.1.self_attn.rotary_emb") for r in tr)
    # the same Program with per-layer rotary modules (shared_rope=False): one op_path per layer, invocation 0, no aliases
    tr_u, t_u = assert_three_way(prog, q=RV.Query(shared_rope=False))
    assert t_u.aliases == {} and any(r["op_path"] == "model.layers.2.self_attn.rotary_emb" and r["invocation"] == 0 for r in tr_u)


def test_shared_inputs_weights_once_per_program_and_once_per_workload():
    prog = dense_df(layers=2)
    tr, t = assert_three_way(prog)
    w = [r for r in tr if r["family"] == "weights"]
    assert len(w) == len(set(r["op_path"] for r in w)) == 10 and all(r["step"] == "*" and r["element_range"] == [0, None] for r in w)
    assert [r["op_path"] for r in w] == sorted(r["op_path"] for r in w)
    # a workload of two requests of the same model: the shared weights appear ONCE in the fold, every other identity per request
    fixture = {"fixture": "t_b2", "requests": [{"request_id": "A", "index": 0, "prompt_len": 3, "max_tokens": 2, "arrive_step": 0},
                                                {"request_id": "B", "index": 1, "prompt_len": 2, "max_tokens": 2, "arrive_step": 0}]}
    progs = {"r0": dense_df(LP=3, T=1), "r1": dense_df(LP=2, T=1)}
    g = RV.global_manifest_from_traversal(fixture, copy.deepcopy(progs), workload_digest="w" * 64)
    assert g["populations"]["by_family"]["weights"] == 10 and g["populations"]["by_family"]["input_token_ids"] == 4
    assert all(r["program_digest"] == "w" * 64 for r in g["identities"])
    old = RM.build_global_manifest(fixture, copy.deepcopy(progs), workload_digest="w" * 64)
    eq = RV.manifest_equal(old, g)
    assert eq["equal"], eq["why"]


def test_multiple_outputs_every_member_required_and_droppable_only_named():
    prog = dense_df()
    tr, t = assert_three_way(prog)
    fused = {(r["op_path"], r["output_member"]) for r in tr if r["step"] == 0 and str(r.get("spec", "")).startswith("RMSNormFusedCuda")}
    assert ("model.layers.1.input_layernorm", "0") in fused and ("model.layers.1.input_layernorm", "1") in fused and ("model.norm", "1") in fused
    rope = {(r["op_path"], r["output_member"], r["element_range"][1]) for r in tr if r["step"] == 0 and r.get("spec") == "RoPE_v1"}
    assert ("model.layers.0.self_attn.rotary_emb", "0", 3 * 2 * 16) in rope and ("model.layers.0.self_attn.rotary_emb", "1", 3 * 1 * 16) in rope
    man = RV.manifest_from_traversal(copy.deepcopy(prog))
    lay = _layouts_from(man)
    assert RM.coverage_check(man, lay)["ok"]
    lay1 = {s: [m for m in ms if m["name"] != "model.layers.1.input_layernorm/1"] for s, ms in lay.items()}
    r1 = RM.coverage_check(man, lay1)
    assert not r1["ok"] and all(x["output_member"] == "1" for x in r1["missing"]) and r1["missing_n"] == 2


def test_empty_arrays_never_a_zero_width_identity_and_fail_closed():
    """MAN-02: a member of zero width (Gemm N=0, SiluMul I=0) is never yielded, the population is incomplete, coverage refuses; the
    traversal, the reference and the inventory agree on the identities that remain and on the named defect."""
    for fam_edit in (("model.model.layers.1.mlp.down_proj/", "Gemm_v1{K=64,N=0}"), ("model.model.layers.0.mlp.act_fn/", "SiluMul_v1{I=0}")):
        prog = dense_df()
        for r in prog["instances"]["rows"]:
            if r["name"].startswith(fam_edit[0]):
                r["spec"] = fam_edit[1]
        tr, t = assert_three_way(prog)
        assert not t.complete and any("width 0" in k for k in t.unmodelled)
        assert not any(r["element_range"][1] == 0 for r in tr) and all(r["element_range"][1] is None or r["element_range"][1] > 0 for r in tr)
        man = RV.manifest_from_traversal(copy.deepcopy(prog))
        res = RM.coverage_check(man, _layouts_from(man))
        assert res["ok"] is False and "width 0" in res["error"]
    # an empty weights record field list is not an identity either (no `weights` rows when the Program names none)
    prog = dense_df()
    prog["instances"]["params"] = [["prompt", "Array<3,Value<32>>"], ["weights", "{}"]]
    tr, t = traversal_rows(prog)
    assert not any(r["family"] == "weights" for r in tr) and t.complete


def test_nested_calls_workload_components_lazy_fold_equals_global_inventory():
    """GP-01 Workload_v1: three staggered components; the traversal recurses per component with its arrival, yields the shared weights
    once and the per-engine-step hidden union; equals build_global_manifest and a fold of the per-component references."""
    fixture = {"fixture": "t_b3", "requests": [{"request_id": "r0", "index": 0, "prompt_len": 4, "max_tokens": 2, "arrive_step": 0},
                                                {"request_id": "r1", "index": 1, "prompt_len": 3, "max_tokens": 3, "arrive_step": 1},
                                                {"request_id": "r2", "index": 2, "prompt_len": 5, "max_tokens": 2, "arrive_step": 3}]}
    progs = {"r0": dense_df(LP=4, T=1), "r1": dense_df(LP=3, T=2), "r2": dense_df(LP=5, T=1)}
    g_old = RM.build_global_manifest(fixture, copy.deepcopy(progs), workload_digest="w" * 64)
    g_new = RV.global_manifest_from_traversal(fixture, copy.deepcopy(progs), workload_digest="w" * 64)
    eq = RV.manifest_equal(g_old, g_new)
    assert eq["equal"], eq["why"]
    for k in ("engine_steps", "components", "requests_declared", "composition_reference", "steps", "requests", "component_digests", "program_digest_kind", "addressing"):
        assert g_old[k] == g_new[k], k
    # [REQ-01] the canonical 8-tuple is unique at B=1 and NOT for the global hidden union rows (engine_step / requests are annotations):
    # counted and named in the traversal manifest, never silently deduplicated -- the row count and the digest stay the inventory's
    uk = g_new["engine"]["identity_keys"]
    assert uk["rows"] == len(g_new["identities"]) == g_old["populations"]["identities"]
    if not uk["unique"]:
        assert set(uk["families"]) == {"fa2_hidden_m1_stream"} and uk["rows"] - uk["distinct_canonical"] == uk["rows_sharing_a_key"] - uk["colliding_keys"]
        ann = {(r["engine_step"], tuple(r["requests"])) for r in g_new["identities"] if r["family"] == "fa2_hidden_m1_stream"}
        assert len(ann) == sum(1 for r in g_new["identities"] if r["family"] == "fa2_hidden_m1_stream") // len({r["op_path"] for r in g_new["identities"] if r["family"] == "fa2_hidden_m1_stream"})
    assert RV.manifest_from_traversal(dense_df(LP=4, T=2), {"sweep": {"arrive_steps": [0]}})["engine"]["identity_keys"]["unique"]
    # the components' references fold to the same request-local identities (weights once; hidden rows are the union, checked by count)
    wl = RV.Workload.of(fixture, copy.deepcopy(progs), workload_digest="w" * 64)
    folded: list[dict] = []
    seen_w = set()
    for c in wl.components:
        ref, _s = RV.reference_required_values(c.program, RV.Query(request_id=c.request_id, arrive_step=c.arrive_step))
        for x in ref:
            if x.family == "weights":
                if x.op_path in seen_w:
                    continue
                seen_w.add(x.op_path)
            if x.family.endswith("hidden_m1_stream"):
                continue
            folded.append(x.row())
    new_local = [r for r in g_new["identities"] if not r["family"].endswith("hidden_m1_stream")]
    key = lambda r: (r["request_id"], str(r["step"]), r["op_path"], r["output_member"], r["invocation"])
    assert sorted(map(key, folded)) == sorted(map(key, new_local))
    hidden = [r for r in g_new["identities"] if r["family"].endswith("hidden_m1_stream")]
    assert len(hidden) == 2 * len(g_new["engine_steps"]) and all(r["request_id"] == "*" for r in hidden)
    e1 = next(r for r in hidden if r["engine_step"] == 1 and r["op_path"] == "fa2.m1/L0")
    assert e1["requests"] == ["r0", "r1"] and e1["geometry"]["n_seq"] == 2
    # the same fixture without a workload digest folds the component digests (kind components-fold) identically in both
    a, b = RM.build_global_manifest(fixture, copy.deepcopy(progs)), RV.global_manifest_from_traversal(fixture, copy.deepcopy(progs))
    assert a["program_digest"] == b["program_digest"] and a["manifest_digest"] == b["manifest_digest"] and b["program_digest_kind"] == "components-fold"
    # laziness: identities of r0 are yielded before r2's rows are read
    consumed = {"r2": 0}
    p2 = copy.deepcopy(progs["r2"])
    rows2 = p2["instances"]["rows"]

    def gen():
        for r in rows2:
            consumed["r2"] += 1
            yield r
    v2 = RV.ProgramView.of(p2)
    v2.rows = gen()
    wl2 = RV.Workload([RV.Component("r0", 0, 0, 4, 2, RV.ProgramView.of(copy.deepcopy(progs["r0"]))), RV.Component("r2", 2, 3, 5, 2, v2)], "w" * 64)
    it = iter(RV.WorkloadTraversal(wl2))
    first = [next(it) for _ in range(20)]
    assert consumed["r2"] == 0 and any(x.family == "instance_outputs" and x.request_id == "r0" for x in first)
    rest = list(it)
    assert consumed["r2"] == len(rows2) and any(x.request_id == "r2" for x in rest)


def test_moe_block_stream_planes_and_routing_geometry():
    """A fused-MoE layer beside a dense one: six planes per MoE layer per step from the Program's routing statics (M x TOPK), the router
    Gemm `mlp.gate` as member 0, the block's rows as the aggregate class (their values are bound under the planes); MoeSum crosses the
    module boundary (read by the next norm) and the audit classes it `aggregate`, not `unnamed_crossing`."""
    prog = dense_df(layers=2, moe_layers=(1,), M_topk=2, E=4)
    tr, t = assert_three_way(prog)
    planes = sorted((r["output_member"], r["element_range"][1]) for r in tr if r["step"] == 0 and r["family"] == "moe_block_stream")
    M, K, H, I, N = 3, 2, 32, 64, 128
    assert planes == sorted([("router_w", M * K), ("router_ids", M * K), ("up", M * K * N), ("act", M * K * I), ("down", M * K * H), ("sum", M * H)])
    assert all(r["op_path"] == "moe/L1" and r["moe_module"] == "model.layers.1.mlp.experts" for r in tr if r["family"] == "moe_block_stream")
    assert any(r["op_path"] == "model.layers.1.mlp.gate" and r["output_member"] == "0" and r["element_range"] == [0, M * 4] for r in tr if r["step"] == 0)
    assert not any(r["op_path"] == "model.layers.1.mlp.experts" for r in tr)
    assert "moe_block_stream" in t.query_info["required_families"] and t.query_info["moe_layers"] == [1]
    aud = t.boundary_audit["families"]
    assert set(aud["MoeSum_v1"]) == {"aggregate"} and aud["MoeSum_v1"]["aggregate"]["crossing"] == aud["MoeSum_v1"]["aggregate"]["rows"]
    assert set(aud["MoeExpertGemm_v1"]) == {"aggregate"} and aud["MoeExpertGemm_v1"]["aggregate"]["crossing"] == 0
    assert set(aud["SiluMul_v1"]) == {"aggregate", "boundary"}                     # experts' act rows vs the dense layer's act_fn
    assert t.boundary_audit["unnamed_crossing"] == {}
    # routing-dependent packing declared by the Program: an expert-GEMM count that is not M x TOPK is refused by name in both
    bad = copy.deepcopy(prog)
    rows = [r for r in bad["instances"]["rows"] if not r["name"].startswith("model.model.layers.1.mlp.experts/expert_up_1[0]")]
    for k, r in enumerate(rows):
        r["i"] = k
    bad["instances"]["rows"] = rows
    with pytest.raises(ValueError, match="expert-GEMM instances != M\\*TOPK"):
        list(RV.Traversal(bad))
    with pytest.raises(ValueError, match="expert-GEMM instances != M\\*TOPK"):
        RM.build_manifest(bad)
    # a collector without the planes fails coverage on the family by name
    man = RV.manifest_from_traversal(copy.deepcopy(prog))
    lay = _layouts_from(man)
    lay_np = {s: [m for m in ms if not m["name"].startswith("moe/")] for s, ms in lay.items()}
    r = RM.coverage_check(man, lay_np)
    assert not r["ok"] and r["families_missing_entirely"] == ["moe_block_stream"]


def test_tp_rank_partials_and_collective_outputs_both_required():
    """TP-04: rank partials (`part_rank<r>`, `shard_rank<r>`) AND the collective outputs, per rank; the partial feeds the module's own
    collective, so the dataflow keeps it inside the module -- the audit classes it `declared` (Q requires it), never `named_interior`."""
    for rank in (0, 1):
        prog = _tp_program(rank)
        tr, t = assert_three_way(prog, q=RV.Query(rank=rank, ranks=2), workload={"sweep": {"row_id": "tp2-test", "batch": 1, "arrive_steps": [0]}})
        fam = t.populations()["by_family"]
        assert fam["tp_rank_partials"] == (2 * 2 + 2) * 3 and all(r["rank"] == rank for r in tr)
        assert {r["output_member"] for r in tr if r["family"] == "tp_rank_partials"} == {f"part_rank{rank}", f"shard_rank{rank}"}
        assert any(r.get("spec") == "AllReduce2_v1" and r["output_member"] == "0" for r in tr) and any(r.get("spec") == "AllGather2_v1" for r in tr)
        assert t.query_info["tp"]["rank"] == rank
    # a wired TP fixture: Gemm partial -> AllReduce2 inside one module (no crossing) is `declared`; the collective output crosses
    b = DF(LP=2, T=0)
    H = b.H
    b.weight("model.model.layers.0.mlp.down_proj.weight", f"Array<{H},Array<{H // 2},Value<16>>>")
    e = [b.emit(f"Embedding_v1{{V={b.V},H={H}}}", f"model.model.embed_tokens/embedding[{r}]", params=[(0, r, r + 1)]) for r in range(2)]
    parts = [b.emit(f"Gemm_v1{{K={H // 2},N={H}}}", f"model.model.layers.0.mlp.down_proj/triton_launch[{r}]", inputs=[e[r]]) for r in range(2)]
    ar = [b.emit(f"AllReduce2_v1{{N={H}}}", f"model.model.layers.0.mlp.down_proj/all_reduce2_1[{r}]/all_reduce", inputs=[parts[r]]) for r in range(2)]
    fin = [b.emit(f"RMSNormTriton_v1{{N={H},{EPS}}}", f"model.model.norm/triton_launch[{r}]", inputs=[ar[r]]) for r in range(2)]
    lg = b.emit(f"Gemm_v1{{K={H},N={b.V}}}", f"model.logits_processor/triton_launch_9[0]", inputs=[fin[-1]])
    b.emit(f"TokenSelect_v1{{V={b.V}}}", "argmax[0]", inputs=[lg])
    prog = b.program(layers=0, attn_obs={}, tp={"world": 2, "rank": 0})
    tr, t = assert_three_way(prog, q=RV.Query(rank=0, ranks=2))
    aud = t.boundary_audit["families"]
    assert set(aud["Gemm_v1"]) == {"declared", "boundary"} and aud["Gemm_v1"]["declared"] == {"rows": 2, "crossing": 0}
    assert aud["AllReduce2_v1"]["boundary"]["crossing"] == 2 and t.boundary_audit["named_interior"] == {} and t.boundary_audit["unnamed_crossing"] == {}
    # the rank rule: a rank Program refuses another rank's / world's query in both implementations
    with pytest.raises(ValueError, match="rank Program mismatch"):
        list(RV.Traversal(_tp_program(1), RV.Query(rank=0, ranks=2)))


def _tp_df(rank: int, *, world: int = 2, stray_reader: bool = False, down_proj_twice: bool = False) -> dict:
    """A WIRED TP rank Program in the derive_step --tp vocabulary (the llama TP2 B1 record's shape at toy size): per step k = 0 all_reduce on
    the embedding shard, k = 1 all_reduce on down_proj (row-parallel Gemm), k = 2 all_gather on the logits shard; every collective row
    reads the rank's OWN partial (the producer row) and the RECEIVED peer partial = the Program input `peers_<k>` (tp_links.json:
    peer_input peers_k, peer_rank 1-r).  `stray_reader`: the norm also reads `peers_1` directly (a peers input some non-collective row
    reads -- must stay unnamed by name).  `down_proj_twice` [revb F-r17b-17]: step 0 ISSUES the down_proj module path a second time (a
    second row-parallel Gemm + all_reduce on the same module, reading a further peer input `peers_6`) -- the same module path committed
    twice in one step."""
    b = DF(LP=2, T=1)
    H, V = b.H, b.V
    b.weight("model.model.layers.0.mlp.down_proj.weight", f"Array<{H},Array<{H // world},Value<16>>>")
    b.weight("model.lm_head.weight", f"Array<{V // world},Array<{H},Value<16>>>")
    peers = {}
    for k, (rows, n) in enumerate([(2, H), (2, H), (1, V // world), (1, H), (1, H), (1, V // world)] + ([(2, H)] if down_proj_twice else [])):
        peers[k] = b.param(f"peers_{k}", f"Array<{rows},Array<{n},Value<16>>>")
    k = 0
    for s, rows in ((0, [0, 1]), (1, [0])):
        sh = [b.emit(f"EmbeddingShard_v1{{VS={V // world},H={H},START={rank * (V // world)}}}", f"model.model.embed_tokens/embedding_shard_{k}[{r}]/shard",
                     params=[(0, s + r, s + r + 1)]) for r in rows]
        ar = [b.emit(f"AllReduce2_v1{{N={H}}}", f"model.model.embed_tokens/all_reduce2_{k}[{r}]/all_reduce", inputs=[sh[r]], params=[(peers[k], r, r + 1)]) for r in rows]
        k += 1
        g = [b.emit(f"Gemm_v1{{K={H // world},N={H}}}", f"model.model.layers.0.mlp.down_proj/triton_launch[{r}]", inputs=[ar[r]]) for r in rows]
        ar2 = [b.emit(f"AllReduce2_v1{{N={H}}}", f"model.model.layers.0.mlp.down_proj/all_reduce2_{k}[{r}]/all_reduce", inputs=[g[r]], params=[(peers[k], r, r + 1)]) for r in rows]
        k += 1
        if down_proj_twice and s == 0:                 # the second issue of the SAME module path in the step (peers_6 = its received partial)
            g2 = [b.emit(f"Gemm_v1{{K={H // world},N={H}}}", f"model.model.layers.0.mlp.down_proj/triton_launch_b[{r}]", inputs=[ar2[r]]) for r in rows]
            ar2 = [b.emit(f"AllReduce2_v1{{N={H}}}", f"model.model.layers.0.mlp.down_proj/all_reduce2_b[{r}]/all_reduce", inputs=[g2[r]], params=[(peers[6], r, r + 1)]) for r in rows]
        fin = [b.emit(f"RMSNormTriton_v1{{N={H},{EPS}}}", f"model.model.norm/triton_launch[{r}]", inputs=[ar2[r]],
                      params=([(peers[1], 0, 1)] if stray_reader else [])) for r in rows]
        lg = b.emit(f"Gemm_v1{{K={H},N={V // world}}}", "model.logits_processor/triton_launch_9[0]", inputs=[fin[-1]])
        ag = b.emit(f"AllGather2_v1{{N={V // world}}}", f"model.logits_processor/all_gather2_{k}[0]/all_gather", inputs=[lg], params=[(peers[k], 0, 1)])
        k += 1
        b.emit(f"TokenSelect_v1{{V={V}}}", f"argmax[{s}]", inputs=[ag])
    return b.program(digest=f"{rank}" * 64, layers=0, attn_obs={}, tp={"world": world, "rank": rank})


def test_a_rank_programs_received_peer_partials_are_named_by_the_peer_ranks_identity_never_left_unnamed():
    """[R17 finding 7, TP2 B1 rung (1d)] the llama TP2 B1 record: identical 16,134 identities and digest on all three sides, but the armed
    traversal flagged the 1088 `peers_<k>` inputs per rank as `unnamed_input` (complete False vs the inventory's True).  The RECEIVED peer
    partial is the peer rank's own partial (`tp_rank_partials` part_rank<1-r> / shard_rank<1-r>) at the same (request, step, module) -- an
    identity the merged population already carries: named by RULE (every reader a collective row, world 2), never a new identity."""
    for rank in (0, 1):
        prog = _tp_df(rank)
        tr, t = assert_three_way(prog, q=RV.Query(rank=rank, ranks=2), workload={"sweep": {"row_id": "tp2-df", "batch": 1, "arrive_steps": [0]}})
        ba = t.boundary_audit
        assert ba["available"] and ba["fully_wired"] and t.complete, t.unmodelled
        assert ba["unnamed_input"] == {} and ba["unnamed_crossing"] == {}
        assert {n for n, f in ba["named_input"].items() if f == "tp_rank_partials"} == {f"peers_{k}" for k in range(6)}
        assert ba["named_input"]["prompt"] == "input_token_ids"                # (the toy's Gemms carry no weights operand run)
        pi = ba["peer_inputs"]
        assert pi["world"] == 2 and pi["rank"] == rank and pi["peer_rank"] == 1 - rank and pi["params"] == 6
        assert pi["reader_rows"] == (2 + 2 + 1) + (1 + 1 + 1) and pi["reader_rows_by_kind"] == {"all_reduce": 6, "all_gather": 2}   # READER rows (revb M-0143 b)
        # the population is the SAME identities as before the rule (the alias adds no row): partials + collective outputs per TP-04
        fam = t.populations()["by_family"]
        assert fam["tp_rank_partials"] == 3 * 2 and all(r["rank"] == rank for r in tr)     # 3 collective modules x 2 steps (one identity per module-step)
        # the guard's header line names the run once, with the count
        g = RV.guard_summary(ba, engine="traversal")
        assert g["armed"] and g["flags"] == {"unnamed_crossing": 0, "unnamed_input": 0, "hidden_bn_disagreement": 0}
        assert g["named_input"] == ["peers_<0..5> (6) -> tp_rank_partials", "prompt -> input_token_ids"]
    # a peers input some NON-collective row reads is not the received partial: unnamed by name, the population not complete on it
    t = RV.Traversal(_tp_df(0, stray_reader=True), RV.Query(rank=0, ranks=2))
    list(t)
    assert not t.complete and set(t.boundary_audit["unnamed_input"]) == {"peers_1"}
    assert t.boundary_audit["unnamed_input"]["peers_1"]["readers"] == {"AllReduce2_v1": 2, "RMSNormTriton_v1": 3}
    assert any(k.startswith("input `peers_1`") for k in t.unmodelled)
    assert "peers_1" not in t.boundary_audit["named_input"] and t.boundary_audit["named_input"]["peers_0"] == "tp_rank_partials"
    # no numbered run of 1-2 params is compressed; a run of 3+ is one line
    assert RV.named_input_summary({"peers_0": "tp_rank_partials", "peers_1": "tp_rank_partials", "prompt": "input_token_ids"}) == \
        ["peers_0 -> tp_rank_partials", "peers_1 -> tp_rank_partials", "prompt -> input_token_ids"]


def test_the_tp_merge_binds_every_collective_output_to_the_peers_partial_identity_for_identity_and_names_a_missing_one():
    """[R17 finding 7] `merge_manifests` of the rank parts: every AllReduce2/AllGather2 output on rank r has the peer's own partial on rank
    1-r at the same (request, step, invocation, module), and every partial has its collective output there (both directions); a missing one
    is named and the merge is not complete.  Both engines' parts bind identically (the identities are the same)."""
    wl = {"sweep": {"row_id": "tp2-df", "batch": 1, "arrive_steps": [0]}}
    for eng in ("inventory", "traversal"):
        parts = [RM.build_manifest(_tp_df(rk), wl, rank=rk, ranks=2, engine=eng) for rk in (0, 1)]
        row = RM.merge_manifests(parts)
        b = row["tp_peer_binding"]
        assert b["world"] == 2 and b["collective_outputs"] == 2 * 3 * 2 and b["peer_partials"] == 2 * 3 * 2     # ranks x modules x steps
        assert b["bound"] == b["collective_outputs"] and b["unbound"] == [] and b["n_unbound"] == 0 and row["complete"]
        assert row["guard"]["armed"] == (eng == "traversal")
        if eng == "traversal":
            assert row["guard"]["named_input"] == ["peers_<0..5> (6) -> tp_rank_partials", "prompt -> input_token_ids"]
    # NEGATIVE (forward): rank 1 lacks the down_proj partial at step 1 -> rank 0's all_reduce output there has no peer identity
    parts = [RM.build_manifest(_tp_df(rk), wl, rank=rk, ranks=2, engine="traversal") for rk in (0, 1)]
    bad1 = copy.deepcopy(parts[1])
    bad1["identities"] = [r for r in bad1["identities"] if not (r["family"] == "tp_rank_partials" and r["op_path"] == "model.layers.0.mlp.down_proj" and r["step"] == 1)]
    row = RM.merge_manifests([parts[0], bad1])
    assert not row["complete"] and row["tp_peer_binding"]["n_unbound"] == 1
    assert row["tp_peer_binding"]["unbound"] == ["rank 0 model.layers.0.mlp.down_proj step 1 all_reduce output: no part_rank1 identity on rank 1 (request r0)"]
    assert any(k.startswith("tp peer binding: 1 collective output(s)") for k in row["unmodelled"])
    # NEGATIVE (reverse): rank 0 lacks the logits all_gather OUTPUT at step 0 -> rank 1's shard there has no collective output on rank 0
    bad0 = copy.deepcopy(parts[0])
    bad0["identities"] = [r for r in bad0["identities"] if not (r.get("spec") == "AllGather2_v1" and r["step"] == 0)]
    row = RM.merge_manifests([bad0, parts[1]])
    assert not row["complete"] and row["tp_peer_binding"]["unbound"] == ["rank 1 logits_processor step 0 partial shard_rank1: no collective output identity on rank 0 (request r0)"]
    # [revb M-0143 (a)] the RANGE binds too: an all_reduce partial narrower than its collective output is named (the key alone would bind it)
    assert row["tp_peer_binding"]["width_checked"] > 0
    narrow = copy.deepcopy(parts[1])
    part = next(r for r in narrow["identities"] if r["family"] == "tp_rank_partials" and r["op_path"] == "model.layers.0.mlp.down_proj" and r["step"] == 1)
    lo, hi = part["element_range"]
    part["element_range"] = [lo, lo + (hi - lo) // 2]
    row = RM.merge_manifests([parts[0], narrow])
    assert not row["complete"] and row["tp_peer_binding"]["n_unbound"] == 1
    assert row["tp_peer_binding"]["unbound"][0].startswith(f"rank 0 model.layers.0.mlp.down_proj step 1 all_reduce output width {hi - lo}: part_rank1 on rank 1 is {(hi - lo) // 2} elements, not {hi - lo}")
    # and the all_gather rule is 1/world: the shard's width is the gathered output's over the world (holds on the fixture: both sides sized)
    good = RM.merge_manifests(parts)["tp_peer_binding"]
    assert good["width_checked"] == good["bound"] == good["collective_outputs"]
    # a TP1 merge of request components has no binding (not a TP row)
    assert RM.merge_manifests([RM.build_manifest(dense_df(), None, request_id="r0"), RM.build_manifest(dense_df(), None, request_id="r1")])["tp_peer_binding"] is None


def test_a_world_other_than_two_leaves_the_peer_inputs_unnamed_and_the_population_incomplete_by_name():
    """[revb M-0143 (c)] the `tp_rank_partials` naming rule is written for world 2 (the peer is 1 - r; the rows the sweep has).  A rank
    Program of a wider world with `peers_<k>` inputs read by collectives: the rule does NOT apply, the inputs stay `unnamed_input`, the
    population is not complete by name -- never a silent guess at which peer a received partial came from.  The inventory (no dataflow) is
    blind and complete: exactly the differential the one-release cross-check turns into a row failing by name."""
    prog = _tp_df(0, world=4)
    t = RV.Traversal(copy.deepcopy(prog), RV.Query(rank=0, ranks=4))
    ids = list(t)
    ba = t.boundary_audit
    assert not t.complete and set(ba["unnamed_input"]) == {f"peers_{k}" for k in range(6)}
    assert all(set(v["readers"]) <= {"AllReduce2_v1", "AllGather2_v1"} for v in ba["unnamed_input"].values())   # collective readers, still unnamed
    assert ba["peer_inputs"]["world"] == 4 and ba["peer_inputs"]["peer_rank"] is None and ba["peer_inputs"]["params"] == 0
    assert not any(f == "tp_rank_partials" for f in ba["named_input"].values())
    assert sum(k.startswith("input `peers_") for k in t.unmodelled) == 6
    assert len(ids) > 0 and RV.guard_summary(ba, engine="traversal")["flags"]["unnamed_input"] == 6
    wl = {"sweep": {"row_id": "tp4-df", "batch": 1, "arrive_steps": [0]}}
    trav = RM.build_manifest(copy.deepcopy(prog), wl, rank=0, ranks=4, engine="traversal")
    inv = RM.build_manifest(copy.deepcopy(prog), wl, rank=0, ranks=4, engine="inventory")
    assert trav["complete"] is False and inv["complete"] is True and trav["identities"] == inv["identities"]
    cc = RM.cross_check_manifests(trav, inv, primary_engine="traversal", other_engine="inventory")
    assert not cc["agree"] and cc["rows_equal"] and not cc["complete_equal"] and "complete False (traversal) != True (inventory)" in cc["why"]


def test_a_module_path_issued_twice_in_one_step_is_one_identity_and_the_second_issue_fails_at_coverage_by_name_not_at_the_peer_binding():
    """[revb F-r17b-17] `invocation` is NOT a per-step issue ordinal: both engines emit invocation 0 per (module, member, step) and a module
    path issued twice in a step is ONE identity whose element_range is both calls' rows.  Consequences, pinned: (a) the identities are the
    same on both engines (one row, hi = both calls); (b) `tp_peer_binding` sees one partial/output pair per (step, module) and says BOUND;
    (c) the collector commits the module path twice -> `coverage_check` binds the k-th occurrence to invocation k, the first occurrence's
    numel (one call) is < the identity's hi -> the row is MISSING by name.  Fail-closed by the range arithmetic, one stage after the binding
    -- never a silent pass, never two identities the statement would have to invent."""
    from verity_capture.commit.tests.test_required_manifest import _layouts_from
    H = 32
    wl = {"sweep": {"row_id": "tp2-df-twice", "batch": 1, "arrive_steps": [0]}}
    parts_by_engine = {}
    for eng in ("inventory", "traversal"):
        parts_by_engine[eng] = [RM.build_manifest(_tp_df(rk, down_proj_twice=True), wl, rank=rk, ranks=2, engine=eng) for rk in (0, 1)]
    assert [p["identities"] for p in parts_by_engine["inventory"]] == [p["identities"] for p in parts_by_engine["traversal"]]
    parts = parts_by_engine["traversal"]
    # (a) ONE identity per (step 0, down_proj, member), invocation 0, range = both issues' rows (2 rows x 2 calls x H)
    outs = [r for r in parts[0]["identities"] if r["op_path"] == "model.layers.0.mlp.down_proj" and r["step"] == 0 and r["family"] == "instance_outputs"]
    prts = [r for r in parts[0]["identities"] if r["op_path"] == "model.layers.0.mlp.down_proj" and r["step"] == 0 and r["family"] == "tp_rank_partials"]
    assert len(outs) == 1 and len(prts) == 1
    assert outs[0]["invocation"] == 0 and outs[0]["element_range"] == [0, 2 * 2 * H] and outs[0]["rows"] == 4
    assert prts[0]["invocation"] == 0 and prts[0]["element_range"] == [0, 2 * 2 * H]
    once = [r for r in RM.build_manifest(_tp_df(0), wl, rank=0, ranks=2, engine="traversal")["identities"]
            if r["op_path"] == "model.layers.0.mlp.down_proj" and r["step"] == 0 and r["family"] == "instance_outputs"]
    assert once[0]["element_range"] == [0, 2 * H] and once[0]["invocation"] == 0                          # the single-issue Program, for scale
    assert all(r["invocation"] == 0 for p in parts for r in p["identities"] if r["family"] != "weights")   # no ordinal anywhere on this Program
    assert parts[0]["complete"] and parts[1]["complete"]                                                     # peers_6 is named by the rule too
    # (b) the peer binding is by (step, module): one pair, bound, width-checked -- it cannot see the second issue
    row = RM.merge_manifests(parts)
    b = row["tp_peer_binding"]
    assert row["complete"] and b["n_unbound"] == 0 and b["bound"] == b["collective_outputs"] == b["width_checked"] == 2 * 3 * 2
    assert "issued TWICE in one step is ONE identity" in b["rule"] and "fails one stage later at `coverage_check`" in b["rule"]
    # (c) the collector commits the module path TWICE in step 0 (each tensor = one call's rows): coverage names the identity missing
    lay = _layouts_from(parts[0])
    twice = []
    for m in lay[0]:
        if m["name"] in ("model.layers.0.mlp.down_proj/0", "model.layers.0.mlp.down_proj/part_rank0"):   # the all_reduce output (tuple slot 0) + own partial
            twice += [dict(m, numel=m["numel"] // 2), dict(m, numel=m["numel"] // 2)]        # the two committed occurrences, one call each
        else:
            twice.append(m)
    lay[0] = twice
    cov = RM.coverage_check(parts[0], lay, request_id="r0", rank=0)
    assert not cov["ok"] and cov["missing_n"] == 2
    miss = {(m["family"], m["output_member"]): m for m in cov["missing"]}
    assert set(miss) == {("instance_outputs", "0"), ("tp_rank_partials", "part_rank0")}
    for m in miss.values():
        assert m["op_path"] == "model.layers.0.mlp.down_proj" and m["step"] == 0 and m["invocation"] == 0
        assert m["want_range"] == [0, 4 * H] and m["committed_numel"] == [2 * H]       # the FIRST occurrence's numel (< hi) is what bound to invocation 0
    # the single-issue collector covers the single-issue manifest (the positive), and the twice manifest is NOT covered by one call's tensor
    assert RM.coverage_check(parts[0], _layouts_from(parts[0]), request_id="r0", rank=0)["ok"]


# ---------------------------------------------------------------------------------------------------------------------------------
# (3) a missing required family cannot vanish from both the expectation and the collector
# ---------------------------------------------------------------------------------------------------------------------------------


def test_unmodelled_family_fails_closed_in_expectation_and_coverage():
    """An op family present in P with no naming rule: the traversal yields NO identity for it (never an empty one), records
    UnmodelledFamily, `complete` is False, the manifest's coverage_check refuses even a collector that committed every other identity,
    and the reference agrees.  The same on a workload (one bad component poisons the fold, named)."""
    prog = dense_df()
    for r in prog["instances"]["rows"]:
        if r["name"].startswith("model.model.layers.1.mlp.act_fn/"):
            r["spec"] = "GeGLU_v9{I=64}"
    tr, t = assert_three_way(prog)
    assert not t.complete and t.unmodelled == {"GeGLU_v9 under model.layers.1.mlp.act_fn": 4}
    assert not any(r["op_path"] == "model.layers.1.mlp.act_fn" for r in tr)
    assert not any(r["element_range"][1] == 0 for r in tr)
    assert t.boundary_audit["families"]["GeGLU_v9"] == {"unmodelled": {"rows": 4, "crossing": 4}}
    man = RV.manifest_from_traversal(copy.deepcopy(prog))
    res = RM.coverage_check(man, _layouts_from(man))
    assert res["ok"] is False and "GeGLU_v9 under model.layers.1.mlp.act_fn" in res["error"] and res["families_missing_entirely"] == ["GeGLU_v9 under model.layers.1.mlp.act_fn"]
    fixture = {"fixture": "t", "requests": [{"request_id": "r0", "index": 0, "prompt_len": 3, "max_tokens": 2, "arrive_step": 0},
                                             {"request_id": "r1", "index": 1, "prompt_len": 3, "max_tokens": 2, "arrive_step": 0}]}
    g = RV.global_manifest_from_traversal(fixture, {"r0": dense_df(), "r1": copy.deepcopy(prog)})
    assert g["complete"] is False and "GeGLU_v9 under model.layers.1.mlp.act_fn" in g["unmodelled"] and not RM.coverage_check(g, {})["ok"]
    # the vocabulary itself raises (no silent []), so a traversal rule table cannot grow an implicit "skip"
    with pytest.raises(RM.UnmodelledFamily):
        RM.members_for("GeGLU_v9", {"I": 64}, "model.model.layers.1.mlp.act_fn", "")


def test_dataflow_audit_promotes_a_crossing_family_the_vocabulary_calls_interior():
    """The second gate: suppose a family is (wrongly) listed as interior -- by the vocabulary alone the manifest would be complete and a
    collector binding nothing of it would pass coverage.  The Program's own dataflow still says its output is read by ANOTHER module.
    [R19 int-20] The gate used to be a refusal (`unnamed_crossing`, complete:false, `audit` exits 1); it is now a PROMOTION: the crossing
    rows become a required member (`out`) of their module, class `interior_crossing`, width / dtype from the registry Definition's output
    (MeanTriton returns one f32 per row), the reading module recorded as its consumer -- the manifest is complete and a collector binding
    nothing of it FAILS coverage, because the member is required.  Here `MeanTriton_v1` (an interior norm-chain op) is wired as a module
    output; all three engines emit the promoted identity (row for row: the same pass)."""
    b = DF(LP=2, T=0)
    H, V = b.H, b.V
    e = [b.emit(f"Embedding_v1{{V={V},H={H}}}", f"model.model.embed_tokens/embedding[{r}]", params=[(0, r, r + 1)]) for r in range(2)]
    mean = [b.emit(f"MeanTriton_v1{{N={H}}}", f"model.model.layers.0.input_layernorm/mean[{r}]", inputs=[e[r]]) for r in range(2)]
    g = [b.emit(f"Gemm_v1{{K={H},N={H}}}", f"model.model.layers.0.mlp.down_proj/triton_launch[{r}]", inputs=[mean[r]]) for r in range(2)]
    lg = b.emit(f"Gemm_v1{{K={H},N={V}}}", "model.logits_processor/triton_launch_9[0]", inputs=[g[-1]])
    b.emit(f"TokenSelect_v1{{V={V}}}", "argmax[0]", inputs=[lg])
    prog = b.program(layers=0, attn_obs={})
    tr, t = traversal_rows(prog)
    pm = [r for r in tr if r["op_path"] == "model.layers.0.input_layernorm"]
    assert len(pm) == 1 and pm[0]["output_member"] == "out" and pm[0]["promoted"] is True and pm[0]["boundary_class"] == "interior_crossing", pm
    assert pm[0]["spec"] == "MeanTriton_v1" and pm[0]["rows"] == 2 and pm[0]["width"] == 1 and pm[0]["dtype"] == "f32" and pm[0]["element_range"] == [0, 2]
    assert pm[0]["consumers"] == ["model.layers.0.mlp.down_proj"]
    assert t.complete and t.boundary_audit["unnamed_crossing"] == {} and t.boundary_audit["promoted"] == {"MeanTriton_v1": 2}, t.boundary_audit
    assert t.boundary_audit["families"]["MeanTriton_v1"] == {"interior_crossing": {"rows": 2, "crossing": 2}}
    # the inventory (in-memory dataflow pass) and the traversal emit the same identities; the manifest is complete on both
    old = RM.build_manifest(copy.deepcopy(prog), {"sweep": {"arrive_steps": [0]}}, engine="inventory")
    new = RV.manifest_from_traversal(copy.deepcopy(prog), {"sweep": {"arrive_steps": [0]}})
    assert old["complete"] is True and new["complete"] is True and new["identities"] == old["identities"]
    assert any(r.get("promoted") for r in old["identities"]) and old["query"]["promotion"]["rows"] == 2
    # the second gate, kept: a collector's layouts that cover everything BUT the promoted member do not pass coverage
    lay = _layouts_from(new)
    assert RM.coverage_check(new, lay)["ok"]
    without = {s: [m for m in ms if m["name"] != "model.layers.0.input_layernorm"] for s, ms in lay.items()}
    assert sum(len(ms) for ms in without.values()) == sum(len(ms) for ms in lay.values()) - 1
    assert not RM.coverage_check(new, without)["ok"]
    _ref, summ = RV.reference_required_values(copy.deepcopy(prog), RV.Query())
    assert summ["complete"] and summ["promotion"]["rows"] == 2 and summ["promotion"]["faults"] == {}, summ.get("promotion")
    # the same Program where the mean stays inside its module (read by a same-module op only): no finding
    b2 = DF(LP=2, T=0)
    e = [b2.emit(f"Embedding_v1{{V={V},H={H}}}", f"model.model.embed_tokens/embedding[{r}]", params=[(0, r, r + 1)]) for r in range(2)]
    mean = [b2.emit(f"MeanTriton_v1{{N={H}}}", f"model.model.layers.0.input_layernorm/mean[{r}]", inputs=[e[r]]) for r in range(2)]
    nrm = [b2.emit(f"RMSNormTriton_v1{{N={H},{EPS}}}", f"model.model.layers.0.input_layernorm/triton_launch[{r}]", inputs=[mean[r], e[r]]) for r in range(2)]
    g = [b2.emit(f"Gemm_v1{{K={H},N={H}}}", f"model.model.layers.0.mlp.down_proj/triton_launch[{r}]", inputs=[nrm[r]]) for r in range(2)]
    lg = b2.emit(f"Gemm_v1{{K={H},N={V}}}", "model.logits_processor/triton_launch_9[0]", inputs=[g[-1]])
    b2.emit(f"TokenSelect_v1{{V={V}}}", "argmax[0]", inputs=[lg])
    tr2, t2 = traversal_rows(b2.program(layers=0, attn_obs={}))
    assert t2.complete and t2.boundary_audit["unnamed_crossing"] == {} and t2.boundary_audit["families"]["MeanTriton_v1"] == {"interior": {"rows": 2, "crossing": 0}}


def test_audit_cli_exit_codes_and_streaming_reader(tmp_path):
    """`iter_rows` streams the rows of instances.json.gz (never the list); `ProgramView.streaming` traverses to the same identities as
    the in-memory view; the `audit` CLI exits 0 on a clean Program and 1 when the audit names an unnamed crossing or the population is
    incomplete; the `diff` CLI exits 0 iff old == new."""
    import subprocess
    import sys

    def write(prog: dict, d):
        d.mkdir()
        with gzip.open(d / "instances.json.gz", "wt") as f:
            json.dump(prog["instances"], f)
        (d / "result.json").write_text(json.dumps(prog["result"]))
        (d / "artifact.json").write_text(json.dumps(prog["artifact"]))
        return d

    prog = dense_df(LP=4, T=2)
    d = write(prog, tmp_path / "ok")
    assert [r["i"] for r in RV.iter_rows(str(d))] == [r["i"] for r in prog["instances"]["rows"]]
    assert list(RV.iter_rows(str(d))) == prog["instances"]["rows"]
    v = RV.ProgramView.streaming(str(d))
    assert v.digest == prog["artifact"]["program_digest"] and v.params == [tuple(p) for p in prog["instances"]["params"]]
    st = rows_of(list(RV.Traversal(v)))
    mem, _t = traversal_rows(prog)
    assert st == mem
    env = {"PYTHONPATH": ".:vllm-poc"}
    import os
    env = dict(os.environ, **env)
    run = lambda *args: subprocess.run([sys.executable, "-m", "verity_capture.commit.required_values", *args], capture_output=True, text=True, env=env, cwd=os.getcwd())
    r = run("audit", "--program", str(d), "--out", str(tmp_path / "a.json"))
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    rec = json.loads((tmp_path / "a.json").read_text())
    assert rec["complete"] and rec["boundary_audit"]["unnamed_crossing"] == {} and rec["identities"] == len(mem)
    r = run("diff", "--program", str(d), "--out", str(tmp_path / "d.json"))
    assert r.returncode == 0 and json.loads((tmp_path / "d.json").read_text())["equal"] is True
    bad = copy.deepcopy(prog)
    for row in bad["instances"]["rows"]:
        if row["name"].startswith("model.model.layers.1.mlp.act_fn/"):
            row["spec"] = "GeGLU_v9{I=64}"
    d2 = write(bad, tmp_path / "bad")
    assert run("audit", "--program", str(d2)).returncode == 1


GEMMA_VARIANTS = {"default": {}, "no_hint_narrow": dict(hint_narrow=False), "two_hinted": dict(two_hinted=True), "no_cap": dict(softcap_logits=False),
                  "unhinted_cap": dict(hint_cap=False), "cap_width_replace": dict(cap_width=8), "swap_narrows": dict(swap_narrows=True),
                  "cross_operands": dict(cross_operands=True), "narrow_param": dict(narrow_operand="param"), "narrow_none": dict(narrow_operand="none"),
                  "LP5_T2_L3": dict(LP=5, T=2, layers=3)}


@pytest.mark.parametrize("variant", sorted(GEMMA_VARIANTS))
def test_gemma_dense_variants_traversal_equals_inventory(variant):
    """The R15 dense fixtures (unfused norm chain with hinted narrowings + producer ordinals / operands, soft-capped logits, sliding
    window, the D1/D2 defects): traversal == inventory on identities, digest, completeness AND the named defects; the reference agrees on
    every complete Program and on every defect's family; on the D2 'replace with another width' defect it keeps the hinted producer where
    the inventory keeps the default claim (both incomplete -> coverage refuses either way; documented, not load-bearing)."""
    from verity_capture.commit.tests.test_required_manifest_dense import _gemma_program
    wl = {"request_ids": [0], "sweep": {"arrive_steps": [0]}}
    prog = _gemma_program(**GEMMA_VARIANTS[variant])
    if variant == "default":
        obs = prog["result"]["attention_impls_observed"]
        obs["model.layers.0.self_attn.attn"]["sliding_window"] = [4095, 0]
        obs["model.layers.1.self_attn.attn"]["sliding_window"] = [-1, -1]
    old = RM.build_manifest(copy.deepcopy(prog), wl)
    new = RV.manifest_from_traversal(copy.deepcopy(prog), wl)
    eq = RV.manifest_equal(old, new)
    assert eq["equal"], eq["why"]
    assert old["unmodelled"] == new["unmodelled"]
    ref, summ = RV.reference_required_values(copy.deepcopy(prog), RV.Query())
    assert summ["complete"] == new["complete"]
    if variant != "cap_width_replace":
        assert _family_keys(summ["unmodelled"]) == _family_keys(new["unmodelled"])
        assert [x.row() for x in ref] == new["identities"]
    else:                                                                       # D2 replace defect: the reference keeps the hinted producer and
        assert not summ["complete"] and "Bf16MulScalar_v1 under logits_processor" in _family_keys(summ["unmodelled"])   # names it; no D1 finding then
    # this fixture wires only the narrowings' args and [R19 int-20] the embed scale's (its `model/mul` rows read the embedding, layer 0's norms
    # read them -- so the audit is armed on every variant, `narrow_none` included: that variant strips the NARROWINGS' operands only): an
    # observed crossing is evidence, `named_interior` is honestly withheld (not fully wired); the one crossing interior family is promoted
    aud = new["boundary_audit"]
    assert aud["available"] and not aud["fully_wired"] and aud["named_interior"] is None and aud["unnamed_crossing"] == {}
    assert aud["promoted"] == {"Bf16MulScalarTensor_v1": GEMMA_VARIANTS[variant].get("LP", 3) + GEMMA_VARIANTS[variant].get("T", 1)}
    assert old["query"]["promotion"] == new["query"]["promotion"] and old["query"]["promotion"]["faults"] == {}


def test_stochastic_and_reordered_programs_traversal_equals_inventory():
    from verity_capture.commit.tests.test_required_manifest_dense import _gemma_program
    from verity_capture.commit.tests.test_required_manifest_stoch import _stochastic_program
    old, new = RM.build_manifest(_stochastic_program()), RV.manifest_from_traversal(_stochastic_program())
    assert RV.manifest_equal(old, new)["equal"] and "sampler_seed" in new["query"]["required_families"]
    wl = {"request_ids": [0], "sweep": {"arrive_steps": [0]}}
    for kw in ({}, dict(cap_width=8)):
        prog = _gemma_program(**kw)
        rows = prog["instances"]["rows"]
        idx_mul = next(k for k, r in enumerate(rows) if r["name"].startswith("model.logits_processor/mul_2/out"))
        idx_gemm = next(k for k, r in enumerate(rows) if r["name"].startswith("model.logits_processor/triton_launch_9["))
        rows.insert(idx_gemm, rows.pop(idx_mul))
        for k, r in enumerate(rows):
            r["i"] = k
        old, new = RM.build_manifest(copy.deepcopy(prog), wl), RV.manifest_from_traversal(copy.deepcopy(prog), wl)
        eq = RV.manifest_equal(old, new)
        assert eq["equal"] and old["unmodelled"] == new["unmodelled"], eq["why"]


def test_traversal_is_lazy_and_single_pass():
    """Identities of step 0 are yielded before the rows of step 1 are consumed; the rows are read exactly once; a second iteration is refused."""
    prog = dense_df(LP=3, T=2)
    rows = prog["instances"]["rows"]
    first_select = next(r["i"] for r in rows if r["spec"].startswith("TokenSelect"))
    consumed = {"n": 0}

    def gen():
        for r in rows:
            consumed["n"] += 1
            yield r
    v = RV.ProgramView.of(prog)
    v.rows = gen()
    t = RV.Traversal(v)
    it = iter(t)
    out = []
    while True:
        x = next(it)
        out.append(x)
        if x.family == "sampled_token_ids":
            break
    assert consumed["n"] == first_select + 1                                     # exactly the rows of step 0 (+ its sampling event)
    step0 = [x for x in out if x.family != "weights"]
    assert step0 and all(x.step == 0 for x in step0)
    rest = list(it)
    assert consumed["n"] == len(rows) and t.steps == 3
    with pytest.raises(RuntimeError):
        list(t)
    # the same identities as the eager path
    assert rows_of(out + rest) == traversal_rows(prog)[0]


# ----------------------------------------------------------------------------------------------------------------------------------------
# unit 5: the flag (default OFF) -- `required_manifest.build_manifest(engine=...)` / VERITY_REQUIRED_VALUES_ENGINE / --engine
# ----------------------------------------------------------------------------------------------------------------------------------------

def _write_program_dir(tmp_path, prog: dict) -> str:
    d = tmp_path / "build_request"
    d.mkdir()
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump(prog["instances"], f)
    (d / "result.json").write_text(json.dumps(prog["result"]))
    if prog.get("artifact") is not None:
        (d / "artifact.json").write_text(json.dumps(prog["artifact"]))
    return str(d)


def test_streamed_global_fold_equals_loaded_and_the_build_global_cli_under_the_flag(tmp_path):
    """The traversal engine streams the component Programs (`ProgramView.streaming(dir)`; the CLI passes dirs): same identities and digest
    as the loaded fold and as the inventory's build-global, with the rows never all in memory (Llama B8 offline: 58 MB vs 1 GB peak)."""
    import subprocess
    import sys
    fixture = {"fixture": "t_b2", "requests": [{"request_id": "a", "index": 0, "prompt_len": 4, "max_tokens": 2, "arrive_step": 0},
                                                {"request_id": "b", "index": 1, "prompt_len": 3, "max_tokens": 3, "arrive_step": 1}]}
    progs = {"a": dense_df(LP=4, T=1), "b": dense_df(LP=3, T=2)}
    root = tmp_path / "programs"
    dirs = {}
    for rid, prog in progs.items():
        d = root / f"build_request_LP{prog['result']['LP']}_T{prog['result']['T']}"
        d.mkdir(parents=True)
        with gzip.open(d / "instances.json.gz", "wt") as f:
            json.dump(prog["instances"], f)
        (d / "result.json").write_text(json.dumps(prog["result"]))
        dirs[rid] = str(d)
    loaded = RV.global_manifest_from_traversal(fixture, copy.deepcopy(progs), workload_digest="w" * 64)
    streamed = RV.global_manifest_from_traversal(fixture, dirs, workload_digest="w" * 64)                       # dirs -> ProgramView.streaming
    views = RV.global_manifest_from_traversal(fixture, {rid: RV.ProgramView.streaming(d) for rid, d in dirs.items()}, workload_digest="w" * 64)
    inventory = RM.build_global_manifest(fixture, copy.deepcopy(progs), workload_digest="w" * 64)
    assert loaded["manifest_digest"] == streamed["manifest_digest"] == views["manifest_digest"] == inventory["manifest_digest"]
    assert loaded["identities"] == streamed["identities"] == views["identities"]
    assert RM._program_dirs_for(fixture, str(root)) == dirs
    fx = tmp_path / "fx.json"
    fx.write_text(json.dumps(fixture))
    env = {**__import__("os").environ, "PYTHONPATH": ".:vllm-poc"}
    env.pop(RM.ENGINE_ENV, None)
    outs = {}
    for eng in ("inventory", "traversal"):
        out = tmp_path / f"g_{eng}.json"
        r = subprocess.run([sys.executable, "-m", "verity_capture.commit.required_manifest", "--engine", eng, "build-global", "--fixture", str(fx),
                            "--programs-root", str(root), "--workload-digest", "w" * 64, "--out", str(out)], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr + r.stdout
        outs[eng] = json.loads(out.read_text())
    assert outs["inventory"]["manifest_digest"] == outs["traversal"]["manifest_digest"] == inventory["manifest_digest"]
    assert "engine" not in outs["inventory"] and outs["traversal"]["engine"]["schema"] == RV.SCHEMA
    assert outs["traversal"]["engine"]["identity_keys"]["rows"] == inventory["populations"]["identities"]
    # the consumer side (`commit_delta --required-manifest` -> binding.load_required_manifest): the traversal-built FILE passes the binder's
    # schema / program_digest / recomputed-digest refusals and expands to the identical population; the extra `engine` block is ignored
    from verity_capture.commit import binding as B
    pops = {}
    for eng in ("inventory", "traversal"):
        pop, hdr = B.load_required_manifest(str(tmp_path / f"g_{eng}.json"), outs[eng]["program_digest"], requests=["a", "b"], steps=len(outs[eng]["steps"]),
                                            ranks=[0], arrivals={"a": 0, "b": 1})
        assert hdr["manifest_digest"] == inventory["manifest_digest"] and hdr["complete"] and not hdr["unmodelled"]
        pops[eng] = sorted(map(repr, pop))
    assert pops["inventory"] == pops["traversal"] and len(pops["traversal"]) > 0


def test_engine_flag_default_is_the_traversal_for_tp1_and_the_inventory_is_a_cache_of_the_same_manifest(monkeypatch):
    """[R17 reqval task 3 -- the flip, r17-int-2] the default engine for a TP1 Program is the traversal (promotion ladder (1a)-(1d) identical:
    out/gen/r17/REQUIRED-VALUES-PROMOTION.md; revb S-r17b-02 / M-0099).  The inventory is reached by name (`engine=` / the env), produces
    the same identities and the same digest, and is what the one-release cross-check computes beside the traversal (next test).  A RANK
    Program (ranks > 1) takes the same default: rung (1d) TP2 B1 was posted (M-0124, finding 7) and revb reproduced it and extended
    S-r17b-02 to the TP paths (M-0143)."""
    monkeypatch.delenv(RM.ENGINE_ENV, raising=False)
    prog = dense_df(layers=2)
    wl = {"sweep": {"row_id": "flag", "batch": 1, "arrive_steps": [0]}}
    assert RM.manifest_engine() == "traversal" and RM.DEFAULT_ENGINE == "traversal" and not RM.engine_explicit()
    assert RM.manifest_engine("inventory") == "inventory" and RM.engine_explicit("inventory")
    default = RM.build_manifest(copy.deepcopy(prog), wl)
    trav = RM.build_manifest(copy.deepcopy(prog), wl, engine="traversal")
    inv = RM.build_manifest(copy.deepcopy(prog), wl, engine="inventory")
    assert default == trav and default["engine"]["traversal"].endswith("required_values.Traversal") and "boundary_audit" in default
    assert "engine" not in inv and "boundary_audit" not in inv                                       # the inventory's output is byte-for-byte what R15 wrote
    assert trav["manifest_digest"] == inv["manifest_digest"] and trav["identities"] == inv["identities"] and RV.manifest_equal(inv, trav)["equal"]
    assert default["guard"]["armed"] is True and inv["guard"]["armed"] is False                      # [F-r17b-09] the guard is armed on the default
    monkeypatch.setenv(RM.ENGINE_ENV, "inventory")
    assert RM.build_manifest(copy.deepcopy(prog), wl) == inv and RM.manifest_engine() == "inventory" and RM.engine_explicit()
    assert RM.build_manifest(copy.deepcopy(prog), wl, engine="traversal") == trav                   # an explicit engine beats the env
    monkeypatch.setenv(RM.ENGINE_ENV, "sometimes")
    with pytest.raises(ValueError, match="not one of"):
        RM.build_manifest(copy.deepcopy(prog), wl)
    monkeypatch.delenv(RM.ENGINE_ENV)
    # the global adapter takes the same default
    fx = {"fixture": "t_b2", "requests": [{"request_id": "r0", "index": 0, "prompt_len": 4, "max_tokens": 2, "arrive_step": 0},
                                          {"request_id": "r1", "index": 1, "prompt_len": 3, "max_tokens": 3, "arrive_step": 1}]}
    progs = {"r0": dense_df(LP=4, T=1), "r1": dense_df(LP=3, T=2)}
    g_def = RM.build_global_manifest(fx, copy.deepcopy(progs), workload_digest="w" * 64)
    g_inv = RM.build_global_manifest(fx, copy.deepcopy(progs), workload_digest="w" * 64, engine="inventory")
    g_trav = RM.build_global_manifest(fx, copy.deepcopy(progs), workload_digest="w" * 64, engine="traversal")
    assert g_def == g_trav and g_inv["manifest_digest"] == g_trav["manifest_digest"] and "engine" in g_trav and "engine" not in g_inv
    # a rank Program takes the default too (rank 0 of 2, wired peers: the toy TP fixture): the traversal, guard armed, peers named by rule
    tp0 = RM.build_manifest(_tp_df(rank=0), wl, rank=0, ranks=2)
    assert "engine" in tp0 and tp0["ranks"] == 2 and tp0["guard"]["armed"] and tp0["complete"]
    assert tp0 == RM.build_manifest(_tp_df(rank=0), wl, rank=0, ranks=2, engine="traversal")
    assert tp0["identities"] == RM.build_manifest(_tp_df(rank=0), wl, rank=0, ranks=2, engine="inventory")["identities"]


def test_the_one_release_cross_check_computes_both_engines_and_fails_the_row_by_name_when_they_disagree(monkeypatch):
    """[R17 task 3] `cross_check_manifests`: the traversal's manifest (of record) beside the inventory's -- agree on every row, the digest,
    `complete`, the unmodelled families.  A disagreement names the first differing row and the sides; the CLI writes no manifest then."""
    monkeypatch.delenv(RM.ENGINE_ENV, raising=False)
    prog = dense_df(layers=2)
    wl = {"sweep": {"row_id": "cc", "batch": 1, "arrive_steps": [0]}}
    trav = RM.build_manifest(copy.deepcopy(prog), wl, engine="traversal")
    inv = RM.build_manifest(copy.deepcopy(prog), wl, engine="inventory")
    cc = RM.cross_check_manifests(trav, inv, primary_engine="traversal", other_engine="inventory")
    assert cc["agree"] and cc["rows_equal"] and cc["digest_equal"] and cc["complete_equal"] and cc["first_difference"] is None and "why" not in cc
    assert cc["identities"] == {"traversal": len(trav["identities"]), "inventory": len(inv["identities"])} and cc["identities"]["traversal"] > 0
    assert cc["digests"]["traversal"] == cc["digests"]["inventory"] and "one release" in cc["rule"]
    # one row's annotation differs (the inventory side edited): the check names index, both rows, the digest, and fails
    bad = copy.deepcopy(inv)
    bad["identities"][3]["rows"] = 999
    cc2 = RM.cross_check_manifests(trav, bad, primary_engine="traversal", other_engine="inventory")
    assert not cc2["agree"] and not cc2["rows_equal"] and cc2["first_difference"]["index"] == 3
    assert cc2["first_difference"]["inventory"]["rows"] == 999 and cc2["first_difference"]["traversal"] == trav["identities"][3]
    assert cc2["why"].startswith("ENGINES DISAGREE") and "rows differ at index 3" in cc2["why"]
    # a missing row: the lengths differ, the first difference is where the two lists part
    short = copy.deepcopy(inv)
    del short["identities"][0]
    cc3 = RM.cross_check_manifests(trav, short, primary_engine="traversal", other_engine="inventory")
    assert not cc3["agree"] and cc3["first_difference"]["index"] == 0 and cc3["identities"]["inventory"] == cc3["identities"]["traversal"] - 1
    # `complete` disagreeing alone fails too (the r16 TP2 shape of finding 7: same rows, one side complete:false)
    inc = copy.deepcopy(inv)
    inc["complete"] = False
    inc["unmodelled"] = {"peers_0: unnamed input": {"rows": 1}}
    cc4 = RM.cross_check_manifests(trav, inc, primary_engine="traversal", other_engine="inventory")
    assert not cc4["agree"] and cc4["rows_equal"] and not cc4["complete_equal"] and not cc4["unmodelled_equal"] and "complete True" in cc4["why"]
    assert cc4["unmodelled_families"] == {"traversal": [], "inventory": ["peers_0: unnamed input"]}
    # the switch: "0" disables (never on a row of record), anything else keeps it on
    assert RM.cross_check_enabled()
    monkeypatch.setenv(RM.CROSS_CHECK_ENV, "0")
    assert not RM.cross_check_enabled()
    monkeypatch.setenv(RM.CROSS_CHECK_ENV, "1")
    assert RM.cross_check_enabled()


def test_cli_verify_with_traversal_engine_cross_checks_both_engines(tmp_path):
    """`required_manifest --engine traversal verify` is the pod's integration probe: rebuild with the traversal AND the inventory, `ok` only
    when the used manifest equals both; the record names the engines and carries the boundary audit (without the per-family table)."""
    import subprocess
    import sys
    prog = dense_df(layers=2)
    d = _write_program_dir(tmp_path, prog)
    wl = tmp_path / "wl.json"
    wl.write_text(json.dumps({"sweep": {"row_id": "flag", "batch": 1, "arrive_steps": [0]}}))
    man = tmp_path / "manifest.json"
    env = {**__import__("os").environ, "PYTHONPATH": ".:vllm-poc"}
    env.pop(RM.ENGINE_ENV, None)
    r = subprocess.run([sys.executable, "-m", "verity_capture.commit.required_manifest", "build", "--program", d, "--workload", str(wl), "--out", str(man)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr + r.stdout
    used = json.loads(man.read_text())
    # [R17 task 3] the default build is the traversal's, WITH the one-release cross-check on the header (both computed, they agreed)
    assert used["engine"]["schema"] == RV.SCHEMA and used["cross_check"]["agree"] and used["cross_check"]["primary_engine"] == "traversal"
    assert used["cross_check"]["digests"]["inventory"] == used["cross_check"]["digests"]["traversal"] == used["manifest_digest"]
    assert used["cross_check"]["identities"]["inventory"] == used["cross_check"]["identities"]["traversal"] == len(used["identities"])
    # explicit `--engine inventory` builds the R15 manifest verbatim (no engine, no cross-check: one side was asked for) -- same digest
    man_inv = tmp_path / "manifest_inv.json"
    r = subprocess.run([sys.executable, "-m", "verity_capture.commit.required_manifest", "--engine", "inventory", "build", "--program", d, "--workload", str(wl),
                        "--out", str(man_inv)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr + r.stdout
    inv = json.loads(man_inv.read_text())
    assert "engine" not in inv and "cross_check" not in inv and inv["manifest_digest"] == used["manifest_digest"] and inv["identities"] == used["identities"]
    out = tmp_path / "verify.json"
    r = subprocess.run([sys.executable, "-m", "verity_capture.commit.required_manifest", "--engine", "traversal", "verify", "--manifest", str(man),
                        "--program", d, "--workload", str(wl), "--out", str(out)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr + r.stdout
    rec = json.loads(out.read_text())
    assert rec["ok"] and rec["rebuilt_engine"] == "traversal" and rec["cross_check"]["agree"]
    assert rec["cross_check"]["inventory_digest"] == rec["cross_check"]["traversal_digest"] == used["manifest_digest"]
    assert rec["cross_check"]["boundary_audit"]["fully_wired"] and "families" not in rec["cross_check"]["boundary_audit"]
    # the env var selects the engine for the module CLI too, and a traversal-built manifest verifies under the default (inventory) engine
    man2 = tmp_path / "manifest_trav.json"
    r = subprocess.run([sys.executable, "-m", "verity_capture.commit.required_manifest", "build", "--program", d, "--workload", str(wl), "--out", str(man2)],
                       capture_output=True, text=True, env={**env, RM.ENGINE_ENV: "traversal"})
    assert r.returncode == 0, r.stderr + r.stdout
    assert json.loads(man2.read_text())["engine"]["schema"] == RV.SCHEMA
    r = subprocess.run([sys.executable, "-m", "verity_capture.commit.required_manifest", "verify", "--manifest", str(man2), "--program", d, "--workload", str(wl),
                        "--out", str(tmp_path / "v2.json")], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and json.loads((tmp_path / "v2.json").read_text())["ok"], r.stderr + r.stdout
    # and the R15 (inventory-built) manifest verifies under the new default: the default `verify` rebuilds with the traversal AND cross-checks
    r = subprocess.run([sys.executable, "-m", "verity_capture.commit.required_manifest", "verify", "--manifest", str(man_inv), "--program", d, "--workload", str(wl),
                        "--out", str(tmp_path / "v3.json")], capture_output=True, text=True, env=env)
    v3 = json.loads((tmp_path / "v3.json").read_text())
    assert r.returncode == 0 and v3["ok"] and v3["rebuilt_engine"] == "traversal" and v3["cross_check"]["agree"], r.stderr + r.stdout


def test_cli_build_under_the_default_fails_by_name_and_writes_no_manifest_when_the_engines_disagree(tmp_path, monkeypatch, capsys):
    """[R17 task 3] the one-release cross-check in the CLI: the inventory side is made to disagree (one row's annotation) -> `build` returns 1,
    writes `<out>.cross_check_fail.json` (the header + the check, no identities) and NO manifest; `--engine traversal` (explicit) skips the
    check and writes; VERITY_REQUIRED_VALUES_CROSS_CHECK=0 skips it too (never set on a row of record)."""
    import sys
    monkeypatch.delenv(RM.ENGINE_ENV, raising=False)
    monkeypatch.delenv(RM.CROSS_CHECK_ENV, raising=False)
    prog = dense_df(layers=2)
    d = _write_program_dir(tmp_path, prog)
    wl = tmp_path / "wl.json"
    wl.write_text(json.dumps({"sweep": {"row_id": "cc", "batch": 1, "arrive_steps": [0]}}))
    real = RM.build_manifest

    def skewed(*a, **kw):
        m = real(*a, **kw)
        if kw.get("engine") == "inventory":
            m["identities"][2]["rows"] = 4242
            m["manifest_digest"] = RM.manifest_digest(m["identities"])
        return m
    monkeypatch.setattr(RM, "build_manifest", skewed)
    man = tmp_path / "manifest.json"
    monkeypatch.setattr(sys, "argv", ["required_manifest", "build", "--program", d, "--workload", str(wl), "--out", str(man)])
    assert RM._main() == 1
    assert not man.exists() and (tmp_path / "manifest.json.cross_check_fail.json").exists()
    fail = json.loads((tmp_path / "manifest.json.cross_check_fail.json").read_text())
    assert "identities" not in fail and not fail["cross_check"]["agree"] and fail["cross_check"]["first_difference"]["index"] == 2
    assert fail["cross_check"]["first_difference"]["inventory"]["rows"] == 4242
    assert "ENGINES DISAGREE" in capsys.readouterr().out
    # explicit engine: one side asked for, no cross-check, the manifest is written
    monkeypatch.setattr(sys, "argv", ["required_manifest", "--engine", "traversal", "build", "--program", d, "--workload", str(wl), "--out", str(man)])
    assert RM._main() == 0 and man.exists() and "cross_check" not in json.loads(man.read_text())
    man.unlink()
    monkeypatch.setenv(RM.CROSS_CHECK_ENV, "0")
    monkeypatch.setattr(sys, "argv", ["required_manifest", "build", "--program", d, "--workload", str(wl), "--out", str(man)])
    assert RM._main() == 0 and man.exists() and "cross_check" not in json.loads(man.read_text())


def derived_s_program(*, LP: int = 3, T: int = 2, name_splits: bool = True) -> dict:
    """stoch M-0254 (derived S, int-3 candidate): params [prompt, seed, splits, weights]; every sampling event is GumbelTopPTokenSelect_v1{V}
    reading the seed (param 1) and ITS step's `splits` entry (param 2, [s, s+1)) as its 6th operand.  `name_splits=False` leaves the
    same reads on a param the vocabulary does not know under another name (the finding names the param, not a string)."""
    prog = dense_df(LP=LP, T=T)
    inst = prog["instances"]
    prompt, weights = inst["params"]
    pname = "splits" if name_splits else "num_splits_per_event"
    inst["params"] = [prompt, ["seed", "Array<1,Value<64>>"], [pname, f"Array<{T + 1},Value<32>>"], weights]
    for r in inst["rows"]:                                                     # weights moved from param 1 to param 3
        for operand in r.get("args") or []:
            for ref in operand:
                if ref[0] == "p" and ref[1] == 1:
                    ref[1] = 3
    s = 0
    for r in inst["rows"]:
        if r["spec"].startswith("TokenSelect_v1"):
            r["spec"] = r["spec"].replace("TokenSelect_v1", "GumbelTopPTokenSelect_v1", 1)
            r["name"] = f"gumbel_topp_token_select{'' if s == 0 else f'[{s}]'}[0]" if "argmax" in r["name"] else r["name"]
            r["args"] = r["args"] + [[["p", 1, 0, 1]], [["p", 2, s, s + 1]]]
            s += 1
    assert s == T + 1
    return prog


def test_a_program_input_no_family_names_is_an_unnamed_input_and_the_population_is_incomplete():
    """A Program input some row reads that NO family names (here the derived-S reads on a param under another name, `num_splits_per_event`):
    the traversal (from the dataflow) names the input and the reading family, and the population is NOT complete on it -- coverage refuses;
    the inventory is blind (complete) -- the differential shows the gap.  (Under its own name `splits` the input IS named: `sampler_splits`,
    next test.)"""
    prog = derived_s_program(name_splits=False)
    wl = {"sweep": {"row_id": "derived-s", "batch": 1, "arrive_steps": [0]}}
    old = RM.build_manifest(copy.deepcopy(prog), wl, engine="inventory")
    assert old["complete"] and "sampler_seed" in old["populations"]["by_family"]          # the seed IS named (R15 stoch): 3 identities
    assert "sampler_splits" not in old["query"]["required_families"]                        # no `splits` param: the family is not emitted
    t = RV.Traversal(copy.deepcopy(prog), RV.Query(arrive_step=0))
    ids = list(t)
    aud = t.boundary_audit
    assert aud["inputs_read"] == {"prompt": 3, "seed": 3, "num_splits_per_event": 3, "weights": aud["inputs_read"]["weights"]}
    assert set(aud["unnamed_input"]) == {"num_splits_per_event"}
    info = aud["unnamed_input"]["num_splits_per_event"]
    assert info["readers"] == {"GumbelTopPTokenSelect_v1": 3} and info["param"] == 2 and info["type"] == "Array<3,Value<32>>"
    assert t.complete is False and any("unnamed input" in k and "num_splits_per_event" in k for k in t.unmodelled)
    # the seed rows are still there (named); the identity rows otherwise equal the inventory's -- only completeness differs
    assert sorted(map(repr, map(RM.canonical_identity, (x.row() for x in ids)))) == sorted(map(repr, map(RM.canonical_identity, old["identities"])))
    new = RV.manifest_from_traversal(copy.deepcopy(prog), wl)
    assert new["complete"] is False and old["complete"] is True and new["manifest_digest"] == old["manifest_digest"]
    lay = _layouts_from(new)
    cov = RM.coverage_check(new, lay, request_id=None, rank=0)
    assert not cov["ok"] and "incomplete" in json.dumps(cov).lower()
    assert RM.coverage_check(old, lay, request_id=None, rank=0)["ok"]                    # the inventory would have let the row pass
    # a wired Program whose params are all named has no finding (the dense fixture)
    t3 = RV.Traversal(dense_df(), RV.Query(arrive_step=0))
    list(t3)
    assert t3.boundary_audit["unnamed_input"] == {} and t3.complete and set(t3.boundary_audit["inputs_read"]) == {"prompt", "weights"}


def test_the_derived_s_splits_input_is_the_sampler_splits_family_in_both_engines_and_the_reference():
    """REQ-03 (stoch M-0254 derived S; global M-0287 'the required family is yours'): a Program with the request-owned `splits` param and a
    stochastic selector emits `sampler_splits` = (step s, invocation 0, runner.sampler, member `splits`, [0, n_seq), i32) beside `sampler_seed`,
    in the inventory, the traversal and the expanded reference alike; the population is then complete (the input is named), coverage binds it
    like the seed; a Program WITHOUT the param (greedy, R15 static-form {V,S}) is byte-for-byte unchanged."""
    prog = derived_s_program()
    wl = {"sweep": {"row_id": "derived-s", "batch": 1, "arrive_steps": [0]}}
    old = RM.build_manifest(copy.deepcopy(prog), wl)
    new = RV.manifest_from_traversal(copy.deepcopy(prog), wl)
    assert RV.manifest_equal(old, new)["equal"] and old["complete"] and new["complete"]
    assert old["populations"]["by_family"]["sampler_splits"] == 3 == old["populations"]["by_family"]["sampler_seed"]
    assert old["query"]["required_families"].index("sampler_splits") == old["query"]["required_families"].index("sampler_seed") + 1
    assert "F-stoch-01" in old["query"]["why"]["sampler_splits"]
    sp = sorted((r["step"], r["output_member"], r["element_range"], r["dtype"], r["spec"]) for r in old["identities"] if r["family"] == "sampler_splits")
    assert sp == [(s, "splits", [0, 1], "i32", "GumbelTopPTokenSelect_v1") for s in range(3)]      # spec = the selector FAMILY (as sampler_seed)
    assert all(r["op_path"] == "runner.sampler" and r["invocation"] == 0 for r in old["identities"] if r["family"] == "sampler_splits")
    # the traversal's audit: `splits` is a NAMED input now (read by the selector only) -- no finding, complete
    t = RV.Traversal(copy.deepcopy(prog), RV.Query(arrive_step=0))
    ids = list(t)
    assert t.complete and t.boundary_audit["unnamed_input"] == {} and t.boundary_audit["inputs_read"]["splits"] == 3
    # the expanded reference agrees identity for identity
    ref, _summary = RV.reference_required_values(copy.deepcopy(prog), RV.Query(arrive_step=0))
    assert rows_of(ids) == rows_of(ref)
    # coverage: a collector that binds runner.sampler/splits per step passes; one that binds the seed but not the splits fails BY NAME
    lay = _layouts_from(new)
    assert RM.coverage_check(new, lay, request_id=None, rank=0)["ok"]
    lay_no_splits = {step: [m for m in ms if not m["name"].endswith("runner.sampler/splits")] for step, ms in lay.items()}
    assert sum(map(len, lay.values())) - sum(map(len, lay_no_splits.values())) == 3
    cov = RM.coverage_check(new, lay_no_splits, request_id=None, rank=0)
    assert not cov["ok"] and "splits" in json.dumps(cov)
    # `splits` read by a row that is NOT the selector: the family is still emitted (the param exists) but the input is not named by it -> incomplete
    prog2 = derived_s_program()
    emb = next(r for r in prog2["instances"]["rows"] if r["spec"].startswith("Embedding"))
    emb["args"] = emb["args"] + [[["p", 2, 0, 1]]]
    t2 = RV.Traversal(prog2, RV.Query(arrive_step=0))
    list(t2)
    assert t2.complete is False and set(t2.boundary_audit["unnamed_input"]) == {"splits"} and "Embedding" in str(t2.boundary_audit["unnamed_input"]["splits"]["readers"])
    # no `splits` param: nothing changes (the R15 static-S stochastic fixture -- on the derived-S tree `_stochastic_program` itself carries
    # the splits param, stoch ef0f86587 -- and the greedy dense fixture)
    from verity_capture.commit.tests.test_required_manifest_stoch import _static_s_program, _stochastic_program
    assert RM.has_splits_param(_stochastic_program()["instances"]["params"]) and not RM.has_splits_param(_static_s_program()["instances"]["params"])
    for pr in (_static_s_program(), dense_df()):
        m = RM.build_manifest(copy.deepcopy(pr))
        assert "sampler_splits" not in m["query"]["required_families"] and m["populations"]["by_family"].get("sampler_splits", 0) == 0
        assert RV.manifest_equal(m, RV.manifest_from_traversal(copy.deepcopy(pr)))["equal"]


def test_cli_verify_a_tp_rank_fold_manifest_of_record_with_repeated_program_and_rank(tmp_path):
    """A TP row's manifest of record is `build --program r0 --rank 0 --program r1 --rank 1 --ranks 2` (per-rank manifests merged with the
    rank rule); `verify` takes the same repeated arguments (B=1 and, with --programs-root, B>=2) and cross-checks both engines."""
    import subprocess
    import sys
    dirs = []
    for rank in (0, 1):
        prog = _tp_program(rank)
        d = tmp_path / f"rank{rank}" / "build_request"
        d.mkdir(parents=True)
        with gzip.open(d / "instances.json.gz", "wt") as f:
            json.dump(prog["instances"], f)
        (d / "result.json").write_text(json.dumps(prog["result"]))
        if prog.get("artifact") is not None:
            (d / "artifact.json").write_text(json.dumps(prog["artifact"]))
        dirs.append(str(d))
    wl = tmp_path / "wl.json"
    wl.write_text(json.dumps({"sweep": {"row_id": "tp2-test", "batch": 1, "arrive_steps": [0]}}))
    env = {**__import__("os").environ, "PYTHONPATH": ".:vllm-poc"}
    env.pop(RM.ENGINE_ENV, None)
    mod = [sys.executable, "-m", "verity_capture.commit.required_manifest"]
    man = tmp_path / "manifest.json"
    r = subprocess.run(mod + ["build", "--program", dirs[0], "--rank", "0", "--program", dirs[1], "--rank", "1", "--ranks", "2", "--workload", str(wl),
                              "--out", str(man), "--compact"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr + r.stdout
    used = json.loads(man.read_text())
    assert used["program_digest_kind"] == "tp-ranks-fold" and used["ranks"] == 2 and {x["rank"] for x in used["identities"]} == {0, 1}
    out = tmp_path / "verify.json"
    r = subprocess.run(mod + ["--engine", "traversal", "verify", "--manifest", str(man), "--program", dirs[0], "--rank", "0", "--program", dirs[1], "--rank", "1",
                              "--ranks", "2", "--workload", str(wl), "--out", str(out)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr + r.stdout
    rec = json.loads(out.read_text())
    assert rec["ok"] and rec["cross_check"]["agree"] and rec["rebuilt_engine"] == "traversal"
    assert rec["cross_check"]["inventory_digest"] == rec["rebuilt_digest"] == used["manifest_digest"] and rec["rebuilt_program_digest"] == used["program_digest"]
    # one rank Program alone is not the manifest of record of a TP2 row: the rank-0 rebuild is compared and refused by name (fewer
    # identities, another program digest) -- a per-rank manifest is what a single --program legitimately verifies (below)
    r = subprocess.run(mod + ["verify", "--manifest", str(man), "--program", dirs[0], "--rank", "0", "--ranks", "2", "--workload", str(wl)],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "not the manifest of record" in (r.stderr + r.stdout), r.stderr + r.stdout
    # two Programs both declared rank 0 for a TP2 world: refused before any rebuild
    r = subprocess.run(mod + ["verify", "--manifest", str(man), "--program", dirs[0], "--rank", "0", "--program", dirs[1], "--rank", "0", "--ranks", "2", "--workload", str(wl)],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "one --program per rank" in (r.stderr + r.stdout), r.stderr + r.stdout
    # the single-Program form is unchanged: rank 0 of the fold alone verifies against its own per-rank manifest
    man0 = tmp_path / "manifest_r0.json"
    r = subprocess.run(mod + ["build", "--program", dirs[0], "--rank", "0", "--ranks", "2", "--workload", str(wl), "--out", str(man0), "--compact"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr + r.stdout
    r = subprocess.run(mod + ["--engine", "traversal", "verify", "--manifest", str(man0), "--program", dirs[0], "--rank", "0", "--ranks", "2", "--workload", str(wl),
                              "--out", str(tmp_path / "v0.json")], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and json.loads((tmp_path / "v0.json").read_text())["ok"], r.stderr + r.stdout


# ----------------------------------------------------------------------------------------------------------------------------------------
# the bot-lifted Program (pad, padded baseline experimental; board M-0069 / M-0050): RequiredValues(Lift(P), Q) == RequiredValues(P, Q)
# ----------------------------------------------------------------------------------------------------------------------------------------

def lift_program(prog: dict, *, eos: int = 17, guard: bool = True, continuation: bool = False) -> dict:
    """Mechanically lift a request Program the way `verity_vllm/registry/lifted.py` lifts a Definition: every row's spec becomes
    `Lifted[<id>]_v1{same statics}`; with `guard`, the step guard of `LServe_v1{GUARD=in}` is inserted before every decode step's
    embedding (`s=<s>/live` = LStepLive1_v1(out[s-1], eos), `s=<s>/tok_in` = LSelect32_v1(live, out[s-1], bot32)) and the embedding
    reads `tok_in` instead of the previous step's sampled id; `eos[0]` (Const33) and `bot32` (LBot32_v1) open the Program."""
    prog = copy.deepcopy(prog)
    rows = prog["instances"]["rows"]
    out: list[dict] = []
    remap: dict[int, int] = {}

    def emit(spec, name, args):
        out.append({"i": len(out), "spec": spec, "form": "primitive" if not spec.startswith("Lifted[") else "call", "name": name, "args": args})
        return len(out) - 1

    def lift_spec(spec: str) -> str:
        head, _, rest = spec.partition("{")
        return f"Lifted[{head}]_v1" + ("{" + rest if rest else "")

    def ref_of(j: int):
        return [["n", j, 0, 1]]
    eos_i = emit(f"Const33[0x{eos:08x}]_v1", "eos[0]", []) if guard else None
    bot_i = emit("LBot32_v1", "bot32", []) if guard else None
    last_select: int | None = None
    step = 0
    for r in rows:
        fam = RM.spec_family(r["spec"])
        name = r["name"]
        new_args = [[[k, (remap[int(ref[1])] if k == "n" else ref[1]), ref[2], ref[3]] for ref in operand for k in [ref[0]]] for operand in r["args"]]
        if guard and fam.startswith("Embedding") and last_select is not None and name.startswith(f"model.model.embed_tokens/embedding_{step}["):
            live = emit("LStepLive1_v1", f"s={step}/live", [ref_of(remap[last_select]), ref_of(eos_i)])
            tok_in = emit("LSelect32_v1", f"s={step}/tok_in", [ref_of(live), ref_of(remap[last_select]), ref_of(bot_i)])
            if continuation:
                # [R17-32 Continuation_v1] the controller bits pad's lift emits per decode step: is_eos = (out[s-1] == eos), emit = live & ~is_eos
                is_eos = emit("LIsEOS1_v1", f"slot={step}/is_eos", [ref_of(remap[last_select]), ref_of(eos_i)])
                emit("LEmitNext[H=1]_v1", f"slot={step}/emit", [ref_of(is_eos), ref_of(live)])
            new_args = [ref_of(tok_in) if operand == ref_of(remap[last_select]) else operand for operand in new_args]
        j = emit(lift_spec(r["spec"]), name, new_args)
        remap[int(r["i"])] = j
        if RM.is_token_select(fam):
            last_select = int(r["i"])
            step += 1
    prog["instances"]["rows"] = out
    return prog


def test_lifted_program_has_the_same_required_values_as_the_wrapped_program():
    """pad's rule request (M-0069): RequiredValues(Lift(P), Q) keys == RequiredValues(P, Q) keys -- the wrapped families' members, the
    same element counts (one lifted value = w+1 bits), the step boundary at Lifted[TokenSelect]; the guard rows carry no identity and
    the audit classes them `guard` (their crossing is by design: tok_in is what the lifted Embedding names as input_token_ids)."""
    base = dense_df(LP=3, T=2, layers=2)
    lifted = lift_program(base)
    specs = {r["spec"].split("{")[0] for r in lifted["instances"]["rows"]}
    assert {"LStepLive1_v1", "LSelect32_v1", "LBot32_v1"} <= specs and all(sp.startswith(("Lifted[", "L", "Const33")) for sp in specs)
    t0 = RV.Traversal(RV.ProgramView.of(copy.deepcopy(base)), RV.Query())
    ids0 = rows_of(list(t0))
    t1 = RV.Traversal(RV.ProgramView.of(copy.deepcopy(lifted)), RV.Query())
    ids1 = rows_of(list(t1))
    assert t1.complete and t1.unmodelled == {} and t1.lifted_rows == len(base["instances"]["rows"]) and t1.guard_rows == 2 * 2 + 1   # live+tok_in per decode step, bot32 once
    assert [RM.canonical_identity(r) for r in ids1] == [RM.canonical_identity(r) for r in ids0]
    assert [ {k: v for k, v in r.items() if k != "spec"} for r in ids1] == [ {k: v for k, v in r.items() if k != "spec"} for r in ids0]
    assert t1.steps == t0.steps == 3 and t1.query_info["lifted"] and t1.query_info["guard_rows"] == 5
    aud = t1.boundary_audit
    assert aud["fully_wired"] and aud["unnamed_crossing"] == {} and aud["named_interior"] == {}
    assert aud["families"]["LSelect32_v1"] == {"guard": {"rows": 2, "crossing": 2}} and aud["families"]["LStepLive1_v1"]["guard"]["crossing"] == 0   # live is read only by its own step's tok_in
    assert aud["families"]["LBot32_v1"]["guard"]["rows"] == 1
    # the reference agrees; the adapter's manifest has the SAME digest as the wrapped Program's (the manifest is a cache of the keys)
    ref, summ = RV.reference_required_values(copy.deepcopy(lifted), RV.Query())
    assert summ["complete"] and rows_of(ref) == ids1
    m0 = RV.manifest_from_traversal(copy.deepcopy(base), {"sweep": {"arrive_steps": [0]}})
    m1 = RV.manifest_from_traversal(copy.deepcopy(lifted), {"sweep": {"arrive_steps": [0]}})
    assert m1["manifest_digest"] == m0["manifest_digest"] and m1["query"]["lifted"] is True and "lifted" not in m0["query"]
    # [revb S-r17b-16 R4] the controller / guard rows are NAMED per family in the query header (identities 0 by rule; rows counted)
    assert m1["query"]["guard_families"] == {"LBot": 1, "LSelect": 2, "LStepLive": 2} and sum(m1["query"]["guard_families"].values()) == m1["query"]["guard_rows"]
    # the R15 inventory has no lift rule: it fails closed on the lifted Program (named), never an empty or a partial population
    old = RM.build_manifest(copy.deepcopy(lifted), {"sweep": {"arrive_steps": [0]}}, engine="inventory")
    assert old["complete"] is False and any(k.startswith("Lifted[") for k in old["unmodelled"])


def test_r4_c1_continuation_controller_rows_are_guard_class_with_no_identity_and_the_same_digest():
    """[revb S-r17b-20 R4 c1 / R17-32 pad Continuation_v1] the controller bits pad's lift emits per decode step -- `LIsEOS<E>`
    (slot=<t>/is_eos) and `LEmitNext[H]` (slot=<t>/emit) -- are GUARD rows under this rule: counted per family in the query header,
    no identity, no unmodelled family, the manifest digest of the wrapped Program unchanged.  Before c1 they were 'unmodelled' -- the
    lifted Program's population statement would have failed closed on pad's tree while passing here (two statements for one lift)."""
    base = dense_df(LP=3, T=2, layers=2)
    lifted = lift_program(copy.deepcopy(base), continuation=True)
    specs = {r["spec"].split("{")[0] for r in lifted["instances"]["rows"]}
    assert {"LIsEOS1_v1", "LEmitNext[H=1]_v1"} <= specs
    t0 = RV.Traversal(RV.ProgramView.of(copy.deepcopy(base)), RV.Query())
    ids0 = rows_of(list(t0))
    t1 = RV.Traversal(RV.ProgramView.of(copy.deepcopy(lifted)), RV.Query())
    ids1 = rows_of(list(t1))
    assert t1.complete and t1.unmodelled == {}
    assert t1.guard_rows == 2 * 4 + 1 and t1.query_info["guard_families"] == {"LBot": 1, "LEmitNext": 2, "LIsEOS": 2, "LSelect": 2, "LStepLive": 2}
    assert [RM.canonical_identity(r) for r in ids1] == [RM.canonical_identity(r) for r in ids0]
    aud = t1.boundary_audit
    assert aud["fully_wired"] and aud["unnamed_crossing"] == {} and aud["named_interior"] == {}
    assert aud["families"]["LIsEOS1_v1"]["guard"]["rows"] == 2 and aud["families"]["LEmitNext[H=1]_v1"]["guard"]["rows"] == 2
    m0 = RV.manifest_from_traversal(copy.deepcopy(base), {"sweep": {"arrive_steps": [0]}})
    m1 = RV.manifest_from_traversal(copy.deepcopy(lifted), {"sweep": {"arrive_steps": [0]}})
    assert m1["manifest_digest"] == m0["manifest_digest"] and m1["query"]["guard_rows"] == 9
    assert set(RV.GUARD_PREFIXES) == {"LStepLive", "LSelect", "LBot", "LIsEOS", "LEmitNext"}


def test_a_lifted_moe_body_keeps_the_block_stream_population_and_the_digest():
    """pad M-0566 (3), the MoE stage on a real body: `Lifted[MoeRouterTopK_v1]` / `[MoeExpertGemm_v1]` / `[MoeExpertGemmW_v1]` / `[MoeSum_v1]`
    rows unlift to the aggregate families the block stream covers, so RequiredValues(Lift(P), Q) == RequiredValues(P, Q) row for row on a
    dense+MoE Program -- the six planes per MoE layer x step, the same digest, complete, no unmodelled family; the reference agrees."""
    base = dense_df(moe_layers=(1,))
    lifted = lift_program(copy.deepcopy(base))
    moe_lifted = {r["spec"].split("{")[0] for r in lifted["instances"]["rows"] if "Moe" in r["spec"]}
    assert moe_lifted == {"Lifted[MoeRouterTopK_v1]_v1", "Lifted[MoeExpertGemm_v1]_v1", "Lifted[MoeExpertGemmW_v1]_v1", "Lifted[MoeSum_v1]_v1"}, moe_lifted
    rows_b, tb = traversal_rows(copy.deepcopy(base), RV.Query())
    rows_l, tl = traversal_rows(copy.deepcopy(lifted), RV.Query())
    assert rows_l == rows_b and RM.manifest_digest(rows_l) == RM.manifest_digest(rows_b)
    assert tl.complete and tl.unmodelled == {} and tl.moe_layers == tb.moe_layers == {1}
    assert tl.populations()["by_family"]["moe_block_stream"] == tb.populations()["by_family"]["moe_block_stream"] == 6 * tb.steps * len(tb.moe_layers)
    ref, summ = RV.reference_required_values(copy.deepcopy(lifted), RV.Query())
    assert summ["complete"] and [x.row() for x in ref] == rows_l


def test_guard_rows_without_a_lift_and_the_block_level_tiny_pad_program_fail_closed():
    """(a) the guard in an UN-lifted Program is unmodelled (it is only meaningful as part of a lift); (b) pad's TINY-PAD root Program
    (`Program(L.lserve(TINY, LP, D_MAX, eos))`: Lifted[Prefill_v1] + per step live / tok_in / Lifted[StepBody_v1]{T=LP+s}) is
    block-level -- not a vLLM Build -- so the wrapped families Prefill_v1 / StepBody_v1 have no rule: UnmodelledFamily, named."""
    base = dense_df(LP=3, T=1)
    guarded_unlifted = lift_program(base)
    for r in guarded_unlifted["instances"]["rows"]:                                # strip the lift, keep the guard
        r["spec"] = RV.unlift(r["spec"])[0]
    t = RV.Traversal(RV.ProgramView.of(guarded_unlifted), RV.Query())
    list(t)
    assert not t.complete and any("guard rows" in k for k in t.unmodelled) and t.lifted_rows == 0 and t.guard_rows == 3
    _ref, summ = RV.reference_required_values(guarded_unlifted, RV.Query())
    assert not summ["complete"] and any("guard rows" in k for k in summ["unmodelled"])
    # (b) the TINY-PAD shape (row names and specs as lifted.py emits them; statics abbreviated)
    C = '{C={"dict":{"D":16,"H":2}},LP=4,GEMM=x,ATTENTION=y,NORM=z}'
    rows = [{"i": 0, "spec": "Lifted[Prefill_v1]_v1" + C, "form": "call", "name": "prefill", "args": [[["p", 0, 0, 4]], [["p", 1, 0, 100]]]},
            {"i": 1, "spec": "Const33[0x00000011]_v1", "form": "primitive", "name": "eos[0]", "args": []},
            {"i": 2, "spec": "LBot32_v1", "form": "primitive", "name": "bot32", "args": []}]
    prev = 0
    for s_ in range(1, 4):
        k = len(rows)
        rows.append({"i": k, "spec": "LStepLive1_v1", "form": "primitive", "name": f"s={s_}/live", "args": [[["n", prev, 0, 1]], [["n", 1, 0, 1]]]})
        rows.append({"i": k + 1, "spec": "LSelect32_v1", "form": "primitive", "name": f"s={s_}/tok_in", "args": [[["n", k, 0, 1]], [["n", prev, 0, 1]], [["n", 2, 0, 1]]]})
        rows.append({"i": k + 2, "spec": f"Lifted[StepBody_v1]_v1{{T={4 + s_}}}", "form": "call", "name": f"s={s_}/step", "args": [[["n", k + 1, 0, 1]], [["n", prev, 1, 65]], [["p", 1, 0, 100]]]})
        prev = k + 2
    tiny = {"instances": {"schema": "cb-a/instance-sequence/v1", "digest": "t" * 64, "params": [["prompt", "Array<4,Value<32>>"], ["weights", "{w: Array<4,Value<16>>}"]], "rows": rows},
            "result": {"program": {"digest": "t" * 64, "function": "LServe_v1"}, "wrapper": "request", "LP": 4, "T": 3, "attention_impls_observed": {}}, "artifact": None, "dir": "<tiny-pad>"}
    t = RV.Traversal(RV.ProgramView.of(copy.deepcopy(tiny)), RV.Query())
    ids = [x.row() for x in t]
    assert not t.complete and t.lifted_rows == 4 and t.guard_rows == 7
    assert any(k.startswith("StepBody_v1 under") for k in t.unmodelled) and any(k.startswith("Prefill_v1 under") for k in t.unmodelled), t.unmodelled
    assert all(r["family"] == "weights" for r in ids)                                # nothing but the shared weights: no silent partial population
    man = RV.manifest_from_traversal(copy.deepcopy(tiny))
    assert man["complete"] is False and man["query"]["lifted"] is True


def _invariant_fixtures() -> list[tuple[str, dict, RV.Query]]:
    from verity_capture.commit.tests.test_required_manifest import _program
    from verity_capture.commit.tests.test_required_manifest_stoch import _stochastic_program
    from verity_capture.commit.tests.test_required_manifest_tp import _tp_program
    return [("dense", dense_df(), RV.Query()),
            ("dense_moe", dense_df(moe_layers=(1,)), RV.Query()),
            ("gemma_r15", _program(), RV.Query()),
            ("stochastic_r15", _stochastic_program(), RV.Query()),
            ("derived_s", derived_s_program(), RV.Query()),
            ("tp_rank0", _tp_program(0), RV.Query(rank=0, ranks=2)),
            ("tp_rank1", _tp_program(1), RV.Query(rank=1, ranks=2)),
            ("lifted_dense", lift_program(dense_df()), RV.Query())]


@pytest.mark.parametrize("name,prog,q", _invariant_fixtures(), ids=lambda v: v if isinstance(v, str) else "")
def test_handoff_invariants_on_every_fixture_class(name, prog, q):
    """REQUIRED-VALUES-HANDOFF s7, on every fixture class the lane has: the traversal on the same (P, Q) is byte-identical (rows and
    manifest_digest); no identity has element_range lo == hi (MAN-02); no two rows claim one identity (canonical key + the B>=2 annotations);
    every claimed identity names ONE producer row (`spec` + the module of record) -- and the inventory agrees on all of it."""
    rows1, t1 = traversal_rows(prog, q)
    rows2, t2 = traversal_rows(prog, q)
    assert rows1 == rows2 and RM.manifest_digest(rows1) == RM.manifest_digest(rows2) and t1.complete == t2.complete
    assert rows1, name
    for r in rows1:
        lo, hi = r["element_range"]
        if hi is None:                                                             # the inventory's whole-field convention: weights only
            assert r["family"] == "weights" and lo == 0, (name, r)
        else:
            assert 0 <= lo < hi, (name, r)                                         # MAN-02: never a zero-width identity
        assert r["family"] in RM.FAMILIES and r["program_digest"] and r["op_path"] and r["output_member"] != "", (name, r)
    keys = [repr(RM.canonical_identity(r)) + repr((r.get("engine_step"), tuple(r.get("requests") or ()))) for r in rows1]
    assert len(keys) == len(set(keys)), (name, [k for k in keys if keys.count(k) > 1][:3])       # one row per identity
    # one producer per claimed identity: a module-output / sampler row names the producing family (`spec`); a stream row names the
    # producing module (`attention_module` / `moe_module`) -- the prescribed inputs (weights, prompt ids) have no producer row by definition
    for r in rows1:
        fam = r["family"]
        if fam in ("weights", "input_token_ids"):
            continue
        if fam.endswith("hidden_m1_stream"):
            assert r.get("attention_module") and r.get("geometry"), (name, r)
        elif fam == "moe_block_stream":
            assert r.get("moe_module") and r.get("rows"), (name, r)
        else:
            assert r.get("spec"), (name, r)
    # the inventory (the old side) agrees row for row when the family set is the same (the lift is traversal-only by design)
    if name != "lifted_dense":
        old = RM.build_manifest(copy.deepcopy(prog), request_id=q.request_id, rank=q.rank, ranks=q.ranks)
        assert old["identities"] == rows1 and old["manifest_digest"] == RM.manifest_digest(rows1) and old["complete"] == t1.complete


# ---------------------------------------------------------------------------------------------------------------------------------
# (7) Hopper preflight: the fa3 stream under a hopper profile, and the Program's own attention block size vs the launch table
# ---------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["hopper", "vllm-sm90-fa3", "h100"])
@pytest.mark.parametrize("fixture", ["dense", "gemma_gemm_v2", "r15"])
def test_a_hopper_profile_selects_the_fa3_stream_in_the_traversal_the_reference_and_the_inventory(profile, fixture):
    """Ruling 37 names the first Hopper rows (Gemm_v2{HopperBF16WgmmaDot16}, Attention_v2 FA3): under every spelling of the hopper profile
    the query's attention hidden family is `fa3_hidden_m1_stream` with `_fa3_geometry` (one launch per attention layer x step), the three
    engines agree byte for byte, and the population is complete -- no Hopper Program exists offline, so the profile switch is pinned here."""
    from verity_capture.commit.tests.test_required_manifest import _program
    from verity_capture.commit.tests.test_required_manifest_dense import _gemma_program
    prog = {"dense": dense_df, "gemma_gemm_v2": _gemma_program, "r15": _program}[fixture]()
    tr, t = assert_three_way(prog, q=RV.Query(profile=profile))
    assert t.query_info["attention_hidden_family"] == "fa3_hidden_m1_stream" and t.complete
    fam = t.populations()["by_family"]
    assert fam["fa3_hidden_m1_stream"] == len(t.hidden_layers) * t.steps and "fa2_hidden_m1_stream" not in fam
    for r in tr:
        if r["family"] == "fa3_hidden_m1_stream":
            assert r["op_path"].startswith("fa3.m1/L") and r["output_member"] == f"g{r['geometry']['HB']}x{r['geometry']['M']}x{r['geometry']['NB']}x{r['geometry']['BN']}x{r['geometry']['D']}"
    # these fixtures are FA2 Builds (Attention_v3 BN=128): wherever the fa3 table's tile differs the audit names it, the population stays complete
    for k, v in t.boundary_audit["hidden_bn_disagreement"].items():
        assert k.startswith("fa3.m1/L") and v["program_BN"] == [128] and v["table_BN"] == [192], (k, v)


def test_an_attention_block_size_the_program_carries_that_disagrees_with_the_launch_table_is_named_in_the_audit():
    """The hidden-stream geometry comes from the launch table (`_fa2_geometry` / `_fa3_geometry`), the Program's attention rows carry their
    own `BN`.  When they agree (every Program of record so far: Attention_v3 BN=128 at D=64, BN=64 at D=128) the audit is empty; when they
    do not (an FA2-forced Build under a hopper profile, or a table drift) the audit names the stream, both block sizes and the module --
    while the identities stay the inventory's (no semantics change: the required stream name is then one no launch exports, and coverage
    fails closed on it)."""
    prog = dense_df()
    tr, t = traversal_rows(prog, RV.Query())
    assert t.boundary_audit["hidden_bn_disagreement"] == {}
    # (a) fa2 table at D=64 says BN=128; a Program whose attention rows say BN=64
    bad = copy.deepcopy(prog)
    n = 0
    for r in bad["instances"]["rows"]:
        if r["spec"].startswith("Attention_v3{") and "BN=128" in r["spec"]:
            r["spec"] = r["spec"].replace("BN=128", "BN=64")
            n += 1
    assert n > 0
    tr2, t2 = traversal_rows(bad, RV.Query())
    dis = t2.boundary_audit["hidden_bn_disagreement"]
    assert set(dis) == {f"fa2.m1/L{L}" for L in range(2)}, dis
    for k, v in dis.items():
        assert v["program_BN"] == [64] and v["table_BN"] == [128] and v["steps"] == t2.steps and v["attention_module"].endswith(".self_attn.attn")
    assert t2.complete and tr2 == tr                                              # identities unchanged: the audit names, the coverage fails closed
    q = RV.Query()
    old = RM.build_manifest(copy.deepcopy(bad), None, request_id=q.request_id, rank=q.rank, ranks=q.ranks, profile=q.profile, shared_rope=q.shared_rope)
    assert old["identities"] == tr2
    # (b) the hopper profile on an FA2 Build at D=64 (the R15 Program): the fa3 table (one warp group) says BN=192, the Program says 128 -> named
    from verity_capture.commit.tests.test_required_manifest import _program
    tr3, t3 = traversal_rows(_program(), RV.Query(profile="hopper"))
    dis3 = t3.boundary_audit["hidden_bn_disagreement"]
    assert set(dis3) == {f"fa3.m1/L{L}" for L in range(2)} and all(v["program_BN"] == [128] and v["table_BN"] == [192] for v in dis3.values()), dis3
    # ... and at D=16 (this fixture) the fa3 table's BN is 128 = the Program's: agreement, nothing named
    _, t3b = traversal_rows(copy.deepcopy(prog), RV.Query(profile="hopper"))
    assert t3b.boundary_audit["hidden_bn_disagreement"] == {}
    # (c) attention rows without a BN static (AttentionHead_v1 style): nothing to compare, nothing named
    nobn = copy.deepcopy(prog)
    for r in nobn["instances"]["rows"]:
        if r["spec"].startswith("Attention_v3{"):
            r["spec"] = r["spec"].replace(",BN=128", "")
    _, t4 = traversal_rows(nobn, RV.Query())
    assert t4.boundary_audit["hidden_bn_disagreement"] == {} and t4.complete


def test_under_a_b2_fold_the_attention_block_size_is_compared_against_the_engine_steps_launch_not_the_components_own_rows():
    """B>=2: the FA3 tile reads the LAUNCH's max_seqlen_q (verity_vllm.registry.targets.fa3_kblock_n on the declared launch context), so a
    decoding request's own rows (1 x g <= 64: one warp group, BN 192 at D=64) are not the launch when another request prefills in the same
    engine step (BN 128).  The per-request audits therefore do not compare; the fold compares its own launch against the union of the
    components' BN at that engine step, under `boundary_audit["*"]`.  With the FA2-built fixtures under the hopper profile the fold
    names every stream where the 192 tile disagrees with the components' 128 and nothing else; the identities are the inventory's."""
    from verity_capture.commit.tests.test_required_manifest import _program
    progs = {"r0": _program(LP=3, T=2), "r1": _program(LP=5, T=2)}
    fx = {"fixture": "t_b2_hopper", "requests": [{"request_id": "r0", "index": 0, "prompt_len": 3, "max_tokens": 3, "arrive_step": 0},
                                                  {"request_id": "r1", "index": 1, "prompt_len": 5, "max_tokens": 3, "arrive_step": 0}]}
    old = RM.build_global_manifest(fx, copy.deepcopy(progs), workload_digest="w" * 64, profile="hopper")
    new = RV.global_manifest_from_traversal(fx, copy.deepcopy(progs), workload_digest="w" * 64, profile="hopper")
    eq = RV.manifest_equal(old, new)
    assert eq["equal"], eq["why"]
    ba = new["boundary_audit"]
    assert set(ba) == {"r0", "r1", "*"}
    for rid in ("r0", "r1"):
        assert ba[rid]["hidden_bn_disagreement"] == {}, rid                       # a component under a fold does not compare its own rows
    fold = ba["*"]["hidden_bn_disagreement"]
    assert set(fold) == {"fa3.m1/L0", "fa3.m1/L1"}, fold
    for v in fold.values():
        assert v["program_BN"] == [128] and v["table_BN"] == [192] and v["steps"] == 3, v
    # the same fold under the row's ampere profile (the fixtures' FA2 Build): agreement everywhere
    new2 = RV.global_manifest_from_traversal(fx, copy.deepcopy(progs), workload_digest="w" * 64)
    assert new2["boundary_audit"]["*"]["hidden_bn_disagreement"] == {}


def test_audit_flags_flatten_the_disagreement_lists_for_b1_and_for_a_fold():
    """`audit_flags` is the console line of `required_values diff`: the disagreement lists only, keyed by request (or "r" for B=1, "*"
    for the fold's launches); {} everywhere + fully_wired is the clean preflight."""
    from verity_capture.commit.tests.test_required_manifest import _program
    _, t = traversal_rows(dense_df(), RV.Query())                                # wired fixture (args on every row)
    assert RV.audit_flags(t.boundary_audit) == {"unnamed_crossing": {}, "unnamed_input": {}, "hidden_bn_disagreement": {}, "fully_wired": True}
    _, tu = traversal_rows(_program(), RV.Query())                               # the R15 fixture carries no args: not wired, nothing else flagged
    assert RV.audit_flags(tu.boundary_audit) == {"unnamed_crossing": {}, "unnamed_input": {}, "hidden_bn_disagreement": {}, "fully_wired": False}
    _, th = traversal_rows(_program(), RV.Query(profile="hopper"))
    fl = RV.audit_flags(th.boundary_audit)
    assert set(fl["hidden_bn_disagreement"]) == {"r"} and set(fl["hidden_bn_disagreement"]["r"]) == {"fa3.m1/L0", "fa3.m1/L1"}
    progs = {"r0": _program(LP=3, T=2), "r1": _program(LP=5, T=2)}
    fx = {"fixture": "t_b2_hopper", "requests": [{"request_id": "r0", "index": 0, "prompt_len": 3, "max_tokens": 3, "arrive_step": 0},
                                                  {"request_id": "r1", "index": 1, "prompt_len": 5, "max_tokens": 3, "arrive_step": 0}]}
    g = RV.global_manifest_from_traversal(fx, copy.deepcopy(progs), workload_digest="w" * 64, profile="hopper")
    fg = RV.audit_flags(g["boundary_audit"])
    assert set(fg["hidden_bn_disagreement"]) == {"*"} and fg["unnamed_crossing"] == {} and fg["unnamed_input"] == {}
    g2 = RV.global_manifest_from_traversal(fx, copy.deepcopy(progs), workload_digest="w" * 64)
    assert RV.audit_flags(g2["boundary_audit"]) == {"unnamed_crossing": {}, "unnamed_input": {}, "hidden_bn_disagreement": {}, "fully_wired": False}


def test_the_token_input_is_named_by_rule_not_by_the_params_name_so_a_step_wrapper_input_ids_is_a_named_input():
    """[R17 revb F-r17b-08] the R15 step wrapper calls its token input `input_ids` (the request wrapper: `prompt`).  A param-name whitelist
    flagged it as an unnamed input -- the population carried the value (`input_token_ids` x steps) while `complete` said False: same rows,
    same digest, opposite `complete` between the engines.  The rule of record: an input every reader of which is an Embedding row (or a
    lifted guard row) IS named by `input_token_ids`, whatever the param is called; a token input some OTHER row reads is not named by it."""
    prog = dense_df()
    prog["instances"]["params"][0][0] = "input_ids"                               # the step wrapper's name for the same input
    old = RM.build_manifest(copy.deepcopy(prog), None)
    t = RV.Traversal(copy.deepcopy(prog), RV.Query(arrive_step=0))
    ids = list(t)
    aud = t.boundary_audit
    assert aud["available"] and aud["unnamed_input"] == {} and aud["named_input"] == {"input_ids": "input_token_ids", "weights": "weights"}
    assert t.complete is True and old["complete"] is True
    assert sorted(map(repr, map(RM.canonical_identity, (x.row() for x in ids)))) == sorted(map(repr, map(RM.canonical_identity, old["identities"])))
    new = RV.manifest_from_traversal(copy.deepcopy(prog), None)
    assert new["manifest_digest"] == old["manifest_digest"] and new["complete"] is old["complete"] is True
    # `prompt` under the request wrapper: the same rule names it (not the whitelist)
    t2 = RV.Traversal(dense_df(), RV.Query(arrive_step=0))
    list(t2)
    assert t2.boundary_audit["named_input"] == {"prompt": "input_token_ids", "weights": "weights"}
    # the token input read by a NON-Embedding row too: the value is no longer named by `input_token_ids` alone -> unnamed, incomplete, by name
    prog3 = dense_df()
    gemm = next(r for r in prog3["instances"]["rows"] if r["spec"].startswith("Gemm"))
    gemm["args"] = gemm["args"] + [[["p", 0, 0, 1]]]
    t3 = RV.Traversal(prog3, RV.Query(arrive_step=0))
    list(t3)
    assert set(t3.boundary_audit["unnamed_input"]) == {"prompt"} and "Gemm_v1" in t3.boundary_audit["unnamed_input"]["prompt"]["readers"]
    assert t3.complete is False and any("unnamed input" in k and "`prompt`" in k for k in t3.unmodelled)


def test_the_manifest_header_states_whether_the_guard_was_armed_in_both_engines():
    """[R17 revb F-r17b-09] the 'missing family cannot vanish from both sides' guard runs only over rows with operand runs; a Program without
    them leaves the audit unavailable and `complete` untouched -- so every manifest header (and, through it, the Commit record) SAYS whether
    the guard was armed: the traversal on a wired Program = armed with zero flags; on an unwired one = not armed by name; the inventory = not
    armed (no dataflow audit), never read as clean."""
    from verity_capture.commit.tests.test_required_manifest import _program
    wired = RV.manifest_from_traversal(dense_df(), None)
    g = wired["guard"]
    assert g["engine"] == "traversal" and g["armed"] is True and g["available"] is True and g["fully_wired"] is True and g["requests_audited"] == 1
    assert g["flags"] == {"unnamed_crossing": 0, "unnamed_input": 0, "hidden_bn_disagreement": 0}
    assert g["named_input"] == ["prompt -> input_token_ids", "weights -> weights"] and g["rule"] == RV.GUARD_RULE
    unwired = RV.manifest_from_traversal(_program(), None)                      # the R15 fixture carries no args
    assert unwired["guard"]["armed"] is False and unwired["guard"]["available"] is False and unwired["complete"] is True
    inv = RM.build_manifest(dense_df(), None, engine="inventory")
    assert inv["guard"]["engine"] == "inventory" and inv["guard"]["armed"] is False and inv["guard"]["flags"] is None and "no dataflow audit" in inv["guard"]["why"]
    # a fold: armed only when every component's audit is available; the flags are counted over the requests + the fold's launches
    progs = {"r0": dense_df(LP=3, T=2), "r1": dense_df(LP=5, T=2)}
    fx = {"fixture": "t_b2", "requests": [{"request_id": "r0", "index": 0, "prompt_len": 3, "max_tokens": 3, "arrive_step": 0},
                                           {"request_id": "r1", "index": 1, "prompt_len": 5, "max_tokens": 3, "arrive_step": 0}]}
    gm = RV.global_manifest_from_traversal(fx, copy.deepcopy(progs), workload_digest="w" * 64)
    assert gm["guard"]["armed"] is True and gm["guard"]["requests_audited"] == 2 and gm["guard"]["flags"] == {"unnamed_crossing": 0, "unnamed_input": 0, "hidden_bn_disagreement": 0}
    mixed = {"r0": dense_df(LP=3, T=2), "r1": _program(LP=5, T=2)}
    gm2 = RV.global_manifest_from_traversal(fx, copy.deepcopy(mixed), workload_digest="w" * 64)
    assert gm2["guard"]["armed"] is False and gm2["guard"]["requests_audited"] == 2
    # a TP merge of rank manifests: armed iff every part was
    m = RM.merge_manifests([RV.manifest_from_traversal(dense_df(), None), RV.manifest_from_traversal(dense_df(), None)])
    assert m["guard"]["armed"] is True and m["guard"]["engine"] == "traversal" and len(m["guard"]["parts"]) == 2
    m2 = RM.merge_manifests([RV.manifest_from_traversal(dense_df(), None), RM.build_manifest(dense_df(), None, engine="inventory")])
    assert m2["guard"]["armed"] is False and m2["guard"]["engine"] == ["inventory", "traversal"] and m2["guard"]["flags"] is None


def test_padded_moe_block_is_one_module_boundary_row_and_its_slot_bodies_are_interior_under_q_module():
    """[moe MOE-PAD-SPEC route A (3), reqval RequiredValues authority] the padded fixed-slot construction emits ONE `MoeFfnFixedL_v1` call per
    token row where MOE-01 has the pair rows.  Under Q_module_v1 that row is the module-boundary VU of `mlp.experts`: `out` (H) is required per
    row, `moe_block_stream` is ABSENT (no router/pair/sum rows exist), the manifest is complete on all three engines, and the header states the
    ruling (the E slot bodies are interior on this Program; moe's 5,121 words/slot planes are not R(P_pad, Q_module_v1)).  The hand-lifted twin
    `LMoeFfnL_v1` is a lifted row of the same population (Lift(Q): ids modulo the lift)."""
    prog = dense_df(layers=2, moe_layers=(1,), M_topk=2, E=4, padded_moe="MoeFfnFixedL_v1")
    tr, t = assert_three_way(prog)
    H = 32
    outs = [r for r in tr if r["op_path"] == "model.layers.1.mlp.experts"]
    # one identity per (step, module): the token rows packed (step 0: LP 3 rows x H; step 1: one decode row x H)
    assert sorted((r["step"], r["output_member"], tuple(r["element_range"]), r["rows"]) for r in outs) == [(0, "out", (0, 3 * H), 3), (1, "out", (0, H), 1)], outs
    assert all(r["family"] == "instance_outputs" for r in outs)
    assert not any(r["family"] == "moe_block_stream" for r in tr) and "moe_block_stream" not in t.query_info["required_families"]
    assert t.complete and not t.unmodelled
    mc = t.query_info["moe_construction"]
    assert mc["construction"] == "padded" and mc["rows"] == 4 and "NOT R(P_pad, Q_module_v1)" in mc["not_required_under_Q_module_v1"]
    aud = t.boundary_audit["families"]
    assert set(aud["MoeFfnFixedL_v1"]) == {"boundary"} and t.boundary_audit["unnamed_crossing"] == {}
    # the router logits Gemm stays a module output of `mlp.gate` (the block's second input, by linkage)
    assert any(r["op_path"] == "model.layers.1.mlp.gate" and r["output_member"] == "0" for r in tr)
    # the hand lift: same identities modulo the lift, counted as lifted rows (the traversal's; the inventory has no lift rule)
    lifted = dense_df(layers=2, moe_layers=(1,), M_topk=2, E=4, padded_moe="LMoeFfnL_v1")
    trl, tl = traversal_rows(lifted)
    assert tl.lifted_rows == 4 and tl.padded_moe_rows == 4 and tl.complete
    strip = lambda rows: sorted((r["op_path"], r["output_member"], r["step"], tuple(r["element_range"])) for r in rows if r["family"] == "instance_outputs")
    assert strip(trl) == strip(tr)
    # the MOE-01 construction of the same layer is a DIFFERENT population (R17-45 (1)): planes present there, absent here -- never #Q equality
    tr01, t01 = traversal_rows(dense_df(layers=2, moe_layers=(1,), M_topk=2, E=4))
    assert any(r["family"] == "moe_block_stream" for r in tr01) and "moe_construction" not in t01.query_info


def test_q_module_pad_names_the_slot_units_as_a_different_query_and_leaves_q_module_untouched():
    """[revb S-r17b-51 (b)] `Q_module_pad_v1` (boundary `module-pad`) = Q_module_v1 + the padded clause: on every MoeFfnFixedL_v1 row the slot
    units (`slot<e>/up` 2I, `/act` I, `/down` H, `/g` 1 f32; moe's 5,121 words/slot union basis, M-1119 (b)) + `router_ids` / `router_w` (TOPK)
    are identities beside `out`.  Its OWN id and digest on the header; traversal == reference; Q_module_v1's population on the same Program
    is untouched (the pad query adds rows, never rewrites one); the slot-unit bytes are the construction's planes, by the query's text."""
    prog = dense_df(layers=2, moe_layers=(1,), M_topk=2, E=4, padded_moe="MoeFfnFixedL_v1")
    H, I, E, TOPK = 32, 64, 4, 2
    q = RV.Query(boundary="module-pad")
    tr, t = traversal_rows(prog, q)
    ref, summ = RV.reference_required_values(copy.deepcopy(prog), q)
    assert tr == rows_of(ref) and t.complete and summ["complete"] and not t.unmodelled
    qi = t.query_info
    assert qi["query_id"] == "Q_module_pad_v1" and qi["boundary"] == "module-pad" and qi["based_on"]["query_id"] == "Q_module_v1"
    assert qi["query_id_digest"] != qi["based_on"]["query_id_digest"] and "padded_slots" in qi["why"]
    assert qi["moe_construction"]["slot_units"].startswith("REQUIRED")
    ex = {(r["step"], r["output_member"]): (tuple(r["element_range"]), r["dtype"]) for r in tr if r["op_path"] == "model.layers.1.mlp.experts"}
    per_step = 1 + 2 + 4 * E                                                         # out + router_ids/router_w + 4 members per slot
    assert len(ex) == 2 * per_step, len(ex)
    for step, rows in ((0, 3), (1, 1)):                                              # LP 3 rows at step 0, one decode row at step 1
        assert ex[(step, "out")] == ((0, rows * H), "bf16")
        # [revb F-r17b-52 (iii)] the slot units and router members are LIFTED words: l17 = Value<17> (u32, bf16 payload), l33 = Value<33> (u64)
        assert ex[(step, "router_ids")] == ((0, rows * TOPK), "l33") and ex[(step, "router_w")] == ((0, rows * TOPK), "l33")
        for e in range(E):
            assert ex[(step, f"slot{e}/up")] == ((0, rows * 2 * I), "l17") and ex[(step, f"slot{e}/act")] == ((0, rows * I), "l17")
            assert ex[(step, f"slot{e}/down")] == ((0, rows * H), "l17") and ex[(step, f"slot{e}/g")] == ((0, rows), "l33")
    assert RM.lifted_bot("l17") == 1 << 16 and RM.lifted_bot("l33") == 1 << 32 and "enc(bot) = 1 << w" in RM.q_module_pad_text("ampere", "fa2_hidden_m1_stream", "d")
    assert "a_e = 0" in qi["why"]["padded_slots"] and "1 << w" in qi["why"]["padded_slots"]
    # words per token row of the block = moe's basis (5,121 at OLMoE E=64/H=2048/I=1024: (2I + I + H + 1) per slot) x E + 2 TOPK
    words_row = sum(w for (st, m), ((lo, w), _d) in ex.items() if st == 1 and m != "out")
    assert words_row == E * (2 * I + I + H + 1) + 2 * TOPK
    assert RM.padded_slot_members({"E": 64, "TOPK": 8, "VPT": 8, "H": 2048, "I": 1024}) [2:][:4] == [("slot0/up", 2048, "l17"), ("slot0/act", 1024, "l17"), ("slot0/down", 2048, "l17"), ("slot0/g", 1, "l33")]
    assert sum(w for _m, w, _d in RM.padded_slot_members({"E": 64, "TOPK": 8, "VPT": 8, "H": 2048, "I": 1024})) == 64 * 5121 + 16
    # Q_module_v1 on the same Program: exactly the rows of the pad query minus the slot units and router members -- nothing rewritten
    tr0, t0 = traversal_rows(prog)
    strip = lambda rows: [r for r in rows if not (r["op_path"].endswith("mlp.experts") and r["output_member"] != "out")]
    assert strip(tr) == tr0 and t0.query_info["query_id"] == "Q_module_v1" and t0.query_info["moe_construction"]["slot_units"].startswith("NOT required")
    # the un-padded (MOE-01) Program under module-pad: the clause has nothing to name, the population is Q_module_v1's row for row
    prog01 = dense_df(layers=2, moe_layers=(1,), M_topk=2, E=4)
    tr01p, _ = traversal_rows(prog01, RV.Query(boundary="module-pad"))
    tr01, _ = traversal_rows(prog01)
    assert tr01p == tr01
    # the manifest builder accepts the boundary by name and the sampled_replay's row index reads it from the header
    assert RM.manifest_boundary("module-pad") == "module-pad"
    from verity_capture.commit import sampled_replay as SR
    assert SR.manifest_boundary_of({"query": qi}) == "module-pad" and SR.manifest_boundary_of({"query": t0.query_info}) == "module"
