"""verity_vllm.predict: predicted Programs equal the traced Builds of record, and the scorer says where they do not.

The pinned digests are Programs the Build traced on rtxpro6000 (cc 12.0, 188 SMs) from the checkpoints' `config.json`
(`fixtures/`): SmolLM2-135M from cov-k01-10, Llama-3.2-1B from cov-k05-7, TinyLlama-1.1B from cov-k04-7, Qwen2.5-0.5B from
cov-k03-8 (its biased q/k/v linear one `GemmBias_v2{K,N,DOT}` Call), Phi-3-mini from cov-k07-3, Qwen3-4B from cov-k09-3, Gemma-2-2B from cov-k06-5, Llama-3.2-1B's TP2 ranks from cov-p000,
Gemma-2-2B's from cov-p058-2, Qwen3-30B-A3B's from cov-p081 and OLMoE's from cov-p069.
A library change that moves a Definition moves the traced digest too: re-trace the unit (`verity-vllm build --wrapper
step --M 1 --pos0 8`, CPU only) and pin the new digest with the predictor's.
`SiluMul_v2{I}` as the `silu_mul` kind moved every pin but Gemma-2's (GeGLU): each is the Build's re-trace of its unit from the
same checkpoints on the declared rtxpro6000 target (run r20261002-123618-cee3, CPU only), and with `SiluMul_v1{I}` in its place
the predictor gives the rows' traced digests.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pytest

from verity_vllm.config import UsageError
from verity_vllm.predict import score as S
from verity_vllm.predict.build import Unexpressible, build_program, weight_leaves, model_shape
from verity_vllm.predict.phi import Phi

FIX = Path(__file__).parent / "fixtures"
RTXPRO6000 = {"compute_capability": [12, 0], "num_sms": 188}
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu"}
STOCH = {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": "derived"}


def _phi(cfg: dict, requests: list[dict] | None = None, **kw) -> Phi:
    return Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=dict(RTXPRO6000), workload={"requests": requests or []}, **kw)


@pytest.mark.parametrize("model,digest", [
    ("smollm2-135m", "3c792f4807b296f1d7a372cd246aa80eed6b8b929f927900da5059155e2af697"),
    ("llama32-1b", "3b438e0e35d365105b36d0697023dad70da86ecd27b27d476f2445a5641fabeb"),
    ("tinyllama-11b", "c12cd8cc914bda1fb13136fe0a8ead108e4c1721fe4e9ab05714a4f41a1be19f"),
    ("qwen25-05b", "221e97b683f184f265c4baf1f2e490f674c59ec86490c91fa61fe69d2235c28c"),
    ("phi3-mini", "9ac4fc245509b18c08e6a8e091748a415bb0b7dbb220fcbd3353e2841d3a3fc9"),
    ("qwen3-4b", "69ebe6af722d17985687e59aa529902771aaa8d323ae4931999b3fcf8a676108"),
    ("olmoe-1b-7b", "99d789b704be9c42bdb429d5e45baa71380de3226b622b2c37dc9d521ebf1887"),
    ("qwen3-30b-a3b", "e3655459083291eafb440a562e38ed9b5ae5addfbde2f0e39114e89277021d78"),
    ("gemma2-2b", "c062d1b305e1bcf5fb614dbfd70c7686e73df57dd7c9c23671a66b8107c10ddd"),
])
def test_step_program_equals_the_traced_build(model, digest):
    r = build_program(_phi(json.loads((FIX / f"{model}.config.json").read_text())), "step", M=1, pos0=8)
    cls = {"qwen25-05b": "Qwen2ForCausalLM", "phi3-mini": "Phi3ForCausalLM", "qwen3-4b": "Qwen3ForCausalLM", "olmoe-1b-7b": "OlmoeForCausalLM",
           "qwen3-30b-a3b": "Qwen3MoeForCausalLM", "gemma2-2b": "Gemma2ForCausalLM"}.get(model, "LlamaForCausalLM")
    assert r.function == f"Derived[vLLMStep[{cls}]]_v1{{SRC={{f8a263a011b5}}}}"
    assert r.digest == digest


@pytest.mark.slow
def test_request_and_workload_programs_equal_the_traced_build():
    """SmolLM2-135M B1 256/32 greedy (cov-k01-10): the request Program (LP 256, T 31) and GP-01's workload Program over it.
    About 20 s: the only test that composes a workload, which the step pins cannot reach."""
    phi = _phi(json.loads((FIX / "smollm2-135m.config.json").read_text()),
               [{"request_id": "u0", "index": 0, "prompt_len": 256, "max_tokens": 32, "arrive_step": 0}])
    r = build_program(phi, "request", LP=256, T=31)
    assert r.digest == "c18f32dba2ceef5c38caf64c9348cbc498f9fe5122edfe55f61cb15e20278824"
    assert r.n_root_nodes == 86738
    assert S.predict_workload(phi)["digest"] == "5a03914457d2b2d9597a620a285566e7649b00537acb13994a67963bff016636"


@pytest.mark.parametrize("model,function,rank,digest", [
    ("llama32-1b", "LlamaForCausalLM]]_v1{SRC={ffedfc14b0e0}}", 0, "b61d007ab219e9161a8c16e7854af7cbd3b7deda267930fe236038f1170f1459"),
    ("llama32-1b", "LlamaForCausalLM]]_v1{SRC={ffedfc14b0e0}}", 1, "7c37593eda24d3ebe09527b33c27516a5d290c92d83d015ac4e842d2d28928b5"),
    ("gemma2-2b", "Gemma2ForCausalLM]]_v1{SRC={33928f49cee7}}", 0, "7ca0319a25893edc2bccf7866f25cd29cc28f58b6eef3158b0c2b04feda0c753"),
    ("gemma2-2b", "Gemma2ForCausalLM]]_v1{SRC={33928f49cee7}}", 1, "7da1ceee0fc64bbd16aa0267c5c283708eb75740ac532659a6e7c64c2bac05db"),
    ("qwen3-30b-a3b", "Qwen3MoeForCausalLM]]_v1{SRC={3f2e2f30d60d}}", 0, "de6d9f143bc8a485a3ec4dcdeb6ecbe43013ea87d444dfa167cb1143921e8ef3"),
    ("qwen3-30b-a3b", "Qwen3MoeForCausalLM]]_v1{SRC={3f2e2f30d60d}}", 1, "40da70bce2941f84f5d149ca70f69194b24e60240e998e5ca4cb0e17fc7356e0"),
    ("olmoe-1b-7b", "OlmoeForCausalLM]]_v1{SRC={32f6bda2809e}}", 0, "c4ebc70a471394983db8ba424cbb2ce35851bd9dc71a485fb6f4187865bb3626"),
    ("olmoe-1b-7b", "OlmoeForCausalLM]]_v1{SRC={32f6bda2809e}}", 1, "3e9c7925d46c5b0d2748762abe840b802e8a741cb1a3194ec8aedc06ffd89dcd"),
])
def test_tensor_parallel_rank_step_equals_the_traced_build(model, function, rank, digest):
    """Llama-3.2-1B (cov-p000), Gemma-2-2B (cov-p058-2), Qwen3-30B-A3B (cov-p081) and OLMoE (cov-p069) at TP2: each rank's step
    over its shards, with a peer Input per collective in call order (the embedding's all-reduce, each layer's o and down / MoE
    all-reduces, the logits all-gather) and the two partials in rank order.  Gemma-2's normalizer scales the reduced embedding and
    its soft-cap the gathered logits; the fused MoE keeps its router and experts whole over each expert's width shard; OLMoE's q / k
    norm over all heads reads its q and k rows all-gathered, then keeps this rank's slice."""
    r = build_program(_phi(json.loads((FIX / f"{model}.config.json").read_text()), tp={"world": 2, "rank": rank}), "step", M=1, pos0=8)
    assert r.function == "Derived[vLLMStep[" + function
    assert r.digest == digest


@pytest.mark.slow
def test_tensor_parallel_rank_workload_equals_the_traced_build():
    """cov-p000's rank-1 workload Program (B1 256/32 at TP2): the only test that composes a rank's request component, whose
    `peers_*` Inputs GP-01 keeps request-local.  About 15 s."""
    phi = _phi(json.loads((FIX / "llama32-1b.config.json").read_text()),
               [{"request_id": "u0", "index": 0, "prompt_len": 256, "max_tokens": 32, "arrive_step": 0}], tp={"world": 2, "rank": 1})
    assert S.predict_workload(phi)["digest"] == "e2d93ebe35e1ee77d568ade81b93a79e1a39cd1ca3caf7ca91d2aeb8b6758351"


@pytest.mark.parametrize("sampling,M,digest", [
    ({**STOCH}, 8, "68a5d8274937c6d89c63bcb509ee5c97818485af19cbf13d99a6577bc8241493"),
    ({**STOCH, "top_p": 1.0, "splits": 32}, 1, "902f910d8ad69adf3c2dec4f88f17e53b25833be68043c664c5de5cf8eaa770f"),
])
def test_stochastic_step_program_equals_the_traced_build(sampling, M, digest):
    """SmolLM2-135M under the seeded top-p policy: a B8 row's step (the split count a `splits` Input, cov-g230) and a B1 row's (the
    single request's constant S = 32 on 188 SMs, cov-g253-2)."""
    r = build_program(_phi(json.loads((FIX / "smollm2-135m.config.json").read_text()), sampling=sampling), "step", M=M, pos0=8)
    assert r.digest == digest


def test_sampling_follows_the_builds_policy():
    one = {"sweep": {"sampling": "stoch-t0.8-p0.95", "sampling_params": {"temperature": 0.8, "top_p": 0.95, "top_k": 0}, "sampling_seeds": [7]},
           "requests": [{"prompt_len": 8, "max_tokens": 3, "seed": 7}]}
    assert S.sampling_of(one, _phi(TINY)) == {**STOCH, "splits": 32}
    assert S.sampling_of({**one, "requests": one["requests"] * 8}, _phi(TINY)) == STOCH
    assert S.sampling_of({"requests": [{"prompt_len": 8, "max_tokens": 3}]}, _phi(TINY)) == {"kind": "greedy"}
    with pytest.raises(Unexpressible, match="the Build refuses the policy"):
        S.sampling_of({**one, "sweep": {**one["sweep"], "sampling_params": {"temperature": 0.8, "top_p": 0.95, "top_k": 50}}}, _phi(TINY))


def test_weights_record_follows_the_module_order():
    C = model_shape(TINY)
    names = [n for n, _ in weight_leaves(C)]
    assert names[:3] == ["model.model.embed_tokens.weight", "model.model.layers.0.self_attn.qkv_proj.weight",
                         "model.model.layers.0.self_attn.o_proj.weight"]
    assert "model.lm_head.weight" not in names  # tied: the embedding's leaf
    assert len(names) == 1 + 6 * 2 + 1 + 1 + 4 * 2
    shards = [(n.split("layers.0.")[-1], t) for n, t in weight_leaves(C, 2)[:7]]
    assert [(n, t.leaves) for n, t in shards] == [
        ("model.model.embed_tokens.weight", 128 * 64), ("self_attn.qkv_proj.weight", (2 + 2 * 1) * 16 * 64), ("self_attn.o_proj.weight", 64 * 32),
        ("mlp.gate_up_proj.weight", 128 * 64), ("mlp.down_proj.weight", 64 * 64), ("input_layernorm.weight", 64), ("post_attention_layernorm.weight", 64)]
    untied = [n for n, _ in weight_leaves(model_shape({**TINY, "tie_word_embeddings": False}))]
    assert untied.index("model.lm_head.weight") == untied.index("model.model.norm.weight") + 1
    biased = [n for n, _ in weight_leaves(model_shape({**TINY, "model_type": "qwen2", "use_sliding_window": False, "sliding_window": 32}))]
    assert biased[1:4] == ["model.model.layers.0.self_attn.qkv_proj.weight", "model.model.layers.0.self_attn.qkv_proj.bias",
                           "model.model.layers.0.self_attn.o_proj.weight"]
    normed = [n for n, _ in weight_leaves(model_shape({**TINY, "model_type": "qwen3"}))]
    assert normed[2:5] == ["model.model.layers.0.self_attn.o_proj.weight", "model.model.layers.0.self_attn.q_norm.weight",
                           "model.model.layers.0.self_attn.k_norm.weight"]
    moe = weight_leaves(model_shape(json.loads((FIX / "olmoe-1b-7b.config.json").read_text())))
    assert [n.split("layers.0.")[-1] for n, _ in moe[1:10]] == [
        "self_attn.qkv_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.o_proj.weight", "mlp.gate.weight",
        "mlp.experts.w13_weight", "mlp.experts.w2_weight", "input_layernorm.weight", "post_attention_layernorm.weight"]
    gemma = [n for n, _ in weight_leaves(model_shape(json.loads((FIX / "gemma2-2b.config.json").read_text())))]
    assert [n.split("layers.0.")[-1] for n in gemma[1:9]] == [
        "self_attn.qkv_proj.weight", "self_attn.o_proj.weight", "mlp.gate_up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight",
        "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"]
    assert gemma.index("model.model.normalizer") == gemma.index("model.model.norm.weight") + 1 == \
        gemma.index("model.model.layers.0.self_attn.rotary_emb.cos_sin_cache") - 1


@pytest.mark.parametrize("target,kind", [({"compute_capability": [8, 0], "num_sms": 108}, "GemmBias_v1"), (RTXPRO6000, "GemmBias_v2")])
def test_the_biased_linear_follows_the_targets_rule(target, kind):
    """Qwen2's q/k/v bias: one `GemmBias_v1{K,N}` Call per row under the accepted kinds (A100, FA2), and where the target's record
    names a biased-linear step (`gemm_bias_dot`: rtxpro6000) one `GemmBias_v2{K,N,DOT}` Call with the target's step; never the GEMM
    then `BiasAdd_v1`, which only a family whose record names no step would keep."""
    cfg = {**TINY, "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "use_sliding_window": False, "sliding_window": 32}
    r = build_program(Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=target, workload={"requests": []}), "step")
    names = {re.match(r"\w+", k).group() for k in r.desc["definitions"]}
    assert names & {"GemmBias_v1", "GemmBias_v2", "BiasAdd_v1"} == {kind}


@pytest.mark.parametrize("cfg,phi_kw,fact", [
    ({**TINY, "model_type": "gpt2"}, {}, "model_type 'gpt2': not in the family facts table"),
    ({**TINY, "attention_bias": True}, {}, "attention_bias"),
    ({**TINY, "mlp_bias": True}, {}, "MLP biases"),
    ({**TINY, "sliding_window": 32}, {}, "sliding window 32"),
    ({**TINY, "model_type": "phi3", "sliding_window": 8}, {}, "a history of 9 keys under the sliding window of 7 keys"),
    ({**TINY, "hidden_act": "gelu"}, {}, "hidden_act 'gelu'"),
    ({**json.loads((FIX / "gemma2-2b.config.json").read_text()), "query_pre_attn_scalar": 144}, {}, "query_pre_attn_scalar 144"),
    ({**json.loads((FIX / "qwen3-30b-a3b.config.json").read_text()), "num_experts": 32}, {}, "32 experts, top-8"),
    ({**json.loads((FIX / "qwen3-30b-a3b.config.json").read_text()), "shared_expert_intermediate_size": 512}, {}, "shared experts"),
    ({**json.loads((FIX / "qwen3-30b-a3b.config.json").read_text()), "mlp_only_layers": [0]}, {}, "dense MLP layers among the MoE ones"),
    (TINY, {"sampling": {**STOCH, "top_k": 50}}, "top_k 50"),
    (TINY, {"sampling": {**STOCH, "splits": None}}, "without a split count"),
    (TINY, {"sampling": {"kind": "mixed"}}, "sampling 'mixed'"),
    (TINY, {"tp": {"world": 4, "rank": 0}}, "tensor parallel world 4"),
    ({**TINY, "hidden_size": 96, "num_attention_heads": 6, "num_key_value_heads": 3}, {"tp": {"world": 2, "rank": 0}},
     "6 query and 3 key/value heads over 2 ranks"),
    ({**TINY, "vocab_size": 250}, {"tp": {"world": 2, "rank": 1}}, "a vocabulary of 250 over 2 ranks"),
    (TINY, {"tp": {"world": 2, "rank": 0}, "sampling": dict(STOCH)}, "stochastic sampling under tensor parallelism"),
])
def test_unexpressible_names_the_fact(cfg, phi_kw, fact):
    with pytest.raises(Unexpressible, match=re.escape(fact)):
        build_program(_phi(cfg, **phi_kw), "step")


def _row(tmp_path: Path, row: str, builds: dict, sweep: dict | None = None) -> dict:
    return {"dir": str(tmp_path / "rows" / "job" / row), "job": "job", "row": row, "summary": {}, "builds": builds, "source_root": None,
            "workload_path": None, "workload": {"model": "org/tiny", "tokenizer_revision": "rev", "target": {}, **({"sweep": sweep} if sweep else {}),
                                                "requests": [{"request_id": "u0", "index": 0, "prompt_len": 8, "max_tokens": 3, "arrive_step": 0,
                                                              **({"seed": 7} if sweep else {})}]}}


def test_score_reports_exact_divergence_and_refusals(tmp_path):
    cfgp = tmp_path / "hf" / "models--org--tiny" / "snapshots" / "rev" / "config.json"
    cfgp.parent.mkdir(parents=True)
    cfgp.write_text(json.dumps(TINY))
    phi = _phi(TINY, [{"request_id": "u0", "index": 0, "prompt_len": 8, "max_tokens": 3, "arrive_step": 0}])
    step = build_program(phi, "step", M=1, pos0=8)
    req = build_program(phi, "request", LP=8, T=2)
    wl = S.predict_workload(phi)
    # the traced request descriptor: the prediction with one root reference moved (the attention's key history of row 3)
    bad = json.loads(json.dumps(req.desc))
    nodes = bad["definitions"][bad["root"]]["body"]["nodes"]
    k = next(i for i, n in enumerate(nodes) if bad["callees"][n["fn"]].startswith("Attention") and len(n["args"]) == 3
             and "concat" in n["args"][1]["refs"])
    nodes[k]["args"][1]["refs"] = nodes[k]["args"][2]["refs"]
    dpath = tmp_path / "traced_request.json.gz"
    with gzip.open(dpath, "wt") as f:
        json.dump(bad, f)
    greedy = "smollm2-135m__bf16__rtxpro6000__tp1__b1__i8__o3__mixed__greedy__bi-eager"
    stoch = "smollm2-135m__bf16__rtxpro6000__tp1__b1__i8__o3__mixed__stoch-t0.8-p0.95__bi-eager"
    topk = "smollm2-135m__bf16__rtxpro6000__tp1__b1__i8__o4__mixed__stoch-t0.8-p0.95__bi-eager"
    sweep = {"sampling": "stoch-t0.8-p0.95", "sampling_params": {"temperature": 0.8, "top_p": 0.95, "top_k": 0}, "sampling_seeds": [7]}
    sstep = build_program(_phi(TINY, sampling={**STOCH, "splits": 32}), "step", M=1, pos0=8)
    tp2 = "smollm2-135m__bf16__rtxpro6000__tp2__b1__i8__o3__mixed__greedy__bi-eager"
    rank_step = {r: build_program(_phi(TINY, tp={"world": 2, "rank": r}), "step", M=1, pos0=8).digest for r in (0, 1)}
    rank1_wl = S.predict_workload(_phi(TINY, phi.requests, tp={"world": 2, "rank": 1}))["digest"]
    assert len({rank_step[0], rank_step[1], step.digest}) == 3
    prog = lambda d: {"digest": d, "function": None, "n_root_nodes": None}  # noqa: E731
    inv = {"schema": "verity-predict/inventory/v1", "root": str(tmp_path / "rows"), "rows": [
        _row(tmp_path, greedy, {
            "build_step": {"wrapper": "step", "M": 1, "pos0": 8, "LP": 8, "T": 4, "ok": True, "program": prog(step.digest)},
            "build_request": {"wrapper": "request", "M": 1, "pos0": 8, "LP": 8, "T": 2, "ok": True, "program": prog("0" * 64), "descriptor": str(dpath)},
            "build_request_LP9_T2": {"wrapper": None, "ok": False, "message": "Device string must not be empty"},
            "build_workload": {"ok": True, "workload_digest": wl["digest"], "lag": 1, "n_requests": 1}}),
        _row(tmp_path, stoch, {"build_step": {"wrapper": "step", "M": 1, "pos0": 8, "ok": True, "program": prog(sstep.digest)}}, sweep),
        _row(tmp_path, topk, {"build_step": {"wrapper": "step", "M": 1, "pos0": 8, "ok": True, "program": prog("1" * 64)}},
             {**sweep, "sampling_params": {**sweep["sampling_params"], "top_k": 50}}),
        _row(tmp_path, tp2, {f"rank{r}/build_step": {"wrapper": "step", "M": 1, "pos0": 8, "ok": True, "program": prog(rank_step[r])} for r in (0, 1)}
             | {"rank1/build_workload": {"ok": True, "workload_digest": rank1_wl, "lag": 1, "n_requests": 1}}),
    ]}
    cache = str(tmp_path / "cache.jsonl")
    res = S.score(inv, hf_home=str(tmp_path / "hf"), workers=0, max_rows=64, cache=cache)
    assert S.score(inv, hf_home=str(tmp_path / "hf"), workers=0, max_rows=64, cache=cache, cached_only=True)["summary"] == res["summary"]
    assert S.score(inv, hf_home=str(tmp_path / "hf"), workers=0, max_rows=64, cache=cache, cached_only=True, max_diverge=0)["summary"] == \
        res["summary"], "a divergence in the cache is reported whatever the budget for new ones"
    cold = S.score(inv, hf_home=str(tmp_path / "hf"), workers=0, max_rows=64, cache=str(tmp_path / "empty.jsonl"), cached_only=True)
    assert cold["summary"]["units_compared"] == 0 and cold["summary"]["units_by_status"]["not_run"] == 7
    tight = S.score(inv, hf_home=str(tmp_path / "hf"), workers=0, max_rows=64, max_mem_mb=1, cache=cache, cached_only=True)
    assert tight["summary"]["units_by_status"]["over_budget"] == 7
    units = {(r["row"], u["unit"]): u for r in res["rows"] for u in r["units"]}
    assert units[(greedy, "build_step")]["status"] == "exact"
    assert units[(greedy, "build_workload")]["status"] == "exact"
    u = units[(greedy, "build_request")]
    assert u["status"] == "differs" and u["predicted_digest"] == req.digest
    assert u["first_divergence"]["where"] == "root node" and u["first_divergence"]["node"] == k and u["first_divergence"]["arg"] == 1
    assert u["equivalence"]["verdict"] != "equivalent"
    assert units[(greedy, "build_request_LP9_T2")]["status"] == "no_trace"
    assert units[(stoch, "build_step")]["status"] == "exact"
    assert units[(topk, "build_step")]["status"] == "unexpressible" and "top_k = 50" in units[(topk, "build_step")]["reason"]
    assert all(units[(tp2, n)]["status"] == "exact" for n in ("rank0/build_step", "rank1/build_step", "rank1/build_workload"))
    s = res["summary"]
    assert (s["units_compared"], s["units_exact"]) == (7, 6)
    assert s["units_by_granularity"]["step:exact"] == 4
    assert "| first divergence | units |" in S.table_md(res)
    entries = [json.loads(line) for line in open(cache)]
    assert res["code"] == S.code_version() and {e["code"] for e in entries} == {res["code"]} and s["compared_by_code"] == {res["code"]: 7}
    other, unstamped = str(tmp_path / "other.jsonl"), str(tmp_path / "unstamped.jsonl")
    with open(other, "w") as f:
        f.writelines(json.dumps({**e, "code": "0" * 16}) + "\n" for e in entries)
    with open(unstamped, "w") as f:
        f.writelines(json.dumps({k: v for k, v in e.items() if k != "code"}) + "\n" for e in entries)
    for path, version in ((other, "0" * 16), (unstamped, S.UNSTAMPED)):
        kw = dict(hf_home=str(tmp_path / "hf"), workers=0, max_rows=64, cache=path, cached_only=True)
        assert S.score(inv, **kw)["summary"]["units_by_status"]["not_run"] == 7, "another code's results are not this code's"
        old = S.score(inv, **kw, reuse_code=(version,))["summary"]
        assert old["compared_by_code"] == {version: 7} and {**old, "compared_by_code": 0} == {**s, "compared_by_code": 0}
    # whatever the order written: this code's result for a key, then the reused versions' in the order named
    e = entries[0]
    mixed = str(tmp_path / "mixed.jsonl")
    with open(mixed, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in ({**e, "digest": "own"}, {**e, "code": "a", "digest": "a"}, {**e, "code": "b", "digest": "b"}))
    assert S.Cache(mixed, e["code"], ("a", "b")).get(e["key"])["digest"] == "own"
    assert S.Cache(mixed, "c", ("a", "b")).get(e["key"])["digest"] == "a"
    assert S.Cache(mixed, "c", ("b", "a")).get(e["key"])["digest"] == "b"


def test_memory_estimate_is_above_a_measured_peak():
    """A Qwen3-4B request of LP+T = 79 peaked at 814 MB resident (`_child`'s max_rss_mb); the pool admits units by the estimate."""
    assert S.mem_estimate_mb(model_shape(json.loads((FIX / "qwen3-4b.config.json").read_text())), [79]) > 814


def test_score_never_writes_into_a_rows_root(tmp_path):
    rows = tmp_path / "rows"
    (rows / "job" / "row").mkdir(parents=True)
    with pytest.raises(UsageError, match="never written into a Build row"):
        S._refuse_inside(str(rows / "job" / "row" / "score.json"), [str(rows)])
    S._refuse_inside(str(tmp_path / "out" / "score.json"), [str(rows)])


def test_no_record_path_module_reads_the_predictor():
    """The Build, Match and Commit never import `verity_vllm.predict`; the CLI's command table names it as a string only, and the
    Build's input trace lists the package as a forbidden input."""
    from verity_vllm.program.frontend import inputs_trace
    pkg = Path(S.__file__).resolve().parents[1]
    pat = re.compile(r"^\s*(from|import)\s+(verity_vllm\.predict|\.+predict)\b|import_module\([^)]*predict", re.M)
    hits = [str(p.relative_to(pkg)) for p in pkg.rglob("*.py") if "predict" not in p.relative_to(pkg).parts[:1] and pat.search(p.read_text())]
    assert hits == []
    assert any(m in str(Path(S.__file__).resolve()) for m in inputs_trace._FORBIDDEN_MARKERS)


def test_code_version_is_one_tree_over_a_namespaces_portions(monkeypatch, tmp_path):
    """`verity` as a namespace: the digest is the same over any order of its portions and with a cwd directory that joined it
    without a module, and moves with an edit."""
    import verity

    for rel in ("upper/verity/ml/__init__.py", "lower/verity/ir/__init__.py", "lower/verity/ir/defs.py",
                "cwd/verity/primitives/circuits/src/verity/ir/defs.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(f"# {rel}\n")

    def version(*portions: str) -> str:
        monkeypatch.setattr(verity, "__path__", [str(tmp_path / p / "verity") for p in portions])
        return S.code_version()

    a = version("upper", "lower")
    assert version("lower", "upper") == version("cwd", "upper", "lower") == a
    (tmp_path / "lower" / "verity" / "ir" / "defs.py").write_text("# edited\n")
    assert version("upper", "lower") != a
