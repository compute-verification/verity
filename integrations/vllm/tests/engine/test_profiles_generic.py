"""`profiles.generic`: every checkpoint role's κ profile from its HF config.json and the family facts table -- no per-model
module, no role -> module table.  Every shipped config.json builds a profile or is refused with its reason."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json

import pytest

from verity_vllm.program import model as analytic
from verity_vllm.observe.events import RunHeader
from verity_vllm.observe.fold.patterns import TokenSelectArgmax
from verity_vllm.engine.profiles import dense_generic as dg
from verity_vllm.engine.profiles import family_facts as FF
from verity_vllm.observe.fold.profiles import generic as G
from verity_vllm.observe.fold.profiles import profile_for_case, profile_module_for_case
from verity_vllm.observe.fold.profiles import vllm_d9105ea80_sm89_eager as BASE
from verity_vllm.observe.fold.patterns.collectives import EmbeddingShardCuda, TpCollective

SHIPPED = sorted(p.name[: -len(".config.json")] for p in dg.HF_CONFIG_DIR.glob("*.config.json"))
#: the tensor-parallel worlds with recorded runs (the retired `*_tp2rank` hand modules and the Qwen3-MoE TP2 half)
TP2_ROLES = ("B1", "LLAMA32_1B", "QWEN3_30B_A3B")
ROLES = [(r, 1) for r in SHIPPED] + [(r, 2) for r in TP2_ROLES]
#: shipped configs the constructor refuses, by the reason it states
REFUSED = {"QWEN15_AWQ": "awq"}


# ------------------------------------------------------------------------------------------------------------------- shipped configs

def test_shipped_config_sha_is_the_manifest_bytes():
    """A role's config.json is the manifest's byte copy (sha256 == checkpoints.json's config_sha256)."""
    for role in SHIPPED:
        hit = dg.find_hf_config(role)
        assert hit is not None, role
        path, source, entry = hit
        assert source == "package:hf_configs", (role, source)
        if entry and entry.get("config_sha256"):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["config_sha256"], role


def test_a_pin_reads_its_config_of_record_and_an_unpinned_model_none(monkeypatch):
    """`pinned_hf_config`, the one config.json a served pin's φ reads on the prover's side (`advice_compile`) and the verifier's
    (`prove_path.phi_of_serve`): the int8 pin's is its own, with its quantization config, never the bf16 one's."""
    int8 = dg.pinned_hf_config("verity-int8/Llama-3.2-1B-INT8", "15b0976d400d796ee589e7737d03a919526bf800")
    assert int8 == dg.HF_CONFIG_DIR / "LLAMA32_1B_INT8.config.json"
    assert json.loads(int8.read_text())["quantization_config"]["quant_method"] == "verity_int8"
    assert dg.pinned_hf_config("unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da") == dg.HF_CONFIG_DIR / "LLAMA32_1B.config.json"
    assert dg.pinned_hf_config("verity-int8/Llama-3.2-1B-INT8", "0" * 40) is None
    assert dg.pinned_hf_config("m/x", "r") is None
    monkeypatch.setattr(dg, "_sha256", lambda p: "0" * 64)
    with pytest.raises(ValueError, match="config_sha256"):
        dg.pinned_hf_config("verity-int8/Llama-3.2-1B-INT8", "15b0976d400d796ee589e7737d03a919526bf800")


# ------------------------------------------------------------------------------------------------------------ names, roles, the finder

def test_module_name_grammar():
    assert G.module_name("B1") == "derived_B1" and G.module_name("B1", 2) == "derived_B1_tp2"
    assert G.parse_module("verity_vllm.observe.fold.profiles.derived_QWEN3_30B_A3B_tp2") == ("QWEN3_30B_A3B", 2)
    assert G.parse_module("derived_LLAMA32_1B") == ("LLAMA32_1B", 1)
    assert G.parse_module("dense_MISTRAL7B") == ("MISTRAL7B", 1)                  # the name recorded runs cite
    assert G.parse_module("gen_llama_llama32_1b") is None and G.parse_module("vllm_d9105ea80_sm89_eager") is None


def test_finder_materialises_every_role_module_and_the_x07_name():
    for role, world in ROLES:
        if role in REFUSED:
            continue
        mod = importlib.import_module(f"verity_vllm.observe.fold.profiles.{G.module_name(role, world)}")
        assert mod.PROFILE.id == G.profile_id(role, mod.C, world)
        assert callable(mod.weight_field) and callable(mod.root_class) and callable(mod.check_header) and callable(mod.build_profile)
    dense = importlib.import_module("verity_vllm.observe.fold.profiles.dense_MISTRAL7B")
    assert dense.PROFILE.id == importlib.import_module("verity_vllm.observe.fold.profiles.derived_MISTRAL7B").PROFILE.id


def test_the_x07_name_is_for_dense_architectures_only():
    with pytest.raises(ImportError, match="derived_OLMOE"):
        importlib.import_module("verity_vllm.observe.fold.profiles.dense_OLMOE")
    with pytest.raises(ImportError, match="derived_PYTHIA160M"):
        importlib.import_module("verity_vllm.observe.fold.profiles.dense_PYTHIA160M")


def test_profile_module_for_case_is_derived_not_looked_up():
    from verity_vllm.observe.fold import profiles
    assert not hasattr(profiles, "PROFILE_BY_CASE")
    for role, world in ROLES:
        if role in REFUSED:
            continue
        assert profile_module_for_case(role, world) == G.module_name(role, world)
        assert profile_for_case(role, world).notes["case"] == role
    for role, why in REFUSED.items():
        with pytest.raises(KeyError, match=why):
            profile_module_for_case(role)
    with pytest.raises(KeyError, match="no HF config.json"):
        profile_module_for_case("NO_SUCH_ROLE")


def test_derived_b0_is_the_protected_base_profile_behind_the_tuned_gemm():
    """The golden corpus's profile (`vllm_d9105ea80_sm89_eager`) is what the generic constructor derives from SmolLM2's config.json: the
    same C up to the identity fields and unset keys, the same pattern list after the target's tuned-GEMM specialization."""
    derived = importlib.import_module("verity_vllm.observe.fold.profiles.derived_B0").PROFILE

    def facts(C):
        return {k: v for k, v in C.items() if k not in ("family", "revision", "profile") and v is not None}

    assert facts(derived.config) == facts(BASE.PROFILE.config)
    assert [p.name for p in derived.patterns] == ["gemm.matmul_kernel_persistent.tuned"] + [p.name for p in BASE.PROFILE.patterns]


# ------------------------------------------------------------------------------------------------------------------- family facts

def test_family_facts_rules_agree_with_analytic():
    """The table's `rules` are `analytic.FAMILY_RULES` where analytic has the family (one truth, two readers)."""
    for fam, f in FF.FAMILY_FACTS.items():
        if fam in analytic.FAMILY_RULES:
            assert f["rules"] == analytic.FAMILY_RULES[fam], fam
        assert f["arch"] in ("dense", "dense-ln", "moe", "dsa-moe", "mhc-moe"), fam
        assert "ForCausalLM" in f["vllm_class"], fam
        assert any(s in f["rules_provenance"] for s in ("vllm/model_executor/models/", "vllm/models/", "transformers")), fam


def test_every_hand_fact_cites_where_vllm_hard_codes_it():
    for fam, f in FF.FAMILY_FACTS.items():
        for key, fact in f["facts"].items():
            assert fact.get("provenance"), (fam, key)
            assert FF.VLLM in fact["provenance"] or "vllm/" in fact["provenance"] or "csrc/" in fact["provenance"], (fam, key, fact["provenance"])
            assert sum(k in fact for k in ("const", "key", "derive")) == 1, (fam, key, "exactly one of const / key / derive")
        for p in ("params",):
            if f.get(p):
                assert f[p].get("provenance"), (fam, p)


def test_gemma2_hand_facts_resolve_to_vllms_values():
    cfg = json.loads((dg.HF_CONFIG_DIR / "GEMMA2_2B.config.json").read_text())
    C = profile_for_case("GEMMA2_2B").config
    assert C["tie_word_embeddings"] is True and C["attn_logit_softcapping"] == 50.0 and C["final_logit_softcapping"] == 30.0
    assert C["query_pre_attn_scalar"] == cfg["query_pre_attn_scalar"] and C["embed_normalizer"] == pytest.approx(cfg["hidden_size"] ** 0.5)
    assert C["rmsnorm_plus_one"] is True and C["layer_norms"] == 4 and C["hidden_act"] == "gelu_pytorch_tanh"


def test_pythia_hand_facts_resolve_to_vllms_values():
    C = profile_for_case("PYTHIA160M").config
    assert C["family"] == "gpt_neox" and C["BIAS"] is True and C["use_parallel_residual"] is True and C["NORM"] == "layernorm"
    assert C["dense_bias"] is True and C["layer_norms"] == 2 and C["hidden_act"] == "gelu"
    assert C["ROPE_THETA"] == 10000.0 and C["ROT"] == int(C["D"] * 0.25)


def test_moe_hand_facts_and_derived_router_vpt():
    o, q = profile_for_case("OLMOE").config, profile_for_case("QWEN3_30B_A3B").config
    assert o["RENORMALIZE"] is False and o["QK_NORM"] is True and o["QK_NORM_EPS"] == 1e-5 and o["EPS"] == 1e-5 and o["NUM_EXPERTS"] == 64 and o["ROUTER_VPT"] == 8
    assert q["RENORMALIZE"] is True and q["QK_NORM"] == "head" and q["NUM_EXPERTS"] == 128 and q["ROUTER_VPT"] == 8
    assert q["I"] == q["MOE_I_FULL"] == 768
    assert G.router_vpt_bf16(64) == 8 and G.router_vpt_bf16(128) == 8 and G.router_vpt_bf16(16) == 8 and G.router_vpt_bf16(256) == 8
    with pytest.raises(ValueError):
        G.router_vpt_bf16(96)


# --------------------------------------------------------------------------------------------------------------------- world > 1

@pytest.mark.parametrize("role", TP2_ROLES)
def test_rank_profile_is_the_world_divided_configuration(role):
    """The claim the retired `*_tp2rank` docstrings made, checked: heads / kv heads / the (expert) width / the vocabulary divided by the
    world, the head width kept, the two TP patterns first, the sampler over the FULL vocabulary."""
    full, rank = profile_for_case(role, 1), profile_for_case(role, 2)
    Cf, Cr = full.config, rank.config
    assert (Cr["NH"], Cr["KVH"], Cr["I"], Cr["V"]) == (Cf["NH"] // 2, Cf["KVH"] // 2, Cf["I"] // 2, Cf["V"] // 2)
    assert Cr["D"] == Cf["D"] and Cr["QKV"] == (Cr["NH"] + 2 * Cr["KVH"]) * Cr["D"] and Cr["WORLD"] == 2
    assert isinstance(rank.patterns[0], EmbeddingShardCuda) and isinstance(rank.patterns[1], TpCollective) and rank.patterns[1].world == 2
    assert [type(p).__name__ for p in rank.patterns[2:]] == [type(p).__name__ for p in full.patterns]
    sampler = next(p for p in rank.patterns if isinstance(p, TokenSelectArgmax))
    assert sampler.vocab == Cf["V"]
    assert rank.id.endswith("-tp2-rank")


def test_rank_check_header_reshards_the_recorded_full_config():
    cfg = json.loads((dg.HF_CONFIG_DIR / "LLAMA32_1B.config.json").read_text())
    mod = importlib.import_module("verity_vllm.observe.fold.profiles.derived_LLAMA32_1B_tp2")
    h = RunHeader("verity-capture/log/v1", "2026-09-23T00:00:00Z", "pid", {}, {"repo": mod.REPO_ID, "config": cfg, "tp": 2}, {"requests": []})
    assert mod.check_header(h) == []
    assert any("tensor_parallel_size" in m for m in mod.check_header(dataclasses.replace(h, model={**h.model, "tp": 1})))
    bad = dataclasses.replace(h, model={**h.model, "config": {**cfg, "num_attention_heads": 30}})
    assert any("num_attention_heads" in m for m in mod.check_header(bad))


def test_world_must_divide_the_sharded_keys():
    with pytest.raises(ImportError, match="not divisible"):
        importlib.import_module("verity_vllm.observe.fold.profiles.derived_B0_tp2")          # SmolLM2-135M: 9 heads


# ------------------------------------------------------------------------------------------------------------------- refusals

def test_structural_refusals_are_stated():
    cfg = json.loads((dg.HF_CONFIG_DIR / "LLAMA32_1B.config.json").read_text())
    assert G.structural_refusal(cfg) is None
    assert "model_type" in G.structural_refusal({**cfg, "model_type": "falcon"})
    assert "sliding_window" in G.structural_refusal({**cfg, "sliding_window": 4096})
    q = json.loads((dg.HF_CONFIG_DIR / "QWEN05.config.json").read_text())
    assert q["sliding_window"] and not q["use_sliding_window"] and G.structural_refusal(q) is None        # Qwen2's gated window is inert
    assert "sliding_window" in G.structural_refusal({**q, "use_sliding_window": True})
    m = json.loads((dg.HF_CONFIG_DIR / "QWEN3_30B_A3B.config.json").read_text())
    assert "mlp_only_layers" in G.structural_refusal({**m, "mlp_only_layers": [0]})
    assert "awq" in G.structural_refusal(json.loads((dg.HF_CONFIG_DIR / "QWEN15_AWQ.config.json").read_text()))


def test_the_sparse_attention_family_builds_its_h200_dsa_and_shared_moe_profile():
    """GLM-5.3's facts resolve over a TP8 rank and register every DSA fold before the shared-expert MoE scope."""
    from pathlib import Path

    from verity_vllm.observe.fold.patterns.dsa import (
        INDEX_SHARE_SRC_SHA,
        NORM_SRC_SHA,
        Q_SRC_SHA,
        DsaIndexShare,
        DsaMqaAllocation,
        DsaMqaGather,
        DsaNormRope,
        DsaQ,
        DsaSparseMla,
        DsaTopK,
    )
    from verity_vllm.observe.fold.profiles.gen_shared_moe_patterns import MoEFusedSharedExpertsFp8

    cfg = json.loads((Path(__file__).parents[1] / "check" / "fixtures" / "glm_moe_dsa" / "config.json").read_text())
    assert FF.facts_for(cfg)["arch"] == "dsa-moe"
    assert G.structural_refusal(cfg) is None
    profile = G.build_profile("GLM53", 8, hf_config=cfg, repo="zai-org/GLM-5.3-FP8", revision="aca966e4")
    assert profile.config["MOE_I_FULL"] == 2048 and profile.config["I"] == 256 and profile.config["ROUTER_VPT"] == 8
    assert [type(p) for p in profile.patterns[2:9]] == [
        DsaNormRope, DsaQ, DsaMqaGather, DsaMqaAllocation, DsaTopK, DsaIndexShare, DsaSparseMla]
    assert (NORM_SRC_SHA, Q_SRC_SHA, INDEX_SHARE_SRC_SHA) == (
        "2dcf56892eb2b61c0f0249965013fc1f63968c00e473b3df35cb186d42c650de",
        "7380a1385db787ceded52869fe395160e859d22710ca46f20d1d0fdef2f41043",
        "ad40bdb415664d3b65e3e4163863158096f2ea52f2e2a3e01ff8be63aeb98f63",
    )
    shared = next(p for p in profile.patterns if isinstance(p, MoEFusedSharedExpertsFp8))
    assert shared.group == 128 and shared.gate_spec is None
    wf = profile.weight_field                                           # the family's `params`: dense below FIRST_DENSE = 3, MoE from it on
    assert wf("model.layers.0.mlp.gate_up_proj.weight") == ("dense_layers", 0, "gate_up_w")
    assert wf("model.layers.3.mlp.experts.routed_experts.w13_weight") == ("layers", 0, "w13")
    assert wf("model.layers.6.self_attn.indexer.wq_b.weight") == ("indexers", 3, "wq_b_w")      # INDEXER_LAYERS = (0, 1, 2, 6, ...)
    assert wf("model.layers.4.self_attn.indexer.k_norm.bias") is None
    assert profile.notes["moe"]["status"].startswith("constructive")
    vals, prov = FF.resolve_facts(G.rank_hf_config(cfg, 8), {})
    assert {k: vals[k] for k in ("NUM_EXPERTS", "N_GROUP", "TOPK_GROUP", "RENORMALIZE", "SCORING", "ROUTED_SCALING_FACTOR", "SHARED_I",
                                 "INDEX_HEADS", "INDEX_HEAD_DIM", "INDEX_TOPK", "INDEX_TOPK_FREQ", "INDEX_SKIP_TOPK_OFFSET", "FIRST_DENSE")} == {
        "NUM_EXPERTS": 256, "N_GROUP": 1, "TOPK_GROUP": 1, "RENORMALIZE": True, "SCORING": "sigmoid", "ROUTED_SCALING_FACTOR": 2.5,
        "SHARED_I": 256, "INDEX_HEADS": 32, "INDEX_HEAD_DIM": 128, "INDEX_TOPK": 2048, "INDEX_TOPK_FREQ": 4, "INDEX_SKIP_TOPK_OFFSET": 3,
        "FIRST_DENSE": 3}
    assert vals["ATTN_SCALE"] == 256 ** -0.5 and vals["ROPE_NEOX"] is False and prov["SHARED_I"] == "derived"


def test_the_hyper_connected_family_resolves_its_rank_facts():
    """DeepSeek-V4-Pro's facts resolve over a TP8 rank (the experts' width and the heads divided, the rest whole)."""
    cfg = json.loads((dg.HF_CONFIG_DIR / "DSV4_PRO.config.json").read_text())
    assert FF.facts_for(cfg)["arch"] == "mhc-moe" and G.structural_refusal(cfg) is None
    vals, prov = FF.resolve_facts(G.rank_hf_config(cfg, 8), {})
    assert {k: vals[k] for k in ("Q_LORA", "HEAD_DIM", "QK_ROPE", "O_GROUPS", "O_LORA", "WINDOW", "PADDED_HEADS", "INDEX_HEADS", "NUM_EXPERTS",
                                 "TOPK", "HASH_LAYERS", "SCORING", "SCORE_BIAS", "SHARED_I", "EXPERT_DTYPE", "HC_MULT", "HC_REPEAT")} == {
        "Q_LORA": 1536, "HEAD_DIM": 512, "QK_ROPE": 64, "O_GROUPS": 16, "O_LORA": 1024, "WINDOW": 128, "PADDED_HEADS": 64, "INDEX_HEADS": 64,
        "NUM_EXPERTS": 384, "TOPK": 6, "HASH_LAYERS": 3, "SCORING": "sqrtsoftplus", "SCORE_BIAS": "noaux_tc", "SHARED_I": 384,
        "EXPERT_DTYPE": "fp4", "HC_MULT": 4, "HC_REPEAT": 20}
    assert vals["ATTN_SCALE"] == 512 ** -0.5 and len(vals["COMPRESS_RATIOS"]) == cfg["num_hidden_layers"] + 1 and prov["SHARED_I"] == "derived"


def test_the_hyper_connected_rank_profile_states_the_builds_weights_record():
    """DeepSeek-V4-Pro's TP8 rank profile: `C` from the facts and the rank's config.json, a weights Record of the family's params in their
    order at the allocated shapes (the fold's `weights` parameter), every non-Marlin parameter bound to its own field and the Marlin-served
    experts to none (their words are no leaves of the checkpoint-layout fields), and the header check over the recorded full config."""
    from verity_vllm.observe.fold.fold import weights_type_of
    mod = importlib.import_module(f"verity_vllm.observe.fold.profiles.{G.module_name('DSV4_PRO', 8)}")
    prof, C, W = mod.PROFILE, mod.C, mod.PROFILE.weights_type
    assert {k: C[k] for k in ("H", "NL", "NH", "V", "VOCAB", "I", "SHARED_I", "FP8_BLOCK", "WORLD")} == {
        "H": 7168, "NL": 61, "NH": 16, "V": 16160, "VOCAB": 129280, "I": 384, "SHARED_I": 384, "FP8_BLOCK": 128, "WORLD": 8}
    names = [n for n, _ in W.fields]
    assert names[:5] == ["model.hc_head_fn", "model.hc_head_base", "model.hc_head_scale", "model.embed_tokens.weight", "model.layers.0.hc_attn_fn"]
    assert names[-3:] == ["model.norm.weight", "lm_head.weight", "model.layers.0.attn.rotary_emb.cos_sin_cache"]
    assert "model.layers.0.hc_attn_fn_broadcast" in names and "model.layers.1.hc_attn_fn_broadcast" not in names
    assert "model.layers.2.attn.indexer.wq_b.weight" in names and "model.layers.3.attn.indexer.wq_b.weight" not in names    # ratios 4 / 128
    assert "model.layers.2.ffn.gate.tid2eid" in names and "model.layers.3.ffn.gate.e_score_correction_bias" in names      # HASH_LAYERS 3

    def field(n):
        return W.fields[W.index(n)][1]
    assert (field("model.layers.0.attn.wo_a.weight").leaves, field("model.layers.0.attn.wo_a.weight").leaf_widths()[0]) == (2 * 1024 * 4096, 8)
    assert field("model.layers.5.ffn.experts.routed_experts.w13_weight").leaves == 384 * 768 * 3584
    assert prof.weight_field("model.layers.5.attn.wq_b.weight_scale_inv") == ("model.layers.5.attn.wq_b.weight_scale_inv",)
    assert prof.weight_field("model.layers.5.ffn.experts.routed_experts.w2_weight_scale") is None
    assert mod.root_class("model.layers.5.ffn.experts.routed_experts.w2_weight_scale") == "marlin-served"
    assert mod.root_class("model.layers.5.attn.wq_b.weight") == "weight" and mod.root_class("model.layers.5.attn.k_scale") == "unknown"
    assert weights_type_of(prof, C) is W and weights_type_of(BASE.PROFILE, BASE.C) == analytic.weights_type(BASE.C)
    from verity_vllm.observe.fold.patterns import dsv4_patterns
    names, own = [p.name for p in prof.patterns], [p.name for p in dsv4_patterns()]
    assert own[-4:] == ["fp8.dynamic_deepgemm_linear", "gemm.cublas_linear_default", "gemm.cublas_mm_f32out", "norm.rms_norm_cuda"]
    assert names[:len(own)] == own and names[len(own):len(own) + 4] == [
        "tp.embedding_shard_cuda", "tp.collective", "moe.marlin_shared_experts.mxfp4", "residual_add.aten_add"]
    assert prof.witnesses == ("schedule",)
    h = RunHeader("verity-capture/log/v1", "2026-10-18T00:00:00Z", "pid", {}, {"repo": mod.REPO_ID, "config": mod.HF_CONFIG, "tp": 8},
                  {"requests": []})
    assert mod.check_header(h) == []
    bad = dataclasses.replace(h, model={**h.model, "config": {**mod.HF_CONFIG, "num_hidden_layers": 60, "n_routed_experts": 256}, "tp": 4})
    assert mod.check_header(bad) == ["num_hidden_layers=60 != C[NL]=61", "n_routed_experts=256 != C[NUM_EXPERTS]=384", "tensor_parallel_size=4 != WORLD=8"]


def test_olmoe_rank_record_holds_the_whole_all_heads_qk_norm():
    """OLMoE's q/k RMSNorm spans every head (`RMSNorm(total_num_heads * head_dim)`): a TP rank all-gathers q and k first and holds the
    whole replicated weight, so the rank record's `q_norm` / `k_norm` have the full model's width while its heads are the rank's."""
    full, rank = (importlib.import_module(f"verity_vllm.observe.fold.profiles.{G.module_name('OLMOE', w)}") for w in (1, 2))

    def layer(C):
        wt = analytic.weights_type(C)
        return {n: t.leaves for n, t in wt.fields[wt.index("layers")][1].elem.fields}

    f, r = layer(full.C), layer(rank.C)
    assert rank.C["NH"] * 2 == full.C["NH"] and r["o_w"] * 2 == f["o_w"]
    assert r["q_norm"] == f["q_norm"] == full.C["NH"] * full.C["D"] and r["k_norm"] == f["k_norm"] == full.C["KVH"] * full.C["D"]
