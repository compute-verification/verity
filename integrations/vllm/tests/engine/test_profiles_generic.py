"""[P6, 2026-09-23] `profiles.generic`: every checkpoint role's κ profile from its HF config.json and the family facts table -- no per-model
module, no role -> module table.  The per-role fixtures `profiles/expected/derived_<ROLE>[_tp<W>].json` pin the canonical form
(`profiles.canonical`) the generic constructor produced when the hand-written modules were retired (they were shown equal to it, object by
object, by the `test_profiles_generic_equals_hand.py` that left with those modules); a change here is a deliberate re-pin
(`verity-vllm canonical-profiles --write`, reviewed in the diff)."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json

import pytest

from verity_vllm.program import model as analytic
from verity_vllm.observe.events import RunHeader
from verity_vllm.observe.fold.patterns import TokenSelectArgmax
from verity_vllm.engine.profiles import canonical as K
from verity_vllm.engine.profiles import dense_generic as dg
from verity_vllm.engine.profiles import family_facts as FF
from verity_vllm.engine.profiles import generic as G
from verity_vllm.engine.profiles import profile_for_case, profile_module_for_case
from verity_vllm.observe.fold.patterns.collectives import TpCollective
from verity_vllm.collectives.embedding_shard import EmbeddingShardCuda

FIXTURES = sorted(K.EXPECTED_DIR.glob("derived_*.json"))
ROLES = K.fixture_roles()


def _diff(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: {'only expected' if k in a else 'only built'}")
            else:
                _diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: expected {json.dumps(a, default=repr)[:200]} built {json.dumps(b, default=repr)[:200]}")
    return out


# --------------------------------------------------------------------------------------------------------------------- the pinned form

def test_every_shipped_config_has_a_fixture_and_vice_versa():
    """The fixture set IS the shipped-config set (at TP1 and the recorded TP worlds) minus the roles the constructor refuses."""
    names = {p.stem for p in FIXTURES}
    expected = set()
    for role, world in ROLES:
        try:
            G.make_module(f"verity_vllm.engine.profiles.{G.module_name(role, world)}")
        except ImportError:
            continue
        expected.add(G.module_name(role, world))
    assert names == expected, (sorted(names - expected), sorted(expected - names))


@pytest.mark.parametrize("fixture", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_generic_profile_equals_its_fixture(fixture):
    role, world = G.parse_module(fixture.stem)
    built = json.loads(K.dumps(K.canonical_of_role(role, world)))     # through JSON: the fixture's own value domain
    expected = json.loads(fixture.read_text())
    delta = _diff(expected, built)
    assert not delta, f"{fixture.name}: {len(delta)} differences (a deliberate re-pin is `verity-vllm canonical-profiles --write`):\n  " + "\n  ".join(delta[:40])


def test_fixture_config_sha_is_the_manifest_bytes():
    """The config.json a fixture was derived from is the manifest's byte copy (sha256 == checkpoints.json's config_sha256)."""
    for role, world in ROLES:
        if world != 1:
            continue
        hit = dg.find_hf_config(role)
        assert hit is not None, role
        path, source, entry = hit
        assert source == "package:hf_configs", (role, source)
        if entry and entry.get("config_sha256"):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["config_sha256"], role


# ------------------------------------------------------------------------------------------------------------ names, roles, the finder

def test_module_name_grammar():
    assert G.module_name("B1") == "derived_B1" and G.module_name("B1", 2) == "derived_B1_tp2"
    assert G.parse_module("verity_vllm.engine.profiles.derived_QWEN3_30B_A3B_tp2") == ("QWEN3_30B_A3B", 2)
    assert G.parse_module("derived_LLAMA32_1B") == ("LLAMA32_1B", 1)
    assert G.parse_module("dense_MISTRAL7B") == ("MISTRAL7B", 1)                  # the X-07 name recorded runs cite
    assert G.parse_module("gen_llama_llama32_1b") is None and G.parse_module("vllm_d9105ea80_sm89_eager") is None


def test_finder_materialises_every_fixture_module_and_the_x07_name():
    for fixture in FIXTURES:
        mod = importlib.import_module(f"verity_vllm.engine.profiles.{fixture.stem}")
        assert mod.PROFILE.id == json.loads(fixture.read_text())["id"]
        assert callable(mod.weight_field) and callable(mod.root_class) and callable(mod.check_header) and callable(mod.build_profile)
    dense = importlib.import_module("verity_vllm.engine.profiles.dense_MISTRAL7B")
    assert dense.PROFILE.id == importlib.import_module("verity_vllm.engine.profiles.derived_MISTRAL7B").PROFILE.id


def test_the_x07_name_is_for_dense_architectures_only():
    with pytest.raises(ImportError, match="derived_OLMOE"):
        importlib.import_module("verity_vllm.engine.profiles.dense_OLMOE")
    with pytest.raises(ImportError, match="derived_PYTHIA160M"):
        importlib.import_module("verity_vllm.engine.profiles.dense_PYTHIA160M")


def test_profile_module_for_case_is_derived_not_looked_up():
    from verity_vllm.engine import profiles
    assert not hasattr(profiles, "PROFILE_BY_CASE")
    refused = {"QWEN15_AWQ": "awq"}                                             # shipped configs the constructor refuses by name
    for role, world in ROLES:
        if role in refused:
            continue
        assert profile_module_for_case(role, world) == G.module_name(role, world)
        assert profile_for_case(role, world).notes["case"] == role
    for role, why in refused.items():
        with pytest.raises(KeyError, match=why):
            profile_module_for_case(role)
    with pytest.raises(KeyError, match="no HF config.json"):
        profile_module_for_case("NO_SUCH_ROLE")


# ------------------------------------------------------------------------------------------------------------------- family facts

def test_family_facts_rules_agree_with_analytic():
    """The table's `rules` are `analytic.FAMILY_RULES` where analytic has the family (one truth, two readers)."""
    for fam, f in FF.FAMILY_FACTS.items():
        if fam in analytic.FAMILY_RULES:
            assert f["rules"] == analytic.FAMILY_RULES[fam], fam
        assert f["arch"] in ("dense", "dense-ln", "moe", "dsa-moe"), fam
        assert "ForCausalLM" in f["vllm_class"], fam
        assert "vllm/model_executor/models/" in f["rules_provenance"] or "transformers" in f["rules_provenance"], fam


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

TP2_ROLES = sorted(r for r, w in K.fixture_roles() if w == 2)


def test_the_recorded_tp2_worlds_are_pinned():
    """The two retired `*_tp2rank` hand modules and the Qwen3-MoE TP2 half: three roles have a TP2 fixture (data, not a table)."""
    assert len(TP2_ROLES) == 3


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
    mod = importlib.import_module("verity_vllm.engine.profiles.derived_LLAMA32_1B_tp2")
    h = RunHeader("verity-capture/log/v1", "2026-09-23T00:00:00Z", "pid", {}, {"repo": mod.REPO_ID, "config": cfg, "tp": 2}, {"requests": []})
    assert mod.check_header(h) == []
    assert any("tensor_parallel_size" in m for m in mod.check_header(dataclasses.replace(h, model={**h.model, "tp": 1})))
    bad = dataclasses.replace(h, model={**h.model, "config": {**cfg, "num_attention_heads": 30}})
    assert any("num_attention_heads" in m for m in mod.check_header(bad))


def test_world_must_divide_the_sharded_keys():
    with pytest.raises(ImportError, match="not divisible"):
        importlib.import_module("verity_vllm.engine.profiles.derived_B0_tp2")          # SmolLM2-135M: 9 heads


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


def test_the_sparse_attention_family_is_described_and_refused():
    """GLM-5.3's `glm_moe_dsa`: the facts resolve over a TP8 rank's configuration to what the MoE scope and the capture map read, and the
    profile refuses the class (no fold patterns for the indexer or sparse MLA)."""
    from pathlib import Path

    cfg = json.loads((Path(__file__).parents[1] / "check" / "fixtures" / "glm_moe_dsa" / "config.json").read_text())
    assert FF.facts_for(cfg)["arch"] == "dsa-moe"
    assert "sparse attention" in G.structural_refusal(cfg)
    with pytest.raises(ValueError, match="sparse attention"):
        G.derive_config(cfg, repo=None, revision=None, world=8)
    vals, prov = FF.resolve_facts(G.rank_hf_config(cfg, 8), {})
    assert {k: vals[k] for k in ("NUM_EXPERTS", "N_GROUP", "TOPK_GROUP", "RENORMALIZE", "SCORING", "ROUTED_SCALING_FACTOR", "SHARED_I",
                                 "INDEX_HEADS", "INDEX_HEAD_DIM", "INDEX_TOPK", "INDEX_TOPK_FREQ", "INDEX_SKIP_TOPK_OFFSET", "FIRST_DENSE")} == {
        "NUM_EXPERTS": 256, "N_GROUP": 1, "TOPK_GROUP": 1, "RENORMALIZE": True, "SCORING": "sigmoid", "ROUTED_SCALING_FACTOR": 2.5,
        "SHARED_I": 256, "INDEX_HEADS": 32, "INDEX_HEAD_DIM": 128, "INDEX_TOPK": 2048, "INDEX_TOPK_FREQ": 4, "INDEX_SKIP_TOPK_OFFSET": 3,
        "FIRST_DENSE": 3}
    assert vals["ATTN_SCALE"] == 256 ** -0.5 and vals["ROPE_NEOX"] is False and prov["SHARED_I"] == "derived"


def test_olmoe_rank_record_holds_the_whole_all_heads_qk_norm():
    """OLMoE's q/k RMSNorm spans every head (`RMSNorm(total_num_heads * head_dim)`): a TP rank all-gathers q and k first and holds the
    whole replicated weight, so the rank record's `q_norm` / `k_norm` have the full model's width while its heads are the rank's."""
    full, rank = (importlib.import_module(f"verity_vllm.engine.profiles.{G.module_name('OLMOE', w)}") for w in (1, 2))

    def layer(C):
        wt = analytic.weights_type(C)
        return {n: t.leaves for n, t in wt.fields[wt.index("layers")][1].elem.fields}

    f, r = layer(full.C), layer(rank.C)
    assert rank.C["NH"] * 2 == full.C["NH"] and r["o_w"] * 2 == f["o_w"]
    assert r["q_norm"] == f["q_norm"] == full.C["NH"] * full.C["D"] and r["k_norm"] == f["k_norm"] == full.C["KVH"] * full.C["D"]
