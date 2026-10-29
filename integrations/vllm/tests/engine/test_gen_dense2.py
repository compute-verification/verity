"""The two profile modules are exactly `analytic.config_of` + `build_profile` with the listed hand
fields; the weight-name maps bind every parameter name of the served models and nothing else; the workloads mirror the B1
one-request workload (32 prompt tokens, 16 greedy tokens, no accepted tokens); positive AND negative recognizer tests of the
patterns (`gen_dense2_patterns`, when present)."""
from __future__ import annotations

import importlib

import json
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.program import model as analytic
from verity_vllm.observe.events import RunHeader
from verity_vllm.observe.fold.profiles import profile_module_for_case

gemma = importlib.import_module("verity_vllm.observe.fold.profiles.derived_GEMMA2_2B")
phi3 = importlib.import_module("verity_vllm.observe.fold.profiles.derived_PHI3_MINI")
from verity.ml.definitions.registry import b1

REPO = Path(__file__).resolve().parents[2]
PROFILES = REPO / "workloads"

STATIC_KEYS = ("H", "NL", "NH", "KVH", "D", "I", "V", "QKV", "GU", "EPS", "ROPE_THETA", "MAX_POS", "BIAS", "tie_word_embeddings", "BN")


# --- configuration --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mod", [phi3, gemma])
def test_embedded_hf_config_is_the_registered_one(mod):
    if not mod.HF_CONFIG_PATH.exists():
        pytest.skip("data/hf_configs is not shipped to pods")
    assert json.loads(mod.HF_CONFIG_PATH.read_text()) == mod.HF_CONFIG


def test_phi3_config_is_config_of_plus_identity_only():
    C, prov = analytic.config_of(dict(phi3.HF_CONFIG), model=phi3.MODEL, revision=phi3.REVISION)
    assert {k: phi3.C[k] for k in STATIC_KEYS} == {**{k: C[k] for k in STATIC_KEYS if k != "BN"}, "BN": b1.fa2_kblock_n(C["D"])}
    assert phi3.C["family"] == "phi3" and phi3.C["D"] == 96 and phi3.C["BIAS"] is False and phi3.C["tie_word_embeddings"] is False
    assert phi3.C["rope_scaling"] is None and phi3.C["ROPE_THETA"] == 10000.0 and phi3.C["MAX_POS"] == 4096
    assert sorted(k for k, v in prov.items() if v == "hand") == ["model", "revision"] == sorted(phi3.HAND)
    assert phi3.PROFILE.notes["hand_fields"] == ["model", "revision"]
    assert [f for f, _ in analytic.weights_type(phi3.C).fields] == ["embed", "layers", "norm", "cos_sin", "lm_head"]


def test_gemma2_config_hand_fields_are_listed_and_only_those():
    C, prov = analytic.config_of(dict(gemma.HF_CONFIG), model=gemma.MODEL, revision=gemma.REVISION)
    # config.json facts unchanged by the hand fields
    for k in ("H", "NL", "NH", "KVH", "D", "I", "V", "EPS", "ROPE_THETA", "MAX_POS", "BIAS", "QKV", "GU"):
        assert gemma.C[k] == C[k], k
    assert gemma.C["D"] == 256 and gemma.C["BN"] == b1.fa2_kblock_n(256) == 64
    # the one config_of default the hand fields override: no tie_word_embeddings key in config.json -> config_of says False,
    # the checkpoint has no lm_head.weight and vLLM ties (family fact)
    assert C["tie_word_embeddings"] is False and gemma.C["tie_word_embeddings"] is True
    hand = gemma.PROFILE.notes["hand_fields"]
    assert hand == sorted(gemma.HAND)
    assert {k for k, v in gemma.PROVENANCE.items() if v == "hand"} <= set(hand)      # a `key` fact's value is config.json's (hf-config)
    assert set(hand) == {"model", "revision", "tie_word_embeddings", "attn_logit_softcapping", "final_logit_softcapping",
                         "query_pre_attn_scalar", "sliding_window", "hidden_act", "embed_normalizer", "rmsnorm_plus_one",
                         "layer_norms", "extra_layer_norms"}
    assert gemma.C["embed_normalizer"] == 48.0 == gemma.C["H"] ** 0.5 and gemma.C["query_pre_attn_scalar"] == gemma.C["D"]
    # tied: the shared weights Record has no lm_head field; the sqrt(H) embedding-scale buffer is the LAST field
    assert [f for f, _ in analytic.weights_type(gemma.C).fields] == ["embed", "layers", "norm", "cos_sin", "normalizer"]


def test_profiles_are_discovered_by_case_and_ids_are_distinct():
    assert profile_module_for_case("PHI3_MINI") == "derived_PHI3_MINI"
    assert profile_module_for_case("GEMMA2_2B") == "derived_GEMMA2_2B"
    assert profile_module_for_case("B1") == "derived_B1"
    assert phi3.PROFILE.id != gemma.PROFILE.id and phi3.PROFILE.accepted is False and gemma.PROFILE.accepted is False
    assert phi3.PROFILE.config["V"] == 32064 and gemma.PROFILE.config["V"] == 256000


# --- weight-name maps ---------------------------------------------------------------------------------------------------

def _llama_like_names(NL: int, tied: bool, bias: bool) -> list[str]:
    names = ["model.embed_tokens.weight", "model.norm.weight", "model.layers.0.self_attn.rotary_emb.cos_sin_cache"]
    if not tied:
        names.append("lm_head.weight")
    for L in range(NL):
        names += [f"model.layers.{L}.{s}" for s in ("input_layernorm.weight", "self_attn.qkv_proj.weight", "self_attn.o_proj.weight",
                                                    "post_attention_layernorm.weight", "mlp.gate_up_proj.weight", "mlp.down_proj.weight")]
        if bias:
            names.append(f"model.layers.{L}.self_attn.qkv_proj.bias")
    return names


def test_phi3_weight_field_binds_llama_fused_names_and_untied_lm_head():
    wf = phi3.weight_field
    names = _llama_like_names(32, tied=False, bias=False)
    assert len(names) == 4 + 32 * 6 == 196
    paths = [wf(n) for n in names]
    assert None not in paths and len(set(paths)) == len(paths)
    assert wf("lm_head.weight") == ("lm_head",) and wf("model.embed_tokens.weight") == ("embed",)
    assert wf("model.layers.31.self_attn.qkv_proj.weight") == ("layers", 31, "qkv_w")
    assert wf("model.layers.7.mlp.gate_up_proj.weight") == ("layers", 7, "gate_up_w")
    # negatives: no bias field (BIAS False), Gemma-only norms, the rotary alias of a later layer, the KV scale placeholders
    assert wf("model.layers.0.self_attn.qkv_proj.bias") is None
    assert wf("model.layers.0.pre_feedforward_layernorm.weight") is None
    assert wf("model.layers.3.self_attn.rotary_emb.cos_sin_cache") is None and phi3.root_class("model.layers.3.self_attn.rotary_emb.cos_sin_cache") == "alias-of-cos_sin"
    assert phi3.root_class("model.layers.0.self_attn.attn.k_scale") == "ignored"
    assert phi3.root_class("model.normalizer") == "unknown"
    # the fused checkpoint map covers exactly the per-layer fields of the weights Record
    layer_rec = dict(analytic.weights_type(phi3.C).fields)["layers"].elem
    # the fused-name map is the base module's (`make_weight_field`): every per-layer Record field is bound by exactly one root name
    bound = {}
    for suffix in ("input_layernorm.weight", "self_attn.qkv_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
                   "mlp.gate_up_proj.weight", "mlp.down_proj.weight"):
        path = wf(f"model.layers.0.{suffix}")
        assert path is not None and path[:2] == ("layers", 0), suffix
        bound[path[2]] = suffix
    assert sorted(bound) == sorted(f for f, _ in layer_rec.fields)


def test_gemma2_weight_field_adds_the_two_mlp_norms_and_ties_lm_head():
    wf = gemma.weight_field
    names = _llama_like_names(26, tied=True, bias=False)
    names += [f"model.layers.{L}.{s}" for L in range(26) for s in ("pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight")]
    paths = [wf(n) for n in names]
    assert None not in paths and len(set(paths)) == len(paths)
    assert wf("model.layers.4.pre_feedforward_layernorm.weight") == ("layers", 4, "ln_pre_mlp")
    assert wf("model.layers.4.post_feedforward_layernorm.weight") == ("layers", 4, "ln_post_mlp")
    assert wf("lm_head.weight") == ("embed",)                    # tied (vLLM computes the logits with model.embed_tokens)
    assert wf("model.normalizer") == ("normalizer",)              # the one-word analytic table
    assert wf("model.layers.0.self_attn.qkv_proj.bias") is None


# --- the sqrt(H) embedding-scale buffer as an analytic-table root --------------------------------

def _f32_path_word(H: int) -> int:
    """The other plausible construction (`torch.tensor(sqrt(H))` as f32, then `.to(bf16)`): must agree for every Gemma-2 width."""
    return int(analytic.f32_to_bf16_words(np.array([np.float32(float(H) ** 0.5)], dtype=np.float32))[0])


@pytest.mark.parametrize("H, word, value", [(2304, 0x4240, 48.0), (3584, 0x426F, 59.75), (4608, 0x4288, 68.0), (1, 0x3F80, 1.0), (2, 0x3FB5, 1.4140625)])
def test_normalizer_word_is_bf16_of_sqrt_h(H, word, value):
    w = analytic.normalizer_word(H)
    assert w == word == _f32_path_word(H)
    assert np.array([w << 16], dtype=np.uint32).view(np.float32)[0] == np.float32(value)


def test_normalizer_word_matches_torch_bf16_rounding_on_many_widths():
    """Nearest-even from the Python float agrees with the f32-intermediate path on every hidden width up to 2^15 except where
    the two roundings legitimately differ (none below 32768 — the assertion is the evidence)."""
    for H in range(1, 32768):
        assert analytic.normalizer_word(H) == _f32_path_word(H), H


def test_check_normalizer_is_exact_and_reads_the_hand_value():
    C = dict(gemma.C)
    assert analytic.check_normalizer(C, np.array([0x4240], dtype=np.uint16))["status"] == "pass"
    assert analytic.check_normalizer(C, [0x4240])["status"] == "pass"
    bad = analytic.check_normalizer(C, np.array([0x4241], dtype=np.uint16))
    assert bad["status"] == "fail" and "0x4241" in bad["reason"] and bad["expected_word"] == 0x4240
    two = analytic.check_normalizer(C, np.array([0x4240, 0x4240], dtype=np.uint16))
    assert two["status"] == "fail" and "2 elements" in two["reason"]
    drift = analytic.check_normalizer({**C, "embed_normalizer": 47.0}, np.array([0x4240], dtype=np.uint16))
    assert drift["status"] == "fail" and "embed_normalizer" in drift["reason"]
    # a configuration without the hand field has no normalizer table and no field
    assert "normalizer" not in dict(analytic.weights_type({k: v for k, v in C.items() if k != "embed_normalizer"}).fields)
    assert "normalizer" not in dict(analytic.weights_type(phi3.C).fields)


def test_normalizer_field_is_one_bf16_leaf_after_every_existing_interval():
    from verity_vllm.program.frontend.rules.family import _field_intervals

    wt = analytic.weights_type(gemma.C)
    ivs = _field_intervals(wt, "")
    lo, hi, name = ivs[-1]
    assert name == "normalizer" and hi - lo == 1 and hi == wt.leaves
    without = analytic.weights_type({k: v for k, v in gemma.C.items() if k != "embed_normalizer"})
    assert [(a, b, n) for a, b, n in ivs[:-1]] == list(_field_intervals(without, ""))   # every existing interval unchanged


def test_weight_source_serves_the_normalizer_word_without_a_checkpoint():
    from verity_vllm.check.fold_compare import WeightSource
    from verity_vllm.program.frontend.rules.family import _field_intervals

    wt = analytic.weights_type(gemma.C)
    ws = WeightSource(wt, safetensors=[], cos_sin=None, repo=None)   # no safetensors: only the analytic fields are servable
    lo, hi, name = _field_intervals(wt, "")[-1]
    assert name == "normalizer"
    assert ws.words(lo, hi).tolist() == [0x4240]


def test_weight_source_serves_gemma2_pre_and_post_mlp_norms_from_the_checkpoint_names():
    """`analytic.weights_type` adds `ln_pre_mlp` / `ln_post_mlp` per layer from `C["extra_layer_norms"]`; `WeightSource` must
    know their checkpoint tensors (`model.layers.L.pre_feedforward_layernorm.weight`, `...post_feedforward_layernorm.weight`)
    or tier A skips every GemmaRMSNorm instance that reads them ('parameter 1 leaves [...]: ln_pre_mlp').  Llama / Phi-3
    Records add no such field."""
    from verity_vllm.check.fold_compare import WeightSource, _extra_norm_layer_fields

    wt = analytic.weights_type(gemma.C)
    fields = _extra_norm_layer_fields(wt)
    assert fields == {"ln_pre_mlp": ("w=layers.{L}.pre_feedforward_layernorm.weight", ["model.layers.{L}.pre_feedforward_layernorm.weight"]),
                      "ln_post_mlp": ("w=layers.{L}.post_feedforward_layernorm.weight", ["model.layers.{L}.post_feedforward_layernorm.weight"])}
    assert _extra_norm_layer_fields(analytic.weights_type(phi3.C)) == {}

    ws = WeightSource(wt, safetensors=[], cos_sin=None, repo=None)
    H = gemma.C["H"]
    ws.tensors = {"model.layers.3.pre_feedforward_layernorm.weight": np.full(H, 0x3F80, dtype=np.uint16),
                  "model.layers.3.post_feedforward_layernorm.weight": np.full(H, 0x4000, dtype=np.uint16)}
    assert ws._field_words("layers[3].ln_pre_mlp").tolist() == [0x3F80] * H
    assert ws._field_words("layers[3].ln_post_mlp").tolist() == [0x4000] * H
    from verity_vllm.program.frontend.rules.family import _field_intervals
    lo, hi, name = next((lo, hi, n) for lo, hi, n in _field_intervals(wt, "") if n == "layers[3].ln_pre_mlp")
    assert hi - lo == H and ws.words(lo, hi).tolist() == [0x3F80] * H


def test_gemma2_extra_norm_fields_vs_the_shared_weights_record():
    """Without the PROPOSED `analytic.weights_type` change (REPORT §6) the shared layer Record has the six base fields only and
    the fold raises a precise FoldError on `('layers', 0, 'ln_pre_mlp')`; with it, the two extra `[H]` bf16 norms follow the
    base fields in the order the profile declares (`C["extra_layer_norms"]`).  Either way the base fields are untouched."""
    layer_rec = dict(analytic.weights_type(gemma.C).fields)["layers"].elem
    names = [f for f, _ in layer_rec.fields]
    base = ["ln1", "qkv_w", "o_w", "ln2", "gate_up_w", "down_w"]
    assert names in (base, base + list(gemma.C["extra_layer_norms"])), names
    for f in gemma.C["extra_layer_norms"]:
        if f in dict(layer_rec.fields):
            assert dict(layer_rec.fields)[f].leaves == gemma.C["H"]


# --- header consistency -------------------------------------------------------------------------------------------------

def _header(cfg: dict) -> RunHeader:
    return RunHeader(schema="test", created_utc="", profile_id="x", profile={}, model={"config": cfg}, workload={})


def test_check_header_phi3_accepts_untied_and_rejects_shape_drift():
    assert phi3.check_header(_header(dict(phi3.HF_CONFIG))) == []
    bad = dict(phi3.HF_CONFIG, hidden_size=3073)
    assert any("hidden_size" in m for m in phi3.check_header(_header(bad)))


def test_check_header_gemma2_checks_the_softcap_and_window_facts():
    assert gemma.check_header(_header(dict(gemma.HF_CONFIG))) == []
    for key, wrong in (("attn_logit_softcapping", 30.0), ("final_logit_softcapping", 50.0), ("sliding_window", 2048), ("hidden_act", "silu"),
                       ("query_pre_attn_scalar", 128)):
        ms = gemma.check_header(_header(dict(gemma.HF_CONFIG, **{key: wrong})))
        assert ms and key in ms[0], (key, ms)


# --- workloads ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, case, model, V", [("workload_phi3_mini_32x16_1req.json", "PHI3_MINI", phi3.MODEL, 32064),
                                                  ("workload_gemma2_2b_32x16_1req.json", "GEMMA2_2B", gemma.MODEL, 256000)])
def test_workloads_mirror_the_b1_one_request_workload(name, case, model, V):
    ref = json.loads((PROFILES / "workload_qwen15_32x16_1req.json").read_text())
    doc = json.loads((PROFILES / name).read_text())
    assert doc["schema"] == ref["schema"] == "verity-vllm/workload/v2" and doc["case"] == case and doc["model"] == model
    assert doc["variants"] == ref["variants"] and doc["prompt_lengths"] == [32] and doc["max_tokens"] == [16]
    (r,) = doc["requests"]
    assert len(r["prompt_token_ids"]) == 32 == r["prompt_len"] and r["max_tokens"] == 16
    assert all(0 <= t < V for t in r["prompt_token_ids"])
    assert "accepted_token_ids" not in r and "accepted_from" not in r        # no accepted tokens: the control run is the reference


# --- patterns: AttentionScopeInertWindow (Phi-3 `sliding_window` 2047 -> FA2 window_size_left 2046 >= max_seqlen_k) ----------

import dataclasses

from verity_vllm.observe.events import DispatchEnter
from verity_vllm.observe.fold.patterns import AttentionScope
from verity_vllm.observe.fold.profiles import vllm_d9105ea80_sm89_eager as B1PROF
from verity_vllm.observe.fold.profiles.gen_dense2_patterns import AttentionScopeInertWindow, with_dense2_patterns
from verity_vllm.observe.fold.resolver import Compute, Unsupported, resolve_step
from tests.observe.test_patterns_synthetic import by_spec, context, decode_step
from verity_vllm.observe.fold.tree import step_events

FA2 = "_vllm_fa2_C.varlen_fwd.default"
SCOPE = "vllm.unified_attention_with_output.default"


def _fa2_edit(events, **updates):
    """The synthetic decode step with the named FA2 schema arguments replaced (positional binding, as vLLM emits them)."""
    from verity_vllm.observe.fold.patterns import schema_param_names
    for i, e in enumerate(events):
        if isinstance(e, DispatchEnter) and e.op == FA2:
            names = schema_param_names(e.schema)
            args = list(e.args)
            for k, v in updates.items():
                args[names.index(k)] = v
            events[i] = dataclasses.replace(e, args=args)
    return events


def _ctx_with_dense2(start, sample):
    ctx = context(start, sample)
    prof = dataclasses.replace(ctx.profile, patterns=with_dense2_patterns(ctx.profile.patterns, ctx.config))
    return dataclasses.replace(ctx, profile=prof)


def _attention_specs(res):
    return sorted(k for k in by_spec(res) if k.startswith("Attention_v3{"))


def test_dense2_patterns_insert_the_inert_window_scope_ahead_of_the_base_scope():
    ps = with_dense2_patterns(B1PROF.PROFILE.patterns, B1PROF.C)
    kinds = [type(p) for p in ps]
    i, j = kinds.index(AttentionScopeInertWindow), kinds.index(AttentionScope)
    assert i == j - 1 and kinds.count(AttentionScopeInertWindow) == 1 == kinds.count(AttentionScope)
    assert ps[i].bn == ps[j].bn and ps[i].num_splits == ps[j].num_splits and ps[i].name == "attention.fa2_varlen_fwd.inert_window"
    assert len(ps) == len(B1PROF.PROFILE.patterns) + 1 and [p for p in ps if p is not ps[i]] == B1PROF.PROFILE.patterns
    assert phi3.PROFILE.notes["dense2_patterns"] == ["attention.fa2_varlen_fwd.inert_window"]
    assert any(type(p) is AttentionScopeInertWindow for p in gemma.PROFILE.patterns)


def test_base_scope_alone_reports_the_phi3_window_as_unsupported():
    events, start = decode_step()
    se = step_events(_fa2_edit(events, window_size_left=2046, window_size_right=0))
    res = resolve_step(se, context(start, se.sample))
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert len(uns) == 1 and uns[0].op == SCOPE and "sliding window" in uns[0].reason


@pytest.mark.parametrize("left", [2046, 4095, 33])       # Phi-3 (2047-1), Gemma-2 (4096-1), and the boundary left == max_seqlen_k (33 here)
def test_inert_window_resolves_to_the_same_attention_computes_as_the_windowless_launch(left):
    events, start = decode_step()
    fa = next(e for e in events if isinstance(e, DispatchEnter) and e.op == FA2)
    from verity_vllm.observe.fold.patterns import schema_param_names
    assert fa.args[schema_param_names(fa.schema).index("max_seqlen_k")] == 33 and left >= 33
    se0 = step_events(list(events))
    base = resolve_step(se0, context(start, se0.sample))
    events, start = decode_step()
    se = step_events(_fa2_edit(events, window_size_left=left, window_size_right=0))
    res = resolve_step(se, _ctx_with_dense2(start, se.sample))
    assert not [r for r in res if isinstance(r, Unsupported)]
    assert _attention_specs(res) == _attention_specs(base) == ["Attention_v3{T=33,NH=9,KVH=3,D=64,BN=128}"]
    (c,) = by_spec(res)["Attention_v3{T=33,NH=9,KVH=3,D=64,BN=128}"]
    (c0,) = by_spec(base)["Attention_v3{T=33,NH=9,KVH=3,D=64,BN=128}"]
    assert (c.operands, c.outputs, c.attribution, c.consumed) == (c0.operands, c0.outputs, c0.attribution, c0.consumed)
    scope = next(r for r in res if getattr(r, "pattern", "") == "attention.fa2_varlen_fwd.inert_window")
    assert set(scope.consumed) == next(n for n in se.roots if n.op == SCOPE).ids()


def test_a_real_local_window_is_still_unsupported_with_the_dense2_patterns():
    events, start = decode_step()
    se = step_events(_fa2_edit(events, window_size_left=16, window_size_right=0))     # 16 < max_seqlen_k 33: a local kernel
    res = resolve_step(se, _ctx_with_dense2(start, se.sample))
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert len(uns) == 1 and uns[0].op == SCOPE and "sliding window" in uns[0].reason


@pytest.mark.parametrize("edit, needle", [({"window_size_left": 2046, "window_size_right": 0, "is_causal": False}, "not causal"),
                                          ({"window_size_left": 2046, "window_size_right": 5}, "right window")])
def test_inert_window_pattern_rejects_non_causal_and_right_windows(edit, needle):
    events, start = decode_step()
    se = step_events(_fa2_edit(events, **edit))
    res = resolve_step(se, _ctx_with_dense2(start, se.sample))
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert len(uns) == 1 and uns[0].op == SCOPE and needle in uns[0].reason, [u.reason for u in uns]


# --- PROPOSED analytic._inv_freq: ATen CUDA scalar division is a reciprocal multiply ----------------------------------------

def test_inv_freq_models_cuda_reciprocal_multiply_and_is_identity_for_power_of_two_D():
    import numpy as np
    for D, theta in ((64, 500000.0), (128, 1000000.0), (256, 10000.0)):
        C = {"D": D, "ROPE_THETA": theta, "rope_scaling": None}
        div = (np.float32(1.0) / np.power(np.float32(theta), (np.arange(0, D, 2, dtype=np.float32) / np.float32(D)).astype(np.float32),
                                          dtype=np.float32)).astype(np.float32)
        assert np.array_equal(analytic._inv_freq(C).view(np.uint32), div.view(np.uint32)), D
    C = {"D": 96, "ROPE_THETA": 10000.0, "rope_scaling": None}
    inv = analytic._inv_freq(C)
    div = (np.float32(1.0) / np.power(np.float32(10000.0), (np.arange(0, 96, 2, dtype=np.float32) / np.float32(96)).astype(np.float32),
                                      dtype=np.float32)).astype(np.float32)
    ulps = inv.view(np.int32).astype(np.int64) - div.view(np.int32).astype(np.int64)
    assert (ulps != 0).sum() == 14 and ulps.min() == -9 and ulps.max() == 0, ulps.tolist()      # the D=96 discrepancy this models


def test_phi3_captured_rotary_table_is_within_the_analytic_bound():
    p = Path(__file__).resolve().parents[2] / "out" / "gen" / "lanes" / "dense2" / "runs" / "phi3_mini_32x16" / "capture" / "cos_sin_cache.npy"
    if not p.exists():
        pytest.skip("run artefact not present")
    r = analytic.check_cos_sin(phi3.C, p)
    assert r["status"] == "pass" and r["n_outside_bound"] == 0 and r["max_ulp"] <= 1, {k: r[k] for k in ("n_outside_bound", "max_ulp", "n_diff")}
