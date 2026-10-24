"""[GEN cov-ln] the lane's tests: the Pythia-160M profile is `analytic.config_of` + `build_profile` with the listed hand fields;
its weight-name map binds every parameter / buffer name of vLLM's `GPTNeoXForCausalLM` and nothing else; the LayerNorm-family
weights Record; the shared-code adapters (F16 checkpoint words, the fused-QKV permutation, the family-neutral root-name
classes); the lane's Definitions (GeluErfBf16's measured-table facts, LayerNormAten's body on a small row); the patterns'
placement in the profile's list.  Bit-exactness of the Definitions is the pod difftest's business, not this file's."""
from __future__ import annotations

import importlib

import json
import re

import numpy as np
import pytest

from verity_vllm.program import model as analytic; from verity_vllm.check import root_policy
from verity_vllm.check.fold_compare import _f16_words_to_bf16, _layernorm_family_fields, _qkv_neox_permute
from verity_vllm.observe.events import RunHeader
from verity_vllm.engine.profiles import profile_module_for_case
from verity_vllm.engine.profiles.gen_ln_patterns import (GeluErfPattern, LayerNormAtenPattern, ResidualAddPattern,
                                                     RotaryEmbeddingPartial)

pythia = importlib.import_module("verity_vllm.engine.profiles.derived_PYTHIA160M")
from verity_vllm.observe.fold.patterns import BiasAdd, RotaryEmbedding
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import act, gelu_erf_table, layer_norm

C = pythia.C


# --- configuration --------------------------------------------------------------------------------------------------------

def test_embedded_hf_config_is_the_registered_one():
    if not pythia.HF_CONFIG_PATH.exists():
        pytest.skip("data/hf_configs is not shipped to pods")
    assert json.loads(pythia.HF_CONFIG_PATH.read_text()) == pythia.HF_CONFIG


def test_config_is_config_of_plus_the_listed_hand_fields():
    base, _ = analytic.config_of(dict(pythia.HF_CONFIG), model=pythia.MODEL, revision=pythia.REVISION, family_rules=pythia.FAMILY_RULES)
    for k in ("H", "NL", "NH", "KVH", "D", "I", "V", "EPS", "ROPE_THETA", "MAX_POS", "BIAS", "QKV", "GU", "tie_word_embeddings"):
        assert C[k] == base[k], k
    assert (C["H"], C["NL"], C["NH"], C["KVH"], C["D"], C["I"], C["V"]) == (768, 12, 12, 12, 64, 3072, 50304)
    assert C["BIAS"] is True and C["tie_word_embeddings"] is False and C["BN"] == b1.fa2_kblock_n(64) == 128
    hand = pythia.PROFILE.notes["hand_fields"]
    assert hand == sorted(pythia.HAND)
    assert {k for k, v in pythia.PROVENANCE.items() if v == "hand"} <= set(hand)     # [P6] a `key` fact's value is config.json's (hf-config)
    assert set(hand) == {"model", "revision", "ROPE_THETA", "ROT", "NORM", "hidden_act", "use_parallel_residual", "layer_norms", "dense_bias"}
    assert C["ROT"] == 16 == int(64 * pythia.HF_CONFIG["rotary_pct"]) and analytic.rotary_dim(C) == 16


def test_profile_is_discovered_by_case():
    assert profile_module_for_case("PYTHIA160M") == "derived_PYTHIA160M"
    assert pythia.PROFILE.notes["case"] == "PYTHIA160M" and pythia.PROFILE.id == pythia.PROFILE_ID


# --- weights Record + names -----------------------------------------------------------------------------------------------

def test_layernorm_family_weights_record():
    wt = analytic.weights_type(C)
    assert [f for f, _ in wt.fields] == ["embed", "layers", "norm", "norm_b", "cos_sin", "lm_head"]
    layer = dict(wt.fields)["layers"].elem
    assert [f for f, _ in layer.fields] == ["ln1", "ln1_b", "qkv_w", "qkv_b", "o_w", "o_b", "ln2", "ln2_b", "up_w", "up_b", "down_w", "down_b"]
    cs = dict(wt.fields)["cos_sin"]
    assert cs.n == 2048 and cs.elem.n == 16                       # partial rotary: the table is [MAX_POS, ROT]
    assert analytic.cos_sin_table(C).shape == (2048, 16)
    # RMSNorm families are unchanged
    from verity_vllm.engine.profiles import vllm_d9105ea80_sm89_eager as base
    assert analytic.weights_type(base.C).fields == b1.weights_type(base.C).fields or "lm_head" in dict(analytic.weights_type(base.C).fields)


def _vllm_gpt_neox_names(NL: int) -> list[str]:
    names = ["gpt_neox.embed_in.weight", "gpt_neox.final_layer_norm.weight", "gpt_neox.final_layer_norm.bias", "embed_out.weight"]
    per = ["input_layernorm.weight", "input_layernorm.bias", "attention.query_key_value.weight", "attention.query_key_value.bias",
           "attention.dense.weight", "attention.dense.bias", "post_attention_layernorm.weight", "post_attention_layernorm.bias",
           "mlp.dense_h_to_4h.weight", "mlp.dense_h_to_4h.bias", "mlp.dense_4h_to_h.weight", "mlp.dense_4h_to_h.bias"]
    for L in range(NL):
        names += [f"gpt_neox.layers.{L}.{p}" for p in per]
    return names


def test_weight_field_binds_every_parameter_and_the_layer0_rotary_buffer_only():
    names = _vllm_gpt_neox_names(C["NL"])
    bound = {n: pythia.weight_field(n) for n in names}
    assert all(v is not None for v in bound.values()), [n for n, v in bound.items() if v is None]
    assert bound["gpt_neox.layers.3.attention.query_key_value.bias"] == ("layers", 3, "qkv_b")
    assert bound["embed_out.weight"] == ("lm_head",) and bound["gpt_neox.final_layer_norm.bias"] == ("norm_b",)
    assert pythia.weight_field("gpt_neox.layers.0.attention.rotary_emb.cos_sin_cache") == ("cos_sin",)
    assert pythia.weight_field("gpt_neox.layers.1.attention.rotary_emb.cos_sin_cache") is None
    assert pythia.root_class("gpt_neox.layers.1.attention.rotary_emb.cos_sin_cache") == "alias-of-cos_sin"
    assert pythia.root_class("gpt_neox.layers.5.attention.attn.k_scale") == "ignored"
    assert pythia.root_class("model.layers.0.self_attn.qkv_proj.weight") == "unknown"
    # every Record field is reachable from some name
    fields = {f for f, _ in dict(analytic.weights_type(C).fields)["layers"].elem.fields}
    assert {v[2] for v in bound.values() if len(v) == 3} == fields


def test_root_policy_frozen_classes_are_family_neutral():
    for fam in ("model.layers.4.self_attn", "gpt_neox.layers.4.attention"):
        assert root_policy.frozen_root_class(f"{fam}.rotary_emb.cos_sin_cache", lambda n: None) == "alias-of-cos_sin"
        assert root_policy.frozen_root_class(f"{fam}.attn._k_scale", lambda n: None) == "ignored"
    assert not root_policy.ALIAS_OF_COS_SIN.match("gpt_neox.layers.0.attention.rotary_emb.cos_sin_cache")


def test_compare_family_fields_and_checkpoint_conversions():
    top, layer = _layernorm_family_fields(C)
    assert top["lm_head"][1] == ["embed_out.weight"] and top["norm_b"][1] == ["gpt_neox.final_layer_norm.bias"]
    assert layer["qkv_w"][1] == ["gpt_neox.layers.{L}.attention.query_key_value.weight"]
    assert _layernorm_family_fields({"NORM": "rmsnorm"}) is None and _layernorm_family_fields(None) is None
    # fp16 -> bf16: exact widening then RNE (1 + 2^-10 -> 1.0; 65504 -> 65536; min subnormal 2^-24 keeps its value)
    w = np.array([0x3C00, 0x3C01, 0x3C03, 0xBC00, 0x7BFF, 0x0001], dtype=np.uint16)
    assert [hex(v) for v in _f16_words_to_bf16(w)] == ["0x3f80", "0x3f80", "0x3f80", "0xbf80", "0x4780", "0x3380"]
    # the checkpoint's [NH, 3, D] rows -> vLLM's [3, NH, D] rows
    NH, D, H = 2, 4, 5
    q = np.arange(NH * 3 * D * H).reshape(NH * 3 * D, H)
    p = _qkv_neox_permute(q, NH, D)
    for which in range(3):
        for nh in range(NH):
            for d in range(D):
                assert (p[which * NH * D + nh * D + d] == q[nh * 3 * D + which * D + d]).all()


# --- header check -----------------------------------------------------------------------------------------------------------

def test_check_header_accepts_the_registered_config_and_reports_drift():
    def header(cfg):
        return RunHeader("verity-capture/log/v1", "2026-09-23T00:00:00Z", "pid", {}, {"repo": pythia.MODEL, "config": cfg}, {"requests": []})
    assert pythia.check_header(header(dict(pythia.HF_CONFIG))) == []
    out = pythia.check_header(header({**pythia.HF_CONFIG, "rotary_pct": 1.0, "use_parallel_residual": False}))
    assert any("rotary_pct" in m for m in out) and any("use_parallel_residual" in m for m in out)


# --- Definitions --------------------------------------------------------------------------------------------------------------

def test_gelu_table_is_measured_and_satisfies_the_kernel_facts():
    assert gelu_erf_table.SHA256 and len(gelu_erf_table.TABLE_WORDS) == 2 * 65536
    g = act.gelu_erf_bf16_bits
    assert g(0x0000) == 0x0000 and g(0x8000) == 0x8000                    # +-0
    assert g(0x7F80) == 0x7F80 and g(0xFF80) == 0x7FFF                    # +inf -> +inf, -inf -> NaN (-inf * 0)
    assert g(0x7FC0) == 0x7FFF and g(0xFFC0) == 0x7FFF                    # NaN -> canonical
    f = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
    big = np.flatnonzero(np.isfinite(f) & (f >= 16.0))
    assert all(g(int(x)) == int(x) for x in big)                          # erff = 1 -> gelu(x) = x
    neg = np.flatnonzero(np.isfinite(f) & (f <= -16.0))
    assert all(g(int(x)) == 0x8000 for x in neg)                          # erff = -1 -> (-x * 0.5) * 0 = -0
    assert g(0x3F80) == 0x3F57 and g(0xBF80) == 0xBE22                    # gelu(1) = 0.8413 -> 0.83984375; gelu(-1) -> -0.158203125


def test_layer_norm_and_gelu_erf_are_registry_definitions():
    from verity.ml.circuits.defs import REGISTRY
    fns = (layer_norm.LayerNormAten, layer_norm.LayerNormAtenV2, act.GeluErf, act.GeluErfBf16)
    assert [f.id for f in fns] == ["LayerNormAten_v1", "LayerNormAten_v2", "GeluErf_v1", "GeluErfBf16_v1"]
    assert all(REGISTRY[f.id] is f for f in fns)
    assert layer_norm.VEC == 4 and layer_norm.NTHREADS == 128 and layer_norm.MEAN_FMA_ORDER == "A"


def test_layer_norm_body_matches_a_float64_layer_norm_within_bf16_rounding():
    from verity_vllm.properties.admission import evaluate_spec
    from verity.ml.circuits.defs import bind
    N = 64
    rng = np.random.default_rng(0)
    x, w, b = (rng.standard_normal(N).astype(np.float32).view(np.uint32) >> 16 for _ in range(3))
    got = np.array(evaluate_spec(bind(layer_norm.LayerNormAten, N=N, EPS=1e-5), [a.astype(np.uint16) for a in (x, w, b)]), dtype=np.uint32)

    def bf(u):
        return (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    xf, wf, bf_ = bf(x), bf(w), bf(b)
    ref = wf * (xf - xf.mean()) / np.sqrt(((xf - xf.mean()) ** 2).mean() + 1e-5) + bf_
    assert np.abs(bf(got) - ref).max() <= 2 ** -7 * np.abs(ref).max()   # one bf16 ulp at the row's scale


# --- patterns in the profile ----------------------------------------------------------------------------------------------------

def test_patterns_are_placed_ahead_of_the_base_rotary_and_bias_patterns():
    kinds = [type(p) for p in pythia.PROFILE.patterns]
    assert kinds.index(RotaryEmbeddingPartial) < kinds.index(RotaryEmbedding)
    for k in (LayerNormAtenPattern, GeluErfPattern, ResidualAddPattern):
        assert kinds.index(k) < kinds.index(BiasAdd)
    ln = next(p for p in pythia.PROFILE.patterns if isinstance(p, LayerNormAtenPattern))
    rp = next(p for p in pythia.PROFILE.patterns if isinstance(p, RotaryEmbeddingPartial))
    assert ln.eps == 1e-5 and rp.rot == 16
    assert re.match(pythia.PROFILE.notes["alias_of_cos_sin"], "gpt_neox.layers.7.attention.rotary_emb.cos_sin_cache")
