"""[GEN dense2] κ profile for microsoft/Phi-3-mini-4k-instruct (checkpoint role PHI3_MINI, revision
f39ac1d28e925b323eae81227eaba4464caced4e) under the SAME pins as `vllm_d9105ea80_sm89_eager` (vLLM d9105ea80, torch
2.13.0+cu129, Triton 3.7.1, RTX 4090, eager, `VLLM_BATCH_INVARIANT=1`, bf16).  Nothing model-level is authored here: the
profile is `build_profile(config_of(config.json))`; vLLM serves the model as `Phi3ForCausalLM(LlamaForCausalLM)`.

The static configuration `C` (every key from the checkpoint's config.json through `analytic.config_of`, family rules
`analytic.FAMILY_RULES["phi3"]`): H=3072 NL=32 NH=32 KVH=32 D=96 I=8192 V=32064, QKV=9216 GU=16384, EPS=1e-5,
ROPE_THETA=1e4 (config `rope_theta`; `rope_scaling` is null -- the 4k variant, standard RoPE), MAX_POS=4096,
BIAS=False (`attention_bias: false`), tie_word_embeddings=False (untied `lm_head.weight` -> the `lm_head` field of
`analytic.weights_type`), FA2 static BN = fa2_kblock_n(96) = 64.

Hand fields (`hand=` of `config_of`, listed as the brief asks): `model`, `revision` only (both from checkpoints.json).

Family facts checked against the capture, not assumed (REPORT §2):
  * fused checkpoint weights: Phi-3's safetensors store `self_attn.qkv_proj.weight` [9216, 3072] and `mlp.gate_up_proj.weight`
    [16384, 3072] already fused -- vLLM's parameter names are the SAME as Llama's fused modules, so `make_weight_field(C)` (the
    base map) binds them unchanged.  What differs is the checkpoint-side map `verity_vllm.correspondence.core._LAYER_FIELDS`
    (`qkv_w` <- q_proj + k_proj + v_proj) used by `compare.WeightSource` for the weights' checkpoint identity (root policy /
    tier b): `PHI3_LAYER_FIELDS` below is the fused-name map a WeightSource for this checkpoint needs (see the REPORT).
  * `sliding_window: 2047` in config.json: vLLM's Llama-architecture attention takes no per-layer window, the model-level
    `cache_config.sliding_window` becomes FA2's `window_size` -- for a 48-token request numerically the causal kernel.
    The kernel census / AttentionScope decide (an `Unsupported` if the launch differs from the pinned one).
"""

from __future__ import annotations

import json
from pathlib import Path

from verity_capture import analytic
from verity_capture.events import RunHeader

from .gen_dense2_patterns import with_dense2_patterns
from .vllm_d9105ea80_sm89_eager import (build_profile, make_check_header, make_root_class, make_weight_field,  # noqa: F401
                                        with_geometry)

PROFILE_ID = "vllm-d9105ea80-sm89-eager-bf16-phi3-mini-4k"
CASE = "PHI3_MINI"
MODEL = "microsoft/Phi-3-mini-4k-instruct"
REVISION = "f39ac1d28e925b323eae81227eaba4464caced4e"
#: the registered copy of the checkpoint's config.json (`verity_capture.bench.add_checkpoint`); the profile embeds the document
#: so it is self-contained on a pod (`out/` is not shipped) -- `test_gen_dense2` asserts the two agree.
HF_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "hf_configs" / "PHI3_MINI.config.json"
HF_CONFIG: dict = {
    "_name_or_path": "Phi-3-mini-4k-instruct", "architectures": ["Phi3ForCausalLM"], "attention_bias": False, "attention_dropout": 0.0,
    "auto_map": {"AutoConfig": "configuration_phi3.Phi3Config", "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM"},
    "bos_token_id": 1, "embd_pdrop": 0.0, "eos_token_id": 32000, "hidden_act": "silu", "hidden_size": 3072, "initializer_range": 0.02,
    "intermediate_size": 8192, "max_position_embeddings": 4096, "model_type": "phi3", "num_attention_heads": 32, "num_hidden_layers": 32,
    "num_key_value_heads": 32, "original_max_position_embeddings": 4096, "pad_token_id": 32000, "resid_pdrop": 0.0, "rms_norm_eps": 1e-05,
    "rope_scaling": None, "rope_theta": 10000.0, "sliding_window": 2047, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
    "transformers_version": "4.40.2", "use_cache": True, "vocab_size": 32064,
}

#: `hand` fields passed to `config_of` (each recorded with provenance "hand"): only the checkpoint identity.
HAND = {"model": MODEL, "revision": REVISION}


def config() -> tuple[dict, dict]:
    """(C, provenance) from the checkpoint's config.json (`HF_CONFIG`, the registered `data/hf_configs/PHI3_MINI.config.json`)."""
    C, prov = analytic.config_of(dict(HF_CONFIG), model=MODEL, revision=REVISION)
    return with_geometry(C), prov


C, PROVENANCE = config()
assert C["family"] == "phi3" and C["H"] == 3072 and C["NL"] == 32 and C["NH"] == 32 and C["KVH"] == 32 and C["D"] == 96, C
assert C["I"] == 8192 and C["V"] == 32064 and C["EPS"] == 1e-5 and C["ROPE_THETA"] == 10000.0 and C["MAX_POS"] == 4096, C
assert C["BIAS"] is False and C["tie_word_embeddings"] is False and C["rope_scaling"] is None and C["BN"] == 64, C
assert C["QKV"] == 9216 and C["GU"] == 16384, C

#: checkpoint-side (safetensors) tensor names of the per-layer weights fields for THIS checkpoint (fused projections), in the
#: shape of `verity_vllm.correspondence.core._LAYER_FIELDS`: field -> (capture canonical id, checkpoint tensors concatenated along dim 0).
PHI3_LAYER_FIELDS = {
    "ln1": ("w=layers.{L}.input_layernorm.weight", ["model.layers.{L}.input_layernorm.weight"]),
    "qkv_w": ("w=layers.{L}.self_attn.qkv_proj.weight", ["model.layers.{L}.self_attn.qkv_proj.weight"]),
    "o_w": ("w=layers.{L}.self_attn.o_proj.weight", ["model.layers.{L}.self_attn.o_proj.weight"]),
    "ln2": ("w=layers.{L}.post_attention_layernorm.weight", ["model.layers.{L}.post_attention_layernorm.weight"]),
    "gate_up_w": ("w=layers.{L}.mlp.gate_up_proj.weight", ["model.layers.{L}.mlp.gate_up_proj.weight"]),
    "down_w": ("w=layers.{L}.mlp.down_proj.weight", ["model.layers.{L}.mlp.down_proj.weight"]),
}

weight_field = make_weight_field(C)          # vLLM parameter names are Llama's fused names; `lm_head.weight` -> ("lm_head",)
root_class = make_root_class(weight_field)
_base_check_header = make_check_header(C)


def check_header(header: RunHeader) -> list[str]:
    """The base consistency check minus its tied-embeddings clause: this profile binds the untied `lm_head.weight` to the
    `lm_head` field of `analytic.weights_type(C)` (C["tie_word_embeddings"] is False), so `tie_word_embeddings=False` in the
    run's config is the expected fact, not a mismatch."""
    return [m for m in _base_check_header(header) if not m.startswith("tie_word_embeddings=False")]


PROFILE = build_profile(C, profile_id=PROFILE_ID, case=CASE,
                        model_note="Phi-3-mini-4k-instruct @ f39ac1d2 (analytic.config_of): H=3072 NL=32 NH=32 KVH=32 D=96 I=8192 "
                                   "V=32064, no QKV bias, untied lm_head, rope_scaling null, FA2 kBlockN 64; fused qkv_proj / "
                                   "gate_up_proj checkpoint tensors (PHI3_LAYER_FIELDS); config sliding_window 2047 -> FA2 "
                                   "window_size_left 2046 >= max_seqlen_k: the causal kernel (attention.fa2_varlen_fwd.inert_window)")
#: the lane's more specific patterns ahead of the base ones (`AttentionScopeInertWindow` before `AttentionScope`)
PROFILE.patterns = with_dense2_patterns(PROFILE.patterns, C)
PROFILE.notes["dense2_patterns"] = [p.name for p in PROFILE.patterns if p.name.startswith("attention.fa2_varlen_fwd.inert_window")]
PROFILE.notes["hand_fields"] = sorted(HAND)
PROFILE.notes["config_provenance"] = dict(PROVENANCE)
PROFILE.notes["checkpoint_layer_fields"] = "gen_dense2_phi3_mini.PHI3_LAYER_FIELDS (fused qkv_proj / gate_up_proj safetensors names)"
