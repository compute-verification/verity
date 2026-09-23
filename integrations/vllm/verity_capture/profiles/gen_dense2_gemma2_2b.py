"""[GEN dense2] κ profile for unsloth/gemma-2-2b (checkpoint role GEMMA2_2B, revision 25319945f7fd83b8b903e12081777b7eef2ba993)
under the SAME pins as `vllm_d9105ea80_sm89_eager` (vLLM d9105ea80, torch 2.13.0+cu129, Triton 3.7.1, RTX 4090, eager,
`VLLM_BATCH_INVARIANT=1`, bf16).  vLLM serves the model as `Gemma2ForCausalLM`.

The static configuration `C` (from the checkpoint's config.json through `analytic.config_of`, family rules
`analytic.FAMILY_RULES["gemma2"]`): H=2304 NL=26 NH=8 KVH=4 D=256 (`head_dim`) I=9216 V=256000, QKV=4096 GU=18432, EPS=1e-6,
ROPE_THETA=1e4, MAX_POS=8192, BIAS=False (`attention_bias: false`), FA2 static BN = fa2_kblock_n(256) = 64.

Hand fields (`hand=` of `config_of`; each recorded with provenance "hand"; every one is a family fact of vLLM's
`Gemma2ForCausalLM` / the checkpoint, NOT something the capture is allowed to assume -- REPORT §2 checks each against the log):
  * `model`, `revision`                 -- the checkpoint identity (checkpoints.json)
  * `tie_word_embeddings = True`        -- config.json has no key; the checkpoint has no `lm_head.weight`; vLLM computes the
                                           logits with `model.embed_tokens` (transformers' Gemma2Config default is True)
  * `attn_logit_softcapping = 50.0`     -- config.json key `config_of` does not lift: FA2 `softcap` (tanh) on the attention logits
  * `final_logit_softcapping = 30.0`    -- config.json key: `LogitsProcessor(soft_cap=30.0)`: logits = 30 * tanh(logits / 30)
  * `query_pre_attn_scalar = 256`       -- config.json key: attention scaling = query_pre_attn_scalar ** -0.5 (== D ** -0.5 here)
  * `sliding_window = 4096`             -- config.json key: even layers (layer_idx % 2 == 0) use a 4096 window, odd layers are global
  * `hidden_act = "gelu_pytorch_tanh"`  -- config.json key: GeGLU, vLLM `GeluAndMul(approximate="tanh")`
  * `embed_normalizer = 48.0`           -- sqrt(H): `Gemma2Model.normalizer` buffer, `hidden_states *= normalizer` after the embedding
  * `rmsnorm_plus_one = True`           -- `GemmaRMSNorm`: x * (1 + w) (the `(1 + w)` is in the norm, the weights are stored as w)
  * `layer_norms = 4`                   -- input_layernorm, post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm

Weights: `weight_field` extends the base map with the two extra per-layer norms (`ln_pre_mlp`, `ln_post_mlp`).  The shared
`analytic.weights_type(C)` (PROTECTED) has no such fields -- the fold raises a precise FoldError on the first Gemma-2 layer
(`weight field path ('layers', 0, 'ln_pre_mlp') ... is not in ...`).  The PROPOSED shared-code change (REPORT §6,
branch `lane/gen-dense2-shared`) adds `C["extra_layer_norms"]` to `analytic.weights_type`; this profile declares the same
tuple in `C` so the two agree the moment the proposal lands.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from verity_capture import analytic
from verity_capture.events import RunHeader

from .gen_dense2_patterns import with_dense2_patterns
from .gen_dense_gemma2_patterns import with_gemma2_patterns
from .gen_dense_softcap_patterns import with_softcap_patterns
from .vllm_d9105ea80_sm89_eager import (build_profile, make_check_header, make_root_class, make_weight_field,  # noqa: F401
                                        with_geometry)

PROFILE_ID = "vllm-d9105ea80-sm89-eager-bf16-gemma-2-2b"
CASE = "GEMMA2_2B"
MODEL = "unsloth/gemma-2-2b"
REVISION = "25319945f7fd83b8b903e12081777b7eef2ba993"
#: the registered copy of the checkpoint's config.json (`verity_capture.bench.add_checkpoint`); the profile embeds the document
#: so it is self-contained on a pod (`out/` is not shipped) -- `test_gen_dense2` asserts the two agree.
HF_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "hf_configs" / "GEMMA2_2B.config.json"
HF_CONFIG: dict = {
    "_name_or_path": "unsloth/gemma-2-2b", "architectures": ["Gemma2ForCausalLM"], "attention_bias": False, "attention_dropout": 0.0,
    "attn_logit_softcapping": 50.0, "bos_token_id": 2, "cache_implementation": "hybrid", "eos_token_id": 1, "final_logit_softcapping": 30.0,
    "head_dim": 256, "hidden_act": "gelu_pytorch_tanh", "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 2304,
    "initializer_range": 0.02, "intermediate_size": 9216, "max_position_embeddings": 8192, "model_type": "gemma2", "num_attention_heads": 8,
    "num_hidden_layers": 26, "num_key_value_heads": 4, "pad_token_id": 0, "query_pre_attn_scalar": 256, "rms_norm_eps": 1e-06,
    "rope_theta": 10000.0, "sliding_window": 4096, "torch_dtype": "bfloat16", "transformers_version": "4.44.2", "unsloth_version": "2024.9",
    "use_cache": True, "vocab_size": 256000,
}

#: the two per-layer norms Llama/Qwen do not have, in the order vLLM applies them (before / after the MLP)
EXTRA_LAYER_NORMS = ("ln_pre_mlp", "ln_post_mlp")

#: `hand` fields passed to `config_of` (each recorded with provenance "hand"); see the module docstring for the fact behind each.
HAND = {
    "model": MODEL,
    "revision": REVISION,
    "tie_word_embeddings": True,
    "attn_logit_softcapping": 50.0,
    "final_logit_softcapping": 30.0,
    "query_pre_attn_scalar": 256,
    "sliding_window": 4096,
    "hidden_act": "gelu_pytorch_tanh",
    "embed_normalizer": 48.0,
    "rmsnorm_plus_one": True,
    "layer_norms": 4,
    "extra_layer_norms": EXTRA_LAYER_NORMS,
}


def config() -> tuple[dict, dict]:
    """(C, provenance) from the checkpoint's config.json (`HF_CONFIG`, the registered `data/hf_configs/GEMMA2_2B.config.json`) + HAND."""
    C, prov = analytic.config_of(dict(HF_CONFIG), model=MODEL, revision=REVISION,
                                 hand={k: v for k, v in HAND.items() if k not in ("model", "revision")})
    return with_geometry(C), prov


C, PROVENANCE = config()
assert C["family"] == "gemma2" and C["H"] == 2304 and C["NL"] == 26 and C["NH"] == 8 and C["KVH"] == 4 and C["D"] == 256, C
assert C["I"] == 9216 and C["V"] == 256000 and C["EPS"] == 1e-6 and C["ROPE_THETA"] == 10000.0 and C["MAX_POS"] == 8192, C
assert C["BIAS"] is False and C["tie_word_embeddings"] is True and C["rope_scaling"] is None and C["BN"] == 64, C
assert C["QKV"] == 4096 and C["GU"] == 18432 and C["embed_normalizer"] == C["H"] ** 0.5, C

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_EXTRA_FIELDS = {"pre_feedforward_layernorm.weight": "ln_pre_mlp", "post_feedforward_layernorm.weight": "ln_post_mlp"}
_base_weight_field = make_weight_field(C)


def weight_field(name: str) -> tuple[str | int, ...] | None:
    """The base map (embed / norm / cos_sin / ln1 / qkv_w / o_w / ln2 / gate_up_w / down_w; `lm_head.weight` -> embed, tied)
    plus Gemma-2's `pre_feedforward_layernorm` -> `ln_pre_mlp` and `post_feedforward_layernorm` -> `ln_post_mlp`, and
    `model.normalizer` (the 0-d bf16 `sqrt(H)` embedding-scale buffer, not a checkpoint tensor) -> the one-word `normalizer`
    field `analytic.weights_type` declares when `C["embed_normalizer"]` is set (root class `analytic-table`, validated exactly by
    `analytic.check_normalizer`; lane cov-dense 2026-09-17 — before that the root was class "unknown" and the fold's very first
    read of the scaled embedding was Unresolved)."""
    if name == "model.normalizer":
        return ("normalizer",)
    m = _LAYER_RE.match(name)
    if m and m.group(2) in _EXTRA_FIELDS:
        return ("layers", int(m.group(1)), _EXTRA_FIELDS[m.group(2)])
    return _base_weight_field(name)


root_class = make_root_class(weight_field)
_base_check_header = make_check_header(C)


def check_header(header: RunHeader) -> list[str]:
    """The base consistency check plus the Gemma-2 softcapping / window / activation facts recorded in the run's config."""
    out = list(_base_check_header(header))
    cfg = (header.model or {}).get("config") or {}
    for hf, key in (("attn_logit_softcapping", "attn_logit_softcapping"), ("final_logit_softcapping", "final_logit_softcapping"),
                    ("query_pre_attn_scalar", "query_pre_attn_scalar"), ("sliding_window", "sliding_window"), ("hidden_act", "hidden_act")):
        if hf in cfg and cfg[hf] != C[key]:
            out.append(f"{hf}={cfg[hf]!r} != C[{key}]={C[key]!r}")
    return out


PROFILE = build_profile(C, profile_id=PROFILE_ID, case=CASE,
                        model_note="gemma-2-2b @ 25319945 (analytic.config_of + hand family facts): H=2304 NL=26 NH=8 KVH=4 D=256 "
                                   "I=9216 V=256000, no QKV bias, tied embeddings, GeGLU (gelu_pytorch_tanh), GemmaRMSNorm x*(1+w) "
                                   "x4 per layer, sqrt(H) embed scale, attention softcap 50 / final softcap 30, sliding window 4096 "
                                   "on even layers, FA2 kBlockN 64")
PROFILE.weight_field = weight_field
PROFILE.patterns = with_dense2_patterns(PROFILE.patterns, C)      # the even layers' 4096 window is inert at this length
PROFILE.patterns = with_gemma2_patterns(PROFILE.patterns)         # [ov-dense] unfused GemmaRMSNorm / GeGLU / softcap chain / embed scale -> quarantine.dense
PROFILE.patterns = with_softcap_patterns(PROFILE.patterns)        # [r8 E-cleanup] FA2 softcap=50 launches -> quarantine.dense.AttentionSoftcap_v1{..., CAP=50.0}
PROFILE.notes["softcap_pattern"] = [p.name for p in PROFILE.patterns if p.name == "attention.fa2_varlen_fwd.softcap"]
PROFILE.notes["hand_fields"] = sorted(HAND)
PROFILE.notes["config_provenance"] = dict(PROVENANCE)
PROFILE.notes["extra_layer_norms"] = list(EXTRA_LAYER_NORMS)
