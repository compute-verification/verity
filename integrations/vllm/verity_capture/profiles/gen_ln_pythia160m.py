"""[GEN cov-ln] κ profile for EleutherAI/pythia-160m (checkpoint role PYTHIA160M, revision 50f5173d932e8e61f858120bcb800b97af589f46)
under the SAME pins as `vllm_d9105ea80_sm89_eager` (vLLM d9105ea80, torch 2.13.0+cu129, Triton 3.7.1, RTX 4090, eager,
`VLLM_BATCH_INVARIANT=1`, bf16).  vLLM serves the model as `GPTNeoXForCausalLM` (vllm/model_executor/models/gpt_neox.py).

Static configuration `C` (config.json through `analytic.config_of`, family rules passed as `family_rules` — no edit of the shared
table): H=768 NL=12 NH=12 KVH=12 (no GQA) D=64 I=3072 V=50304, QKV=2304 GU=6144 (unused: the MLP has no gate), EPS=1e-5
(`layer_norm_eps`), ROPE_THETA=1e4 (`rotary_emb_base`), MAX_POS=2048, BIAS=True (GPTNeoXAttention: `bias=True` unconditionally),
tie_word_embeddings=False (`embed_out` is its own tensor), FA2 static BN = fa2_kblock_n(64) = 128.

Hand fields (each recorded with provenance "hand"; each a family fact of vLLM's `GPTNeoXForCausalLM` / the checkpoint):
  * `ROT = 16`                       -- `rotary_pct = 0.25`: `get_rope(head_size=64, rotary_dim=int(64 * 0.25))`; the cos/sin cache is `[2048, 16]`
  * `NORM = "layernorm"`             -- `nn.LayerNorm(H, eps)` with weight AND bias (selects `analytic.layernorm_layer_weights_type`)
  * `hidden_act = "gelu"`            -- `get_act_fn("gelu")` = `nn.GELU()` (erf), applied between `dense_h_to_4h` and `dense_4h_to_h`
  * `use_parallel_residual = True`   -- `hidden_states = mlp(ln2(x)) + attn(ln1(x)) + x` (two bf16 adds per layer)
  * `layer_norms = 2`, `dense_bias = True` -- every dense projection (fused qkv, dense, h_to_4h, 4h_to_h) carries a bias

Weights: `weight_field` maps vLLM's GPT-NeoX parameter names (`gpt_neox.layers.L.attention.query_key_value.weight`, ...) onto the
`analytic.weights_type(C)` Record of the LayerNorm family (`ln1, ln1_b, qkv_w, qkv_b, o_w, o_b, ln2, ln2_b, up_w, up_b, down_w,
down_b` per layer; `embed`, `norm`, `norm_b`, `cos_sin [2048][16]`, `lm_head`).  The checkpoint is fp16: vLLM loads it into bf16
parameters (`param.copy_(loaded)`: one RNE per word), so checkpoint identity is `bf16_rne(f32(fp16 word))` — and the fused QKV
tensor is stored head-interleaved `[NH, 3, D]` in the checkpoint while vLLM's parameter is `[3, NH, D]` (`load_weights`
transposes); both facts live in `compare.WeightSource` ([SHARED-CODE lane cov-ln]).
"""
from __future__ import annotations

import re
from pathlib import Path

from verity_capture import analytic
from verity_capture.events import RunHeader

from .gen_ln_patterns import with_ln_patterns
from .vllm_d9105ea80_sm89_eager import build_profile, with_geometry  # noqa: F401

PROFILE_ID = "vllm-d9105ea80-sm89-eager-bf16-pythia-160m"
CASE = "PYTHIA160M"
MODEL = "EleutherAI/pythia-160m"
REVISION = "50f5173d932e8e61f858120bcb800b97af589f46"
#: the registered copy of the checkpoint's config.json (`verity_capture.bench.add_checkpoint`); embedded so the profile is
#: self-contained on a pod (`out/` is not shipped) -- `test_gen_ln` asserts the two agree.
HF_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "hf_configs" / "PYTHIA160M.config.json"
HF_CONFIG: dict = {
    "architectures": ["GPTNeoXForCausalLM"], "bos_token_id": 0, "eos_token_id": 0, "hidden_act": "gelu", "hidden_size": 768,
    "initializer_range": 0.02, "intermediate_size": 3072, "layer_norm_eps": 1e-05, "max_position_embeddings": 2048,
    "model_type": "gpt_neox", "num_attention_heads": 12, "num_hidden_layers": 12, "rotary_emb_base": 10000, "rotary_pct": 0.25,
    "tie_word_embeddings": False, "torch_dtype": "float16", "transformers_version": "4.24.0", "use_cache": True,
    "use_parallel_residual": True, "vocab_size": 50304,
}

#: family facts vLLM hard-codes for `gpt_neox` (not in config.json): the fused QKV / dense / MLP biases are unconditional
FAMILY_RULES = {"gpt_neox": {"qkv_bias": True, "default_theta": 10_000.0}}

ROT = int(HF_CONFIG["hidden_size"] // HF_CONFIG["num_attention_heads"] * HF_CONFIG["rotary_pct"])   # 16

HAND = {
    "model": MODEL,
    "revision": REVISION,
    "ROPE_THETA": float(HF_CONFIG["rotary_emb_base"]),
    "ROT": ROT,
    "NORM": "layernorm",
    "hidden_act": "gelu",
    "use_parallel_residual": True,
    "layer_norms": 2,
    "dense_bias": True,
}


def config() -> tuple[dict, dict]:
    C, prov = analytic.config_of(dict(HF_CONFIG), model=MODEL, revision=REVISION, family_rules=FAMILY_RULES,
                                 hand={k: v for k, v in HAND.items() if k not in ("model", "revision")})
    return with_geometry(C), prov


C, PROVENANCE = config()
assert C["family"] == "gpt_neox" and C["H"] == 768 and C["NL"] == 12 and C["NH"] == 12 and C["KVH"] == 12 and C["D"] == 64, C
assert C["I"] == 3072 and C["V"] == 50304 and C["EPS"] == 1e-5 and C["ROPE_THETA"] == 10000.0 and C["MAX_POS"] == 2048, C
assert C["BIAS"] is True and C["tie_word_embeddings"] is False and C["rope_scaling"] is None and C["BN"] == 128, C
assert C["QKV"] == 2304 and C["ROT"] == 16 and analytic.rotary_dim(C) == 16, C

# ---------------------------------------------------------------------------------------------------------
# weights: vLLM GPT-NeoX parameter / buffer names -> analytic.weights_type(C) Record paths
# ---------------------------------------------------------------------------------------------------------

_LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")
LAYER_FIELDS = {
    "input_layernorm.weight": "ln1", "input_layernorm.bias": "ln1_b",
    "attention.query_key_value.weight": "qkv_w", "attention.query_key_value.bias": "qkv_b",
    "attention.dense.weight": "o_w", "attention.dense.bias": "o_b",
    "post_attention_layernorm.weight": "ln2", "post_attention_layernorm.bias": "ln2_b",
    "mlp.dense_h_to_4h.weight": "up_w", "mlp.dense_h_to_4h.bias": "up_b",
    "mlp.dense_4h_to_h.weight": "down_w", "mlp.dense_4h_to_h.bias": "down_b",
}
TOP_FIELDS = {
    "gpt_neox.embed_in.weight": ("embed",),
    "gpt_neox.final_layer_norm.weight": ("norm",),
    "gpt_neox.final_layer_norm.bias": ("norm_b",),
    "embed_out.weight": ("lm_head",),
}
COS_SIN_BUFFER = "attention.rotary_emb.cos_sin_cache"
ALIAS_OF_COS_SIN = re.compile(r"^gpt_neox\.layers\.[1-9]\d*\.attention\.rotary_emb\.cos_sin_cache$")
IGNORED_ROOTS = (re.compile(r"^gpt_neox\.layers\.\d+\.attention\.attn\.(k_scale|v_scale|q_scale|prob_scale|_k_scale|_v_scale|_q_scale|_prob_scale)$"),)


def weight_field(name: str) -> tuple[str | int, ...] | None:
    if name in TOP_FIELDS:
        return TOP_FIELDS[name]
    m = _LAYER_RE.match(name)
    if m:
        layer, rest = int(m.group(1)), m.group(2)
        if rest == COS_SIN_BUFFER:
            return ("cos_sin",) if layer == 0 else None
        f = LAYER_FIELDS.get(rest)
        if f is not None:
            return ("layers", layer, f)
    return None


def root_class(name: str) -> str:
    if weight_field(name) is not None:
        return "weight"
    if ALIAS_OF_COS_SIN.match(name):
        return "alias-of-cos_sin"
    if any(r.match(name) for r in IGNORED_ROOTS):
        return "ignored"
    return "unknown"


_HF_KEYS = {"hidden_size": "H", "num_hidden_layers": "NL", "num_attention_heads": "NH", "intermediate_size": "I", "vocab_size": "V",
            "layer_norm_eps": "EPS", "rotary_emb_base": "ROPE_THETA", "max_position_embeddings": "MAX_POS",
            "hidden_act": "hidden_act", "use_parallel_residual": "use_parallel_residual", "tie_word_embeddings": "tie_word_embeddings"}


def check_header(header: RunHeader) -> list[str]:
    """Mismatches between the run's recorded HF config and C (the base check hard-codes RMSNorm keys and tied embeddings)."""
    cfg = (header.model or {}).get("config") or {}
    out = []
    for hf, key in _HF_KEYS.items():
        if hf in cfg and cfg[hf] != C[key]:
            out.append(f"{hf}={cfg[hf]!r} != C[{key}]={C[key]!r}")
    if "rotary_pct" in cfg and int(C["D"] * cfg["rotary_pct"]) != C["ROT"]:
        out.append(f"rotary_pct={cfg['rotary_pct']!r}: rot_dim {int(C['D'] * cfg['rotary_pct'])} != C[ROT]={C['ROT']}")
    if "num_key_value_heads" in cfg and cfg["num_key_value_heads"] != C["KVH"]:
        out.append(f"num_key_value_heads={cfg['num_key_value_heads']!r} != C[KVH]={C['KVH']!r}")
    return out


PROFILE = build_profile(C, profile_id=PROFILE_ID, case=CASE,
                        model_note="pythia-160m @ 50f5173d (GPTNeoXForCausalLM; analytic.config_of + hand family facts): H=768 NL=12 "
                                   "NH=KVH=12 D=64 I=3072 V=50304, nn.LayerNorm(eps=1e-5) with bias, erf-GELU MLP (no gate), parallel "
                                   "residual, biases on every dense layer incl. fused qkv, partial rotary rot_dim=16 (rotary_pct 0.25), "
                                   "untied embed_out, fp16 checkpoint loaded as bf16, FA2 kBlockN 128")
PROFILE.weight_field = weight_field
PROFILE.patterns = with_ln_patterns(PROFILE.patterns, C)
PROFILE.notes["hand_fields"] = sorted(HAND)
PROFILE.notes["config_provenance"] = dict(PROVENANCE)
PROFILE.notes["family_rules"] = FAMILY_RULES
PROFILE.notes["alias_of_cos_sin"] = ALIAS_OF_COS_SIN.pattern
PROFILE.notes["ignored_roots"] = [r.pattern for r in IGNORED_ROOTS]
PROFILE.notes["root_class"] = "profiles.gen_ln_pythia160m.root_class(name)"
PROFILE.notes["checkpoint_dtype"] = "float16 (loaded into bf16 parameters: one RNE per word; compare.WeightSource converts)"
