"""[P6, 2026-09-23] The declared FAMILY FACTS table: what vLLM d9105ea80 hard-codes per HF `model_type` and `config.json` does not say.

This is the ONE place in `verity_vllm/engine/profiles/` where a model family name may appear (by-name lint class L, `retires: stays`;
`tests/by_name_allowlist.json`, decision "P6 2026-09-23: family facts table is the declared home of vLLM's per-family hard-codes").
Everything else a κ profile needs is a function of the role's `config.json` (`analytic.config_of`) and of the κ pins
(`vllm_d9105ea80_sm89_eager`); `profiles.generic.build_profile(role)` is that function.  Before P6 these facts were spread over nine
hand-written per-model modules (`gen_dense2_gemma2_2b`, `gen_ln_pythia160m`, `gen_moe_olmoe`, `gen_ov_moe_qwen3`, ...); their findings
harvest is on Notion (findings index 3e4399515d9e8188ade0c8d9164c85b6).

Every fact cites the vLLM source that hard-codes it (`provenance`: file:line @ d9105ea80, read from the pinned tree).  Kinds:
  * `const`  -- a value vLLM's model code fixes (Gemma-2's `x * (1 + w)` norm; OLMoE's `renormalize=False`; GPT-NeoX's LayerNorm);
  * `key`    -- a `config.json` key vLLM's model code READS that `analytic.config_of` does not lift (softcaps, sliding window, activation,
                partial rotary, router renormalisation); `default` when the key may be absent (Gemma-2's `tie_word_embeddings`);
  * `derive` -- a value computed from the configuration the way vLLM computes it (Gemma-2's `normalizer = sqrt(H)`; Pythia's rotary width).
`rules` are the `analytic.config_of` family rules (QKV bias, default rope theta, per-head q/k norm); for the families `analytic.FAMILY_RULES`
already lists they MUST agree (`tests/test_profiles_generic.py` pins that), the others (`gpt_neox`, `qwen3_moe`) are passed as `family_rules`.
`arch` selects the constructor class in `profiles.generic` (`dense`, `dense-ln`, `moe`; `dsa-moe` is described and refused); `params` are the architecture's vLLM parameter
names that the base Llama map (`vllm_d9105ea80_sm89_eager._LAYER_FIELDS`) does not know, per weights-Record field.
"""
from __future__ import annotations

import math
from typing import Any

VLLM = "vllm @ d9105ea80"
_LLAMA = "vllm/model_executor/models/llama.py"
_QWEN2 = "vllm/model_executor/models/qwen2.py"
_QWEN3 = "vllm/model_executor/models/qwen3.py"
_GEMMA2 = "vllm/model_executor/models/gemma2.py"
_NEOX = "vllm/model_executor/models/gpt_neox.py"
_PHI3 = "vllm/model_executor/models/phi3.py"
_OLMOE = "vllm/model_executor/models/olmoe.py"
_Q3MOE = "vllm/model_executor/models/qwen3_moe.py"
_DSV2 = "vllm/model_executor/models/deepseek_v2.py"
_DSA = "vllm/models/deepseek_v32/nvidia/model.py"
_DSA_ATTN = "vllm/models/deepseek_v32/attention.py"
_FMOE = "vllm/model_executor/layers/fused_moe/layer.py"
_LN = "vllm/model_executor/layers/layernorm.py"


def _rules_llama_like(theta: float) -> dict[str, Any]:
    return {"qkv_bias_key": "attention_bias", "default_theta": theta}


FAMILY_FACTS: dict[str, dict[str, Any]] = {
    # ----------------------------------------------------------------------------------------------------------------------------- dense
    "llama": {
        "arch": "dense", "vllm_class": "LlamaForCausalLM",
        "rules": _rules_llama_like(10_000.0),
        "rules_provenance": f"{_LLAMA}:267-274 LlamaAttention: attention_bias = getattr(config, 'attention_bias', False) (or config.bias / qkv_bias); "
                            f"rope theta from config (transformers default 1e4)",
        "facts": {},
    },
    "mistral": {
        "arch": "dense", "vllm_class": "MistralForCausalLM (= LlamaForCausalLM)",
        "rules": _rules_llama_like(10_000.0),
        "rules_provenance": f"{_LLAMA}:267-274 (MistralForCausalLM is registered onto LlamaForCausalLM)",
        "facts": {},
    },
    "qwen2": {
        "arch": "dense", "vllm_class": "Qwen2ForCausalLM",
        "rules": {"qkv_bias": True, "default_theta": 1_000_000.0},
        "rules_provenance": f"{_QWEN2}:156-163 Qwen2Attention: QKVParallelLinear(..., bias=True) unconditionally; :247 set_default_rope_theta(config, 1000000)",
        "facts": {},
        # Qwen2.5 config.json files carry `sliding_window` next to `use_sliding_window: false`: transformers' Qwen2Config sets
        # `self.sliding_window = sliding_window if use_sliding_window else None`, so vLLM's Qwen2Attention (no per-layer window unless
        # `layer_types` says so, {_QWEN2}:349-359) never sees it.  The window is a fact of the configuration only when the gate is true.
        "sliding_window_gate": "use_sliding_window",
        "sliding_window_gate_provenance": f"transformers Qwen2Config.__init__ (sliding_window = sliding_window if use_sliding_window else None); {_QWEN2}:349-359",
    },
    "qwen3": {
        "arch": "dense", "vllm_class": "Qwen3ForCausalLM",
        "rules": {"qkv_bias_key": "attention_bias", "default_theta": 1_000_000.0, "qk_norm": "head"},
        "rules_provenance": f"{_QWEN3}:150-151 Qwen3Attention: q_norm = RMSNorm(head_dim), k_norm = RMSNorm(head_dim) applied per head before RoPE; "
                            f":204 qkv_bias = getattr(config, 'attention_bias', False)",
        "facts": {},
    },
    "phi3": {
        "arch": "dense", "vllm_class": "Phi3ForCausalLM(LlamaForCausalLM)",
        "rules": _rules_llama_like(10_000.0),
        "rules_provenance": f"{_PHI3}:10-18 Phi3ForCausalLM inherits LlamaForCausalLM; packed_modules_mapping keeps the checkpoint's fused qkv_proj / gate_up_proj",
        # the model-level `sliding_window` becomes FA2's window_size_left = sliding_window - 1; at max_seqlen_k <= that it is the causal kernel
        # (gen_dense2_patterns.AttentionScopeInertWindow decides from the LAUNCH, never from this key) -- no C field, no hand value
        "facts": {},
        "patterns": ["dense2"],
    },
    "gemma2": {
        "arch": "dense", "vllm_class": "Gemma2ForCausalLM",
        "rules": _rules_llama_like(10_000.0),
        "rules_provenance": f"{_GEMMA2}:136,143 Gemma2Attention: bias=config.attention_bias",
        "facts": {
            "tie_word_embeddings": {"key": "tie_word_embeddings", "default": True,
                                    "provenance": f"{_GEMMA2}:364-368 Gemma2ForCausalLM.compute_logits: logits_processor(self.model.embed_tokens, ...): no lm_head parameter; "
                                                  f"transformers Gemma2Config default tie_word_embeddings=True (the key is absent from the checkpoint's config.json)"},
            "attn_logit_softcapping": {"key": "attn_logit_softcapping",
                                       "provenance": f"{_GEMMA2}:202 attn_logits_soft_cap=config.attn_logit_softcapping -> :165 Attention(logits_soft_cap=...) (FA2 softcap)"},
            "final_logit_softcapping": {"key": "final_logit_softcapping",
                                        "provenance": f"{_GEMMA2}:342-343 LogitsProcessor(config.vocab_size, soft_cap=config.final_logit_softcapping)"},
            "query_pre_attn_scalar": {"key": "query_pre_attn_scalar",
                                      "provenance": f"{_GEMMA2}:129 self.scaling = config.query_pre_attn_scalar**-0.5"},
            "sliding_window": {"key": "sliding_window",
                               "provenance": f"{_GEMMA2}:155-156 is_sliding = config.layer_types[layer_idx] == 'sliding_attention'; sliding_window = config.sliding_window "
                                             f"if is_sliding else None (transformers Gemma2Config: layer_types defaults to sliding on even layer_idx, global on odd)"},
            "hidden_act": {"key": ("hidden_activation", "hidden_act"),
                           "provenance": f"{_GEMMA2}:81-86 Gemma2MLP asserts hidden_activation == 'gelu_pytorch_tanh' and uses GeluAndMul(approximate='tanh'); :209"},
            "embed_normalizer": {"derive": lambda cfg, C: float(C["H"]) ** 0.5,
                                 "provenance": f"{_GEMMA2}:276-277 normalizer = hidden_size**0.5 registered as a bf16 buffer; :283 embed_tokens(ids) * normalizer"},
            "rmsnorm_plus_one": {"const": True,
                                 "provenance": f"{_LN}:132-136 GemmaRMSNorm: x * (1 + w) instead of x * w; {_GEMMA2}:213-220,270 every norm is GemmaRMSNorm"},
            "layer_norms": {"const": 4,
                            "provenance": f"{_GEMMA2}:213-220 input_layernorm, post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm"},
            "extra_layer_norms": {"const": ("ln_pre_mlp", "ln_post_mlp"),
                                  "provenance": f"{_GEMMA2}:241-245 pre_feedforward_layernorm before the MLP, post_feedforward_layernorm after it (the two [H] fields "
                                                f"analytic.weights_type appends per layer)"},
        },
        "params": {
            "layer": {"pre_feedforward_layernorm.weight": "ln_pre_mlp", "post_feedforward_layernorm.weight": "ln_post_mlp"},
            "top": {"model.normalizer": ("normalizer",)},
            "provenance": f"{_GEMMA2}:217-220 (the two extra GemmaRMSNorm parameters), :277 (the 0-d `normalizer` buffer; analytic.check_normalizer validates it)",
        },
        "patterns": ["dense2", "unfused-norm-chains", "softcap"],
    },
    # -------------------------------------------------------------------------------------------------------------------------- dense-ln
    "gpt_neox": {
        "arch": "dense-ln", "vllm_class": "GPTNeoXForCausalLM",
        "rules": {"qkv_bias": True, "default_theta": 10_000.0},
        "rules_provenance": f"{_NEOX}:70 self.bias = getattr(config, 'attention_bias', True) -> :80,:87 query_key_value / dense bias; the MLP linears carry biases (:128-139)",
        "facts": {
            "ROPE_THETA": {"key": "rotary_emb_base",
                           "provenance": f"{_NEOX}:92-96 get_rope(head_size, rope_parameters=config.rope_parameters) -- transformers GPTNeoXConfig maps rotary_emb_base "
                                         f"to rope_parameters.rope_theta"},
            "ROT": {"derive": lambda cfg, C: int(C["D"] * cfg["rotary_pct"]), "check_key": "rotary_pct",
                    "provenance": f"{_NEOX}:92-96 get_rope over rope_parameters (partial_rotary_factor = rotary_pct): rotary_dim = int(head_size * rotary_pct)"},
            "NORM": {"const": "layernorm", "provenance": f"{_NEOX}:159-164 nn.LayerNorm(hidden_size, eps=layer_norm_eps) with weight and bias; :224 final_layer_norm"},
            "hidden_act": {"key": "hidden_act", "provenance": f"{_NEOX}:140 self.act = get_act_fn(config.hidden_act) ('gelu' = nn.GELU(), erf)"},
            "use_parallel_residual": {"key": "use_parallel_residual",
                                      "provenance": f"{_NEOX}:158,181-186 x + attn(ln1(x)) + mlp(ln2(x)) when use_parallel_residual"},
            "layer_norms": {"const": 2, "provenance": f"{_NEOX}:159-164"},
            "dense_bias": {"const": True, "provenance": f"{_NEOX}:80,87 (bias=self.bias, default True), :128-139 ColumnParallelLinear / RowParallelLinear default bias=True"},
        },
        "params": {
            "layer_prefix": "gpt_neox.layers",
            "layer": {"input_layernorm.weight": "ln1", "input_layernorm.bias": "ln1_b",
                      "attention.query_key_value.weight": "qkv_w", "attention.query_key_value.bias": "qkv_b",
                      "attention.dense.weight": "o_w", "attention.dense.bias": "o_b",
                      "post_attention_layernorm.weight": "ln2", "post_attention_layernorm.bias": "ln2_b",
                      "mlp.dense_h_to_4h.weight": "up_w", "mlp.dense_h_to_4h.bias": "up_b",
                      "mlp.dense_4h_to_h.weight": "down_w", "mlp.dense_4h_to_h.bias": "down_b"},
            "cos_sin": "attention.rotary_emb.cos_sin_cache",
            "attn_scales": "attention.attn",
            "top": {"gpt_neox.embed_in.weight": ("embed",), "gpt_neox.final_layer_norm.weight": ("norm",), "gpt_neox.final_layer_norm.bias": ("norm_b",),
                    "embed_out.weight": ("lm_head",)},
            "provenance": f"{_NEOX}:75-89 (query_key_value, dense), :128-139 (dense_h_to_4h, dense_4h_to_h), :159-164, :215-224 (embed_in, final_layer_norm), :287-291 (embed_out)",
        },
        "header_keys": {"layer_norm_eps": "EPS"},
        "patterns": ["ln"],
    },
    # ------------------------------------------------------------------------------------------------------------------------------- moe
    "olmoe": {
        "arch": "moe", "vllm_class": "OlmoeForCausalLM",
        "rules": {"qkv_bias_key": "attention_bias", "default_theta": 10_000.0, "default_eps": 1e-5},
        "rules_provenance": f"{_OLMOE}:158,169 bias=False on qkv_proj / o_proj (attention_bias False in the checkpoint); :239-240,292 RMSNorm(hidden_size, eps=1e-5) "
                            f"hard-coded (config.json carries no epsilon)",
        "facts": {
            "QK_NORM": {"const": True, "provenance": f"{_OLMOE}:164-165 q_norm = RMSNorm(total_num_heads * head_dim, eps=1e-5), k_norm likewise over ALL heads, "
                                                     f"applied before RoPE (:190-211)"},
            "QK_NORM_EPS": {"const": 1e-5, "provenance": f"{_OLMOE}:164-165"},
            "RENORMALIZE": {"const": False, "provenance": f"{_OLMOE}:100 FusedMoE(..., renormalize=False): the top-k softmax weights are used as they are"},
            "CLIP_QKV": {"key": "clip_qkv", "default": None,
                         "provenance": f"{_OLMOE} has no clip_qkv handling at d9105ea80 (the checkpoint's config.json says null; a non-null value would be "
                                       f"a fact vLLM ignores -- check_header flags it)"},
        },
        "params": {
            "layer": {"self_attn.q_norm.weight": "q_norm", "self_attn.k_norm.weight": "k_norm", "mlp.gate.weight": "gate_w",
                      "mlp.experts.routed_experts.w13_weight": "w13", "mlp.experts.routed_experts.w2_weight": "w2"},
            "provenance": f"{_OLMOE}:164-165 (q_norm, k_norm), :87-92 (gate: ReplicatedLinear(hidden, num_experts, bias=False)), :94-103 FusedMoE(...) -> routed_experts "
                          f"(w13_weight [E, 2I, H], w2_weight [E, H, I])",
        },
    },
    "qwen3_moe": {
        "arch": "moe", "vllm_class": "Qwen3MoeForCausalLM",
        "rules": {"qkv_bias_key": "attention_bias", "default_theta": 1_000_000.0, "qk_norm": "head"},
        "rules_provenance": f"{_Q3MOE}:319-320 Qwen3MoeAttention: q_norm = RMSNorm(head_dim, eps), k_norm likewise, per head before RoPE; :368 qkv_bias = "
                            f"getattr(config, 'attention_bias', False)",
        "facts": {
            "RENORMALIZE": {"key": "norm_topk_prob", "provenance": f"{_Q3MOE}:206 FusedMoE(..., renormalize=config.norm_topk_prob)"},
            "SHARED_EXPERTS": {"key": "shared_expert_intermediate_size", "default": 0,
                               "provenance": f"{_Q3MOE}:175-197 shared_expert only when shared_expert_intermediate_size > 0 (else shared_experts_input=None)"},
        },
        "params": {
            "layer": {"mlp.gate.weight": "gate_w", "mlp.experts.routed_experts.w13_weight": "w13", "mlp.experts.routed_experts.w2_weight": "w2"},
            "provenance": f"{_Q3MOE}:167-172 (gate: ReplicatedLinear, bias=False, full E on every rank), :198-210 FusedMoE(...) -> routed_experts (w13 [E, 2I_r, H], w2 [E, H, I_r])",
        },
        "moe_width_key": "moe_intermediate_size",
        "moe_width_provenance": f"{_Q3MOE}:186-190,:213-219 the experts' width is moe_intermediate_size; intermediate_size is the dense-MLP width of mlp_only_layers "
                                f"(none here: mlp_only_layers empty, decoder_sparse_step 1)",
    },
    # --------------------------------------------------------------------------------------------------------------------------- dsa-moe
    # Sparse attention (multi-head latent attention whose decode attends the top-k positions a lightning indexer picks) and a sigmoid
    # grouped-top-k MoE with a shared expert.  `structural_refusal` refuses the class: the κ fold has patterns for the MoE block
    # (`gen_shared_moe_patterns`) but none for the indexer or sparse MLA, so a profile would be silently mis-fitted.  The facts below are
    # what the Build's lowering, the MoE scope and the capture map read.
    "glm_moe_dsa": {
        "arch": "dsa-moe", "vllm_class": "GlmMoeDsaForCausalLM (= DeepseekV32ForCausalLM)",
        "rules": {"qkv_bias_key": "attention_bias"},
        "rules_provenance": f"vllm/models/deepseek_v32/__init__.py:18-21 GlmMoeDsaForCausalLM is the CUDA DeepseekV32ForCausalLM; {_DSA_ATTN}:255-261 "
                            f"o_proj and the MLA projections without bias; MoE blocks are {_DSV2}:287-421 DeepseekV2MoE",
        "facts": {
            "Q_LORA": {"key": "q_lora_rank", "provenance": f"{_DSA_ATTN}:137 q_lora_rank = config.q_lora_rank (fused_qkv_a_proj, q_a_layernorm, q_b_proj)"},
            "KV_LORA": {"key": "kv_lora_rank", "provenance": f"{_DSA_ATTN}:138,177-180 kv_b_proj = ColumnParallelLinear(kv_lora_rank, heads * (qk_nope + v_head))"},
            "QK_NOPE": {"key": "qk_nope_head_dim", "provenance": f"{_DSA_ATTN}:134,144 qk_head_dim = qk_nope_head_dim + qk_rope_head_dim"},
            "QK_ROPE": {"key": "qk_rope_head_dim", "provenance": f"{_DSA_ATTN}:135,263-268 rotary_emb = get_rope(qk_rope_head_dim, ...)"},
            "V_HEAD": {"key": "v_head_dim", "provenance": f"{_DSA_ATTN}:136,255-256 o_proj = RowParallelLinear(heads * v_head_dim, hidden)"},
            "ROPE_NEOX": {"const": False, "provenance": f"{_DSA_ATTN}:263-268 get_rope(..., is_neox_style=False): interleaved (even, odd) rotary pairs"},
            "ATTN_SCALE": {"derive": lambda cfg, C: float(int(cfg["qk_nope_head_dim"]) + int(cfg["qk_rope_head_dim"])) ** -0.5,
                           "provenance": f"{_DSA_ATTN}:144-145 scaling = qk_head_dim**-0.5 (:148-159 times the yarn mscale only when rope_type != default)"},
            "INDEX_HEADS": {"key": "index_n_heads", "provenance": f"{_DSA_ATTN}:63 DeepseekV32Indexer.n_head = config.index_n_heads"},
            "INDEX_HEAD_DIM": {"key": "index_head_dim", "provenance": f"{_DSA_ATTN}:64 DeepseekV32Indexer.head_dim = config.index_head_dim"},
            "INDEX_TOPK": {"key": "index_topk", "provenance": f"{_DSA_ATTN}:62 DeepseekV32Indexer.topk_tokens = config.index_topk; {_DSA}:182-188 the shared "
                                                              f"top-k indices buffer"},
            "INDEX_TOPK_FREQ": {"key": "index_topk_freq", "default": 1,
                                "provenance": f"{_DSA_ATTN}:161-168 skip_topk = max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0 "
                                              f"(a layer that skips builds no indexer and reuses the shared top-k)"},
            "INDEX_SKIP_TOPK_OFFSET": {"key": "index_skip_topk_offset", "default": 2, "provenance": f"{_DSA_ATTN}:164-168"},
            "INDEX_ROPE_INTERLEAVE": {"key": "indexer_rope_interleave", "default": False,
                                      "provenance": f"{_DSA_ATTN}:236,270-276 indexer_rope_emb = get_rope(..., is_neox_style=not indexer_rope_interleave)"},
            "FIRST_DENSE": {"key": "first_k_dense_replace",
                            "provenance": f"{_DSA}:85-108 layers from first_k_dense_replace on are DeepseekV2MoE, the ones below DeepseekV2MLP(intermediate_size)"},
            "NUM_EXPERTS": {"key": "n_routed_experts", "provenance": f"{_DSV2}:317-321,373 GateLinear(hidden, n_routed_experts), FusedMoE num_experts"},
            "N_GROUP": {"key": "n_group", "provenance": f"{_DSV2}:379-380 use_grouped_topk=True, num_expert_group=getattr(config, 'n_group', 1)"},
            "TOPK_GROUP": {"key": "topk_group", "provenance": f"{_DSV2}:381 topk_group=getattr(config, 'topk_group', 1)"},
            "RENORMALIZE": {"key": "norm_topk_prob", "provenance": f"{_DSV2}:377 renormalize=config.norm_topk_prob"},
            "SCORING": {"key": "scoring_func", "provenance": f"{_DSV2}:383 scoring_func=getattr(config, 'scoring_func', 'softmax') (sigmoid in the checkpoint)"},
            "SCORE_BIAS": {"key": "topk_method", "provenance": f"{_DSV2}:323-328 e_score_correction_bias = an f32 [n_routed_experts] parameter when "
                                                               f"topk_method == 'noaux_tc', else None; :386 passed to the router"},
            "ROUTED_SCALING_FACTOR": {"key": "routed_scaling_factor",
                                      "provenance": f"{_DSA}:98 DeepseekV2MoE(..., apply_routed_scale_to_output=False) -> {_FMOE}:295-302 the router "
                                                    f"multiplies the top-k weights by routed_scaling_factor (grouped_topk's routed_scaling_factor)"},
            "ROUTER_DTYPE": {"const": "float32", "provenance": f"{_DSV2}:131-141 _get_moe_router_dtype returns float32 for model_type glm_moe_dsa; "
                                                               f":317-321 GateLinear(out_dtype=router_dtype)"},
            "GATE_IN_EXPERTS": {"const": True, "provenance": f"{_DSV2}:370-372,419-421 FusedMoEFactory(shared_experts=..., gate=self.gate) called with "
                                                              f"router_logits=hidden_states: the gate and the shared expert run inside moe_forward_shared"},
            "SHARED_EXPERTS": {"key": "n_shared_experts", "provenance": f"{_DSV2}:355-368 shared_experts = DeepseekV2MLP(moe_intermediate_size * "
                                                                        f"n_shared_experts, reduce_results=False) unless the fused shared expert is enabled"},
            "SHARED_I": {"derive": lambda cfg, C: int(cfg["moe_intermediate_size"]) * int(cfg["n_shared_experts"]),
                         "provenance": f"{_DSV2}:358 intermediate_size = moe_intermediate_size * n_shared_experts (a rank's: the rank's moe_intermediate_size)"},
        },
        "moe_width_key": "moe_intermediate_size",
        "moe_width_provenance": f"{_DSV2}:376 FusedMoE(intermediate_size=config.moe_intermediate_size); intermediate_size is the dense layers' width "
                                f"({_DSA}:101-108)",
    },
}

#: `model_type`s whose vLLM d9105ea80 implementation this table describes; anything else refuses (no silently mis-fitted profile).
FAMILIES: tuple[str, ...] = tuple(FAMILY_FACTS)


def family_of(cfg: dict[str, Any]) -> str:
    return str(cfg.get("model_type", "")).lower()


def facts_for(cfg: dict[str, Any]) -> dict[str, Any]:
    fam = family_of(cfg)
    if fam not in FAMILY_FACTS:
        raise ValueError(f"model_type {fam!r} is not in the family facts table {FAMILIES} (needs its own mechanism record)")
    return FAMILY_FACTS[fam]


def resolve_facts(cfg: dict[str, Any], C: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """The family's hand fields for a configuration: `(values, provenance)`; `key` facts absent from `cfg` and without a default are skipped."""
    vals: dict[str, Any] = {}
    prov: dict[str, str] = {}
    for name, f in facts_for(cfg)["facts"].items():
        if "const" in f:
            vals[name], prov[name] = f["const"], "hand"
        elif "key" in f:
            keys = f["key"] if isinstance(f["key"], tuple) else (f["key"],)
            hit = next((k for k in keys if k in cfg and cfg[k] is not None), None)
            if hit is not None:
                vals[name], prov[name] = cfg[hit], "hf-config"
            elif "default" in f:
                vals[name], prov[name] = f["default"], "hand"
        elif "derive" in f:
            vals[name], prov[name] = f["derive"](cfg, C), "derived"
        else:
            raise ValueError(f"family fact {name!r} has no const / key / derive")
    return vals, prov


def header_checks(cfg_family: dict[str, Any], C: dict[str, Any], header_cfg: dict[str, Any]) -> list[str]:
    """The family's `key` facts compared to the run's recorded HF config (the base check covers the Llama-vocabulary keys)."""
    out: list[str] = []
    fam = facts_for(cfg_family)
    for name, f in fam["facts"].items():
        if "key" in f:
            keys = f["key"] if isinstance(f["key"], tuple) else (f["key"],)
            for hf in keys:
                if hf in header_cfg and name in C and header_cfg[hf] != C[name]:
                    out.append(f"{hf}={header_cfg[hf]!r} != C[{name}]={C[name]!r}")
                    break
        elif "derive" in f and f.get("check_key") in header_cfg:
            new = f["derive"](header_cfg, C)
            if new != C.get(name):
                out.append(f"{f['check_key']}={header_cfg[f['check_key']]!r}: {name} {new!r} != C[{name}]={C.get(name)!r}")
    for hf, key in (fam.get("header_keys") or {}).items():
        if hf in header_cfg and header_cfg[hf] != C.get(key):
            out.append(f"{hf}={header_cfg[hf]!r} != C[{key}]={C.get(key)!r}")
    return out


def table() -> list[dict[str, Any]]:
    """The facts as rows (family, field, kind, value/key, provenance) -- the report's table."""
    rows: list[dict[str, Any]] = []
    for fam, F in FAMILY_FACTS.items():
        rows.append({"family": fam, "field": "<rules>", "kind": "rules", "value": F["rules"], "provenance": F["rules_provenance"]})
        for name, f in F["facts"].items():
            kind = "const" if "const" in f else "key" if "key" in f else "derive"
            val = f.get("const", f.get("key", "f(config)"))
            if "default" in f:
                val = f"{val} (default {f['default']!r})"
            rows.append({"family": fam, "field": name, "kind": kind, "value": val, "provenance": f["provenance"]})
        if F.get("params"):
            rows.append({"family": fam, "field": "<params>", "kind": "names", "value": {k: v for k, v in F["params"].items() if k != "provenance"},
                         "provenance": F["params"]["provenance"]})
    return rows


__all__ = ["FAMILY_FACTS", "FAMILIES", "family_of", "facts_for", "resolve_facts", "header_checks", "table", "math"]
