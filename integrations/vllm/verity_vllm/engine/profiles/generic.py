"""[P6, 2026-09-23] The configuration-derived κ profile of ANY checkpoint role: `build_profile(role, world=1)`.

One constructor per ARCHITECTURE class, selected by the family facts table (`family_facts.FAMILY_FACTS[model_type]["arch"]`):
  * `dense`     -- the Llama-architecture vocabulary (`LlamaForCausalLM`, Qwen2, Mistral, Qwen3, Phi-3, Gemma-2): the base κ module's
                   patterns / weight map / header check (`vllm_d9105ea80_sm89_eager.build_profile`), the tuned-GEMM target specialization
                   (`gen_llama_patterns.GemmLaunchTuned`, a TARGET fact: it precedes the base `GemmLaunch` for every role and owns only launches
                   whose (N, K) is a key of the record target's table), and -- when the configuration says so -- the inert-window FA2 scope
                   (`sliding_window` in config.json), the Gemma-2 unfused-norm / GeGLU / embed-scale / softcap chains (family facts);
  * `dense-ln`  -- the GPT-NeoX architecture (`nn.LayerNorm` with bias, biases on every projection, parallel residual, partial rotary):
                   the family's parameter names (`FAMILY_FACTS["gpt_neox"]["params"]`) and `gen_ln_patterns`;
  * `moe`       -- the fused-MoE decoder (`OlmoeForCausalLM`, `Qwen3MoeForCausalLM`): the constructive MoE block (`gen_ov_moe_patterns.
                   MoEFusedExperts`, router renormalisation from `C["RENORMALIZE"]`, the router kernel's VPT from E), the experts' width from
                   `moe_intermediate_size` when the family has one, the MoE / q-k-norm weight fields.
  * `dsa-moe`   -- GLM-5.3's sparse-attention MLA and sigmoid shared-expert MoE: the DSA folds in
                   `observe.fold.patterns.dsa`, and `gen_shared_moe_patterns.MoEFusedSharedExpertsFp8`.
`world > 1` is a generic branch of every class (the rank-local shard configuration: heads, kv heads, the MLP / expert width and the vocabulary
divided by the world; `tp.embedding_shard.EmbeddingShardCuda` and `tp.collective_pattern.TpCollective(world)` registered first; the sampler over the
all-gathered vocabulary; the header check re-shards the recorded FULL config) -- it replaced the two hand-written `*_tp2rank` modules and the
TP2 half of `gen_ov_moe_qwen3` after the claim in their docstrings ("the rank profile is the base module's build_profile over the world-divided
configuration") was checked object by object (`tests/test_profiles_generic.py`, fixtures `profiles/expected/`).

Module names (the `--profile NAME` every consumer imports as `verity_vllm.engine.profiles.NAME`, materialised by the finder in `profiles/__init__.py`):
  `derived_<ROLE>`          -- the TP1 profile of a checkpoint role (any architecture),
  `derived_<ROLE>_tp<W>`    -- ONE rank of a W-way tensor-parallel engine,
  `derived_<ROLE>[..]_ovs`  -- the experimental sampled-row variant (`TokenSampleGumbel` first; lanes cov-difr / ov-sampling),
  `dense_<ROLE>`            -- [X-07] the pre-P6 name, still accepted for dense-architecture roles (recorded runs cite it).
The role's HF `config.json` is `dense_generic.find_hf_config(role)`: the byte copy `profiles/hf_configs/<ROLE>.config.json` (sha256 checked against
the manifest entry), else the manifest's local snapshot / the hub cache.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import types
from typing import Any, Callable

from verity_vllm.program import model as analytic
from verity_vllm.observe.events import RunHeader
from verity_vllm.observe.fold.patterns import TokenSelectArgmax, TokenSelectGumbelTopP, TokenSelectGumbelTwoStage
from verity_vllm.observe.fold.resolver import Profile
from verity_vllm.observe.fold.patterns.collectives import TpCollective
from verity_vllm.collectives.embedding_shard import EmbeddingShardCuda
from verity_vllm.engine.profiles import dense_generic as _dg
from verity_vllm.engine.profiles import family_facts as FF
from verity_vllm.engine.profiles import vllm_d9105ea80_sm89_eager as base
from verity_vllm.engine.profiles.gen_dense2_patterns import with_dense2_patterns
from verity_vllm.engine.profiles.gen_dense_gemma2_patterns import with_gemma2_patterns
from verity_vllm.engine.profiles.gen_dense_softcap_patterns import with_softcap_patterns
from verity_vllm.observe.fold.patterns.dsa import dsa_patterns
from verity_vllm.engine.profiles.gen_llama_patterns import ADA_TUNED_GEMM, with_tuned_gemm
from verity_vllm.engine.profiles.gen_ln_patterns import with_ln_patterns
from verity_vllm.engine.profiles.gen_ov_moe_patterns import (FUSED_MOE_NUM_STAGES, FUSED_MOE_NUM_WARPS, FUSED_MOE_PINS, MOE_CONSTRUCTION_PATTERNS,
                                  MOE_FP8_CONSTRUCTION_PATTERNS, ROUTER_VPT_BY_E, MoEFusedExperts, MoEFusedExpertsFp8)
from verity_vllm.engine.profiles.gen_shared_moe_patterns import MoEFusedSharedExpertsFp8
from verity_vllm.program.frontend.target_profile import TargetProfile

MODULE_PREFIX = "derived_"
OV_SAMPLING_SUFFIX = "_ovs"
_MODULE_RE = re.compile(r"^derived_([A-Z0-9_]+?)(?:_tp(\d+))?(_ovs)?$")
_LAYER_RE_FMT = r"^{prefix}\.(\d+)\.(.+)$"
_ATTN_SCALES = r"(k_scale|v_scale|q_scale|prob_scale|_k_scale|_v_scale|_q_scale|_prob_scale)"
_SCALE_INV = "_scale_inv"
_MOE_ARCHES = frozenset({"moe", "dsa-moe"})


# ------------------------------------------------------------------------------------------------------------------------------- names

def module_name(role: str, world: int = 1, *, ov_sampling: bool = False) -> str:
    return f"{MODULE_PREFIX}{role}" + (f"_tp{world}" if world > 1 else "") + (OV_SAMPLING_SUFFIX if ov_sampling else "")


def parse_module(name: str) -> tuple[str, int] | None:
    """`derived_<ROLE>[_tp<W>][_ovs]` -> (ROLE, W); `dense_<ROLE>` -> (ROLE, 1) (the X-07 name); else None."""
    short = name.rsplit(".", 1)[-1]
    m = _MODULE_RE.match(short)
    if m:
        return m.group(1), int(m.group(2) or 1)
    legacy = _dg.role_of_module(short)
    return (legacy, 1) if legacy else None


def is_ov_sampling(name: str) -> bool:
    """`derived_<ROLE>[_tp<W>]_ovs`: the experimental sampled-row variant (`gen_ov_sampling_patterns.TokenSampleGumbel` registered first;
    lanes cov-difr / ov-sampling; NOT an accepted profile) -- what the per-role `gen_ov_sampling_b0` / `_b1` modules were before P6."""
    m = _MODULE_RE.match(name.rsplit(".", 1)[-1])
    return bool(m and m.group(3))


# ------------------------------------------------------------------------------------------------------------------------- configuration

def load_hf_config(role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(config.json document, source record)` of a role, or ImportError with the three places looked at."""
    hit = _dg.find_hf_config(role)
    if hit is None:
        raise ImportError(f"no HF config.json for role {role!r} (profiles/hf_configs/{role}.config.json, the manifest entry's local_path, "
                          f"or the $HF_HOME snapshot) -- ship the byte copy with the checkpoints.json entry")
    path, source, entry = hit
    cfg = json.loads(path.read_text())
    return cfg, {"path": str(path), "source": source, "sha256": _dg._sha256(path), "repo": (entry or {}).get("repo"),
                 "revision": (entry or {}).get("revision")}


def moe_width_key(cfg: dict[str, Any]) -> str | None:
    fam = FF.facts_for(cfg)
    k = fam.get("moe_width_key")
    return k if k and cfg.get(k) is not None else None


def sharded_keys(cfg: dict[str, Any]) -> tuple[str, ...]:
    """The HF keys vLLM's parallel layers divide by the world: heads, kv heads, the MLP (or expert) width, the vocabulary."""
    return ("num_attention_heads", "num_key_value_heads", moe_width_key(cfg) or "intermediate_size", "vocab_size")


def rank_hf_config(cfg: dict[str, Any], world: int) -> dict[str, Any]:
    """The rank-local HF configuration (`world` = 1: the full one, with the experts' width as `intermediate_size` for MoE families)."""
    out = dict(cfg)
    if "num_key_value_heads" not in out or out["num_key_value_heads"] is None:
        out["num_key_value_heads"] = out["num_attention_heads"]
    if world > 1:
        if not out.get("head_dim"):
            out["head_dim"] = int(out["hidden_size"]) // int(out["num_attention_heads"])   # the FULL model's head width: a rank keeps it
        for k in sharded_keys(cfg):
            if k == "num_key_value_heads" and out[k] < world and world % out[k] == 0:
                out[k] = 1          # QKVParallelLinear: each key/value head on world // heads ranks (num_kv_head_replicas)
                continue
            if out[k] % world:
                raise ValueError(f"{k}={out[k]} is not divisible by the world {world}")
            out[k] = out[k] // world
    mk = moe_width_key(cfg)
    if mk:
        out["intermediate_size"] = out[mk]
    return out


def effective_sliding_window(cfg: dict[str, Any]) -> int | None:
    """The `sliding_window` vLLM's model code sees: the config key, unless the family gates it (Qwen2's `use_sliding_window: false`)."""
    gate = FF.FAMILY_FACTS.get(FF.family_of(cfg), {}).get("sliding_window_gate")
    if gate is not None and not cfg.get(gate):
        return None
    return cfg.get("sliding_window") or None


def _quant_config_for_profile(cfg: dict[str, Any], arch: str) -> dict[str, Any] | None:
    """The checkpoint quantization contract that generic ``config_of`` validates.

    GLM-5.3 deliberately keeps its DSA projections, norms, biases, embeddings,
    lm_head, and MoE gates in their served non-FP8 dtypes.  Their profile and
    Build rules model those roles separately; ``quant_refusal`` still checks
    the block-FP8 scheme and that every MoE router is kept in bf16.
    """
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict) or arch != "dsa-moe":
        return qc
    routers = [f"model.layers.{n}.mlp.gate" for n in range(int(cfg.get("num_hidden_layers") or 0))]
    return {**qc, "modules_to_not_convert": routers}


def structural_refusal(cfg: dict[str, Any]) -> str | None:
    """None when the family facts table describes the configuration's vLLM implementation; else why it is not served."""
    fam = FF.family_of(cfg)
    if fam not in FF.FAMILY_FACTS:
        return f"model_type {fam!r} is not in the family facts table {FF.FAMILIES} (needs its own mechanism record)"
    arch = FF.FAMILY_FACTS[fam]["arch"]
    if arch in ("mla-moe", "dsa-moe"):
        if int(cfg.get("n_group", 1) or 1) != 1 or int(cfg.get("topk_group", 1) or 1) != 1:
            return f"{fam}: n_group={cfg.get('n_group')!r} topk_group={cfg.get('topk_group')!r}: the sigmoid router is stated over one expert group"
        if (cfg.get("rope_scaling") or {}).get("rope_type", "default") not in ("default", None):
            return f"{fam}: rope_scaling={cfg.get('rope_scaling')!r}: the MLA profile states the default (unscaled) rotary table"
    if effective_sliding_window(cfg) and "dense2" not in (FF.FAMILY_FACTS[fam].get("patterns") or ()):
        # only families whose vLLM code hands the window to FA2 as an inert `window_size_left >= max_seqlen_k` (Phi-3, Gemma-2: the
        # inert-window scope decides from the LAUNCH) are served; any other windowed configuration is outside the pinned splitkv patterns
        return f"sliding_window={cfg['sliding_window']!r}: windowed attention is outside the pinned FA2 splitkv patterns"
    if FF.FAMILY_FACTS[fam]["rules"].get("qk_norm") == "head" and not cfg.get("head_dim"):
        return f"{fam} without head_dim in config.json: the per-head q/k norm width is not derivable"
    if arch in _MOE_ARCHES:
        if cfg.get("mlp_only_layers"):
            return f"mlp_only_layers={cfg['mlp_only_layers']!r}: the MoE profile states every layer as a fused-MoE layer"
        if int(cfg.get("decoder_sparse_step", 1) or 1) != 1:
            return f"decoder_sparse_step={cfg['decoder_sparse_step']!r}: the MoE profile states every layer as a fused-MoE layer"
        E = cfg.get("num_experts") or cfg.get("num_local_experts") or cfg.get("n_routed_experts")
        if not E:
            return "MoE family without num_experts, num_local_experts, or n_routed_experts in config.json"
    why = analytic.quant_refusal(_quant_config_for_profile(cfg, arch), cfg)
    if why:
        return why
    return None


def router_vpt_bf16(E: int) -> int:
    """`topkGatingSoftmax<T=bf16, EXPERTS=E>`'s values per thread (vLLM csrc/moe/topk_softmax_kernels.cu): 16-byte loads = `ELTS_PER_LDG =
    min(16, sizeof(T) * E) / sizeof(T)` bf16 words, `VECs_PER_THREAD = max(1, E / (ELTS_PER_LDG * 32))`, `VPT = ELTS_PER_LDG * VECs_PER_THREAD`.
    8 at E = 64 (OLMoE, bit-exact on the recorded cases) and at E = 128 (Qwen3-30B-A3B, the Build's binding); `ROUTER_VPT_BY_E` where it has
    the entry -- the two must agree (asserted), the rule covers the tiny test configurations."""
    if E <= 0 or E & (E - 1):
        raise ValueError(f"E={E}: the router kernel is instantiated for powers of two only")
    elts = min(8, E)
    vpt = elts * max(1, E // (elts * 32))
    if E in ROUTER_VPT_BY_E and ROUTER_VPT_BY_E[E] != vpt:
        raise ValueError(f"E={E}: router_vpt_bf16 gives {vpt}, ROUTER_VPT_BY_E pins {ROUTER_VPT_BY_E[E]}")
    return vpt


def derive_config(cfg: dict[str, Any], *, repo: str | None, revision: str | None, world: int = 1) -> tuple[dict[str, Any], dict[str, str]]:
    """`analytic.config_of` over the (rank-local) HF configuration with the family's rules and hand facts, plus FA2 geometry; fail-closed."""
    why = structural_refusal(cfg)
    if why:
        raise ValueError(why)
    fam = FF.facts_for(cfg)
    rcfg = rank_hf_config(cfg, world)
    analytic_cfg = dict(rcfg)
    analytic_cfg["quantization_config"] = _quant_config_for_profile(rcfg, fam["arch"])
    # the hand facts that config_of needs BEFORE deriving (none today) vs the ones that read C (derive: sqrt(H), rotary width): two passes
    C0, _ = analytic.config_of(analytic_cfg, model=repo, revision=revision, family_rules={FF.family_of(cfg): fam["rules"]})
    hand, hprov = FF.resolve_facts(rcfg, C0)
    C, prov = analytic.config_of(analytic_cfg, model=repo, revision=revision, family_rules={FF.family_of(cfg): fam["rules"]}, hand=hand)
    if rcfg.get("quantization_config"):
        C["quantization_config"] = dict(rcfg["quantization_config"])
    prov.update({k: v for k, v in hprov.items() if k in C})
    C["ROPE_THETA"] = float(C["ROPE_THETA"])                            # config.json writes 100000 or 100000.0; the profiles always said float
    if fam["arch"] in ("mla-moe", "dsa-moe"):
        # the prefill FA2 call sees q / k at qk_nope + qk_rope and v padded to it; the decode path never reads D
        C["D"], prov["D"] = int(C["QK_NOPE"]) + int(C["QK_ROPE"]), "derived"
        C["ROT"], prov["ROT"] = int(C["QK_ROPE"]), "derived"
        C["QKV"], prov["QKV"] = int(C["Q_LORA"]) + int(C["KV_LORA"]) + int(C["QK_ROPE"]), "derived"
        if int(cfg["intermediate_size"]) % world:
            raise ValueError(f"intermediate_size={cfg['intermediate_size']} is not divisible by the world {world}")
        C["I_DENSE"], prov["I_DENSE"] = int(cfg["intermediate_size"]) // world, "hf-config"
        indexer_types = tuple(cfg.get("indexer_types") or ())
        if indexer_types:
            if len(indexer_types) < int(C["NL"]):
                raise ValueError(f"indexer_types has {len(indexer_types)} entries for NL={C['NL']}")
            C["INDEXER_LAYERS"], prov["INDEXER_LAYERS"] = tuple(i for i, kind in enumerate(indexer_types[:int(C["NL"])])
                                                                if kind == "full"), "hf-config"
    C = base.with_geometry(C)
    if fam["arch"] in ("mla-moe", "dsa-moe"):
        mk = moe_width_key(cfg)
        if mk:
            C["MOE_I_FULL"], prov["MOE_I_FULL"] = int(cfg[mk]), "hf-config"
        C["ROUTER_VPT"], prov["ROUTER_VPT"] = router_vpt_bf16(int(C["NUM_EXPERTS"])), "derived"
        if C["V_HEAD"] > C["D"]:
            raise ValueError(f"V_HEAD={C['V_HEAD']} > D={C['D']}: the prefill pads v up to the q/k width, never down")
        if C["H"] % 64 or C["I"] % 32 or C["I_DENSE"] % 64:
            raise ValueError(f"H={C['H']} I={C['I']} I_DENSE={C['I_DENSE']}: the pinned GEMM tilings need K a multiple of 64 / 32 (fused_moe_kernel)")
    elif fam["arch"] == "moe":
        mk = moe_width_key(cfg)
        if mk:
            C["MOE_I_FULL"], prov["MOE_I_FULL"] = int(cfg[mk]), "hf-config"
        C["ROUTER_VPT"], prov["ROUTER_VPT"] = router_vpt_bf16(int(C["NUM_EXPERTS"])), "derived"
        if C["H"] % 64 or C["I"] % 32:
            raise ValueError(f"H={C['H']} I={C['I']}: the pinned GEMM tilings need K a multiple of 64 (dense) / 32 (fused_moe_kernel)")
    elif C["H"] % 64 or C["I"] % 64:
        raise ValueError(f"H={C['H']} I={C['I']}: the pinned Gemm tiling (BLOCK_SIZE_K 64) needs K a multiple of 64")
    if world > 1:
        C["WORLD"], prov["WORLD"] = int(world), "hand"
    return C, prov


# ------------------------------------------------------------------------------------------------------------------------------ weights

#: vLLM parameter names of a multi-head-latent-attention MoE decoder layer (`Glm4MoeLite*` / `DeepseekV2*`) -> `analytic.mla_layer_weights_type`
#: fields; layers below FIRST_DENSE carry the dense MLP names, the rest the MoE ones
_MLA_LAYER_FIELDS = {
    "input_layernorm.weight": "ln1", "post_attention_layernorm.weight": "ln2",
    "self_attn.fused_qkv_a_proj.weight": "qkv_a_w", "self_attn.q_a_layernorm.weight": "q_a_ln", "self_attn.q_b_proj.weight": "q_b_w",
    "self_attn.kv_a_layernorm.weight": "kv_a_ln", "self_attn.kv_b_proj.weight": "kv_b_w", "self_attn.o_proj.weight": "o_w",
    "self_attn.fused_qkv_a_proj.weight_scale_inv": "qkv_a_ws", "self_attn.q_b_proj.weight_scale_inv": "q_b_ws",
    "self_attn.kv_b_proj.weight_scale_inv": "kv_b_ws", "self_attn.o_proj.weight_scale_inv": "o_ws",
}
_MLA_DENSE_FIELDS = {
    "mlp.gate_up_proj.weight": "gate_up_w", "mlp.gate_up_proj.weight_scale_inv": "gate_up_ws",
    "mlp.down_proj.weight": "down_w", "mlp.down_proj.weight_scale_inv": "down_ws",
}
_MLA_MOE_FIELDS = {
    "mlp.gate.weight": "gate_w", "mlp.gate.e_score_correction_bias": "gate_bias",
    "mlp.shared_experts.gate_up_proj.weight": "sh_gate_up_w", "mlp.shared_experts.down_proj.weight": "sh_down_w",
    "mlp.experts.routed_experts.w13_weight": "w13", "mlp.experts.routed_experts.w2_weight": "w2",
    "mlp.shared_experts.gate_up_proj.weight_scale_inv": "sh_gate_up_ws",
    "mlp.shared_experts.down_proj.weight_scale_inv": "sh_down_ws",
    "mlp.experts.routed_experts.w13_weight_scale_inv": "w13_ws",
    "mlp.experts.routed_experts.w2_weight_scale_inv": "w2_ws",
}
_MLA_INDEXER_FIELDS = {
    "self_attn.indexer.wq_b.weight": "wq_b_w",
    "self_attn.indexer.wq_b.weight_scale_inv": "wq_b_ws",
    "self_attn.indexer.wk_weights_proj.weight": "wk_w",
    "self_attn.indexer.k_norm.weight": "k_norm_w",
    "self_attn.indexer.k_norm.bias": "k_norm_b",
}
#: `MLAAttention`'s absorbed up-projections: views of `kv_b_proj.weight`'s own storage (W_UK_T = the k_nope rows transposed, W_UV the v
#: rows), so they are no Record field; and its kv/q scale placeholders (`kv_cache_dtype=auto`: never read by a modelled kernel)
MLA_KV_B_VIEWS = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(?:mla_attn\.mla_attn\.)?(W_UK_T|W_UV)$")
_MLA_SCALES = re.compile(rf"^model\.layers\.\d+\.self_attn\.(?:mla_attn\.mla_attn\.)?{_ATTN_SCALES}$")


def _mla_weight_field(C: dict[str, Any]) -> Callable[[str], tuple[str | int, ...] | None]:
    FD, NL = int(C["FIRST_DENSE"]), int(C["NL"])
    indexers = {layer: i for i, layer in enumerate(C.get("INDEXER_LAYERS") or ())}

    def weight_field(name: str) -> tuple[str | int, ...] | None:
        if name == "model.embed_tokens.weight":
            return ("embed",)
        if name == "lm_head.weight":
            return ("lm_head",)
        if name == "model.norm.weight":
            return ("norm",)
        m = base._LAYER_RE.match(name)
        if not m:
            return None
        layer, rest = int(m.group(1)), m.group(2)
        if layer >= NL:
            return None
        if rest == base.COS_SIN_BUFFER:
            return ("cos_sin",) if layer == 0 else None
        if rest in _MLA_INDEXER_FIELDS:
            return ("indexers", indexers[layer], _MLA_INDEXER_FIELDS[rest]) if layer in indexers else None
        f = _MLA_LAYER_FIELDS.get(rest) or (_MLA_DENSE_FIELDS if layer < FD else _MLA_MOE_FIELDS).get(rest)
        if f is None:
            return None
        return ("dense_layers", layer, f) if layer < FD else ("layers", layer - FD, f)

    return weight_field


def make_weight_field(cfg: dict[str, Any], C: dict[str, Any]) -> Callable[[str], tuple[str | int, ...] | None]:
    """`Root.name` -> path into `analytic.weights_type(C)`: the base Llama map plus the family's parameter names, each bound only when the
    Record has the field (a name the Record has no field for stays unbound -- never silently mapped)."""
    fam = FF.facts_for(cfg)
    if fam["arch"] in ("mla-moe", "dsa-moe"):
        return _mla_weight_field(C)
    params = fam.get("params") or {}
    layer_type = analytic.weights_type(C).fields[analytic.weights_type(C).index("layers")][1].elem
    layer_fields = frozenset(n for n, _ in layer_type.fields)
    top_fields = frozenset(n for n, _ in analytic.weights_type(C).fields)
    extra_layer = dict(params.get("layer") or {})
    extra_top = {k: tuple(v) for k, v in (params.get("top") or {}).items()}
    prefix = params.get("layer_prefix")
    if prefix:
        # a family with its own module tree (GPT-NeoX): the family names ARE the map; the base map does not apply
        layer_re = re.compile(_LAYER_RE_FMT.format(prefix=re.escape(prefix)))
        cos_sin = params["cos_sin"]

        def weight_field(name: str) -> tuple[str | int, ...] | None:
            if name in extra_top:
                return extra_top[name] if extra_top[name][0] in top_fields else None
            m = layer_re.match(name)
            if m:
                layer, rest = int(m.group(1)), m.group(2)
                if rest == cos_sin:
                    return ("cos_sin",) if layer == 0 else None
                f = extra_layer.get(rest)
                if f is not None and f in layer_fields:
                    return ("layers", layer, f)
            return None

        return weight_field

    base_wf = base.make_weight_field(C)
    layer_re = base._LAYER_RE
    dense_only = ("mlp.gate_up_proj.weight", "mlp.down_proj.weight", "self_attn.qkv_proj.bias") if fam["arch"] in _MOE_ARCHES else ()

    def weight_field(name: str) -> tuple[str | int, ...] | None:
        if name in extra_top:
            return extra_top[name] if extra_top[name][0] in top_fields else None
        m = layer_re.match(name)
        if m:
            layer, rest = int(m.group(1)), m.group(2)
            if rest in dense_only:
                return None
            f = extra_layer.get(rest)
            if f is None and rest.endswith(_SCALE_INV) and extra_layer.get(rest[:-len(_SCALE_INV)]):
                f = f"{extra_layer[rest[:-len(_SCALE_INV)]]}_ws"          # an FP8 expert stack's block scales (`analytic._fp8_fields`)
            if f is not None:
                return ("layers", layer, f) if f in layer_fields else None
            path = base_wf(name)
            return path if path is None or len(path) < 3 or path[2] in layer_fields else None
        return base_wf(name)

    return weight_field


#: parameters of MoE variants the constructive block does not state (shared experts, expert-parallel maps, router bias)
_UNMODELLED_MOE = re.compile(r"^model\.layers\.\d+\.mlp\.(shared_expert|shared_experts|shared_expert_gate|gate\.e_score_correction_bias|experts\.expert_map)")


def make_root_class(cfg: dict[str, Any], weight_field: Callable[[str], Any]) -> Callable[[str], str]:
    """'weight' / 'alias-of-cos_sin' / 'ignored' / 'unknown' as the base module, over the family's module tree; MoE families add
    'unbound-moe-weight' (a MoE parameter the Record has no field for) and 'unmodelled-moe-variant' (shared experts, EP maps, router bias)."""
    fam = FF.facts_for(cfg)
    if fam["arch"] in ("mla-moe", "dsa-moe"):
        def mla_root_class(name: str) -> str:
            if weight_field(name) is not None:
                return "weight"
            if base.ALIAS_OF_COS_SIN.match(name):
                return "alias-of-cos_sin"
            if MLA_KV_B_VIEWS.match(name):
                return "alias-of-kv_b"
            if _MLA_SCALES.match(name):
                return "ignored"
            return "unknown"
        return mla_root_class
    params = fam.get("params") or {}
    prefix = params.get("layer_prefix")
    if prefix:
        alias = re.compile(rf"^{re.escape(prefix)}\.[1-9]\d*\.{re.escape(params['cos_sin'])}$")
        ignored = (re.compile(rf"^{re.escape(prefix)}\.\d+\.{re.escape(params['attn_scales'])}\.{_ATTN_SCALES}$"),)
    else:
        alias, ignored = base.ALIAS_OF_COS_SIN, base.IGNORED_ROOTS
    moe_names = frozenset((params.get("layer") or {}).keys()) if fam["arch"] in _MOE_ARCHES else frozenset()

    def root_class(name: str) -> str:
        if weight_field(name) is not None:
            return "weight"
        if alias.match(name):
            return "alias-of-cos_sin"
        if any(r.match(name) for r in ignored):
            return "ignored"
        if moe_names:
            if _UNMODELLED_MOE.match(name):
                return "unmodelled-moe-variant"
            m = base._LAYER_RE.match(name)
            if m and m.group(2) in moe_names:
                return "unbound-moe-weight"
        return "unknown"

    return root_class


# ------------------------------------------------------------------------------------------------------------------------------- header

_MOE_HF_KEYS = {"num_experts": "NUM_EXPERTS", "num_experts_per_tok": "TOPK", "norm_topk_prob": "NORM_TOPK_PROB"}


def make_check_header(cfg: dict[str, Any], C: dict[str, Any], world: int) -> Callable[[RunHeader], list[str]]:
    """Mismatches between the run's recorded HF config and `C`: the base check (over the recorded FULL config re-sharded to this rank when
    `world > 1`), the family's `key` facts, the MoE shape keys, the tensor-parallel size."""
    fam = FF.facts_for(cfg)
    base_check = base.make_check_header(C)
    mk = moe_width_key(cfg)

    def check_header(header: RunHeader) -> list[str]:
        hcfg = (header.model or {}).get("config") or {}
        out: list[str] = []
        if hcfg:
            try:
                rk = rank_hf_config(hcfg, world) if (world > 1 or mk) else dict(hcfg)
            except (KeyError, ValueError) as e:
                return [f"recorded config cannot be sharded by {world}: {e}"]
            if mk:
                rk.pop("intermediate_size", None)          # the dense width of no layer here; MOE_I_FULL is checked below
            out += base_check(dataclasses.replace(header, model={**(header.model or {}), "config": rk}))
            out += FF.header_checks(cfg, C, hcfg)
            if fam["arch"] in _MOE_ARCHES:
                for hf, key in _MOE_HF_KEYS.items():
                    if hf in hcfg and hcfg[hf] != C.get(key):
                        out.append(f"{hf}={hcfg[hf]!r} != C[{key}]={C.get(key)!r}")
                if mk and hcfg.get(mk) is not None and hcfg[mk] != C.get("MOE_I_FULL"):
                    out.append(f"{mk}={hcfg[mk]!r} != C[MOE_I_FULL]={C.get('MOE_I_FULL')!r}")
                if hcfg.get("mlp_only_layers"):
                    out.append(f"mlp_only_layers={hcfg['mlp_only_layers']!r}: the MoE profile states every layer as a fused-MoE layer")
            if fam["arch"] in ("mla-moe", "dsa-moe"):
                expected = {
                    "intermediate_size": int(C["I_DENSE"]) * world,
                    "moe_intermediate_size": C.get("MOE_I_FULL"),
                    "num_experts_per_tok": C.get("TOPK"),
                }
                for hf, value in expected.items():
                    if hf in hcfg and hcfg[hf] != value:
                        out.append(f"{hf}={hcfg[hf]!r} != expected {value!r} from the rank profile")
        tp = (header.model or {}).get("tp") or (header.model or {}).get("tensor_parallel_size")
        if tp is not None and int(tp) != world:
            out.append(f"tensor_parallel_size={tp!r} != WORLD={world}")
        return out

    return check_header


# ------------------------------------------------------------------------------------------------------------------------------ profile

def profile_id(role: str, C: dict[str, Any], world: int) -> str:
    dtype_tag = f"fp8-block{C['FP8_BLOCK']}" if C.get("FP8_BLOCK") else "bf16"
    return f"vllm-d9105ea80-sm89-eager-{dtype_tag}-{role.lower()}" + (f"-tp{world}-rank" if world > 1 else "")


def _dense_patterns(cfg: dict[str, Any], C: dict[str, Any], patterns: list) -> list:
    fam = FF.facts_for(cfg)
    kinds = tuple(fam.get("patterns") or ())
    out = with_tuned_gemm(patterns)                                   # the target's tuned-GEMM table rows, ahead of the base GemmLaunch
    if effective_sliding_window(cfg):
        out = with_dense2_patterns(out, C)                            # the inert-window FA2 scope decides from the LAUNCH (window >= max_seqlen_k)
    if "unfused-norm-chains" in kinds:
        out = with_gemma2_patterns(out)                               # unfused GemmaRMSNorm(+1) / GeGLU / embed-scale scalar chains -> quarantine.dense
    if "softcap" in kinds and C.get("attn_logit_softcapping"):
        out = with_softcap_patterns(out)                              # FA2 softcap launches -> quarantine.dense.AttentionSoftcap_v1{..., CAP}
    if "ln" in kinds:
        out = with_ln_patterns(out, C)
    return out


def _moe_patterns(C: dict[str, Any], patterns: list, *, world: int, moe_construction: str | None, fused_pins, fused_num_warps, fused_num_stages) -> list:
    key = moe_construction or TargetProfile.DEFAULT_MOE_CONSTRUCTION
    table = MOE_FP8_CONSTRUCTION_PATTERNS if C.get("FP8_BLOCK") else MOE_CONSTRUCTION_PATTERNS
    if key not in table:
        raise ValueError(f"moe_construction {moe_construction!r} is not one of {sorted(table)}" + (" (block-FP8 experts)" if C.get("FP8_BLOCK") else ""))
    # the rank's scale grid, vLLM's refined block (64 for the 235B at TP8), is the FP8 scope's group
    group = {"group": analytic.fp8_moe_block(int(C["FP8_BLOCK"]), int(C["I"]), int(C["H"]))} if C.get("FP8_BLOCK") else {}
    coll = TpCollective(world=world) if world > 1 else None
    moe = table[key](C, router_vpt={**ROUTER_VPT_BY_E, int(C["NUM_EXPERTS"]): int(C["ROUTER_VPT"])},
                     renormalize=bool(C.get("RENORMALIZE", False)), scope_patterns=(coll,) if coll else (),
                     fused_pins=dict(FUSED_MOE_PINS if fused_pins is None else fused_pins),
                     fused_num_warps=FUSED_MOE_NUM_WARPS if fused_num_warps is None else int(fused_num_warps),
                     fused_num_stages=FUSED_MOE_NUM_STAGES if fused_num_stages is None else int(fused_num_stages), **group)
    out = with_tuned_gemm(patterns)
    out.insert(next(i for i, p in enumerate(out) if type(p).__name__ == "GemmLaunchTuned") + 1, moe)   # tuned GEMM, MoE scope, then the base list
    return out


def _dsa_patterns(C: dict[str, Any], patterns: list) -> list:
    """GLM-5.3's pinned DSA kernels and block-FP8 shared-expert scope."""
    if int(C.get("FP8_BLOCK") or 0) != 128:
        raise ValueError(f"dsa-moe requires the recorded block-128 FP8 experts, got FP8_BLOCK={C.get('FP8_BLOCK')!r}")
    out = with_tuned_gemm(patterns)
    moe = MoEFusedSharedExpertsFp8(C, group=128, gate_spec=None)
    out.insert(next(i for i, p in enumerate(out) if type(p).__name__ == "GemmLaunchTuned") + 1, moe)
    return dsa_patterns() + out


def build_profile(role: str, world: int = 1, *, moe_construction: str | None = None, fused_moe_pins: dict[str, Any] | None = None,
                  fused_moe_num_warps: int | None = None, fused_moe_num_stages: int | None = None, ov_sampling: bool = False,
                  hf_config: dict[str, Any] | None = None, repo: str | None = None, revision: str | None = None) -> Profile:
    """The κ profile of a checkpoint role (one rank of a `world`-way engine when `world > 1`) from its HF config.json and the family facts.

    `moe_construction` (MoE roles): None = `TargetProfile.DEFAULT_MOE_CONSTRUCTION`; `indexed-read` = the MOE-01 pairwise fold;
    `indexed-read-ordered` = MOE-01 with the router in selection order; `padded` = one fixed-slot Compute per token (refused at
    fold time under a renormalising router).  `fused_moe_pins` / `..._num_warps` / `..._num_stages`: the fused_moe_kernel tiling to pin instead
    of `gen_ov_moe_patterns.FUSED_MOE_PINS` (a κ fact of the capture; never loosened, only re-pinned from a rank log).  `hf_config`: the
    document to derive from (default: `load_hf_config(role)`; tests pass a synthetic one).  `ov_sampling`: the experimental sampled-row
    variant (`TokenSampleGumbel` first, id `+ov-sampling`; greedy headers resolve exactly as without it)."""
    if hf_config is None:
        cfg, src = load_hf_config(role)
        repo, revision = repo or src["repo"], revision or src["revision"]
    else:
        cfg, src = dict(hf_config), {"source": "caller", "path": None, "sha256": hashlib.sha256(json.dumps(hf_config, sort_keys=True).encode()).hexdigest(),
                                     "repo": repo, "revision": revision}
    C, prov = derive_config(cfg, repo=repo, revision=revision, world=world)
    fam = FF.facts_for(cfg)
    arch = fam["arch"]
    facts = fam["facts"]
    hand_note = ", ".join(f"{k}={C[k]!r}" for k in facts if k in C)
    note = (f"{repo or role} @ {(revision or '')[:8] or '?'} ({fam['vllm_class']}; profiles.generic over config.json + FAMILY_FACTS[{FF.family_of(cfg)!r}])"
            + (f" ONE RANK of TP{world} (rank-local shard configuration)" if world > 1 else "")
            + f": H={C['H']} NL={C['NL']} NH={C['NH']} KVH={C['KVH']} D={C['D']} I={C['I']} V={C['V']} QKV={C['QKV']}, "
            + ("QKV bias" if C["BIAS"] else "no QKV bias") + ", " + ("tied" if C["tie_word_embeddings"] else "untied") + f" embeddings, FA2 kBlockN {C['BN']}"
            + (f"; E={C['NUM_EXPERTS']} top-{C['TOPK']} " + ("renormalised" if C.get("RENORMALIZE") else "unnormalised") + " router"
               if arch in _MOE_ARCHES else "")
            + (f"; hand facts: {hand_note}" if hand_note else ""))
    prof = base.build_profile(C, profile_id=profile_id(role, C, world), case=role, model_note=note)
    wf = make_weight_field(cfg, C)
    if arch == "moe":
        patterns = _moe_patterns(C, list(prof.patterns), world=world, moe_construction=moe_construction, fused_pins=fused_moe_pins,
                                 fused_num_warps=fused_moe_num_warps, fused_num_stages=fused_moe_num_stages)
    elif arch == "dsa-moe":
        patterns = _dsa_patterns(C, list(prof.patterns))
    elif arch == "mla-moe":
        from verity_vllm.engine.profiles.gen_mla_moe_patterns import with_mla_moe_patterns
        patterns = with_mla_moe_patterns(with_tuned_gemm(list(prof.patterns)), C)
    else:
        patterns = _dense_patterns(cfg, C, list(prof.patterns))       # [R15 fp8] the fp8 patterns are the base list's own (make_patterns)
    if world > 1:
        patterns = [EmbeddingShardCuda(), TpCollective(world=world)] + patterns
        for pt in patterns:                                             # the sampler sees the all-gathered logits: the full vocabulary
            if isinstance(pt, (TokenSelectGumbelTopP, TokenSelectGumbelTwoStage, TokenSelectArgmax)):
                pt.vocab = int(cfg["vocab_size"])
    notes = dict(prof.notes)
    notes["pins"] = dict(notes["pins"])
    notes["pins"]["gemm_tuned_ada"] = {f"N={n},K={k}": row for (n, k), row in ADA_TUNED_GEMM.items()}
    notes.update({
        "derived_by": "profiles.generic [P6]: no per-model module; applicability = patterns (kernels) + check_header (facts); family facts = profiles.family_facts",
        "family": FF.family_of(cfg), "arch": arch, "vllm_class": fam["vllm_class"],
        "hand_facts": {k: {"value": C.get(k), "provenance": f["provenance"]} for k, f in facts.items() if k in C},
        # the fields NOT read off config.json by `analytic.config_of`'s generic rules: the identity and the family's hand facts (a `key`
        # fact's VALUE is config.json's; that vLLM reads the key is the family fact) -- the retired hand modules' `hand_fields`
        "hand": {**({"model": repo} if repo else {}), **({"revision": revision} if revision else {}), **{k: C[k] for k in facts if k in C}},
        "hand_fields": sorted([*(["model"] if repo else []), *(["revision"] if revision else []), *[k for k in facts if k in C]]),
        "family_rules": {FF.family_of(cfg): fam["rules"]}, "family_rules_provenance": fam["rules_provenance"],
        "provenance": dict(prov), "config_provenance": dict(prov),
        "hf_config": {"source": src["source"], "path": src["path"], "sha256": src["sha256"], "revision": revision},
        "root_class": "profiles.generic.make_root_class(cfg, profile.weight_field)(name)",
    })
    names = [p.name for p in patterns]
    notes["pattern_names"] = names
    notes["dense2_patterns"] = [n for n in names if n == "attention.fa2_varlen_fwd.inert_window"]
    notes["softcap_pattern"] = [n for n in names if n == "attention.fa2_varlen_fwd.softcap"]
    if C.get("extra_layer_norms"):
        notes["extra_layer_norms"] = list(C["extra_layer_norms"])
    params = fam.get("params") or {}
    if params.get("layer_prefix"):
        notes["alias_of_cos_sin"] = rf"^{re.escape(params['layer_prefix'])}\.[1-9]\d*\.{re.escape(params['cos_sin'])}$"
    else:
        notes["alias_of_cos_sin"] = base.ALIAS_OF_COS_SIN.pattern
    if C.get("FP8_BLOCK"):
        notes["fp8"] = {"scheme": "block-scaled dynamic e4m3 FP8 (vLLM Fp8LinearMethod -> CutlassFp8BlockScaledMMKernel)",
                        "weight_block_size": [C["FP8_BLOCK"], C["FP8_BLOCK"]], "activation_group": C["FP8_BLOCK"],
                        "kernels": ["_C.per_token_group_fp8_quant (per_token_group_quant_8bit_kernel)",
                                    "_C.cutlass_scaled_mm blockwise (cutlass_3x_gemm_fp8_blockwise<bf16>, sm90)"],
                        "env": ["VLLM_USE_DEEP_GEMM=0 (the DeepGEMM JIT path is not modelled)", "VLLM_USE_FLASHINFER_SAMPLER=0"],
                        "families": ["Fp8GroupQuant_v1{K,G}", "ScaledMmFp8BlockSharedScale_v1{K,N,G} (fp8_scale_construction per-coordinate: "
                                     "ScaledMmFp8Block_v1{K,N,G})"], "quantization_config": C.get("quantization_config")}
    if arch in _MOE_ARCHES:
        moe = next(p for p in patterns if isinstance(p, (MoEFusedExperts, MoEFusedSharedExpertsFp8)))
        E, T, I, H = C["NUM_EXPERTS"], C["TOPK"], C["I"], C["H"]
        if isinstance(moe, MoEFusedExpertsFp8):
            fp8 = f"G={moe.group},KB={moe.fused_pins['BLOCK_SIZE_K']},SWAP=False,DOT=the record target's e4m3 step"
            experts = [f"Fp8GroupQuant_v1{{K={H},G={moe.group}}}", f"MoeExpertFp8Gemm_v1{{E={E},K={H},N={2 * I},{fp8}}}", f"SiluMul_v2{{I={I}}}",
                       f"Fp8GroupQuant_v1{{K={I},G={moe.group}}}", f"MoeExpertFp8GemmW_v1{{E={E},K={I},N={H},{fp8}}}"]
        else:
            experts = [f"MoeExpertGemm_v1{{E={E},K={H},N={2 * I}}}", f"SiluMul_v2{{I={I}}}", f"MoeExpertGemmW_v1{{E={E},K={I},N={H}}}"]
        notes["moe"] = {"status": ("constructive except the gate: no registered bf16 x bf16 -> f32 gate Definition"
                                     if arch == "dsa-moe" else "constructive: Computes over registry.moe definitions per step per rank"),
                        "scope_pattern": f"gen_ov_moe_patterns.{type(moe).__name__} (vllm.moe_forward*), renormalize={bool(C.get('RENORMALIZE', False))}",
                        "definitions": [f"MoeRouterTopK{'Ordered' if moe.router == 'ordered' else ''}{'Norm' if C.get('RENORMALIZE') else ''}_v1"
                                        f"{{E={E},TOPK={T},VPT={C['ROUTER_VPT']}}}", *experts,
                                        f"MoeSum_v1{{TOPK={T},H={H}}}"] + ([f"AllReduce2_v1{{N={H}}} (the block's TP all-reduce, in or after the scope)"] if world > 1 else []),
                        "pins": {"fused_moe_kernel": {**moe._numeric_pins(), **moe.fused_pins, "num_warps": moe.fused_num_warps, "num_stages": moe.fused_num_stages,
                                                      "source": "gen_ov_moe_patterns.FUSED_MOE_PINS (vLLM BI default, M-invariant)" if fused_moe_pins is None
                                                                else "injected (build_profile(fused_moe_pins=...))"},
                                 "router_vpt_by_E": {E: C["ROUTER_VPT"]},
                                 "alignment": "moe_align_block_size is Transparent(noop): a pair->expert-block schedule with no numerical effect"},
                        "construction": ("shared-experts-pairwise" if arch == "dsa-moe"
                                         else moe_construction or TargetProfile.DEFAULT_MOE_CONSTRUCTION),
                        "sharding": {"router": "replicated (full E logits on every rank)", "experts": f"intermediate axis / {world}: w13 [E, 2*{I}, H], w2 [E, H, {I}]",
                                     "reduce": "moe_sum rank-local, then AllReduce2_v1{N=H} of the routed sum"} if world > 1 else "none (TP1)",
                        "record_fields_present": sorted(f for f in ("q_norm", "k_norm", "gate_w", "w13", "w2")
                                                        if f in {n for n, _ in analytic.weights_type(C).fields[analytic.weights_type(C).index("layers")][1].elem.fields})}
    if world > 1:
        notes["tp_rank"] = {"world": world, "rank_config_divided": list(sharded_keys(cfg)), "recomputed": {"QKV": C["QKV"], "GU": C["GU"]},
                            "sampler_vocab": int(cfg["vocab_size"]), "own_patterns": [EmbeddingShardCuda.name, TpCollective.name],
                            "collectives": "[R17 D90] vllm.all_reduce / vllm.all_gather -> AllReduce2_v1{N} / AllGather2_v1{N} per row over (own partial, "
                                           "peer partial) in rank order; the peer partial is the fold's `peers_<request>` root parameter (root class "
                                           "collective_received), linked to the other rank's committed partial by the cross-rank leg (tp2_rank_match)"}
    pid = prof.id
    if ov_sampling:
        from verity_vllm.engine.profiles.gen_ov_sampling_patterns import ov_sampling_patterns
        patterns = ov_sampling_patterns(C) + patterns
        pid = prof.id + "+ov-sampling"
        notes["ov_sampling"] = {"pattern": "gen_ov_sampling_patterns.TokenSampleGumbel (_gumbel_sample_kernel; seeded rows at T=1 and T not in {0,1}, no top-k/top-p)",
                                "definition": "quarantine ov-sampling/GumbelArgmaxBf16@v1 {V, SEED, POS}: argmax_v(f32(logits[v]) + g(SEED, POS)[v]); "
                                              "GumbelArgmaxTempBf16@v1 {V, SEED, POS, T} for tempered rows (div.full.f32 quotient)",
                                "seed_provenance": "SEED static from the run header's request record (children: parent seed + index); "
                                                   "the device seeds buffer is not a checked fact -- PROPOSED Root(role=sampling_seed)"}
    return dataclasses.replace(prof, id=pid, patterns=patterns, weight_field=wf, notes=notes, accepted=False)


def with_moe_construction(profile: Profile, moe_construction: str | None) -> Profile:
    """The fold's hook (run_config `fold_summary_result`): this profile under the workload's DECLARED MoE construction."""
    if moe_construction in (None, TargetProfile.DEFAULT_MOE_CONSTRUCTION):
        return profile
    return build_profile(profile.notes["case"], int(profile.config.get("WORLD", 1)), moe_construction=moe_construction)


# ------------------------------------------------------------------------------------------------------------------------------- module

def make_module(fullname: str) -> types.ModuleType:
    """The module object behind `verity_vllm.engine.profiles.derived_<ROLE>[_tp<W>]` (and the X-07 `dense_<ROLE>`): PROFILE / C / PROVENANCE /
    HF_CONFIG / weight_field / root_class / check_header / build_profile / with_moe_construction."""
    parsed = parse_module(fullname)
    if parsed is None:
        raise ImportError(f"{fullname}: not a derived_<ROLE>[_tp<W>] profile name")
    role, world = parsed
    ovs = is_ov_sampling(fullname)
    try:
        cfg, src = load_hf_config(role)
        if fullname.rsplit(".", 1)[-1].startswith(_dg.MODULE_PREFIX) and FF.FAMILY_FACTS.get(FF.family_of(cfg), {}).get("arch") != "dense":
            raise ImportError(f"{fullname}: the dense_<ROLE> name is for dense-architecture roles; use {module_name(role, world)}")
        prof = build_profile(role, world, ov_sampling=ovs)
    except ValueError as e:
        raise ImportError(f"{fullname}: {e}") from e
    mod = types.ModuleType(fullname)
    mod.__doc__ = f"[P6] configuration-derived κ profile for role {role}" + (f", one rank of TP{world}" if world > 1 else "") + f" ({src['source']}: {src['path']})"
    mod.CASE, mod.ROLE, mod.WORLD, mod.REPO_ID, mod.REVISION = role, role, world, src["repo"], src["revision"]
    mod.MODEL = src["repo"]                                             # the retired hand modules' name for the repo id
    mod.HAND = dict(prof.notes["hand"])                                 # identity + the family's hand facts as resolved for this role
    mod.HF_CONFIG_PATH = __import__("pathlib").Path(src["path"]) if src["path"] else None
    mod.CONFIG_JSON, mod.CONFIG_SHA256, mod.HF_CONFIG = mod.HF_CONFIG_PATH, src["sha256"], cfg
    mod.FAMILY, mod.FAMILY_RULES = FF.family_of(cfg), {FF.family_of(cfg): FF.facts_for(cfg)["rules"]}
    mod.C, mod.PROVENANCE = prof.config, prof.notes["provenance"]
    mod.PROFILE_ID, mod.PROFILE = prof.id, prof
    mod.weight_field = prof.weight_field
    mod.root_class = make_root_class(cfg, prof.weight_field)
    mod.check_header = make_check_header(cfg, prof.config, world)
    mod.build_profile = lambda world=world, moe_construction=None, **kw: build_profile(role, world, moe_construction=moe_construction,
                                                                                       **{"ov_sampling": ovs, **kw})
    mod.derive_config = lambda hf_config, world=1, **kw: derive_config(hf_config, repo=kw.get("repo"), revision=kw.get("revision"), world=world)
    mod.with_moe_construction = with_moe_construction if FF.facts_for(cfg)["arch"] == "moe" else None
    if FF.facts_for(cfg)["arch"] == "dense":
        from verity_vllm.engine.profiles.gen_ov_easy_family import serve_family                    # the untied / tied serving-family reference `compare` may consult
        mod.SERVE_FAMILY = serve_family
    mod.__file__ = __file__
    return mod
