"""verity_vllm/program/model.py — Phase 0 of the generalization sweep: what a configuration *is*, and the roots that are
functions of it.

    config_of(hf_config, model=..., revision=...)  -> (C, provenance)   static configuration from a checkpoint's config.json
    weights_type(C)                                -> Record            b1.weights_type(C) plus an `lm_head` field when untied
    cos_sin_table(C)                               -> uint16 [MAX_POS, D]   the rotary table vLLM computes, recomputed here
    check_cos_sin(C, captured_npy)                 -> dict              gate I9: captured table within COS_SIN_MAX_ULP of analytic

Why this exists (plan §2 I9, §3): the M6 comparison took the rotary table from vLLM's own buffer, so nothing checked that the
table is the function of (ROPE_THETA, D, MAX_POS, rope_scaling) it claims to be.  Here it is recomputed from `C` with the same
formula as vLLM's `RotaryEmbedding._compute_cos_sin_cache` (fp32 `inv_freq = 1 / base ** (arange(0, D, 2) / D)`, `freqs = outer(t,
inv_freq)`, `cat(cos, sin)`, then bf16 round-to-nearest-even) and, for `rope_scaling.rope_type == "llama3"`, vLLM's
`Llama3RotaryEmbedding._compute_inv_freq` wavelength blend.  vLLM evaluates cosf/sinf/powf on the GPU (CUDA libm) and numpy on
the CPU (system libm); those differ by an fp32 ulp or so, which flips a bf16 rounding only at a boundary — hence the gate is
"within 1 bf16 ulp", not bit-exact, and the captured table stays the committed root; the check catches a wrong θ, D, MAX_POS,
layout or scaling rule, which is what a configuration error looks like.

`config_of` is the compression target for the sweep's cards: every field carries a provenance ("hf-config": read from
config.json; "derived": computed from other fields; "hand": supplied by the caller because the checkpoint does not carry it).
Torch-free.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.program.registry import b1
from verity_vllm.program.registry.prims import BF16
from verity.ml.circuits.types import Array, Record

COS_SIN_MAX_ULP = 1     # gate I9 (frozen with gates.py; change only through an approved decision)

#: model_type -> family facts vLLM hard-codes (not in config.json).  `qkv_bias`: Qwen2Attention passes bias=True unconditionally;
#: Llama uses `attention_bias` (default False).  Anything not listed here is "hand" for BIAS.
FAMILY_RULES: dict[str, dict[str, Any]] = {
    "llama": {"qkv_bias_key": "attention_bias", "default_theta": 10_000.0},
    "qwen2": {"qkv_bias": True, "default_theta": 1_000_000.0},
    "mistral": {"qkv_bias_key": "attention_bias", "default_theta": 10_000.0},
    "gemma2": {"qkv_bias_key": "attention_bias", "default_theta": 10_000.0},
    "phi3": {"qkv_bias_key": "attention_bias", "default_theta": 10_000.0},
    "olmoe": {"qkv_bias_key": "attention_bias", "default_theta": 10_000.0, "default_eps": 1e-5},
    # [R15 fp8] Qwen3 (vLLM `Qwen3Attention`, models/qwen3.py): the Llama-architecture dense block PLUS a per-head RMSNorm on q and on k
    # before RoPE (`self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)`, applied to `q.view(..., NH, head_dim)`; likewise k over KVH
    # heads), no QKV bias (`attention_bias` False), head_dim in config.json.  `qk_norm: "head"` is the family fact (`C["QK_NORM"]`).
    "qwen3": {"qkv_bias_key": "attention_bias", "default_theta": 1_000_000.0, "qk_norm": "head"},
}

#: [R15 fp8] the ONE checkpoint quantization the sweep serves: vLLM `Fp8Config` from a `quantization_config` with `quant_method: fp8`,
#: `fmt: e4m3`, `weight_block_size: [G, G]` (block-scaled weights `weight` e4m3 [N, K] + `weight_scale_inv` f32 [N/G, K/G]) and
#: `activation_scheme: dynamic` (per-token-group activation quantisation, group G, at run time).  `config_of` records it as
#: `C["FP8_BLOCK"] = G`; any other quantization_config is refused (`quant_refusal`) -- never served as a looser approximation.
FP8_BLOCK_QUANT = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}


def _moe_routers(cfg: dict[str, Any] | None) -> set[str]:
    """The router module of every layer of a fused-MoE configuration (`mlp.gate`, the bf16 `gate_w` of `moe_layer_weights_type`); empty
    for a dense configuration or none."""
    cfg = cfg or {}
    if not any(cfg.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts")):
        return set()
    return {f"model.layers.{n}.mlp.gate" for n in range(int(cfg.get("num_hidden_layers") or 0))}


def quant_refusal(qc: dict[str, Any] | None, cfg: dict[str, Any] | None = None) -> str | None:
    """None when `qc` is absent (bf16 checkpoint) or exactly the block-scaled dynamic FP8 scheme; else why it is not served.  `cfg` is
    the checkpoint's config.json: a fused-MoE one serves FP8 experts with bf16 routers, so it must list every layer's router."""
    if not qc:
        return None
    for k, v in FP8_BLOCK_QUANT.items():
        if str(qc.get(k, "")).lower() != v:
            return f"quantization_config.{k}={qc.get(k)!r} is not {v!r} (only block-scaled dynamic e4m3 FP8 is modelled)"
    bs = qc.get("weight_block_size")
    if not (isinstance(bs, (list, tuple)) and len(bs) == 2 and bs[0] == bs[1] and int(bs[0]) > 0 and int(bs[0]) % 32 == 0):
        return f"quantization_config.weight_block_size={bs!r} is not a square block [G, G] with G a multiple of 32"
    # `modules_to_not_convert` may only name modules that are no linear projection anyway (the tied `lm_head`, the RMSNorms -- Qwen3-4B-FP8
    # lists exactly those) or, on a MoE, the routers the MoE weights type keeps bf16; any other LINEAR left in bf16 would make the layer
    # mixed-precision, which is not modelled
    kept = list(qc.get("modules_to_not_convert") or ())
    routers = _moe_routers(cfg)
    mixed = [m for m in kept if not (m == "lm_head" or m.endswith("norm") or m in routers)]
    if mixed:
        return f"quantization_config.modules_to_not_convert names linears kept in bf16 {mixed[:4]!r}: mixed-precision layers are not modelled"
    fp8_routers = sorted(routers - set(kept), key=lambda m: int(m.split(".")[2]))
    if fp8_routers:
        return (f"quantization_config quantizes the MoE routers {fp8_routers[:4]!r} ({len(fp8_routers)} of {len(routers)} layers not in "
                f"modules_to_not_convert): the MoE weights type keeps every router bf16, an FP8 router is not modelled")
    return None


def config_of(hf_config: str | Path | dict, *, model: str | None = None, revision: str | None = None,
              hand: dict[str, Any] | None = None, family_rules: dict[str, dict[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """The static configuration `C` (keys of `b1.B0_CONFIG` / `B1_CONFIG` plus `family`, `rope_scaling`, `BN`) from a
    checkpoint's `config.json`, with a provenance per key.  `hand` overrides / supplies keys the checkpoint lacks (each is
    recorded as "hand")."""
    cfg = json.loads(Path(hf_config).read_text()) if not isinstance(hf_config, dict) else dict(hf_config)
    prov: dict[str, str] = {}
    C: dict[str, Any] = {}

    def take(key: str, *hf_keys: str, default: Any = None, required: bool = True) -> None:
        for k in hf_keys:
            if k in cfg and cfg[k] is not None:
                C[key] = cfg[k]
                prov[key] = "hf-config"
                return
        if default is not None or not required:
            C[key] = default
            prov[key] = "derived"
            return
        raise KeyError(f"config.json has none of {hf_keys} for {key}")

    family = str(cfg.get("model_type", "")).lower()
    rules = {**FAMILY_RULES, **(family_rules or {})}.get(family, {})    # lanes pass their family's facts here (no edit of this file)
    C["model"], prov["model"] = model, "hand" if model else "derived"
    C["revision"], prov["revision"] = revision, "hand" if revision else "derived"
    C["family"], prov["family"] = family, "hf-config"
    take("H", "hidden_size")
    take("NL", "num_hidden_layers")
    take("NH", "num_attention_heads")
    take("KVH", "num_key_value_heads", default=C["NH"])
    if cfg.get("head_dim"):
        C["D"], prov["D"] = int(cfg["head_dim"]), "hf-config"
    else:
        C["D"], prov["D"] = C["H"] // C["NH"], "derived"
    take("I", "intermediate_size")
    take("V", "vocab_size")
    # OLMoE's config.json carries no epsilon: vLLM's OlmoeForCausalLM hard-codes RMSNorm(eps=1e-5) (family fact, "hand")
    take("EPS", "rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon", default=rules.get("default_eps"), required="default_eps" not in rules)
    if prov.get("EPS") == "derived" and "default_eps" in rules:
        prov["EPS"] = "hand"
    # MoE facts, when present (the routing itself is a mechanism question; these are just the static shape facts)
    for key, hf in (("NUM_EXPERTS", "num_experts"), ("TOPK", "num_experts_per_tok"), ("NORM_TOPK_PROB", "norm_topk_prob")):
        if hf in cfg and cfg[hf] is not None:
            C[key], prov[key] = cfg[hf], "hf-config"
    for hf in ("num_local_experts", "n_routed_experts"):
        if "NUM_EXPERTS" not in C and hf in cfg and cfg[hf] is not None:
            C["NUM_EXPERTS"], prov["NUM_EXPERTS"] = cfg[hf], "hf-config"
    rp = cfg.get("rope_parameters") if isinstance(cfg.get("rope_parameters"), dict) else None
    if cfg.get("rope_theta") is None and rp and rp.get("rope_theta") is not None:
        C["ROPE_THETA"], prov["ROPE_THETA"] = float(rp["rope_theta"]), "hf-config"     # transformers >= 5: rope_parameters.rope_theta
    else:
        take("ROPE_THETA", "rope_theta", default=float(rules.get("default_theta", 10_000.0)))
    take("MAX_POS", "max_position_embeddings")
    C["QKV"], prov["QKV"] = (C["NH"] + 2 * C["KVH"]) * C["D"], "derived"
    C["GU"], prov["GU"] = 2 * C["I"], "derived"
    if "qkv_bias" in rules:
        C["BIAS"], prov["BIAS"] = bool(rules["qkv_bias"]), "derived"
    elif rules.get("qkv_bias_key") in cfg:
        C["BIAS"], prov["BIAS"] = bool(cfg[rules["qkv_bias_key"]]), "hf-config"
    elif "attention_bias" in cfg:
        C["BIAS"], prov["BIAS"] = bool(cfg["attention_bias"]), "hf-config"
    else:
        C["BIAS"], prov["BIAS"] = False, "derived"
    take("tie_word_embeddings", "tie_word_embeddings", default=False)
    rs = cfg.get("rope_scaling") or rp
    if isinstance(rs, dict) and (rs.get("rope_type") or rs.get("type") or "default") == "default" and set(rs) <= {"rope_type", "type", "rope_theta"}:
        rs = None                                    # v5 `rope_parameters` with only the theta: no scaling
    C["rope_scaling"], prov["rope_scaling"] = rs, "hf-config"
    C["BN"], prov["BN"] = b1.fa2_kblock_n(C["D"]), "derived"
    # [R15 fp8] family fact: the per-head q/k RMSNorm (Qwen3) -- "hand" because vLLM's model code applies it, config.json does not say
    if rules.get("qk_norm"):
        C["QK_NORM"], prov["QK_NORM"] = rules["qk_norm"], "hand"
    # [R15 fp8] checkpoint quantization: block-scaled dynamic e4m3 FP8 (`C["FP8_BLOCK"] = G`) or none; anything else refuses here
    qc = cfg.get("quantization_config")
    if qc:
        why = quant_refusal(qc, cfg)
        if why:
            raise ValueError(why)
        C["FP8_BLOCK"], prov["FP8_BLOCK"] = int(qc["weight_block_size"][0]), "hf-config"
        C["quantization_config"], prov["quantization_config"] = dict(qc), "hf-config"
    for k, v in (hand or {}).items():
        C[k], prov[k] = v, "hand"
    return C, prov


def moe_layer_weights_type(C: dict[str, Any]) -> Record:
    """PROPOSED (lane moe, OLMoE): the per-layer Record of a fused-MoE decoder layer.  `b1.layer_weights_type(C)` with the
    dense MLP (`gate_up_w`, `down_w`) replaced by the router and the expert stacks, plus the qk-norm weights when the family
    applies them (`C["QK_NORM"]`, a "hand" fact of vLLM's `OlmoeAttention`).  Shapes are vLLM d9105ea80's parameters:
    `mlp.gate.weight [E][H]`, `mlp.experts.routed_experts.w13_weight [E][2I][H]` (gate|up fused), `w2_weight [E][H][I]`.
    Field order = parameter order in the checkpoint's layer (b1's order for the shared fields).  With `C["FP8_BLOCK"]` the linears but
    the router are e4m3, each followed by its block scales (`_fp8_fields`)."""
    H, QKV, E, I = C["H"], C["QKV"], C["NUM_EXPERTS"], C["I"]
    fields: list[tuple[str, Any]] = [("ln1", Array(H, BF16)), ("qkv_w", Array(QKV, Array(H, BF16)))]
    if C.get("BIAS", True):
        fields.append(("qkv_b", Array(QKV, BF16)))
    if C.get("QK_NORM") == "head":
        # [R19 qwen3] Qwen3-MoE (`Qwen3MoeAttention`): the per-head q/k RMSNorm of the dense Qwen3 family -- ONE `[D]` vector shared by every
        # head (`dense_layer_weights_type`), not OLMoE's all-heads `[NH*D]` / `[KVH*D]` norm.  Same field names, so every reader is unchanged.
        fields += [("q_norm", Array(C["D"], BF16)), ("k_norm", Array(C["D"], BF16))]
    elif C.get("QK_NORM"):
        # OLMoE's norm spans ALL heads (`RMSNorm(total_num_heads * head_dim)`): a TP rank all-gathers q / k first and holds the whole
        # weight, replicated -- C's NH / KVH are the rank's heads, so the width is theirs times the world
        W = int(C.get("WORLD", 1))
        fields += [("q_norm", Array(C["NH"] * C["D"] * W, BF16)), ("k_norm", Array(C["KVH"] * C["D"] * W, BF16))]
    fields += [("o_w", Array(H, Array(C["NH"] * C["D"], BF16))), ("ln2", Array(H, BF16)),
               ("gate_w", Array(E, Array(H, BF16))),
               ("w13", Array(E, Array(2 * I, Array(H, BF16)))),
               ("w2", Array(E, Array(H, Array(I, BF16))))]
    G = int(C.get("FP8_BLOCK") or 0)
    return Record(*(_fp8_fields(fields, G, fp8_moe_block(G, I, H)) if G else fields))


def fp8_moe_block(G: int, I_r: int, H: int) -> int:
    """The block of a rank's FP8 expert scales, vLLM d9105ea80 `refine_fp8_moe_block_shape` (fused_moe/oracle/fp8.py:272): the
    checkpoint's `G` when the rank's expert width `I_r` is a multiple of it, else `g = gcd(G, I_r, H)` when g >= 32 -- the loader then
    copies each checkpoint scale onto (G/g)^2 cells of a g-block grid, losslessly.  Any other width vLLM refuses
    (`Fp8MoEMethod.create_weights`: "not divisible by weight quantization block_n"), so this raises."""
    if I_r % G == 0:
        return G
    g = math.gcd(G, I_r, H)
    if g < 32:
        raise ValueError(f"FP8 MoE: the rank's expert width {I_r} is not a multiple of the block {G} and gcd({G}, {I_r}, H={H}) = {g} < 32: "
                         f"vLLM refines no scale grid for it and refuses the checkpoint")
    return g


def _fp8_fields(fields: list[tuple[str, Any]], G: int, g: int) -> list[tuple[str, Any]]:
    """A layer's fields under a block-FP8 checkpoint: each linear of `FP8_LINEAR_FIELDS` is e4m3 `Value<8>` followed by its f32 block
    scales on the checkpoint's G-block grid; a fused-MoE layer's expert stacks `w13`, `w2` likewise, their scales `w13_ws [E][2·cdiv(I,
    g)][cdiv(H, g)]` and `w2_ws [E][cdiv(H, g)][cdiv(I, g)]` on the rank's g-block grid (`fp8_moe_block`): the shapes vLLM's
    `Fp8MoEMethod.create_weights` gives `w13_weight_scale_inv` / `w2_weight_scale_inv`.  The router `gate_w` and the norms stay bf16
    (`quant_refusal` admits the router only in `modules_to_not_convert`)."""
    from verity_vllm.program.registry.fp8 import E4M3, weight_blocks
    from verity_vllm.program.registry.prims import F32

    def cdiv(a: int, b: int) -> int:
        return -(-a // b)

    scale_of = dict(FP8_LINEAR_FIELDS)
    out: list[tuple[str, Any]] = []
    for n, t in fields:
        if n in scale_of:
            N, K = t.n, t.elem.n
            if K % G:
                raise ValueError(f"{n}: [{N}][{K}]: K is not a multiple of the FP8 block {G}")
            out += [(n, Array(N, Array(K, E4M3))), (scale_of[n], Array(weight_blocks(N, G), Array(K // G, F32)))]
        elif isinstance(t, Array) and isinstance(t.elem, Array) and isinstance(t.elem.elem, Array):     # an expert stack [E][N][K]
            # g divides I (fp8_moe_block), so w13's cdiv(2I, g) rows are vLLM's 2·cdiv(I, g)
            E, N, K = t.n, t.elem.n, t.elem.elem.n
            out += [(n, Array(E, Array(N, Array(K, E4M3)))), (f"{n}_ws", Array(E, Array(cdiv(N, g), Array(cdiv(K, g), F32))))]
        else:
            out.append((n, t))
    return out


def layernorm_layer_weights_type(C: dict[str, Any]) -> Record:
    """[SHARED-CODE lane cov-ln, GPT-NeoX / Pythia] the per-layer Record of a LayerNorm-with-bias decoder layer whose every
    dense projection carries a bias (vLLM d9105ea80 `GPTNeoXLayer`): `input_layernorm.{weight,bias}` -> `ln1`, `ln1_b`;
    `attention.query_key_value.{weight,bias}` -> `qkv_w` [QKV][H], `qkv_b` [QKV]; `attention.dense.{weight,bias}` -> `o_w`
    [H][NH*D], `o_b` [H]; `post_attention_layernorm.{weight,bias}` -> `ln2`, `ln2_b`; `mlp.dense_h_to_4h.{weight,bias}` ->
    `up_w` [I][H], `up_b` [I] (no gate: the MLP is `down(gelu(up(x)))`); `mlp.dense_4h_to_h.{weight,bias}` -> `down_w`
    [H][I], `down_b` [H].  The base field names (`ln1`, `qkv_w`, `qkv_b`, `o_w`, `ln2`, `down_w`) keep b1's meaning; the
    bias / `up_w` fields are the family's additions.  Selected by `C["NORM"] == "layernorm"` (a hand fact of the lane's
    profile: `nn.LayerNorm` with bias, `use_parallel_residual`); RMSNorm families are unchanged."""
    H, QKV, I = C["H"], C["QKV"], C["I"]
    return Record(("ln1", Array(H, BF16)), ("ln1_b", Array(H, BF16)),
                  ("qkv_w", Array(QKV, Array(H, BF16))), ("qkv_b", Array(QKV, BF16)),
                  ("o_w", Array(H, Array(C["NH"] * C["D"], BF16))), ("o_b", Array(H, BF16)),
                  ("ln2", Array(H, BF16)), ("ln2_b", Array(H, BF16)),
                  ("up_w", Array(I, Array(H, BF16))), ("up_b", Array(I, BF16)),
                  ("down_w", Array(H, Array(I, BF16))), ("down_b", Array(H, BF16)))


def rotary_dim(C: dict[str, Any]) -> int:
    """[SHARED-CODE lane cov-ln] the rotary width: `C["ROT"]` when the family rotates only part of each head (GPT-NeoX
    `rotary_pct`: Pythia rot_dim = D/4 = 16, vLLM `get_rope(head_size, rotary_dim=int(head_size * rotary_pct))`), else `D`.
    The cos/sin table vLLM builds is `[MAX_POS, ROT]` (`[cos(t·inv_freq) | sin(t·inv_freq)]`, inv_freq over `arange(0, ROT, 2) /
    ROT`), so every analytic-table function below reads this width; full-rotary configurations (no `ROT`) are unchanged."""
    return int(C.get("ROT") or C["D"])


#: [R15 fp8] the block-scaled FP8 linears of a layer (a MoE layer has the first two): base weight field -> (vLLM parameter suffix, scale field).  The scale
#: field `<w>s` is f32 `[cdiv(N, G)][K/G]` (`weight_scale_inv`, one scale per GxG weight block, the last one partial when N ends
#: mid-block; vLLM's `process_fp8_weight_block_strategy` only re-lays it out) and follows its weight field so every earlier leaf
#: interval of the layer Record is unchanged.
FP8_LINEAR_FIELDS: tuple[tuple[str, str], ...] = (("qkv_w", "qkv_ws"), ("o_w", "o_ws"), ("gate_up_w", "gate_up_ws"), ("down_w", "down_ws"))


def dense_layer_weights_type(C: dict[str, Any]) -> Record:
    """[R15 fp8] `b1.layer_weights_type(C)` with the two family facts a dense Llama-architecture layer may add:
    * `C["QK_NORM"] == "head"` (Qwen3): `q_norm`, `k_norm` bf16 `[D]` after `qkv_w`/`qkv_b` (vLLM `self_attn.q_norm.weight`,
      `self_attn.k_norm.weight`: ONE head-width vector shared by every head);
    * `C["FP8_BLOCK"] == G`: every linear's weight is e4m3 `[N][K]` (`Value<8>` words, the checkpoint's stored bytes) followed by its
      f32 block-scale field `[cdiv(N, G)][K/G]` (`FP8_LINEAR_FIELDS`).  The norms stay bf16 (vLLM keeps them unquantised).
    Configurations without these facts get exactly `b1.layer_weights_type(C)`."""
    base = b1.layer_weights_type(C)
    fields: list[tuple[str, Any]] = list(base.fields)
    if C.get("QK_NORM") == "head":
        names = [n for n, _ in fields]
        i = names.index("qkv_b") if "qkv_b" in names else names.index("qkv_w")
        fields[i + 1:i + 1] = [("q_norm", Array(C["D"], BF16)), ("k_norm", Array(C["D"], BF16))]
    G = int(C.get("FP8_BLOCK") or 0)
    return Record(*(_fp8_fields(fields, G, G) if G else fields))


def weights_type(C: dict[str, Any]) -> Record:
    """`b1.weights_type(C)` (embed, layers, norm, cos_sin) plus an `lm_head` field `[V][H]` when the checkpoint does not tie
    the output projection to the embedding.  `b1.py` is frozen (D57); tied configurations get the identical Record.
    PROPOSED (lane moe): a configuration with `NUM_EXPERTS` (a fused-MoE family: OLMoE) gets `moe_layer_weights_type(C)` as
    its `layers` element instead of the dense `b1.layer_weights_type(C)`; dense configurations are unchanged.
    [SHARED-CODE lane cov-ln] a configuration with `C["NORM"] == "layernorm"` (GPT-NeoX / Pythia) gets
    `layernorm_layer_weights_type(C)` as its `layers` element, a `norm_b` `[H]` field (the final LayerNorm's bias) after
    `norm`, and a `cos_sin` table of width `rotary_dim(C)` (partial rotary); RMSNorm configurations are unchanged.
    [R15 fp8] `C["QK_NORM"] == "head"` (Qwen3) and/or `C["FP8_BLOCK"]` (block-scaled FP8 checkpoint) get
    `dense_layer_weights_type(C)` as the `layers` element (per-head norm weights; e4m3 weights + f32 block scales); a fused-MoE one
    takes `moe_layer_weights_type(C)`, which states both itself."""
    base = b1.weights_type(C)
    fields = list(base.fields)
    if not C.get("NUM_EXPERTS") and (C.get("QK_NORM") == "head" or C.get("FP8_BLOCK")):
        fields = [(n, Array(C["NL"], dense_layer_weights_type(C))) if n == "layers" else (n, t) for n, t in fields]
    if C.get("NORM") == "layernorm":
        fields = [(n, Array(C["NL"], layernorm_layer_weights_type(C))) if n == "layers" else (n, t) for n, t in fields]
        i = next(k for k, (n, _) in enumerate(fields) if n == "norm")
        fields.insert(i + 1, ("norm_b", Array(C["H"], BF16)))
    if rotary_dim(C) != C["D"]:
        fields = [(n, Array(C["MAX_POS"], Array(rotary_dim(C), BF16))) if n == "cos_sin" else (n, t) for n, t in fields]
    if C.get("NUM_EXPERTS"):
        fields = [(n, Array(C["NL"], moe_layer_weights_type(C))) if n == "layers" else (n, t) for n, t in fields]
    extra = tuple(C.get("extra_layer_norms") or ())
    if extra:
        # [SHARED-CODE lane dense, re-created from wave-1 dense2] families with more than two norms per layer (Gemma-2:
        # pre-/post-feedforward RMSNorms) declare the extra `[H]` bf16 fields in `C["extra_layer_norms"]` (a hand field of
        # the lane's profile); appended to each layer Record after the base fields so every existing path
        # (`("layers", L, "ln1")`, ...) is unchanged.
        i = next(k for k, (n, _) in enumerate(fields) if n == "layers")
        layer = fields[i][1].elem
        fields[i] = ("layers", Array(C["NL"], Record(*layer.fields, *[(n, Array(C["H"], BF16)) for n in extra])))
    if not C.get("tie_word_embeddings", True) and "lm_head" not in dict(fields):
        fields.append(("lm_head", Array(C["V"], Array(C["H"], BF16))))   # guard: b1.weights_type may carry the untied field itself
    if C.get("embed_normalizer") is not None:
        # [ADAPTER lane cov-dense, 2026-09-17] Gemma-2 scales the embedding by `model.normalizer`, a 0-d bf16 *buffer* vLLM's
        # `Gemma2Model.__init__` registers as `torch.tensor(hidden_size ** 0.5)` at the model dtype — a root with no checkpoint
        # tensor.  It is an analytic table of one word (`normalizer_word(H)`, validated by `check_normalizer`, root class
        # `analytic-table` like `cos_sin`); declared as the LAST field so every existing leaf interval is unchanged.
        fields.append(("normalizer", Array(1, BF16)))
    return Record(*fields)


# --- the embedding normalizer (Gemma-2) ---------------------------------------------------------------------------

def normalizer_word(H: int) -> int:
    """The bf16 word of Gemma-2's `model.normalizer` buffer: `torch.tensor(hidden_size ** 0.5)` created under vLLM's
    `set_default_torch_dtype(bf16)`, i.e. the Python float `sqrt(H)` rounded ONCE (nearest-even) to bf16.  For every Gemma-2
    width (2304 / 3584 / 4608) the value is also what the f32-intermediate path gives (`test_gen_dense2.py`)."""
    x = float(H) ** 0.5
    if not (x > 0.0 and math.isfinite(x)):
        raise ValueError(f"normalizer_word: sqrt({H}) is not a positive finite number")
    m, e = math.frexp(x)                      # x = m * 2**e, 0.5 <= m < 1
    q = round(m * 256.0)                      # 8 significant bits (7 stored + implicit), Python round = nearest-even, exact here
    if q == 256:
        q, e = 128, e + 1
    exp = e - 1 + 127                         # x = (q / 128) * 2**(e-1) with q in [128, 256)
    if not 1 <= exp <= 254:
        raise ValueError(f"normalizer_word: sqrt({H}) = {x} is outside the bf16 normal range")
    return (exp << 7) | (q - 128)


def check_normalizer(C: dict[str, Any], captured: np.ndarray | list[int] | int) -> dict[str, Any]:
    """Root-policy validator of the `model.normalizer` root (class `analytic-table`): the captured bf16 word(s) of the buffer
    equal `normalizer_word(C["H"])`; the profile's hand value `C["embed_normalizer"]`, when present, must be `sqrt(H)` too.
    Exact — the buffer is one number computed from the configuration, there is no libm in the way."""
    got = np.atleast_1d(np.asarray(captured)).astype(np.uint32).reshape(-1)
    H = int(C["H"])
    exp = normalizer_word(H)
    out: dict[str, Any] = {"status": "fail", "H": H, "expected_word": exp, "expected_value": float(H) ** 0.5,
                           "captured_words": [int(w) for w in got[:4]], "n_elements": int(got.size)}
    hand = C.get("embed_normalizer")
    if hand is not None and float(hand) != float(H) ** 0.5:
        out["reason"] = f"C[embed_normalizer]={hand!r} != sqrt(H)={float(H) ** 0.5!r}"
        return out
    if got.size != 1:
        out["reason"] = f"normalizer root has {got.size} elements, expected 1"
        return out
    if int(got[0]) != exp:
        out["reason"] = f"captured word 0x{int(got[0]):04X} != bf16(sqrt({H})) = 0x{exp:04X}"
        return out
    out["status"] = "pass"
    return out


# --- the rotary table --------------------------------------------------------------------------------------------

def _inv_freq(C: dict[str, Any]) -> np.ndarray:
    D = rotary_dim(C)                       # [SHARED-CODE lane cov-ln] the rotary width (== D unless C["ROT"], partial rotary)
    base = np.float32(C["ROPE_THETA"])
    # [PROPOSED GEN dense2] vLLM builds the cache on the CUDA device (`RotaryEmbedding._compute_inv_freq`: `torch.arange(0, D, 2) /
    # D` under the model's device context), and ATen's CUDA true division by a CPU scalar is a multiplication by the fp32
    # reciprocal (BinaryDivTrueKernel.cu: `a * (1/b)`), not a correctly rounded division.  For power-of-two D the two agree
    # bit for bit (Llama 64/128, Qwen 128, Gemma-2 256); for D=96 (Phi-3-mini) they differ by one fp32 ulp at 15 of the 48
    # exponents, which `base ** e` amplifies to up to 9 ulps of inv_freq (14 of 48) and check_cos_sin's bound then rejects
    # 198 words of the captured table (positions t with t * inv_freq_k near a zero of cos / sin).
    exps = (np.arange(0, D, 2, dtype=np.float32) * (np.float32(1.0) / np.float32(D))).astype(np.float32)
    inv = (np.float32(1.0) / np.power(base, exps, dtype=np.float32)).astype(np.float32)
    rs = C.get("rope_scaling") or {}
    rtype = (rs.get("rope_type") or rs.get("type") or "").lower() if isinstance(rs, dict) else ""
    if rtype == "llama3":
        # vLLM Llama3RotaryEmbedding._compute_inv_freq (fp32): blend between inv_freq and inv_freq / factor by wavelength
        factor = np.float32(rs["factor"])
        low = np.float32(rs["low_freq_factor"])
        high = np.float32(rs["high_freq_factor"])
        orig = np.float32(rs["original_max_position_embeddings"])
        low_wl = orig / low
        high_wl = orig / high
        wave_len = (np.float32(2 * math.pi) / inv).astype(np.float32)
        if low == high:
            smooth = np.zeros_like(inv)
        else:
            smooth = ((orig / wave_len - low) / (high - low)).astype(np.float32)
        new = ((1 - smooth) * inv / factor + smooth * inv).astype(np.float32)
        inv = np.where(wave_len < high_wl, inv, np.where(wave_len > low_wl, (inv / factor).astype(np.float32), new)).astype(np.float32)
    elif rtype not in ("", "default"):
        raise NotImplementedError(f"rope_scaling type {rtype!r} has no analytic table here (write one; do not skip the gate)")
    return inv


def f32_to_bf16_words(x: np.ndarray) -> np.ndarray:
    """IEEE round-to-nearest-even of float32 values to bf16, as uint16 words (torch `.to(torch.bfloat16)`)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + bias) >> 16).astype(np.uint16)


def cos_sin_table(C: dict[str, Any], max_pos: int | None = None) -> np.ndarray:
    """uint16 bf16 words, shape [MAX_POS, ROT] (ROT = `rotary_dim(C)`, == D for full rotary): `[cos(t·inv_freq) | sin(t·inv_freq)]`
    per position, vLLM's layout."""
    D = rotary_dim(C)
    n = int(max_pos if max_pos is not None else C["MAX_POS"])
    inv = _inv_freq(C)
    t = np.arange(n, dtype=np.float32)
    freqs = (t[:, None] * inv[None, :]).astype(np.float32)          # one fp32 product per element == torch.einsum outer
    cos = np.cos(freqs, dtype=np.float32)
    sin = np.sin(freqs, dtype=np.float32)
    words = np.concatenate([f32_to_bf16_words(cos), f32_to_bf16_words(sin)], axis=1)
    assert words.shape == (n, D), words.shape
    return words


def _bf16_monotone(w: np.ndarray) -> np.ndarray:
    """bf16 words -> integers monotone in the represented value (so |Δ| is the distance in bf16 ulps)."""
    w = w.astype(np.int64)
    neg = (w & 0x8000) != 0
    return np.where(neg, -(w & 0x7FFF), w)


def bf16_words_to_f32(w: np.ndarray) -> np.ndarray:
    return (w.astype(np.uint32) << 16).view(np.float32)


#: fp32 libm slack the bound allows the producer (CUDA) relative to numpy: `powf` differences in `inv_freq` and `cosf`/`sinf`
#: result error, both in fp32 ulps of the respective quantity.  CUDA's documented maximum errors are 2 ulp for each.
POW_ULPS = 2
TRIG_ULPS = 2


def check_cos_sin(C: dict[str, Any], captured: str | Path | np.ndarray, *, max_ulp: int = COS_SIN_MAX_ULP,
                  rows: int = 4096) -> dict[str, Any]:
    """Gate I9: the captured rotary table (uint16 words [MAX_POS, D], `m1_capture.dump_cos_sin`) is the analytic function of
    `C`.  Two views are reported: the bf16-ulp distance to numpy's fp32 evaluation (`max_ulp`, histogram; informational — at
    large positions and near the zeros of cos/sin the two fp32 libms legitimately land on different bf16 words) and the
    **bound check** that decides `status`: every captured word must round from a value within the error the fp32 pipeline
    can produce — `|Δarg| ≤ t · POW_ULPS · ulp(inv_freq_k) + ulp(arg)` on the argument, `TRIG_ULPS` fp32 ulps on the result,
    plus half a bf16 ulp of rounding.  A wrong θ, D, MAX_POS, layout or scaling rule violates this bound at essentially every
    element (the report's `n_outside` then says so); a legitimate table violates it nowhere."""
    got = np.load(captured) if not isinstance(captured, np.ndarray) else captured
    out: dict[str, Any] = {"status": "fail", "max_ulp_allowed_informational": max_ulp, "shape": list(got.shape),
                           "pow_ulps": POW_ULPS, "trig_ulps": TRIG_ULPS}
    D = rotary_dim(C)                       # [SHARED-CODE lane cov-ln] table width = the rotary width (partial rotary: ROT < D)
    n = got.shape[0] if got.ndim == 2 else 0
    try:
        inv = _inv_freq(C)
    except NotImplementedError as e:
        out.update(status="error", reason=str(e))
        return out
    out["expected_shape"] = [int(C["MAX_POS"]), D]
    out["shape_ok"] = got.ndim == 2 and tuple(got.shape) == (int(C["MAX_POS"]), D) and got.dtype == np.uint16
    if not out["shape_ok"]:
        out["reason"] = f"captured {got.dtype} {got.shape} vs analytic {out['expected_shape']} for D={D} MAX_POS={C['MAX_POS']}"
        return out
    hist = np.zeros(9, dtype=np.int64)
    n_diff = n_outside = 0
    max_ulp_seen = 0
    max_excess = 0.0
    offending: list[dict] = []
    ulp_inv = np.spacing(inv).astype(np.float64)                  # fp32 spacing of each inv_freq
    for lo in range(0, n, rows):
        hi = min(n, lo + rows)
        t = np.arange(lo, hi, dtype=np.float32)
        freqs = (t[:, None] * inv[None, :]).astype(np.float32)   # the fp32 product both pipelines compute
        cos = np.cos(freqs, dtype=np.float32)
        sin = np.sin(freqs, dtype=np.float32)
        exp_words = np.concatenate([f32_to_bf16_words(cos), f32_to_bf16_words(sin)], axis=1)
        g = got[lo:hi]
        d = np.abs(_bf16_monotone(g) - _bf16_monotone(exp_words))
        hist += np.bincount(np.minimum(d, 8).reshape(-1), minlength=9)
        n_diff += int((d > 0).sum())
        max_ulp_seen = max(max_ulp_seen, int(d.max()) if d.size else 0)
        # the bound, in float64
        darg = (t.astype(np.float64)[:, None] * (POW_ULPS * ulp_inv)[None, :]) + np.spacing(freqs).astype(np.float64)
        darg = np.concatenate([darg, darg], axis=1)
        exact = np.concatenate([cos, sin], axis=1).astype(np.float64)
        slope = np.concatenate([np.abs(sin), np.abs(cos)], axis=1).astype(np.float64)   # |d cos| = |sin|·|Δarg|, |d sin| = |cos|·|Δarg|
        f32v = np.concatenate([cos, sin], axis=1)
        bound = slope * darg + 0.5 * darg ** 2 + TRIG_ULPS * np.abs(np.spacing(f32v)).astype(np.float64)
        gv = bf16_words_to_f32(g).astype(np.float64)
        half_ulp = 0.5 * np.abs(np.spacing(bf16_words_to_f32(g))).astype(np.float64) * 65536.0   # bf16 ulp = 2^16 fp32 ulps at the same exponent
        excess = np.abs(gv - exact) - (bound + half_ulp)
        bad = excess > 0
        n_outside += int(bad.sum())
        if bad.any():
            max_excess = max(max_excess, float(excess[bad].max()))
            for p, c in np.argwhere(bad)[: max(0, 8 - len(offending))]:
                offending.append({"pos": int(lo + p), "col": int(c), "got": int(g[p, c]), "expected": int(exp_words[p, c]),
                                  "ulp": int(d[p, c]), "excess": float(excess[p, c])})
    out["ulp_histogram"] = {str(i) if i < 8 else "8+": int(c) for i, c in enumerate(hist) if c}
    out["n_diff"] = n_diff
    out["n_elements"] = int(n * D)
    out["max_ulp"] = max_ulp_seen
    out["n_outside_bound"] = n_outside
    out["max_excess"] = max_excess
    out["first_offending"] = offending
    out["status"] = "pass" if n_outside == 0 else "fail"
    return out
