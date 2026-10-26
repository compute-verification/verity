"""The mHC MoE's (DeepSeek-V4's) configuration and weights Record, as `generic.build_profile` derives them for the `mhc-moe` architecture.

`C` is the family's facts over the rank's config.json (`analytic.config_of` states the Llama vocabulary, which this architecture does not
have) with the sizes `MHC_HF_KEYS` names; the weights Record is the family's `params` in their order at the shapes the model allocates
(`mhc_weight_shapes`), the Build's weights struct.  The routed experts as served are no leaf of their fields (`MARLIN_SERVED`).
"""
from __future__ import annotations

import re
from typing import Any

from verity.ml.circuits.types import Array, Record, Type, Value
from verity_vllm.engine.profiles import family_facts as FF

#: (config.json key, C key) of the sizes `mhc_config` reads off the rank's config.json; the header check compares the recorded ones
MHC_HF_KEYS = (("hidden_size", "H"), ("rms_norm_eps", "EPS"), ("num_hidden_layers", "NL"), ("num_attention_heads", "NH"),
               ("vocab_size", "V"), ("intermediate_size", "I"), ("max_position_embeddings", "MAX_POS"),
               ("tie_word_embeddings", "tie_word_embeddings"))

#: the routed experts as served: `process_weights_after_loading` repacks the Record's checkpoint-layout e2m1 codes into Marlin's tile
#: layout and permutes their scales, so no served word is a leaf of its field (`program.frontend.marlin_moe_leaf`)
MARLIN_SERVED = re.compile(r"^model\.layers\.\d+\.ffn\.experts\.routed_experts\.w(13|2)_weight(_scale)?$")


def mhc_config(cfg: dict[str, Any], rcfg: dict[str, Any], world: int) -> tuple[dict[str, Any], dict[str, str]]:
    """The mHC MoE's `C`: the family's facts over the rank-local config.json `rcfg` (heads, vocabulary and expert width divided by the
    world: `generic.rank_hf_config`), the sizes `MHC_HF_KEYS` names, the full vocabulary (the hash router's table) and the block-FP8
    group."""
    C, prov = FF.resolve_facts(rcfg, {})
    for hf, key in MHC_HF_KEYS:
        C[key], prov[key] = rcfg[hf], "hf-config"
    C["VOCAB"], prov["VOCAB"] = int(cfg["vocab_size"]), "hf-config"
    C["FP8_BLOCK"], prov["FP8_BLOCK"] = int(cfg["quantization_config"]["weight_block_size"][0]), "hf-config"
    if world > 1:
        C["WORLD"], prov["WORLD"] = int(world), "hand"
    return C, prov


def mhc_weight_shapes(C: dict[str, Any]) -> dict[str, tuple[tuple[int, ...], int]]:
    """Every parameter and buffer of one rank of the mHC MoE by name, as (shape, bit width) at the shape the model allocates: the
    head's mixes, embedding, final norm and lm_head; per layer the mHC mixes (layer 0's broadcast pre and the one rope cache too), the
    attention (an indexer at ratio 4, a compressor of the D-wide rows at every ratio, overlapping two windows at ratio 4), the router
    (its token table below HASH_LAYERS, else its correction bias), the shared expert and the MXFP4 routed experts in the checkpoint's
    [E][N][K/2] e2m1 bytes and [E][N][K/32] e8m0 scales (`program.frontend.marlin_moe_leaf`)."""
    from verity.ml.definitions.registry import marlin_moe as MM
    H, HC, V, D, DR, QL, NH, NHP, E, I, SI, IH, ID, OL, TOPK, G = (int(C[k]) for k in (
        "H", "HC_MULT", "V", "HEAD_DIM", "QK_ROPE", "Q_LORA", "NH", "PADDED_HEADS", "NUM_EXPERTS", "I", "SHARED_I", "INDEX_HEADS",
        "INDEX_HEAD_DIM", "O_LORA", "TOPK", "FP8_BLOCK"))
    OUT, OG = HC * (2 + HC), int(C["O_GROUPS"]) // int(C.get("WORLD", 1))

    def fp8(name: str, N: int, K: int) -> dict[str, tuple[tuple[int, ...], int]]:
        return {f"{name}.weight": ((N, K), 8), f"{name}.weight_scale_inv": ((-(-N // G), -(-K // G)), 32)}

    def compressor(prefix: str, width: int, r: int) -> dict[str, tuple[tuple[int, ...], int]]:
        cw = width * (2 if r == 4 else 1)
        return {f"{prefix}.ape": ((r, cw), 32), f"{prefix}.fused_wkv_wgate.weight": ((2 * cw, H), 16), f"{prefix}.norm.weight": ((width,), 16)}

    out = {"model.hc_head_fn": ((HC, HC * H), 32), "model.hc_head_base": ((HC,), 32), "model.hc_head_scale": ((1,), 32),
           "model.embed_tokens.weight": ((V, H), 16), "model.norm.weight": ((H,), 16), "lm_head.weight": ((V, H), 16)}
    for L, r in enumerate(max(1, int(r)) for r in C["COMPRESS_RATIOS"][:int(C["NL"])]):
        lay: dict[str, tuple[tuple[int, ...], int]] = {f"hc_{s}_{k}": v for s in ("attn", "ffn")
                                                       for k, v in (("fn", ((OUT, HC * H), 32)), ("base", ((OUT,), 32)), ("scale", ((3,), 32)))}
        if L == 0:
            lay |= {"hc_attn_fn_broadcast": ((OUT, H), 32), "attn.rotary_emb.cos_sin_cache": ((int(C["MAX_POS"]), DR), 32)}
        lay |= {"attn.attn_sink": ((NHP,), 32), "attn.q_norm.weight": ((QL,), 16), "attn.kv_norm.weight": ((D,), 16)}
        lay |= fp8("attn.fused_wqa_wkv", QL + D, H) | fp8("attn.wq_b", NH * D, QL) | fp8("attn.wo_a", OG * OL, NH * D // OG) | fp8("attn.wo_b", H, OG * OL)
        if r == 4:
            lay |= fp8("attn.indexer.wq_b", IH * ID, QL) | compressor("attn.indexer.compressor", ID, r)
            lay["attn.indexer.weights_proj.weight"] = ((IH, H), 16)
        lay |= compressor("attn.compressor", D, r)
        lay["ffn.gate.weight"] = ((E, H), 16)
        lay |= {"ffn.gate.tid2eid": ((int(C["VOCAB"]), TOPK), 32)} if L < int(C["HASH_LAYERS"]) else {"ffn.gate.e_score_correction_bias": ((E,), 32)}
        lay |= fp8("ffn.shared_experts.gate_up_proj", 2 * SI, H) | fp8("ffn.shared_experts.down_proj", H, SI)
        for k, N, K in (("w13", 2 * I, H), ("w2", H, I)):
            lay |= {f"ffn.experts.routed_experts.{k}_weight": ((E, N, K // 2), 8), f"ffn.experts.routed_experts.{k}_weight_scale": ((E, N, K // MM.GROUP), 8)}
        lay |= {"attn_norm.weight": ((H,), 16), "ffn_norm.weight": ((H,), 16)}
        out |= {f"model.layers.{L}.{n}": s for n, s in lay.items()}
    return out


def mhc_weights_type(cfg: dict[str, Any], C: dict[str, Any]) -> Record:
    """The rank's weights Record: one field per parameter and buffer, named as vLLM names it, in the family's `params` order (`top`,
    each layer's `order`, `tail`, each layer's `buffers`: the order `named_parameters` then `named_buffers` give, the Build's weights
    struct), at `mhc_weight_shapes`; a layer's name the order does not list refuses."""
    P, shapes = FF.facts_for(cfg)["params"], mhc_weight_shapes(C)
    layers = [f"model.layers.{L}." for L in range(int(C["NL"]))]
    names = [*P["top"], *(p + n for p in layers for n in P["order"]), *P["tail"], *(p + n for p in layers for n in P["buffers"])]
    if (stray := sorted(set(shapes) - set(names))):
        raise ValueError(f"the family's params do not order {stray[:4]}")

    def arr(shape: tuple[int, ...], w: int) -> Type:
        t: Type = Value(w)
        for n in reversed(shape):
            t = Array(n, t)
        return t

    return Record(*((n, arr(*shapes[n])) for n in names if n in shapes))
