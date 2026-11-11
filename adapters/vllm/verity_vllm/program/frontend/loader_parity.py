"""Loader parity: what vLLM's loader does to a model after it loads the weights, done on the meta instantiation
(`vllm_meta.instantiate_meta`), which never loads.  Each step leaves the model with the Parameters the served model has, in the served
shapes and dtypes and under the served names, so the derived Program reads the roots the collector commits, and returns the fact it
consulted for the Build's report; `as_the_loader_does` runs them in the loader's order."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn


def retie_lm_head_as_the_loader_does(model: nn.Module, vllm_config) -> dict:
    """Loader parity for a checkpoint whose config says `tie_word_embeddings` but which SHIPS an `lm_head.weight` of its own
    (quantisation tooling keeps a redundant copy: Qwen3-4B-Instruct-2507-FP8).  vLLM's `ModelConfig.maybe_untie_word_embeddings`
    then builds the head untied (`hf_config.tie_word_embeddings = False`, `word_embeddings_untied_by_checkpoint = True`), loads both
    tensors, and `maybe_retie_word_embeddings` (model_loader/utils.py, `process_weights_after_loading`) re-ties them -- ONE shared
    Parameter -- iff `torch.equal(lm_head.weight, embed_tokens.weight)`.  A meta instantiation never loads, so without this the
    derived Program carries two Inputs (embed, lm_head) where the served model has one root: a structural mismatch at Match.

    The decision is reproduced exactly as the loader makes it, from the pinned checkpoint's bytes (a checkpoint fact, read like its
    config.json -- never assumed): the two tensors are read from the snapshot's safetensors on the CPU and compared with `torch.equal`;
    equal -> vLLM's own `_UntiedLMHead.tie` on the meta model (the head's `weight` becomes the embedding's Parameter, which
    `derive._weights_of` collapses to one leaf with an alias).  Not equal -> left untied, as the loader leaves it (and the checkpoint
    is then genuinely untied: two fields).  Returns the fact as consulted for the report; `{"applies": False}` when vLLM did not untie."""
    fact: dict[str, Any] = {"applies": False}
    mc = vllm_config.model_config
    if not bool(getattr(mc, "word_embeddings_untied_by_checkpoint", False)):
        return fact
    from vllm.model_executor.model_loader.weight_tying import _get_untied_lm_head
    untied = _get_untied_lm_head(model)
    fact.update(applies=True, config_tie_word_embeddings=True, untied_for_loading=True, head=None if untied is None else untied.name)
    if untied is None:
        fact["note"] = "no single unquantised (lm_head, embed_tokens) pair of one shape: the loader leaves the model untied"
        return fact
    import glob
    import json as _json
    import struct
    from safetensors import safe_open
    root = mc.model
    if not os.path.isdir(root):
        raise RuntimeError(f"retie_lm_head: model path {root!r} is not a local snapshot directory (HF_HUB_OFFLINE resolution failed)")
    names = {"embed": None, "lm_head": None}
    files = sorted(glob.glob(os.path.join(root, "*.safetensors")))
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            keys = set(_json.loads(fh.read(n)))
        for tag, nm in (("embed", "model.embed_tokens.weight"), ("lm_head", "lm_head.weight")):
            if nm in keys:
                names[tag] = (f, nm)
    if names["embed"] is None or names["lm_head"] is None:
        raise RuntimeError(f"retie_lm_head: checkpoint {root!r} lacks one of model.embed_tokens.weight / lm_head.weight in {files}")
    tensors = {}
    for tag, (f, nm) in names.items():
        with safe_open(f, framework="pt", device="cpu") as sf:
            tensors[tag] = sf.get_tensor(nm)
    equal = tensors["embed"].shape == tensors["lm_head"].shape and tensors["embed"].dtype == tensors["lm_head"].dtype \
        and bool(torch.equal(tensors["embed"], tensors["lm_head"]))
    fact.update(checkpoint_files={k: os.path.basename(v[0]) for k, v in names.items()},
                shapes={k: list(t.shape) for k, t in tensors.items()}, dtypes={k: str(t.dtype) for k, t in tensors.items()},
                lm_head_equals_embed=equal, retied=equal,
                rule="vllm.model_executor.model_loader.weight_tying.maybe_retie_word_embeddings: torch.equal -> tie_weights")
    del tensors
    if equal:
        untied.tie(model)
        fact["shared_parameter"] = bool(untied.lm_head.weight is untied.embed_tokens.weight)
    return fact


def mla_absorbed_weights_as_the_loader_does(model: nn.Module, vllm_config) -> dict:
    """Loader parity for MLA (DeepSeek-V3.2, GLM-5.3): `MLAAttention.process_weights_after_loading` (mla_attention.py:1150-1252) splits
    the dequantised `kv_b_proj` weight into the absorbed projections and registers them on the layer, `W_UK_T` [N, P, L] and `W_UV`
    [N, L, V] (`replace_parameter`), which the forward reads (`bmm(q_nope^T, W_UK_T)`, `bmm(attn^T, W_UV)`).  A meta instantiation
    never loads, so each MLA layer without them gets them here, empty, in the served shapes and the activation dtype: the Program then
    reads them as weights under the served names, the roots the collector commits.  A layer on the AMX or aiter bmm paths (other
    platforms' packed `W_K` / `W_V`) is listed, not given them."""
    from vllm.model_executor.layers.attention import MLAAttention
    fact: dict[str, Any] = {"layers": [], "skipped": []}
    dt = vllm_config.model_config.dtype
    for name, m in model.named_modules():
        if not isinstance(m, MLAAttention) or (hasattr(m, "W_UV") and hasattr(m, "W_UK_T")):
            continue
        if any(getattr(m, f, False) for f in ("is_amx_bmm_enabled", "is_aiter_triton_fp4_bmm_enabled", "is_aiter_triton_fp8_bmm_enabled")):
            fact["skipped"].append(name)
            continue
        w = m.kv_b_proj.weight
        N, P, L, V = m.num_heads, m.qk_nope_head_dim, m.kv_lora_rank, m.v_head_dim
        m.register_parameter("W_UK_T", nn.Parameter(torch.empty((N, P, L), dtype=dt, device=w.device), requires_grad=False))
        m.register_parameter("W_UV", nn.Parameter(torch.empty((N, L, V), dtype=dt, device=w.device), requires_grad=False))
        fact["layers"].append(name)
    if fact["layers"]:
        fact.update(dtype=str(dt).removeprefix("torch."), W_UK_T=[N, P, L], W_UV=[N, L, V],
                    rule=f"{MLAAttention.__qualname__}.process_weights_after_loading: replace_parameter(W_UV = "
                         "W_UV.transpose(0, 1), W_UK_T = W_UK.permute(1, 2, 0)) from kv_b_proj")
    return fact


def fp8_block_weights_as_the_loader_does(model: nn.Module) -> dict:
    """Loader parity for the block-FP8 linears' DeepGEMM post-process (`fp8_utils.deepgemm_post_process_fp8_weight_block`, which the
    sm_90 block-FP8 kernels' `process_weights_after_loading` runs): checkpoint UE8M0 block scales (`float8_e8m0fnu`, DeepSeek-V4) are
    upcast to the f32 words the kernels read, and a layer marked `is_bmm` (DeepSeek-V4's grouped o-projection `wo_a`, the einsum's
    B) has its weight [g*r, d] and scales [g*r/128, d/128] as [g, r, d] and [g, r/128, d/128].  A meta instantiation never loads,
    so each such layer gets them here, empty, in the served shapes and dtypes, under the served names.  A model whose block-FP8
    scales are f32 and that marks no layer `is_bmm` is unchanged."""
    fact: dict[str, Any] = {"upcast": [], "bmm": []}
    for name, m in model.named_modules():
        w, s = getattr(m, "weight", None), getattr(m, "weight_scale_inv", None)
        if not (isinstance(w, nn.Parameter) and isinstance(s, nn.Parameter) and w.dtype == torch.float8_e4m3fn):
            continue
        upcast, g = s.dtype == torch.float8_e8m0fnu, getattr(m, "bmm_batch_size", 0) if getattr(m, "is_bmm", False) else 0
        if not upcast and not g:
            continue
        wshape, sshape = tuple(w.shape), tuple(s.shape)
        if g:
            wshape, sshape = (g, wshape[0] // g, wshape[1]), (g, sshape[0] // g, sshape[1])
            m.register_parameter("weight", nn.Parameter(torch.empty(wshape, dtype=w.dtype, device=w.device), requires_grad=False))
            fact["bmm"].append(name)
        m.register_parameter("weight_scale_inv", nn.Parameter(torch.empty(sshape, dtype=torch.float32, device=s.device), requires_grad=False))
        if upcast:
            fact["upcast"].append(name)
    if fact["upcast"] or fact["bmm"]:
        fact["rule"] = ("deepgemm_post_process_fp8_weight_block: _upcast_e8m0_to_fp32(ws) for float8_e8m0fnu scales; is_bmm: "
                        "wq.view(g, r, d), ws.view(g, r // 128, d // 128)")
    return fact


def _plain_tensors(model: nn.Module) -> dict[tuple[str, str], torch.Tensor]:
    return {(mname, n): v for mname, m in model.named_modules() for n, v in vars(m).items()
            if isinstance(v, torch.Tensor) and not isinstance(v, nn.Parameter)}


def post_load_tensors_as_the_loader_does(model: nn.Module) -> dict:
    """Loader parity for the model-level post-load hook, the loader's last step (`model_loader/utils.py`
    `process_weights_after_loading`: `model.process_weights_after_loading()` after the per-layer quant finalize).  A hook may make a
    tensor from loaded weights and keep it as a plain attribute that the forward reads (DeepSeek-V4's first-layer
    `hc_attn_fn_broadcast`, its mHC `hc_attn_fn` summed over the residual streams, `finalize_mhc_broadcast_weights`).  The hook runs
    here on meta, and each plain tensor attribute it makes is registered as a Parameter under the same name: the Program reads it as a
    weight under the served name, the root the collector commits (the meta inventory walks plain tensor attributes).  A model without
    the hook is unchanged."""
    fact: dict[str, Any] = {"applies": False}
    hook = getattr(model, "process_weights_after_loading", None)
    if not callable(hook):
        return fact
    before = _plain_tensors(model)
    with torch.no_grad():
        hook()
    made = {k: v for k, v in _plain_tensors(model).items() if before.get(k) is not v}
    modules = dict(model.named_modules())
    for (mname, n), v in made.items():
        m = modules[mname]
        delattr(m, n)
        m.register_parameter(n, nn.Parameter(v, requires_grad=False))
    fact.update(applies=True, hook=f"{type(model).__qualname__}.process_weights_after_loading",
                parameters={f"{mname}.{n}" if mname else n: {"shape": list(v.shape), "dtype": str(v.dtype).removeprefix("torch.")}
                            for (mname, n), v in sorted(made.items())})
    return fact


def as_the_loader_does(model: nn.Module, vllm_config) -> dict:
    """Every loader-parity step on `model`, by the Build report's key."""
    return {"lm_head_tying": retie_lm_head_as_the_loader_does(model, vllm_config),
            "mla_absorbed_weights": mla_absorbed_weights_as_the_loader_does(model, vllm_config),
            "fp8_block_weights": fp8_block_weights_as_the_loader_does(model),
            "post_load_tensors": post_load_tensors_as_the_loader_does(model)}
