"""Checkpoint words for `fold_compare.WeightSource`: raw word views of safetensors shards, and the checkpoint sources of the
`mla-moe` weights Record's layer fields (`analytic.mla_weights_type`, GLM-4.7-Flash)."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from verity.ml.circuits.types import Type


def load_safetensors_words(paths: Iterable[Path]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """`correspondence.load_safetensors` for checkpoints that are not bf16: memory-mapped raw uint16 views of every BF16 / F16
    tensor across the shards plus the stored dtype per tensor (Pythia's checkpoint is F16; vLLM loads it into bf16 parameters,
    see `fold_compare._f16_words_to_bf16`).  F32 tensors get raw uint32 views (an f32 Record field such as GLM-4.7-Flash's
    `e_score_correction_bias`).  Tensors of any other dtype are listed in the dtype map but get no words view (asking for one
    raises KeyError).  Pythia's checkpoint also carries the HF-era buffers `attention.bias` (U8 causal mask),
    `attention.masked_bias` and `rotary_emb.inv_freq`, which vLLM's `load_weights` skips and no Record field names."""
    out: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    for p in paths:
        with open(p, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        base = 8 + n
        mm = np.memmap(p, dtype=np.uint8, mode="r")
        for k, v in header.items():
            if k == "__metadata__":
                continue
            dtypes[k] = v["dtype"]
            if v["dtype"] not in ("BF16", "F16", "F32"):
                continue                                    # no words view (see the docstring); the name stays known
            a, b = v["data_offsets"]
            out[k] = mm[base + a:base + b].view(np.uint32 if v["dtype"] == "F32" else np.uint16).reshape(v["shape"])
    return out, dtypes


#: checkpoint sources of `analytic.mla_weights_type`'s layer fields (`arch` mla-moe: GLM-4.7-Flash,
#: `Glm4MoeLite*`): the layers below FIRST_DENSE are `dense_layers[d]` = `model.layers.{d}`, the MoE layers `layers[i]` =
#: `model.layers.{i + FIRST_DENSE}`.  vLLM fuses `q_a_proj` | `kv_a_proj_with_mqa` into `fused_qkv_a_proj` and `gate_proj` |
#: `up_proj` into each `gate_up_proj` (rows in that order); the router gate is a bf16 checkpoint tensor loaded into an f32
#: parameter (`_MLA_WIDEN`), its `e_score_correction_bias` an F32 tensor.
_MLA_ATTN_SOURCES = {"ln1": ["input_layernorm.weight"], "ln2": ["post_attention_layernorm.weight"],
                     "qkv_a_w": ["self_attn.q_a_proj.weight", "self_attn.kv_a_proj_with_mqa.weight"],
                     "q_a_ln": ["self_attn.q_a_layernorm.weight"], "q_b_w": ["self_attn.q_b_proj.weight"],
                     "kv_a_ln": ["self_attn.kv_a_layernorm.weight"], "kv_b_w": ["self_attn.kv_b_proj.weight"],
                     "o_w": ["self_attn.o_proj.weight"]}
_MLA_DENSE_SOURCES = {"gate_up_w": ["mlp.gate_proj.weight", "mlp.up_proj.weight"], "down_w": ["mlp.down_proj.weight"]}
_MLA_MOE_SOURCES = {"gate_w": ["mlp.gate.weight"], "gate_bias": ["mlp.gate.e_score_correction_bias"],
                    "sh_gate_up_w": ["mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight"],
                    "sh_down_w": ["mlp.shared_experts.down_proj.weight"]}
_MLA_WIDEN = frozenset({"gate_w"})
MLA_LAYER_FIELD = re.compile(r"(dense_layers|layers)\[(\d+)\]\.(\w+)$")

MlaSources = tuple[int, dict[str, list[str]], dict[str, list[str]]]


def mla_moe_sources(weights_type: Type) -> MlaSources | None:
    """`(FIRST_DENSE, dense layer sources, MoE layer sources)` of an mla-moe weights Record (one with `dense_layers`), each source
    a list of `model.layers.{L}.`-relative checkpoint names concatenated along dim 0; None for every other Record."""
    try:
        dense = weights_type.fields[weights_type.index("dense_layers")][1]
        moe = weights_type.fields[weights_type.index("layers")][1].elem
    except (AttributeError, KeyError, ValueError, IndexError):
        return None
    E = moe.fields[moe.index("w13")][1].n
    experts = {"w13": [f"mlp.experts.{e}.{p}_proj.weight" for e in range(E) for p in ("gate", "up")],
               "w2": [f"mlp.experts.{e}.down_proj.weight" for e in range(E)]}
    return dense.n, {**_MLA_ATTN_SOURCES, **_MLA_DENSE_SOURCES}, {**_MLA_ATTN_SOURCES, **_MLA_MOE_SOURCES, **experts}


def mla_field_words(sources: MlaSources, name: str, tensor_words: Callable[[str, str], np.ndarray]) -> np.ndarray:
    """The words of layer field `name` (`dense_layers[d].f` / `layers[i].f`, matching `MLA_LAYER_FIELD`): its source tensors'
    words (`tensor_words(checkpoint name, field)`) concatenated, the f32 router gate widened from bf16."""
    m = MLA_LAYER_FIELD.match(name)
    fd, dense, moe = sources
    L = int(m[2]) + (0 if m[1] == "dense_layers" else fd)
    srcs = (dense if m[1] == "dense_layers" else moe)[m[3]]
    w = np.concatenate([tensor_words(f"model.layers.{L}.{t}", m[3]) for t in srcs])
    return w.astype(np.uint32) << 16 if m[3] in _MLA_WIDEN else w
