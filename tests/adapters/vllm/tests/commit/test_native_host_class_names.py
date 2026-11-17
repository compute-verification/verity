"""The native committer's vLLM layer classes where they reach a record: the step bound (`step_bound_bytes`, its `widths` buckets)
and the module classes and families the coverage declaration names (`_module_class`, `FAMILY_OF_MODULE`)."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import torch

from verity_vllm.commit.committer import native_host as NH


def _module(*mro: str, weight: int = 0, **attrs):
    """A module whose MRO holds `mro`, most derived first, the way vLLM's layer hierarchy does."""
    cls = torch.nn.Module
    for name in reversed(mro):
        cls = type(name, (cls,), {})
    m = cls()
    if weight:
        m.weight = torch.nn.Parameter(torch.zeros(weight, dtype=torch.bfloat16))
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


MODEL = SimpleNamespace(config=SimpleNamespace(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                               intermediate_size=128, num_hidden_layers=2, vocab_size=1000))
# vLLM's bases where they matter, and some classes without them (a flat MergedColumnParallelLinear is no Linear to `_module_class`)
MODULES = [
    ("model.embed_tokens", _module("VocabParallelEmbedding", weight=64, embedding_dim=64)),
    ("model.layers.0.input_layernorm", _module("RMSNorm", weight=64)),
    ("model.layers.0.self_attn.qkv_proj", _module("QKVParallelLinear", "ColumnParallelLinear", "LinearBase", weight=8,
                                                  output_size_per_partition=128)),
    ("model.layers.0.self_attn.rotary_emb", _module("RotaryEmbedding")),
    ("model.layers.0.self_attn.attn", _module("Attention")),
    ("model.layers.0.self_attn.o_proj", _module("RowParallelLinear", "LinearBase", weight=8, output_size_per_partition=64)),
    ("model.layers.0.post_attention_layernorm", _module("GemmaRMSNorm", weight=64)),
    ("model.layers.0.mlp.gate_up_proj", _module("MergedColumnParallelLinear", weight=8, output_size=256)),
    ("model.layers.0.mlp.act_fn", _module("SiluAndMul")),
    ("model.layers.0.mlp.down_proj", _module("ReplicatedLinear", weight=8)),
    ("model.layers.1.mlp.act_fn", _module("GeluAndMul")),
    ("model.layers.1.mlp.act_sparse", _module("GeluAndMulSparse")),
    ("model.layers.1.mlp.act_mul_silu", _module("MulAndSilu")),
    ("model.layers.1.mlp.act_clamp", _module("SiluAndMulWithClamp")),
    ("model.layers.1.input_layernorm", _module("LayerNorm", weight=64)),
    ("model.layers.1.self_attn.qkv_proj", _module("ColumnParallelLinear", "LinearBase", weight=8)),
    ("model.layers.1.self_attn.base", _module("LinearBase", weight=8)),
    ("model.norm", _module("RMSNorm", weight=64)),
    ("logits_processor", _module("LogitsProcessor")),
    ("sampler", _module("Sampler")),
    ("lm_head", _module("ParallelLMHead", "VocabParallelEmbedding", weight=64)),
    ("model.layers.1.mlp.experts", _module("FusedMoE")),
]


def _committer(modules):
    c = NH.NativeHostCommitter.__new__(NH.NativeHostCommitter)
    c._modules = modules
    return c


def test_the_step_bound_and_the_declared_module_classes_are_pinned():
    bound = NH.step_bound_bytes(MODEL, MODULES, {"batch": 2, "max_prompt": 16, "max_new": 4}, hidden_m1=False)
    live = _committer(MODULES)
    names = [f"{n}/0" for n, _m in MODULES] + ["runner.sampler/sampled_token_ids", "model.layers.9.unknown/out"]
    declared = {n: live._module_class(n) for n in names}
    offline = {n: _committer([])._module_class(n) for n in ("model.layers.0.input_layernorm/0", "model.layers.0.mlp.down_proj/out")}
    assert bound["widths"] == {"VocabParallelEmbedding": 264, "RMSNorm": 512, "LinearBase": 416, "RotaryEmbedding": 384, "Attention": 128,
                               "other": 2192, "SiluAndMul": 256, "LogitsProcessor": 0, "Sampler": 0}, bound["widths"]
    assert declared["model.layers.0.mlp.gate_up_proj/0"] == "MergedColumnParallelLinear" and declared["lm_head/0"] == "VocabParallelEmbedding"
    assert offline == {"model.layers.0.input_layernorm/0": "RMSNorm", "model.layers.0.mlp.down_proj/out": "down_proj"}
    blob = json.dumps({"bound": bound, "declared": declared, "offline": offline,
                       "family_of_module": NH.NativeHostCommitter.FAMILY_OF_MODULE}).encode()
    assert hashlib.sha256(blob).hexdigest() == "b2499ffd1f8c4111cb8e9db4127f597a479221dfa469660fca6502a9ce848600", (bound, declared)


def test_a_module_s_width_counts_its_first_parameter_s_element_size_and_two_bytes_where_it_has_none():
    """`verity_vllm.commit.committer.staging_bounds.step_bound_bytes` sizes a linear's, a norm's and an embedding's output by the
    element size of the module's first parameter; a module with no parameter counts two bytes an element."""
    def widths(m):
        return NH.step_bound_bytes(MODEL, [("m", m)], {"batch": 1, "max_prompt": 4, "max_new": 1}, hidden_m1=False)["widths"]

    def typed(m, **weights):
        for name, dtype in weights.items():
            setattr(m, name, torch.nn.Parameter(torch.zeros(8, dtype=dtype)))
        return m

    assert widths(_module("LinearBase", output_size_per_partition=64)) == {"LinearBase": 128}                 # no parameter
    assert widths(_module("LinearBase", weight=8, output_size_per_partition=64)) == {"LinearBase": 128}       # bf16
    assert widths(typed(_module("LinearBase", output_size_per_partition=64), weight=torch.float32)) == {"LinearBase": 256}
    assert widths(typed(_module("LinearBase", output_size_per_partition=64), weight=torch.float64)) == {"LinearBase": 512}
    # the first parameter in registration order decides, whatever the later ones are
    assert widths(typed(_module("LinearBase", output_size_per_partition=64), bias=torch.float32, weight=torch.bfloat16)) == {"LinearBase": 256}
    assert widths(typed(_module("LinearBase", output_size_per_partition=64), bias=torch.bfloat16, weight=torch.float32)) == {"LinearBase": 128}
    # a buffer is no parameter
    buffered = _module("LinearBase", output_size_per_partition=64)
    buffered.register_buffer("scale", torch.zeros(8, dtype=torch.float32))
    assert widths(buffered) == {"LinearBase": 128}
    assert widths(typed(_module("RMSNorm"), weight=torch.float32)) == {"RMSNorm": 2 * 8 * 4}
    assert widths(_module("RMSNorm")) == {"RMSNorm": 2 * 64 * 2}
    assert widths(typed(_module("VocabParallelEmbedding", embedding_dim=64), weight=torch.float32)) == {"VocabParallelEmbedding": 64 * 4 + 4}
    assert widths(_module("VocabParallelEmbedding", embedding_dim=64)) == {"VocabParallelEmbedding": 64 * 2 + 4}
