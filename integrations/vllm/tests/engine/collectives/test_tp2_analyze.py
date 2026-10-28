"""The torch-free helpers of bench/tp2_analyze (the collective body arithmetic and the shard / position classifiers)."""

from __future__ import annotations

import numpy as np

from tests.engine.collectives import analyze as ta


def test_bf16_round_trip_and_rne_ties():
    vals = np.array([0.0, 1.0, -2.5, 1.0 / 3.0, 65504.0, 1e-3], dtype=np.float32)
    bits = ta.f32_to_bf16_rne(vals)
    back = ta.bf16_to_f32(bits)
    # every bf16 value is exactly representable in f32, and re-rounding is idempotent
    assert np.array_equal(ta.f32_to_bf16_rne(back), bits)
    assert np.all(np.abs(back - vals) <= np.abs(vals) * 2.0 ** -8 + 1e-30)
    # 1 + 2^-8 is exactly halfway between the bf16 neighbours 1.0 and 1 + 2^-7: RNE rounds to the even mantissa (1.0)
    tie = np.array([1.0 + 2.0 ** -8], dtype=np.float32)
    assert ta.bf16_to_f32(ta.f32_to_bf16_rne(tie))[0] == 1.0
    # 1 + 3 * 2^-8 is halfway between 1 + 2^-7 and 1 + 2^-6: rounds to even = 1 + 2^-6
    tie2 = np.array([1.0 + 3.0 * 2.0 ** -8], dtype=np.float32)
    assert ta.bf16_to_f32(ta.f32_to_bf16_rne(tie2))[0] == 1.0 + 2.0 ** -6


def test_all_reduce_body_is_single_rounding_of_the_f32_sum():
    rng = np.random.default_rng(0)
    a = ta.bf16_to_f32(ta.f32_to_bf16_rne(rng.standard_normal(4096).astype(np.float32)))
    b = ta.bf16_to_f32(ta.f32_to_bf16_rne(rng.standard_normal(4096).astype(np.float32)))
    body = ta.f32_to_bf16_rne(a + b)
    # commutative for world 2 (the order of the two shards cannot matter)
    assert np.array_equal(body, ta.f32_to_bf16_rne(b + a))
    # and equals the sequential bf16 chain, which for two operands is the same single rounding
    assert np.array_equal(body, ta.f32_to_bf16_rne(ta.bf16_to_f32(ta.f32_to_bf16_rne(a)) + b))


def test_classify_shard_rules():
    w = 2
    assert ta.classify_shard((1536,), (1536,), w, "model.norm.weight", {}) == "replicated"
    assert ta.classify_shard((151936, 1536), (75968, 1536), w, "model.embed_tokens.weight", {}).startswith("dim0: contiguous vocab range")
    assert ta.classify_shard((2048, 1536), (1024, 1536), w, "model.layers.0.self_attn.qkv_proj.weight", {}).startswith("dim0: fused [q | k | v]")
    assert ta.classify_shard((2048,), (1024,), w, "model.layers.0.self_attn.qkv_proj.bias", {}).startswith("dim0: fused [q | k | v] bias")
    assert ta.classify_shard((17920, 1536), (8960, 1536), w, "model.layers.0.mlp.gate_up_proj.weight", {}).startswith("dim0: fused [gate | up]")
    assert ta.classify_shard((1536, 1536), (1536, 768), w, "model.layers.0.self_attn.o_proj.weight", {}).startswith("dim1: contiguous input-feature")
    assert ta.classify_shard((1536, 8960), (1536, 4480), w, "model.layers.0.mlp.down_proj.weight", {}).startswith("dim1:")
    assert ta.classify_shard((3, 4), (2, 4), w, "x", {}).startswith("other:")


def test_collective_positions_and_op_filter():
    assert ta.is_collective_op("vllm.all_reduce.default")
    assert ta.is_collective_op("vllm.all_gather.default")
    assert ta.is_collective_op("_C.vocab_parallel_embedding.default")
    assert not ta.is_collective_op("aten.mm.default")
    assert ta.position_of("model.layers.3.self_attn.o_proj", "vllm.all_reduce.default").startswith("layer 3: after o_proj")
    assert ta.position_of("model.layers.27.mlp.down_proj", "vllm.all_reduce.default").startswith("layer 27: after down_proj")
    assert ta.position_of("model.embed_tokens", "vllm.all_reduce.default").startswith("embedding: sum")
    assert ta.position_of("model.embed_tokens", "_C.vocab_parallel_embedding.default").startswith("embedding: fused masked lookup")
    assert ta.position_of("logits_processor", "vllm.all_gather.default").startswith("logits: gather")
