"""The TP2 RANK profiles the per-rank fold Match folds a rank log with -- Llama-3.2-1B (the B1/B8 workloads of record)
and Qwen2.5-1.5B (the qwen TP2 B8 workload): the rank-local shard configuration is the full configuration with heads / kv heads / intermediate /
vocab divided by the world and the derived widths recomputed; the rank's own patterns (`EmbeddingShardCuda`, `TpCollective`) come first;
the sampler sees the full vocabulary.  Torch-free (profile construction only).  The rank profiles are `profiles.generic.build_profile(role, world=2)`."""
from __future__ import annotations

import pytest

from verity_vllm.observe.fold.patterns import TokenSelectArgmax, TokenSelectGumbelTwoStage


@pytest.mark.parametrize("mod, shard, full_vocab, qkv, gu, bias", [
    ("derived_LLAMA32_1B_tp2", dict(H=2048, NL=16, NH=16, KVH=4, D=64, I=4096, V=64128), 128256, 1536, 8192, False),
    ("derived_B1_tp2", dict(H=1536, NL=28, NH=6, KVH=1, D=128, I=4480, V=75968), 151936, 1024, 8960, True),
])
def test_rank_profile_states_the_shard_configuration_and_the_rank_patterns(mod, shard, full_vocab, qkv, gu, bias):
    import importlib
    p = importlib.import_module(f"verity_vllm.observe.fold.profiles.{mod}")
    assert p.WORLD == 2
    assert {k: p.C[k] for k in shard} == shard
    assert (p.C["QKV"], p.C["GU"], bool(p.C["BIAS"])) == (qkv, gu, bias)
    names = [type(x).__name__ for x in p.PROFILE.patterns]
    assert names[:2] == ["EmbeddingShardCuda", "TpCollective"], names[:4]     # the rank's own patterns before the base ones
    assert p.PROFILE.patterns[1].world == 2
    sel = [x for x in p.PROFILE.patterns if isinstance(x, (TokenSelectArgmax, TokenSelectGumbelTwoStage))]
    assert sel and all(x.vocab == full_vocab for x in sel), "the sampler runs over the all-gathered logits (full V), not the rank's shard"
    assert "tp2-rank" in p.PROFILE_ID


def test_qwen_rank_profile_shares_the_llama_rank_facts_not_a_copy():
    import importlib
    from verity_vllm.observe.fold.patterns.collectives import EmbeddingShardCuda
    llama = importlib.import_module("verity_vllm.observe.fold.profiles.derived_LLAMA32_1B_tp2")
    qwen = importlib.import_module("verity_vllm.observe.fold.profiles.derived_B1_tp2")
    assert type(qwen.PROFILE.patterns[0]) is type(llama.PROFILE.patterns[0]) is EmbeddingShardCuda   # one generic `world > 1` branch
    assert qwen.C["BN"] == 64 and llama.C["BN"] == 128                          # head dim 128 vs 64: FA2 kBlockN differs, nothing else is model-authored
    assert qwen.C["tie_word_embeddings"] is True
