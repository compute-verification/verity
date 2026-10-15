"""Tensor parallelism with fewer key/value heads than ranks (Qwen3-235B-A22B at TP8: 64 query and 4 key/value heads): vLLM's
QKVParallelLinear gives every rank one key/value head, head r // (world // heads), shared by world // heads ranks.  The rank's HF
configuration, the weights of record and the Build predictor's rank shape follow it at worlds 2, 4 and 8; an even split is unchanged."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from verity_vllm.check import weights_of_record as WR
from verity_vllm.engine.profiles.generic import rank_hf_config

NH, KVH, D, H = 64, 4, 2, 8
CFG = {"model_type": "qwen3_moe", "hidden_size": H, "num_attention_heads": NH, "num_key_value_heads": KVH, "head_dim": D,
       "moe_intermediate_size": 16, "intermediate_size": 32, "num_experts": 4, "vocab_size": 64}


@pytest.mark.parametrize("W,heads,kv_heads", [(2, 32, 2), (4, 16, 1), (8, 8, 1)])
def test_rank_config_replicates_kv_heads_past_the_kv_head_count(W, heads, kv_heads):
    r = rank_hf_config(CFG, W)
    assert (r["num_attention_heads"], r["num_key_value_heads"], r["head_dim"]) == (heads, kv_heads, D)


def test_rank_config_refuses_a_world_the_kv_heads_neither_divide_nor_replicate_into():
    with pytest.raises(ValueError, match="num_key_value_heads=4"):
        rank_hf_config(dict(CFG, num_attention_heads=48), 6)


@dataclass
class _Ck:
    arrays: dict
    config: dict = field(default_factory=lambda: dict(CFG))

    @property
    def tensors(self):
        return {n: ("model.safetensors", "BF16", list(a.shape), 0, a.nbytes) for n, a in self.arrays.items()}

    def has(self, name):
        return name in self.arrays

    def array(self, name, rows=None):
        return self.arrays[name] if rows is None else self.arrays[name][rows[0]:rows[1]]

    def dtype(self, name):
        return "BF16"


def _qkv_ck():
    rng = np.random.default_rng(0)
    t = {f"model.layers.0.self_attn.{p}_proj.weight": rng.integers(0, 1 << 16, (n * D, H), dtype=np.uint16)
         for p, n in (("q", NH), ("k", KVH), ("v", KVH))}
    return _Ck(t), t


@pytest.mark.parametrize("W", [2, 4, 8])
def test_weights_of_record_give_each_rank_its_q_slice_and_its_kv_head(W):
    ck, t = _qkv_ck()
    q, k, v = (t[f"model.layers.0.self_attn.{p}_proj.weight"] for p in "qkv")
    nq, nkv = NH // W, max(1, KVH // W)
    for rank in range(W):
        out, note = WR.compose("model.layers.0.self_attn.qkv_proj.weight", [(nq + 2 * nkv) * D, H], 16, ck, world=W, rank=rank)
        kv = rank // (W // KVH) if W > KVH else rank
        want = np.concatenate([q[rank * nq * D:(rank + 1) * nq * D], k[kv * nkv * D:(kv + 1) * nkv * D], v[kv * nkv * D:(kv + 1) * nkv * D]])
        assert np.array_equal(out, want), (W, rank)
        assert ("replicated on" in note["shard"]) == (W > KVH)


def test_ranks_sharing_a_kv_head_hold_the_same_kv_rows():
    ck, _ = _qkv_ck()
    shard = [WR.compose("model.layers.0.self_attn.qkv_proj.weight", [(8 + 2) * D, H], 16, ck, world=8, rank=r)[0] for r in range(8)]
    for r in range(0, 8, 2):
        assert np.array_equal(shard[r][8 * D:], shard[r + 1][8 * D:]) and not np.array_equal(shard[r][:8 * D], shard[r + 1][:8 * D])
    assert not np.array_equal(shard[0][8 * D:], shard[2][8 * D:])


@pytest.mark.parametrize("W,kv", [(2, 2), (4, 1), (8, 1)])
def test_the_build_predictor_rank_shape_replicates_kv_heads(W, kv):
    from verity_vllm.predict import build as PB
    C = PB.ModelShape(arch="moe", model_class="Qwen3MoeForCausalLM", H=4096, V=151936, NL=94, NH=64, KVH=4, D=128, I=1536, EPS=1e-6,
                      tie=False, max_pos=40960, rot=128, E=128, TOPK=8)
    r = PB.rank_shape(C, W)
    assert (r.NH, r.KVH, r.I) == (64 // W, kv, 1536 // W)
