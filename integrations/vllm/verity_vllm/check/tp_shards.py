"""A TP rank's slice of a checkpoint tensor, as vLLM's parallel linears take it: an even split along one dimension, and
QKVParallelLinear's rule when there are fewer key/value heads than ranks (rank r holds key/value head r // (world // heads), shared
by world // heads ranks: `num_kv_head_replicas`)."""
from __future__ import annotations

import re
from typing import Any

import numpy as np


def shard_dim0(a: np.ndarray, world: int, rank: int) -> np.ndarray:
    return shard_dim(a, 0, world, rank)


def shard_dim(a: np.ndarray, dim: int, world: int, rank: int) -> np.ndarray:
    n = a.shape[dim]
    if n % world:
        raise ValueError(f"dim {dim} ({n}) not divisible by world {world}")
    k = n // world
    idx = [slice(None)] * a.ndim
    idx[dim] = slice(rank * k, (rank + 1) * k)
    return a[tuple(idx)]


def replicated_qkv(name: str, shape: list[int], parts: list[np.ndarray], config: dict[str, Any], world: int,
                   rank: int) -> tuple[list[np.ndarray], str] | None:
    """Rank `rank`'s q|k|v parts of a fused qkv tensor whose key/value heads are fewer than the ranks, and the shard note; None when
    the tensor isn't one (the even split applies)."""
    hkv = int(config.get("num_key_value_heads") or config.get("num_attention_heads") or 0)
    if not (world > 1 and len(parts) == 3 and re.search(r"\.(qkv_proj|wqkv)\.\w+$", name) and 0 < hkv < world and world % hkv == 0
            and shape and shape[0] == parts[0].shape[0] // world + 2 * (parts[1].shape[0] // hkv)):
        return None
    rep = world // hkv
    return ([shard_dim0(parts[0], world, rank)] + [shard_dim0(p, hkv, rank // rep) for p in parts[1:]],
            f"rank {rank}/{world}: q dim-0 slice; k|v head {rank // rep}/{hkv}, replicated on {rep} ranks")
