"""The collective kind of a Definition family or id (`registry.b1_tp2`), and the cross-rank families of a world: what the
Commit's cross-rank pass and the query compose read instead of spelling a family prefix."""

from __future__ import annotations

from verity_vllm.program.registry.b1_tp2 import AllGather, AllGather2, AllReduce, AllReduce2, EmbeddingShard, all_gather, all_reduce


def collective_kind(family_or_id: str) -> str | None:
    """[lane vllm-tp-n] `"all_reduce"` / `"all_gather"` for a collective Definition family or id of ANY world (`AllReduce2_v1`, `AllReduce_v2{WORLD,N}`,
    `AllGather2_v1`, `AllGather_v1{WORLD,N}`), else None -- from the registry's composites, so a consumer never spells a family prefix."""
    fam = str(family_or_id or "").split("{", 1)[0]
    return _COLLECTIVE_KIND.get(fam)


_COLLECTIVE_KIND: dict[str, str] = {f"{c.name}_v{c.version}": k for c, k in ((AllReduce2, "all_reduce"), (AllReduce, "all_reduce"),
                                                                              (AllGather2, "all_gather"), (AllGather, "all_gather"))}


def cross_rank_families(world: int) -> frozenset[str]:
    """[lane vllm-tp-n] The Definition families of a `world`-rank Program whose output (or, for the embedding shard, whose partial) is a CROSS-RANK
    fact -- evaluated by the Commit's cross-rank collective pass, never by the rank-local sampled replay: the received-output families the
    registry's own selectors bind at this world (`all_reduce` / `all_gather`) and `EmbeddingShard_v1`.  Derived from the composites, not typed."""
    return frozenset(f"{c.name}_v{c.version}" for c in (all_reduce(world, 1), all_gather(world, 1), EmbeddingShard))
