"""The collective kind of a Definition family or id (`registry.b1_tp2`, and the captured routes of `registry.tp_allreduce`), and
the cross-rank families of a world: what the Commit's cross-rank pass and the query compose read instead of spelling a family prefix."""

from __future__ import annotations

from verity.ml.definitions.registry.b1_tp2 import AllGather, AllGather2, AllReduce, AllReduce2, EmbeddingShard, all_gather, all_reduce
from verity.ml.definitions.registry.tp_allreduce import (AllReduceCustom, AllReduceMnnvl, AllReduceMultimem, AllReduceNcclTree,
                                                      AllReduceRmsNormMnnvl)


def collective_kind(family_or_id: str) -> str | None:
    """`"all_reduce"` / `"all_gather"` for a collective Definition family or id of ANY world (`AllReduce2_v1`, `AllReduce_v2{WORLD,N}`,
    `AllGather2_v1`, `AllGather_v1{WORLD,N}`, the captured routes `AllReduceMnnvl_v1`, `AllReduceMultimem_v1`, `AllReduceCustom_v1`,
    `AllReduceNcclTree_v1` and the fused `AllReduceRmsNormMnnvl_v1`), else None -- from the registry's composites, so a consumer never spells a family prefix."""
    fam = str(family_or_id or "").split("{", 1)[0]
    return _COLLECTIVE_KIND.get(fam)


def trailing_operands(family_or_id: str) -> int:
    """How many operands a collective reads after its `parts` (the rank-local partial and the received peer partials): the fused
    all-reduce + residual add + RMSNorm reads the residual and the norm weight, every other collective nothing."""
    return _TRAILING.get(str(family_or_id or "").split("{", 1)[0], 0)


_COLLECTIVE_KIND: dict[str, str] = {f"{c.name}_v{c.version}": k for c, k in ((AllReduce2, "all_reduce"), (AllReduce, "all_reduce"),
                                                                              (AllGather2, "all_gather"), (AllGather, "all_gather"),
                                                                              (AllReduceMnnvl, "all_reduce"), (AllReduceMultimem, "all_reduce"),
                                                                              (AllReduceCustom, "all_reduce"), (AllReduceNcclTree, "all_reduce"),
                                                                              (AllReduceRmsNormMnnvl, "all_reduce"))}
_TRAILING: dict[str, int] = {f"{AllReduceRmsNormMnnvl.name}_v{AllReduceRmsNormMnnvl.version}": 2}


def cross_rank_families(world: int) -> frozenset[str]:
    """The Definition families of a `world`-rank Program whose output (or, for the embedding shard, whose partial) is a CROSS-RANK
    fact -- evaluated by the Commit's cross-rank collective pass, never by the rank-local sampled replay: the received-output families the
    registry's own selectors bind at this world (`all_reduce` / `all_gather`) and `EmbeddingShard_v1`.  Derived from the composites, not typed."""
    return frozenset(f"{c.name}_v{c.version}" for c in (all_reduce(world, 1), all_gather(world, 1), EmbeddingShard))


def evaluated_cross_rank(family_or_id: str, families: frozenset[str] | set[str] | list[str]) -> bool:
    """Whether the cross-rank pass evaluates a family: one of `families` (`cross_rank_families(world)`) or any all-reduce route
    a Program binds (`AllReduceMnnvl_v1`, `AllReduceMultimem_v1`, `AllReduceCustom_v1`, `AllReduceNcclTree_v1`, the fused
    `AllReduceRmsNormMnnvl_v1`), each
    by its routed Definition.  The one rule both the rank's unreplayable sites and the record's `evaluated_by` read."""
    fam = str(family_or_id or "").split("{", 1)[0]
    return fam in families or collective_kind(fam) == "all_reduce"
