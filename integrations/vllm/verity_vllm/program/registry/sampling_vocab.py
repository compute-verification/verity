"""The sampling family's vocabulary: which stochastic select a target's `sampler_construction` binds.

`SHARED_GREEDY` maps each restated select (`GumbelTokenSelect_v1`, `GumbelTopPTokenSelect_v1`) to the Definition of `registry.sampling`
that computes `temp == 0` once, and `select_for` picks one of the two under a construction. The Definitions are `sampling`'s; this
module names choices among them."""
from __future__ import annotations

from verity_vllm.program.registry.sampling import GumbelTokenSelectSharedGreedy, GumbelTopPTokenSelectSharedGreedy


#: each restated select by the Definition it restates (bound under `TargetProfile.sampler_construction = "shared-greedy"`, the default)
SHARED_GREEDY = {"GumbelTokenSelect_v1": GumbelTokenSelectSharedGreedy, "GumbelTopPTokenSelect_v1": GumbelTopPTokenSelectSharedGreedy}


def select_for(select, construction: str):
    """The stochastic select a `sampler_construction` binds for `select` (a `SHARED_GREEDY` key): its `SHARED_GREEDY` entry under
    `shared-greedy`, `select` itself under `greedy-check-per-stage`."""
    return SHARED_GREEDY[select.id] if construction == "shared-greedy" else select


__all__ = ["SHARED_GREEDY", "select_for"]
