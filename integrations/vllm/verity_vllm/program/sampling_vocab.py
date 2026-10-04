"""Which sampler Definition a target binds: each stochastic select under the `TargetProfile.sampler_construction` it is bound
under.  The Definitions are `registry.sampling`'s; this is the integration's vocabulary over them (`sampling_event.CONSTRUCTION_OF`
maps the other way, from a family to its construction)."""
from __future__ import annotations

from verity_vllm.program.registry.sampling import GumbelTokenSelectSharedGreedy, GumbelTopPTokenSelectSharedGreedy


#: each restated select by the Definition it restates (bound under `TargetProfile.sampler_construction = "shared-greedy"`, the default)
SHARED_GREEDY = {"GumbelTokenSelect_v1": GumbelTokenSelectSharedGreedy, "GumbelTopPTokenSelect_v1": GumbelTopPTokenSelectSharedGreedy}


def select_for(select, construction: str):
    """The stochastic select a `sampler_construction` binds for `select` (a `SHARED_GREEDY` key): its `SHARED_GREEDY` entry under
    `shared-greedy`, `select` itself under `greedy-check-per-stage`."""
    return SHARED_GREEDY[select.id] if construction == "shared-greedy" else select


__all__ = ["SHARED_GREEDY", "select_for"]
