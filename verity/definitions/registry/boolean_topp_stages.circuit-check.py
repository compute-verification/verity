"""circuit-check's bindings of `verity.definitions.registry.boolean_topp_stages`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """Each stage family on bits at the statics its word family is bound at (`topp_stages`' sidecar): units of
    `GumbelTopPTokenSelect_v1{V=64}`'s cut at S = 4, and the carried forms (a slice's later stats block, a later step block, a
    middle select chunk) that its slices of one block reach only at a larger V.  All but the first are over the suite's
    gate limit (`targets.MAX_GATES`: a lane of a step block alone is), so each holds a ``digests`` pin, which keeps it in
    the suite where the limit would keep only its family's smallest.  The word gates' Boolean versions the module defines
    (`BitXor_v2`, `U32Xor_v2`, `SelectBit_v2`, `TopPNumkeep_v2`) are static-free, catalog roots already, and these bodies
    Call them."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_topp_stages as BT
    from verity.definitions.registry import topp_stages as T

    return [
        bind(BT.TopPStatsHead, N=8, FIRST=False),
        bind(BT.TopPStatsExp, N=8, SPT=4, AT=0, LAST=True, FIRST=False),
        bind(BT.TopPProbs, V=64, S=4, N=16, HEAD=True),
        bind(BT.TopPProbs, V=64, S=4, N=16, HEAD=False),
        bind(BT.TopPStep, V=64, S=4, N=16, FIRST=False, COMBINE=-1),
        bind(BT.TopPStep, V=64, S=4, N=16, FIRST=True, COMBINE=0),
        bind(BT.TopPStep, V=64, S=4, N=16, FIRST=True, COMBINE=T.R - 1),
        bind(BT.GumbelTopPSelect, V=64, N=2, FIRST=False, LAST=False),
        bind(BT.GumbelTopPSelect, V=64, N=2, FIRST=False, LAST=True),
    ]
