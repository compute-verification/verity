"""circuit-check's bindings of `verity.definitions.registry.topp_stages`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Units of `GumbelTopPTokenSelect_v1{V=64}`'s cut (`sampling.CUTS`) at S = 4, and the carried forms (a slice's later stats
    block, a later step block, a middle select chunk) that its slices of one block reach only at a larger V."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import topp_stages as T

    return [
        ("TopPStatsHead_v1", lambda: bind(T.TopPStatsHead, N=8, FIRST=False)),
        ("TopPStatsExp_v1", lambda: bind(T.TopPStatsExp, N=8, SPT=4, AT=0, LAST=True, FIRST=False)),
        ("TopPProbs_v1-head", lambda: bind(T.TopPProbs, V=64, S=4, N=16, HEAD=True)),
        ("TopPProbs_v1", lambda: bind(T.TopPProbs, V=64, S=4, N=16, HEAD=False)),
        ("TopPStep_v1", lambda: bind(T.TopPStep, V=64, S=4, N=16, FIRST=False, COMBINE=-1)),
        ("TopPStep_v1-combine", lambda: bind(T.TopPStep, V=64, S=4, N=16, FIRST=True, COMBINE=0)),
        ("TopPStep_v1-scalars", lambda: bind(T.TopPStep, V=64, S=4, N=16, FIRST=True, COMBINE=T.R - 1)),
        ("GumbelTopPSelect_v1", lambda: bind(T.GumbelTopPSelect, V=64, N=2, FIRST=False, LAST=False)),
        ("GumbelTopPSelect_v1-last", lambda: bind(T.GumbelTopPSelect, V=64, N=2, FIRST=False, LAST=True)),
    ]
