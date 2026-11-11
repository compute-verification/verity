"""circuit-check's bindings of `verity.definitions.registry.topp_standalone_words`' Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _boolean_roots() -> list[Any]:
    """The standalone top-p select over V = 16, which reaches its mask and keep word."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import topp_standalone_words as TSW

    return [bind(TSW.GumbelTopPStandaloneSelect, V=16)]


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """GLM-5.3's served select when the sampler batch is over `topp_geometry.SPLIT_MAX_BATCH` rows, at its vocabulary
    (V = 154,880).  It is over `targets.MAX_GATES`, so ``--all`` builds it without checking it: a build that fails only at the
    served V fails here."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import topp_standalone_words as TSW

    return [("GumbelTopPStandaloneSelect_v1-glm53", lambda: bind(TSW.GumbelTopPStandaloneSelect, V=154880))]
