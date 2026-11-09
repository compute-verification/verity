"""circuit-check's bindings of `verity.ml.definitions.registry.greedy_stages`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Units of `TokenSelect_v1{V=200}`'s cut (`b1.CUTS`): its first chunk, a later one and the last of 8 lanes, and their
    combine; and the combine of GLM-5.3's vocabulary (V = 154,880 in ten chunks), whose token's high word is 1 or 2 past
    2^16 lanes."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import greedy_stages as G

    return [
        ("GreedyChunk_v1-first", lambda: bind(G.GreedyChunk, N=64, FIRST=True)),
        ("GreedyChunk_v1", lambda: bind(G.GreedyChunk, N=64, FIRST=False)),
        ("GreedyChunk_v1-last", lambda: bind(G.GreedyChunk, N=8, FIRST=False)),
        ("GreedyCombine_v1", lambda: bind(G.GreedyCombine, V=200, N=64)),
        ("GreedyCombine_v1-glm53", lambda: bind(G.GreedyCombine, V=154880, N=16384)),
    ]
