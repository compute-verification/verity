"""circuit-check's bindings of `verity.ml.definitions.registry.greedy_stages`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Units of `TokenSelect_v1{V=100}`'s cut (`b1.CUTS`): its first chunk, a later one and the last of 4 lanes, and their
    combine, each small enough that its lowering is compared whole; and the combine of GLM-5.3's vocabulary (V = 154,880 in
    ten chunks), whose token's high word is 1 or 2 past 2^16 lanes."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import greedy_stages as G

    return [
        ("GreedyChunk_v1-first", lambda: bind(G.GreedyChunk, N=32, FIRST=True)),
        ("GreedyChunk_v1", lambda: bind(G.GreedyChunk, N=32, FIRST=False)),
        ("GreedyChunk_v1-last", lambda: bind(G.GreedyChunk, N=4, FIRST=False)),
        ("GreedyCombine_v1", lambda: bind(G.GreedyCombine, V=100, N=32)),
        ("GreedyCombine_v1-glm53", lambda: bind(G.GreedyCombine, V=154880, N=16384)),
    ]
