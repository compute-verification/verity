"""circuit-check's bindings of `verity.definitions.registry.greedy_stages`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Units of `TokenSelect_v1{V=100}`'s cut (`b1.CUTS`): its first chunk, a later one, and the combine, which reads the last
    chunk of 4 lanes; and a combine whose last chunk of 32 lanes sits past 2^16 lanes (V = 65,568 in chunks of 16,384), so its
    token's high word is 1.  Each is small enough that its lowering is compared whole."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import greedy_stages as G

    return [
        ("GreedyChunk_v1-first", lambda: bind(G.GreedyChunk, N=32, FIRST=True)),
        ("GreedyChunk_v1", lambda: bind(G.GreedyChunk, N=32, FIRST=False)),
        ("GreedyCombine_v1", lambda: bind(G.GreedyCombine, V=100, N=32)),
        ("GreedyCombine_v1-high", lambda: bind(G.GreedyCombine, V=65568, N=16384)),
    ]
