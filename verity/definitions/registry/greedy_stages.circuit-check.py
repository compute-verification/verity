"""circuit-check's bindings of `verity.definitions.registry.greedy_stages`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Units of `TokenSelect_v1{V=640}`'s cut (`b1.CUTS`), 20 chunks of 32 lanes: its first chunk, a later one, and the combine,
    which reads the last chunk; a combine with a carry (V = 67,000 in 67 chunks of 1,000 lanes: 2^16 falls inside chunk 65 at
    its lane 536, `greedy_stages.carries`, and the last chunk's high word is 1); and the select whose body is the plan, on a
    row of 640 and at GLM-5.3's statics.  The served select is past whole lowering: its descriptor's digest is pinned
    (``digests``), which keeps it in ``--all``; each other is small enough that its lowering is compared whole."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import greedy_stages as G

    return [
        ("GreedyChunk_v1-first", lambda: bind(G.GreedyChunk, N=32, FIRST=True)),
        ("GreedyChunk_v1", lambda: bind(G.GreedyChunk, N=32, FIRST=False)),
        ("GreedyCombine_v2", lambda: bind(G.GreedyCombine, V=640, N=32)),
        ("GreedyCombine_v2-carry", lambda: bind(G.GreedyCombine, V=67000, N=1000)),
        ("GreedySelect_v1", lambda: G.select(640)),
        ("GreedySelect_v1-served", lambda: G.select(154880)),
    ]
