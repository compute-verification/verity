"""circuit-check's bindings of `verity.definitions.registry.greedy_stages`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Units of `TokenSelect_v1{V=320}`'s cut (`b1.CUTS`), 20 chunks of 16 lanes: its first chunk, a later one, and the combine,
    which reads the last chunk; and the select whose body is the plan, on a row of 32 in 4 chunks: each small enough that its
    lowering is compared whole.  A combine with a carry is past that (a multiple of 2^16 falls inside a chunk only past
    65,536 lanes): V = 67,000 in 67 chunks of 1,000 lanes, where 2^16 falls inside chunk 65 at its lane 536
    (`greedy_stages.carries`) and the last chunk's high word is 1.  The select at GLM-5.3's statics is held to the block by
    its units (`greedy_stages.CUTS`) and its descriptor's digest is pinned (``digests``), which keeps it in ``--all``."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import greedy_stages as G

    return [
        ("GreedyChunk_v1-first", lambda: bind(G.GreedyChunk, N=16, FIRST=True)),
        ("GreedyChunk_v1", lambda: bind(G.GreedyChunk, N=16, FIRST=False)),
        ("GreedyCombine_v2", lambda: bind(G.GreedyCombine, V=320, N=16)),
        ("GreedyCombine_v2-carry", lambda: bind(G.GreedyCombine, V=67000, N=1000)),
        ("GreedySelect_v1", lambda: G.select(32, 4)),
        ("GreedySelect_v1-served", lambda: G.select(154880)),
    ]
