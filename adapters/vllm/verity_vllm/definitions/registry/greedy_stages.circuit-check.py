"""circuit-check's bindings of `verity_vllm.definitions.registry.greedy_stages`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Units of `TokenSelect_v1{V=320}`'s cut (`b1.CUTS`), 20 chunks of 16 lanes: its first chunk, a later one, and the combine,
    which reads the last chunk; a chunk a multiple of 2^16 falls inside (24 lanes at OFF = 65,520, from its lane 16:
    `greedy_stages.carry`); and the select whose body is the plan, on a row of 32 in 4 chunks: each small enough that its
    lowering is compared whole.  The select at GLM-5.3's statics is held to the block by its units (`greedy_stages.CUTS`) and
    its descriptor's digest is pinned (``digests``), which keeps it in ``--all``.  A class's two selects at a class of 4
    served ids over that small combine with two stop ids, and at GLM-5.3's (its three EOS ids, a model of 131,072 ids) at
    O = 128, each under `MAX_GATES`, so kept in ``--all`` with no pin."""
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import greedy_stages as G

    small = dict(V=320, N=16, O=4, STOP=(37, 301), MAX_MODEL_LEN=64)
    served = dict(V=154880, N=7744, O=128, STOP=(154820, 154827, 154829), MAX_MODEL_LEN=131072)
    return [
        ("GreedyCombineFirst_v1", lambda: bind(G.GreedyCombineFirst, **small)),
        ("GreedyCombineLive_v1", lambda: bind(G.GreedyCombineLive, **small)),
        ("GreedyCombineFirst_v1-served", lambda: bind(G.GreedyCombineFirst, **served)),
        ("GreedyCombineLive_v1-served", lambda: bind(G.GreedyCombineLive, **served)),
        ("GreedyChunk_v2-first", lambda: bind(G.GreedyChunk, N=16, OFF=0)),
        ("GreedyChunk_v2", lambda: bind(G.GreedyChunk, N=16, OFF=48)),
        ("GreedyChunk_v2-carry", lambda: bind(G.GreedyChunk, N=24, OFF=65520)),
        ("GreedyCombine_v2", lambda: bind(G.GreedyCombine, V=320, N=16)),
        ("GreedyCombineToken_v1", lambda: bind(G.GreedyCombineToken, V=320, N=16)),
        ("GreedySelect_v1", lambda: G.select(32, 4)),
        ("GreedySelect_v1-served", lambda: G.select(154880)),
    ]
