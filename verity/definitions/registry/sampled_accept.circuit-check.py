"""circuit-check's bindings of `verity.definitions.registry.sampled_accept`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The sampled accept at the smallest layout it takes, 17 tiles of 512 logits with the last tile 16 logits, under two
    draft tokens (the verdicts' chain and the row select have a second member), with its row test and log-sum-exp there; a
    tile whole, cut to 16 logits, and at the pin's 8192 logits (sixteen vectors a thread); the index and the residual at
    V = 16; and the accept at GLM-5.3's served statics (five draft tokens, V = 154,880, tiles of 8192), which is over
    `targets.MAX_GATES`: ``--all`` builds it, so a body that fails only at the served layout (19 tiles, the last 7,424
    logits) fails here, and checks the tiles it reaches."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import sampled_accept as A

    tile, v = 512, 16 * 512 + 16
    return [
        ("AcceptExpLane_v1", lambda: bind(A.AcceptExpLane)),
        ("AcceptUniform_v1", lambda: bind(A.AcceptUniform)),
        ("AcceptTileStats_v1", lambda: bind(A.AcceptTileStats, N=tile, TILE=tile)),
        ("AcceptTileStats_v1-short", lambda: bind(A.AcceptTileStats, N=16, TILE=tile)),
        ("AcceptTileStats_v1-pin", lambda: bind(A.AcceptTileStats, N=A.STATS_TILE, TILE=A.STATS_TILE)),
        ("AcceptRowLse_v1", lambda: bind(A.AcceptRowLse, V=v, TILE=tile)),
        ("LogitAt_v1", lambda: bind(A.LogitAt, V=16)),
        ("AcceptResidual_v1", lambda: bind(A.AcceptResidual, V=16)),
        ("SampledAcceptRow_v1", lambda: bind(A.SampledAcceptRow, V=v, TILE=tile)),
        ("SampledAccept_v1", lambda: bind(A.SampledAccept, K=2, V=v, TILE=tile)),
        ("SampledAccept_v1-glm53", lambda: bind(A.SampledAccept, K=5, V=154880, TILE=A.STATS_TILE)),
    ]
