"""circuit-check's bindings of `verity.definitions.registry.sampled_accept`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The exponential lane and the uniform, each lowered whole; each unit of the Call at small statics (a head in its three
    places, a tile's part of two threads, one that finishes a tile from carried thread sums, the row end at 17 tiles, a
    resample chunk in its three places) and at GLM-5.3's (154,880 logits in tiles of 8,192, five draft tokens), where each
    is held to the Flock block; and the Call at its smallest layout, 17 tiles of 512 logits with a last tile of 16, held to
    the block by its cut."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import sampled_accept as A

    return [
        ("AcceptExpLane_v1", lambda: A.AcceptExpLane),
        ("AcceptUniform_v1", lambda: A.AcceptUniform),
        ("AcceptTileHead_v1-first", lambda: bind(A.AcceptTileHead, N=16, FIRST=True, LAST=False)),
        ("AcceptTileHead_v1", lambda: bind(A.AcceptTileHead, N=16, FIRST=False, LAST=False)),
        ("AcceptTileHead_v1-last", lambda: bind(A.AcceptTileHead, N=16, FIRST=False, LAST=True)),
        ("AcceptTileSum_v1", lambda: bind(A.AcceptTileSum, N=512, AT=0, LAST=False)),
        ("AcceptTileSum_v1-last", lambda: bind(A.AcceptTileSum, N=6, AT=0, LAST=True)),
        ("AcceptTileSum_v1-carried", lambda: bind(A.AcceptTileSum, N=16, AT=512, LAST=True)),
        ("AcceptRowEnd_v1", lambda: bind(A.AcceptRowEnd, T=17, N=16, AT=0)),
        ("AcceptRowEnd_v1-carried", lambda: bind(A.AcceptRowEnd, T=17, N=16, AT=512)),
        ("AcceptResample_v1-first", lambda: bind(A.AcceptResample, K=2, N=2, FIRST=True, LAST=False)),
        ("AcceptResample_v1", lambda: bind(A.AcceptResample, K=2, N=2, FIRST=False, LAST=False)),
        ("AcceptResample_v1-last", lambda: bind(A.AcceptResample, K=2, N=2, FIRST=False, LAST=True)),
        ("AcceptTileHead_v1-glm53-first", lambda: bind(A.AcceptTileHead, N=8192, FIRST=True, LAST=False)),
        ("AcceptTileHead_v1-glm53", lambda: bind(A.AcceptTileHead, N=8192, FIRST=False, LAST=False)),
        ("AcceptTileHead_v1-glm53-last", lambda: bind(A.AcceptTileHead, N=7424, FIRST=False, LAST=True)),
        ("AcceptTileSum_v1-glm53", lambda: bind(A.AcceptTileSum, N=4096, AT=0, LAST=False)),
        ("AcceptTileSum_v1-glm53-last", lambda: bind(A.AcceptTileSum, N=4096, AT=4096, LAST=True)),
        ("AcceptRowEnd_v1-glm53", lambda: bind(A.AcceptRowEnd, T=19, N=3328, AT=4096)),
        ("AcceptResample_v1-glm53-first", lambda: bind(A.AcceptResample, K=5, N=32, FIRST=True, LAST=False)),
        ("AcceptResample_v1-glm53", lambda: bind(A.AcceptResample, K=5, N=32, FIRST=False, LAST=False)),
        ("AcceptResample_v1-glm53-last", lambda: bind(A.AcceptResample, K=5, N=32, FIRST=False, LAST=True)),
        ("SampledAccept_v1", lambda: bind(A.SampledAccept, K=2, V=16 * 512 + 16, TILE=512)),
    ]


def _by_calls_roots() -> list[Any]:
    """`SampledAccept_v1` at GLM-5.3's statics, checked by its calls: its 5,125 units are past what circuit-check builds
    whole, and each is a target of `_model_roots`."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import sampled_accept as A

    return [bind(A.SampledAccept, K=5, V=154880, TILE=A.STATS_TILE)]
