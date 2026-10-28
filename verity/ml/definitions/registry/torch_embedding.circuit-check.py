"""circuit-check's bindings of `verity.ml.definitions.registry.torch_embedding`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """torch's embedding backward for one entry and feature: the sort path at the least M it runs at (one run of 3073
    tokens) and a run of 12, and the feature kernel over 40 tokens (chunks of 32 and 8); then the step's blocks: an entry's row
    on each kernel, the tied gradient over 130 tokens and 130 features, whose transpose has a tile of each shape, and the
    forward over 3 tokens of a 64-word vocabulary; then version 2 of the tied gradient, the ids sorted once and routed, on the
    feature kernel (40 tokens of a 9-word vocabulary, 3 features) and on the sort path (3073 tokens, 2 words, 1 feature), and
    the run scan on each kernel and the sort path's segment scan over 12 tokens, since past its gate budget circuit-check
    checks only a family's smallest binding."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import torch_embedding as TE

    return [
        bind(TE.EmbeddingBackwardTorch, M=TE.FEATURE_MAX + 1),
        bind(TE.EmbeddingBackwardSortRun, N=12),
        bind(TE.EmbeddingBackwardFeature, M=40),
        bind(TE.EmbeddingBackwardEntry, M=40, H=3),
        bind(TE.EmbeddingBackwardEntry, M=TE.FEATURE_MAX + 1, H=2),
        bind(TE.EmbeddingBackwardTied, M=130, V=2, H=130),
        bind(TE.EmbeddingTorch, M=3, V=64, H=4),
        bind(TE.EmbeddingBackwardTiedV2, M=40, V=9, H=3),
        bind(TE.EmbeddingBackwardTiedV2, M=TE.FEATURE_MAX + 1, V=2, H=1),
        bind(TE.EmbeddingBackwardSegmentRun, N=12, V=9, NP=4, FEATURE=0),
        bind(TE.EmbeddingBackwardSortedRun, N=12, FEATURE=0),
        bind(TE.EmbeddingBackwardSortedRun, N=12, FEATURE=1),
    ]
