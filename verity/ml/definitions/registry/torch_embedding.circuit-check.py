"""circuit-check's bindings of `verity.ml.definitions.registry.torch_embedding`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """torch's embedding backward for one entry and feature: the sort path at the least M it runs at (one run of 3073
    tokens) and a run of 12, and the feature kernel over 40 tokens (chunks of 32 and 8); then the step's blocks: an entry's row
    on each kernel, and the tied gradient over 130 tokens and 130 features, whose transpose has a tile of each shape."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import torch_embedding as TE

    return [
        bind(TE.EmbeddingBackwardTorch, M=TE.FEATURE_MAX + 1),
        bind(TE.EmbeddingBackwardSortRun, N=12),
        bind(TE.EmbeddingBackwardFeature, M=40),
        bind(TE.EmbeddingBackwardEntry, M=40, H=3),
        bind(TE.EmbeddingBackwardEntry, M=TE.FEATURE_MAX + 1, H=2),
        bind(TE.EmbeddingBackwardTied, M=130, V=2, H=130),
    ]
