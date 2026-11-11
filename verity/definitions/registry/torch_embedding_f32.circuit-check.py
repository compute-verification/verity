"""circuit-check's bindings of `verity.definitions.registry.torch_embedding_f32`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The f32 table's gradient on the feature kernel (40 tokens of a 9-word vocabulary, 3 features) and on the sort path at
    the least M it runs at (3073 tokens, 2 words, 1 feature), whose bodies reach every Definition here."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import torch_embedding_f32 as TF
    from verity.definitions.registry.torch_embedding import FEATURE_MAX

    return [
        ("EmbeddingGradF32_v1", lambda: bind(TF.EmbeddingGradF32, M=40, V=9, H=3)),
        ("EmbeddingGradF32_v1/sort", lambda: bind(TF.EmbeddingGradF32, M=FEATURE_MAX + 1, V=2, H=1)),
    ]
