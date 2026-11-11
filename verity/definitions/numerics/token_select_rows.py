"""`TokenSelectChunk_v2{C}`'s row kernel over many chunks at once: the units `TokenSelect_v4` (`registry.dead_free`) splits a
logits row into, each its chunk's first maximum above -inf and that maximum's index."""
from __future__ import annotations

import numpy as np

from verity.definitions.registry.b1 import NEG_INF_BF16


def token_select_chunk_bests(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`TokenSelectChunk_v2{C}` on each row of bf16 words ``x`` (its last axis): (best, its index), the first maximum above
    -inf, a NaN never taken, and (-inf, 0) where there is none."""
    x = np.asarray(x, dtype=np.uint16)
    f = (x.astype(np.uint32) << 16).view(np.float32)
    f = np.where(np.isnan(f), -np.inf, f)
    i = np.argmax(f, axis=-1)
    none = ~(f > -np.inf).any(axis=-1)
    best = np.take_along_axis(x, i[..., None], axis=-1)[..., 0]
    return np.where(none, np.uint16(NEG_INF_BF16), best).astype(np.uint16), np.where(none, 0, i)
