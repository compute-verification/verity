"""`SiluMul_v2{I}` — vLLM `_C.silu_and_mul` on one row: `SiluMul_v1`'s shape (gate = gu[:I], up = gu[I:]) over the element
`SiluMulBf16_v2` (`verity.ml.definitions.registry.act.SiluMulBf16`).

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.act.SiluMul`, bound by the
`b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports the same object so the quarantine's difftest adapter,
`ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.act import SiluMul

SEMANTIC_ID = "act/SiluMul@v2"

__all__ = ["SEMANTIC_ID", "SiluMul"]
