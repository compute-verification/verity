"""circuit-check's bindings of `verity.ml.definitions.registry.mla`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """MLA: Triton's one-split decode at one head of two keys over a 16 + 16 latent (a partial block, so its masked columns),
    which reaches the head, the block, the score and every element; the absorb bmm at two heads, K = 32, N = 3; the
    shared-expert combine at 8 elements with GLM-4.7-Flash's routed_scaling_factor."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.definitions.registry import mla

    HOP = BG.HopperBF16WgmmaDot16
    return [
        bind(mla.TritonMlaDecode, H=1, T=2, DL=16, DR=16, BN=32, SPLITS=1, SCALE=0x3D800000, DOT=HOP),
        bind(mla.MlaBmm, NHEADS=2, K=32, N=3, DOT=HOP),
        bind(mla.MoeSharedCombine, H=8, RSF=0x3FE66666),
    ]
