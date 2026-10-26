"""The FP8 vocabulary: which blockwise GEMM Definition a scale construction binds (`registry.fp8`'s v1 and v3 forms), the capture
mapping's call names of the FP8 serving port, and the fused-MoE config under batch invariance (`registry.fp8_moe`)."""
from __future__ import annotations

from verity.ml.circuits.defs import bind
from verity_vllm.program.registry import hopper as H
from verity_vllm.program.registry.fp8 import (
    ScaledMmFp8Block, ScaledMmFp8BlockSharedScale, ScaledMmFp8BlockSharedScaleV3, ScaledMmFp8BlockV3,
)


#: each restated GEMM by the Definition it restates (bound instead under `fp8_scale_construction = shared-per-block`)
SHARED_SCALE = {"ScaledMmFp8Block_v1": ScaledMmFp8BlockSharedScale}


#: the blockwise GEMM's Definition by `fp8_scale_construction`: the v1 forms (Hopper's step as written) and the v3 forms over a step
BLOCK_GEMMS = {None: {"per-coordinate": ScaledMmFp8Block, "shared-per-block": ScaledMmFp8BlockSharedScale},
               "DOT": {"per-coordinate": ScaledMmFp8BlockV3, "shared-per-block": ScaledMmFp8BlockSharedScaleV3}}


def block_gemm(K: int, N: int, G: int, scale: str, dot=None):
    """The blockwise FP8 GEMM bound at (K, N, G) under the scale construction `scale`: the v1 form when `dot` is None, else the v3 form
    over the e4m3 k32 step `dot`."""
    if dot is None:
        return bind(BLOCK_GEMMS[None][scale], K=int(K), N=int(N), G=int(G))
    return bind(BLOCK_GEMMS["DOT"][scale], K=int(K), N=int(N), G=int(G), DOT=dot)


#: TA1 call name -> this port's call name (the capture mapping's child selectors are written against TA1's names)
CALL_NAMES: dict[str, str] = {**H.CALL_NAMES, "LayerPre": "LayerPreFp8", "LayerPost": "LayerPostFp8", "Serve": "ServeFp8"}


#: `get_default_config` under VLLM_BATCH_INVARIANT (fused_moe.py:1308-1315) clamped by the block (fused_moe.py:844-846)
BATCH_INVARIANT_KB = 32
BATCH_INVARIANT_SWAP = False
