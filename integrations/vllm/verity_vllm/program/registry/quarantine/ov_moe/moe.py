"""Lane ov-moe's provisional MoE definitions -- PROMOTED to `verity_vllm.program.registry.moe` (R13 lane moe, MOE-01). This module re-exports them
so the ov-moe difftest adapters and the `gen_ov_moe_patterns` eager profile keep working; the ids are unchanged (same names / versions)."""
from __future__ import annotations

from verity_vllm.program.registry.moe import (  # noqa: F401
    ALL, C_252, C_HALF, C_LOG2E_HI, C_LOG2E_LO, C_LOG2E_OVER_252, C_MAGIC, C_NEG_MAGIC_BIAS, F32BitsShl23, F32FmaRm, F32GtStrict,
    F32IsFinite, F32Neg, F32Sat, F32Sub, MoeExpertCoordinate, MoeExpertCoordinateW, MoeExpertGemm, MoeExpertGemmW, MoeExpertRow,
    MoeRouterTopK, MoeSum, MoeSumCoordinate, NEG_10000, NvExpf, ONE32, RouterArgmaxStep, coordinate_major,
)
from verity_vllm.program.kernels.moe_twins import nv_expf_bits  # noqa: F401

SEMANTIC_ID = "ov-moe/MoeExpertGemm@v1"
