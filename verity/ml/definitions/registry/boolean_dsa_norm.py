"""`dsa_norm`'s RMSNorms on bits: `DsaRMSNorm_v2{N, EPS}` and `TritonBlockRMSNorm_v2{N, BLOCK, WARPS, EPS}`, `dsa_norm.rms_norm` over
`boolean_dsa.bit_ops` (core's Boolean FP32 pieces: `Bf16ToF32`, `F32Fma`, `F32Mul`, `F32Add`, `DivFullScaleA`, `DivFullRcp`,
`RsqrtApprox`) and `F2fpBf16_v2`, each traced as one circuit (`boolean_dsa.flat`), so its word view is the `_v1` at the same statics."""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.types import Array
from verity.ml.definitions.registry import dsa_norm as DN
from verity.ml.definitions.registry.boolean_dsa import bit_ops, flat

BF16 = Array(16, BL.BIT)


def _rms(B, S, x, w):
    """`DsaRMSNorm_v1`'s body on bits."""
    return DN.rms_norm(bit_ops(B), S, x, w, BG.F2fpBf16, BF16)


DsaRMSNormV2 = flat("DsaRMSNorm", 2, ["N", "EPS"], lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)),
                    _rms, DN.DsaRMSNorm)


def _block_rms(B, S, x, w):
    """`TritonBlockRMSNorm_v1`'s body on bits."""
    return DN.block_rms_norm(bit_ops(B), S, x, w, BG.F2fpBf16, BF16)


TritonBlockRMSNormV2 = flat("TritonBlockRMSNorm", 2, ["N", "BLOCK", "WARPS", "EPS"],
                            lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)), _block_rms,
                            DN.TritonBlockRMSNorm)

BOOLEAN_DSA_NORM = [DsaRMSNormV2, TritonBlockRMSNormV2]
