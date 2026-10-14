"""The replay row kernels of the sm_90 block-FP8 linears (`registry.deepgemm`), one token row each, on the numpy twins of
`deepgemm_twins`: `DeepGemmFp8Linear_v1{K,N}` (the M >= 32 branch: q, s and y), `FlashInferFp8Linear_v1{K,N}` (the M < 32 branch: y)
and `ScaledMmFp8BlockGrouped_v1{H,K,N,G}` (DeepSeek-V4's `wo_a` einsum: H x N).  A coordinate with a NaN e4m3 operand or a non-finite
accumulator is outside the GEMM twin's domain and declines the row by name, as `rows.scaled_mm_fp8_block` does.  The row kernels
register from `rows` (kernel "rows").
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from verity_vllm.program.kernels import deepgemm_twins as T
from verity_vllm.program.kernels.kernel_registry import Declined
from verity_vllm.program.registry.fp8 import FP8_BLOCK, weight_blocks

EVALUATORS = {
    "DeepGemmFp8Linear_v1": "verity_vllm.program.kernels.deepgemm_twins.deepgemm_linear_row (Fp8GroupQuantUe8m0_v1{K,128}: the fmaxf fold "
                            "from 1e-10, div.rn by 448, the ue8m0 ceiling, div.rn, cvt.rn.satfinite.e4m3; then derived_rows."
                            "scaled_mm_fp8_block_row, ScaledMmFp8BlockSharedScale_v1{K,N,128}) -> q, s, y",
    "FlashInferFp8Linear_v1": "verity_vllm.program.kernels.deepgemm_twins.flashinfer_linear_row (FlashInferFp8Quant1x128_v1{K}: the 32-lane "
                              "fmaxf, the setp.lt.ftz tree, MUFU.RCP and FTZ multiplies, cvt.rn.satfinite.e4m3; then derived_rows."
                              "scaled_mm_fp8_block_row, ScaledMmFp8BlockSharedScale_v1{K,N,128}) -> y",
    "ScaledMmFp8BlockGrouped_v1": "verity_vllm.program.kernels.deepgemm_twins.scaled_mm_fp8_block_grouped_row (per group h, derived_rows."
                                  "scaled_mm_fp8_block_row on its own operands: ScaledMmFp8BlockSharedScale_v1{K,N,G}) -> H x N y",
}


def _declines(family: str, nonfin: np.ndarray) -> None:
    if nonfin.any():
        raise Declined(f"{family}: {int(nonfin.sum())} output coordinates have a NaN e4m3 operand or a non-finite accumulator "
                       "(outside the vectorised twin's domain)")


def _linear_operands(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K, N = int(statics["K"]), int(statics["N"])
    return (np.asarray(ops[0], dtype=np.uint16).reshape(K), np.asarray(ops[1], dtype=np.uint8).reshape(N, K),
            np.asarray(ops[2], dtype=np.uint32).reshape(weight_blocks(N, FP8_BLOCK), K // FP8_BLOCK))


def deepgemm_linear(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`DeepGemmFp8Linear_v1{K,N}(x, w, ws)`: ports 0 (K e4m3 words), 1 (K/128 f32 scales) and 2 (N bf16 words), in return order."""
    q, s, y, nonfin = T.deepgemm_linear_row(*_linear_operands(statics, ops))
    _declines("DeepGemmFp8Linear", nonfin)
    return {"0": np.asarray(q, dtype=np.uint8), "1": np.asarray(s, dtype=np.uint32), "2": np.asarray(y, dtype=np.uint16)}


def flashinfer_linear(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`FlashInferFp8Linear_v1{K,N}(x, w, ws)` -> y (N bf16 words)."""
    y, nonfin = T.flashinfer_linear_row(*_linear_operands(statics, ops))
    _declines("FlashInferFp8Linear", nonfin)
    return {"0": np.asarray(y, dtype=np.uint16)}


def scaled_mm_fp8_block_grouped(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`ScaledMmFp8BlockGrouped_v1{H,K,N,G}(xq, sx, w, ws)` on one token -> y (H x N bf16 words, group-major)."""
    H, K, N, G = (int(statics[k]) for k in ("H", "K", "N", "G"))
    y, nonfin = T.scaled_mm_fp8_block_grouped_row(
        np.asarray(ops[0], dtype=np.uint8).reshape(H, K), np.asarray(ops[1], dtype=np.uint32).reshape(H, K // G),
        np.asarray(ops[2], dtype=np.uint8).reshape(H, N, K), np.asarray(ops[3], dtype=np.uint32).reshape(H, weight_blocks(N, G), K // G), G=G)
    _declines("ScaledMmFp8BlockGrouped", nonfin)
    return {"0": np.asarray(y, dtype=np.uint16).reshape(H * N)}


#: family -> (row kernel, the members it returns in the Definition's return order)
ROWS: dict[str, tuple[Callable[..., dict[str, np.ndarray]], tuple[str, ...]]] = {
    "DeepGemmFp8Linear_v1": (deepgemm_linear, ("0", "1", "2")),
    "FlashInferFp8Linear_v1": (flashinfer_linear, ("0",)),
    "ScaledMmFp8BlockGrouped_v1": (scaled_mm_fp8_block_grouped, ("0",)),
}


def check_targets(family: str) -> list:
    """The small specialisations `family`'s self-check binds: two 128-groups, and a last weight block of 64 rows (N = 192)."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import deepgemm as DG
    return {
        "DeepGemmFp8Linear_v1": [bind(DG.DeepGemmFp8Linear, K=256, N=n) for n in (128, 192)],
        "FlashInferFp8Linear_v1": [bind(DG.FlashInferFp8Linear, K=256, N=n) for n in (128, 192)],
        "ScaledMmFp8BlockGrouped_v1": [bind(DG.ScaledMmFp8BlockGrouped, H=2, K=256, N=192, G=FP8_BLOCK)],
    }[family]


__all__ = ["EVALUATORS", "ROWS", "check_targets", "deepgemm_linear", "flashinfer_linear", "scaled_mm_fp8_block_grouped"]
