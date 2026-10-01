"""The e4m3 k32 steps vectorised by the primitive id a DOT static names, for `ScaledMmFp8Block_v2{K,N,G,DOT}` and its shared-scale
restatement (`registry.fp8`): Hopper's `HopperE4m3QgmmaDot32_v1` is `derived_rows.tc_dot32_e4m3`, and sm_120's
`BlackwellE4m3QmmaDot32_v1` is core's `group_sum_total_batch` on `BLACKWELL_SM120_E4M3_M16N8K32`."""
from __future__ import annotations

import numpy as np

from verity_vllm.program.kernels import derived_rows as D

_E4M3_NAN_MASK = 0x7F


def tc_dot32_e4m3_blackwell(acc: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`BlackwellE4m3QmmaDot32_v1` vectorised: acc (...,) f32 words, a/b (..., 32) e4m3 bytes -> (f32 words, nonfinite mask).  The
    lanes `tc_dot32_e4m3` flags (an e4m3 NaN byte, a non-finite accumulator) are flagged here too, so both twins decline one domain."""
    from verity.ml.kernels import group_sum_total_batch
    from verity.ml.tc.models import BLACKWELL_SM120_E4M3_M16N8K32
    acc = np.asarray(acc, dtype=np.uint32)
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    nonfin = ((a & _E4M3_NAN_MASK) == _E4M3_NAN_MASK).any(axis=-1) | ((b & _E4M3_NAN_MASK) == _E4M3_NAN_MASK).any(axis=-1) \
        | ((acc & 0x7F800000) == 0x7F800000)
    word = group_sum_total_batch(BLACKWELL_SM120_E4M3_M16N8K32, acc.reshape(-1), a.reshape(-1, 32), b.reshape(-1, 32))[0]
    return np.asarray(word, dtype=np.uint32).reshape(acc.shape), nonfin


TC_DOT32_E4M3_BY_PRIM = {"HopperE4m3QgmmaDot32_v1": D.tc_dot32_e4m3, "BlackwellE4m3QmmaDot32_v1": tc_dot32_e4m3_blackwell}

#: provenance of the v2 rows' evaluator (`rows.EVALUATOR_PROVENANCE`): the sm_120 twin against the cc 12.0 capture
SM120_TWIN_PROVENANCE = {
    "fixture": "tests/program/data/fp8_block_cc12_operands.npz",
    "fixture_sha256": "67306cfecd9de152f91abaf5a3f69410f72d503ce21d9bb05e6415d4561eb1d6",
    "captured_from": ["_C.per_token_group_fp8_quant", "_C.cutlass_scaled_mm blockwise, cc 12.0 (CutlassFp8BlockScaledMMKernel, VLLM_USE_DEEP_GEMM=0)"],
    "device": "vy-nebius-1, NVIDIA RTX PRO 6000 Blackwell Server Edition (cc 12.0), vLLM 0.28.1rc1.dev472+gd9105ea80",
    "captured_utc": "2026-09-30", "runs": ["r20260930-164401-c04d", "r20260930-164854-5f2c"],
    "tests": ["tests/program/test_fp8_block_cc12.py"],
    "definitions": "verity_vllm/program/registry/fp8.py: ScaledMmFp8Block_v2{K,N,G,DOT}, ScaledMmFp8BlockSharedScale_v2{K,N,G,DOT}",
}

__all__ = ["SM120_TWIN_PROVENANCE", "TC_DOT32_E4M3_BY_PRIM", "tc_dot32_e4m3_blackwell"]
