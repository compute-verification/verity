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

__all__ = ["TC_DOT32_E4M3_BY_PRIM", "tc_dot32_e4m3_blackwell"]
