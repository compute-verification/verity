"""The numpy twin of `tf32_wgmma.HopperTf32WgmmaDot8_v1` over a leading row axis, on u32 (f32) words, with the k8 step's
`GroupSum` and each operand's conversion to TF32 (``"trunc"``, ``"rna"``, ``"rne"``; Hopper is RNE) as parameters."""
from __future__ import annotations

import numpy as np
from verity.circuits.kernels import tc as TC
from verity.circuits.silicon.models import GroupSum

from verity_vllm.definitions.registry import tf32_wgmma as W

U32 = np.uint32
CANDIDATE = W.HOPPER_TF32_WGMMA_K8_CANDIDATE
B_ROUNDINGS = ("trunc", "rna", "rne")


def _u(x) -> np.ndarray:
    return np.asarray(x, dtype=U32)


def _is_nan(u: np.ndarray) -> np.ndarray:
    return ((u & 0x7F800000) == 0x7F800000) & ((u & 0x7FFFFF) != 0)


def round_tf32(w, rounding: str = "rne") -> np.ndarray:
    """f32 words as the k8 step would read them: ``"trunc"`` leaves them for a low-bit discard; ``"rna"`` /
    ``"rne"`` rounded to 19 bits first (`cvt.rna.tf32.f32` / to nearest even), a NaN kept."""
    w = _u(w)
    if rounding == "trunc":
        return w
    if rounding not in B_ROUNDINGS:
        raise ValueError(f"round_tf32: one of {B_ROUNDINGS} (got {rounding!r})")
    bump = U32(0x1000) if rounding == "rna" else (U32(0xFFF) + ((w >> 13) & 1)).astype(U32)
    r = ((w.astype(np.uint64) + bump) & ~np.uint64(0x1FFF)).astype(U32)
    return np.where(_is_nan(w) | ((w & 0x7F800000) == 0x7F800000), w, r).astype(U32)


def _tf32_codes(w, rounding: str) -> np.ndarray:
    w = _u(w)
    code = round_tf32(w, rounding) >> U32(13)
    return np.where(_is_nan(w), code | U32(1), code).astype(U32)


def tf32_dot8(acc, a, b, model: GroupSum = CANDIDATE, a_round: str = "rne", b_round: str = "rne") -> np.ndarray:
    """`HopperTf32WgmmaDot8_v1` (at the candidate ``model``) on acc [R] and a, b [R, 8] f32 words."""
    out, _ = TC.group_sum_total_batch(model, _u(acc).reshape(-1), _tf32_codes(a, a_round).reshape(-1, 8),
                                      _tf32_codes(b, b_round).reshape(-1, 8))
    return np.asarray(out, dtype=U32)
