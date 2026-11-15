"""CPU bit-models of the Inductor-generated kernels examined in the compiled/eager divergence
diagnosis (W4c, CONTRACT §16.6). The generated Triton source IS the semantics; each model below
transcribes one kernel's arithmetic (FP32 compute, BF16 loads/stores, which intermediate
roundings exist and which were elided by fusion). numpy only.

Kernels (SmolLM2-135M, stock compiled vLLM, ``custom_ops none``, IR-op priority ``native``):

* ``triton_per_fused__to_copy_embedding_rms_norm_t_0``  embedding gather + rms_norm, ONE rounding:
  ``out = bf16((x * s) * w)``, ``s = libdevice.rsqrt(sum(x*x)/H + eps)`` (FP32).
* ``triton_per_fused_fused_add_rms_norm_t_0``  ``z = x + residual`` kept in FP32 (the graph's
  bf16 convert is elided), ``out = bf16((z * s(z)) * w)``; no residual output.

The eager counterparts (vLLM CUDA / Triton kernels, modelled by W4b in ``beyond_gemm.py``) round
the residual sum to BF16 before normalising.

The per-row scalar ``s`` is an approximate instruction on the GPU (``rsqrt.approx`` for
libdevice.rsqrt; ``sqrt.approx`` + division in vLLM's Triton kernel), so the models take ``s`` as
``s0 + k ulp`` around the correctly rounded IEEE value (the ``k`` argument). ``k == 0`` means the
kernel's scalar is the IEEE one.
"""

from __future__ import annotations

import numpy as np


def bf16_to_f32(w: np.ndarray) -> np.ndarray:
    return (np.asarray(w, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """RNE FP32 -> BF16 words (NaN handling as in W4b's model)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    lsb = (u >> 16) & 1
    r = ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)
    r[nan] = 0x7FC0
    return r


def ieee_inv_rms(x_f32: np.ndarray, eps: float) -> np.ndarray:
    """Correctly rounded FP32 ``1/sqrt(mean(x^2) + eps)`` per row (exact mean in float64)."""
    x = np.asarray(x_f32, dtype=np.float64)
    mean = (x * x).sum(axis=1) / x.shape[1]
    return (1.0 / np.sqrt(mean + eps)).astype(np.float32)


def ulp_offset(s: np.ndarray, k: int) -> np.ndarray:
    return (np.asarray(s, dtype=np.float32).view(np.int32) + np.int32(k)).view(np.float32)


def rms_rows(x_f32: np.ndarray, w_bf16: np.ndarray, s: np.ndarray) -> np.ndarray:
    """``bf16((x * s) * w)`` with FP32 arithmetic, one rounding (all four kernels' epilogue)."""
    y = (np.asarray(x_f32, dtype=np.float32) * s[:, None]).astype(np.float32)
    y = (y * bf16_to_f32(w_bf16)[None, :]).astype(np.float32)
    return f32_to_bf16(y)


def rms_norm_compiled(x_bf16: np.ndarray, w_bf16: np.ndarray, eps: float, k: int | np.ndarray = 0) -> np.ndarray:
    """Model of ``triton_per_fused__to_copy_embedding_rms_norm_t_0`` on already gathered rows."""
    x = bf16_to_f32(x_bf16)
    s = ieee_inv_rms(x, eps)
    s = ulp_offset(s, k) if np.isscalar(k) else (s.view(np.int32) + np.asarray(k, np.int32)).view(np.float32)
    return rms_rows(x, w_bf16, s)


def fused_add_rms_norm_compiled(x_bf16: np.ndarray, residual_bf16: np.ndarray, w_bf16: np.ndarray, eps: float, k: int = 0) -> np.ndarray:
    """Model of ``triton_per_fused_fused_add_rms_norm_t_0``: normalises the UNROUNDED FP32 sum."""
    z = (bf16_to_f32(x_bf16) + bf16_to_f32(residual_bf16)).astype(np.float32)
    return rms_rows(z, w_bf16, ulp_offset(ieee_inv_rms(z, eps), k))


def fused_add_rms_norm_eager(x_bf16: np.ndarray, residual_bf16: np.ndarray, w_bf16: np.ndarray, eps: float, k: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """W4b's model of vLLM's CUDA ``fused_add_rms_norm_kernel``: ``residual' = bf16(x + residual)``,
    normalises the ROUNDED value. Returns (normed, residual_out)."""
    r = f32_to_bf16((bf16_to_f32(x_bf16) + bf16_to_f32(residual_bf16)).astype(np.float32))
    z = bf16_to_f32(r)
    return rms_rows(z, w_bf16, ulp_offset(ieee_inv_rms(z, eps), k)), r
