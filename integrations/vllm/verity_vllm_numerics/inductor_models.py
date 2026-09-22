"""CPU bit-models of the Inductor-generated kernels examined in the compiled/eager divergence
diagnosis (W4c, CONTRACT §16.6). The generated Triton source IS the semantics; each model below
transcribes one kernel's arithmetic (FP32 compute, BF16 loads/stores, which intermediate
roundings exist and which were elided by fusion). numpy only.

Kernels (SmolLM2-135M, stock compiled vLLM, ``custom_ops none``, IR-op priority ``native``):

* ``triton_per_fused__to_copy_embedding_rms_norm_t_0``  embedding gather + rms_norm, ONE rounding:
  ``out = bf16((x * s) * w)``, ``s = libdevice.rsqrt(sum(x*x)/H + eps)`` (FP32).
* ``triton_per_fused_fused_add_rms_norm_t_0``  ``z = x + residual`` kept in FP32 (the graph's
  bf16 convert is elided), ``out = bf16((z * s(z)) * w)``; no residual output.
* ``triton_per_fused_fused_add_rms_norm_t_2``  ``z = down + (o_proj + residual)`` both adds FP32,
  ``residual_out = bf16(z)``, ``out = bf16((z * s(z)) * w)``.
* ``triton_poi_fused_mul_silu_slice_t_1``  ``out = bf16((g / (1 + exp(-g))) * u)``, ONE rounding.

The eager counterparts (vLLM CUDA / Triton kernels, modelled by W4b in ``beyond_gemm.py``) round
the residual sum to BF16 before normalising and round ``silu(g)`` to BF16 before the multiply.

The per-row scalar ``s`` is an approximate instruction on the GPU (``rsqrt.approx`` for
libdevice.rsqrt; ``sqrt.approx`` + division in vLLM's Triton kernel), so the models take ``s`` as
``s0 + k ulp`` around the correctly rounded IEEE value and report the ``k`` that reproduces each
row (``fit_ulp_offsets``). ``k == 0`` for every row means the kernel's scalar is the IEEE one on
those inputs.
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


def fused_add_rms_norm_two_outputs_compiled(down_bf16: np.ndarray, o_proj_bf16: np.ndarray, residual_bf16: np.ndarray, w_bf16: np.ndarray, eps: float, k: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Model of ``triton_per_fused_fused_add_rms_norm_t_2``: two FP32 adds, residual rounded once at
    the store, normalisation from the unrounded chain. Returns (normed, residual_out)."""
    z = (bf16_to_f32(down_bf16) + (bf16_to_f32(o_proj_bf16) + bf16_to_f32(residual_bf16)).astype(np.float32)).astype(np.float32)
    return rms_rows(z, w_bf16, ulp_offset(ieee_inv_rms(z, eps), k)), f32_to_bf16(z)


def silu_mul_compiled(gate_up_bf16: np.ndarray) -> np.ndarray:
    """Model of ``triton_poi_fused_mul_silu_slice_t_1``: ``bf16(silu_f32(g) * u)``, one rounding.
    silu is evaluated exactly (float64) then rounded to FP32; on the captured operands this matched
    the kernel's libdevice.exp + division on every element."""
    H = gate_up_bf16.shape[1] // 2
    g = bf16_to_f32(gate_up_bf16[:, :H]).astype(np.float64)
    u = bf16_to_f32(gate_up_bf16[:, H:])
    silu = (g / (1.0 + np.exp(-g))).astype(np.float32)
    return f32_to_bf16((silu * u).astype(np.float32))


def silu_mul_eager(gate_up_bf16: np.ndarray) -> np.ndarray:
    """W4b's model of vLLM's CUDA ``act_and_mul_kernel<silu>``: ``bf16(float(bf16(silu(g))) * u)``."""
    H = gate_up_bf16.shape[1] // 2
    g = bf16_to_f32(gate_up_bf16[:, :H]).astype(np.float64)
    u = bf16_to_f32(gate_up_bf16[:, H:])
    silu_bf = bf16_to_f32(f32_to_bf16((g / (1.0 + np.exp(-g))).astype(np.float32)))
    return f32_to_bf16((silu_bf * u).astype(np.float32))


def fit_ulp_offsets(gpu_bf16: np.ndarray, x_f32: np.ndarray, w_bf16: np.ndarray, eps: float, kmax: int = 4) -> np.ndarray:
    """Per row, the smallest |k| such that ``rms_rows(x, w, s0 + k ulp)`` equals the GPU row; 99 if none."""
    s0 = ieee_inv_rms(x_f32, eps)
    best = np.full(gpu_bf16.shape[0], 99, np.int32)
    for k in sorted(range(-kmax, kmax + 1), key=abs):
        ok = np.all(rms_rows(x_f32, w_bf16, ulp_offset(s0, k)) == gpu_bf16, axis=1) & (best == 99)
        best[ok] = k
    return best


def first_diff(expected: np.ndarray, actual: np.ndarray) -> dict | None:
    """First differing word in row-major order as (position, coordinate) with both bit patterns."""
    e, a = expected.reshape(-1), actual.reshape(-1)
    d = np.nonzero(e != a)[0]
    if d.size == 0:
        return None
    i = int(d[0])
    row_len = int(np.prod(expected.shape[1:])) if expected.ndim > 1 else 1
    pos, coord = divmod(i, row_len)
    return {
        "n_diff": int(d.size),
        "position": int(pos),
        "coordinate": int(coord),
        "expected_bits": f"0x{int(e[i]):04x}",
        "actual_bits": f"0x{int(a[i]):04x}",
        "expected_value": float(bf16_to_f32(e[i : i + 1])[0]),
        "actual_value": float(bf16_to_f32(a[i : i + 1])[0]),
    }
