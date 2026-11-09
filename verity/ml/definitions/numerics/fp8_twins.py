"""CPU twins of the FP8 Definitions (`registry.fp8`) on numpy arrays: the per-token and per-tensor quantization kernels and the
e4m3 conversions, for the tests that hold them to a production run's words and the pod experiments, and the scale products a
host committer commits for `ScaledMmFp8BlockSharedScale_v1`."""
from __future__ import annotations

import numpy as np

from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.fp8 import E4M3_NAN, MIN_SCALE_BITS
from verity.ml.definitions.registry.prims import f32_of


def scale_products(x_s: np.ndarray, w_s: np.ndarray) -> np.ndarray:
    """The f32 words a host committer commits as `ScaledMmFp8BlockSharedScale_v1`'s scale products: `F32Mul_v1(x_s[kb], w_s[nb][kb])`
    for every weight block nb and k block kb, computed by the evaluator (`evaluate_batch` on `F32Mul_v1`), NaN payloads included.  `x_s` is one
    token row's K/G activation scales ([K/G], or [R, K/G] for R rows) and `w_s` the checkpoint's [N/G, K/G] block scales, both as u32
    words; returns [N/G, K/G] (or [R, N/G, K/G]) u32 words.  numpy's array multiply is `F32Mul_v1` on every product but a NaN, whose
    payload it takes from the other operand: those lanes go through the evaluator."""
    from verity.ml.circuits.evaluation import evaluate_batch
    xs = np.asarray(x_s, dtype=np.uint32)
    ws = np.asarray(w_s, dtype=np.uint32)
    if ws.ndim != 2 or xs.ndim not in (1, 2) or xs.shape[-1] != ws.shape[1]:
        raise ValueError(f"scale_products: x_s {xs.shape} and w_s {ws.shape} are not [.., K/G] and [N/G, K/G]")
    rows = xs.reshape(-1, 1, ws.shape[1])
    a, b = (np.ascontiguousarray(v).reshape(-1) for v in np.broadcast_arrays(rows, ws[None, :, :]))
    out = (a.view(np.float32) * b.view(np.float32)).view(np.uint32)      # RN-even: F32Mul_v1 on every product that is not a NaN
    nan = ((out & 0x7F800000) == 0x7F800000) & ((out & 0x7FFFFF) != 0)
    if nan.any():                                                          # a NaN's payload is the evaluator's, never numpy's operand choice
        out[nan] = evaluate_batch(P.F32Mul, a[nan], b[nan]).astype(np.uint32)
    return out.reshape(xs.shape[:-1] + ws.shape)


def np_f32_to_e4m3_sat(x: np.ndarray) -> np.ndarray:
    """Vectorised `F32ToE4m3Sat` over an f32 array (the same function, for whole tensors)."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    sign = ((u >> 31) << 7).astype(np.uint8)
    a = np.abs(x).astype(np.float64)
    out = np.zeros(x.shape, dtype=np.uint8)
    nan = np.isnan(x)
    sat = (~nan) & (a >= 464.0)
    small = (~nan) & (a < 2.0 ** -6)
    normal = (~nan) & (~sat) & (~small)
    # normal binades: e = floor(log2 a); step 2^(e-3); RNE
    e = np.zeros(x.shape, dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        e[normal] = np.floor(np.log2(a[normal])).astype(np.int64)
    step = np.exp2((e - 3).astype(np.float64))
    q = np.zeros(x.shape, dtype=np.float64)
    q[normal] = a[normal] / step[normal]
    q[small] = a[small] / 2.0 ** -9
    n = np.rint(q)  # numpy rint: round half to even, exact on these magnitudes
    # a value that rounds up to the next binade (n == 16) or to 480 (n == 15 at e == 8)
    val = np.where(normal, n * step, np.where(small, n * 2.0 ** -9, 0.0))
    val = np.where(nan, np.nan, val)
    sat2 = (~nan) & (val >= 464.0)
    zero = (~nan) & (val == 0.0)
    sub = (~nan) & (~zero) & (val < 2.0 ** -6)
    norm2 = (~nan) & (~zero) & (~sub) & (~sat2) & (~sat)
    e2 = np.zeros(x.shape, dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        e2[norm2] = np.floor(np.log2(val[norm2])).astype(np.int64)
    m3 = np.zeros(x.shape, dtype=np.int64)
    m3[norm2] = np.rint((val[norm2] / np.exp2(e2[norm2].astype(np.float64)) - 1.0) * 8).astype(np.int64)
    out[norm2] = (((e2[norm2] + 7) << 3) | m3[norm2]).astype(np.uint8)
    out[sub] = np.rint(val[sub] / 2.0 ** -9).astype(np.uint8)
    out[sat | sat2] = 0x7E
    out = (out | np.where(nan, 0, sign)).astype(np.uint8)
    out[nan] = E4M3_NAN
    return out


def np_e4m3_to_f32(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.uint8)
    sign = np.where(b & 0x80, -1.0, 1.0)
    e = ((b >> 3) & 0xF).astype(np.int64)
    m = (b & 0x7).astype(np.float64)
    val = np.where(e == 0, m * 2.0 ** -9, np.exp2((e - 7).astype(np.float64)) * (1.0 + m / 8.0))
    val = np.where((b & 0x7F) == 0x7F, np.nan, val)
    return (sign * val).astype(np.float32)


def np_bf16_to_f32(w: np.ndarray) -> np.ndarray:
    return (np.asarray(w, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def np_dynamic_per_token_quant(x_bf16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CPU twin of `_C.dynamic_per_token_scaled_fp8_quant` on bf16 rows [M, K] -> (x_q [M, K] u8, s_x [M] f32)."""
    xf = np_bf16_to_f32(x_bf16)
    absmax = np.nanmax(np.abs(xf), axis=1, initial=np.float32(0.0)).astype(np.float32)  # fmaxf drops NaN; starts at 0
    absmax = np.where(np.isnan(absmax), np.float32(0.0), absmax).astype(np.float32)
    s = np.maximum((absmax / np.float32(448.0)).astype(np.float32), np.float32(f32_of(MIN_SCALE_BITS))).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r = (xf / s[:, None]).astype(np.float32)
    r = np.fmin(r, np.float32(448.0))     # fminf(x, 448): NaN -> 448
    r = np.fmax(np.float32(-448.0), r)    # fmaxf(-448, x)
    return np_f32_to_e4m3_sat(r), s


def np_online_weight_quant(w_bf16: np.ndarray) -> tuple[np.ndarray, np.float32]:
    """CPU twin of `Fp8PerTensorOnlineLinearMethod.process_weights_after_loading` on the bf16 tensor [N, K]:
    amax over the tensor (aminmax + abs + maximum in bf16 — exact), s_w = f32(amax) / 448 (torch f32 RN),
    W_q = cvt(f32(W) * (1.0f / s_w)) (static_scaled_fp8_quant: reciprocal multiply, then clamp ±448, then cvt).  Held to the
    production run's committed words in `verity/ml/definitions/tests/test_fp8_twins.py`."""
    wf = np_bf16_to_f32(w_bf16)
    amax = np.float32(np.max(np.abs(wf)))
    s_w = np.float32(amax / np.float32(448.0))
    inv = np.float32(np.float32(1.0) / s_w)
    r = (wf * inv).astype(np.float32)
    r = np.fmax(np.float32(-448.0), np.fmin(r, np.float32(448.0)))
    return np_f32_to_e4m3_sat(r), s_w
