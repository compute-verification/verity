"""numpy twins of `registry.deepgemm` on whole rows, and of the load-time weight processing whose words the linears bind as derived
Inputs (`deepgemm_post_process_fp8_weight_block`, vLLM d9105ea80 fp8_utils.py:1111-1175): DeepSeek-V4's e8m0 scale upcast and
GLM-5.3's ue8m0 requantisation.  The capture (tests/properties/deepgemm_sm90_capture_gpu.py) compares the served words with these
and the word Definitions with the GPU; tests/program/test_deepgemm.py pins each twin to its Definition."""
from __future__ import annotations

import numpy as np

from verity_vllm.program.kernels import derived_rows as R
from verity_vllm.program.kernels.fp8_twins import np_e4m3_to_f32, np_f32_to_e4m3_sat
from verity_vllm.program.registry.deepgemm import (F_1EM10, F_448, F_ONE, FI_LANES, UE8M0_KEEP, UE8M0_ROUND,
                                                   fi_lane_elements)
from verity_vllm.program.registry.fp8 import FP8_BLOCK

#: the f32 ulps of `amax / 448` around a power of two where Triton's `div.full.f32` (within 2 ulps of the quotient) may cross it
REQUANT_DIV_ULPS = 2


def requant_band(sf0: np.ndarray) -> np.ndarray:
    """Where `per_block_cast_to_fp8`'s ue8m0 power may differ from the exact one this twin takes: `sf0 = amax / 448` within
    `REQUANT_DIV_ULPS` of a power of two 2^k, or above it by up to 2^(floor(log2 |k|) + 1) more ulps, where `log2f` (within an ulp
    of a result near k, whose ulp is 2^(floor(log2 |k|) - 23)) may return k itself and its ceiling stay k."""
    sf0 = np.asarray(sf0, dtype=np.uint32)
    mant = (sf0 & 0x7FFFFF).astype(np.int64)
    k = np.maximum(np.abs(((sf0 >> 23) & 0xFF).astype(np.int64) - 127), 1)
    above = REQUANT_DIV_ULPS + (np.int64(2) << np.floor(np.log2(k)).astype(np.int64))
    finite = (sf0 & 0x7F800000) != 0x7F800000
    return finite & ((mant <= above) | (mant >= (1 << 23) - REQUANT_DIV_ULPS))


def ue8m0_ceil(y: np.ndarray) -> np.ndarray:
    """`Fp8Ue8m0Ceil_v1` on f32 words: `(bits(m) + 0x7FFFFF) & 0xFF800000`, m = `fmaxf(fabsf(y), 1e-10f)` (a NaN y gives 1e-10)."""
    a = np.asarray(y, dtype=np.uint32) & np.uint32(0x7FFFFFFF)
    m = np.where((a > 0x7F800000) | (a < F_1EM10), np.uint32(F_1EM10), a)
    return (m + np.uint32(UE8M0_ROUND)) & np.uint32(UE8M0_KEEP)


def _clamped_quant(xf: np.ndarray, s: np.ndarray) -> np.ndarray:
    """`cvt.rn.satfinite.e4m3(fminf(fmaxf(xf / s, -448), 448))` per element, ``s`` broadcast over the last axis."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        q = R.as_f32(R.div_rn(R.as_u32(xf), np.broadcast_to(s[..., None], xf.shape)))
    q = np.where(np.isnan(q), np.float32(-448.0), np.maximum(q, np.float32(-448.0)))
    return np_f32_to_e4m3_sat(np.minimum(q, np.float32(448.0)).astype(np.float32))


def ue8m0_group_quant_row(x: np.ndarray, *, G: int = FP8_BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """`Fp8GroupQuantUe8m0_v1{K,G}` on one bf16 row: x (K,) bf16 words -> (q (K,) e4m3 bytes, s (K/G,) f32 words); s =
    `ue8m0_ceil(fmaxf-fold(1e-10, |x|) / 448)`."""
    x = np.asarray(x, dtype=np.uint16).reshape(-1)
    K = x.size
    if K % G:
        raise ValueError(f"Fp8GroupQuantUe8m0: K={K} is not a multiple of G={G}")
    xf = R.as_f32(R.bf16_to_f32_words(x)).reshape(K // G, G)
    ax = np.abs(xf)
    m = np.max(np.where(np.isnan(ax), np.float32(-np.inf), ax), axis=1, initial=np.float32(1e-10)).astype(np.float32)
    s = ue8m0_ceil(R.div_rn(R.as_u32(m), np.full(K // G, F_448, dtype=np.uint32)))
    return _clamped_quant(xf, s).reshape(K), s


def ue8m0_group_quant_rows(x: np.ndarray, *, G: int = FP8_BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """`ue8m0_group_quant_row` over the rows of x (M, K): (q (M, K), s (M, K/G))."""
    x = np.asarray(x, dtype=np.uint16)
    q, s = zip(*(ue8m0_group_quant_row(r, G=G) for r in x)) if len(x) else ((), ())
    return np.asarray(q, dtype=np.uint8).reshape(x.shape), np.asarray(s, dtype=np.uint32).reshape(x.shape[0], -1)


def _lt_ftz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`F32LtFtz_v1`: ftz(a) < ftz(b) as f32 values (a NaN compares false)."""
    return R.as_f32(R.ftz(a)) < R.as_f32(R.ftz(b))


def flashinfer_quant_1x128_row(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`FlashInferFp8Quant1x128_v1{K}` on one bf16 row: x (K,) bf16 words -> (q (K,) e4m3 bytes, stored scales (K/128,) f32 words),
    the lanes, the `setp.lt.ftz` tree and the FTZ arithmetic of `FlashInferFp8Quant1x128Group_v1` (registry.deepgemm's docstring)."""
    x = np.asarray(x, dtype=np.uint16).reshape(-1)
    K = x.size
    if K % FP8_BLOCK:
        raise ValueError(f"FlashInferFp8Quant1x128: K={K} is not a multiple of {FP8_BLOCK}")
    xw = R.bf16_to_f32_words(x).reshape(K // FP8_BLOCK, FP8_BLOCK)
    lanes = R.as_f32(xw & np.uint32(0x7FFFFFFF))[:, np.array([fi_lane_elements(l) for l in range(FI_LANES)])]
    v = R.as_u32(np.fmax.reduce(lanes, axis=-1, initial=np.float32(0.0)).astype(np.float32))      # fmaxf from +0: a NaN is dropped
    while v.shape[1] > 1:
        h = v.shape[1] // 2
        a, b = v[:, :h], v[:, h:2 * h]
        v = np.where(_lt_ftz(a, b), b, a)
    amax = np.ascontiguousarray(v[:, 0])
    nz = R.as_f32(R.f2fp_bf16(amax).astype(np.uint32) << 16) > 0
    with np.errstate(invalid="ignore", over="ignore"):
        scale = np.where(nz, R.mul_ftz(R.mufu_rcp_ftz(amax), np.uint32(F_448)), np.uint32(F_ONE)).astype(np.uint32)
        q = np_f32_to_e4m3_sat(R.as_f32(R.mul_ftz(np.broadcast_to(scale[:, None], xw.shape), xw)))
    return q.reshape(K), R.mufu_rcp_ftz(scale)


def flashinfer_quant_1x128_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`flashinfer_quant_1x128_row` over the rows of x (M, K)."""
    x = np.asarray(x, dtype=np.uint16)
    q, s = zip(*(flashinfer_quant_1x128_row(r) for r in x)) if len(x) else ((), ())
    return np.asarray(q, dtype=np.uint8).reshape(x.shape), np.asarray(s, dtype=np.uint32).reshape(x.shape[0], -1)


def deepgemm_linear_row(x: np.ndarray, w: np.ndarray, ws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`DeepGemmFp8Linear_v1{K,N}` on one bf16 row: (q, s, y bf16 words, y's nonfinite mask)."""
    q, s = ue8m0_group_quant_row(x)
    y, nf = R.scaled_mm_fp8_block_row(q, s, w, ws, G=FP8_BLOCK)
    return q, s, y, nf


def flashinfer_linear_row(x: np.ndarray, w: np.ndarray, ws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`FlashInferFp8Linear_v1{K,N}` on one bf16 row: (y bf16 words, nonfinite mask)."""
    q, s = flashinfer_quant_1x128_row(x)
    return R.scaled_mm_fp8_block_row(q, s, w, ws, G=FP8_BLOCK)


def scaled_mm_fp8_block_grouped_row(xq: np.ndarray, sx: np.ndarray, w: np.ndarray, ws: np.ndarray, *,
                                    G: int = FP8_BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """`ScaledMmFp8BlockGrouped_v1{H,K,N,G}` on one token: xq (H, K) e4m3, sx (H, K/G) f32 words, w (H, N, K) e4m3, ws (H, cdiv(N, G),
    K/G) f32 words -> (y (H, N) bf16 words, nonfinite mask): group h is `scaled_mm_fp8_block_row` of its own operands."""
    xq = np.asarray(xq, dtype=np.uint8)
    rows = [R.scaled_mm_fp8_block_row(xq[h], np.asarray(sx)[h], np.asarray(w)[h], np.asarray(ws)[h], G=G) for h in range(xq.shape[0])]
    return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])


def upcast_e8m0(b: np.ndarray) -> np.ndarray:
    """`_upcast_e8m0_to_fp32` (fp8_utils.py:1071-1080): the f32 word `byte << 23` (byte 0 is +0, 0xFF is +inf)."""
    return np.asarray(b, dtype=np.uint8).astype(np.uint32) << 23


def per_block_cast_to_fp8_ue8m0(x: np.ndarray, block: tuple[int, int] = (FP8_BLOCK, FP8_BLOCK)
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`per_block_cast_to_fp8(x, block, use_ue8m0=True)` (vllm/utils/deep_gemm.py:720-740, torch.compile'd) on an f32 [N, K] matrix:
    zero-padded to whole blocks, `amax = max|x|` per block (a NaN propagates) `.clamp(1e-4)`, `sf = amax / 448`, then
    `2^ceil(log2|sf|)` taken exactly, `q = e4m3(x * (1 / sf))` (`1 / sf` exact, a power of two), cropped.  Returns (q [N, K] bytes,
    sf [cdiv(N, bm), cdiv(K, bk)] f32 words, unresolved [same] bool): a block whose `amax / 448` is in `requant_band` is where the
    GPU's division and log2 decide the power, so its words are not claimed."""
    x = np.asarray(x, dtype=np.uint32)
    N, K = x.shape
    bm, bk = block
    nb, kb = -(-N // bm), -(-K // bk)
    xp = np.zeros((nb * bm, kb * bk), dtype=np.uint32)
    xp[:N, :K] = x
    xv = R.as_f32(xp).reshape(nb, bm, kb, bk)
    amax = np.abs(xv).max(axis=(1, 3))
    amax = np.where(np.isnan(amax), amax, np.maximum(amax, np.float32(1e-4))).astype(np.float32)
    sf0 = R.div_rn(R.as_u32(amax), np.full(amax.shape, F_448, dtype=np.uint32))
    nan = ((sf0 & 0x7F800000) == 0x7F800000) & ((sf0 & 0x7FFFFF) != 0)
    sf = np.where(nan, sf0, (sf0 + np.uint32(UE8M0_ROUND)) & np.uint32(UE8M0_KEEP)).astype(np.uint32)
    unresolved = ~nan & requant_band(sf0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        inv = R.div_rn(np.full(sf.shape, F_ONE, dtype=np.uint32), sf)
        q = np_f32_to_e4m3_sat(R.as_f32(R.mul_rn(R.as_u32(xv), np.broadcast_to(inv[:, None, :, None], xv.shape))))
    return q.reshape(nb * bm, kb * bk)[:N, :K], sf, unresolved


def requant_weight_ue8m0(w: np.ndarray, s: np.ndarray, block: tuple[int, int] = (FP8_BLOCK, FP8_BLOCK)
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`requant_weight_ue8m0_inplace` (fp8_utils.py:1011-1068) on one [N, K] e4m3 matrix and its f32 block scales: `w_dq = f32(w) *
    s` (each scale repeated over its block, cropped; RN), then `per_block_cast_to_fp8_ue8m0(w_dq)`: (q, sf, unresolved)."""
    w = np.asarray(w, dtype=np.uint8)
    N, K = w.shape
    s = np.asarray(s, dtype=np.uint32)
    s_exp = np.repeat(np.repeat(s, block[0], axis=0), block[1], axis=1)[:N, :K]
    with np.errstate(invalid="ignore", over="ignore"):
        w_dq = R.mul_rn(R.as_u32(np_e4m3_to_f32(w)), s_exp)
    return per_block_cast_to_fp8_ue8m0(w_dq, block)
