"""The numpy twins of `registry.dsv4_moe`'s Programs (vLLM d9105ea80's DeepSeek-V4 routers and clamped SwiGLU): `router_twin`
(`MoeRouterSqrtSoftplus_v1`, the Triton `dsv4_topk`), `hash_twin` (`MoeRouterHash_v1`, the CUDA `dsv4HashTopkSoftplusSqrt`) and
`clamp_twin` (`SiluMulClamp_v1`, `_C.silu_and_mul_with_clamp`), each on rows and word for word the Program's operations.  The
Programs are the reference: tests/properties/test_dsv4_moe_capture.py holds each twin and element function to them, and the GPU
capture tests/properties/dsv4_moe_capture_gpu.py holds the kernels to the twins on every captured row."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from verity.ml.circuits.ops import mufu as MU
from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics.act_rows import silu_mul_expf_rows
from verity.ml.definitions.numerics.libdevice import nv_logf
from verity.ml.definitions.registry import dsv4_moe as D
from verity.ml.definitions.registry import mla_check as MC

U32 = np.uint32
ONE, TWO, QNAN = 0x3F800000, 0x40000000, 0x7FFFFFFF
TOPK = D.TOPK


def _w(v) -> np.ndarray:
    return np.asarray(v, np.float32).view(U32)


def _gpu_nan(w) -> np.ndarray:
    w = np.asarray(w, U32)
    return np.where(MC.is_nan(w), U32(QNAN), w).astype(U32)


@lru_cache(maxsize=None)
def _lg2_tables() -> tuple[np.ndarray, ...]:
    c = np.array(D.LG2_COEFFS, np.int64)
    return c[:, 0], c[:, 1], c[:, 2], D.truncated_square(np.arange(1 << D.LG2_OFFSET_BITS, dtype=np.int64))


def lg2_words(x) -> np.ndarray:
    """`MufuLg2Mantissa_v1` on an array of f32 words: the interpolant G at x's mantissa m, G 2^-36 rounded toward zero, +0 at m = 0."""
    m = (np.asarray(x, U32) & U32(0x7FFFFF)).astype(np.int64)
    c0, c1, c2, q = _lg2_tables()
    g, u = m >> D.LG2_OFFSET_BITS, m & ((1 << D.LG2_OFFSET_BITS) - 1)
    G = (c0[g] + c1[g] * u + c2[g] * q[u]) >> D.LG2_SHIFT
    e = np.frexp(np.maximum(G, 1).astype(np.float64))[1].astype(np.int64) - 1          # floor(log2 G): G < 2^37 is exact
    frac = np.where(e <= 23, G << np.clip(23 - e, 0, 23), G >> np.clip(e - 23, 0, 40))
    return np.where(m == 0, 0, ((e + 127 + D.LG2_SCALE) << 23) | (frac & 0x7FFFFF)).astype(U32)


def _sqrt_rn(v) -> np.ndarray:
    with np.errstate(all="ignore"):
        return MC.u32(np.sqrt(MC.f32(v)))


def hash_weight_words(x) -> np.ndarray:
    """`Dsv4HashWeight_v1`: ``y = Ex2Approx(|x| * -log2 e) + 1``, MUFU.LG2 of y (1 at y = 2, 0x7FFFFFFF at a NaN), ``fma(lg2, ln 2,
    fmaxf(x, 0))``, sqrt.rn, a NaN +0."""
    x = np.asarray(x, U32)
    y = MC.add(MC.ex2_approx(MC.mul(x & U32(0x7FFFFFFF), U32(D.NEG_LOG2E))), U32(ONE))
    lg2 = np.where(MC.is_nan(y), U32(QNAN), np.where(y == U32(TWO), U32(ONE), lg2_words(y)))
    r = _sqrt_rn(MC.fma(lg2, U32(D.LN2), MC.fmax(x, U32(0))))
    return np.where(MC.is_nan(r), U32(0), r).astype(U32)


def score_words(x) -> np.ndarray:
    """`Dsv4SqrtSoftplus_v1`: ``MufuSqrtFtz(x > 20 ? x : NvLogf_v2(1 + Ex2Approx(x * log2 e)))``."""
    x = np.asarray(x, U32)
    y = MC.add(U32(ONE), MC.ex2_approx(MC.mul(x, U32(D.LOG2E))))
    with np.errstate(all="ignore"):
        lg = _gpu_nan(MC.u32(nv_logf(MC.f32(y))))
    return MU.mufu_sqrt_words(np.where(MC.f32(x) > np.float32(20.0), x, lg), MU.table("sqrt"))


def lane_sum(w: np.ndarray) -> np.ndarray:
    """The six slots' sum both kernels compute (`moe._lane0_sum`; `tl.sum` of eight slots, two +0): ``((w0 + w4) + w2) + ((w1 + w5) +
    w3)``."""
    return MC.add(MC.add(MC.add(w[:, 0], w[:, 4]), w[:, 2]), MC.add(MC.add(w[:, 1], w[:, 5]), w[:, 3]))


def _denominator(total: np.ndarray) -> np.ndarray:
    return np.where(MC.f32(total) > np.float32(0.0), total, U32(ONE)).astype(U32)


def router_twin(x, b, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """`MoeRouterSqrtSoftplus_v1{E, SCALE=scale}` on rows x (n, E) under bias b (E,) or (n, E): weights (n, 6) f32 words and ids (n,
    6).  ``current = score + bias``, a NaN -1e30; six slots of the lowest index of the largest current, its score, the current then
    -inf; ``w * div.full(SCALE, sum > 0 ? sum : 1)`` (the product's NaN 0x7FFFFFFF)."""
    x = np.asarray(x, U32)
    n = x.shape[0]
    s = score_words(x)
    cur = MC.add(s, np.broadcast_to(np.asarray(b, U32), x.shape))
    cur = MC.f32(np.where(MC.is_nan(cur), U32(D.NAN_SCORE), cur)).copy()
    rows = np.arange(n)
    sel, ids = np.zeros((n, TOPK), U32), np.zeros((n, TOPK), np.int32)
    for k in range(TOPK):
        j = np.argmax(cur == cur.max(axis=1, keepdims=True), axis=1)
        sel[:, k], ids[:, k] = s[rows, j], j
        cur[rows, j] = -np.inf
    q = MC.div_full(_w(scale), _denominator(lane_sum(sel)))
    return _gpu_nan(MC.mul(sel, q[:, None])), ids


def hash_twin(x, ids, scale: float) -> np.ndarray:
    """`MoeRouterHash_v1{E, SCALE=scale}` on rows x (n, E) and their table ids (n, 6) in [0, E): ``w * SCALE / (sum > 0 ? sum :
    1)``, FMUL then div.rn, each NaN 0x7FFFFFFF.  The ids are written unchanged."""
    w = hash_weight_words(np.take_along_axis(np.asarray(x, U32), np.asarray(ids, np.int64), 1))
    num = _gpu_nan(MC.mul(w, _w(scale)))
    return _gpu_nan(DR.div_rn(num, _denominator(lane_sum(w))[:, None]))


def clamp_inputs(gu: np.ndarray, limit: float) -> np.ndarray:
    """The clamped gate and up as bf16 words: ``fminf(g, LIMIT)`` and ``fmaxf(fminf(u, LIMIT), -LIMIT)`` (a NaN LIMIT), both exact."""
    gu = np.asarray(gu, np.uint16)
    half = gu.shape[1] // 2
    lim = U32(D.limit_bits(limit) >> 16)
    f = MC.f32(DR.bf16_to_f32_words(gu))
    nan = MC.is_nan(DR.bf16_to_f32_words(gu))
    hi = nan | (f > np.float32(limit))
    lo = np.zeros_like(hi)
    lo[:, half:] = f[:, half:] < np.float32(-limit)
    return np.where(hi, lim, np.where(lo, lim | U32(0x8000), gu)).astype(np.uint16)


def clamp_twin(gu: np.ndarray, limit: float) -> np.ndarray:
    """`SiluMulClamp_v1{I, LIMIT=limit}` on rows gu (n, 2I) of bf16 words: `SiluMul_v2` of the clamped gate and up (n, I)."""
    return silu_mul_expf_rows(clamp_inputs(gu, limit))["out"].astype(np.uint16)


__all__ = ["clamp_inputs", "clamp_twin", "hash_twin", "hash_weight_words", "lane_sum", "lg2_words", "router_twin", "score_words"]
