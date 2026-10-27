"""numpy kernels for the tensor-core steps and GEMM coordinates, bit-exact batch forms of `verity.ml.circuits.silicon`.

    group_sum_total_batch      `GroupSum` on every encoding (`total.tc_dot_total` for BF16, `total_fp8.tc_dot_total_e4m3`
                               for E4M3): NaN / sticky and saturating infinities / `+0` zero, as the scalar total rules
    block_scaled_batch         `BlockScaledAlignAdd.step_scaled` (NVFP4 / MXFP4); declines what the scalar model rejects
                               (non-finite accumulators, invalid scale codes, a sum leaving binary32)
    chain_batch                an ascending-K fold of a step from a `+0` accumulator (one GEMM coordinate's accumulator)
    f2fp_bf16_batch, ...       the FP32 <-> BF16 word casts

Encodings are unsigned integer arrays with a leading batch axis; the arithmetic is int64 (every intermediate is below
2^53).  The IR kernels are registered next to their Definitions (`prims`, `gemm`).  The FP8 and NVFP4 steps have no core
Definition yet, so their references are `ScalarReference`s over the `verity.ml.circuits.silicon` functions, registered here.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from verity.ml.circuits.evaluation import ScalarReference, register_kernel
from verity.ml.circuits.types import Array, Value

from verity.ml.circuits.silicon.models import ADA_E4M3_M16N8K32, BLACKWELL_SM120_MXF4, BLACKWELL_SM120_NVF4, BlockScaledAlignAdd, GroupSum
from verity.ml.circuits.silicon.term import FORMATS
from verity.ml.circuits.silicon.total_fp8 import HOPPER_E4M3_WGMMA_K32, tc_dot_total_e4m3

__all__ = [
    "HOPPER_E4M3_DOT",
    "HOPPER_E4M3_STEP",
    "SM120_MXF4_DOT",
    "SM120_MXF4_STEP",
    "SM120_NVF4_DOT",
    "SM120_NVF4_STEP",
    "bf16_to_f32_batch",
    "block_scaled_batch",
    "chain_batch",
    "dot_kernel",
    "f2fp_bf16_batch",
    "f32_to_bf16_rn_batch",
    "gemm_coordinate_kernel",
    "gemm_kernel",
    "group_sum_total_batch",
    "sample_tc",
    "sample_tc_e5m2",
]

F32 = Value(32)
_I64 = np.int64
_NAN_WORD = np.uint32(0x7FFFFFFF)
_INF_WORD = np.uint32(0x7F800000)
_SIGN = np.uint32(0x80000000)


def _bitlen(x: np.ndarray) -> np.ndarray:
    """``int.bit_length`` of non-negative int64 values below 2^53."""
    return np.frexp(x.astype(np.float64))[1].astype(_I64)


def _pack_finite(neg: np.ndarray, mag: np.ndarray, exp: np.ndarray) -> np.ndarray:
    """`term.pack_fp32` of in-range terms (24-bit magnitudes, exponent <= 127); zero is `+0`."""
    field = exp + 127 - ((mag & (1 << 23)) == 0)
    word = (neg.astype(_I64) << 31) | (field << 23) | (mag & 0x7FFFFF)
    return np.where(mag == 0, 0, word).astype(np.uint32)


# ---- casts ------------------------------------------------------------------------------------------------------------

def bf16_to_f32_batch(a: np.ndarray) -> tuple[np.ndarray, None]:
    return a.astype(np.uint32) << 16, None


def _f32_to_bf16(u: np.ndarray, nan: int) -> np.ndarray:
    u = u.astype(np.uint32).astype(_I64)
    rne = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
    is_nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x7FFFFF) != 0)
    return np.where(is_nan, nan, rne).astype(np.uint16)


def f2fp_bf16_batch(a: np.ndarray) -> tuple[np.ndarray, None]:
    """`cast.f32_to_bf16_hw_word`: RNE, every NaN -> 0x7FFF."""
    return _f32_to_bf16(a, 0x7FFF), None


def nvf4_step_total_batch(model: BlockScaledAlignAdd, acc: np.ndarray, a_codes: np.ndarray, b_codes: np.ndarray,
                          a_scales: np.ndarray, b_scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`prims.BlackwellNvf4OmmaDot64` on every encoding: a NaN-coded scale byte -> 0x7FFFFFFF; else a NaN accumulator ->
    0x7FFFFFFF and +-inf -> itself; else `block_scaled_batch` with bit 7 of every scale byte cleared."""
    acc = np.asarray(acc, dtype=np.uint32)
    sa, sb = np.asarray(a_scales, dtype=np.uint8), np.asarray(b_scales, dtype=np.uint8)
    nan_scale = ((sa & 0x7F) == 0x7F).any(axis=1) | ((sb & 0x7F) == 0x7F).any(axis=1)
    special = (acc & 0x7F800000) == 0x7F800000
    skip = nan_scale | special
    words, declined = block_scaled_batch(model, np.where(skip, np.uint32(0), acc), a_codes, b_codes,
                                         np.where(skip[:, None], np.uint8(0), sa & 0x7F), np.where(skip[:, None], np.uint8(0), sb & 0x7F))
    words = np.where(special, np.where((acc & 0x7FFFFF) != 0, np.uint32(0x7FFFFFFF), acc), np.asarray(words, dtype=np.uint32))
    return np.where(nan_scale, np.uint32(0x7FFFFFFF), words), np.asarray(declined, dtype=bool) & ~skip


def mxf4_step_total_batch(model: BlockScaledAlignAdd, acc: np.ndarray, a_codes: np.ndarray, b_codes: np.ndarray,
                          a_scales: np.ndarray, b_scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`prims.BlackwellMxf4OmmaDot64` on every encoding: a 0xFF scale byte -> 0x7FFFFFFF; else a NaN accumulator ->
    0x7FFFFFFF and +-inf -> itself; else `block_scaled_batch` with a sum leaving binary32 -> +-inf of its sign."""
    acc = np.asarray(acc, dtype=np.uint32)
    sa, sb = np.asarray(a_scales, dtype=np.uint8), np.asarray(b_scales, dtype=np.uint8)
    nan_scale = (sa == 0xFF).any(axis=1) | (sb == 0xFF).any(axis=1)
    special = (acc & 0x7F800000) == 0x7F800000
    skip = nan_scale | special
    words, declined = block_scaled_batch(model, np.where(skip, np.uint32(0), acc), a_codes, b_codes,
                                         np.where(skip[:, None], np.uint8(0), sa), np.where(skip[:, None], np.uint8(0), sb),
                                         overflow_inf=True)
    words = np.where(special, np.where((acc & 0x7FFFFF) != 0, np.uint32(0x7FFFFFFF), acc), np.asarray(words, dtype=np.uint32))
    return np.where(nan_scale, np.uint32(0x7FFFFFFF), words), np.asarray(declined, dtype=bool) & ~skip


def f32_to_e2m1_sat_batch(a: np.ndarray, nan: int) -> tuple[np.ndarray, None]:
    """`cast.f32_to_e2m1_sat_word` on every word."""
    from verity.ml.circuits.silicon.cast import E2M1_UP

    u = np.asarray(a, dtype=np.uint32)
    sign = ((u >> 31) << 3).astype(np.uint8)
    mag = u & np.uint32(0x7FFFFFFF)
    code = np.zeros(u.shape, dtype=np.uint8)
    for bound, up in E2M1_UP:
        code += ((mag > bound) | ((mag == bound) if up else False)).astype(np.uint8)
    return np.where(mag > 0x7F800000, np.uint8(nan), code | sign), None


_E4M3_TO_F32 = None


def e4m3_to_f32_batch(a: np.ndarray) -> tuple[np.ndarray, None]:
    """`cast.e4m3_to_f32_word` on every byte (a 256-entry table of it)."""
    global _E4M3_TO_F32
    if _E4M3_TO_F32 is None:
        from verity.ml.circuits.silicon.cast import e4m3_to_f32_word

        _E4M3_TO_F32 = np.array([e4m3_to_f32_word(b) for b in range(256)], dtype=np.uint32)
    return _E4M3_TO_F32[np.asarray(a, dtype=np.int64) & 0xFF], None


def f32_to_bf16_rn_batch(a: np.ndarray) -> tuple[np.ndarray, None]:
    """`cast.f32_to_bf16_word`: RNE, every NaN -> 0x7FC0."""
    return _f32_to_bf16(a, 0x7FC0), None


# ---- GroupSum on every encoding -------------------------------------------------------------------------------------

def _operands(fmt, w: np.ndarray):
    """Per word: (nan, inf, zero, sign, significand, biased exponent) under `fmt`'s encoding rules."""
    w = w.astype(_I64)
    e = (w >> fmt.man_bits) & ((1 << fmt.exp_bits) - 1)
    m = w & ((1 << fmt.man_bits) - 1)
    all_ones = e == (1 << fmt.exp_bits) - 1
    if fmt.nan_rule == "ieee":
        nan, inf = all_ones & (m != 0), all_ones & (m == 0)
    elif fmt.nan_rule == "e4m3":
        nan, inf = all_ones & (m == (1 << fmt.man_bits) - 1), np.zeros(w.shape, bool)
    else:
        nan = inf = np.zeros(w.shape, bool)
    sig = m | ((e != 0).astype(_I64) << fmt.man_bits)
    return nan, inf, sig == 0, (w >> (fmt.bits - 1)) & 1, sig, e


def group_sum_total_batch(model: GroupSum, acc: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, None]:
    """`model`'s step on every encoding: acc ``(N,)`` FP32 words, a/b ``(N, k)`` operand words -> ``(N,)`` FP32 words.
    Equal to `total.tc_dot_total` (BF16) / `total_fp8.tc_dot_total_e4m3` (E4M3) / `total_fp8.tc_dot_total_e5m2` (E5M2) on every
    input; declines nothing."""
    fmt = FORMATS[model.dtype]
    na, ia, za, sa, ga, ea = _operands(fmt, a)
    nb, ib, zb, sb, gb, eb = _operands(fmt, b)
    p_nan = na | nb | (ia & zb) | (ib & za)
    p_inf = (ia | ib) & ~p_nan
    p_neg = (sa ^ sb) == 1
    p_fin = ~(p_nan | p_inf)
    p_mag = np.where(p_fin, (ga * gb) << fmt.product_shift, 0)
    p_exp = np.maximum(ea, 1) + np.maximum(eb, 1) + fmt.product_exponent_offset

    acc = acc.astype(np.uint32).astype(_I64)
    a_field, a_frac = (acc >> 23) & 0xFF, acc & 0x7FFFFF
    state = np.where(a_field == 0xFF, np.where(a_frac != 0, 1, 2), 0)  # 0 finite, 1 NaN, 2 infinity
    neg = (acc >> 31) == 1
    mag = np.where(state == 0, np.where(a_field == 0, a_frac, a_frac | (1 << 23)), 0)
    exp = np.where(a_field == 0, -126, a_field - 127)

    width, rescale = model.width, model.width - 24
    start = 0
    for size in model.groups:
        sl = slice(start, start + size)
        start += size
        pos_inf = ((state == 2) & ~neg) | (p_inf[:, sl] & ~p_neg[:, sl]).any(-1)
        neg_inf = ((state == 2) & neg) | (p_inf[:, sl] & p_neg[:, sl]).any(-1)
        is_nan = (state == 1) | p_nan[:, sl].any(-1) | (pos_inf & neg_inf)
        is_inf = ~is_nan & (pos_inf | neg_inf)

        mags = np.concatenate([mag[:, None], p_mag[:, sl]], axis=1)
        exps = np.concatenate([exp[:, None], p_exp[:, sl]], axis=1)
        negs = np.concatenate([neg[:, None], p_neg[:, sl]], axis=1)
        live = mags != 0
        top = np.maximum(np.where(live, exps, model.zero_exponent).max(axis=1), model.zero_exponent)
        scaled = mags << rescale if rescale >= 0 else mags >> -rescale
        aligned = np.where(live, scaled >> np.minimum(top[:, None] - exps, 63), 0)
        total = np.where(negs, -aligned, aligned).sum(axis=1)
        s_neg = total < 0
        s_mag = np.abs(total)
        bl = _bitlen(s_mag)
        s_exp = top + bl - width
        s_mag = np.where(bl > width, s_mag >> np.clip(bl - width, 0, 63), s_mag << np.clip(width - bl, 0, 63))
        s_mag >>= np.clip(-126 - s_exp, 0, 63)
        s_exp = np.maximum(s_exp, -126)
        s_mag = s_mag >> rescale if rescale >= 0 else s_mag << -rescale
        over = ~is_nan & ~is_inf & (s_mag != 0) & (s_exp > 127)

        new_state = np.where(is_nan, 1, np.where(is_inf | over, 2, 0))
        neg = np.where(is_inf, neg_inf, s_neg & (s_mag != 0))
        mag = np.where(new_state == 0, s_mag, 0)
        exp = s_exp
        state = new_state

    word = _pack_finite(neg, mag, np.minimum(exp, 127))
    word = np.where(state == 2, np.where(neg, _SIGN | _INF_WORD, _INF_WORD), word)
    return np.where(state == 1, _NAN_WORD, word).astype(np.uint32), None


# ---- BlockScaledAlignAdd (NVFP4 / MXFP4) ------------------------------------------------------------------------------

_E2M1_HALVES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=_I64)
_FP32_SUBNORMAL_EXPONENT = -149
_NONE = -(1 << 40)


def _scale_batch(scale: str, code: np.ndarray):
    """(mantissa, exponent, invalid) of block-scale bytes (`models._scale_dyadic`)."""
    c = code.astype(_I64)
    if scale == "ue8m0":
        return np.full(c.shape, 8, _I64), c - 130, c == 0xFF
    e, m = (c >> 3) & 0xF, c & 7                    # the low 7 bits: sm_120 ignores bit 7 (`models._scale_dyadic`)
    invalid = (e == 15) & (m == 7)
    return np.where(e == 0, m, 8 + m), np.where(e == 0, -9, e - 10), invalid


def _trunc_rz(mant: np.ndarray, exp: np.ndarray, lsb: np.ndarray) -> np.ndarray:
    up = np.clip(exp - lsb, 0, 62)
    q = np.abs(mant) >> np.clip(lsb - exp, 0, 63)
    return np.where(exp >= lsb, mant << up, np.where(mant < 0, -q, q))


def block_scaled_batch(model: BlockScaledAlignAdd, acc: np.ndarray, a_codes: np.ndarray, b_codes: np.ndarray,
                       a_scales: np.ndarray, b_scales: np.ndarray, *, overflow_inf: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """`model.step_scaled` on ``(N,)`` FP32 words, ``(N, k)`` E2M1 codes and ``(N, k / block)`` scale bytes per operand.
    Returns ``(words, declined)``; declined are the instances on which the scalar model raises, except that with
    ``overflow_inf`` a sum leaving binary32 is ±inf of its sign."""
    n, k, g = acc.shape[0], model.k, model.group
    groups = k // g
    prod = _E2M1_HALVES[a_codes.astype(_I64)] * _E2M1_HALVES[b_codes.astype(_I64)]
    gd = prod.reshape(n, groups, g).sum(axis=2)
    participates = (prod != 0).reshape(n, groups, g).any(axis=2)
    mant, unit = gd, np.full((n, groups), -2, _I64)
    declined = np.zeros(n, bool)
    if model.scale:
        per = model.block // g
        ma, ea, ia = _scale_batch(model.scale, np.repeat(a_scales, per, axis=1))
        mb, eb, ib = _scale_batch(model.scale, np.repeat(b_scales, per, axis=1))
        declined |= ia.any(axis=1) | ib.any(axis=1)
        mant, unit = mant * ma * mb, unit + ea + eb
        participates &= (ma != 0) & (mb != 0)

    acc = acc.astype(np.uint32).astype(_I64)
    field, frac = (acc >> 23) & 0xFF, acc & 0x7FFFFF
    declined |= field == 0xFF
    acc_m = np.where(field == 0, frac, frac | (1 << 23))
    acc_m = np.where((acc >> 31) == 1, -acc_m, acc_m)
    acc_m = np.where(field == 0xFF, 0, acc_m)
    acc_e = np.where(field == 0, _FP32_SUBNORMAL_EXPONENT, field - 150)

    anchor = np.where(participates, unit - model.frac_bits, _NONE).max(axis=1)
    acc_anchor = np.where(acc_m != 0, np.maximum(field, 1) - 127 - (model.acc_window - 1), _NONE)
    anchor = np.maximum(anchor, acc_anchor)
    lsb = np.maximum(anchor, model.grid_floor)
    total = np.where(mant != 0, _trunc_rz(mant, unit, lsb[:, None]), 0).sum(axis=1)
    total += np.where(acc_m != 0, _trunc_rz(acc_m, acc_e, lsb), 0)

    t_neg = total < 0
    t_mag = np.abs(total)
    e_val = _bitlen(t_mag) - 1 + lsb
    grid = np.where(e_val >= -126, e_val - 23, _FP32_SUBNORMAL_EXPONENT)
    shift = grid - lsb
    q = np.where(shift > 0, t_mag >> np.clip(shift, 0, 63), t_mag << np.clip(-shift, 0, 62))
    e = _bitlen(q) - 1 + grid
    live = (anchor != _NONE) & (total != 0)
    over = live & (q != 0) & (e > 127)
    bits = np.where(e >= -126, ((e + 127) << 23) | (q & 0x7FFFFF), q)
    word = np.where(t_neg, bits | (1 << 31), bits)
    if model.underflow_zero == "signed":
        word = np.where(q == 0, np.where(t_neg, 1 << 31, 0), word)
    else:
        word = np.where(q == 0, 0, word)
    if overflow_inf:
        word = np.where(over, np.where(t_neg, int(_SIGN | _INF_WORD), int(_INF_WORD)), word)
    else:
        declined |= over
    return np.where(live & ~declined, word, 0).astype(np.uint32), declined


# ---- chains and the IR kernels ------------------------------------------------------------------------------------------

def chain_batch(step: Callable[..., tuple[np.ndarray, Any]], k: int, n_steps: int, *operands: np.ndarray,
                per_step: tuple[int, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fold ``step(acc, *slices)`` over ``n_steps`` from a `+0` accumulator; operand ``j`` contributes ``per_step[j]``
    columns per step (``k`` for all when omitted).  An instance declined by any step is declined."""
    n = operands[0].shape[0]
    widths = per_step or (k,) * len(operands)
    acc = np.zeros(n, np.uint32)
    declined = np.zeros(n, bool)
    for s in range(n_steps):
        acc, d = step(acc, *(op[:, s * w:(s + 1) * w] for op, w in zip(operands, widths)))
        if d is not None:
            declined |= d
    return acc, declined


def dot_kernel(steps: Mapping[Any, GroupSum]) -> Callable[..., Any]:
    """The `DotBf16_v2{K, DOT}` kernel for the ``DOT`` step objects in ``steps`` (matched by identity, never by id)."""

    def kernel(x: np.ndarray, w: np.ndarray, *, K: int, DOT: Any):
        model = steps.get(DOT)
        if model is None:
            return NotImplemented
        return chain_batch(lambda acc, a, b: group_sum_total_batch(model, acc, a, b), model.k, K // model.k, x, w)

    return kernel


def gemm_coordinate_kernel(steps: Mapping[Any, GroupSum]) -> Callable[..., Any]:
    """`GemmCoordinate_v2{K, DOT}`: the dot, then `cvt.rn.bf16.f32`."""
    dot = dot_kernel(steps)

    def kernel(x: np.ndarray, w: np.ndarray, *, K: int, DOT: Any):
        res = dot(x, w, K=K, DOT=DOT)
        if res is NotImplemented:
            return res
        return f2fp_bf16_batch(res[0])[0], res[1]

    return kernel


def block_scaled_coordinate_kernel(steps: Mapping[Any, tuple[BlockScaledAlignAdd, Callable[..., Any]]]) -> Callable[..., Any]:
    """`GemmCoordinateNvf4_v1` / `GemmCoordinateMxf4_v1{K, DOT}` for the ``DOT`` steps in ``steps`` (by identity; each with its
    model and total step kernel): K/64 steps from `+0`, then `cvt.rn.bf16.f32`."""

    def kernel(x: np.ndarray, w: np.ndarray, sx: np.ndarray, sw: np.ndarray, *, K: int, DOT: Any):
        entry = steps.get(DOT)
        if entry is None:
            return NotImplemented
        model, fire = entry
        ns = model.scales_per_step
        acc, declined = chain_batch(lambda acc, *ops: fire(model, acc, *ops), model.k, K // model.k, x, w, sx, sw,
                                    per_step=(model.k, model.k, ns, ns))
        return f2fp_bf16_batch(acc)[0], declined

    return kernel


def gemm_kernel(steps: Mapping[Any, GroupSum]) -> Callable[..., Any]:
    """`Gemm_v2{K, N, DOT}`: ``x (B, K)``, ``w (B, N, K)`` -> ``(B, N)``, one coordinate per output."""
    coord = gemm_coordinate_kernel(steps)

    def kernel(x: np.ndarray, w: np.ndarray, *, K: int, N: int, DOT: Any):
        b = x.shape[0]
        xs = np.broadcast_to(x[:, None, :], (b, N, K)).reshape(b * N, K)
        res = coord(xs, w.reshape(b * N, K), K=K, DOT=DOT)
        if res is NotImplemented:
            return res
        return res[0].reshape(b, N), res[1].reshape(b, N).any(axis=1)

    return kernel


def gemm_coordinate_pad_kernel(steps: Mapping[Any, GroupSum]) -> Callable[..., Any]:
    """`GemmCoordinatePad_v1{T, K, DOT}`: the coordinate on ``x`` and ``w`` (B, T), each padded with K - T +0 words."""
    coord = gemm_coordinate_kernel(steps)

    def kernel(x: np.ndarray, w: np.ndarray, *, T: int, K: int, DOT: Any):
        def pad(a):
            return np.pad(a, ((0, 0), (0, K - T)))
        return coord(pad(x), pad(w), K=K, DOT=DOT)

    return kernel


def pv_row_kernel(steps: Mapping[Any, GroupSum]) -> Callable[..., Any]:
    """`PvRow_v1{T, K, D, DOT}`: ``p (B, T)`` and the T key rows ``v_t (B, D)`` -> ``(B, D)``, dim d the padded coordinate of
    ``p`` and ``(v_0[d], ..., v_{T-1}[d])``."""
    coord = gemm_coordinate_kernel(steps)

    def kernel(p: np.ndarray, *vs: np.ndarray, T: int, K: int, D: int, DOT: Any):
        b = p.shape[0]
        xs = np.zeros((b, D, K), p.dtype)
        xs[:, :, :T] = p[:, None, :]
        ws = np.zeros((b, D, K), p.dtype)
        ws[:, :, :T] = np.stack(vs, axis=2)
        res = coord(xs.reshape(b * D, K), ws.reshape(b * D, K), K=K, DOT=DOT)
        if res is NotImplemented:
            return res
        return res[0].reshape(b, D), res[1].reshape(b, D).any(axis=1)

    return kernel


def pv_row_pad_kernel(steps: Mapping[Any, GroupSum]) -> Callable[..., Any]:
    """`PvRowPad_v1{T, K, D, DOT}`: `PvRow_v1{T, K, D, DOT}` with key t's p word and V row zeroed unless t < n (n an
    unsigned u16 word, ``(B,)``)."""
    row = pv_row_kernel(steps)

    def kernel(p: np.ndarray, *rest: np.ndarray, T: int, K: int, D: int, DOT: Any):
        *vs, n = rest
        keep = np.arange(T)[None, :] < (np.asarray(n).astype(np.int64) & 0xFFFF)[:, None]
        return row(np.where(keep, p, 0).astype(p.dtype), *(np.where(keep[:, t, None], v, 0).astype(v.dtype)
                                                           for t, v in enumerate(vs)), T=T, K=K, D=D, DOT=DOT)

    return kernel


# ---- instance sampling for self_check ------------------------------------------------------------------------------------

def _words(rng: np.random.Generator, fmt, shape: tuple[int, ...], specials: tuple[int, ...], p_special: np.ndarray,
           regime: np.ndarray, spread: int = 6) -> np.ndarray:
    """Operand words with exponents clustered around a per-instance centre (so products align, cancel and overflow),
    a uniform word one time in eight, and ``specials`` with probability ``p_special`` (per input).  The centre is
    uniform (regime 0), at the unit binade (1), or where products land at the bottom of the FP32 range (2)."""
    n = shape[0]
    emax = (1 << fmt.exp_bits) - 1
    tiny = max(0, (2 * fmt.bias - 140) // 2)
    centre = np.select([regime == 1, regime == 2], [fmt.bias, tiny + rng.integers(-4, 5, size=n)], rng.integers(0, emax + 1, size=n))
    centre = centre.reshape((n,) + (1,) * (len(shape) - 1))
    e = np.clip(centre + rng.integers(-spread, spread + 1, size=shape), 0, emax)
    m = rng.integers(0, 1 << fmt.man_bits, size=shape)
    s = rng.integers(0, 2, size=shape)
    w = (s << (fmt.bits - 1)) | (e << fmt.man_bits) | m
    w = np.where(rng.random(shape) < 0.125, rng.integers(0, 1 << fmt.bits, size=shape), w)
    if specials:
        sp = np.asarray(specials)[rng.integers(0, len(specials), size=shape)]
        w = np.where(rng.random(shape) < p_special.reshape((n,) + (1,) * (len(shape) - 1)), sp, w)
    return w


def _acc_words(rng: np.random.Generator, n: int, regime: np.ndarray) -> np.ndarray:
    """FP32 accumulators: uniform words, moderate exponents, subnormals, both zeros, the largest finite, NaN, infinities;
    regime-2 inputs (tiny products) get a zero or subnormal accumulator."""
    kind = rng.integers(0, 16, size=n)
    frac = rng.integers(0, 1 << 23, size=n)
    sign = rng.integers(0, 2, size=n) << 31
    mid = sign | (rng.integers(90, 170, size=n) << 23) | frac
    out = np.select(
        [kind < 2, kind < 4, kind == 4, kind == 5, kind == 6],
        [rng.integers(0, 1 << 32, size=n), sign | frac, sign, sign | 0x7F7FFFFF,
         np.asarray([0x7F800000, 0xFF800000, 0x7FC00000, 0xFFFFFFFF, 0x7F800001])[rng.integers(0, 5, size=n)]],
        mid)
    out = np.where(regime == 2, np.where(kind < 8, sign, sign | (frac >> rng.integers(0, 23, size=n))), out)
    return out.astype(np.uint32)


#: The special operand words `sample_tc` draws, per 8-bit format: zeros, NaNs, the largest finite, the least subnormal (and E5M2's infinities).
_FP8_SPECIALS = {"e4m3": (0x00, 0x80, 0x7F, 0xFF, 0x7E, 0x01), "e5m2": (0x00, 0x80, 0x7C, 0xFC, 0x7D, 0xFF, 0x7B, 0x01)}


def sample_tc(rng: np.random.Generator, n: int, params, *, fp8: str = "e4m3", **statics: Any) -> list[np.ndarray]:
    """Instances for the tensor-core kernels, by parameter width: 32 an accumulator, 16 BF16, 8 `fp8` operands.  A quarter
    of the instances carry special operand words; an eighth are exact cancellations (the second half of ``a`` negates the
    first, ``b`` repeats)."""
    p_special = np.where(rng.random(n) < 0.25, 0.03, 0.0)
    regime = rng.choice(3, size=n, p=(0.5, 0.25, 0.25))
    out: list[np.ndarray] = []
    for pt in params:
        if isinstance(pt, Value):
            out.append(_acc_words(rng, n, regime))
            continue
        w = pt.elem.w
        fmt = FORMATS["bf16" if w == 16 else fp8]
        specials = (0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0xFFFF, 0x0001) if w == 16 else _FP8_SPECIALS[fp8]
        out.append(_words(rng, fmt, (n, pt.n), specials, p_special, regime).astype(np.uint16 if w == 16 else np.uint8))
    if len(out) >= 2 and out[-1].ndim == 2 and out[-1].shape[1] % 2 == 0:
        a, b = out[-2].copy(), out[-1].copy()
        h = a.shape[1] // 2
        c = rng.random(n) < 0.125
        sign = np.asarray(1 << (a.dtype.itemsize * 8 - 1), dtype=a.dtype)
        a[c, h:] = a[c, :h] ^ sign
        b[c, h:] = b[c, :h]
        out[-2], out[-1] = a, b
    return out


def sample_pv_row_pad(rng: np.random.Generator, n: int, params, **statics: Any) -> list[np.ndarray]:
    """`sample_tc` for p and the key rows, and a key count n: mostly 0..T + 1, an eighth uniform u16 words."""
    T = params[0].n
    counts = np.where(rng.random(n) < 0.125, rng.integers(0, 1 << 16, n), rng.integers(0, T + 2, n)).astype(np.uint16)
    return [*sample_tc(rng, n, params[:-1], **statics), counts]


def sample_tc_e5m2(rng: np.random.Generator, n: int, params, **statics: Any) -> list[np.ndarray]:
    """`sample_tc` with E5M2 operands."""
    return sample_tc(rng, n, params, fp8="e5m2", **statics)


def _sample_block_scaled(model: BlockScaledAlignAdd):
    def sample(rng: np.random.Generator, n: int, params, **statics: Any) -> list[np.ndarray]:
        k, ns = params[-4].n, params[-2].n
        codes = []
        for _ in range(2):
            c = rng.integers(0, 16, size=(n, k))
            sparse = rng.random((n, 1)) < 0.25
            codes.append(np.where(sparse & (rng.random((n, k)) < 0.9), c & 8, c).astype(np.uint8))
        scales = []
        for _ in range(2):
            if model.scale == "ue4m3":
                e = np.clip(rng.integers(0, 16, size=(n, 1)) + rng.integers(-2, 3, size=(n, ns)), 0, 15)
                s = (e << 3) | rng.integers(0, 8, size=(n, ns))
                s = np.where(s == 0x7F, 0x7E, s)
                odd = np.asarray([0x7F, 0x80, 0xFE, 0x00])[rng.integers(0, 4, size=n)]
            else:
                s = np.clip(rng.integers(0, 254, size=(n, 1)) + rng.integers(-3, 4, size=(n, ns)), 0, 254)
                odd = np.asarray([0xFF, 0x00, 0xFE])[rng.integers(0, 3, size=n)]
            hit = rng.random(n) < 0.05  # one odd scale byte (invalid, zero or extreme) in 5% of the instances
            s[hit, rng.integers(0, ns, size=n)[hit]] = odd[hit]
            scales.append(s.astype(np.uint8))
        arrays = [*codes, *scales]
        return [_acc_words(rng, n, rng.choice(3, size=n)), *arrays] if len(params) == 5 else arrays
    return sample


# ---- FP8 and NVFP4 references and their kernels (the FP8 step is also the Definition `prims.HopperE4m3QgmmaDot32`) -----

_E4M3, _E2M1, _SCALE = Value(8), Value(4), Value(8)


def _e4m3_dot(x, w, *, K):
    acc = 0
    for s in range(0, K, 32):
        acc = tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, acc, x[s:s + 32], w[s:s + 32])
    return acc


HOPPER_E4M3_STEP = ScalarReference(
    "tc.hopper_e4m3_wgmma_k32.step",
    lambda acc, a, b: tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, acc, a, b),
    lambda: ((F32, Array(32, _E4M3), Array(32, _E4M3)), F32),
    doc="one accumulator coordinate of wgmma m64nNk32 f32.e4m3.e4m3 on every encoding (total_fp8.tc_dot_total_e4m3)")
HOPPER_E4M3_DOT = ScalarReference(
    "tc.hopper_e4m3_wgmma_k32.dot", _e4m3_dot, lambda K: ((Array(K, _E4M3), Array(K, _E4M3)), F32), statics=("K",),
    doc="the FP32 accumulator of one GEMM coordinate: K/32 ascending E4M3 steps from +0 (vLLM's DotE4m3_v1{K})")


def _ada_e4m3_dot(x, w, *, K):
    acc = 0
    for s in range(0, K, 32):
        acc = ADA_E4M3_M16N8K32.step(acc, x[s:s + 32], w[s:s + 32])
    return acc


#: sm_89's E4M3 step is pinned on finite operands and accumulators only (RTX 4090, 24,898,304 elements); no special-value
#: campaign ran on it, so these references raise outside that domain and their kernels decline it.
ADA_E4M3_STEP = ScalarReference(
    "tc.ada_e4m3_mma_k32.step", lambda acc, a, b: ADA_E4M3_M16N8K32.step(acc, a, b),
    lambda: ((F32, Array(32, _E4M3), Array(32, _E4M3)), F32),
    doc="one accumulator coordinate of mma.sync.m16n8k32 f32.e4m3.e4m3 on sm_89 (models.ADA_E4M3_M16N8K32: two groups of 16, "
        "14-bit adder), finite operands and accumulator")
ADA_E4M3_DOT = ScalarReference(
    "tc.ada_e4m3_mma_k32.dot", _ada_e4m3_dot, lambda K: ((Array(K, _E4M3), Array(K, _E4M3)), F32), statics=("K",),
    doc="the FP32 accumulator of one GEMM coordinate on sm_89: K/32 ascending E4M3 steps from +0, finite throughout")


def _finite_step(model: GroupSum) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """``model``'s step, declining every instance with a NaN operand, a non-finite accumulator in or out."""
    def step(acc: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out, _ = group_sum_total_batch(model, acc, a, b)
        acc = np.asarray(acc, dtype=np.uint32)
        nan = ((np.asarray(a) & 0x7F) == 0x7F).any(axis=1) | ((np.asarray(b) & 0x7F) == 0x7F).any(axis=1)
        inf = ((acc & 0x7F800000) == 0x7F800000) | ((np.asarray(out, dtype=np.uint32) & 0x7F800000) == 0x7F800000)
        return out, nan | inf
    return step


def _block_scaled_refs(model: BlockScaledAlignAdd, tag: str) -> tuple[ScalarReference, ScalarReference]:
    k, ns, block = model.k, model.scales_per_step, model.block
    step = ScalarReference(
        f"tc.{tag}.step", model.step_scaled,
        lambda: ((F32, Array(k, _E2M1), Array(k, _E2M1), Array(ns, _SCALE), Array(ns, _SCALE)), F32),
        doc=f"one mma.sync kind::mxf4nvf4 accumulator coordinate ({model.name}.step_scaled)")
    dot = ScalarReference(
        f"tc.{tag}.dot", lambda ac, bc, as_, bs, *, K: model.chain(0, ac, bc, as_, bs),
        lambda K: ((Array(K, _E2M1), Array(K, _E2M1), Array(K // block, _SCALE), Array(K // block, _SCALE)), F32),
        statics=("K",), doc=f"the FP32 accumulator of one GEMM coordinate: K/{k} ascending {model.name} firings from +0")
    return step, dot


SM120_NVF4_STEP, SM120_NVF4_DOT = _block_scaled_refs(BLACKWELL_SM120_NVF4, "blackwell_sm120_nvf4_m16n8k64")
SM120_MXF4_STEP, SM120_MXF4_DOT = _block_scaled_refs(BLACKWELL_SM120_MXF4, "blackwell_sm120_mxf4_m16n8k64")


def _register_references() -> None:
    fp8 = HOPPER_E4M3_WGMMA_K32
    fp8_step = lambda acc, a, b: group_sum_total_batch(fp8, acc, a, b)  # noqa: E731
    register_kernel(HOPPER_E4M3_STEP, fp8_step, domain="every encoding", sample=sample_tc)
    register_kernel(HOPPER_E4M3_DOT, lambda x, w, *, K: chain_batch(fp8_step, 32, K // 32, x, w),
                    domain="every encoding", sample=sample_tc)
    ada_step = _finite_step(ADA_E4M3_M16N8K32)
    finite = "finite E4M3 operands and FP32 accumulators (in and out); the rest declined"
    register_kernel(ADA_E4M3_STEP, ada_step, domain=finite, sample=sample_tc)
    register_kernel(ADA_E4M3_DOT, lambda x, w, *, K: chain_batch(ada_step, 32, K // 32, x, w), domain=finite, sample=sample_tc)
    for model, step, dot in ((BLACKWELL_SM120_NVF4, SM120_NVF4_STEP, SM120_NVF4_DOT),
                             (BLACKWELL_SM120_MXF4, SM120_MXF4_STEP, SM120_MXF4_DOT)):
        fire = (lambda m: lambda acc, ac, bc, as_, bs: block_scaled_batch(m, acc, ac, bc, as_, bs))(model)
        domain = "finite accumulators, valid scale codes, sums inside binary32"
        register_kernel(step, fire, domain=domain, sample=_sample_block_scaled(model))
        register_kernel(dot, (lambda f, m: lambda ac, bc, as_, bs, *, K: chain_batch(
            f, m.k, K // m.k, ac, bc, as_, bs, per_step=(m.k, m.k, m.scales_per_step, m.scales_per_step)))(fire, model),
            domain=domain, sample=_sample_block_scaled(model))


_register_references()
