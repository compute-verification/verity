"""[DSV4] numpy twins of DeepSeek-V4-Pro's mHC Definitions (`registry.mhc`) over a leading token axis: the same operations in
the same order, on u32 (f32) and u16 (bf16) words, for the H200 capture (`tests/properties/dsv4_mhc_capture_gpu.py`) and the
Definitions' tests.  Weights (`fn`, `scale`, `base`, `w`) carry no token axis.  Each primitive is its word Definition's semantics:
the f32 add, multiply and divide numpy's (a NaN lane recomputed by the primitive, whose payload numpy's array loops can take from
the other operand), the FMA `derived_rows.fma_rn`, `rsqrt.approx` `derived_rows.rsqrt_approx`, expf `moe_twins.nv_expf_bits`.

The TF32 GEMM takes its k8 step as a `GroupSum` (`HOPPER_TF32_WGMMA_K8_CANDIDATE` by default), each operand's conversion to TF32
(``"trunc"``, ``"rna"``, ``"rne"``; Hopper is RNE) and the squared sum's contraction (``sqr``: ``"lo"``, the Definitions'
GPU `fma(x[k], x[k], x[k+4] x[k+4])`; ``"hi"``, the FMA on x[k + 4]; ``"mul"``, none) as parameters, so the capture can name
which of them the H200 runs.  A is widened bf16 and therefore already exact in TF32.  On bf16 operands a square is exact unless
it is an f32 subnormal, so only such operands tell the three contractions apart."""
from __future__ import annotations

import numpy as np

from verity.ml.circuits.ops.scalar import F32Fmaxf, F32_NAN_PTX
from verity.ml.circuits.silicon.models import GroupSum
from verity.ml.kernels import tc as TC
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.kernels.moe_twins import nv_expf_bits
from verity_vllm.program.registry import mhc as M
from verity_vllm.program.registry import prims as P

U32 = np.uint32
SIGN = U32(0x80000000)
ONE, ZERO, NEG_INF = U32(M.F_ONE), U32(0), U32(M.F_NEG_INF)
CANDIDATE = M.HOPPER_TF32_WGMMA_K8_CANDIDATE
B_ROUNDINGS = ("trunc", "rna", "rne")
SQR_FORMS = ("lo", "hi", "mul")


def word(v: float) -> np.uint32:
    return U32(M.f32_bits(v))


def _u(x) -> np.ndarray:
    return np.asarray(x, dtype=U32)


def _f(x) -> np.ndarray:
    return _u(x).view(np.float32)


def _is_nan(u: np.ndarray) -> np.ndarray:
    return ((u & 0x7F800000) == 0x7F800000) & ((u & 0x7FFFFF) != 0)


def _binop(a, b, op, prim) -> np.ndarray:
    a, b = np.broadcast_arrays(_u(a), _u(b))
    with np.errstate(all="ignore"):
        out = np.asarray(op(a.view(np.float32), b.view(np.float32)), dtype=np.float32).view(U32).copy()
    nan = _is_nan(out)
    if nan.any():
        out[nan] = [prim.evaluate(int(x), int(y)) for x, y in zip(a[nan], b[nan])]
    return out


def add(a, b) -> np.ndarray:
    return _binop(a, b, np.add, P.F32Add)


def mul(a, b) -> np.ndarray:
    return _binop(a, b, np.multiply, P.F32Mul)


def div(a, b) -> np.ndarray:
    return _binop(a, b, np.divide, P.F32Div)


def fma(a, b, c) -> np.ndarray:
    a, b, c = np.broadcast_arrays(_u(a), _u(b), _u(c))
    return DR.fma_rn(a, b, c).reshape(a.shape)


def neg(x) -> np.ndarray:
    return _u(x) ^ SIGN


def fmaxf(a, b) -> np.ndarray:
    """`F32Fmaxf_v1`: a NaN yields the other operand, two NaNs `F32_NAN_PTX`, max(+0, -0) = +0, else the larger."""
    a, b = np.broadcast_arrays(_u(a), _u(b))
    na, nb = _is_nan(a), _is_nan(b)

    def order(u):
        m = (u & 0x7FFFFFFF).astype(np.int64)
        return np.where(u >> 31 == 1, -m, m)
    ka, kb = order(a), order(b)
    zeros = ((a & 0x7FFFFFFF) == 0) & ((b & 0x7FFFFFFF) == 0)
    out = np.where(ka > kb, a, np.where(ka < kb, b, a))
    out = np.where(zeros, np.where((a == 0) | (b == 0), U32(0), SIGN), out)
    out = np.where(na, b, np.where(nb, a, out))
    return np.where(na & nb, U32(F32Fmaxf.evaluate(0x7FC00000, 0x7FC00000)), out).astype(U32)


def expf(x) -> np.ndarray:
    """`NvExpf_v1` on each distinct word."""
    x = _u(x)
    vals, inv = np.unique(x, return_inverse=True)
    return np.array([nv_expf_bits(int(v)) for v in vals], dtype=U32)[inv.reshape(-1)].reshape(x.shape)


def rsqrt(x) -> np.ndarray:
    x = _u(x)
    return np.asarray(DR.rsqrt_approx(x.reshape(-1)), dtype=U32).reshape(x.shape)


def bf16(x) -> np.ndarray:
    x = _u(x)
    return np.asarray(DR.f2fp_bf16(x.reshape(-1)), dtype=np.uint16).reshape(x.shape)


def cvt(x) -> np.ndarray:
    return np.asarray(x, dtype=np.uint16).astype(U32) << 16


def tree(v: np.ndarray, op=add, axis: int = -1) -> np.ndarray:
    """`dsa.halving_tree` along ``axis`` (a power-of-two length): slot t < h becomes `op(v[t], v[t + h])`."""
    v = np.moveaxis(_u(v), axis, -1)
    n = v.shape[-1]
    if n & (n - 1):
        raise ValueError(f"a halving tree over a power-of-two count (n={n})")
    while v.shape[-1] > 1:
        h = v.shape[-1] // 2
        v = op(v[..., :h], v[..., h:])
    return v[..., 0]


def chain(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """`mhc.chain` along ``axis``: ``((0 + v0) + v1) + ...``."""
    v = np.moveaxis(_u(v), axis, -1)
    acc = np.zeros(v.shape[:-1], dtype=U32)
    for i in range(v.shape[-1]):
        acc = add(acc, v[..., i])
    return acc


# --- the sigmoid, the Sinkhorn normalisation, the mixes and the fused RMSNorm ---------------------------------------------------

def sigmoid(z) -> np.ndarray:
    """`MhcSigmoid_v1`."""
    return div(ONE, add(ONE, expf(add(ZERO, neg(z)))))


def _normalize(cm: np.ndarray, eps, rows: bool) -> np.ndarray:
    axis = -1 if rows else -2
    sums = add(tree(add(ZERO, cm), add, axis), eps)
    return div(cm, sums[..., :, None] if rows else sums[..., None, :])


def sinkhorn(x, REPEAT: int, EPS: float) -> np.ndarray:
    """`MhcSinkhorn_v1{HC, REPEAT, EPS}` on [..., HC, HC] logits."""
    x = _u(x)
    M._check_hc(x.shape[-1])
    if REPEAT < 1:
        raise ValueError(f"MhcSinkhorn: at least one iteration (REPEAT={REPEAT})")
    eps = word(EPS)
    nmax = neg(tree(fmaxf(NEG_INF, x), fmaxf))
    e = expf(add(x, nmax[..., None]))
    s = tree(add(ZERO, e))
    cm = _normalize(add(div(e, s[..., None]), eps), eps, rows=False)
    for _ in range(REPEAT - 1):
        cm = _normalize(_normalize(cm, eps, rows=True), eps, rows=False)
    return cm


def _hc(OUT: int) -> int:
    HC = int(round((1 + OUT) ** 0.5)) - 1
    if HC * (2 + HC) != OUT:
        raise ValueError(f"mHC: OUT = HC (2 + HC) mixes (OUT={OUT})")
    return HC


def mixes(yp, rp, scale, base, HC: int, K: int, REPEAT: int, EPS: float, HC_EPS: float):
    """`MhcMixes_v1`: yp [T, NS, OUT], rp [T, NS] -> (pre [T, HC], post [T, HC], comb [T, HC, HC])."""
    yp, rp, scale, base = _u(yp), _u(rp), _u(scale), _u(base)
    M._check_hc(HC)
    rms = rsqrt(add(div(chain(rp), word(K)), word(EPS)))
    m = mul(chain(yp, axis=-2), rms[..., None])
    pre = add(sigmoid(fma(m[..., :HC], scale[0], base[:HC])), word(HC_EPS))
    s = sigmoid(fma(m[..., HC:2 * HC], scale[1], base[HC:2 * HC]))
    x = fma(m[..., 2 * HC:], scale[2], base[2 * HC:])
    return pre, add(s, s), sinkhorn(x.reshape(x.shape[:-1] + (HC, HC)), REPEAT, HC_EPS)


def prenorm(rows, pre, w, TN: int, EPS: float) -> np.ndarray:
    """`MhcPreNorm_v1` on the HC streams' f32 words ``rows`` [T, HC, H] (broadcastable), pre [T, HC], w [H] bf16 -> [T, H] bf16."""
    pre = _u(pre)
    T, HC, H = pre.shape[0], pre.shape[1], np.shape(rows)[-1]
    rows = np.broadcast_to(_u(rows), (T, HC, H))
    M._check_norm(H, TN)
    ol = np.zeros((T, H), dtype=U32)
    for j in range(HC):
        ol = fma(pre[:, j, None], rows[:, j, :], ol)
    blocks = ol.reshape(T, H // (16 * TN), 2, TN, 8)
    ss = np.zeros((T, 2, TN, 8), dtype=U32)
    for b in range(blocks.shape[1]):
        ss = fma(blocks[:, b], blocks[:, b], ss)
    thread = chain(np.stack([ss[:, rv & 1, :, rv >> 1] for rv in range(16)], axis=-1))
    rn = rsqrt(add(div(tree(thread), word(H)), word(EPS)))
    return bf16(mul(mul(cvt(bf16(ol)), rn[:, None]), cvt(w)))


def prenorm_broadcast(x, pre, w, TN: int, EPS: float) -> np.ndarray:
    """`MhcPreNormBroadcast_v1`: x [T, H] bf16."""
    return prenorm(cvt(x)[:, None, :], pre, w, TN, EPS)


def big_fuse(yp, rp, scale, base, residual, w, TN: int, REPEAT: int, EPS: float, HC_EPS: float, broadcast: bool = False):
    """`MhcPreBigFuse_v1` (residual [T, HC, H] bf16) or `MhcPreBigFuseBroadcast_v1` (residual the embedding [T, H] bf16):
    (layer input [T, H] bf16, post, comb)."""
    HC = _hc(_u(base).shape[0])
    H = _u(residual).shape[-1]
    pre, post, comb = mixes(yp, rp, scale, base, HC, H if broadcast else HC * H, REPEAT, EPS, HC_EPS)
    out = prenorm_broadcast(residual, pre, w, TN, EPS) if broadcast else prenorm(cvt(residual), pre, w, TN, EPS)
    return out, post, comb


# --- the post and the fused split (<= 16 tokens) -------------------------------------------------------------------------------

def new_residual(xf, rf, post, comb) -> np.ndarray:
    """`mhc.new_residual` at every element: xf [T, N], rf [T, HC, N] f32 words, post [T, HC], comb [T, HC, HC] -> [T, HC, N]."""
    xf, rf, post, comb = _u(xf), _u(rf), _u(post), _u(comb)
    acc = fma(post[:, :, None], xf[:, None, :], mul(comb[:, 0, :, None], rf[:, 0, None, :]))
    for k in range(1, rf.shape[1]):
        acc = fma(comb[:, k, :, None], rf[:, k, None, :], acc)
    return acc


def post(x, residual, post_mix, comb) -> np.ndarray:
    """`MhcPost_v1{HC, N}`: x [T, N] bf16, residual [T, HC, N] bf16 -> residual' [T, HC, N] bf16."""
    return bf16(new_residual(cvt(x), cvt(residual), post_mix, comb))


def _warp_reduce(v: np.ndarray, TH: int) -> np.ndarray:
    lanes = min(32, TH)
    return chain(tree(v.reshape(v.shape[:-1] + (TH // lanes, lanes))))


def fused_split(x, residual, post_mix, comb, fn, TH: int):
    """`MhcFusedSplit_v1{HC, N, OUT, TH}`: x [T, N], residual [T, HC, N] bf16, fn [OUT, HC, N] f32 -> (residual' [T, HC, N] bf16,
    yp [T, OUT], rp [T])."""
    fn = _u(fn)
    T, HC, N = np.shape(residual)
    OUT = fn.shape[0]
    M._check_split(N, TH)
    nr = new_residual(cvt(x), cvt(residual), post_mix, comb)
    steps = N // TH
    nrr = nr.reshape(T, HC, steps, TH)
    fnr = fn.reshape(OUT, HC, steps, TH)
    sqr = np.zeros((T, TH), dtype=U32)
    acc = np.zeros((T, OUT, TH), dtype=U32)
    for it in range(steps):
        for j in range(HC):
            sqr = fma(nrr[:, j, it], nrr[:, j, it], sqr)
        for j in range(HC):
            acc = fma(fnr[None, :, j, it], nrr[:, None, j, it], acc)
    return bf16(nr), _warp_reduce(acc, TH), _warp_reduce(sqr, TH)


def fused_partials(x, residual, post_mix, comb, fn, NS: int, TH: int):
    """`mhc_fused_tilelang`'s NS splits of H: (residual' [T, HC, H] bf16, yp [T, NS, OUT], rp [T, NS])."""
    T, HC, H = np.shape(residual)
    if H % NS:
        raise ValueError(f"mhc_fused splits H evenly (H={H}, NS={NS})")
    N, fn3 = H // NS, _u(fn).reshape(-1, HC, H)
    parts = [fused_split(np.asarray(x)[:, N * s:N * s + N], np.asarray(residual)[:, :, N * s:N * s + N], post_mix, comb,
                         fn3[:, :, N * s:N * s + N], TH) for s in range(NS)]
    res = np.concatenate([p[0] for p in parts], axis=-1)
    return res, np.stack([p[1] for p in parts], axis=1), np.stack([p[2] for p in parts], axis=1)


def fused_post_pre(x, residual, post_mix, comb, fn, scale, base, w, NS: int, TH: int, TN: int, REPEAT: int, EPS: float,
                   HC_EPS: float):
    """`MhcFusedPostPre_v1`: x [T, H], residual [T, HC, H] bf16, fn [OUT, HC * H] -> (residual', layer input, post', comb')."""
    res, yp, rp = fused_partials(x, residual, post_mix, comb, fn, NS, TH)
    out, post2, comb2 = big_fuse(yp, rp, scale, base, res, w, TN, REPEAT, EPS, HC_EPS)
    return res, out, post2, comb2


# --- the TF32 path ----------------------------------------------------------------------------------------------------------------

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


def gpu_add(a, b) -> np.ndarray:
    """GPU FADD: `add`, with every NaN result canonicalised to 0x7FFFFFFF."""
    out = add(a, b)
    return np.where(_is_nan(out), U32(F32_NAN_PTX), out).astype(U32)


def gpu_mul(a, b) -> np.ndarray:
    """GPU FMUL: `mul`, with every NaN result canonicalised to 0x7FFFFFFF."""
    out = mul(a, b)
    return np.where(_is_nan(out), U32(F32_NAN_PTX), out).astype(U32)


def gpu_fma(a, b, c) -> np.ndarray:
    """GPU FFMA: `fma`, with every NaN result canonicalised to 0x7FFFFFFF."""
    out = fma(a, b, c)
    return np.where(_is_nan(out), U32(F32_NAN_PTX), out).astype(U32)


def _sqr_term(lo: np.ndarray, hi: np.ndarray, sqr: str) -> np.ndarray:
    if sqr == "lo":
        return gpu_fma(lo, lo, gpu_mul(hi, hi))
    if sqr == "hi":
        return gpu_fma(hi, hi, gpu_mul(lo, lo))
    if sqr == "mul":
        return gpu_add(gpu_mul(lo, lo), gpu_mul(hi, hi))
    raise ValueError(f"the squared sum's contraction: one of {SQR_FORMS} (got {sqr!r})")


def _sqrsum(x: np.ndarray, sqr: str) -> np.ndarray:
    """The quad's squared sum over x [..., K] f32 words: four chains v[e] from +0 of `x[k]^2 + x[k+4]^2` contracted as ``sqr``
    (k = 8 i + e), and `(v0 + v2) + (v1 + v3)`."""
    g = x.reshape(x.shape[:-1] + (x.shape[-1] // 8, 8))
    v = np.zeros(x.shape[:-1] + (4,), dtype=U32)
    for i in range(g.shape[-2]):
        v = gpu_add(v, _sqr_term(g[..., i, :4], g[..., i, 4:], sqr))
    return tree(v, gpu_add)


def _tf32_splits(x: np.ndarray, b: np.ndarray, ranges, model: GroupSum, b_round: str):
    """yp [T, NS, OUT] of the splits ``ranges`` over x [T, K] f32 words and b [OUT, K]: equal-length splits at once."""
    T, OUT = x.shape[0], b.shape[0]
    yp = np.zeros((T, len(ranges), OUT), dtype=U32)
    for n in sorted({n for _, n in ranges}):
        idx = [s for s, (_, m) in enumerate(ranges) if m == n]
        xs = np.stack([x[:, ranges[s][0]:ranges[s][0] + n] for s in idx], axis=1)
        bs = np.stack([b[:, ranges[s][0]:ranges[s][0] + n] for s in idx], axis=0)
        av, bv = np.broadcast_arrays(xs[:, :, None, :], bs[None, :, :, :])
        acc = np.zeros(av.shape[:-1], dtype=U32).reshape(-1)
        for k in range(0, n, 8):
            acc = tf32_dot8(acc, av[..., k:k + 8], bv[..., k:k + 8], model, b_round=b_round)
        yp[:, idx, :] = acc.reshape(T, len(idx), OUT)
    return yp


def tf32_split(a, b, model: GroupSum = CANDIDATE, b_round: str = "rne", sqr: str = "lo"):
    """`MhcPrenormTf32Split_v1{K, OUT}`: a [T, K] bf16, b [OUT, K] f32 -> (yp [T, OUT], rp [T])."""
    x, b = cvt(a), _u(b)
    M._check_tf32(x.shape[-1])
    return _tf32_splits(x, b, [(0, x.shape[-1])], model, b_round)[:, 0], _sqrsum(x, sqr)


def prenorm_sqrsum(a, NS: int, sqr: str = "lo") -> np.ndarray:
    """The squared sums of `MhcPrenormTf32_v1{K, NS, OUT}` alone: a [T, K] bf16 -> rp [T, NS]."""
    x = cvt(a)
    return np.stack([_sqrsum(x[:, o:o + n], sqr) for o, n in M.tf32_ranges(x.shape[-1], NS)], axis=1)


def prenorm_tf32(a, b, NS: int, model: GroupSum = CANDIDATE, b_round: str = "rne", sqr: str = "lo"):
    """`MhcPrenormTf32_v1{K, NS, OUT}`: a [T, K] bf16, b [OUT, K] f32 -> (yp [T, NS, OUT], rp [T, NS])."""
    x, b = cvt(a), _u(b)
    return _tf32_splits(x, b, M.tf32_ranges(x.shape[-1], NS), model, b_round), prenorm_sqrsum(a, NS, sqr)


def post_pre_tf32(x, residual, post_mix, comb, fn, scale, base, w, NS: int, TN: int, REPEAT: int, EPS: float, HC_EPS: float,
                  **tf32):
    """`MhcPostPreTf32_v1`: `post`, `prenorm_tf32` over the streams it wrote, `big_fuse`."""
    res = post(x, residual, post_mix, comb)
    yp, rp = prenorm_tf32(res.reshape(res.shape[0], -1), fn, NS, **tf32)
    out, post2, comb2 = big_fuse(yp, rp, scale, base, res, w, TN, REPEAT, EPS, HC_EPS)
    return res, out, post2, comb2


def pre_broadcast(x, fn, scale, base, w, NS: int, TN: int, REPEAT: int, EPS: float, HC_EPS: float, **tf32):
    """`MhcPreBroadcast_v1`: x [T, H] bf16, fn [OUT, H] -> (layer input, post, comb)."""
    yp, rp = prenorm_tf32(x, fn, NS, **tf32)
    return big_fuse(yp, rp, scale, base, x, w, TN, REPEAT, EPS, HC_EPS, broadcast=True)


# --- hc_head ---------------------------------------------------------------------------------------------------------------------

def hc_head(residual, fn, scale, base, TH: int, EPS: float, HC_EPS: float) -> np.ndarray:
    """`HcHead_v1{HC, H, TH, EPS, HC_EPS}`: residual [T, HC, H] bf16, fn [HC, HC * H] f32 -> [T, H] bf16."""
    T, HC, H = np.shape(residual)
    M._check_head(H, TH)
    fn, scale, base = _u(fn), _u(scale), _u(base)
    xs = cvt(residual)
    nb = H // (8 * TH)
    xr = xs.reshape(T, HC, nb, TH, 8)
    fr = fn.reshape(HC, HC, nb, TH, 8)
    sqr = np.zeros((T, TH), dtype=U32)
    mix = np.zeros((T, HC, TH), dtype=U32)
    for mc in range(HC):
        for ih in range(nb):
            for e in range(8):
                sqr = fma(xr[:, mc, ih, :, e], xr[:, mc, ih, :, e], sqr)
            for e in range(8):
                mix = fma(xr[:, None, mc, ih, :, e], fr[None, :, mc, ih, :, e], mix)
    rs = rsqrt(add(div(tree(sqr), word(HC * H)), word(EPS)))
    pre = add(sigmoid(fma(mul(tree(mix), rs[:, None]), scale[0], base[None, :])), word(HC_EPS))
    out = np.zeros((T, H), dtype=U32)
    for k in range(HC):
        out = fma(pre[:, k, None], xs[:, k, :], out)
    return bf16(out)


# --- the served split counts --------------------------------------------------------------------------------------------------

def compute_num_split(block_k: int, k: int, grid: int, sms: int = 132) -> int:
    """vLLM's `compute_num_split` (mhc/tilelang_kernels.py:19) on a device of ``sms`` SMs (the H200's 132)."""
    return max(min(sms // grid, -(-k // block_k) // 4), 1)


def tf32_num_split(T: int, K: int, sms: int = 132) -> int:
    """The split count `tf32_hc_prenorm_gemm` runs at T tokens over K (block 64 x 64): `compute_num_split(64, K, cdiv(T, 64))`."""
    return compute_num_split(64, K, -(-T // 64), sms)
