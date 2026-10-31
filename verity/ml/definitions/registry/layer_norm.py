"""`aten.layer_norm` / `aten.native_layer_norm` on a bf16 `[M, N]` row with bf16 `weight` and `bias` — torch 2.13.0+cu129
`aten/src/ATen/native/cuda/layer_norm_kernel.cu`, the **vectorized** path `vectorized_layer_norm_kernel<c10::BFloat16, float>`
(taken when N <= 2^24, N is a multiple of `vec_size = 4` -- a constexpr of the file, "we could make it dependent on dtype, but
that would lead to different results between float and low-p types" -- and X / Y / gamma / beta are `vec_size * sizeof(T)` =
8-byte aligned, which every contiguous bf16 activation and parameter is): one CTA of `num_threads() = C10_WARP_SIZE * 4 = 128`
threads (`dim3(32, 4)`) per row.

Statistics (`compute_stats`), all f32, no fast-math (IEEE `div.rn.f32` for every `1.f / count`, no FTZ):

    thread t (= threadIdx.x + 32 * threadIdx.y, 0..127) reads the 4-element vectors i = t, t + 128, ... (< N / 4) and folds each
    element in order into a Welford triple (mean, sigma2, count) with `cuWelfordOnlineSum`:
        delta = x - mean;  count += 1;  new_mean = mean + delta * (1.f / count)  [= fma(delta, rcp, mean), rcp IEEE: no fast-math]
        sigma2 = sigma2 + delta * (x - new_mean)                                  [= fma(delta, x - new_mean, sigma2)]
    intra-warp: for offset in 16, 8, 4, 2, 1:  wd = cuWelfordCombine(dataB = wd, dataA = shfl_down(wd, offset))
    inter-warp (blockDim.y = 4): warps 2, 3 publish, warps 0, 1 combine (w0 <- w2, w1 <- w3); warp 1 publishes, warp 0 combines
        cuWelfordCombine(B, A): count = cA + cB; coef = 1.f / count; nA = cA * coef; nB = cB * coef;
            mean = nA * mA + nB * mB;  sigma2 = (sA + sB) + ((delta * delta) * cA) * nB,  delta = mB - mA   (count == 0 -> zeros)
    var = sigma2 / N (IEEE);  rstd = rsqrtf(var + eps)  (`c10::cuda::compat::rsqrt` = MUFU.RSQ with range scaling, `RsqrtApprox`)

Output (T_ACC = float, the store is `c10::BFloat16(float)` = `__float2bfloat16`, RN, NaN -> 0x7FFF):
    y_j = bf16( gamma_j * (rstd * (x_j - mean)) + beta_j )       -- FADD, FMUL, FFMA (nvcc -fmad contracts the last product)

The counts are static (a function of N alone), so every `1.f / count`, `cA * coef`, `cB * coef` is a compile-time exact f32
constant of the body (the kernel computes them at run time with the same IEEE operations; the values coincide bit for bit).
Where a thread holds no elements its triple is (0, 0, 0) and `cuWelfordCombine` folds it through the same arithmetic (nA = 0,
nB = cB * (1.f / cB), which is exactly 1 only for power-of-two counts); the body writes the operations out and lets the
constants decide.  For Pythia (N = H = 768) every thread holds 8 or 4 elements (192 four-wide vectors over 128 threads: threads
0..63 read vectors t and t + 128), so the lane counts are 8 and 4 and the tree's coefficients `cA / count` are inexact from the
first combine on -- the contraction order below is therefore observable.

FMA contraction (nvcc default `-fmad=true`, no `--use_fast_math`): `nA*mA + nB*mB` is `fma(nA, mA, nB*mB)` (the first product
fused -- `MEAN_FMA_ORDER = "A"`, pinned on the pod: 0/184 f32 mean-word mismatches vs `aten.native_layer_norm`'s returned mean
over N in {64, 768, 2304}, against 48/184 for the other order); `sA + sB + delta*delta*cA*nB` is `fma((delta*delta)*cA, nB, sA + sB)`
(`SIGMA_FMA = True`; the two contractions of the sigma2 term were not distinguishable through `rsqrt.approx` on the probe's inputs,
the fused form is the compiler's canonical one).  Its difftest (`layer_norm_difftest.py`) is the bit-exact check.
"""
from __future__ import annotations

import numpy as np

from verity.ml.circuits.defs import composite
from verity.ml.circuits.refs import Coll, concat

from verity.ml.circuits.types import Array
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.b1 import ZERO32, f32c
from verity.ml.definitions.registry.prims import BF16, F32

SEMANTIC_ID = "ln/LayerNormAten@v1"

VEC = 4            # elements per vector load (`constexpr int vec_size = 4`, independent of dtype)
WARP = 32
NWARPS = 4
NTHREADS = WARP * NWARPS

#: `mean = nA * mA + nB * mB` contracted as fma(nA, mA, nB * mB) ("A", measured) or fma(nB, mB, nA * mA) ("B", excluded)
MEAN_FMA_ORDER = "A"
#: `sigma2 = sA + sB + delta * delta * cA * nB` with the last product contracted into the add (True) or not (False)
SIGMA_FMA = True


def _f32(x: float) -> np.float32:
    return np.float32(x)


class _Stat:
    """A Welford triple during body construction: mean / sigma2 are gate refs (Coll), count is a static int."""

    __slots__ = ("mean", "sigma2", "count")

    def __init__(self, mean, sigma2, count: int) -> None:
        self.mean, self.sigma2, self.count = mean, sigma2, count


def _sub(B, a, b):
    """IEEE f32 `a - b` as fma(b, -1, a): the product is exact, one rounding -> identical to FADD a, -b."""
    return B.call(P.F32Fma, b, B.call(f32c(-1.0)), a)


def _online(B, st: _Stat, v) -> _Stat:
    """`cuWelfordOnlineSum(val, curr)`."""
    count = st.count + 1
    rcp = B.call(f32c(float(_f32(1.0) / _f32(count))))          # `1.f / new_count`, IEEE, a static constant here
    delta = _sub(B, v, st.mean)
    new_mean = B.call(P.F32Fma, delta, rcp, st.mean)             # mean + delta * rcp
    t = _sub(B, v, new_mean)
    sigma2 = B.call(P.F32Fma, delta, t, st.sigma2)               # sigma2 + delta * (val - new_mean)
    return _Stat(new_mean, sigma2, count)


def _combine(B, zero, sB: _Stat, sA: _Stat) -> _Stat:
    """`cuWelfordCombine(dataB, dataA)` (note the argument order of the kernel: self first, partner second)."""
    count = sA.count + sB.count
    if count == 0:
        return _Stat(zero, zero, 0)
    coef = _f32(1.0) / _f32(count)
    nA = B.call(f32c(float(_f32(sA.count) * coef)))
    nB = B.call(f32c(float(_f32(sB.count) * coef)))
    delta = _sub(B, sB.mean, sA.mean)
    if MEAN_FMA_ORDER == "A":
        mean = B.call(P.F32Fma, nA, sA.mean, B.call(P.F32Mul, nB, sB.mean))
    else:
        mean = B.call(P.F32Fma, nB, sB.mean, B.call(P.F32Mul, nA, sA.mean))
    s = B.call(P.F32Add, sA.sigma2, sB.sigma2)
    ddc = B.call(P.F32Mul, B.call(P.F32Mul, delta, delta), B.call(f32c(float(sA.count))))
    if SIGMA_FMA:
        sigma2 = B.call(P.F32Fma, ddc, nB, s)
    else:
        sigma2 = B.call(P.F32Add, s, B.call(P.F32Mul, ddc, nB))
    return _Stat(mean, sigma2, count)


def _stats(B, xf, N: int) -> tuple:
    """(mean, var) gate refs of one row: `compute_stats` for `N` f32 values held as `xf[j]`."""
    zero = B.call(ZERO32)
    nvec = N // VEC
    threads: list[_Stat] = []
    for t in range(NTHREADS):
        st = _Stat(zero, zero, 0)
        for i in range(t, nvec, NTHREADS):
            for j in range(VEC):
                st = _online(B, st, xf[VEC * i + j])
        threads.append(st)
    warps: list[_Stat] = []
    for w in range(NWARPS):
        lanes = threads[WARP * w:WARP * (w + 1)]
        for offset in (16, 8, 4, 2, 1):
            # lane i combines with lane i + offset (shfl_down; lanes without a partner keep a value nobody reads)
            lanes = [_combine(B, zero, lanes[i], lanes[i + offset]) if i + offset < WARP else lanes[i] for i in range(WARP)]
        warps.append(lanes[0])
    # inter-warp: offset 2 (w0 <- w2, w1 <- w3), then offset 1 (w0 <- w1)
    w0 = _combine(B, zero, warps[0], warps[2])
    w1 = _combine(B, zero, warps[1], warps[3])
    top = _combine(B, zero, w0, w1)
    var = B.call(P.F32Div, top.sigma2, B.call(f32c(float(N))))
    return top.mean, var


@composite("LayerNormAten", 1, ["N", "EPS"],
           lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16)), ("b", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="quarantine (op-level evidence only): difftest vs torch.nn.functional.layer_norm (bf16, sm_89) on the pod")
def LayerNormAten(B, S, x, w, b, eps=None):
    """One row of torch's vectorized CUDA LayerNorm (see the module docstring): Welford statistics over the 128-thread /
    4-warp reduction tree, `rstd = rsqrtf(var + eps)`, `y_j = bf16(w_j * (rstd * (x_j - mean)) + b_j)`."""
    N = S.N
    xf = B.batch(P.Bf16ToF32, x, axes=(0,))
    mean, var = _stats(B, xf, N)
    rstd = B.call(P.RsqrtApprox, B.call(P.F32Add, var, B.call(f32c(float(_f32(S.EPS)))) if eps is None else eps))
    wf = B.batch(P.Bf16ToF32, w, axes=(0,))
    bf = B.batch(P.Bf16ToF32, b, axes=(0,))
    outs = []
    for j in range(N):
        d = _sub(B, xf[j], mean)
        t = B.call(P.F32Mul, rstd, d)
        outs.append(B.call(P.F2fpBf16, B.call(P.F32Fma, wf[j], t, bf[j])))
    return Coll(Array(N, BF16), concat([o.refs for o in outs]))


@composite("LayerNormAten", 2, ["N"],
           lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16)), ("b", Array(S.N, BF16)), ("eps", F32)), Array(S.N, BF16)),
           conformance="quarantine (op-level evidence only): LayerNormAten_v1's body with eps an input")
def LayerNormAtenV2(B, S, x, w, b, eps):
    """`LayerNormAten_v1{N, EPS}` with the epsilon an input (the f32 word `vectorized_layer_norm_kernel` reads)."""
    return LayerNormAten.body(B, S, x, w, b, eps)
