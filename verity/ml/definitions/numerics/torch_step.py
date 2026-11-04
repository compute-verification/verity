"""numpy models of `registry.torch_step`'s own nodes, as `program_walk` calls them (`program_walk.Model`): F.linear's three
GEMMs along their ROUTE static (`bf16_gemm_check.cublas_route_rows`), `torch_train`'s transpose, RMSNorm
forward, backward and weight gradient and SiLU forward and backward, the residual stream's bf16 add and multiply, and the
optimizer's widening, AdamW step and bf16 copy, each in its kernel's order on bf16 (u16) and f32 (u32) words, with
`torch_kernels`' NaNs and roundings.

`MODELS` are these by family. `STEP_MODELS` adds `torch_blocks.MODELS`, so it models every node of `TrainStep_v1`'s body
that is past the walk's bit budget, and a GEMM's shared weight is read once rather than once per token.
`tests/test_torch_step.py` holds each model to its Definition's bits, and the norm and SiLU models to the GPU's words."""
from __future__ import annotations

from functools import cache
from typing import Any, Callable

import numpy as np

from verity.ml.circuits.ops import mufu as MU
from verity.ml.definitions.registry import bf16_gemm_check as GC
from verity.ml.definitions.registry import torch_train as TT

from . import torch_blocks as TB
from . import torch_kernels as TK

__all__ = ["MODELS", "STEP_MODELS", "adamw", "linear", "rms_backward", "rms_forward", "row_sum", "silu", "silu_backward",
           "weight_grad"]

U32 = np.uint32
ONE = U32(0x3F800000)
TWO = U32(0x40000000)
NEG_HALF = U32(0xBF000000)


def _family(f: Any) -> str:
    return getattr(f, "base_id", None) or f.id


def linear(route: dict[str, Any], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One of F.linear's GEMMs along ``route`` (`cublas_bf16.route_gemm`, `cublas_bf16.train_route` at the root's shapes): each
    row of a against the rows of b; bf16 words (a's rows, b's rows)."""
    return GC.cublas_route_rows(dict(route), a, b)


def row_sum(p: np.ndarray, lanes: int) -> np.ndarray:
    """`Reduce.cuh`'s vectorized sum of each row of p (R, H) of f32 words at ``lanes`` threads (`torch_train.row_sum`)."""
    R, H = p.shape
    sums = []
    for lane in range(lanes):
        acc = [np.zeros(R, U32) for _ in range(4)]
        for v in range(lane, H // 4, lanes):
            acc = [TK.add(a, p[:, 4 * v + i]) for i, a in enumerate(acc)]
        s = acc[0]
        for a in acc[1:]:
            s = TK.add(s, a)
        sums.append(s)
    off = lanes // 2
    while off:
        sums = [TK.add(sums[i], sums[i + off]) for i in range(off)]
        off //= 2
    return sums[0]


def rms_forward(x: np.ndarray, w: np.ndarray, lanes: int, eps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`RmsNormTorch_v1` with RSQRT `mufu.RsqrtApprox_v2` on rows (R, H) of bf16 x and w: (y, hb, r)."""
    x32 = TK.f32(x)
    s = row_sum(TK.mul(x32, x32), lanes)
    v = TK.add(TK.mul(s, U32(TT.mean_factor(x.shape[1]))), U32(eps))
    r = MU.rsqrt_approx_words(np.asarray(v, U32), MU.table("rsq")).astype(U32)
    hb = TK.bf16(TK.mul(x32, r[:, None]))
    return TK.bf16_mul(w, hb), hb, r


def rms_backward(dy: np.ndarray, x: np.ndarray, w: np.ndarray, r: np.ndarray, lanes: int) -> np.ndarray:
    """`RmsNormTorchBackward_v1` on rows (R, H) of bf16 dy, x, w and the saved f32 r (R,): dx (R, H)."""
    H = x.shape[1]
    x32, r = TK.f32(x), np.asarray(r, U32)[:, None]
    g = TK.f32(TK.bf16_mul(dy, w))
    ga = TK.mul(g, r)
    gr = row_sum(TK.mul(g, x32), lanes)[:, None]
    gv = TK.mul(TK.mul(gr, NEG_HALF), TK.mul(TK.mul(r, r), r))
    gp = TK.mul(gv, U32(np.float32(1.0 / H).view(U32)))
    return TK.bf16(TK.add(ga, TK.mul(gp, TK.mul(x32, TWO))))


def weight_grad(dy: np.ndarray, hb: np.ndarray, M: int, H: int) -> np.ndarray:
    """`RmsNormTorchWeightGrad_v1{M, H}` on columns (C, M) of bf16 dy and hb: each column's gradient word (C,)."""
    p = TK.f32(TK.bf16_mul(dy, hb))
    ctas, stride = TT.column_ctas(M, H), 4 * TT.column_ctas(M, H)
    zero = np.zeros(p.shape[0], U32)

    def block_y(ys):
        return TK.add(TK.add(ys[0], ys[2]), TK.add(ys[1], ys[3]))

    parts = []
    for c in range(ctas):
        ys = []
        for y in range(4):
            idx, acc = y + 4 * c, [zero] * 4
            while idx + 3 * stride < M:
                acc = [TK.add(a, p[:, idx + i * stride]) for i, a in enumerate(acc)]
                idx += 4 * stride
            for i in range(4):
                if idx >= M:
                    break
                acc[i] = TK.add(acc[i], p[:, idx])
                idx += stride
            ys.append(TK.add(TK.add(TK.add(acc[0], acc[1]), acc[2]), acc[3]))
        parts.append(block_y(ys))
    if ctas == 1:
        return TK.bf16(parts[0])
    ys = []
    for y in range(4):
        v = zero
        for j in range(y, ctas, 4):
            v = TK.add(v, parts[j])
        ys.append(v)
    return TK.bf16(block_y(ys))


@cache
def _silu_tables() -> dict[str, np.ndarray]:
    """Per bf16 word x: s = RN(1 / RN(1 + expf(-x))), RN(1 - s), and silu's bf16 word; expf CUDA's ``__nv_expf`` on sm_120's
    MUFU.EX2 (`moe_twins.nv_expf_bits`)."""
    from .moe_twins import nv_expf_bits

    x = np.arange(1 << 16, dtype=U32)
    e = np.array([nv_expf_bits(int(v) << 16 ^ 0x80000000) for v in x], U32)
    d = TK.add(ONE, e)
    s = TK.div(ONE, d)
    return {"s": s, "oms": TK.add(ONE, TK.neg(s)), "silu": TK.bf16(TK.div(TK.f32(x), d))}


def silu(x: np.ndarray) -> np.ndarray:
    """`SiluTorch_v1` with EX2 `mufu.MufuEx2Ftz_v2` on bf16 words."""
    return _silu_tables()["silu"][np.asarray(x, np.int64)]


def silu_backward(dy: np.ndarray, x: np.ndarray, contract: bool) -> np.ndarray:
    """`SiluTorchBackward_v1` with EX2 `mufu.MufuEx2Ftz_v2` on bf16 words: bf16(RN(RN(dy s) t)), t = 1 + x RN(1 - s), one FMA
    when ``contract``."""
    T, i = _silu_tables(), np.asarray(x, np.int64)
    x32, s, oms = TK.f32(x), T["s"][i], T["oms"][i]
    t = TK.fma(x32, oms, ONE) if contract else TK.add(ONE, TK.mul(x32, oms))
    return TK.bf16(TK.mul(TK.mul(TK.f32(dy), s), t))


def adamw(p: np.ndarray, g: np.ndarray, m: np.ndarray, v: np.ndarray, k: np.ndarray,
          lerp_fma: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`torch_optim.adamw_element` on f32 words, runs (R, N) of p, g, m and v with each run's seven scalars k (R, 7):
    (p2, m1, v2)."""
    a, w, b2, c2, s2, e, st = (np.asarray(k, U32)[:, i:i + 1] for i in range(7))
    p1 = TK.mul(p, a)
    d = TK.add(g, TK.neg(m))
    om = TK.add(ONE, TK.neg(w))
    small = (w & U32(0x7FFFFFFF)) < U32(0x3F000000)
    if lerp_fma:
        m1 = np.where(small, TK.fma(w, d, m), TK.fma(TK.neg(d), om, g))
    else:
        m1 = np.where(small, TK.add(m, TK.mul(w, d)), TK.add(g, TK.neg(TK.mul(d, om))))
    one = c2 == ONE
    v2 = TK.fma(np.where(one, g, c2), np.where(one, g, TK.mul(g, g)), TK.mul(v, b2))
    dd = TK.add(TK.div(TK.sqrt(v2), s2), e)
    return TK.fma(st, TK.div(m1, dd), p1), m1, v2


def _adamw(S: dict, p, g, m, v, k):
    out = adamw(p, g, m, v, k, bool(S["LERP_FMA"]))
    return (*out, TK.bf16(out[0])) if S["BF16"] else out


def _require(S: dict, key: str, family: str) -> None:
    if _family(S[key]) != family:
        raise ValueError(f"torch_step model: {key} = {_family(S[key])}, modelled only for {family}")


def _rms(S: dict, x, w):
    _require(S, "RSQRT", "RsqrtApprox_v2")
    return rms_forward(x, w, int(S["LANES"]), int(S["EPS"]))


def _silu(S: dict, x):
    _require(S, "EX2", "MufuEx2Ftz_v2")
    return silu(x)


def _silu_backward(S: dict, dy, x):
    _require(S, "EX2", "MufuEx2Ftz_v2")
    return silu_backward(dy, x, bool(S["CONTRACT"]))


def _linear(S: dict, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack([linear(S["ROUTE"], a[m], b[m]) for m in range(a.shape[0])])


MODELS: dict[str, Callable[..., Any]] = {
    "LinearTorch_v1": _linear,
    "LinearTorchDx_v1": _linear,
    "LinearTorchDw_v1": _linear,
    "Transpose_v1": lambda S, x: np.ascontiguousarray(np.swapaxes(x, 1, 2)),
    "RmsNormTorch_v1": _rms,
    "RmsNormTorchBackward_v1": lambda S, dy, x, w, r: rms_backward(dy, x, w, r, int(S["LANES"])),
    "RmsNormTorchWeightGrad_v1": lambda S, dy, hb: weight_grad(dy, hb, int(S["M"]), int(S["H"])),
    "SiluTorch_v1": _silu,
    "SiluTorchBackward_v1": _silu_backward,
    "Bf16AddF2fp_v2": lambda S, a, b: TK.bf16_add(a, b),
    "Bf16MulBf16_v2": lambda S, a, b: TK.bf16_mul(a, b),
    "Bf16ToF32_v2": lambda S, a: TK.f32(a),
    "F2fpBf16_v2": lambda S, a: TK.bf16(a),
    "AdamWForeach_v1": _adamw,
}
STEP_MODELS: dict[str, Callable[..., Any]] = {**TB.MODELS, **MODELS}
