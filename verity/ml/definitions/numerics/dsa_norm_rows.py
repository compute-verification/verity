"""The replay row kernel of `DsaRMSNorm_v1{N, EPS}` (`registry.dsa_norm`, `_fused_norm_rope_kernel`'s `_rms_norm`) and its vectorised
twin, `dsa_rmsnorm_rows`: the registry body over rows, each step `derived_rows`' word function of its primitive.  `block_rmsnorm_rows`
is the same body over a masked block, `TritonBlockRMSNorm_v1{N, BLOCK, WARPS, EPS}` (whose row kernel is `dsv4_rows`').  The row
kernel registers from `rows` (kernel "rows")."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from verity.ml.definitions.numerics import derived_rows as DR

FAMILY = "DsaRMSNorm_v1"
EVALUATOR = ("verity.ml.definitions.numerics.dsa_norm_rows.dsa_rmsnorm_rows (DsaRMSNorm_v1{N,EPS}: per thread of 128 an FMA chain over "
             "its dsa_norm.layout registers from mul(x1, x1), warp xor-butterflies, (w0 + w2) + (w1 + w3), r = RsqrtApprox(eps + "
             "DivFullScaleA(sum, N) * DivFullRcp(N)), y = F2fpBf16((r * x) * w) -- the fused norm + RoPE kernel's _rms_norm)")


def dsa_rmsnorm_rows(x: np.ndarray, w: np.ndarray, eps: float) -> dict[str, np.ndarray]:
    """`DsaRMSNorm_v1{N, EPS=eps}` over rows: x (R, N) bf16 words, w (N,) or (R, N) bf16 words.  Returns the bf16 ``out`` (R, N) and
    the f32 words ``sum``, ``mean`` and ``r`` (each (R,))."""
    from verity.ml.definitions.registry import dsa_norm as DN

    N = np.shape(x)[-1]
    DN.check_statics(N, eps)
    return block_rmsnorm_rows(x, w, eps, N, DN.THREADS // DN.WARP, fused_eps=False)


def block_rmsnorm_rows(x: np.ndarray, w: np.ndarray, eps: float, block: int, warps: int, *, fused_eps: bool = True
                       ) -> dict[str, np.ndarray]:
    """`TritonBlockRMSNorm_v1{N, BLOCK=block, WARPS=warps, EPS=eps}` over rows, `dsa_rmsnorm_rows`' outputs: the block's elements
    from N load +0, so a thread wholly past N sums +0 (no lane's sum is -0, so its +0 in the trees is the Definition's absent lane).
    ``fused_eps`` as `dsa_norm.rms_norm`'s: ``mean`` is then the fma `scale_a(sum) * rcp(N) + eps` that rsqrt takes."""
    from verity.ml.definitions.registry import dsa_norm as DN
    from verity.ml.definitions.registry.prims import bits_of

    x = np.ascontiguousarray(x, dtype=np.uint16)
    R, N = x.shape
    DN.check_block(N, block, warps, eps)
    w2 = np.broadcast_to(np.asarray(w, dtype=np.uint16), (R, N))
    xf = DR.bf16_to_f32_words(x)
    regs = np.pad(xf, ((0, 0), (0, block - N)))[:, np.asarray(DN.layout(block, DN.WARP * warps))]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        s = DR.fma_rn(regs[..., 0], regs[..., 0], DR.mul_rn(regs[..., 1], regs[..., 1]))
        for k in range(2, regs.shape[-1]):
            s = DR.fma_rn(regs[..., k], regs[..., k], s)
        v = s.reshape(R, warps, DN.WARP)
        while v.shape[-1] > 1:
            h = v.shape[-1] // 2
            v = DR.add_rn(v[..., :h], v[..., h:])
        v = v[..., 0]
        while v.shape[-1] > 1:
            h = v.shape[-1] // 2
            v = DR.add_rn(v[..., :h], v[..., h:])
        total = v[:, 0]
        n = np.full(R, bits_of(np.float32(N)), dtype=np.uint32)
        e = np.full(R, bits_of(np.float32(eps)), dtype=np.uint32)
        if fused_eps:
            mean = DR.fma_rn(DR.div_full_scale_a(total, n), DR.div_full_rcp(n), e)
            r = DR.rsqrt_approx(mean)
        else:
            mean = DR.mul_rn(DR.div_full_scale_a(total, n), DR.div_full_rcp(n))
            r = DR.rsqrt_approx(DR.add_rn(e, mean))
        y = DR.mul_rn(DR.mul_rn(r[:, None], xf), DR.bf16_to_f32_words(w2))
    return {"out": DR.f2fp_bf16(y), "sum": total, "mean": mean, "r": r}


def row(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """The row kernel: statics as the spec text renders them (`ir_rows.f64_statics` reads an f64 EPS literal)."""
    from verity.ml.definitions.numerics.ir_rows import f64_statics

    statics = f64_statics(statics)
    N = int(statics["N"])
    r = dsa_rmsnorm_rows(np.asarray(ops[0], dtype=np.uint16).reshape(1, N), np.asarray(ops[1], dtype=np.uint16).reshape(N),
                         float(statics["EPS"]))
    return {"out": r["out"].reshape(N)}


def check_targets(rng: np.random.Generator) -> list:
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import dsa_norm as DN

    return [bind(DN.DsaRMSNorm, N=256, EPS=1e-5), bind(DN.DsaRMSNorm, N=2048, EPS=1e-6)]
