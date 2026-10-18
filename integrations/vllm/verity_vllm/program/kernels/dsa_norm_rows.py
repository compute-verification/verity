"""The replay row kernel of `DsaRMSNorm_v1{N, EPS}` (`registry.dsa_norm`, `_fused_norm_rope_kernel`'s `_rms_norm`) and its vectorised
twin, `dsa_rmsnorm_rows`: the registry body over rows, each step `derived_rows`' word function of its primitive.  The row kernel
registers from `rows` (kernel "rows")."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from verity_vllm.program.kernels import derived_rows as DR

FAMILY = "DsaRMSNorm_v1"
EVALUATOR = ("verity_vllm.program.kernels.dsa_norm_rows.dsa_rmsnorm_rows (DsaRMSNorm_v1{N,EPS}: per thread of 128 an FMA chain over "
             "its dsa_norm.layout registers from mul(x1, x1), warp xor-butterflies, (w0 + w2) + (w1 + w3), r = RsqrtApprox(eps + "
             "DivFullScaleA(sum, N) * DivFullRcp(N)), y = F2fpBf16((r * x) * w) -- the fused norm + RoPE kernel's _rms_norm)")


def dsa_rmsnorm_rows(x: np.ndarray, w: np.ndarray, eps: float) -> dict[str, np.ndarray]:
    """`DsaRMSNorm_v1{N, EPS=eps}` over rows: x (R, N) bf16 words, w (N,) or (R, N) bf16 words.  Returns the bf16 ``out`` (R, N) and
    the f32 words ``sum``, ``mean`` and ``r`` (each (R,))."""
    from verity_vllm.program.registry import dsa_norm as DN
    from verity_vllm.program.registry.prims import bits_of

    x = np.ascontiguousarray(x, dtype=np.uint16)
    R, N = x.shape
    DN.check_statics(N, eps)
    w2 = np.broadcast_to(np.asarray(w, dtype=np.uint16), (R, N))
    xf = DR.bf16_to_f32_words(x)
    regs = xf[:, np.asarray(DN.layout(N))]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        s = DR.fma_rn(regs[..., 0], regs[..., 0], DR.mul_rn(regs[..., 1], regs[..., 1]))
        for k in range(2, regs.shape[-1]):
            s = DR.fma_rn(regs[..., k], regs[..., k], s)
        v = s.reshape(R, DN.THREADS // DN.WARP, DN.WARP)
        while v.shape[-1] > 1:
            h = v.shape[-1] // 2
            v = DR.add_rn(v[..., :h], v[..., h:])
        w4 = v[..., 0]
        total = DR.add_rn(DR.add_rn(w4[:, 0], w4[:, 2]), DR.add_rn(w4[:, 1], w4[:, 3]))
        n = np.full(R, bits_of(np.float32(N)), dtype=np.uint32)
        mean = DR.mul_rn(DR.div_full_scale_a(total, n), DR.div_full_rcp(n))
        r = DR.rsqrt_approx(DR.add_rn(np.full(R, bits_of(np.float32(eps)), dtype=np.uint32), mean))
        y = DR.mul_rn(DR.mul_rn(r[:, None], xf), DR.bf16_to_f32_words(w2))
    return {"out": DR.f2fp_bf16(y), "sum": total, "mean": mean, "r": r}


def row(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """The row kernel: statics as the spec text renders them (`ir_rows.f64_statics` reads an f64 EPS literal)."""
    from verity_vllm.program.kernels.ir_rows import f64_statics

    statics = f64_statics(statics)
    N = int(statics["N"])
    r = dsa_rmsnorm_rows(np.asarray(ops[0], dtype=np.uint16).reshape(1, N), np.asarray(ops[1], dtype=np.uint16).reshape(N),
                         float(statics["EPS"]))
    return {"out": r["out"].reshape(N)}


def check_targets(rng: np.random.Generator) -> list:
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsa_norm as DN

    return [bind(DN.DsaRMSNorm, N=256, EPS=1e-5), bind(DN.DsaRMSNorm, N=2048, EPS=1e-6)]
