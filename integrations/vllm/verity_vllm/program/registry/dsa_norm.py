"""[GLM-5.3] The MLA path's two RMSNorms as `_fused_norm_rope_kernel` runs them: program 2's q_a_layernorm over q_c [2048] and
program 1's kv_a_layernorm over kv_c [512] (vLLM d9105ea80 `deepseek_v32/common/kernels.py`: `_rms_norm` at 25-31, called at 184-196
and 210-219, kv_c stored at 240 and 304; one CTA of `num_warps=4` per (token, program), grid (tokens, 4), kernels.py:539-607).
"The PTX" is Triton 3.7.1's sm_90 output for the specialisation the H200 runs (every pointer and stride divisible by 16, so each
thread's loads are 16- or 8-byte vectors; `_fused_norm_rope_kernel.a3d95dc37831.ptx`, lines 124-418 for program 2 and 446-710 for
program 1).

The kernel's arithmetic, `.reqntid 128` (thread t, warp t // 32, lane t % 32), N = Q_DIM = Q_BLOCK_SIZE or KV_DIM:

    layout  thread t holds E = N / 128 elements, R = E / V repeats of V = min(E, 8) contiguous ones (`ld.global.v4.b32` at N = 2048,
            `v2.b32` at N = 512); register k is element (k // V) * 128V + Vt + k % V, the registers in that order
    s_t     = fma.rn(x_0, x_0, mul.f32(x_1, x_1)), then s_t = fma.rn(x_k, x_k, s_t) for k = 2 .. E-1   (`tl.sum(x * x)`: each square
            contracted into the running add but the first)
    warp    xor-butterfly 16, 8, 4, 2, 1 of add.f32 (`shfl.sync.bfly`), lane 0 to shared memory
    sum     threads 0-3 xor 2 then xor 1: (w_0 + w_2) + (w_1 + w_3)
    mean    div.full.f32(sum, N)                                      0f45000000 at N = 2048, 0f44000000 at N = 512
    r       rsqrt.approx.ftz.f32(add.f32(eps, mean))                  eps the f32 kernel argument
    y_i     cvt.rn.bf16x2.f32(mul.f32(mul.f32(r, x_i), cvt.f32.bf16(w_i)))

where x_i = cvt.f32.bf16 of the row's bf16 word.  It is not `b1.RMSNormTriton_v1` (vLLM's batch-invariant `_rms_norm_kernel`): that
sums 1024-column chunks into +0 one at a time, eight columns a thread, and divides 1 by `sqrt.approx.ftz(fma(rcp(N), sum, eps))`;
here one 128-thread tree takes every column, a thread's chain spans its repeats, and the epilogue is rsqrt of `eps + sum / N`.

`div.full.f32` is `DivFullScaleA` times `DivFullRcp` (as `dsa.k_row`'s mean), and `rsqrt.approx.ftz` is `RsqrtApprox_v1` on an
operand of at least eps >= 2^-126 (`dsa.check_eps`'s reason).  kv_cache_dtype "auto" stores the same bf16 words in the MLA cache.

Registered: `DsaRMSNorm_v1{N, EPS}` (x: N bf16, w: N bf16) -> N bf16, one token's row; `boolean_dsa_norm.DsaRMSNorm_v2` on bits.  N is
a power of two from 256: at N = 128 a thread holds one element and the compiler contracts the first butterfly level instead (as
`dsa.DsaIndexerKRow_v1`'s variance), which this body does not state.
"""
from __future__ import annotations

import numpy as np

from verity.ml.circuits.defs import composite
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity_vllm.program.registry import dsa as D
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.prims import BF16

THREADS, WARP = 128, 32
_CONF = ("model-untested: read from the Triton kernel and its PTX at the commits the module docstring pins; "
         "tests/properties/glm53_fused_rmsnorm_capture_gpu.py is its hardware check")


def check_statics(N: int, EPS: float) -> None:
    if not (N >= 2 * THREADS and N & (N - 1) == 0):
        raise ValueError(f"DsaRMSNorm: the 128-thread layout takes a power-of-two row of at least 256 (N={N})")
    if not float(np.float32(EPS)) >= 2.0 ** -126:
        raise ValueError(f"DsaRMSNorm: EPS {EPS} below 2^-126 (rsqrt.approx.ftz and RsqrtApprox agree on operands from there)")


def layout(N: int) -> list[list[int]]:
    """Thread t's elements in register order: E = N / 128 of them, repeats of V = min(E, 8) contiguous ones 128V apart."""
    E = N // THREADS
    V = min(E, 8)
    return [[(k // V) * THREADS * V + V * t + k % V for k in range(E)] for t in range(THREADS)]


def rms_norm(o: D.Ops, S, x, w, to_bf16, t=BF16) -> Coll:
    """`DsaRMSNorm`'s body over ``o``'s steps (the module docstring's sequence), each output word a Call of ``to_bf16``."""
    check_statics(S.N, S.EPS)
    xf = [o.cvt(x[i]) for i in range(S.N)]
    lanes = []
    for regs in layout(S.N):
        s = o.fma(xf[regs[0]], xf[regs[0]], o.mul(xf[regs[1]], xf[regs[1]]))
        for i in regs[2:]:
            s = o.fma(xf[i], xf[i], s)
        lanes.append(s)
    total = D.halving_tree([D.halving_tree(lanes[WARP * g:WARP * g + WARP], o.add) for g in range(THREADS // WARP)], o.add)
    n = o.word(D.f32_bits(S.N))
    mean = o.mul(o.scale_a(total, n), o.rcp(n))
    r = o.rsqrt(o.add(o.word(D.f32_bits(S.EPS)), mean))
    y = [o.B.call(to_bf16, o.mul(o.mul(r, xf[i]), o.cvt(w[i]))) for i in range(S.N)]
    return Coll(Array(S.N, t), concat([v.refs for v in y]))


@composite("DsaRMSNorm", 1, ["N", "EPS"], lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance=_CONF)
def DsaRMSNorm(B, S, x, w):
    """One token's row of `_fused_norm_rope_kernel`'s `_rms_norm` (programs 1 and 2, the module docstring has the PTX): x and w bf16,
    the sum of squares each thread's FMA chain over its `layout` registers, each warp's butterfly, then `(w0 + w2) + (w1 + w3)`;
    `r = rsqrt.approx.ftz(eps + div.full(sum, N))`; `y = bf16(r * x * f32(w))`, the products in that order."""
    return rms_norm(D.word_ops(B), S, x, w, P.F2fpBf16)


DSA_NORM_COMPOSITES = [DsaRMSNorm]
