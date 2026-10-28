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

`TritonBlockRMSNorm_v1{N, BLOCK, WARPS, EPS}` is the same lowering of a `tl.sum` RMSNorm over a masked block: a row of N live elements
in a power-of-two BLOCK on 32 WARPS threads, the rest loaded as +0 (`tl.load(..., mask=block < SIZE, other=0.0)`), the mean
`div.full(sum, N)`; `boolean_dsa_norm.TritonBlockRMSNorm_v2` on bits.  It is vLLM d9105ea80's `_fused_q_kv_rmsnorm_kernel`
(`models/common/ops/fused_qk_rmsnorm.py`, DeepSeek-V4's q and kv norms: one CTA per (token, program), program 0 q at SIZE = Q_SIZE,
program 1 kv at KV_SIZE, BLOCK = next_pow2 of the larger, 8 warps from BLOCK 2048, else 4), as Triton 3.7.1 compiles it for sm_90
at the H200's specialisation (pointers and strides divisible by 16; the PTX at Q_SIZE 1536, KV_SIZE 512: `.reqntid 256`, thread t
`ld.global.v4.b32` of elements 8t..8t+7 under the one predicate 8t < SIZE, the chain above over them, the five-level butterfly, the
8 warp sums butterflied 4, 2, 1 (a halving tree), `cvt.rn.f32.u32` of SIZE and `div.full.f32`, `add.f32 eps, mean`,
`rsqrt.approx.ftz.f32`, `mul.f32 r, x`, `mul.f32 (.), w`, `cvt.rn.bf16x2.f32`; at BLOCK 1024 the same on 128 threads, at BLOCK 512
and 256 on 128 threads `v2.b32` and `b32` loads of 4 and 2 elements and the same chain, at BLOCK 4096 a thread's two runs 2048 apart,
each under its own predicate; `launch_pdl` adds `griddepcontrol` only).  ptxas contracts `div.full`'s last multiply into that
`add.f32` (no `.rn`, so it may): the SASS is `MUFU.RCP` of the scaled SIZE, then `FFMA rcp, scaled sum, eps`, then `MUFU.RSQ`, so the
operand of rsqrt is `fma(DivFullScaleA(sum, N), DivFullRcp(N), eps)`, rounded once.  At a power-of-two N the product is exact and
the two roundings are that value; at DeepSeek-V4's q (N = 1536) they are not, and the H200's committed rows are the fma's.  At BLOCK
128 on 4 warps a thread holds one element and the first butterfly level is contracted, as above, so BLOCK is at least two elements a
thread.  A thread wholly past N sums +0 (`fma(+0, +0, +0 * +0)`), so the body takes its lane as the constant; a masked element of a
live thread stays in its chain.  `DsaRMSNorm_v1{N, EPS}` computes `TritonBlockRMSNorm_v1{N, N, 4, EPS}`'s words, its N a power of two.
"""
from __future__ import annotations

import numpy as np

from verity.ml.circuits.defs import composite
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.ml.definitions.evidence import EVIDENCE
from verity.ml.definitions.registry import dsa as D
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.prims import BF16

THREADS, WARP = 128, 32
_CONF = ("model-untested: read from the Triton kernel and its PTX at the commits the module docstring pins; "
         "tests/properties/glm53_fused_rmsnorm_capture_gpu.py is its hardware check")
_BLOCK_CONF = ("model-untested: read from vLLM d9105ea80's _fused_q_kv_rmsnorm_kernel and the cc 9.0 PTX and SASS Triton 3.7.1 compiles "
               f"from it (the module docstring); the q rows (N 1536) Commit {EVIDENCE['dsv4_q_norm_commit']['run']} committed on cc 9.0 "
               "are its words "
               "(tests/properties/test_dsv4_q_norm_commit.py), with no probe launch yet")


def check_eps(name: str, EPS: float) -> None:
    if not float(np.float32(EPS)) >= 2.0 ** -126:
        raise ValueError(f"{name}: EPS {EPS} below 2^-126 (rsqrt.approx.ftz and RsqrtApprox agree on operands from there)")


def check_statics(N: int, EPS: float) -> None:
    if not (N >= 2 * THREADS and N & (N - 1) == 0):
        raise ValueError(f"DsaRMSNorm: the 128-thread layout takes a power-of-two row of at least 256 (N={N})")
    check_eps("DsaRMSNorm", EPS)


def check_block(N: int, BLOCK: int, WARPS: int, EPS: float) -> None:
    threads = WARP * WARPS
    if not (WARPS > 0 and WARPS & (WARPS - 1) == 0 and BLOCK >= 2 * threads and BLOCK & (BLOCK - 1) == 0):
        raise ValueError(f"TritonBlockRMSNorm: a power-of-two BLOCK of at least two elements a thread on a power-of-two warp count "
                         f"(BLOCK={BLOCK}, WARPS={WARPS})")
    if not (0 < N <= BLOCK and N % min(BLOCK // threads, 8) == 0):
        raise ValueError(f"TritonBlockRMSNorm: the row fills whole vectors of the block (N={N}, BLOCK={BLOCK}, WARPS={WARPS})")
    check_eps("TritonBlockRMSNorm", EPS)


def layout(N: int, threads: int = THREADS) -> list[list[int]]:
    """Thread t's elements in register order: E = N / threads of them, repeats of V = min(E, 8) contiguous ones threads * V apart."""
    E = N // threads
    V = min(E, 8)
    return [[(k // V) * threads * V + V * t + k % V for k in range(E)] for t in range(threads)]


def rms_norm(o: D.Ops, S, x, w, to_bf16, t=BF16, threads: int = THREADS, block: int | None = None, *,
             fused_eps: bool = False) -> Coll:
    """`DsaRMSNorm`'s body over ``o``'s steps (the module docstring's sequence), each output word a Call of ``to_bf16``; on ``threads``
    over a ``block`` (default N) whose elements from N are +0; ``fused_eps`` takes rsqrt of `fma(scale_a(sum), rcp(N), eps)` (the
    SASS) rather than of `eps + mul(scale_a(sum), rcp(N))` (the PTX)."""
    if block is None:
        check_statics(S.N, S.EPS)
    xf = [o.cvt(x[i]) for i in range(S.N)]
    zeros: list = []

    def zero():
        """The +0 a masked load gives: one constant, made at its first use."""
        if not zeros:
            zeros.append(o.word(0))
        return zeros[0]
    lanes = []
    for regs in layout(block or S.N, threads):
        if regs[0] >= S.N:
            lanes.append(None)
            continue
        v = [xf[i] if i < S.N else zero() for i in regs]
        s = o.fma(v[0], v[0], o.mul(v[1], v[1]))
        for e in v[2:]:
            s = o.fma(e, e, s)
        lanes.append(s)

    def add(a, b):
        return None if a is None and b is None else o.add(zero() if a is None else a, zero() if b is None else b)
    total = D.halving_tree([D.halving_tree(lanes[WARP * g:WARP * g + WARP], add) for g in range(threads // WARP)], add)
    n = o.word(D.f32_bits(S.N))
    if fused_eps:
        r = o.rsqrt(o.fma(o.scale_a(total, n), o.rcp(n), o.word(D.f32_bits(S.EPS))))
    else:
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


def block_rms_norm(o: D.Ops, S, x, w, to_bf16, t=BF16) -> Coll:
    """`TritonBlockRMSNorm`'s body: `rms_norm` on 32 WARPS threads over BLOCK, eps fused into the mean."""
    check_block(S.N, S.BLOCK, S.WARPS, S.EPS)
    return rms_norm(o, S, x, w, to_bf16, t, threads=WARP * S.WARPS, block=S.BLOCK, fused_eps=True)


@composite("TritonBlockRMSNorm", 1, ["N", "BLOCK", "WARPS", "EPS"],
           lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)), conformance=_BLOCK_CONF)
def TritonBlockRMSNorm(B, S, x, w):
    """One row of a Triton `tl.sum` RMSNorm over a masked block (the module docstring has the PTX): N live bf16 elements of a BLOCK on
    32 WARPS threads, each thread's FMA chain over its `layout` registers (a masked one +0), each warp's butterfly, the warp sums'
    halving tree; `r = rsqrt.approx.ftz(fma(DivFullScaleA(sum, N), DivFullRcp(N), eps))`, ptxas's contraction of `eps + div.full(sum,
    N)`; `y = bf16(r * x * f32(w))`, the products in that order."""
    return block_rms_norm(D.word_ops(B), S, x, w, P.F2fpBf16)


DSA_NORM_COMPOSITES = [DsaRMSNorm, TritonBlockRMSNorm]
