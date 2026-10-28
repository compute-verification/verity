"""`verity_int8::row_quant` and `verity_int8::scaled_mm` on CUDA (`engine.int8_ops` registers `row_quant` and `product`):
`quant_rows`, one Triton kernel bit for bit `int8_exact.quantize_rows`, and `scaled_mm`, one Triton GEMM whose epilogue is
`int8_exact.dequantize`, so the exact linear is two launches a call (stock vLLM int8's count) instead of the torch bodies'
roughly fourteen.  Both write into caller-owned buffers, allocate nothing and are graph-capturable once compiled.  Built and
gated in `benchmarks/hash/exactlin` (kgate.py: bit-exact against int8_exact on the served shapes; replay.py: against the
Definitions' replay evaluators).

For x [M, K] bf16, W [N, K] int8 and s_w [N] float32 (`engine.int8_exact`'s statement):

    a[i]   = max_k |f32(x[i, k])|                                  exact (a max)
    s_a[i] = RN32(max(a[i], RN32(1e-30)) / 127)                    div.rn.f32
    X[i,k] = clamp(rint(RN32(f32(x[i, k]) / s_a[i])), -127, 127)   div.rn.f32, then cvt.rni (round half to even)
    Y      = X W^T in int32                                        tl.dot, s8 x s8 -> s32, exact for K <= MAX_K
    y[i,n] = RN_bf16(RN32(RN32(f32(Y[i,n]) * s_a[i]) * s_w[n]))    cvt.rn.f32.s32, two mul.rn, cvt.rn.bf16

Triton's `/` on float32 is div.full.f32 (approximate) and its libdevice calls are flush-to-zero by default, so the kernels
use tl.div_rn, an inline cvt.rni, enable_reflect_ftz=False and enable_fp_fusion=False; `ptx_check` reads the compiled PTX
for exactly these instructions.  The quantizer divides each element by its row's reciprocal and two explicit fma.rn
corrections, div.rn's own fast path, which equals div.rn.f32 on every finite bf16 input (`test_int8_ops`' exhaustive test).  The GEMM is vLLM's `triton_scaled_mm` (the stock arm's kernel) on the pointer path, without
bias, with the same epilogue order (activation scale first); its config per architecture, shape and M is
`engine.profiles.int8_gemm_configs.CONFIGS` (ksweep.py's gated winners), else vLLM's tiles.  A split-K config adds int32 partials (exact in any order) into a
self-zeroing `Workspace`, and the last split applies the epilogue.  A TMA config runs `exactlin_scaled_mm_tma`, the same
product and epilogue with X and W read through host tensor descriptors (kernel parameters, so a graph captures them;
out-of-bounds boxes read zeros), optionally warp-specialized.  The product needs no row
padding: a row of X W^T reads only that row, so M < 17 is just a masked tile (int8_exact pads only because torch._int_mm
refuses M <= 16).  Non-finite activations are outside the statement and not gated.  The quantizer runs `quant_plan`'s
warps and rows a program (`engine.profiles.int8_quant_plans.QUANT_PLANS`, else 16 warps a row); each row's arithmetic is
the same in any plan.

`scaled_mm_zy` (`verity_int8::scaled_mm_zy`'s CUDA body, `product_zy`) is `scaled_mm` at the same config whose epilogue also
writes `int8_exact`'s z/y records from the same int32 accumulator: the y it computes once is both the served y and the
record's, and its grid covers the call's T slots, a row tile at or past M multiplying nothing and writing only its empty
slots' records (z 0, y dq(0, 0, s_w)).  Split-K sums into the same workspace before the one epilogue.

The fused producers (`engine.int8_producers` installs them) write a linear's input and its quantizer's codes and scales in
one launch: `silu_mul_quant` is vLLM's `_C.silu_and_mul` then `quant_rows`, and `add_norm_quant` is vLLM's batch-invariant
`_C.fused_add_rms_norm` (its generic kernel's per-thread sums, warp shuffle tree and warp-aggregate order at min(H, 1024)
threads, `norm_threads`) then `quant_rows`, each bit for bit (`tests/engine/test_int8_producers.py`), at
`producer_warps`' warps.  The silu's act is `_SILU_MUL2`, inline PTX on pairs of elements: written in Triton (libdevice's
expf, tl.div_rn, a scalar bf16 rounding twice an element) it issued about 52 instructions an element, five of them on the
quarter-rate unit, and ran short of HBM's rate; written out it issues about 43, three of them there.
"""
from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from verity_vllm.engine.int8_exact import UNIT, ZY_BYTES, zy_units
from verity_vllm.engine.profiles.int8_gemm_configs import CONFIGS
from verity_vllm.engine.profiles.int8_quant_plans import PRODUCER_WARPS, QUANT_PLANS

OPTS = {"enable_fp_fusion": False, "enable_reflect_ftz": False}


@triton.jit
def _rni_s32(v):
    return tl.inline_asm_elementwise("cvt.rni.s32.f32 $0, $1;", "=r,r", [v], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _membar_gl(v):
    return tl.inline_asm_elementwise("membar.gl;\n\tmov.b32 $0, $1;", "=r,r", [v], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _quant_tile(x, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """x [BLOCK_M, BLOCK_K] f32 (a bf16 row's values, 0 past its end) -> (X int8, s f32 [BLOCK_M]), `quantize_rows` a row."""
    a = tl.max(tl.abs(x), axis=1)
    s = tl.div_rn(tl.maximum(a, 1e-30), 127.0)
    # x / s == (x * 2^e) / (s * 2^e) with s * 2^e in [1, 2): s is normal, so both products are exact (x * 2^e unless it is
    # below 2^-126, where the quotient is too and X is 0 either way), and the rounded quotient is the same.  It is divided
    # as div.rn's fast path does, with the reciprocal taken once a row: y = RN(1/s1), q0 = x y, and two corrections
    # q <- fma(fma(-s1, q, x), y, q); the first leaves q within an ulp, so the second is RN(x / s1) (Markstein's theorem),
    # and a quotient too small to be exact rounds to X = 0 either way.
    sb = s.to(tl.int32, bitcast=True)
    s1 = ((sb & 0x7FFFFF) | 0x3F800000).to(tl.float32, bitcast=True)
    p = ((254 - (sb >> 23)) << 23).to(tl.float32, bitcast=True)
    y = tl.div_rn(tl.full(s1.shape, 1.0, tl.float32), s1)
    xs = x * p[:, None]
    ns = tl.broadcast_to(-s1[:, None], (BLOCK_M, BLOCK_K))
    yb = tl.broadcast_to(y[:, None], (BLOCK_M, BLOCK_K))
    q = xs * yb
    q = tl.fma(tl.fma(ns, q, xs), yb, q)
    q = tl.fma(tl.fma(ns, q, xs), yb, q)
    return tl.minimum(tl.maximum(_rni_s32(q), -127), 127).to(tl.int8), s


@triton.jit
def exactlin_quant_rows(x_ptr, q_ptr, s_ptr, M, K, stride_x, stride_q, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    # BLOCK_M rows a program, each row's arithmetic the same as alone (a max is exact in any order)
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs = tl.arange(0, BLOCK_K)
    live = rows < M
    mask = live[:, None] & (offs < K)[None, :]
    x = tl.load(x_ptr + rows[:, None] * stride_x + offs[None, :], mask=mask, other=0.0).to(tl.float32)
    q, s = _quant_tile(x, BLOCK_M, BLOCK_K)
    tl.store(q_ptr + rows[:, None] * stride_q + offs[None, :], q, mask=mask)
    tl.store(s_ptr + rows, s, mask=live)


@triton.jit
def _rsqrt_approx(v):
    return tl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=r,r", [v], dtype=tl.float32, is_pure=True, pack=1)


# vLLM's silu-and-mul on one pair of a row's elements, $1 its gate and $2 its up as bf16x2 words (the even element in the
# low half), $0 the pair of h: bf16(g / (1.0f + expf(-g * alpha))) at alpha 1, then bf16(that * (u + beta)) at beta 0, where
# u + 0.0f only turns -0 into +0.  expf is libdevice's __nv_expf step for step, with -g's sign moved into the constants it
# multiplies (exact: each product is an fma's), the division is div.rn, and each pair's two roundings to bf16 are one
# cvt.rn.bf16x2.f32 (an ALU instruction; the scalar cvt.rn.bf16.f32 is F2F, on the quarter-rate unit).
_SILU_MUL2 = tl.constexpr("""{
.reg .b32 g<2>, u<2>, j<2>, a;
.reg .f32 x<2>, t<2>, r<2>, e<2>, y<2>, d<2>, q<2>, v<2>, m<2>;
shl.b32 g0, $1, 16;
and.b32 g1, $1, 0xFFFF0000;
mov.b32 x0, g0;
mov.b32 x1, g1;
fma.rn.f32 t0, x0, 0fBBBB989D, 0f3F000000;
fma.rn.f32 t1, x1, 0fBBBB989D, 0f3F000000;
cvt.sat.f32.f32 t0, t0;
cvt.sat.f32.f32 t1, t1;
fma.rm.f32 t0, t0, 0f437C0000, 0f4B400001;
fma.rm.f32 t1, t1, 0f437C0000, 0f4B400001;
add.rn.f32 r0, t0, 0fCB40007F;
add.rn.f32 r1, t1, 0fCB40007F;
neg.f32 r0, r0;
neg.f32 r1, r1;
fma.rn.f32 r0, x0, 0fBFB8AA3B, r0;
fma.rn.f32 r1, x1, 0fBFB8AA3B, r1;
fma.rn.f32 r0, x0, 0fB2A57060, r0;
fma.rn.f32 r1, x1, 0fB2A57060, r1;
mov.b32 j0, t0;
mov.b32 j1, t1;
shl.b32 j0, j0, 23;
shl.b32 j1, j1, 23;
mov.b32 y0, j0;
mov.b32 y1, j1;
ex2.approx.ftz.f32 e0, r0;
ex2.approx.ftz.f32 e1, r1;
mul.rn.f32 e0, e0, y0;
mul.rn.f32 e1, e1, y1;
add.rn.f32 d0, e0, 0f3F800000;
add.rn.f32 d1, e1, 0f3F800000;
div.rn.f32 q0, x0, d0;
div.rn.f32 q1, x1, d1;
cvt.rn.bf16x2.f32 a, q1, q0;
shl.b32 j0, a, 16;
and.b32 j1, a, 0xFFFF0000;
mov.b32 q0, j0;
mov.b32 q1, j1;
shl.b32 u0, $2, 16;
and.b32 u1, $2, 0xFFFF0000;
mov.b32 v0, u0;
mov.b32 v1, u1;
add.rn.f32 v0, v0, 0f00000000;
add.rn.f32 v1, v1, 0f00000000;
mul.rn.f32 m0, q0, v0;
mul.rn.f32 m1, q1, v1;
cvt.rn.bf16x2.f32 $0, m1, m0;
}""")


@triton.jit
def _silu_mul2(g, u):
    return tl.inline_asm_elementwise(_SILU_MUL2, "=r,r,r", [g, u], dtype=tl.bfloat16, is_pure=True, pack=2)


@triton.jit
def exactlin_silu_mul_quant(gu_ptr, h_ptr, q_ptr, s_ptr, M, I, stride_gu, stride_h, stride_q, BLOCK_M: tl.constexpr,
                            BLOCK_K: tl.constexpr, EVEN: tl.constexpr):
    # vLLM's act_and_mul_kernel<bf16, silu_kernel> on each row's [gate | up], then exactlin_quant_rows on its bf16 output.
    # EVEN: I is BLOCK_K and M a multiple of BLOCK_M, so nothing is masked; a masked element's h is +0 (silu(+0) * (+0 + 0)),
    # which leaves its row's max as it was.
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs = tl.arange(0, BLOCK_K)
    g_ptrs = gu_ptr + rows[:, None] * stride_gu + offs[None, :]
    h_ptrs = h_ptr + rows[:, None] * stride_h + offs[None, :]
    q_ptrs = q_ptr + rows[:, None] * stride_q + offs[None, :]
    if EVEN:
        h = _silu_mul2(tl.load(g_ptrs), tl.load(g_ptrs + I))
        tl.store(h_ptrs, h)
        q, s = _quant_tile(h.to(tl.float32), BLOCK_M, BLOCK_K)
        tl.store(q_ptrs, q)
        tl.store(s_ptr + rows, s)
    else:
        live = rows < M
        mask = live[:, None] & (offs < I)[None, :]
        h = _silu_mul2(tl.load(g_ptrs, mask=mask, other=0.0), tl.load(g_ptrs + I, mask=mask, other=0.0))
        tl.store(h_ptrs, h, mask=mask)
        q, s = _quant_tile(h.to(tl.float32), BLOCK_M, BLOCK_K)
        tl.store(q_ptrs, q, mask=mask)
        tl.store(s_ptr + rows, s, mask=live)


@triton.jit
def exactlin_add_norm_quant(x_ptr, r_ptr, w_ptr, q_ptr, s_ptr, H, eps, stride_x, stride_r, stride_q, NT: tl.constexpr,
                            R: tl.constexpr):
    # vLLM's generic fused_add_rms_norm_kernel<bf16, 0, true> on one row at its NT = min(H, 1024) threads, in place (x <- the
    # normed row, r <- x + r), then exactlin_quant_rows on the normed row.  Thread t of the CUDA block owns elements t, t + NT,
    # ...: column t of this [R, NT] tile, so each step of the sum below is the one that thread, its warp or thread 0 takes.
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, R)[:, None] * NT + tl.arange(0, NT)[None, :]
    mask = k < H
    x = tl.load(x_ptr + row * stride_x + k, mask=mask, other=0.0)
    z = (x.to(tl.float32) + tl.load(r_ptr + row * stride_r + k, mask=mask, other=0.0).to(tl.float32)).to(tl.bfloat16)
    tl.store(r_ptr + row * stride_r + k, z, mask=mask)
    zf = z.to(tl.float32)
    # each thread's `variance += x * x` over its elements in order, contracted to an fma (picking element j out of the
    # column adds only +0, which leaves every square and sum as it was)
    var = tl.zeros((NT,), tl.float32)
    for j in tl.static_range(R):
        zj = tl.sum(tl.where(tl.arange(0, R)[:, None] == j, zf, 0.0), axis=0)
        var = tl.fma(zj, zj, var)
    # cub::BlockReduce<float, 1024>: each warp's WarpReduceShfl (lane i adds lane i + off, off = 1, 2, 4, 8, 16), then
    # thread 0 adds the warp aggregates in order from warp 0
    W: tl.constexpr = NT // 32
    lanes = tl.reshape(var, (W, 32))
    for step in tl.static_range(5):
        lanes = tl.sum(tl.reshape(lanes, (W, 16 >> step, 2)), axis=2)
    warps = tl.reshape(lanes, (W,))
    wid = tl.arange(0, W)
    total = tl.sum(tl.where(wid == 0, warps, 0.0), axis=0)
    for j in tl.static_range(1, W):
        total = total + tl.sum(tl.where(wid == j, warps, 0.0), axis=0)
    s_var = _rsqrt_approx(tl.div_rn(total, H.to(tl.float32)) + eps)
    w = tl.load(w_ptr + k, mask=mask, other=0.0).to(tl.float32)
    out = (zf * s_var * w).to(tl.bfloat16)
    tl.store(x_ptr + row * stride_x + k, out, mask=mask)
    q, s = _quant_tile(tl.reshape(tl.where(mask, out.to(tl.float32), 0.0), (1, R * NT)), 1, R * NT)
    tl.store(q_ptr + row * stride_q + tl.reshape(k, (1, R * NT)), q, mask=tl.reshape(mask, (1, R * NT)))
    tl.store(s_ptr + row + tl.arange(0, 1), s)


@triton.jit
def _epilogue(acc, sa_ptr, sb_ptr, c_ptr, offs_m, offs_n, mask_m, mask_n, stride_cm):
    sa = tl.load(sa_ptr + offs_m, mask=mask_m, other=0.0)
    sb = tl.load(sb_ptr + offs_n, mask=mask_n, other=0.0)
    y = acc.to(tl.float32) * sa[:, None]
    y = y * sb[None, :]
    tl.store(c_ptr + stride_cm * offs_m[:, None] + offs_n[None, :], y.to(tl.bfloat16),
             mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def exactlin_scaled_mm(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr, ws_ptr, cnt_ptr, M, N, K, stride_am, stride_bn, stride_cm,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    nk = tl.cdiv(K, BLOCK_K)
    per = tl.cdiv(nk, SPLIT_K)
    k_lo = tl.program_id(1) * per
    k_hi = tl.minimum(k_lo + per, nk)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_ptrs = a_ptr + stride_am * offs_m[:, None] + (k_lo * BLOCK_K + offs_k)[None, :]
    b_ptrs = b_ptr + (k_lo * BLOCK_K + offs_k)[:, None] + stride_bn * offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(k_lo, k_hi):
        mask_k = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    if SPLIT_K == 1:
        _epilogue(acc, sa_ptr, sb_ptr, c_ptr, offs_m, offs_n, mask_m, mask_n, stride_cm)
    else:
        # Integer partial sums add exactly in any order.  Each split adds its tile into the zeroed int32 workspace, and the
        # split that counts last reads the sum back while zeroing it, resets the tile's counter and applies the epilogue
        # once, so the workspace and counters are zero again for the next call (and the next graph replay).
        mask = mask_m[:, None] & mask_n[None, :]
        ws_ptrs = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.atomic_add(ws_ptrs, acc, mask=mask, sem="relaxed")
        _membar_gl(pid)
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel")
        if done == SPLIT_K - 1:
            _membar_gl(pid)
            tot = tl.atomic_xchg(ws_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32), mask=mask, sem="relaxed")
            tl.atomic_xchg(cnt_ptr + pid, 0, sem="relaxed")
            _epilogue(tot, sa_ptr, sb_ptr, c_ptr, offs_m, offs_n, mask_m, mask_n, stride_cm)


@triton.jit
def exactlin_scaled_mm_tma(a_desc, b_desc, sa_ptr, sb_ptr, c_ptr, M, N, K, stride_cm, BLOCK_M: tl.constexpr,
                           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, WARP_SPECIALIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    off_m = (pid // num_pid_n) * BLOCK_M
    off_n = (pid % num_pid_n) * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    # the descriptors' out-of-bounds boxes read zeros, so the tail tiles need no masks
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([off_m, k * BLOCK_K])
        b = b_desc.load([off_n, k * BLOCK_K])
        acc = tl.dot(a, b.T, acc, out_dtype=tl.int32)
    offs_m = off_m + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = off_n + tl.arange(0, BLOCK_N).to(tl.int64)
    _epilogue(acc, sa_ptr, sb_ptr, c_ptr, offs_m, offs_n, offs_m < M, offs_n < N, stride_cm)


@triton.jit
def _zy_epilogue(acc, sa_ptr, sb_ptr, c_ptr, z_ptr, yr_ptr, offs_m, offs_n, M, N, T, stride_cm, UR: tl.constexpr,
                 UC: tl.constexpr):
    mask_m = offs_m < M
    mask_n = offs_n < N
    # a masked split's atomic_xchg leaves rows past M undefined; they are empty slots, whose z is 0
    acc = tl.where(mask_m[:, None], acc, 0)
    sa = tl.load(sa_ptr + offs_m, mask=mask_m, other=0.0)
    sb = tl.load(sb_ptr + offs_n, mask=mask_n, other=0.0)
    y = acc.to(tl.float32) * sa[:, None]
    y = (y * sb[None, :]).to(tl.bfloat16)
    tl.store(c_ptr + stride_cm * offs_m[:, None] + offs_n[None, :], y, mask=mask_m[:, None] & mask_n[None, :])
    # record (i // UR) (N / UC) + j // UC: UR x UC int32 z, then UR x UC bf16 y, each row-major
    u = (offs_m[:, None] // UR) * (N // UC) + offs_n[None, :] // UC
    cell = (offs_m[:, None] % UR) * UC + offs_n[None, :] % UC
    mask = (offs_m < T)[:, None] & mask_n[None, :]
    tl.store(z_ptr + u * (UR * UC * 3 // 2) + cell, acc, mask=mask)
    tl.store(yr_ptr + u * (UR * UC * 3) + 2 * UR * UC + cell, y.to(tl.int16, bitcast=True), mask=mask)


@triton.jit
def exactlin_scaled_mm_zy(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr, z_ptr, yr_ptr, ws_ptr, cnt_ptr, M, N, K, T, stride_am,
                          stride_bn, stride_cm, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          SPLIT_K: tl.constexpr, UR: tl.constexpr, UC: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    live = pid_m * BLOCK_M < M
    nk = tl.cdiv(K, BLOCK_K)
    per = tl.cdiv(nk, SPLIT_K)
    k_lo = tl.program_id(1) * per
    k_hi = tl.where(live, tl.minimum(k_lo + per, nk), k_lo)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_ptrs = a_ptr + stride_am * offs_m[:, None] + (k_lo * BLOCK_K + offs_k)[None, :]
    b_ptrs = b_ptr + (k_lo * BLOCK_K + offs_k)[:, None] + stride_bn * offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(k_lo, k_hi):
        mask_k = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    if SPLIT_K == 1:
        _zy_epilogue(acc, sa_ptr, sb_ptr, c_ptr, z_ptr, yr_ptr, offs_m, offs_n, M, N, T, stride_cm, UR, UC)
    else:
        if live:
            # exactlin_scaled_mm's split-K, then the one epilogue
            mask = mask_m[:, None] & mask_n[None, :]
            ws_ptrs = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
            tl.atomic_add(ws_ptrs, acc, mask=mask, sem="relaxed")
            _membar_gl(pid)
            tl.debug_barrier()
            done = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel")
            if done == SPLIT_K - 1:
                _membar_gl(pid)
                tot = tl.atomic_xchg(ws_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32), mask=mask, sem="relaxed")
                tl.atomic_xchg(cnt_ptr + pid, 0, sem="relaxed")
                _zy_epilogue(tot, sa_ptr, sb_ptr, c_ptr, z_ptr, yr_ptr, offs_m, offs_n, M, N, T, stride_cm, UR, UC)
        elif tl.program_id(1) == 0:
            _zy_epilogue(acc, sa_ptr, sb_ptr, c_ptr, z_ptr, yr_ptr, offs_m, offs_n, M, N, T, stride_cm, UR, UC)


@triton.jit
def exactlin_scaled_mm_zy_tma(a_desc, b_desc, sa_ptr, sb_ptr, c_ptr, z_ptr, yr_ptr, M, N, K, T, stride_cm,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                              WARP_SPECIALIZE: tl.constexpr, UR: tl.constexpr, UC: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    off_m = (pid // num_pid_n) * BLOCK_M
    off_n = (pid % num_pid_n) * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    nk = tl.where(off_m < M, tl.cdiv(K, BLOCK_K), 0)
    for k in tl.range(0, nk, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([off_m, k * BLOCK_K])
        b = b_desc.load([off_n, k * BLOCK_K])
        acc = tl.dot(a, b.T, acc, out_dtype=tl.int32)
    offs_m = off_m + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = off_n + tl.arange(0, BLOCK_N).to(tl.int64)
    _zy_epilogue(acc, sa_ptr, sb_ptr, c_ptr, z_ptr, yr_ptr, offs_m, offs_n, M, N, T, stride_cm, UR, UC)


def quant_warps(k: int) -> int:
    return 16


def quant_plan(m: int, k: int, sm: str | None = None) -> tuple[int, int]:
    """(num_warps, rows a program) of the row quantizer at [m, k] on `sm` (default: the current device's): QUANT_PLANS'
    first plan for k whose M floor m reaches, else one row a program at `quant_warps`.  The plan never changes a value."""
    for floor, warps, rows in QUANT_PLANS.get(sm or arch(torch.cuda.current_device()), {}).get(k, ()):
        if m >= floor:
            return warps, rows
    return quant_warps(k), 1


def tiles(m: int, n: int) -> tuple[int, int, int]:
    """vLLM triton_scaled_mm's heuristic (the stock arm's tiles)."""
    p = max(32, triton.next_power_of_2(m))
    if p <= 32:
        return (64, 64, 256) if n < 8192 else (64, 128, 256)
    if p <= 64:
        return 64, 64, 256
    if p <= 128:
        return 64, 128, 128
    return 128, 128, 128


class Cfg(tuple):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, SPLIT_K, TMA); `Cfg.parse("32x64x256:w4:s3:k2")`.  TMA 1 is
    `exactlin_scaled_mm_tma` (host tensor descriptors, SPLIT_K 1), 2 the same warp-specialized, 0 the pointer kernel."""

    def __new__(cls, bm, bn, bk, warps=4, stages=3, split=1, tma=0):
        return super().__new__(cls, (bm, bn, bk, warps, stages, split, tma))

    @classmethod
    def parse(cls, text: str) -> Cfg:
        dims, *rest = text.split(":")
        kw = {"w": 4, "s": 3, "k": 1, "t": 0}
        for r in rest:
            kw[r[0]] = int(r[1:])
        return cls(*map(int, dims.split("x")), kw["w"], kw["s"], kw["k"], kw["t"])

    def __str__(self) -> str:
        bm, bn, bk, w, s, k, t = self
        return f"{bm}x{bn}x{bk}:w{w}:s{s}:k{k}" + (f":t{t}" if t else "")


def default_cfg(m: int, n: int, k: int) -> Cfg:
    return Cfg(*tiles(m, n))


@functools.cache
def arch(index: int | None = None) -> str:
    major, minor = torch.cuda.get_device_capability(index)
    return f"sm_{major}{minor}"


def pick(m: int, n: int, k: int, sm: str | None = None) -> Cfg:
    """The GEMM config for one call on `sm` (default: the current device's): CONFIGS' first (m_max, cfg) for (n, k)
    with m <= m_max, else vLLM's tiles."""
    for m_max, text in CONFIGS.get(sm or arch(torch.cuda.current_device()), {}).get((n, k), ()):
        if m <= m_max:
            return Cfg.parse(text)
    return default_cfg(m, n, k)


class Workspace:
    """The int32 partial sums and per-tile counters a split-K launch needs, zero between calls; one per device, sized
    once, so a call allocates nothing."""

    def __init__(self, device, words: int = 64 * 32768, tiles: int = 1 << 16):
        self.ws = torch.zeros(words, dtype=torch.int32, device=device)
        self.cnt = torch.zeros(tiles, dtype=torch.int32, device=device)


def quant_rows(x: torch.Tensor, xq: torch.Tensor, s: torch.Tensor, num_warps: int | None = None, rows: int | None = None):
    """x [M, K] bf16 (unit stride along K) -> xq [M, K] int8 and s [M] (or [M, 1]) float32, in place, at `quant_plan`'s
    plan unless `num_warps` or `rows` is given."""
    m, k = x.shape
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1, (x.dtype, x.stride())
    assert xq.dtype == torch.int8 and xq.shape == x.shape and xq.stride(1) == 1
    assert s.dtype == torch.float32 and s.numel() == m and s.is_contiguous()
    warps, r = quant_plan(m, k, arch(x.device.index))
    warps, r = num_warps or warps, rows or r
    return exactlin_quant_rows[(triton.cdiv(m, r),)](x, xq, s, m, k, x.stride(0), xq.stride(0), BLOCK_M=r,
                                                     BLOCK_K=triton.next_power_of_2(k), num_warps=warps, **OPTS)


#: num_warps of a fused producer called where `PRODUCER_WARPS` has no plan (the engine doesn't fuse there)
FUSE_FALLBACK = {"norm": 1, "silu": 16}


def producer_warps(kind: str, m: int, k: int, sm: str | None = None) -> int | None:
    """num_warps of the fused producer `kind` ("norm", "silu") on m rows of k on `sm` (default: the current device's):
    `PRODUCER_WARPS`' first plan for k whose M floor m reaches, where it is measured faster than vLLM's op and the
    quantizer apart, else None."""
    for floor, warps in PRODUCER_WARPS.get(sm or arch(), {}).get(kind, {}).get(k, ()):
        if m >= floor:
            return warps
    return None


def silu_mul_quant(gu: torch.Tensor, h: torch.Tensor, xq: torch.Tensor, s: torch.Tensor, num_warps: int | None = None):
    """gu [M, 2I] bf16 -> h [M, I] bf16, vLLM's `_C.silu_and_mul(h, gu)`, and its rows' `quant_rows` into xq [M, I] int8 and
    s [M] float32, in place in one launch (one row a program)."""
    m, i = h.shape
    assert gu.dtype == h.dtype == torch.bfloat16 and gu.shape == (m, 2 * i) and gu.stride(1) == 1 and h.stride(1) == 1
    assert xq.dtype == torch.int8 and xq.shape == h.shape and xq.stride(1) == 1
    assert s.dtype == torch.float32 and s.numel() == m and s.is_contiguous()
    warps = num_warps or producer_warps("silu", m, i, arch(h.device.index)) or FUSE_FALLBACK["silu"]
    bk = triton.next_power_of_2(i)
    return exactlin_silu_mul_quant[(m,)](gu, h, xq, s, m, i, gu.stride(0), h.stride(0), xq.stride(0), BLOCK_M=1,
                                         BLOCK_K=bk, EVEN=i == bk, num_warps=warps, **OPTS)


def norm_threads(h: int) -> int | None:
    """The thread count of vLLM's batch-invariant fused_add_rms_norm launch on rows of h, min(h, 1024), when
    `add_norm_quant` reproduces its sum (whole warps, a power of two), else None."""
    nt = min(h, 1024)
    return nt if nt >= 32 and nt & (nt - 1) == 0 else None


def add_norm_quant(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, xq: torch.Tensor,
                   s: torch.Tensor, num_warps: int | None = None):
    """vLLM's batch-invariant `_C.fused_add_rms_norm(x, residual, weight, eps)` on x, residual [M, H] bf16 (x <- the normed
    rows, residual <- x + residual) and the normed rows' `quant_rows` into xq [M, H] int8 and s [M] float32, in place in one
    launch (one row a program)."""
    m, h = x.shape
    nt = norm_threads(h)
    assert nt is not None, h
    assert x.dtype == residual.dtype == weight.dtype == torch.bfloat16 and residual.shape == x.shape and weight.shape == (h,)
    assert x.stride(1) == 1 and residual.stride(1) == 1 and weight.is_contiguous()
    assert xq.dtype == torch.int8 and xq.shape == x.shape and xq.stride(1) == 1
    assert s.dtype == torch.float32 and s.numel() == m and s.is_contiguous()
    warps = num_warps or producer_warps("norm", m, h, arch(x.device.index)) or FUSE_FALLBACK["norm"]
    return exactlin_add_norm_quant[(m,)](x, residual, weight, xq, s, h, eps, x.stride(0), residual.stride(0), xq.stride(0),
                                         NT=nt, R=triton.next_power_of_2(triton.cdiv(h, nt)), num_warps=warps, **OPTS)


def scaled_mm(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor,
              cfg: Cfg | None = None, work: Workspace | None = None):
    """xq [M, K] int8, s_a [M] f32, w [N, K] int8 (row-major), s_w [N] or [N, 1] f32 -> y [M, N] bf16, in place.  A
    split-K `cfg` needs `work`."""
    m, k = xq.shape
    n = w.shape[0]
    assert xq.dtype == torch.int8 and w.dtype == torch.int8 and w.shape[1] == k and xq.stride(1) == 1 and w.stride(1) == 1
    assert s_a.dtype == torch.float32 and s_a.numel() == m and s_a.is_contiguous()
    assert s_w.dtype == torch.float32 and s_w.numel() == n and s_w.is_contiguous()
    assert y.dtype == torch.bfloat16 and y.shape == (m, n) and y.stride(1) == 1
    bm, bn, bk, warps, stages, split, tma = cfg or default_cfg(m, n, k)
    tiles_mn = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    if tma:
        from triton.tools.tensor_descriptor import TensorDescriptor
        assert split == 1, cfg
        a_desc = TensorDescriptor.from_tensor(xq, [bm, bk])
        b_desc = TensorDescriptor.from_tensor(w, [bn, bk])
        return exactlin_scaled_mm_tma[(tiles_mn,)](a_desc, b_desc, s_a, s_w, y, m, n, k, y.stride(0), BLOCK_M=bm,
                                                   BLOCK_N=bn, BLOCK_K=bk, WARP_SPECIALIZE=tma == 2, num_warps=warps,
                                                   num_stages=stages, **OPTS)
    if split > 1:
        assert work is not None and m * n <= work.ws.numel() and tiles_mn <= work.cnt.numel(), (m, n, tiles_mn)
        ws, cnt = work.ws, work.cnt
    else:
        ws, cnt = y, y
    return exactlin_scaled_mm[(tiles_mn, split)](xq, w, s_a, s_w, y, ws, cnt, m, n, k, xq.stride(0), w.stride(0), y.stride(0),
                                          BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, num_warps=warps,
                                          num_stages=stages, **OPTS)


def scaled_mm_zy(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor, zy: torch.Tensor,
                 rows: int, cfg: Cfg | None = None, work: Workspace | None = None):
    """`scaled_mm` into y, and zy [units, ZY_BYTES] uint8, the call's records over `rows` slots (`int8_exact.zy_records`),
    from the same accumulator, in place."""
    m, k = xq.shape
    n = w.shape[0]
    assert xq.dtype == torch.int8 and w.dtype == torch.int8 and w.shape[1] == k and xq.stride(1) == 1 and w.stride(1) == 1
    assert s_a.dtype == torch.float32 and s_a.numel() == m and s_a.is_contiguous()
    assert s_w.dtype == torch.float32 and s_w.numel() == n and s_w.is_contiguous()
    assert y.dtype == torch.bfloat16 and y.shape == (m, n) and y.stride(1) == 1
    assert zy.dtype == torch.uint8 and zy.shape == (zy_units(m, n, rows), ZY_BYTES) and zy.is_contiguous(), zy.shape
    bm, bn, bk, warps, stages, split, tma = cfg or default_cfg(m, n, k)
    tiles = triton.cdiv(rows, bm) * triton.cdiv(n, bn)
    z32, y16 = zy.view(torch.int32), zy.view(torch.int16)
    if tma:
        from triton.tools.tensor_descriptor import TensorDescriptor
        assert split == 1, cfg
        a_desc = TensorDescriptor.from_tensor(xq, [bm, bk])
        b_desc = TensorDescriptor.from_tensor(w, [bn, bk])
        return exactlin_scaled_mm_zy_tma[(tiles,)](a_desc, b_desc, s_a, s_w, y, z32, y16, m, n, k, rows, y.stride(0),
                                                   BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, WARP_SPECIALIZE=tma == 2,
                                                   UR=UNIT[0], UC=UNIT[1], num_warps=warps, num_stages=stages, **OPTS)
    if split > 1:
        assert work is not None and m * n <= work.ws.numel() and tiles <= work.cnt.numel(), (m, n, tiles)
        ws, cnt = work.ws, work.cnt
    else:
        ws, cnt = z32, z32
    return exactlin_scaled_mm_zy[(tiles, split)](xq, w, s_a, s_w, y, z32, y16, ws, cnt, m, n, k, rows, xq.stride(0),
                                                 w.stride(0), y.stride(0), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                                 SPLIT_K=split, UR=UNIT[0], UC=UNIT[1], num_warps=warps,
                                                 num_stages=stages, **OPTS)


def linear(x: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, xq: torch.Tensor, s_a: torch.Tensor, y: torch.Tensor):
    """The exact linear into caller-owned (xq, s_a, y)."""
    quant_rows(x, xq, s_a)
    scaled_mm(xq, s_a, w, s_w, y)


def ptx_check(compiled) -> dict:
    """What the PTX of a compiled kernel (a launch's return value) says about its float semantics."""
    ptx = compiled.asm["ptx"]
    name = compiled.metadata.name
    doc = {"kernel": name, "ftz": ptx.count(".ftz"), "div_rn_f32": ptx.count("div.rn.f32"),
           "div_approx_or_full": ptx.count("div.approx") + ptx.count("div.full"), "cvt_rni_s32_f32": ptx.count("cvt.rni.s32.f32"),
           "fma_f32": sum(ptx.count(f) for f in ("fma.rn.f32", "fma.rn.ftz.f32", "mad.f32", "mad.rn.f32")),
           "mad_f32": ptx.count("mad.f32") + ptx.count("mad.rn.f32"),
           "cvt_rn_f32_s32": ptx.count("cvt.rn.f32.s32"), "cvt_rn_bf16": ptx.count("cvt.rn.bf16"),
           # mma.sync's .s32.s8.s8.s32 (sm_80, sm_120), wgmma's .s32.s8.s8 (sm_90), tcgen05's kind::i8 (sm_100)
           "mma_s8": ptx.count(".s8.s8") + ptx.count("kind::i8"),
           "global_scratch": int(getattr(compiled.metadata, "global_scratch_size", 0) or 0),
           "atom_global": ptx.count("atom.global"), "membar_gl": ptx.count("membar.gl")}
    if name.startswith(("exactlin_quant_rows", "exactlin_silu_mul_quant", "exactlin_add_norm_quant")):
        # its fma.rn are the division's corrections, written out (and the norm's contracted squares, libdevice expf's
        # steps); nothing is contracted into a mad.  libdevice's expf takes its 2^f on ex2.approx.ftz, the one .ftz allowed
        doc["ex2_approx_ftz"] = ptx.count("ex2.approx.ftz.f32")
        doc["ok"] = (doc["ftz"] == doc["ex2_approx_ftz"] and doc["div_rn_f32"] >= 2 and doc["div_approx_or_full"] == 0
                     and doc["cvt_rni_s32_f32"] >= 1 and doc["fma_f32"] >= 4 and doc["mad_f32"] == 0
                     and doc["global_scratch"] == 0)
    else:
        doc["ok"] = (doc["ftz"] == 0 and doc["div_approx_or_full"] == 0 and doc["fma_f32"] == 0 and doc["cvt_rn_f32_s32"] >= 1
                     and doc["cvt_rn_bf16"] >= 1 and doc["mma_s8"] >= 1 and doc["global_scratch"] == 0
                     and (doc["atom_global"] == 0 or doc["membar_gl"] >= 2))
    return doc


_WORK: dict[int, Workspace] = {}


def row_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`verity_int8::row_quant`'s CUDA body: (xq, s_a) of x [M, K] bf16, allocating only them."""
    if x.stride(1) != 1:
        x = x.contiguous()
    xq = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    s = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    if x.shape[0]:
        quant_rows(x, xq, s)
    return xq, s


def _plan(m: int, n: int, k: int, device: torch.device) -> tuple[Cfg, Workspace | None]:
    """A call's config (`pick`) and, for a split-K one, the device's workspace, made at the device's first split-K call; a
    capture that would make it runs the config unsplit instead (the same values), since a workspace made inside a capture
    would live in the graph's pool."""
    dev = device.index
    cfg, work = pick(m, n, k, arch(dev)), None
    if cfg[5] > 1:
        if dev not in _WORK and not torch.cuda.is_current_stream_capturing():
            _WORK[dev] = Workspace(device)
        work = _WORK.get(dev)
        if work is None:
            cfg = Cfg(*cfg[:5], 1, cfg[6])
    return cfg, work


def product(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
    """`verity_int8::scaled_mm`'s CUDA body: y [M, N] bf16, allocating only it, at `_plan`'s config."""
    m, n, k = xq.shape[0], w.shape[0], w.shape[1]
    y = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    if m:
        scaled_mm(xq, s_a, w, s_w.reshape(-1), y, *_plan(m, n, k, xq.device))
    return y


def product_zy(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor,
               rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`verity_int8::scaled_mm_zy`'s CUDA body: (y [M, N] bf16, zy [units, ZY_BYTES] uint8), allocating only them, at
    `product`'s config.  A call of no rows has only empty slots, so every record's bytes are zero."""
    m, n, k = xq.shape[0], w.shape[0], w.shape[1]
    y = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    zy = torch.empty((zy_units(m, n, rows), ZY_BYTES), dtype=torch.uint8, device=xq.device)
    if m:
        scaled_mm_zy(xq, s_a.reshape(-1), w, s_w.reshape(-1), y, zy, rows, *_plan(m, n, k, xq.device))
    else:
        zy.zero_()
    return y, zy
