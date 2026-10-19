"""The int8-exact linear fused: `quant_rows`, one Triton kernel bit for bit `int8_exact.quantize_rows`, and `scaled_mm`, one
Triton GEMM whose epilogue is `int8_exact.dequantize`, so the exact linear is two launches a call (stock vLLM int8's count)
instead of `int8_ops`' roughly fourteen.  Both write into caller-owned buffers, allocate nothing and are graph-capturable
once compiled (`warm`).

For x [M, K] bf16, W [N, K] int8 and s_w [N] float32 (`engine.int8_exact`'s statement):

    a[i]   = max_k |f32(x[i, k])|                                  exact (a max)
    s_a[i] = RN32(max(a[i], RN32(1e-30)) / 127)                    div.rn.f32
    X[i,k] = clamp(rint(RN32(f32(x[i, k]) / s_a[i])), -127, 127)   div.rn.f32, then cvt.rni (round half to even)
    Y      = X W^T in int32                                        tl.dot, s8 x s8 -> s32, exact for K <= MAX_K
    y[i,n] = RN_bf16(RN32(RN32(f32(Y[i,n]) * s_a[i]) * s_w[n]))    cvt.rn.f32.s32, two mul.rn, cvt.rn.bf16

Triton's `/` on float32 is div.full.f32 (approximate) and its libdevice calls are flush-to-zero by default, so the kernels
use tl.div_rn, an inline cvt.rni, enable_reflect_ftz=False and enable_fp_fusion=False; `ptx_check` reads the compiled PTX
for exactly these instructions.  The GEMM's tiles are vLLM's `triton_scaled_mm` heuristic (the stock arm's kernel), on the
pointer path, without bias; its epilogue order is the same as vLLM's (activation scale first).  The product needs no row
padding: a row of X W^T reads only that row, so M < 17 is just a masked tile (int8_exact pads only because torch._int_mm
refuses M <= 16).  Non-finite activations are outside the statement and not gated.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

OPTS = {"enable_fp_fusion": False, "enable_reflect_ftz": False}


@triton.jit
def _rni_s32(v):
    return tl.inline_asm_elementwise("cvt.rni.s32.f32 $0, $1;", "=r,r", [v], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def exactlin_quant_rows(x_ptr, q_ptr, s_ptr, K, stride_x, stride_q, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.max(tl.abs(x), axis=0)
    s = tl.div_rn(tl.maximum(a, 1e-30), 127.0)
    q = _rni_s32(tl.div_rn(x, tl.broadcast_to(s, (BLOCK_K,))))
    q = tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)
    tl.store(q_ptr + row * stride_q + offs, q, mask=mask)
    tl.store(s_ptr + row, s)


@triton.jit
def exactlin_scaled_mm(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr, M, N, K, stride_am, stride_bn, stride_cm,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_ptrs = a_ptr + stride_am * offs_m[:, None] + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] + stride_bn * offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    sa = tl.load(sa_ptr + offs_m, mask=mask_m, other=0.0)
    sb = tl.load(sb_ptr + offs_n, mask=mask_n, other=0.0)
    y = acc.to(tl.float32) * sa[:, None]
    y = y * sb[None, :]
    tl.store(c_ptr + stride_cm * offs_m[:, None] + offs_n[None, :], y.to(tl.bfloat16),
             mask=mask_m[:, None] & mask_n[None, :])


def quant_warps(k: int) -> int:
    b = triton.next_power_of_2(k)
    return 4 if b <= 4096 else 8 if b <= 8192 else 16


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


def quant_rows(x: torch.Tensor, xq: torch.Tensor, s: torch.Tensor):
    """x [M, K] bf16 (unit stride along K) -> xq [M, K] int8 and s [M] (or [M, 1]) float32, in place."""
    m, k = x.shape
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1, (x.dtype, x.stride())
    assert xq.dtype == torch.int8 and xq.shape == x.shape and xq.stride(1) == 1
    assert s.dtype == torch.float32 and s.numel() == m and s.is_contiguous()
    return exactlin_quant_rows[(m,)](x, xq, s, k, x.stride(0), xq.stride(0), BLOCK_K=triton.next_power_of_2(k),
                              num_warps=quant_warps(k), **OPTS)


def scaled_mm(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor):
    """xq [M, K] int8, s_a [M] f32, w [N, K] int8 (row-major), s_w [N] or [N, 1] f32 -> y [M, N] bf16, in place."""
    m, k = xq.shape
    n = w.shape[0]
    assert xq.dtype == torch.int8 and w.dtype == torch.int8 and w.shape[1] == k and xq.stride(1) == 1 and w.stride(1) == 1
    assert s_a.dtype == torch.float32 and s_a.numel() == m and s_a.is_contiguous()
    assert s_w.dtype == torch.float32 and s_w.numel() == n and s_w.is_contiguous()
    assert y.dtype == torch.bfloat16 and y.shape == (m, n) and y.stride(1) == 1
    bm, bn, bk = tiles(m, n)
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    return exactlin_scaled_mm[grid](xq, w, s_a, s_w, y, m, n, k, xq.stride(0), w.stride(0), y.stride(0),
                             BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, **OPTS)


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
           "cvt_rn_f32_s32": ptx.count("cvt.rn.f32.s32"), "cvt_rn_bf16": ptx.count("cvt.rn.bf16"),
           "mma_s8": ptx.count(".s8.s8.s32"), "global_scratch": int(getattr(compiled.metadata, "global_scratch_size", 0) or 0)}
    if name.startswith("exactlin_quant_rows"):
        doc["ok"] = (doc["ftz"] == 0 and doc["div_rn_f32"] >= 2 and doc["div_approx_or_full"] == 0
                     and doc["cvt_rni_s32_f32"] >= 1 and doc["fma_f32"] == 0 and doc["global_scratch"] == 0)
    else:
        doc["ok"] = (doc["ftz"] == 0 and doc["div_approx_or_full"] == 0 and doc["fma_f32"] == 0 and doc["cvt_rn_f32_s32"] >= 1
                     and doc["cvt_rn_bf16"] >= 1 and doc["mma_s8"] >= 1 and doc["global_scratch"] == 0)
    return doc


def install() -> None:
    """Serve `verity_int8::row_quant` and `verity_int8::scaled_mm` on CUDA from these kernels (`register_kernel`), each op
    allocating only its own results, as the ops' default bodies do."""
    from verity_vllm.engine import int8_ops

    @int8_ops.row_quant.register_kernel("cuda")
    def _row_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.stride(1) != 1:
            x = x.contiguous()
        xq = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        s = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
        if x.shape[0]:
            quant_rows(x, xq, s)
        return xq, s

    @int8_ops.scaled_mm.register_kernel("cuda")
    def _scaled_mm(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
        y = torch.empty((xq.shape[0], w.shape[0]), dtype=torch.bfloat16, device=xq.device)
        if xq.shape[0]:
            scaled_mm(xq, s_a, w, s_w.reshape(-1), y)
        return y
