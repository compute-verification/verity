"""Triton kernels for BabyBear^6 on the device (SIMT, 32-bit Montgomery arithmetic in registers).

``ext_mul(a, b)``: elementwise extension product of planar ``(..., 6)`` tensors (``b`` may be one element, broadcast).
Values are plain canonical ``[0, p)`` in and out: the product is computed as ``a b R^-1`` (Montgomery, ``R = 2^32``, one
``mul.lo`` + one ``umulhi`` + a conditional add per base product) and fixed by one more Montgomery product with ``R^2``
per coordinate, so no operand needs pre-scaling.  The ``X^6 = W`` wrap goes through ``W b_j`` (``WR = W R mod p``).
Same arithmetic as ``../tensor/fused/field.py`` (a-fused lane); this kernel only replaces the torch composition
(``(a_i b_j) % p`` outer product + float64 reduction matmul) that dominated the v1 prover's memory traffic.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = tl = None

from .field import DEG, P, W

R = 2**32
MU = pow(P, -1, R)
MU_I32 = MU - R
R2_MOD_P = R * R % P
WR = W * R % P

AVAILABLE = triton is not None and torch.cuda.is_available()

if triton is not None:
    PRIME = tl.constexpr(P)
    MU_C = tl.constexpr(MU_I32)
    WR_C = tl.constexpr(WR)
    R2_C = tl.constexpr(R2_MOD_P)

    @triton.jit
    def mmul(a, b):
        lo = a * b
        hi = tl.umulhi(a, b)
        m = lo * MU_C
        u = tl.umulhi(m, PRIME)
        r = hi - u
        return tl.where(r < 0, r + PRIME, r)

    @triton.jit
    def madd(a, b):
        s = (a - PRIME) + b
        return tl.where(s < 0, s + PRIME, s)

    @triton.jit
    def ext_mmul(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5):
        w1 = mmul(b1, WR_C)
        w2 = mmul(b2, WR_C)
        w3 = mmul(b3, WR_C)
        w4 = mmul(b4, WR_C)
        w5 = mmul(b5, WR_C)
        c0 = mmul(a0, b0)
        c0 = madd(c0, mmul(a1, w5))
        c0 = madd(c0, mmul(a2, w4))
        c0 = madd(c0, mmul(a3, w3))
        c0 = madd(c0, mmul(a4, w2))
        c0 = madd(c0, mmul(a5, w1))
        c1 = mmul(a0, b1)
        c1 = madd(c1, mmul(a1, b0))
        c1 = madd(c1, mmul(a2, w5))
        c1 = madd(c1, mmul(a3, w4))
        c1 = madd(c1, mmul(a4, w3))
        c1 = madd(c1, mmul(a5, w2))
        c2 = mmul(a0, b2)
        c2 = madd(c2, mmul(a1, b1))
        c2 = madd(c2, mmul(a2, b0))
        c2 = madd(c2, mmul(a3, w5))
        c2 = madd(c2, mmul(a4, w4))
        c2 = madd(c2, mmul(a5, w3))
        c3 = mmul(a0, b3)
        c3 = madd(c3, mmul(a1, b2))
        c3 = madd(c3, mmul(a2, b1))
        c3 = madd(c3, mmul(a3, b0))
        c3 = madd(c3, mmul(a4, w5))
        c3 = madd(c3, mmul(a5, w4))
        c4 = mmul(a0, b4)
        c4 = madd(c4, mmul(a1, b3))
        c4 = madd(c4, mmul(a2, b2))
        c4 = madd(c4, mmul(a3, b1))
        c4 = madd(c4, mmul(a4, b0))
        c4 = madd(c4, mmul(a5, w5))
        c5 = mmul(a0, b5)
        c5 = madd(c5, mmul(a1, b4))
        c5 = madd(c5, mmul(a2, b3))
        c5 = madd(c5, mmul(a3, b2))
        c5 = madd(c5, mmul(a4, b1))
        c5 = madd(c5, mmul(a5, b0))
        return c0, c1, c2, c3, c4, c5

    @triton.jit
    def ext_mul_kernel(a_ptr, b_ptr, out_ptr, n, sb: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        m = idx < n
        base_a = idx * 6
        base_b = idx * sb
        a0 = tl.load(a_ptr + base_a + 0, mask=m, other=0).to(tl.int32)
        a1 = tl.load(a_ptr + base_a + 1, mask=m, other=0).to(tl.int32)
        a2 = tl.load(a_ptr + base_a + 2, mask=m, other=0).to(tl.int32)
        a3 = tl.load(a_ptr + base_a + 3, mask=m, other=0).to(tl.int32)
        a4 = tl.load(a_ptr + base_a + 4, mask=m, other=0).to(tl.int32)
        a5 = tl.load(a_ptr + base_a + 5, mask=m, other=0).to(tl.int32)
        b0 = tl.load(b_ptr + base_b + 0, mask=m, other=0).to(tl.int32)
        b1 = tl.load(b_ptr + base_b + 1, mask=m, other=0).to(tl.int32)
        b2 = tl.load(b_ptr + base_b + 2, mask=m, other=0).to(tl.int32)
        b3 = tl.load(b_ptr + base_b + 3, mask=m, other=0).to(tl.int32)
        b4 = tl.load(b_ptr + base_b + 4, mask=m, other=0).to(tl.int32)
        b5 = tl.load(b_ptr + base_b + 5, mask=m, other=0).to(tl.int32)
        c0, c1, c2, c3, c4, c5 = ext_mmul(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5)
        tl.store(out_ptr + base_a + 0, mmul(c0, R2_C), mask=m)
        tl.store(out_ptr + base_a + 1, mmul(c1, R2_C), mask=m)
        tl.store(out_ptr + base_a + 2, mmul(c2, R2_C), mask=m)
        tl.store(out_ptr + base_a + 3, mmul(c3, R2_C), mask=m)
        tl.store(out_ptr + base_a + 4, mmul(c4, R2_C), mask=m)
        tl.store(out_ptr + base_a + 5, mmul(c5, R2_C), mask=m)


def ext_mul(a: torch.Tensor, b: torch.Tensor, out_dtype=torch.int64) -> torch.Tensor:
    """``a`` ``(..., 6)``, ``b`` the same shape or a single element ``(6,)`` / ``(1, ..., 6)``; canonical values."""
    lead = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    if a.shape[:-1] != lead:
        a = a.expand(*lead, DEG)
    a_c = a.contiguous()
    n = a_c.numel() // DEG
    if b.numel() == DEG:
        b_c = b.reshape(DEG).contiguous()
        sb = 0
    else:
        if b.shape[:-1] != lead:
            b = b.expand(*lead, DEG)
        b_c = b.contiguous()
        sb = DEG
    out = torch.empty((n, DEG), dtype=out_dtype, device=a.device)
    if n == 0:
        return out.reshape(*lead, DEG)
    BLOCK = 256
    grid = ((n + BLOCK - 1) // BLOCK,)
    ext_mul_kernel[grid](a_c.view(-1), b_c.view(-1), out.view(-1), n, sb=sb, BLOCK=BLOCK, num_warps=4)
    return out.reshape(*lead, DEG)
