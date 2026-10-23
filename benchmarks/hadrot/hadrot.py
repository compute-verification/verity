"""A fused signed Walsh-Hadamard rotation and per-token int8 quantization, one Triton program per row.

For x [M, k] bf16 and compute-accounting's rotated arm (benchmarks/ppl/steps.py `Rotation`, kind `q`, on #3251):
Q = diag(signs) (H_A x R_odd), k = A * odd, A a power of two, H_A = Sylvester / sqrt(A), R_odd a random orthogonal
odd x odd (odd in {1, 3} at GLM-5.3's widths), all seeded by k.  Column c = i * odd + j (i < A, j < odd).  The kernel's
one operation order, in float32 with no FMA and no FTZ:

    t_j[i]  = f32(x[i * odd + j]) * signs[i * odd + j]                           exact (a sign)
    u_j'    = (t_0 * R[0, j'] + t_1 * R[1, j']) + t_2 * R[2, j']                 odd = 3 only; mul.rn, add.rn, in this order
    v_j'    = FWHT_A(u_j'): stages h = 1, 2, 4, ..., A / 2, each (a, b) -> (a + b, a - b) over pairs (i, i + h)
    y       = v * RN32(1 / sqrt(A))                                              one mul.rn
    then int8_fused.exactlin_quant_rows' rule on y: s_a = RN32(max(max|y|, RN32(1e-30)) / 127), codes =
    clamp(rint(RN32(y / s_a)), -127, 127)

and writes the int8 codes [M, k] and s_a [M] float32.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

OPTS = {"enable_fp_fusion": False, "enable_reflect_ftz": False}


@triton.jit
def _rni_s32(v):
    return tl.inline_asm_elementwise("cvt.rni.s32.f32 $0, $1;", "=r,r", [v], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _fwht(x, A: tl.constexpr, LOG: tl.constexpr):
    for s in tl.static_range(LOG):
        h: tl.constexpr = 1 << s
        t = tl.permute(tl.reshape(x, (A // (2 * h), 2, h)), (0, 2, 1))
        a, b = tl.split(t)
        x = tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 2, 1)), (A,))
    return x


@triton.jit
def _scale(amax):
    return tl.div_rn(tl.maximum(amax, 1e-30), 127.0)


@triton.jit
def _codes(y, s, A: tl.constexpr):
    # exactlin_quant_rows' division: y / s == (y * 2^e) / (s * 2^e) with s * 2^e in [1, 2)
    sb = s.to(tl.int32, bitcast=True)
    s1 = ((sb & 0x7FFFFF) | 0x3F800000).to(tl.float32, bitcast=True)
    p = ((254 - (sb >> 23)) << 23).to(tl.float32, bitcast=True)
    ys = y * p
    small = tl.abs(ys) < 0.25
    q = _rni_s32(tl.div_rn(tl.where(small, 1.0, ys), tl.broadcast_to(s1, (A,))))
    q = tl.where(small, 0, q)
    return tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)


@triton.jit
def hadrot_quant_rows(x_ptr, sg_ptr, r_ptr, q_ptr, s_ptr, stride_x, stride_q, norm, A: tl.constexpr, LOG: tl.constexpr,
                      ODD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, A)
    xr = x_ptr + row * stride_x
    qr = q_ptr + row * stride_q
    if ODD == 1:
        t = tl.load(xr + i).to(tl.float32) * tl.load(sg_ptr + i)
        y = _fwht(t, A, LOG) * norm
        s = _scale(tl.max(tl.abs(y), axis=0))
        tl.store(qr + i, _codes(y, s, A))
    else:
        c = i * 3
        t0 = tl.load(xr + c).to(tl.float32) * tl.load(sg_ptr + c)
        t1 = tl.load(xr + c + 1).to(tl.float32) * tl.load(sg_ptr + c + 1)
        t2 = tl.load(xr + c + 2).to(tl.float32) * tl.load(sg_ptr + c + 2)
        u0 = (t0 * tl.load(r_ptr + 0) + t1 * tl.load(r_ptr + 3)) + t2 * tl.load(r_ptr + 6)
        u1 = (t0 * tl.load(r_ptr + 1) + t1 * tl.load(r_ptr + 4)) + t2 * tl.load(r_ptr + 7)
        u2 = (t0 * tl.load(r_ptr + 2) + t1 * tl.load(r_ptr + 5)) + t2 * tl.load(r_ptr + 8)
        y0 = _fwht(u0, A, LOG) * norm
        y1 = _fwht(u1, A, LOG) * norm
        y2 = _fwht(u2, A, LOG) * norm
        amax = tl.maximum(tl.maximum(tl.max(tl.abs(y0), axis=0), tl.max(tl.abs(y1), axis=0)), tl.max(tl.abs(y2), axis=0))
        s = _scale(amax)
        tl.store(qr + c, _codes(y0, s, A))
        tl.store(qr + c + 1, _codes(y1, s, A))
        tl.store(qr + c + 2, _codes(y2, s, A))
    tl.store(s_ptr + row, s)


def sylvester(n: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


class Rotation:
    """#3251's `Rotation(k, "q")` draw (generator seeded k: R first, then the signs), held for the kernel."""

    def __init__(self, k: int, device="cuda"):
        g = torch.Generator().manual_seed(k)
        odd = k
        while odd % 2 == 0:
            odd //= 2
        r = torch.linalg.qr(torch.randn(odd, odd, generator=g, dtype=torch.float64))[0] if odd > 1 else torch.ones(1, 1, dtype=torch.float64)
        assert odd in (1, 3), k
        self.k, self.odd, self.a = k, odd, k // odd
        self.log = self.a.bit_length() - 1
        self.signs64 = torch.randint(0, 2, (k,), generator=g).double() * 2 - 1
        self.r64 = r
        self.signs = self.signs64.float().to(device)
        self.r = r.float().contiguous().to(device)
        self.norm = float(torch.tensor(1 / math.sqrt(self.a), dtype=torch.float32))

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """x Q in float64 (on x's device), through H_A = H_a1 x H_a2."""
        m = x.shape[0]
        dev = x.device
        t = (x.double() * self.signs64.to(dev)).view(m, self.a, self.odd) @ self.r64.to(dev)
        a1 = 1 << (self.log // 2)
        a2 = self.a // a1
        h1, h2 = sylvester(a1).to(dev), sylvester(a2).to(dev)
        y = torch.einsum("ai,mabo,bj->mijo", h1, t.view(m, a1, a2, self.odd), h2)
        return y.reshape(m, self.k)


def hadrot_quant(rot: Rotation, x: torch.Tensor, xq: torch.Tensor, s: torch.Tensor, num_warps: int | None = None):
    m, k = x.shape
    assert k == rot.k and x.dtype == torch.bfloat16 and x.stride(1) == 1
    assert xq.dtype == torch.int8 and xq.shape == x.shape and xq.stride(1) == 1
    assert s.dtype == torch.float32 and s.numel() == m and s.is_contiguous()
    warps = num_warps or (16 if rot.a >= 4096 else 8 if rot.a >= 1024 else 4)
    return hadrot_quant_rows[(m,)](x, rot.signs, rot.r, xq, s, x.stride(0), xq.stride(0), rot.norm, A=rot.a, LOG=rot.log,
                                   ODD=rot.odd, num_warps=warps, **OPTS)
