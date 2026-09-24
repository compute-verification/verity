"""Device-side per-round constants for CUDA-graphed sumcheck loops (lane ``a-fusion``).

A CUDA graph cannot contain host work, so everything a round derives from the verifier's coins (``eq`` tables of the
claim point, the running prefix ``eq(rho_{<i-1}, r_{<i-1})``, the four ``eq(rho_{i-1}, x)`` values, the pre-scaled fold
challenge) is computed here by tiny Triton kernels reading two static device buffers: ``rho (n, 6)`` (the level's claim
point, plain) and ``coins (n, 6)`` (the round challenges ``r_1..r_n``, plain).  Same arithmetic as the host helpers in
:mod:`reference` (exact field values; the transcript does not depend on where they were computed).

* :func:`eq_table_vars` -- ``eq(rho[start:start+nvars], b)`` for ``b < 2^nvars`` as ``(2^nvars, 6)`` plain
  (``rho[start]`` on bit 0: ``reference.eq_table``'s convention).
* :func:`round_consts` -- for round ``i`` (1-based): ``prefix = eq_points(rho[:i-1], coins[:i-1])``, ``eq_x[x] =
  eval_at(1 - rho[i-1], rho[i-1], x)`` for ``x = 0..3``, ``r_R = coins[i-1] . R`` -- 36 int32 in one row of a ``(n, 36)``
  buffer.
* :func:`scale_R` -- ``v . R mod p`` elementwise (the device twin of ``sumcheck_packed.pre_scale``).
* :func:`histogram` -- multiplicities of a range table by ``atomic_add`` (``torch.bincount`` syncs on the max and cannot
  be captured).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .kernels_triton import FIELD_KEY, R2_MOD_P, _ext_mmul6_j, madd, mmul, msub


@triton.jit
def _eq_table_vars_kernel(rho_ptr, start, nvars, out_ptr, n, R2: tl.constexpr, BLOCK: tl.constexpr, FK: tl.constexpr):
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = b < n
    zeros = b * 0
    a0 = zeros + 1
    a1 = zeros
    a2 = zeros
    a3 = zeros
    a4 = zeros
    a5 = zeros
    for j in range(nvars):
        bit = (b >> j) & 1
        r0 = tl.load(rho_ptr + (start + j) * 6 + 0)
        r1 = tl.load(rho_ptr + (start + j) * 6 + 1)
        r2 = tl.load(rho_ptr + (start + j) * 6 + 2)
        r3 = tl.load(rho_ptr + (start + j) * 6 + 3)
        r4 = tl.load(rho_ptr + (start + j) * 6 + 4)
        r5 = tl.load(rho_ptr + (start + j) * 6 + 5)
        # factor = bit ? r : 1 - r, then to Montgomery form (x R) so the product below is plain
        f0 = mmul(tl.where(bit == 1, zeros + r0, msub(zeros + 1, zeros + r0)), R2)
        f1 = mmul(tl.where(bit == 1, zeros + r1, msub(zeros, zeros + r1)), R2)
        f2 = mmul(tl.where(bit == 1, zeros + r2, msub(zeros, zeros + r2)), R2)
        f3 = mmul(tl.where(bit == 1, zeros + r3, msub(zeros, zeros + r3)), R2)
        f4 = mmul(tl.where(bit == 1, zeros + r4, msub(zeros, zeros + r4)), R2)
        f5 = mmul(tl.where(bit == 1, zeros + r5, msub(zeros, zeros + r5)), R2)
        a0, a1, a2, a3, a4, a5 = _ext_mmul6_j(a0, a1, a2, a3, a4, a5, f0, f1, f2, f3, f4, f5)
    o = out_ptr + b.to(tl.int64) * 6
    tl.store(o + 0, a0, mask=mask)
    tl.store(o + 1, a1, mask=mask)
    tl.store(o + 2, a2, mask=mask)
    tl.store(o + 3, a3, mask=mask)
    tl.store(o + 4, a4, mask=mask)
    tl.store(o + 5, a5, mask=mask)


def eq_table_vars(rho: torch.Tensor, start: int, nvars: int, out: torch.Tensor | None = None, BLOCK: int = 256) -> torch.Tensor:
    """``eq(rho[start:start+nvars], b)`` as ``(2^nvars, 6) int32`` plain; ``rho (>= start+nvars, 6)`` plain."""
    n = 1 << nvars
    if out is None:
        out = torch.empty((n, 6), dtype=torch.int32, device=rho.device)
    _eq_table_vars_kernel[(triton.cdiv(n, BLOCK),)](rho, start, nvars, out, n, R2=R2_MOD_P, BLOCK=BLOCK, FK=FIELD_KEY)
    return out


@triton.jit
def _round_consts_kernel(rho_ptr, coins_ptr, i, out_ptr, R2: tl.constexpr, FK: tl.constexpr):
    zero = tl.load(rho_ptr + 0) * 0
    one = zero + 1
    p0 = one
    p1 = zero
    p2 = zero
    p3 = zero
    p4 = zero
    p5 = zero
    for j in range(i - 1):
        a0 = tl.load(rho_ptr + j * 6 + 0)
        a1 = tl.load(rho_ptr + j * 6 + 1)
        a2 = tl.load(rho_ptr + j * 6 + 2)
        a3 = tl.load(rho_ptr + j * 6 + 3)
        a4 = tl.load(rho_ptr + j * 6 + 4)
        a5 = tl.load(rho_ptr + j * 6 + 5)
        b0 = tl.load(coins_ptr + j * 6 + 0)
        b1 = tl.load(coins_ptr + j * 6 + 1)
        b2 = tl.load(coins_ptr + j * 6 + 2)
        b3 = tl.load(coins_ptr + j * 6 + 3)
        b4 = tl.load(coins_ptr + j * 6 + 4)
        b5 = tl.load(coins_ptr + j * 6 + 5)
        # eq(a, b) = 1 - a - b + 2 a b
        m0, m1, m2, m3, m4, m5 = _ext_mmul6_j(mmul(a0, R2), mmul(a1, R2), mmul(a2, R2), mmul(a3, R2), mmul(a4, R2), mmul(a5, R2),
                                              b0, b1, b2, b3, b4, b5)
        e0 = madd(msub(msub(one, a0), b0), madd(m0, m0))
        e1 = madd(msub(msub(zero, a1), b1), madd(m1, m1))
        e2 = madd(msub(msub(zero, a2), b2), madd(m2, m2))
        e3 = madd(msub(msub(zero, a3), b3), madd(m3, m3))
        e4 = madd(msub(msub(zero, a4), b4), madd(m4, m4))
        e5 = madd(msub(msub(zero, a5), b5), madd(m5, m5))
        p0, p1, p2, p3, p4, p5 = _ext_mmul6_j(p0, p1, p2, p3, p4, p5, mmul(e0, R2), mmul(e1, R2), mmul(e2, R2), mmul(e3, R2),
                                              mmul(e4, R2), mmul(e5, R2))
    tl.store(out_ptr + 0, p0)
    tl.store(out_ptr + 1, p1)
    tl.store(out_ptr + 2, p2)
    tl.store(out_ptr + 3, p3)
    tl.store(out_ptr + 4, p4)
    tl.store(out_ptr + 5, p5)
    # eq_x[x] = (1 - x)(1 - gm) + x gm for x = 0..3, gm = rho[i - 1]
    for s in tl.static_range(6):
        g = tl.load(rho_ptr + (i - 1) * 6 + s)
        if s == 0:
            x0 = msub(one, g)
        else:
            x0 = msub(zero, g)
        x1 = g
        x2 = msub(madd(g, g), x0)
        x3 = msub(madd(x2, x2), x1)
        tl.store(out_ptr + 6 + s, x0)
        tl.store(out_ptr + 12 + s, x1)
        tl.store(out_ptr + 18 + s, x2)
        tl.store(out_ptr + 24 + s, x3)
        tl.store(out_ptr + 30 + s, mmul(tl.load(coins_ptr + (i - 1) * 6 + s), R2))


def round_consts(rho: torch.Tensor, coins: torch.Tensor, i: int, out: torch.Tensor) -> torch.Tensor:
    """Round ``i`` (1-based) constants into ``out (36,) int32``: ``[prefix (6) | eq_x (4 x 6) | r_i R (6)]``."""
    _round_consts_kernel[(1,)](rho, coins, i, out, R2=R2_MOD_P, FK=FIELD_KEY)
    return out


@triton.jit
def _round_consts_pfx_kernel(rho_ptr, pfx_ptr, coins_ptr, i, out_ptr, R2: tl.constexpr, FK: tl.constexpr):
    zero = tl.load(rho_ptr + 0) * 0
    one = zero + 1
    for s in tl.static_range(6):
        tl.store(out_ptr + s, tl.load(pfx_ptr + s))
        g = tl.load(rho_ptr + (i - 1) * 6 + s)
        if s == 0:
            x0 = msub(one, g)
        else:
            x0 = msub(zero, g)
        x1 = g
        x2 = msub(madd(g, g), x0)
        x3 = msub(madd(x2, x2), x1)
        tl.store(out_ptr + 6 + s, x0)
        tl.store(out_ptr + 12 + s, x1)
        tl.store(out_ptr + 18 + s, x2)
        tl.store(out_ptr + 24 + s, x3)
        tl.store(out_ptr + 30 + s, mmul(tl.load(coins_ptr + (i - 1) * 6 + s), R2))


def round_consts_pfx(rho: torch.Tensor, pfx: torch.Tensor, coins: torch.Tensor, i: int, out: torch.Tensor) -> torch.Tensor:
    """:func:`round_consts` with the prefix ``prod_{j < i-1} eq(rho_j, r_{j+1})`` read from ``pfx (6,)`` (a running product
    the caller keeps: 1 at a level's start, times ``eq(rho_{i-1}, r_i)`` after round ``i``) instead of recomputed."""
    _round_consts_pfx_kernel[(1,)](rho, pfx, coins, i, out, R2=R2_MOD_P, FK=FIELD_KEY)
    return out


@triton.jit
def _scale_R_kernel(v_ptr, out_ptr, n, R2: tl.constexpr, BLOCK: tl.constexpr, FK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    v = tl.load(v_ptr + i, mask=mask, other=0)
    tl.store(out_ptr + i, mmul(v, R2), mask=mask)


def scale_R(v: torch.Tensor, out: torch.Tensor | None = None, BLOCK: int = 256) -> torch.Tensor:
    """``v . R mod p`` elementwise (int32 plain in, int32 out): the device ``pre_scale``."""
    v = v.contiguous()
    if out is None:
        out = torch.empty_like(v)
    n = v.numel()
    _scale_R_kernel[(triton.cdiv(n, BLOCK),)](v, out, n, R2=R2_MOD_P, BLOCK=BLOCK, FK=FIELD_KEY)
    return out


@triton.jit
def _histogram_kernel(w_ptr, m_ptr, n, Nt, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    w = tl.load(w_ptr + i, mask=mask, other=0)
    ok = mask & (w >= 0) & (w < Nt)
    tl.atomic_add(m_ptr + w, (w * 0 + 1).to(tl.int32), mask=ok)


def histogram(w: torch.Tensor, Nt: int, out: torch.Tensor, BLOCK: int = 1024) -> torch.Tensor:
    """``out[v] = #{j : w[j] = v}`` for ``v < Nt`` (``out`` zeroed here; queries outside ``[0, Nt)`` are ignored)."""
    out.zero_()
    n = w.numel()
    _histogram_kernel[(triton.cdiv(n, BLOCK),)](w, out, n, Nt, BLOCK=BLOCK)
    return out
