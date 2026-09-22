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

    @triton.jit
    def msub(a, b):
        s = a - b
        return tl.where(s < 0, s + PRIME, s)

    @triton.jit
    def load6(ptr, base, m):
        x0 = tl.load(ptr + base + 0, mask=m, other=0).to(tl.int32)
        x1 = tl.load(ptr + base + 1, mask=m, other=0).to(tl.int32)
        x2 = tl.load(ptr + base + 2, mask=m, other=0).to(tl.int32)
        x3 = tl.load(ptr + base + 3, mask=m, other=0).to(tl.int32)
        x4 = tl.load(ptr + base + 4, mask=m, other=0).to(tl.int32)
        x5 = tl.load(ptr + base + 5, mask=m, other=0).to(tl.int32)
        return x0, x1, x2, x3, x4, x5

    @triton.jit
    def fold_kernel(v_ptr, r_ptr, out_ptr, half, BASE: tl.constexpr, BLOCK: tl.constexpr):
        """``out[i] = v[i] + (v[i+half] - v[i]) r``; ``r_ptr`` holds ``r R mod p`` so the Montgomery product is canonical.
        ``BASE``: ``v`` is a base-field vector (stride 1) and ``out`` is extension."""
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        m = idx < half
        r0 = tl.load(r_ptr + 0).to(tl.int32)
        r1 = tl.load(r_ptr + 1).to(tl.int32)
        r2 = tl.load(r_ptr + 2).to(tl.int32)
        r3 = tl.load(r_ptr + 3).to(tl.int32)
        r4 = tl.load(r_ptr + 4).to(tl.int32)
        r5 = tl.load(r_ptr + 5).to(tl.int32)
        ob = idx * 6
        if BASE:
            lo = tl.load(v_ptr + idx, mask=m, other=0).to(tl.int32)
            hi = tl.load(v_ptr + half + idx, mask=m, other=0).to(tl.int32)
            d = msub(hi, lo)
            tl.store(out_ptr + ob + 0, madd(lo, mmul(d, r0)), mask=m)
            tl.store(out_ptr + ob + 1, mmul(d, r1), mask=m)
            tl.store(out_ptr + ob + 2, mmul(d, r2), mask=m)
            tl.store(out_ptr + ob + 3, mmul(d, r3), mask=m)
            tl.store(out_ptr + ob + 4, mmul(d, r4), mask=m)
            tl.store(out_ptr + ob + 5, mmul(d, r5), mask=m)
        else:
            l0, l1, l2, l3, l4, l5 = load6(v_ptr, ob, m)
            h0, h1, h2, h3, h4, h5 = load6(v_ptr, ob + half * 6, m)
            d0 = msub(h0, l0)
            d1 = msub(h1, l1)
            d2 = msub(h2, l2)
            d3 = msub(h3, l3)
            d4 = msub(h4, l4)
            d5 = msub(h5, l5)
            c0, c1, c2, c3, c4, c5 = ext_mmul(d0, d1, d2, d3, d4, d5, r0, r1, r2, r3, r4, r5)
            tl.store(out_ptr + ob + 0, madd(l0, c0), mask=m)
            tl.store(out_ptr + ob + 1, madd(l1, c1), mask=m)
            tl.store(out_ptr + ob + 2, madd(l2, c2), mask=m)
            tl.store(out_ptr + ob + 3, madd(l3, c3), mask=m)
            tl.store(out_ptr + ob + 4, madd(l4, c4), mask=m)
            tl.store(out_ptr + ob + 5, madd(l5, c5), mask=m)

    @triton.jit
    def logup_round_kernel(p0_ptr, p1_ptr, q0_ptr, q1_ptr, e_ptr, lam_ptr, out_ptr, half,
                           PBASE: tl.constexpr, BLOCK: tl.constexpr):
        """One LogUp sumcheck round: for t = 0..3, ``Σ_c e_t(c) [p0_t q1_t + p1_t q0_t + λ q0_t q1_t](c)``,
        with ``x_t = x(c) + t (x(c+half) - x(c))``.  Writes the block's partial sums as ``(4, 6)`` int64 scaled by
        ``R^-2`` (two Montgomery products deep); the host sums the blocks and multiplies by ``R^2``.
        ``lam_ptr`` holds ``λ R mod p``.  ``PBASE``: the p tables are base-field (leaf level)."""
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        m = idx < half
        ob = idx * 6
        oh = ob + half * 6
        m0 = tl.load(lam_ptr + 0).to(tl.int32)
        m1 = tl.load(lam_ptr + 1).to(tl.int32)
        m2 = tl.load(lam_ptr + 2).to(tl.int32)
        m3 = tl.load(lam_ptr + 3).to(tl.int32)
        m4 = tl.load(lam_ptr + 4).to(tl.int32)
        m5 = tl.load(lam_ptr + 5).to(tl.int32)
        b0, b1, b2, b3, b4, b5 = load6(q0_ptr, ob, m)
        bh0, bh1, bh2, bh3, bh4, bh5 = load6(q0_ptr, oh, m)
        d0, d1, d2, d3, d4, d5 = load6(q1_ptr, ob, m)
        dh0, dh1, dh2, dh3, dh4, dh5 = load6(q1_ptr, oh, m)
        e0, e1, e2, e3, e4, e5 = load6(e_ptr, ob, m)
        eh0, eh1, eh2, eh3, eh4, eh5 = load6(e_ptr, oh, m)
        db0 = msub(bh0, b0)
        db1 = msub(bh1, b1)
        db2 = msub(bh2, b2)
        db3 = msub(bh3, b3)
        db4 = msub(bh4, b4)
        db5 = msub(bh5, b5)
        dd0 = msub(dh0, d0)
        dd1 = msub(dh1, d1)
        dd2 = msub(dh2, d2)
        dd3 = msub(dh3, d3)
        dd4 = msub(dh4, d4)
        dd5 = msub(dh5, d5)
        de0 = msub(eh0, e0)
        de1 = msub(eh1, e1)
        de2 = msub(eh2, e2)
        de3 = msub(eh3, e3)
        de4 = msub(eh4, e4)
        de5 = msub(eh5, e5)
        if PBASE:
            a = tl.load(p0_ptr + idx, mask=m, other=0).to(tl.int32)
            ah = tl.load(p0_ptr + half + idx, mask=m, other=0).to(tl.int32)
            c = tl.load(p1_ptr + idx, mask=m, other=0).to(tl.int32)
            ch = tl.load(p1_ptr + half + idx, mask=m, other=0).to(tl.int32)
            da = msub(ah, a)
            dc = msub(ch, c)
        else:
            a0, a1, a2, a3, a4, a5 = load6(p0_ptr, ob, m)
            ah0, ah1, ah2, ah3, ah4, ah5 = load6(p0_ptr, oh, m)
            c0, c1, c2, c3, c4, c5 = load6(p1_ptr, ob, m)
            ch0, ch1, ch2, ch3, ch4, ch5 = load6(p1_ptr, oh, m)
            da0 = msub(ah0, a0)
            da1 = msub(ah1, a1)
            da2 = msub(ah2, a2)
            da3 = msub(ah3, a3)
            da4 = msub(ah4, a4)
            da5 = msub(ah5, a5)
            dc0 = msub(ch0, c0)
            dc1 = msub(ch1, c1)
            dc2 = msub(ch2, c2)
            dc3 = msub(ch3, c3)
            dc4 = msub(ch4, c4)
            dc5 = msub(ch5, c5)
        for t in tl.static_range(4):
            if t > 0:
                b0 = madd(b0, db0)
                b1 = madd(b1, db1)
                b2 = madd(b2, db2)
                b3 = madd(b3, db3)
                b4 = madd(b4, db4)
                b5 = madd(b5, db5)
                d0 = madd(d0, dd0)
                d1 = madd(d1, dd1)
                d2 = madd(d2, dd2)
                d3 = madd(d3, dd3)
                d4 = madd(d4, dd4)
                d5 = madd(d5, dd5)
                e0 = madd(e0, de0)
                e1 = madd(e1, de1)
                e2 = madd(e2, de2)
                e3 = madd(e3, de3)
                e4 = madd(e4, de4)
                e5 = madd(e5, de5)
                if PBASE:
                    a = madd(a, da)
                    c = madd(c, dc)
                else:
                    a0 = madd(a0, da0)
                    a1 = madd(a1, da1)
                    a2 = madd(a2, da2)
                    a3 = madd(a3, da3)
                    a4 = madd(a4, da4)
                    a5 = madd(a5, da5)
                    c0 = madd(c0, dc0)
                    c1 = madd(c1, dc1)
                    c2 = madd(c2, dc2)
                    c3 = madd(c3, dc3)
                    c4 = madd(c4, dc4)
                    c5 = madd(c5, dc5)
            # qq = q0 q1 R^-1 ; g = (p0 q1 + p1 q0 + λ q0 q1) R^-1
            qq0, qq1, qq2, qq3, qq4, qq5 = ext_mmul(b0, b1, b2, b3, b4, b5, d0, d1, d2, d3, d4, d5)
            g0, g1, g2, g3, g4, g5 = ext_mmul(qq0, qq1, qq2, qq3, qq4, qq5, m0, m1, m2, m3, m4, m5)
            if PBASE:
                g0 = madd(g0, madd(mmul(a, d0), mmul(c, b0)))
                g1 = madd(g1, madd(mmul(a, d1), mmul(c, b1)))
                g2 = madd(g2, madd(mmul(a, d2), mmul(c, b2)))
                g3 = madd(g3, madd(mmul(a, d3), mmul(c, b3)))
                g4 = madd(g4, madd(mmul(a, d4), mmul(c, b4)))
                g5 = madd(g5, madd(mmul(a, d5), mmul(c, b5)))
            else:
                x0, x1, x2, x3, x4, x5 = ext_mmul(a0, a1, a2, a3, a4, a5, d0, d1, d2, d3, d4, d5)
                y0, y1, y2, y3, y4, y5 = ext_mmul(c0, c1, c2, c3, c4, c5, b0, b1, b2, b3, b4, b5)
                g0 = madd(g0, madd(x0, y0))
                g1 = madd(g1, madd(x1, y1))
                g2 = madd(g2, madd(x2, y2))
                g3 = madd(g3, madd(x3, y3))
                g4 = madd(g4, madd(x4, y4))
                g5 = madd(g5, madd(x5, y5))
            h0, h1, h2, h3, h4, h5 = ext_mmul(e0, e1, e2, e3, e4, e5, g0, g1, g2, g3, g4, g5)
            h0 = tl.where(m, h0, 0)
            h1 = tl.where(m, h1, 0)
            h2 = tl.where(m, h2, 0)
            h3 = tl.where(m, h3, 0)
            h4 = tl.where(m, h4, 0)
            h5 = tl.where(m, h5, 0)
            ob_out = pid * 24 + t * 6
            tl.store(out_ptr + ob_out + 0, tl.sum(h0.to(tl.int64), axis=0))
            tl.store(out_ptr + ob_out + 1, tl.sum(h1.to(tl.int64), axis=0))
            tl.store(out_ptr + ob_out + 2, tl.sum(h2.to(tl.int64), axis=0))
            tl.store(out_ptr + ob_out + 3, tl.sum(h3.to(tl.int64), axis=0))
            tl.store(out_ptr + ob_out + 4, tl.sum(h4.to(tl.int64), axis=0))
            tl.store(out_ptr + ob_out + 5, tl.sum(h5.to(tl.int64), axis=0))


    @triton.jit
    def load6_2d(ptr, off, m):
        x0 = tl.load(ptr + off + 0, mask=m, other=0).to(tl.int32)
        x1 = tl.load(ptr + off + 1, mask=m, other=0).to(tl.int32)
        x2 = tl.load(ptr + off + 2, mask=m, other=0).to(tl.int32)
        x3 = tl.load(ptr + off + 3, mask=m, other=0).to(tl.int32)
        x4 = tl.load(ptr + off + 4, mask=m, other=0).to(tl.int32)
        x5 = tl.load(ptr + off + 5, mask=m, other=0).to(tl.int32)
        return x0, x1, x2, x3, x4, x5

    @triton.jit
    def phase1_kernel(a_ptr, b_ptr, e_ptr, w_ptr, out_ptr, half, G,
                      BASE: tl.constexpr, BC: tl.constexpr, BG: tl.constexpr):
        """GKR phase-1 round: ``h(t) = Σ_c e_t(c) Σ_g w(g) a_t(c,g) b_t(c,g)`` for t = 0..3, ``x_t = x(c) + t (x(c+half) - x(c))``.
        ``a``/``b`` are ``(2·half, G[, 6])`` (``BASE``: base field), ``e`` ``(2·half, 6)``, ``w_ptr`` = ``w R mod p`` ``(G, 6)``.
        Block partials ``(4, 6)`` int64 scaled by ``R^-2``."""
        pid = tl.program_id(0)
        c = pid * BC + tl.arange(0, BC)
        mc = c < half
        acc00 = tl.zeros([BC], dtype=tl.int32)
        acc01 = tl.zeros([BC], dtype=tl.int32)
        acc02 = tl.zeros([BC], dtype=tl.int32)
        acc03 = tl.zeros([BC], dtype=tl.int32)
        acc04 = tl.zeros([BC], dtype=tl.int32)
        acc05 = tl.zeros([BC], dtype=tl.int32)
        acc10 = tl.zeros([BC], dtype=tl.int32)
        acc11 = tl.zeros([BC], dtype=tl.int32)
        acc12 = tl.zeros([BC], dtype=tl.int32)
        acc13 = tl.zeros([BC], dtype=tl.int32)
        acc14 = tl.zeros([BC], dtype=tl.int32)
        acc15 = tl.zeros([BC], dtype=tl.int32)
        acc20 = tl.zeros([BC], dtype=tl.int32)
        acc21 = tl.zeros([BC], dtype=tl.int32)
        acc22 = tl.zeros([BC], dtype=tl.int32)
        acc23 = tl.zeros([BC], dtype=tl.int32)
        acc24 = tl.zeros([BC], dtype=tl.int32)
        acc25 = tl.zeros([BC], dtype=tl.int32)
        acc30 = tl.zeros([BC], dtype=tl.int32)
        acc31 = tl.zeros([BC], dtype=tl.int32)
        acc32 = tl.zeros([BC], dtype=tl.int32)
        acc33 = tl.zeros([BC], dtype=tl.int32)
        acc34 = tl.zeros([BC], dtype=tl.int32)
        acc35 = tl.zeros([BC], dtype=tl.int32)
        for g0 in range(0, G, BG):
            g = g0 + tl.arange(0, BG)
            mg = g < G
            m2 = mc[:, None] & mg[None, :]
            wg = g * 6
            w0 = tl.load(w_ptr + wg + 0, mask=mg, other=0).to(tl.int32)[None, :]
            w1 = tl.load(w_ptr + wg + 1, mask=mg, other=0).to(tl.int32)[None, :]
            w2 = tl.load(w_ptr + wg + 2, mask=mg, other=0).to(tl.int32)[None, :]
            w3 = tl.load(w_ptr + wg + 3, mask=mg, other=0).to(tl.int32)[None, :]
            w4 = tl.load(w_ptr + wg + 4, mask=mg, other=0).to(tl.int32)[None, :]
            w5 = tl.load(w_ptr + wg + 5, mask=mg, other=0).to(tl.int32)[None, :]
            if BASE:
                off = c[:, None] * G + g[None, :]
                a0 = tl.load(a_ptr + off, mask=m2, other=0).to(tl.int32)
                a1 = tl.load(a_ptr + off + half * G, mask=m2, other=0).to(tl.int32)
                b0 = tl.load(b_ptr + off, mask=m2, other=0).to(tl.int32)
                b1 = tl.load(b_ptr + off + half * G, mask=m2, other=0).to(tl.int32)
                da = msub(a1, a0)
                db = msub(b1, b0)
                for t in tl.static_range(4):
                    if t > 0:
                        a0 = madd(a0, da)
                        b0 = madd(b0, db)
                    pr = mmul(a0, b0)
                    s0 = tl.sum(mmul(pr, w0).to(tl.int64), axis=1)
                    s1 = tl.sum(mmul(pr, w1).to(tl.int64), axis=1)
                    s2 = tl.sum(mmul(pr, w2).to(tl.int64), axis=1)
                    s3 = tl.sum(mmul(pr, w3).to(tl.int64), axis=1)
                    s4 = tl.sum(mmul(pr, w4).to(tl.int64), axis=1)
                    s5 = tl.sum(mmul(pr, w5).to(tl.int64), axis=1)
                    if t == 0:
                        acc00 = madd(acc00, (s0 % PRIME).to(tl.int32))
                        acc01 = madd(acc01, (s1 % PRIME).to(tl.int32))
                        acc02 = madd(acc02, (s2 % PRIME).to(tl.int32))
                        acc03 = madd(acc03, (s3 % PRIME).to(tl.int32))
                        acc04 = madd(acc04, (s4 % PRIME).to(tl.int32))
                        acc05 = madd(acc05, (s5 % PRIME).to(tl.int32))
                    elif t == 1:
                        acc10 = madd(acc10, (s0 % PRIME).to(tl.int32))
                        acc11 = madd(acc11, (s1 % PRIME).to(tl.int32))
                        acc12 = madd(acc12, (s2 % PRIME).to(tl.int32))
                        acc13 = madd(acc13, (s3 % PRIME).to(tl.int32))
                        acc14 = madd(acc14, (s4 % PRIME).to(tl.int32))
                        acc15 = madd(acc15, (s5 % PRIME).to(tl.int32))
                    elif t == 2:
                        acc20 = madd(acc20, (s0 % PRIME).to(tl.int32))
                        acc21 = madd(acc21, (s1 % PRIME).to(tl.int32))
                        acc22 = madd(acc22, (s2 % PRIME).to(tl.int32))
                        acc23 = madd(acc23, (s3 % PRIME).to(tl.int32))
                        acc24 = madd(acc24, (s4 % PRIME).to(tl.int32))
                        acc25 = madd(acc25, (s5 % PRIME).to(tl.int32))
                    else:
                        acc30 = madd(acc30, (s0 % PRIME).to(tl.int32))
                        acc31 = madd(acc31, (s1 % PRIME).to(tl.int32))
                        acc32 = madd(acc32, (s2 % PRIME).to(tl.int32))
                        acc33 = madd(acc33, (s3 % PRIME).to(tl.int32))
                        acc34 = madd(acc34, (s4 % PRIME).to(tl.int32))
                        acc35 = madd(acc35, (s5 % PRIME).to(tl.int32))
            else:
                off = (c[:, None] * G + g[None, :]) * 6
                a0, a1, a2, a3, a4, a5 = load6_2d(a_ptr, off, m2)
                ah0, ah1, ah2, ah3, ah4, ah5 = load6_2d(a_ptr, off + half * G * 6, m2)
                b0, b1, b2, b3, b4, b5 = load6_2d(b_ptr, off, m2)
                bh0, bh1, bh2, bh3, bh4, bh5 = load6_2d(b_ptr, off + half * G * 6, m2)
                da0 = msub(ah0, a0)
                da1 = msub(ah1, a1)
                da2 = msub(ah2, a2)
                da3 = msub(ah3, a3)
                da4 = msub(ah4, a4)
                da5 = msub(ah5, a5)
                db0 = msub(bh0, b0)
                db1 = msub(bh1, b1)
                db2 = msub(bh2, b2)
                db3 = msub(bh3, b3)
                db4 = msub(bh4, b4)
                db5 = msub(bh5, b5)
                for t in tl.static_range(4):
                    if t > 0:
                        a0 = madd(a0, da0)
                        a1 = madd(a1, da1)
                        a2 = madd(a2, da2)
                        a3 = madd(a3, da3)
                        a4 = madd(a4, da4)
                        a5 = madd(a5, da5)
                        b0 = madd(b0, db0)
                        b1 = madd(b1, db1)
                        b2 = madd(b2, db2)
                        b3 = madd(b3, db3)
                        b4 = madd(b4, db4)
                        b5 = madd(b5, db5)
                    p0, p1, p2, p3, p4, p5 = ext_mmul(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5)
                    x0, x1, x2, x3, x4, x5 = ext_mmul(p0, p1, p2, p3, p4, p5, w0, w1, w2, w3, w4, w5)
                    s0 = tl.sum(x0.to(tl.int64), axis=1)
                    s1 = tl.sum(x1.to(tl.int64), axis=1)
                    s2 = tl.sum(x2.to(tl.int64), axis=1)
                    s3 = tl.sum(x3.to(tl.int64), axis=1)
                    s4 = tl.sum(x4.to(tl.int64), axis=1)
                    s5 = tl.sum(x5.to(tl.int64), axis=1)
                    if t == 0:
                        acc00 = madd(acc00, (s0 % PRIME).to(tl.int32))
                        acc01 = madd(acc01, (s1 % PRIME).to(tl.int32))
                        acc02 = madd(acc02, (s2 % PRIME).to(tl.int32))
                        acc03 = madd(acc03, (s3 % PRIME).to(tl.int32))
                        acc04 = madd(acc04, (s4 % PRIME).to(tl.int32))
                        acc05 = madd(acc05, (s5 % PRIME).to(tl.int32))
                    elif t == 1:
                        acc10 = madd(acc10, (s0 % PRIME).to(tl.int32))
                        acc11 = madd(acc11, (s1 % PRIME).to(tl.int32))
                        acc12 = madd(acc12, (s2 % PRIME).to(tl.int32))
                        acc13 = madd(acc13, (s3 % PRIME).to(tl.int32))
                        acc14 = madd(acc14, (s4 % PRIME).to(tl.int32))
                        acc15 = madd(acc15, (s5 % PRIME).to(tl.int32))
                    elif t == 2:
                        acc20 = madd(acc20, (s0 % PRIME).to(tl.int32))
                        acc21 = madd(acc21, (s1 % PRIME).to(tl.int32))
                        acc22 = madd(acc22, (s2 % PRIME).to(tl.int32))
                        acc23 = madd(acc23, (s3 % PRIME).to(tl.int32))
                        acc24 = madd(acc24, (s4 % PRIME).to(tl.int32))
                        acc25 = madd(acc25, (s5 % PRIME).to(tl.int32))
                    else:
                        acc30 = madd(acc30, (s0 % PRIME).to(tl.int32))
                        acc31 = madd(acc31, (s1 % PRIME).to(tl.int32))
                        acc32 = madd(acc32, (s2 % PRIME).to(tl.int32))
                        acc33 = madd(acc33, (s3 % PRIME).to(tl.int32))
                        acc34 = madd(acc34, (s4 % PRIME).to(tl.int32))
                        acc35 = madd(acc35, (s5 % PRIME).to(tl.int32))
        # e_t and the block reduction
        eb = c * 6
        e0, e1, e2, e3, e4, e5 = load6(e_ptr, eb, mc)
        eh0, eh1, eh2, eh3, eh4, eh5 = load6(e_ptr, eb + half * 6, mc)
        de0 = msub(eh0, e0)
        de1 = msub(eh1, e1)
        de2 = msub(eh2, e2)
        de3 = msub(eh3, e3)
        de4 = msub(eh4, e4)
        de5 = msub(eh5, e5)
        for t in tl.static_range(4):
            if t > 0:
                e0 = madd(e0, de0)
                e1 = madd(e1, de1)
                e2 = madd(e2, de2)
                e3 = madd(e3, de3)
                e4 = madd(e4, de4)
                e5 = madd(e5, de5)
            if t == 0:
                h0, h1, h2, h3, h4, h5 = ext_mmul(e0, e1, e2, e3, e4, e5, acc00, acc01, acc02, acc03, acc04, acc05)
            elif t == 1:
                h0, h1, h2, h3, h4, h5 = ext_mmul(e0, e1, e2, e3, e4, e5, acc10, acc11, acc12, acc13, acc14, acc15)
            elif t == 2:
                h0, h1, h2, h3, h4, h5 = ext_mmul(e0, e1, e2, e3, e4, e5, acc20, acc21, acc22, acc23, acc24, acc25)
            else:
                h0, h1, h2, h3, h4, h5 = ext_mmul(e0, e1, e2, e3, e4, e5, acc30, acc31, acc32, acc33, acc34, acc35)
            h0 = tl.where(mc, h0, 0)
            h1 = tl.where(mc, h1, 0)
            h2 = tl.where(mc, h2, 0)
            h3 = tl.where(mc, h3, 0)
            h4 = tl.where(mc, h4, 0)
            h5 = tl.where(mc, h5, 0)
            ob = pid * 24 + t * 6
            tl.store(out_ptr + ob + 0, tl.sum(h0.to(tl.int64), axis=0))
            tl.store(out_ptr + ob + 1, tl.sum(h1.to(tl.int64), axis=0))
            tl.store(out_ptr + ob + 2, tl.sum(h2.to(tl.int64), axis=0))
            tl.store(out_ptr + ob + 3, tl.sum(h3.to(tl.int64), axis=0))
            tl.store(out_ptr + ob + 4, tl.sum(h4.to(tl.int64), axis=0))
            tl.store(out_ptr + ob + 5, tl.sum(h5.to(tl.int64), axis=0))


def phase1_round(a: torch.Tensor, bb: torch.Tensor, e: torch.Tensor, wg: torch.Tensor, half: int, base: bool) -> torch.Tensor:
    """``(4, 6)`` int64 canonical: the GKR phase-1 round polynomial at t = 0..3 (see ``gkr.phase1_round``)."""
    G = a.shape[1]
    BC, BG = 32, 8
    grid = ((half + BC - 1) // BC,)
    partial = torch.empty((grid[0], 4, DEG), dtype=torch.int64, device=e.device)
    wm = _mont(wg[:G].reshape(G, DEG))
    phase1_kernel[grid](a.contiguous().view(-1), bb.contiguous().view(-1), e.contiguous().view(-1), wm.view(-1),
                        partial.view(-1), half, G, BASE=base, BC=BC, BG=BG, num_warps=4)
    s = partial.sum(dim=0) % P
    return (s * R2_MOD_P) % P


def _mont(x: torch.Tensor) -> torch.Tensor:
    """``x R mod p`` (int64, canonical) for a small extension element."""
    return (x.to(torch.int64) * R) % P


def fold(v: torch.Tensor, r: torch.Tensor, base: bool = False, out_dtype=torch.int64) -> torch.Tensor:
    """``v[i] + (v[i + half] - v[i]) r`` over the leading axis.  Extension ``v`` ``(2h, ..., 6)`` -> ``(h, ..., 6)``;
    with ``base`` a base-field ``v`` ``(2h, ...)`` -> ``(h, ..., 6)``.  Canonical in and out."""
    half = v.shape[0] // 2
    v_c = v.contiguous()
    if base:
        n = half * (v_c[0].numel() if v.ndim > 1 else 1)
        out = torch.empty((half,) + tuple(v.shape[1:]) + (DEG,), dtype=out_dtype, device=v.device)
    else:
        n = half * (v_c[0].numel() // DEG)
        out = torch.empty((half,) + tuple(v.shape[1:]), dtype=out_dtype, device=v.device)
    if n == 0:
        return out
    rm = _mont(r.reshape(DEG))
    BLOCK = 256
    grid = ((n + BLOCK - 1) // BLOCK,)
    fold_kernel[grid](v_c.view(-1), rm, out.view(-1), n, BASE=base, BLOCK=BLOCK, num_warps=4)
    return out


def logup_round(p0: torch.Tensor, p1: torch.Tensor, q0: torch.Tensor, q1: torch.Tensor, e: torch.Tensor,
                half: int, lam: torch.Tensor) -> torch.Tensor:
    """``(4, 6)`` int64 canonical: the four evaluations of the LogUp round polynomial (see ``logup.round_poly``)."""
    pbase = p0.ndim == 1
    BLOCK = 128
    grid = ((half + BLOCK - 1) // BLOCK,)
    partial = torch.empty((grid[0], 4, DEG), dtype=torch.int64, device=e.device)
    lm = _mont(lam.reshape(DEG))
    logup_round_kernel[grid](p0.contiguous().view(-1), p1.contiguous().view(-1), q0.contiguous().view(-1),
                             q1.contiguous().view(-1), e.contiguous().view(-1), lm, partial.view(-1), half,
                             PBASE=pbase, BLOCK=BLOCK, num_warps=4)
    s = partial.sum(dim=0) % P                    # < grid * p * ... fits int64 for grid < 2^32
    return (s * R2_MOD_P) % P                     # undo the two Montgomery factors: s R^-2 · R^2


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
