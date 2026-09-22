"""Fused degree-3 sumcheck round over BabyBear^6, plain SIMT integer arithmetic in one Triton kernel.

One launch per round does everything ``ROOFLINE.md`` charges the round for: it reads the three tables of the previous
round once, folds each pair with the previous challenge in registers (``x_e + r (x_o - x_e)``: one extension product),
evaluates the degree-3 round polynomial ``S(0..3) = sum_k f_k(x) g_k(x) h_k(x)`` over pairs of the *folded* elements
(two extension products per point per pair), writes the folded tables once, and emits 24 per-program partial sums.  No
limb decomposition, no INT32 intermediates in HBM.

Layout: a table is ``(deg, t)`` int32 (coefficient-major, values in ``[0, p)``), 24 B per extension element, 4 B per
base element.  Coefficient-major makes every load of a coefficient over a block of elements one contiguous, vectorised
stream; pairs ``(2k, 2k+1)`` are loaded as a ``[BLOCK, 2]`` tile and split.  Arithmetic: 32-bit subtractive Montgomery
(``field.py``), schoolbook 6 x 6 with the ``X^6 = W`` wrap applied to ``W b_j``; everything in int32 registers except the
final block sums (int64).

Modes (constexpr): ``FOLD`` (input pairs are folded with ``r`` first; off for the first round of a layer), ``EVAL`` (the
round polynomial is evaluated; off for the final fold of a layer), ``DEG_IN`` in {1, 6} (base-field tables in the first
two kernels of a layer).
"""
from __future__ import annotations

from typing import Any

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the CPU tests only import field.py
    torch = triton = tl = None

from .. import reference as ref
from . import field

DEG = 6

if triton is not None:
    PRIME = tl.constexpr(field.P)
    MU = tl.constexpr(field.MU_I32)
    WR = tl.constexpr(field.WR)

    @triton.jit
    def mmul(a, b):
        """Montgomery product ``a b R^-1 mod p`` of int32 values in ``[0, p)`` (7 INT32 ops)."""
        lo = a * b
        hi = tl.umulhi(a, b)
        m = lo * MU
        u = tl.umulhi(m, PRIME)
        r = hi - u
        return tl.where(r < 0, r + PRIME, r)

    @triton.jit
    def madd(a, b):
        s = (a - PRIME) + b
        return tl.where(s < 0, s + PRIME, s)

    @triton.jit
    def msub(a, b):
        s = a - b
        return tl.where(s < 0, s + PRIME, s)

    @triton.jit
    def ext_mmul(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5):
        """BabyBear^6 Montgomery product: ``a b R^-1`` coefficientwise (schoolbook, ``X^6 = W`` folded via ``W b_j``)."""
        w1 = mmul(b1, WR)
        w2 = mmul(b2, WR)
        w3 = mmul(b3, WR)
        w4 = mmul(b4, WR)
        w5 = mmul(b5, WR)
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
    def load_pairs(ptr, idx, m, n, c: tl.constexpr):
        """Coefficient ``c`` of elements ``2 idx`` and ``2 idx + 1`` of a ``(deg, n)`` table: two ``[BLOCK]`` vectors."""
        j = tl.arange(0, 2)
        tile = tl.load(ptr + c * n + idx[:, None] * 2 + j[None, :], mask=m[:, None], other=0)
        return tl.split(tile)

    @triton.jit
    def fold_ext(ptr, out_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5):
        """Fold pairs of an extension table: ``e + r (o - e)``; stores and returns the folded coefficients."""
        e0, o0 = load_pairs(ptr, idx, m, n_in, 0)
        e1, o1 = load_pairs(ptr, idx, m, n_in, 1)
        e2, o2 = load_pairs(ptr, idx, m, n_in, 2)
        e3, o3 = load_pairs(ptr, idx, m, n_in, 3)
        e4, o4 = load_pairs(ptr, idx, m, n_in, 4)
        e5, o5 = load_pairs(ptr, idx, m, n_in, 5)
        t0, t1, t2, t3, t4, t5 = ext_mmul(r0, r1, r2, r3, r4, r5,
                                          msub(o0, e0), msub(o1, e1), msub(o2, e2), msub(o3, e3), msub(o4, e4), msub(o5, e5))
        x0 = madd(e0, t0)
        x1 = madd(e1, t1)
        x2 = madd(e2, t2)
        x3 = madd(e3, t3)
        x4 = madd(e4, t4)
        x5 = madd(e5, t5)
        tl.store(out_ptr + 0 * n_out + idx, x0, mask=m)
        tl.store(out_ptr + 1 * n_out + idx, x1, mask=m)
        tl.store(out_ptr + 2 * n_out + idx, x2, mask=m)
        tl.store(out_ptr + 3 * n_out + idx, x3, mask=m)
        tl.store(out_ptr + 4 * n_out + idx, x4, mask=m)
        tl.store(out_ptr + 5 * n_out + idx, x5, mask=m)
        return x0, x1, x2, x3, x4, x5

    @triton.jit
    def fold_base(ptr, out_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5):
        """Fold pairs of a base-field table with an extension challenge (6 base products per element)."""
        e, o = load_pairs(ptr, idx, m, n_in, 0)
        d = msub(o, e)
        x0 = madd(e, mmul(r0, d))
        x1 = mmul(r1, d)
        x2 = mmul(r2, d)
        x3 = mmul(r3, d)
        x4 = mmul(r4, d)
        x5 = mmul(r5, d)
        tl.store(out_ptr + 0 * n_out + idx, x0, mask=m)
        tl.store(out_ptr + 1 * n_out + idx, x1, mask=m)
        tl.store(out_ptr + 2 * n_out + idx, x2, mask=m)
        tl.store(out_ptr + 3 * n_out + idx, x3, mask=m)
        tl.store(out_ptr + 4 * n_out + idx, x4, mask=m)
        tl.store(out_ptr + 5 * n_out + idx, x5, mask=m)
        return x0, x1, x2, x3, x4, x5

    @triton.jit
    def pairs_of(x, HALF: tl.constexpr):
        """A ``[2 HALF]`` vector of consecutive folded elements -> the even and odd ``[HALF]`` vectors."""
        return tl.split(tl.reshape(x, (HALF, 2)))

    @triton.jit
    def at_points(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5):
        """``x(2) = 2b - a`` and ``x(3) = 3b - 2a`` from ``d = b - a``."""
        d0 = msub(b0, a0)
        d1 = msub(b1, a1)
        d2 = msub(b2, a2)
        d3 = msub(b3, a3)
        d4 = msub(b4, a4)
        d5 = msub(b5, a5)
        p0 = madd(b0, d0)
        p1 = madd(b1, d1)
        p2 = madd(b2, d2)
        p3 = madd(b3, d3)
        p4 = madd(b4, d4)
        p5 = madd(b5, d5)
        return p0, p1, p2, p3, p4, p5, madd(p0, d0), madd(p1, d1), madd(p2, d2), madd(p3, d3), madd(p4, d4), madd(p5, d5)

    @triton.jit
    def triple_sum(s_ptr, f0, f1, f2, f3, f4, f5, g0, g1, g2, g3, g4, g5, h0, h1, h2, h3, h4, h5):
        """``sum_k f_k g_k h_k R^-2`` over the block, coefficientwise, stored as 6 int64 at ``s_ptr``."""
        t0, t1, t2, t3, t4, t5 = ext_mmul(f0, f1, f2, f3, f4, f5, g0, g1, g2, g3, g4, g5)
        u0, u1, u2, u3, u4, u5 = ext_mmul(t0, t1, t2, t3, t4, t5, h0, h1, h2, h3, h4, h5)
        tl.store(s_ptr + 0, tl.sum(u0.to(tl.int64), 0))
        tl.store(s_ptr + 1, tl.sum(u1.to(tl.int64), 0))
        tl.store(s_ptr + 2, tl.sum(u2.to(tl.int64), 0))
        tl.store(s_ptr + 3, tl.sum(u3.to(tl.int64), 0))
        tl.store(s_ptr + 4, tl.sum(u4.to(tl.int64), 0))
        tl.store(s_ptr + 5, tl.sum(u5.to(tl.int64), 0))

    @triton.jit
    def eval_ext(s_ptr, fa0, fa1, fa2, fa3, fa4, fa5, fb0, fb1, fb2, fb3, fb4, fb5,
                 ga0, ga1, ga2, ga3, ga4, ga5, gb0, gb1, gb2, gb3, gb4, gb5,
                 ha0, ha1, ha2, ha3, ha4, ha5, hb0, hb1, hb2, hb3, hb4, hb5):
        """The four values of the degree-3 round polynomial over the block's pairs ``(a, b)`` -> 24 int64 at ``s_ptr``."""
        triple_sum(s_ptr + 0, fa0, fa1, fa2, fa3, fa4, fa5, ga0, ga1, ga2, ga3, ga4, ga5, ha0, ha1, ha2, ha3, ha4, ha5)
        triple_sum(s_ptr + 6, fb0, fb1, fb2, fb3, fb4, fb5, gb0, gb1, gb2, gb3, gb4, gb5, hb0, hb1, hb2, hb3, hb4, hb5)
        f20, f21, f22, f23, f24, f25, f30, f31, f32, f33, f34, f35 = at_points(fa0, fa1, fa2, fa3, fa4, fa5, fb0, fb1, fb2, fb3, fb4, fb5)
        g20, g21, g22, g23, g24, g25, g30, g31, g32, g33, g34, g35 = at_points(ga0, ga1, ga2, ga3, ga4, ga5, gb0, gb1, gb2, gb3, gb4, gb5)
        h20, h21, h22, h23, h24, h25, h30, h31, h32, h33, h34, h35 = at_points(ha0, ha1, ha2, ha3, ha4, ha5, hb0, hb1, hb2, hb3, hb4, hb5)
        triple_sum(s_ptr + 12, f20, f21, f22, f23, f24, f25, g20, g21, g22, g23, g24, g25, h20, h21, h22, h23, h24, h25)
        triple_sum(s_ptr + 18, f30, f31, f32, f33, f34, f35, g30, g31, g32, g33, g34, g35, h30, h31, h32, h33, h34, h35)

    @triton.jit
    def fused_round_kernel(f_ptr, g_ptr, h_ptr, fo_ptr, go_ptr, ho_ptr, r_ptr, s_ptr, n_in, n_out,
                           DEG_IN: tl.constexpr, FOLD: tl.constexpr, EVAL: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        if FOLD:
            # BLOCK folded elements per program (= BLOCK input pairs); S over BLOCK/2 pairs of folded elements
            m = idx < n_out
            r0 = tl.load(r_ptr + 0)
            r1 = tl.load(r_ptr + 1)
            r2 = tl.load(r_ptr + 2)
            r3 = tl.load(r_ptr + 3)
            r4 = tl.load(r_ptr + 4)
            r5 = tl.load(r_ptr + 5)
            if DEG_IN == 6:
                f0, f1, f2, f3, f4, f5 = fold_ext(f_ptr, fo_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5)
                g0, g1, g2, g3, g4, g5 = fold_ext(g_ptr, go_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5)
                h0, h1, h2, h3, h4, h5 = fold_ext(h_ptr, ho_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5)
            else:
                f0, f1, f2, f3, f4, f5 = fold_base(f_ptr, fo_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5)
                g0, g1, g2, g3, g4, g5 = fold_base(g_ptr, go_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5)
                h0, h1, h2, h3, h4, h5 = fold_base(h_ptr, ho_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5)
            if EVAL:
                fa0, fb0 = pairs_of(f0, BLOCK // 2)
                fa1, fb1 = pairs_of(f1, BLOCK // 2)
                fa2, fb2 = pairs_of(f2, BLOCK // 2)
                fa3, fb3 = pairs_of(f3, BLOCK // 2)
                fa4, fb4 = pairs_of(f4, BLOCK // 2)
                fa5, fb5 = pairs_of(f5, BLOCK // 2)
                ga0, gb0 = pairs_of(g0, BLOCK // 2)
                ga1, gb1 = pairs_of(g1, BLOCK // 2)
                ga2, gb2 = pairs_of(g2, BLOCK // 2)
                ga3, gb3 = pairs_of(g3, BLOCK // 2)
                ga4, gb4 = pairs_of(g4, BLOCK // 2)
                ga5, gb5 = pairs_of(g5, BLOCK // 2)
                ha0, hb0 = pairs_of(h0, BLOCK // 2)
                ha1, hb1 = pairs_of(h1, BLOCK // 2)
                ha2, hb2 = pairs_of(h2, BLOCK // 2)
                ha3, hb3 = pairs_of(h3, BLOCK // 2)
                ha4, hb4 = pairs_of(h4, BLOCK // 2)
                ha5, hb5 = pairs_of(h5, BLOCK // 2)
                eval_ext(s_ptr + pid * 24, fa0, fa1, fa2, fa3, fa4, fa5, fb0, fb1, fb2, fb3, fb4, fb5,
                         ga0, ga1, ga2, ga3, ga4, ga5, gb0, gb1, gb2, gb3, gb4, gb5,
                         ha0, ha1, ha2, ha3, ha4, ha5, hb0, hb1, hb2, hb3, hb4, hb5)
        else:
            # first round of a layer: S over the input pairs directly, nothing folded
            m = idx < n_in // 2
            if DEG_IN == 6:
                fa0, fb0 = load_pairs(f_ptr, idx, m, n_in, 0)
                fa1, fb1 = load_pairs(f_ptr, idx, m, n_in, 1)
                fa2, fb2 = load_pairs(f_ptr, idx, m, n_in, 2)
                fa3, fb3 = load_pairs(f_ptr, idx, m, n_in, 3)
                fa4, fb4 = load_pairs(f_ptr, idx, m, n_in, 4)
                fa5, fb5 = load_pairs(f_ptr, idx, m, n_in, 5)
                ga0, gb0 = load_pairs(g_ptr, idx, m, n_in, 0)
                ga1, gb1 = load_pairs(g_ptr, idx, m, n_in, 1)
                ga2, gb2 = load_pairs(g_ptr, idx, m, n_in, 2)
                ga3, gb3 = load_pairs(g_ptr, idx, m, n_in, 3)
                ga4, gb4 = load_pairs(g_ptr, idx, m, n_in, 4)
                ga5, gb5 = load_pairs(g_ptr, idx, m, n_in, 5)
                ha0, hb0 = load_pairs(h_ptr, idx, m, n_in, 0)
                ha1, hb1 = load_pairs(h_ptr, idx, m, n_in, 1)
                ha2, hb2 = load_pairs(h_ptr, idx, m, n_in, 2)
                ha3, hb3 = load_pairs(h_ptr, idx, m, n_in, 3)
                ha4, hb4 = load_pairs(h_ptr, idx, m, n_in, 4)
                ha5, hb5 = load_pairs(h_ptr, idx, m, n_in, 5)
                eval_ext(s_ptr + pid * 24, fa0, fa1, fa2, fa3, fa4, fa5, fb0, fb1, fb2, fb3, fb4, fb5,
                         ga0, ga1, ga2, ga3, ga4, ga5, gb0, gb1, gb2, gb3, gb4, gb5,
                         ha0, ha1, ha2, ha3, ha4, ha5, hb0, hb1, hb2, hb3, hb4, hb5)
            else:
                # base x base x base: one coefficient, four values
                fa, fb = load_pairs(f_ptr, idx, m, n_in, 0)
                ga, gb = load_pairs(g_ptr, idx, m, n_in, 0)
                ha, hb = load_pairs(h_ptr, idx, m, n_in, 0)
                fd = msub(fb, fa)
                gd = msub(gb, ga)
                hd = msub(hb, ha)
                f2 = madd(fb, fd)
                g2 = madd(gb, gd)
                h2 = madd(hb, hd)
                f3 = madd(f2, fd)
                g3 = madd(g2, gd)
                h3 = madd(h2, hd)
                tl.store(s_ptr + pid * 24 + 0, tl.sum(mmul(mmul(fa, ga), ha).to(tl.int64), 0))
                tl.store(s_ptr + pid * 24 + 6, tl.sum(mmul(mmul(fb, gb), hb).to(tl.int64), 0))
                tl.store(s_ptr + pid * 24 + 12, tl.sum(mmul(mmul(f2, g2), h2).to(tl.int64), 0))
                tl.store(s_ptr + pid * 24 + 18, tl.sum(mmul(mmul(f3, g3), h3).to(tl.int64), 0))


# ---------------------------------------------------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------------------------------------------------


class FusedTable:
    """``(deg, t)`` int32 on the device, values in ``[0, p)``."""

    def __init__(self, coeffs: Any):
        self.coeffs = coeffs

    @property
    def deg(self) -> int:
        return int(self.coeffs.shape[0])

    @property
    def t(self) -> int:
        return int(self.coeffs.shape[1])

    @classmethod
    def from_ints(cls, values: list[ref.Ext], deg: int, device) -> "FusedTable":
        arr = torch.tensor([[v[c] for v in values] for c in range(deg)], dtype=torch.int32, device=device)
        return cls(arr.reshape(deg, len(values)))

    @classmethod
    def random(cls, t: int, deg: int, device, seed: int) -> "FusedTable":
        gen = torch.Generator(device="cpu").manual_seed(seed)
        return cls(torch.randint(0, field.P, (deg, t), dtype=torch.int32, generator=gen).to(device))

    def to_ints(self) -> list[ref.Ext]:
        a = self.coeffs.cpu().tolist()
        return [tuple(int(a[c][k]) for c in range(self.deg)) + (0,) * (DEG - self.deg) for k in range(self.t)]


def challenge_tensor(r: ref.Ext, device):
    """The challenge pre-scaled by ``R`` as the kernel loads it (6 int32)."""
    return torch.tensor(list(field.to_mont(r)), dtype=torch.int32, device=device)


class FusedRound:
    """Launch wrapper; ``block`` folded elements (or first-round pairs) per program, ``num_warps`` warps."""

    def __init__(self, device, block: int = 256, num_warps: int = 4):
        self.device = device
        self.block = block
        self.num_warps = num_warps

    def alloc_out(self, n_out: int):
        return torch.empty((DEG, n_out), dtype=torch.int32, device=self.device)

    def launch(self, f: FusedTable, g: FusedTable, h: FusedTable, r_mont=None, eval: bool = True, outs=None, s_buf=None):
        """One fused kernel.  ``r_mont`` (6 int32 on device, ``challenge_tensor``) enables the fold; returns
        ``(s_partials (nprog, 24) int64 | None, folded tables | None)``.  Buffers may be passed for graph capture."""
        n_in = f.t
        deg = f.deg
        if g.t != n_in or h.t != n_in or g.deg != deg or h.deg != deg or n_in % 2:
            raise ValueError("tables must have equal even length and equal degree")
        fold = r_mont is not None
        if not fold and not eval:
            raise ValueError("nothing to do")
        if fold:
            n_out = n_in // 2
            if outs is None:
                outs = [self.alloc_out(n_out) for _ in range(3)]
            grid = (triton.cdiv(n_out, self.block),)
        else:
            n_out = 0
            outs = [f.coeffs, f.coeffs, f.coeffs]                              # unused
            grid = (triton.cdiv(n_in // 2, self.block),)
        if s_buf is None:
            s_buf = torch.zeros((grid[0], 24), dtype=torch.int64, device=self.device) if eval else torch.zeros((1, 24), dtype=torch.int64, device=self.device)
        r_arg = r_mont if fold else s_buf
        fused_round_kernel[grid](f.coeffs, g.coeffs, h.coeffs, outs[0], outs[1], outs[2], r_arg, s_buf, n_in, n_out,
                                 DEG_IN=deg, FOLD=fold, EVAL=eval, BLOCK=self.block, num_warps=self.num_warps)
        return (s_buf if eval else None), ([FusedTable(o) for o in outs] if fold else None)

    @staticmethod
    def finish_s(s_partials) -> list[ref.Ext]:
        """Partial sums -> ``S(0), S(1), S(2), S(3)`` as field elements (host: sum, ``mod p``, ``x R^2``)."""
        tot = s_partials.sum(0).cpu().tolist()
        return [tuple(field.fix_s(int(tot[x * 6 + c])) for c in range(6)) for x in range(4)]

    # convenience for tests
    def round(self, f, g, h, r_prev: ref.Ext | None, eval: bool = True):
        r_mont = challenge_tensor(r_prev, self.device) if r_prev is not None else None
        s, outs = self.launch(f, g, h, r_mont, eval)
        return (self.finish_s(s) if s is not None else None), outs
