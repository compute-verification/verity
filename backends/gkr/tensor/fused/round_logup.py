"""Fused round of the logUp fractional-sumcheck GKR (PROTOCOL.md 4.2), SIMT arithmetic, same fusion as ``round_simt``.

One GKR layer of the fraction tree reduces the claims on ``p_k, q_k`` to the layer below through the sumcheck

    sum_x eq(z, x) [ p_l(x) q_r(x) + p_r(x) q_l(x) + lambda q_l(x) q_r(x) ]

(``l, r`` = the two children, ``lambda`` the verifier's claim-combining challenge): degree 3 per variable, **five
tables** ``eq, p_l, q_l, p_r, q_r`` in BabyBear^6, four values per round.  Per pair and point: 5 extension products
(``p_l q_r``, ``p_r q_l``, ``q_l q_r``, ``lambda .``, ``eq .``) -> 20 per pair, plus 5 folds per input pair: ``7.5 n`` products
and ``180 n`` bytes per kernel of ``n`` input elements -- 1.3x the products per byte of the checker round (``ROOFLINE.md``
arithmetic: SIMT needs ~80% of the INT32 peak to stay memory-bound here).

Reference: ``logup_reference`` below (pure Python on ``reference.py``); the kernel is bit-exact against it (``tests``).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .. import reference as ref
from . import field
from .round_simt import (
    DEG,
    FusedTable,
    at_points,
    challenge_tensor,
    ext_mmul,
    fold_ext,
    load_pairs,
    madd,
    pairs_of,
)

PRIME = tl.constexpr(field.P)


def logup_reference(eq, pl, ql, pr, qr, lam: ref.Ext, r_prev: ref.Ext | None):
    """Fold the five tables with ``r_prev`` (if given), then ``S(0..3)``; returns ``(S, folded tables)``."""
    tabs = [eq, pl, ql, pr, qr]
    if r_prev is not None:
        tabs = [ref.fold(tb, r_prev) for tb in tabs]
    eq, pl, ql, pr, qr = tabs
    S = []
    for x in range(4):
        s = ref.ZERO
        for k in range(len(eq) // 2):
            e, a, b, c, d = (ref.eval_at(tb[2 * k], tb[2 * k + 1], x) for tb in tabs)
            u = ref.add(ref.add(ref.mul(a, d), ref.mul(c, b)), ref.mul(lam, ref.mul(b, d)))
            s = ref.add(s, ref.mul(e, u))
        S.append(s)
    return S, tabs


@triton.jit
def logup_point(s_ptr, lam0, lam1, lam2, lam3, lam4, lam5,
                e0, e1, e2, e3, e4, e5, a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5,
                c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5):
    """``sum eq (a d + c b + lambda b d) R^-2`` over the block (a = p_l, b = q_l, c = p_r, d = q_r) -> 6 int64."""
    t0, t1, t2, t3, t4, t5 = ext_mmul(a0, a1, a2, a3, a4, a5, d0, d1, d2, d3, d4, d5)
    u0, u1, u2, u3, u4, u5 = ext_mmul(c0, c1, c2, c3, c4, c5, b0, b1, b2, b3, b4, b5)
    v0, v1, v2, v3, v4, v5 = ext_mmul(b0, b1, b2, b3, b4, b5, d0, d1, d2, d3, d4, d5)
    w0, w1, w2, w3, w4, w5 = ext_mmul(lam0, lam1, lam2, lam3, lam4, lam5, v0, v1, v2, v3, v4, v5)
    x0 = madd(madd(t0, u0), w0)
    x1 = madd(madd(t1, u1), w1)
    x2 = madd(madd(t2, u2), w2)
    x3 = madd(madd(t3, u3), w3)
    x4 = madd(madd(t4, u4), w4)
    x5 = madd(madd(t5, u5), w5)
    y0, y1, y2, y3, y4, y5 = ext_mmul(e0, e1, e2, e3, e4, e5, x0, x1, x2, x3, x4, x5)
    tl.store(s_ptr + 0, tl.sum(y0.to(tl.int64), 0))
    tl.store(s_ptr + 1, tl.sum(y1.to(tl.int64), 0))
    tl.store(s_ptr + 2, tl.sum(y2.to(tl.int64), 0))
    tl.store(s_ptr + 3, tl.sum(y3.to(tl.int64), 0))
    tl.store(s_ptr + 4, tl.sum(y4.to(tl.int64), 0))
    tl.store(s_ptr + 5, tl.sum(y5.to(tl.int64), 0))


@triton.jit
def logup_eval(s_ptr, lam_ptr,
               ea0, ea1, ea2, ea3, ea4, ea5, eb0, eb1, eb2, eb3, eb4, eb5,
               aa0, aa1, aa2, aa3, aa4, aa5, ab0, ab1, ab2, ab3, ab4, ab5,
               ba0, ba1, ba2, ba3, ba4, ba5, bb0, bb1, bb2, bb3, bb4, bb5,
               ca0, ca1, ca2, ca3, ca4, ca5, cb0, cb1, cb2, cb3, cb4, cb5,
               da0, da1, da2, da3, da4, da5, db0, db1, db2, db3, db4, db5):
    lam0 = tl.load(lam_ptr + 0)
    lam1 = tl.load(lam_ptr + 1)
    lam2 = tl.load(lam_ptr + 2)
    lam3 = tl.load(lam_ptr + 3)
    lam4 = tl.load(lam_ptr + 4)
    lam5 = tl.load(lam_ptr + 5)
    logup_point(s_ptr + 0, lam0, lam1, lam2, lam3, lam4, lam5, ea0, ea1, ea2, ea3, ea4, ea5, aa0, aa1, aa2, aa3, aa4, aa5,
                ba0, ba1, ba2, ba3, ba4, ba5, ca0, ca1, ca2, ca3, ca4, ca5, da0, da1, da2, da3, da4, da5)
    logup_point(s_ptr + 6, lam0, lam1, lam2, lam3, lam4, lam5, eb0, eb1, eb2, eb3, eb4, eb5, ab0, ab1, ab2, ab3, ab4, ab5,
                bb0, bb1, bb2, bb3, bb4, bb5, cb0, cb1, cb2, cb3, cb4, cb5, db0, db1, db2, db3, db4, db5)
    e20, e21, e22, e23, e24, e25, e30, e31, e32, e33, e34, e35 = at_points(ea0, ea1, ea2, ea3, ea4, ea5, eb0, eb1, eb2, eb3, eb4, eb5)
    a20, a21, a22, a23, a24, a25, a30, a31, a32, a33, a34, a35 = at_points(aa0, aa1, aa2, aa3, aa4, aa5, ab0, ab1, ab2, ab3, ab4, ab5)
    b20, b21, b22, b23, b24, b25, b30, b31, b32, b33, b34, b35 = at_points(ba0, ba1, ba2, ba3, ba4, ba5, bb0, bb1, bb2, bb3, bb4, bb5)
    c20, c21, c22, c23, c24, c25, c30, c31, c32, c33, c34, c35 = at_points(ca0, ca1, ca2, ca3, ca4, ca5, cb0, cb1, cb2, cb3, cb4, cb5)
    d20, d21, d22, d23, d24, d25, d30, d31, d32, d33, d34, d35 = at_points(da0, da1, da2, da3, da4, da5, db0, db1, db2, db3, db4, db5)
    logup_point(s_ptr + 12, lam0, lam1, lam2, lam3, lam4, lam5, e20, e21, e22, e23, e24, e25, a20, a21, a22, a23, a24, a25,
                b20, b21, b22, b23, b24, b25, c20, c21, c22, c23, c24, c25, d20, d21, d22, d23, d24, d25)
    logup_point(s_ptr + 18, lam0, lam1, lam2, lam3, lam4, lam5, e30, e31, e32, e33, e34, e35, a30, a31, a32, a33, a34, a35,
                b30, b31, b32, b33, b34, b35, c30, c31, c32, c33, c34, c35, d30, d31, d32, d33, d34, d35)


@triton.jit
def fold_or_load(ptr, out_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5, FOLD: tl.constexpr, HALF: tl.constexpr):
    """Folded-and-split (FOLD) or plain (first round) pairs of one table: 12 ``[HALF]`` vectors (even, odd x 6)."""
    if FOLD:
        x0, x1, x2, x3, x4, x5 = fold_ext(ptr, out_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5)
        a0, b0 = pairs_of(x0, HALF)
        a1, b1 = pairs_of(x1, HALF)
        a2, b2 = pairs_of(x2, HALF)
        a3, b3 = pairs_of(x3, HALF)
        a4, b4 = pairs_of(x4, HALF)
        a5, b5 = pairs_of(x5, HALF)
    else:
        pidx = tl.program_id(0) * HALF + tl.arange(0, HALF)                # HALF pairs per program in the first round
        pm = pidx < n_in // 2
        a0, b0 = load_pairs(ptr, pidx, pm, n_in, 0)
        a1, b1 = load_pairs(ptr, pidx, pm, n_in, 1)
        a2, b2 = load_pairs(ptr, pidx, pm, n_in, 2)
        a3, b3 = load_pairs(ptr, pidx, pm, n_in, 3)
        a4, b4 = load_pairs(ptr, pidx, pm, n_in, 4)
        a5, b5 = load_pairs(ptr, pidx, pm, n_in, 5)
    return a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5


@triton.jit
def logup_round_kernel(e_ptr, a_ptr, b_ptr, c_ptr, d_ptr, eo_ptr, ao_ptr, bo_ptr, co_ptr, do_ptr, r_ptr, lam_ptr, s_ptr, n_in, n_out,
                       FOLD: tl.constexpr, EVAL: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n_out
    r0 = tl.load(r_ptr + 0)
    r1 = tl.load(r_ptr + 1)
    r2 = tl.load(r_ptr + 2)
    r3 = tl.load(r_ptr + 3)
    r4 = tl.load(r_ptr + 4)
    r5 = tl.load(r_ptr + 5)
    if FOLD and not EVAL:
        fold_ext(e_ptr, eo_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5)
        fold_ext(a_ptr, ao_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5)
        fold_ext(b_ptr, bo_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5)
        fold_ext(c_ptr, co_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5)
        fold_ext(d_ptr, do_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5)
    else:
        ea0, ea1, ea2, ea3, ea4, ea5, eb0, eb1, eb2, eb3, eb4, eb5 = fold_or_load(e_ptr, eo_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5, FOLD, BLOCK // 2)
        aa0, aa1, aa2, aa3, aa4, aa5, ab0, ab1, ab2, ab3, ab4, ab5 = fold_or_load(a_ptr, ao_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5, FOLD, BLOCK // 2)
        ba0, ba1, ba2, ba3, ba4, ba5, bb0, bb1, bb2, bb3, bb4, bb5 = fold_or_load(b_ptr, bo_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5, FOLD, BLOCK // 2)
        ca0, ca1, ca2, ca3, ca4, ca5, cb0, cb1, cb2, cb3, cb4, cb5 = fold_or_load(c_ptr, co_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5, FOLD, BLOCK // 2)
        da0, da1, da2, da3, da4, da5, db0, db1, db2, db3, db4, db5 = fold_or_load(d_ptr, do_ptr, idx, m, n_in, n_out, r0, r1, r2, r3, r4, r5, FOLD, BLOCK // 2)
        logup_eval(s_ptr + pid * 24, lam_ptr, ea0, ea1, ea2, ea3, ea4, ea5, eb0, eb1, eb2, eb3, eb4, eb5,
                   aa0, aa1, aa2, aa3, aa4, aa5, ab0, ab1, ab2, ab3, ab4, ab5, ba0, ba1, ba2, ba3, ba4, ba5, bb0, bb1, bb2, bb3, bb4, bb5,
                   ca0, ca1, ca2, ca3, ca4, ca5, cb0, cb1, cb2, cb3, cb4, cb5, da0, da1, da2, da3, da4, da5, db0, db1, db2, db3, db4, db5)


class LogUpRound:
    """Five extension tables ``eq, p_l, q_l, p_r, q_r``; ``block`` = folded elements per program (FOLD) or ``2 x`` pairs."""

    name = "logup-simt"
    NTABLES = 5

    def __init__(self, device, block: int = 256, num_warps: int = 4):
        self.device = device
        self.block = block
        self.num_warps = num_warps

    def alloc_out(self, n_out: int):
        return torch.empty((DEG, n_out), dtype=torch.int32, device=self.device)

    def grid_size(self, n_in: int, fold: bool) -> int:
        return triton.cdiv(n_in // 2, self.block if fold else self.block // 2)

    def challenge(self, r: ref.Ext):
        return challenge_tensor(r, self.device)

    def launch(self, tables: list[FusedTable], lam_mont, r_mont=None, eval: bool = True, outs=None, s_buf=None):
        n_in = tables[0].t
        if len(tables) != 5 or any(t.t != n_in or t.deg != DEG for t in tables) or n_in % 2:
            raise ValueError("five extension tables of equal even length")
        fold = r_mont is not None
        if not fold and not eval:
            raise ValueError("nothing to do")
        grid = (self.grid_size(n_in, fold),)
        n_out = n_in // 2 if fold else 0
        if fold and outs is None:
            outs = [self.alloc_out(n_out) for _ in range(5)]
        if not fold:
            outs = [t.coeffs for t in tables]
        if s_buf is None:
            s_buf = torch.zeros((grid[0] if eval else 1, 24), dtype=torch.int64, device=self.device)
        logup_round_kernel[grid](*(t.coeffs for t in tables), *outs, r_mont if fold else lam_mont, lam_mont, s_buf, n_in, n_out,
                                 FOLD=fold, EVAL=eval, BLOCK=self.block, num_warps=self.num_warps)
        return (s_buf if eval else None), ([FusedTable(o) for o in outs] if fold else None)

    def round(self, tables, lam: ref.Ext, r_prev: ref.Ext | None, eval: bool = True):
        s, outs = self.launch(tables, self.challenge(lam), self.challenge(r_prev) if r_prev is not None else None, eval)
        from .round_simt import FusedRound
        return (FusedRound.finish_s(s) if s is not None else None), outs
