"""Data-parallel GKR over one layer (``backends/gkr/src/gkr.rs`` ported to torch, BabyBear^6).

Layer claim  Σ_{g,c} w(g) eq(γ,c) V_out(g,c) = v,  V_out(g,c) = A_g(V_in(·,c)) · B_g(V_in(·,c)).
Phase 1  (b copy variables, degree 3): Σ_c eq(γ,c) Σ_g w(g) Ã_g(c) B̃_g(c)                 -> c*
Phase 2a (s_in variables, degree 2):  Σ_l Ṽ(l) P(l),  P(l) = E* Σ_g w(g) B̃_g(c*) A[g,l]      -> l*, sends Ṽ(l*), P̃(l*)
Phase 2b (s_in variables, degree 2):  Σ_r Ṽ(r) R(r),  R(r) = E* Σ_g w(g) Ã_g(l*) B[g,r]      -> r*, sends Ṽ(r*)
with Ṽ(x) = Σ_c eq(c*,c) V_in(x,c) and E* = eq(γ,c*); the verifier evaluates R̃(r*) from the public wiring.
Two claims on V_in at (l*,c*), (r*,c*) merge with a fresh α into the next layer's w.

Phase 1 is the batch-parallel work: the tables ``a[c,g] = Ã_g(c)``, ``bb[c,g]`` are base-field in round 0
(``mm_mod`` of the input rows with the wiring matrices) and extension tables after the first fold; the round
polynomial is evaluated at t = 0..3 chunk by chunk (``Σ_g w_g a_t b_t`` is an exact float64 limb matmul, the
``a_t · b_t`` products are the elementwise extension multiplications the fused kernel of step 4 replaces).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import field as _field
from .circuit import Layer
from .field import (
    DEG,
    Ext,
    P,
    base_contract_ext,
    e_add,
    e_mul,
    eq_point,
    eq_table,
    ext_contract,
    ext_from_base,
    ext_mul,
    ext_sum,
    fold,
    fold_base,
    interpolate,
    mm_mod,
    py_ext,
    py_exts,
    t_ext,
)
from .transcript import Transcript


class Reader:
    def __init__(self, msgs: list[Ext]) -> None:
        self.msgs = msgs
        self.pos = 0

    def next(self) -> Ext:
        if self.pos >= len(self.msgs):
            raise VerifyError("proof truncated")
        v = self.msgs[self.pos]
        self.pos += 1
        return v

    def take(self, n: int) -> list[Ext]:
        return [self.next() for _ in range(n)]


class VerifyError(Exception):
    pass


@dataclass
class LayerClaim:
    w: torch.Tensor | None          # (>= g_n, 6)
    copy_point: list[Ext]
    value: Ext


FinalClaims = tuple[list[Ext], Ext, list[Ext], Ext]

# rows of the phase-1 tables processed per chunk (extension tables: (chunk, g_n, 36) int64 temporaries)
CHUNK_EXT = 2048
CHUNK_BASE = 32768


def _lerp_base(x0: torch.Tensor, x1: torch.Tensor, t: int) -> torch.Tensor:
    if t == 0:
        return x0
    if t == 1:
        return x1
    return (x0 + (x1 - x0) * t) % P


def _lerp_ext(x0: torch.Tensor, x1: torch.Tensor, t: int) -> torch.Tensor:
    return _lerp_base(x0, x1, t)      # coordinate-wise, a scalar integer t


def phase1_round(a: torch.Tensor, bb: torch.Tensor, e: torch.Tensor, wg: torch.Tensor, half: int, base: bool) -> list[Ext]:
    """``h(t) = Σ_{c<half} E_t[c] Σ_g w[g] a_t[c,g] b_t[c,g]`` at t = 0..3."""
    dev = a.device
    if _field._fast_phase1 is not None and a.is_cuda:
        return py_exts(_field._fast_phase1(a, bb, e, wg, half, base))
    h = torch.zeros((4, DEG), dtype=torch.int64, device=dev)
    chunk = CHUNK_BASE if base else CHUNK_EXT
    for lo in range(0, half, chunk):
        hi = min(half, lo + chunk)
        a0, a1 = a[lo:hi].to(torch.int64), a[half + lo:half + hi].to(torch.int64)
        b0, b1 = bb[lo:hi].to(torch.int64), bb[half + lo:half + hi].to(torch.int64)
        e0, e1 = e[lo:hi].to(torch.int64), e[half + lo:half + hi].to(torch.int64)
        for t in range(4):
            at = _lerp_base(a0, a1, t)
            bt = _lerp_base(b0, b1, t)
            if base:
                inner = base_contract_ext((at * bt) % P, wg)          # (rows, 6)
            else:
                inner = ext_contract(ext_mul(at, bt), wg)
            et = _lerp_ext(e0, e1, t)
            h[t] = (h[t] + ext_sum(ext_mul(et, inner), dim=0)) % P
    return py_exts(h)


def sumcheck_prod(u: torch.Tensor, w: torch.Tensor, tr: Transcript, out: list[Ext]) -> tuple[list[Ext], torch.Tensor, torch.Tensor]:
    """Degree-2 sumcheck of ``Σ_x U(x) W(x)``; returns the bound point and the folded (length-1) tables."""
    point: list[Ext] = []
    while u.shape[0] > 1:
        half = u.shape[0] // 2
        u0, u1, w0, w1 = u[:half], u[half:], w[:half], w[half:]
        s0 = ext_sum(ext_mul(u0, w0))
        s1 = ext_sum(ext_mul(u1, w1))
        s2 = ext_sum(ext_mul((2 * u1 - u0) % P, (2 * w1 - w0) % P))
        s = [py_ext(s0), py_ext(s1), py_ext(s2)]
        out.extend(s)
        tr.absorb_exts(s)
        r = tr.challenge()
        point.append(r)
        rt = t_ext(r, u.device)
        u = fold(u, rt)
        w = fold(w, rt)
    return point, u, w


def prove_layer(layer: Layer, rows: torch.Tensor, b: int, claim: LayerClaim, tr: Transcript, out: list[Ext], last: bool) -> tuple[LayerClaim, FinalClaims | None]:
    """``rows``: the layer's input for every copy, ``(2^b, 2^s_in)`` base (zero rows past the real units)."""
    dev = rows.device
    copies = 1 << b
    g_n = layer.g_n
    wg = claim.w[:g_n].to(torch.int64)
    # phase-1 tables (base field), chunked over copies
    a = torch.empty((copies, g_n), dtype=torch.int64, device=dev)
    bb = torch.empty((copies, g_n), dtype=torch.int64, device=dev)
    for lo in range(0, copies, CHUNK_BASE):
        hi = min(copies, lo + CHUNK_BASE)
        a[lo:hi], bb[lo:hi] = layer.eval_gates(rows[lo:hi].to(torch.int64))
    e = eq_table(claim.copy_point, dev)
    c_star: list[Ext] = []
    length = copies
    base = True
    for _round in range(b):
        half = length // 2
        h = phase1_round(a, bb, e, wg, half, base)
        out.extend(h)
        tr.absorb_exts(h)
        r = tr.challenge()
        c_star.append(r)
        rt = t_ext(r, dev)
        if base:
            a, bb = fold_base(a, rt), fold_base(bb, rt)
            base = False
        else:
            a, bb = fold(a, rt), fold(bb, rt)
        e = fold(e, rt)
        length = half
    e_star = e[0]
    if base:                         # b == 0: single copy, tables still base
        a, bb = ext_from_base(a), ext_from_base(bb)
    b_star = bb[0]                   # (g_n, 6)
    del a, bb
    # Ṽ(·) = Σ_c eq(c*, c) V_in(·, c)
    ec = eq_table(c_star, dev)                                   # (copies, 6)
    vf = mm_mod(ec.T, rows.to(torch.int64)).T.contiguous()       # (n_in, 6)
    wp = ext_mul(wg, e_star)                                     # (g_n, 6)
    A, Bm = layer.matrices(dev)
    # phase 2a
    p = mm_mod(A.T, ext_mul(wp, b_star))                          # (n_in, 6)
    v2 = vf.clone()
    l_star, vfin, pfin = sumcheck_prod(vf, p, tr, out)
    vl, pl = py_ext(vfin[0]), py_ext(pfin[0])
    out.extend([vl, pl])
    tr.absorb_exts([vl, pl])
    # phase 2b
    el = eq_table(l_star, dev)
    wa = ext_mul(wp, layer.eval_a_at(el))
    rr = mm_mod(Bm.T, wa)
    r_star, v2fin, _ = sumcheck_prod(v2, rr, tr, out)
    vr = py_ext(v2fin[0])
    out.append(vr)
    tr.absorb_ext(vr)
    if last:
        return LayerClaim(None, c_star, (0,) * DEG), (l_star, vl, r_star, vr)
    alpha = tr.challenge()
    er = eq_table(r_star, dev)
    nw = (el + ext_mul(er, t_ext(alpha, dev))) % P
    return LayerClaim(nw, c_star, e_add(vl, e_mul(alpha, vr))), None


def verify_sumcheck_prod(nvars: int, running: Ext, tr: Transcript, rd: Reader, what: str) -> tuple[list[Ext], Ext]:
    point = []
    for rnd in range(nvars):
        s = rd.take(3)
        if e_add(s[0], s[1]) != running:
            raise VerifyError(f"{what}: round {rnd} sum mismatch")
        tr.absorb_exts(s)
        r = tr.challenge()
        running = interpolate(s, r)
        point.append(r)
    return point, running


def verify_layer(layer: Layer, b: int, claim: LayerClaim, tr: Transcript, rd: Reader, last: bool, device) -> tuple[LayerClaim, FinalClaims | None]:
    running = claim.value
    c_star: list[Ext] = []
    for rnd in range(b):
        h = rd.take(4)
        if e_add(h[0], h[1]) != running:
            raise VerifyError(f"{layer.name}: phase-1 round {rnd} sum mismatch")
        tr.absorb_exts(h)
        r = tr.challenge()
        running = interpolate(h, r)
        c_star.append(r)
    e_star = eq_point(claim.copy_point, c_star)
    wp = ext_mul(claim.w[: layer.g_n], t_ext(e_star, device))
    l_star, after_l = verify_sumcheck_prod(layer.s_in, running, tr, rd, f"{layer.name} phase-2a")
    vl = rd.next()
    pl = rd.next()
    if e_mul(vl, pl) != after_l:
        raise VerifyError(f"{layer.name}: phase-2a final check")
    tr.absorb_exts([vl, pl])
    r_star, after_r = verify_sumcheck_prod(layer.s_in, pl, tr, rd, f"{layer.name} phase-2b")
    vr = rd.next()
    tr.absorb_ext(vr)
    el = eq_table(l_star, device)
    er = eq_table(r_star, device)
    rt = ext_sum(ext_mul(ext_mul(wp, layer.eval_a_at(el)), layer.eval_b_at(er)))
    if e_mul(vr, py_ext(rt)) != after_r:
        raise VerifyError(f"{layer.name}: phase-2b final check (wiring)")
    if last:
        return LayerClaim(None, c_star, (0,) * DEG), (l_star, vl, r_star, vr)
    alpha = tr.challenge()
    nw = (el + ext_mul(er, t_ext(alpha, device))) % P
    return LayerClaim(nw, c_star, e_add(vl, e_mul(alpha, vr))), None


def eval_columns(cols: torch.Tensor, s_c: int, x: list[Ext], c: list[Ext]) -> Ext:
    """The clear-witness verifier's evaluation of the input layer ``W̃(x, c)`` (padding rows/cols zero)."""
    dev = cols.device
    units, ncols = cols.shape
    ec = eq_table(c, dev)[:units]
    ex = eq_table(x, dev)[:ncols]
    inner = mm_mod(cols.to(torch.int64), ex)          # (units, 6)
    return py_ext(ext_sum(ext_mul(inner, ec)))
