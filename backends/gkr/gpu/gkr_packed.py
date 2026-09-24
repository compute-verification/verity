"""GKR phase 1 (the copy-variable sumcheck) on the a-packed2 kernels, producing a-gpu's messages bit for bit.

Phase 1 of a layer is ``Σ_c eq(γ, c) Σ_g w(g) a[c, g] b[c, g]`` over ``b`` copy variables (``gkr.prove_layer``), bound
top bit first.  The packed sumcheck (``../packed/sumcheck_packed.py``) is ``Σ_j E(j) g(j) h(j)`` over one index with
the weight ``E`` a tensor product ``e_lo(b1) ⊗ e_hi(b2)`` of the low ``κ`` bits and the rest -- and ``e_hi`` is an
arbitrary vector.  Lay the tables out as ``j = g · 2^b + rev_b(c)`` (gate-major, copy index bit-reversed): binding
bit ``i-1`` of ``j`` is binding the top remaining bit of ``c`` (a-gpu's order), ``eq(γ, c)`` with ``γ[0]`` on the top bit
of ``c`` is the packed ``eq(gammas, ·)`` with ``gammas = γ`` on the low bits of ``j``, and the weight of the un-bound
part is ``w(g) ⊗ eq(γ[i+κ:], c_rest)`` = ``e_hi``.  The gate variables are never bound (they are the ``K2`` batch);
after the ``b`` rounds the tables are ``a_g(c*)``, ``b_g(c*)`` for every gate -- what phase 2 needs.  So the first
``k = 3`` rounds run on the *base* gate tables (``a[c, g] b[c, g]`` products are base x base, the 4^(i-1) packing of
Bagad-Domb-Thaler), the rest on ``(t / 8, 6)`` extension tables; the round messages equal ``gkr.phase1_round``'s.
"""
from __future__ import annotations

import torch

from .field import DEG, ONE, P, ZERO, Ext, e_inv, e_mul, e_sub, ext_mul, py_exts, t_ext

try:
    from packed import reference as _ref
    from packed import sumcheck_packed as _sp
    AVAILABLE = True
except ImportError:  # pragma: no cover
    try:
        from backends.gkr.packed import reference as _ref
        from backends.gkr.packed import sumcheck_packed as _sp
        AVAILABLE = True
    except ImportError:
        AVAILABLE = False

_REV: dict = {}


def _bitrev(b: int, device) -> torch.Tensor:
    key = (b, str(device))
    if key not in _REV:
        n = 1 << b
        idx = torch.arange(n)
        rev = torch.zeros(n, dtype=torch.int64)
        for bit in range(b):
            rev |= ((idx >> bit) & 1) << (b - 1 - bit)
        _REV[key] = rev.to(device)
    return _REV[key]


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _layout_kernel(a_ptr, rev_ptr, out_ptr, copies, g_n, BC: tl.constexpr, BG: tl.constexpr):
        """``out[g, j] = a[rev[j], g]`` on ``BC x BG`` tiles: the gather-load is coalesced along ``g``, the store along ``j``."""
        pj = tl.program_id(0)
        pg = tl.program_id(1)
        j = pj * BC + tl.arange(0, BC)
        g = pg * BG + tl.arange(0, BG)
        mj = j < copies
        mg = g < g_n
        c = tl.load(rev_ptr + j, mask=mj, other=0)
        v = tl.load(a_ptr + c[:, None] * g_n + g[None, :], mask=mj[:, None] & mg[None, :], other=0)
        tl.store(out_ptr + g[None, :] * copies + j[:, None], v.to(tl.int32), mask=mj[:, None] & mg[None, :])

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover
    _HAVE_TRITON = False


def _layout(a: torch.Tensor, b: int) -> torch.Tensor:
    """``(2^b, g_n)`` int64 base -> ``(g_n · 2^b,)`` int32 with ``j = g · 2^b + rev_b(c)``."""
    rev = _bitrev(b, a.device)
    copies, g_n = a.shape
    if _HAVE_TRITON and a.is_cuda:
        a = a.contiguous()
        out = torch.empty((g_n, copies), dtype=torch.int32, device=a.device)
        BC, BG = 64, 64
        grid = ((copies + BC - 1) // BC, (g_n + BG - 1) // BG)
        _layout_kernel[grid](a, rev, out, copies, g_n, BC=BC, BG=BG, num_warps=4)
        return out.view(-1)
    return a.to(torch.int32)[rev].t().contiguous().view(-1)


class _EqSlices:
    """``eq(γ[i:i+κ], b1)`` and ``eq(γ[i+κ:], c)`` for every round as strided slices of the one full table
    ``E[j] = eq(γ, j)`` (``γ[j]`` on bit ``j``): ``E[x << i] = eq(γ[:i], 0) · eq(γ[i:i+n], x) · eq(γ[i+n:], 0)`` for
    ``x < 2^n``, so the window tables are ``E[:: 2^i][: 2^n]`` rescaled by the inverse of the constant
    ``Π_{j<i} (1 - γ_j) · Π_{j>=i+n} (1 - γ_j)`` (the ``(1 - γ_j)^-1`` batch-inverted on the host once per layer) -- no
    per-round doubling loops."""

    def __init__(self, E: torch.Tensor, gammas: list[Ext]) -> None:
        self.E = E
        b = len(gammas)
        one_m = [e_sub(ONE, gm) for gm in gammas]
        pref = [ONE]
        for x in one_m:
            pref.append(e_mul(pref[-1], x))                            # pref[i] = Π_{j<i} (1 - γ_j)
        inv_all = e_inv(pref[b])
        inv1 = [ZERO] * b
        for j in range(b - 1, -1, -1):                                 # (1 - γ_j)^-1 = pref[j+1]^-1 · pref[j]
            inv1[j] = e_mul(inv_all, pref[j])
            inv_all = e_mul(inv_all, one_m[j])
        self.invpref = [ONE]
        for x in inv1:
            self.invpref.append(e_mul(self.invpref[-1], x))            # Π_{j<i} (1 - γ_j)^-1
        self.invsuf = [ONE] * (b + 1)
        for i in range(b - 1, -1, -1):
            self.invsuf[i] = e_mul(self.invsuf[i + 1], inv1[i])        # Π_{j>=i} (1 - γ_j)^-1

    def suffix(self, i: int, n: int) -> torch.Tensor:
        """``eq(γ[i:i+n], x)`` for ``x < 2^n`` as ``(2^n, 6)`` int64 (canonical)."""
        sl = self.E[:: 1 << i][: 1 << n]
        return ext_mul(sl, t_ext(e_mul(self.invpref[i], self.invsuf[i + n]), self.E.device))


def _inputs(kern, gammas: list[Ext], rs: list[Ext], i: int, k: int, kappa: int, wgt: torch.Tensor, t: int,
            eqs: _EqSlices, pfx: Ext | None = None):
    """``Kernel.inputs`` with ``e_hi = w(g) ⊗ eq(γ[i+κ:], c_rest)``; ``pfx``: ``eq(γ[:i-1], rs)`` if the caller keeps it."""
    b = len(gammas)
    dev = kern.device
    m = t >> i
    K1 = 1 << kappa
    K2 = m // K1
    e_lo = eqs.suffix(i, kappa).to(torch.int32)                                # (K1, 6)
    eq_c = eqs.suffix(i + kappa, b - i - kappa)                                # (2^(b-i-κ), 6)
    e_hi = ext_mul(wgt[:, None, :], eq_c[None, :, :]).reshape(K2, DEG).to(torch.int32)
    gm = gammas[i - 1]
    eq_x = _sp.to_dev([_ref.eval_at(_ref.sub(_ref.ONE, gm), gm, x) for x in range(4)], dev)
    prefix = _sp.to_dev(_ref.eq_points(gammas[:i - 1], rs) if pfx is None else pfx, dev)
    coeff = None
    if i <= k:
        eq_r = _ref.eq_table(rs)
        coeff = _sp.to_dev([_ref.mul(x, y) for x in eq_r for y in eq_r], dev)
    if kern._tk is not None:      # Montgomery Hadamards: the weight limbs carry the compensating R
        e_lo = _sp.pre_scale(e_lo)
    return _sp.RoundInputs(_sp.decompose(e_lo), e_hi, eq_x, prefix, coeff, None)


def phase1(kern, a: torch.Tensor, bb: torch.Tensor, wg: torch.Tensor, gammas: list[Ext], e: torch.Tensor, tr,
           out: list[Ext], k: int = 3) -> tuple[list[Ext], torch.Tensor, torch.Tensor]:
    """The ``b = len(gammas)`` phase-1 rounds on base gate tables ``a, bb`` ``(2^b, g_n)`` int64 with gate weights
    ``wg (g_n, 6)`` int64 and ``e = field.eq_table(gammas)`` ``(2^b, 6)`` (top-bit-first); absorbs / draws on ``tr``
    exactly as ``gkr.prove_layer``.  Returns ``(c*, a(c*), b(c*))`` with the two tables ``(g_n, 6)`` int64."""
    copies, g_n = a.shape
    b = len(gammas)
    if copies != 1 << b:
        raise ValueError("copies must be 2^b")
    dev = a.device
    t = g_n * copies
    k = min(k, b)
    g, h = _layout(a, b), _layout(bb, b)
    eqs = _EqSlices(e[_bitrev(b, dev)], gammas)
    wgt = (wg[:g_n].to(torch.int64) % P).contiguous()
    saved = kern.k1_log2
    rs: list[Ext] = []
    pfx = _ref.ONE
    gE = hE = None
    try:
        for i in range(1, b + 1):
            kappa = min(saved, b - i)          # b1 = the next κ copy bits; the gate index lives in b2
            kern.k1_log2 = kappa
            inp = _inputs(kern, gammas, rs, i, k, kappa, wgt, t, eqs, pfx)
            if i <= k:
                S = kern.packed_round(g, h, i, inp)
            else:
                S, _, _ = kern.ext_round(gE, hE, i, inp, fold=False)
            s_py = py_exts(S.to(torch.int64))
            out.extend(s_py)
            tr.absorb_exts(s_py)
            r = tr.challenge()
            rs.append(r)
            pfx = _ref.mul(pfx, _ref.eq1(gammas[i - 1], r))
            if i == k:
                kern.set_fold(inp, rs[:-1], k, k, r)
                gE, hE = kern.fold_k(g, k, inp), kern.fold_k(h, k, inp)
                del g, h
            elif i > k:
                kern.set_fold(inp, rs[:-1], i, k, r)
                gE, hE = kern.fold_pair(gE, hE, inp)
        if k == 0:
            gE = torch.zeros((t, DEG), dtype=torch.int32, device=dev)
            gE[:, 0] = g
            hE = torch.zeros_like(gE)
            hE[:, 0] = h
    finally:
        kern.k1_log2 = saved
    return rs, gE.to(torch.int64), hE.to(torch.int64)


def fits(a: torch.Tensor, b: int) -> bool:
    """Use the packed path when the copy dimension is large enough for the fused kernels to matter (the torch
    ``phase1_round`` path produces the identical messages either way)."""
    return AVAILABLE and a.is_cuda and b >= 8 and a.shape[0] * a.shape[1] >= (1 << 16)


__all__ = ["AVAILABLE", "phase1", "fits"]
