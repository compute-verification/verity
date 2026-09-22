"""LogUp fractional-sum GKR (``backends/gkr/src/logup.rs`` in torch, BabyBear^6).

Leaves ``(p_i, q_i)``: one ``(1, z - Σ_k β^k v_k)`` per query of every unit, one ``(-m_j, z - Σ_k β^k t_k)`` per table
row, ``(0, 1)`` padding.  The pair tree combines ``(2x, 2x+1)``: ``p = p0 q1 + p1 q0``, ``q = q0 q1``; the root must be
``(0, q_root)`` with ``q_root != 0``.  Level ``k`` (size ``2^k``) is reduced from level ``k+1`` by a degree-3 sumcheck
of ``Σ_x eq(ρ,x) [p0 q1 + p1 q0 + λ q0 q1](x)`` where ``p0(x) = p_{k+1}(2x)``, ``p1(x) = p_{k+1}(2x+1)`` (top-bit
folding on ``x``), then the four values at ``x*`` are sent and the next claim is
``p(2x*) + μ p(2x*+1) + λ' (q(2x*) + μ q(2x*+1))`` with a fresh ``μ``.  The leaf claim ``Σ_i eq(ρ,i)(p_i + λ q_i)``
is then reduced by the caller to a claim on the unit columns (``prover.py``).

``p`` leaves are base-field values (1, -m, 0) and are stored as one int32 per leaf; every other table is an
extension element (int32 storage, int64 compute).

**Binding order (a-gpu2).**  Since the swap to the packed kernels (``../packed/logup_gpu.py``) every level's sumcheck
binds the *low* bit of the level index first (round ``i`` pairs ``(2y, 2y+1)``; ``logup_reference.py``), and the
claim point is ``rho = [mu] + x*`` (``mu`` on bit 0 = the children pairing).  ``logup.rs`` / the v2e prover bound the
top bit first with ``rho = x* + [mu]``; the two are the same protocol under bit-reversal of the level index, but
the message *values* differ, so the verifier here follows the new schedule (``verify_table``) and the leaf claim
handed to ``prover.add_lookup_claim`` is converted to the top-bit-first convention (``LeafClaim.rho`` reversed)
at the boundary.  The torch code below is the CPU reference of the same schedule (``VERITY_GPU_TORCH_ONLY=1``).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import field as _field
from .circuit import Query, Table, eval_lin_rows
from .field import (
    DEG,
    Ext,
    P,
    e_add,
    e_mul,
    e_sub,
    eq_point,
    eq_table,
    ext_mul,
    ext_scale,
    ext_sum,
    fold,
    fold_base,
    interpolate,
    py_ext,
    py_exts,
    t_ext,
    t_exts,
)
from .gkr import Reader, VerifyError
from .transcript import Transcript

_fast_logup_round = _field._fast_logup_round

CHUNK = 1 << 20


@dataclass
class LeafClaim:
    rho: list[Ext]
    lam: Ext
    value: Ext


def _lerp(x0: torch.Tensor, x1: torch.Tensor, t: int) -> torch.Tensor:
    if t == 0:
        return x0
    if t == 1:
        return x1
    return (x0 + (x1 - x0) * t) % P


def _mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Product where either operand may be base (ndim one less than the other)."""
    if x.ndim < y.ndim:
        return ext_scale(y, x)
    if y.ndim < x.ndim:
        return ext_scale(x, y)
    return ext_mul(x, y)


def _split(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Even / odd entries (``p(2x)``, ``p(2x+1)``)."""
    if v.ndim == 1:
        return v[0::2], v[1::2]
    return v[0::2], v[1::2]


# ---------------------------------------------------------------------------------------------------------------------
# Query values, multiplicities, leaves
# ---------------------------------------------------------------------------------------------------------------------

def query_values(q: Query, cols: torch.Tensor) -> torch.Tensor:
    """``(units, ncols_q)`` int64 of the query's column values."""
    return torch.stack([eval_lin_rows(l, cols) for l in q.cols], dim=1)


def table_rows(t: Table, device) -> torch.Tensor:
    return torch.tensor(t.rows, dtype=torch.int64, device=device).reshape(len(t.rows), t.cols)


def multiplicities(t: Table, trows: torch.Tensor, values: list[torch.Tensor]) -> torch.Tensor:
    """``m_j`` = how many query tuples equal table row ``j`` (int64 ``(rows,)``); raises if a tuple is not a row."""
    dev = trows.device
    m = torch.zeros((trows.shape[0],), dtype=torch.int64, device=dev)
    if t.name == "R16":
        for v in values:
            key = v[:, 0]
            if bool((key >= trows.shape[0]).any()):
                raise ValueError("R16 query out of range")
            m += torch.bincount(key, minlength=trows.shape[0])
        return m
    keys = trows[:, 0]
    order = torch.argsort(keys)
    skeys = keys[order]
    for v in values:
        pos = torch.searchsorted(skeys, v[:, 0]).clamp(max=skeys.shape[0] - 1)
        idx = order[pos]
        ok = (trows[idx] == v).all(dim=1)
        if not bool(ok.all()):
            raise ValueError(f"query tuple not in table {t.name}")
        m += torch.bincount(idx, minlength=trows.shape[0])
    return m


def combine(values: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    """``Σ_k β^k v_k``: ``values`` ``(n, w)`` base, ``betas`` ``(w, 6)`` -> ``(n, 6)``."""
    acc = torch.zeros((values.shape[0], DEG), dtype=torch.int64, device=values.device)
    for k in range(values.shape[1]):
        acc = (acc + ext_scale(betas[k].unsqueeze(0), values[:, k])) % P
    return acc


def beta_powers(beta: Ext, w: int, device) -> torch.Tensor:
    pw, b = [], (1,) + (0,) * (DEG - 1)
    for _ in range(w):
        pw.append(b)
        b = e_mul(b, beta)
    return t_exts(pw, device)


def build_leaves(t: Table, trows: torch.Tensor, values: list[torch.Tensor], m: torch.Tensor, z: Ext, beta: Ext, n: int):
    """``(p_leaf (2^n,) int32 base, q_leaf (2^n, 6) int32)`` in the order [seg0 queries, seg1 queries, table, padding]."""
    dev = trows.device
    size = 1 << n
    p = torch.zeros((size,), dtype=torch.int32, device=dev)
    q = torch.zeros((size, DEG), dtype=torch.int32, device=dev)
    q[:, 0] = 1
    betas = beta_powers(beta, t.cols, dev)
    zt = t_ext(z, dev)
    pos = 0
    for v in values:
        k = v.shape[0]
        p[pos:pos + k] = 1
        for lo in range(0, k, CHUNK):
            hi = min(k, lo + CHUNK)
            q[pos + lo:pos + hi] = ((zt - combine(v[lo:hi], betas)) % P).to(torch.int32)
        pos += k
    k = trows.shape[0]
    p[pos:pos + k] = ((-m) % P).to(torch.int32)
    q[pos:pos + k] = ((zt - combine(trows, betas)) % P).to(torch.int32)
    return p, q


# ---------------------------------------------------------------------------------------------------------------------
# Prover
# ---------------------------------------------------------------------------------------------------------------------

def _combine_level(p: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _field._fast_combine_level is not None and q.is_cuda:
        return _field._fast_combine_level(p, q)
    p0, p1 = _split(p)
    q0, q1 = _split(q)
    n = q0.shape[0]
    np_ = torch.empty((n, DEG), dtype=torch.int32, device=q.device)
    nq = torch.empty((n, DEG), dtype=torch.int32, device=q.device)
    for lo in range(0, n, CHUNK):
        hi = min(n, lo + CHUNK)
        a0, a1 = p0[lo:hi].to(torch.int64), p1[lo:hi].to(torch.int64)
        b0, b1 = q0[lo:hi].to(torch.int64), q1[lo:hi].to(torch.int64)
        np_[lo:hi] = ((_mul(a0, b1) + _mul(a1, b0)) % P).to(torch.int32)
        nq[lo:hi] = ext_mul(b0, b1).to(torch.int32)
    return np_, nq


def round_poly(p0, p1, q0, q1, e, half: int, lam_t: torch.Tensor) -> list[Ext]:
    dev = e.device
    if _fast_logup_round is not None and e.is_cuda:
        return py_exts(_fast_logup_round(p0, p1, q0, q1, e, half, lam_t))
    h = torch.zeros((4, DEG), dtype=torch.int64, device=dev)
    chunk = CHUNK // 4
    for lo in range(0, half, chunk):
        hi = min(half, lo + chunk)
        a0, a1 = p0[lo:hi].to(torch.int64), p0[half + lo:half + hi].to(torch.int64)
        c0, c1 = p1[lo:hi].to(torch.int64), p1[half + lo:half + hi].to(torch.int64)
        b0, b1 = q0[lo:hi].to(torch.int64), q0[half + lo:half + hi].to(torch.int64)
        d0, d1 = q1[lo:hi].to(torch.int64), q1[half + lo:half + hi].to(torch.int64)
        e0, e1 = e[lo:hi].to(torch.int64), e[half + lo:half + hi].to(torch.int64)
        for t in range(4):
            pa, pb = _lerp(a0, a1, t), _lerp(c0, c1, t)
            qa, qb = _lerp(b0, b1, t), _lerp(d0, d1, t)
            qq = ext_mul(qa, qb)
            g = (_mul(pa, qb) + _mul(pb, qa) + ext_mul(qq, lam_t)) % P
            h[t] = (h[t] + ext_sum(ext_mul(_lerp(e0, e1, t), g))) % P
    return py_exts(h)


def _fold_low(v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Bind the *low* bit: ``out[y] = v[2y] + (v[2y+1] - v[2y]) r`` (base ``v`` -> extension)."""
    v0, v1 = v[0::2], v[1::2]
    if v.ndim == 1:
        d = (v1.to(torch.int64) - v0.to(torch.int64)) % P
        o = (d.unsqueeze(-1) * r) % P
        o[..., 0] = (o[..., 0] + v0.to(torch.int64)) % P
        return o
    a, b = v0.to(torch.int64), v1.to(torch.int64)
    return (a + ext_mul((b - a) % P, r)) % P


def round_poly_low(p0, p1, q0, q1, e, lam_t: torch.Tensor) -> list[Ext]:
    """``round_poly`` with the pairs ``(2y, 2y+1)`` (low-bit-first binding); torch reference."""
    dev = e.device
    half = e.shape[0] // 2
    h = torch.zeros((4, DEG), dtype=torch.int64, device=dev)
    chunk = CHUNK // 4
    for lo in range(0, half, chunk):
        hi = min(half, lo + chunk)
        a0, a1 = p0[2 * lo:2 * hi:2].to(torch.int64), p0[2 * lo + 1:2 * hi:2].to(torch.int64)
        c0, c1 = p1[2 * lo:2 * hi:2].to(torch.int64), p1[2 * lo + 1:2 * hi:2].to(torch.int64)
        b0, b1 = q0[2 * lo:2 * hi:2].to(torch.int64), q0[2 * lo + 1:2 * hi:2].to(torch.int64)
        d0, d1 = q1[2 * lo:2 * hi:2].to(torch.int64), q1[2 * lo + 1:2 * hi:2].to(torch.int64)
        e0, e1 = e[2 * lo:2 * hi:2].to(torch.int64), e[2 * lo + 1:2 * hi:2].to(torch.int64)
        for t in range(4):
            pa, pb = _lerp(a0, a1, t), _lerp(c0, c1, t)
            qa, qb = _lerp(b0, b1, t), _lerp(d0, d1, t)
            qq = ext_mul(qa, qb)
            g = (_mul(pa, qb) + _mul(pb, qa) + ext_mul(qq, lam_t)) % P
            h[t] = (h[t] + ext_sum(ext_mul(_lerp(e0, e1, t), g))) % P
    return py_exts(h)


def eq_table_low(rho: list[Ext], device) -> torch.Tensor:
    """``eq(rho, y)`` with ``rho[j]`` on bit ``j`` of ``y`` (the packed kernels' convention)."""
    return eq_table(list(reversed(rho)), device)


def leaf_claim_top(rho_low: list[Ext], lam: Ext, value: Ext) -> LeafClaim:
    """The low-bit-first claim point as a top-bit-first ``LeafClaim`` (what ``prover.add_lookup_claim`` indexes)."""
    return LeafClaim(list(reversed(rho_low)), lam, value)


def prove_table(p_leaf: torch.Tensor, q_leaf: torch.Tensor, n: int, tr: Transcript, out: list[Ext]) -> LeafClaim:
    """Torch reference of the low-bit-first schedule (the GPU path is ``logup_packed.prove_table``); returns the
    leaf claim in the top-bit-first convention."""
    dev = q_leaf.device
    levels_p = [p_leaf]
    levels_q = [q_leaf]
    for _k in range(n, 0, -1):
        np_, nq = _combine_level(levels_p[-1], levels_q[-1])
        levels_p.append(np_)
        levels_q.append(nq)
    # levels[i] has size 2^(n - i); the root is levels[n]
    root_p = py_ext(levels_p[n][0].to(torch.int64)) if levels_p[n].ndim == 2 else (int(levels_p[n][0]),) + (0,) * (DEG - 1)
    root_q = py_ext(levels_q[n][0].to(torch.int64))
    if any(root_p):
        raise ValueError("LogUp: fractional sum is not zero (a query is not in its table)")
    if not any(root_q):
        raise ValueError("LogUp: q_root is zero")
    out.append(root_q)
    tr.absorb_ext(root_q)
    lam = tr.challenge()
    rho: list[Ext] = []
    value = e_mul(lam, root_q)      # p_root + λ q_root with p_root = 0
    for k in range(n):
        # claim: Σ_{x in 2^k} eq(rho, x) (p_k(x) + λ q_k(x)); reduce with level k+1 (children)
        cp, cq = levels_p[n - k - 1], levels_q[n - k - 1]
        p0, p1 = _split(cp)
        q0, q1 = _split(cq)
        levels_p[n - k - 1] = None
        levels_q[n - k - 1] = None
        e = eq_table_low(rho, dev)
        lam_t = t_ext(lam, dev)
        x_star: list[Ext] = []
        length = 1 << k
        while length > 1:
            h = round_poly_low(p0, p1, q0, q1, e, lam_t)
            out.extend(h)
            tr.absorb_exts(h)
            r = tr.challenge()
            x_star.append(r)
            rt = t_ext(r, dev)
            p0, p1, q0, q1, e = (_fold_low(v, rt) for v in (p0, p1, q0, q1, e))
            length //= 2
        vals = []
        for v in (p0, p1, q0, q1):
            if v.ndim == 1:
                vals.append((int(v[0]) % P,) + (0,) * (DEG - 1))
            else:
                vals.append(py_ext(v[0].to(torch.int64)))
        out.extend(vals)
        tr.absorb_exts(vals)
        mu = tr.challenge()
        pv = e_add(vals[0], e_mul(mu, e_sub(vals[1], vals[0])))
        qv = e_add(vals[2], e_mul(mu, e_sub(vals[3], vals[2])))
        lam = tr.challenge()
        value = e_add(pv, e_mul(lam, qv))
        rho = [mu] + x_star
    return leaf_claim_top(rho, lam, value)


# ---------------------------------------------------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------------------------------------------------

def verify_table(n: int, tr: Transcript, rd: Reader) -> LeafClaim:
    root_q = rd.next()
    if not any(root_q):
        raise VerifyError("LogUp: q_root is zero")
    tr.absorb_ext(root_q)
    lam = tr.challenge()
    rho: list[Ext] = []
    value = e_mul(lam, root_q)
    for k in range(n):
        running = value
        x_star = []
        for rnd in range(k):
            h = rd.take(4)
            if e_add(h[0], h[1]) != running:
                raise VerifyError(f"LogUp level {k} round {rnd}: sum mismatch")
            tr.absorb_exts(h)
            r = tr.challenge()
            running = interpolate(h, r)
            x_star.append(r)
        vals = rd.take(4)
        p0, p1, q0, q1 = vals
        # eq(rho, x*) [p0 q1 + p1 q0 + λ q0 q1] == running
        ex = eq_point(rho, x_star)
        g = e_add(e_add(e_mul(p0, q1), e_mul(p1, q0)), e_mul(lam, e_mul(q0, q1)))
        if e_mul(ex, g) != running:
            raise VerifyError(f"LogUp level {k}: final check")
        tr.absorb_exts(vals)
        mu = tr.challenge()
        pv = e_add(p0, e_mul(mu, e_sub(p1, p0)))
        qv = e_add(q0, e_mul(mu, e_sub(q1, q0)))
        lam = tr.challenge()
        value = e_add(pv, e_mul(lam, qv))
        rho = [mu] + x_star          # low-bit-first: mu binds the children bit (bit 0 of the next level's index)
    return leaf_claim_top(rho, lam, value)
