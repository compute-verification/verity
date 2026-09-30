"""Pricing F1′ and F2, the base-split and flatness debit terms of `theory-pearl-c4-domain.md` §6.2–6.3 (13:35Z), on
Pearl-C4 v1: honest completeness at the cap on real models, and the attack families.

F1′ (it replaces item 1), per row of A′ and of B̃, from the clean row, α, ρ, a_E and this salt's line E_i:

    p_t  = 1 − max_c P(code_t = c) under the noise law given E_i, at the block's modal scale byte;
           0 on witness-dead codes and on D-SS's screened blocks
    per block: the 16 p_t sorted and paired; chunk 0 takes sorted pairs 1, 3, 5, 7 and chunk 1 pairs 2, 4, 6, 8
    per aligned 128-window: X = elements beyond 2:4, N_h = changed codes per k64 half, M_w = blocks off their modal byte
    S_w  = E[max(0, ½ − c·X)] + Σ_h E[max(0, ½ − c·N_h)],  c = 16.92/128
    debit = 128·max(0, S_w − min(c·M_w, 0.125)) FP4 units per word the row feeds

The window DPs are bc-a8466279's (`cheap-binding/pearlc4-fix/dnc2.py`). Two noise laws:

    exact  the law the reference forms: D_t = α·x_t + (a_E/64)·n_t, n_t = Σ_j e_ij·f_tj in quarter units, E_i this
           salt's line (half units e) and f_t uniform over the 16 E2M1 codes; per row the pmf of n from the product of
           the 32 terms' characteristic functions, so P(code) is exact given E_i (ties to even at the cell edges); the
           modal byte from 24 draws of the same law, the realised byte from one more
    gauss  dnc2.py's census: a Gaussian at σα with p_t at the clean block scale, bytes from Gaussian draws

F2 (§6.3, parameters pinned in `docs/pouw/fp4-tile-model.md` §7.8): f_A, f_B = the tile's share of A′ and B̃ blocks
at its modal realised byte; save = max(0, 1 − c_L − 4·(2 − f_A − f_B)), charged on the tile's chain, with c_L(m, k, n)
the int8 route's cost at the deepest depth the GEMM's shape admits.

The tile debit is F1′ (A and B) + term 3 (screened pairs) + term 4 (forming, on the witness) + F2, against v1's
creditOf at f_s = 107.34, next to the current v1 debit (term 1 on the witness) from `fp4_v4_coverage.py`.

    python benchmarks/pouw/fp4_f1_pricing.py attack --out attack.json          # the attack families
    python benchmarks/pouw/fp4_f1_pricing.py real Qwen/Qwen2.5-0.5B --out q.json  # honest completeness on a model
    python benchmarks/pouw/fp4_f1_pricing.py real Qwen/Qwen2.5-7B --offload 9GiB    # a model larger than memory
    python benchmarks/pouw/fp4_f1_pricing.py flatness unsloth/Meta-Llama-3.1-70B     # F2 on the weights alone
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fp4_emulation import E2M1_HALVES, e2m1_rn, ue4m3_rn  # noqa: E402
from fp4_v4_coverage import KB1, T, bf16, stats, uval, witness  # noqa: E402

C = 16.92 / 128
XM = 4
FS, CAP = 107.34, 1 / 400
H = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12])     # the 16 E2M1 codes in half units
LMAX = 32 * 12 * 12
G = 16384
EDGES = np.array([-5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
TIE_DOWN = np.array([False, True] * 7)            # RN-even: a tie at the edge goes to the value below it
CELL_VALUES = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6])
_W = 2 * np.pi * np.arange(G // 2 + 1) / G
_MAGS = (1, 2, 3, 4, 6, 8, 12)
_PHI = np.stack([(2 + 2 * sum(np.cos(_W * v * h) for h in _MAGS)) / 16 for v in _MAGS])


# ---- the exact law ---------------------------------------------------------------------------------------------------

def row_cdf(eh):
    """P(n ≤ m) for m in [−LMAX, LMAX], per row, n = Σ_j e_j·f_j with f_j uniform over H. eh: (R, 32) half units."""
    counts = np.stack([(np.abs(eh) == v).sum(1) for v in _MAGS], 1)
    phi = np.ones((len(eh), G // 2 + 1))
    for i in range(len(_MAGS)):
        phi *= np.power(_PHI[i][None, :], counts[:, i:i + 1])
    pmf = np.fft.irfft(phi, G, axis=1)
    pmf = np.maximum(np.concatenate([pmf[:, G - LMAX:], pmf[:, :LMAX + 1]], 1), 0)
    cdf = np.cumsum(pmf, 1)
    return cdf / cdf[:, -1:]


def p_le(cdf, m):
    """P(n ≤ m), per row of cdf, for integer m of shape (R, ...)."""
    idx = m + LMAX
    base = (np.arange(len(cdf)) * (2 * LMAX + 1)).reshape((-1,) + (1,) * (m.ndim - 1))
    out = cdf.ravel()[base + np.clip(idx, 0, 2 * LMAX)]
    return np.where(idx < 0, 0.0, out)


def below(u, delta, tie_down):
    """The largest m with n ≤ m ⇔ v + δn rounds to the value below the edge, u = edge·s − v."""
    z = u / delta
    r = np.rint(z)
    tie = np.abs(z - r) < 1e-9 * np.maximum(1.0, np.abs(z))
    m = np.where(tie, r - np.where(tie_down, 0, 1), np.ceil(z) - 1)
    return np.clip(m, -LMAX - 1, LMAX + 1).astype(np.int64)


def p_exact(v, s, delta, cdf, rows=16):
    """(p_t, the modal value): p_t = 1 − max_c P(code_t = c), v = α·x (R, k), s the block scale value per element
    (R, k), δ = a_E/64 (R,)."""
    R, k = v.shape
    p, mode = np.empty((R, k)), np.empty((R, k))
    for a in range(0, R, rows):
        sl = slice(a, a + rows)
        u = EDGES[None, None, :] * s[sl, :, None] - v[sl, :, None]
        m = below(u, delta[sl, None, None], TIE_DOWN[None, None, :])
        F = p_le(cdf[sl], m)
        cells = np.diff(np.concatenate([np.zeros(F.shape[:2] + (1,)), F, np.ones(F.shape[:2] + (1,))], 2), axis=2)
        p[sl] = 1 - cells.max(2)
        mode[sl] = CELL_VALUES[cells.argmax(2)]
    return np.clip(p, 0, 1), mode


BLOCK = {"nv": 16, "mx": 32}


def scale_byte(amax, fmt):
    """The block's scale byte from its |max|: NVFP4 max(ue4m3(amax/6), 1); MXFP4 the UE8M0 byte of 2^⌈log2(amax/6)⌉."""
    v = amax / 6
    if fmt == "nv":
        return np.maximum(ue4m3_rn(v), 1)
    m, e = np.frexp(np.maximum(v, 2.0 ** -127))
    return np.clip(np.where(m > 0.5, e, e - 1) + 127, 0, 254)


def scale_value(byte, fmt):
    return np.maximum(uval(byte), 2.0 ** -9) if fmt == "nv" else np.exp2(byte.astype(np.float64) - 127)


def stats_fmt(X, fmt):
    """v1's row statistics; on MXFP4 α is floored to a power of two and a_E formed from it (the reference's `stats`)."""
    alpha, ae, dnf, rho2 = stats(X, KB1)
    if fmt == "mx":
        alpha = np.exp2(np.floor(np.log2(alpha)))
        ae = uval(ue4m3_rn(np.sqrt(rho2) / 4 * alpha * KB1))
        dnf = ae >= 2.0 ** -6
    return alpha, ae, dnf, rho2


def witness_fmt(X, alpha, ae, fmt):
    """The reference's witness (C + N, C, C − N with N = 72·a_E, the block's other elements at their clean values) at
    the format's block and scale rule."""
    if fmt == "nv":
        return witness(X, alpha, ae, 72.0, True)
    R, k = X.shape
    B = BLOCK[fmt]
    Cb = (alpha[:, None] * X).reshape(R, k // B, B)
    A = np.abs(Cb)
    top = A.max(2, keepdims=True)
    second = np.sort(A, axis=2)[:, :, -2:-1]
    rest = np.where(A == top, second, top)
    N = (72 * ae)[:, None, None]
    vals = [e2m1_value(e2m1_rn(d / scale_value(scale_byte(np.maximum(np.abs(d), rest), fmt), fmt)))
            for d in (Cb + N, Cb, Cb - N)]
    return ((vals[0] == vals[1]) & (vals[1] == vals[2])).reshape(R, k)


def side_fmt(X, fmt):
    """`fp4_v4_coverage.side` at the format: D-NF, term 1's 2:4 units on the witness, D-SS's screened blocks, and the
    witness-dead count term 4 charges."""
    R, k = X.shape
    B = BLOCK[fmt]
    alpha, ae, dnf, rho2 = stats_fmt(X, fmt)
    dead = np.zeros(X.shape, bool)
    for a in range(0, R, 512):
        dead[a:a + 512] = witness_fmt(X[a:a + 512], alpha[a:a + 512], ae[a:a + 512], fmt)
    scr = (np.abs(X).reshape(R, -1, B).max(2) ** 2) > 90 * rho2[:, None]
    d4 = dead.reshape(R, -1, 4).sum(2)
    units = np.where(d4 == 4, 4, np.where(d4 >= 2, 2, 0)).sum(1).astype(np.float64)
    return dict(dnf=dnf, units=units, scr=scr.astype(np.float64), sdw=dead.sum(1).astype(np.float64), dead=dead,
                screened=scr, alpha=alpha, ae=ae, rho2=rho2)


def draw_bytes(v, eh, delta, rng, fmt="nv"):
    """One draw of the block scale bytes under the law given E_i (a fresh F)."""
    R, k = v.shape
    F = H[rng.integers(0, 16, (k, 32))].astype(np.float64)
    D = v + delta[:, None] * (eh.astype(np.float64) @ F.T)
    return scale_byte(np.abs(D).reshape(R, k // BLOCK[fmt], BLOCK[fmt]).max(2), fmt)


def modal_of(draws):
    B = np.stack(draws)
    best, bc = np.zeros(B.shape[1:], np.int64), np.zeros(B.shape[1:], np.int64)
    for val in range(int(B.min()), int(B.max()) + 1):
        c = (B == val).sum(0)
        upd = c > bc
        best[upd], bc[upd] = val, c[upd]
    return best


# ---- the Gaussian census (dnc2.py) -----------------------------------------------------------------------------------

def ncdf(z):
    import torch
    return torch.special.ndtr(torch.from_numpy(np.asarray(z, np.float64))).numpy()


def p_gauss(X, alpha, rho2):
    R, k = X.shape
    amax = np.abs(X).reshape(R, -1, 16).max(2)
    sb = np.repeat(np.maximum(uval(ue4m3_rn(alpha[:, None] * amax / 6)), 2.0 ** -9), 16, axis=1)
    v = alpha[:, None] * X / sb
    sig = (np.sqrt(rho2) / 4 * alpha)[:, None] / sb
    edges = np.concatenate([[-np.inf], EDGES, [np.inf]])
    p, mode = np.empty((R, k)), np.empty((R, k))
    for a in range(0, R, 64):
        cells = np.diff(ncdf((edges[None, None, :] - v[a:a + 64, :, None]) / sig[a:a + 64, :, None]), axis=-1)
        p[a:a + 64] = 1 - cells.max(-1)
        mode[a:a + 64] = CELL_VALUES[cells.argmax(-1)]
    return p, mode


def gauss_bytes(X, alpha, rho2, rng):
    R, k = X.shape
    D = (alpha[:, None] * X) + rng.standard_normal(X.shape) * (np.sqrt(rho2) / 4 * alpha)[:, None]
    return np.maximum(ue4m3_rn(np.abs(D).reshape(R, k // 16, 16).max(2) / 6), 1)


# ---- one side: p, the modal and realised bytes ----------------------------------------------------------------------

def law_side(X, rng, law="exact", draws=24, fmt="nv", sd=None):
    """p (R, k) with the rule's zeros, the modal and realised bytes (R, k/B), and the salt's E (half units). `sd`: the
    side's `side_fmt`, if already formed."""
    R, k = X.shape
    B = BLOCK[fmt]
    sd = sd or side_fmt(X, fmt)
    alpha, ae, dnf, rho2, scr, dead = sd["alpha"], sd["ae"], sd["dnf"], sd["rho2"], sd["screened"], sd["dead"]
    if law == "exact":
        eh = H[rng.integers(0, 16, (R, 32))]
        delta = ae / 64
        v = alpha[:, None] * X
        mod = modal_of([draw_bytes(v, eh, delta, rng, fmt) for _ in range(draws)])
        rea = draw_bytes(v, eh, delta, rng, fmt)
        s = np.repeat(scale_value(mod, fmt), B, axis=1)
        p, a0 = np.empty((R, k)), np.empty((R, k))
        for a in range(0, R, 256):
            p[a:a + 256], a0[a:a + 256] = p_exact(v[a:a + 256], s[a:a + 256], delta[a:a + 256], row_cdf(eh[a:a + 256]))
    else:
        assert fmt == "nv", "the Gaussian census is NVFP4's"
        mod = modal_of([gauss_bytes(X, alpha, rho2, rng) for _ in range(draws)])
        rea = gauss_bytes(X, alpha, rho2, rng)
        p, a0 = p_gauss(X, alpha, rho2)
    p = np.where(dead | np.repeat(scr, B, axis=1), 0.0, p)
    return dict(p=p, a0=a0, mod=mod, rea=rea, off=(mod != rea), scr=scr, dead=dead, dnf=dnf,
                eh=eh if law == "exact" else None)


# ---- F1′'s window DPs (dnc2.py) -------------------------------------------------------------------------------------

COMBOS = list(itertools.product(range(3), repeat=4))
EXC = []
for _js in COMBOS:
    _dev = sorted(j for j in _js if j > 0)
    EXC.append(min(sum(_dev[:len(_dev) - 2]) if len(_dev) > 2 else 0, XM))
VX = np.array([max(0.0, 0.5 - C * x) for x in range(XM + 1)])


def conv_trunc(a, b, m=XM):
    out = np.zeros_like(a)
    for i in range(m + 1):
        for j in range(m + 1 - i):
            out[..., i + j] += a[..., i] * b[..., j]
    out[..., m] += a.sum(-1) * b.sum(-1) - out.sum(-1)
    return out


def block_excess(p):
    """p: (N, B). The block's excess distribution (N, XM + 1) at the sorted layout: the B codes sorted and paired, the
    B/2 pairs dealt round-robin to the block's B/8 chunks (B = 16: chunk 0 takes sorted pairs 1, 3, 5, 7)."""
    s = np.sort(p, axis=1)
    pa, pb = s[:, 0::2], s[:, 1::2]
    pj = np.stack([(1 - pa) * (1 - pb), pa * (1 - pb) + (1 - pa) * pb, pa * pb], -1)     # (N, B/2, 3)
    nch = p.shape[1] // 8
    d = None
    for c0 in range(nch):
        ch = pj[:, c0::nch]
        dc = np.zeros((p.shape[0], XM + 1))
        for js, x in zip(COMBOS, EXC):
            dc[:, x] += ch[:, 0, js[0]] * ch[:, 1, js[1]] * ch[:, 2, js[2]] * ch[:, 3, js[3]]
        d = dc if d is None else conv_trunc(d, dc)
    return d


def changed_dist(p, m=XM):
    d = np.zeros((p.shape[0], m + 1))
    d[:, 0] = 1
    for t in range(p.shape[1]):
        q = p[:, t:t + 1]
        nd = d * (1 - q)
        nd[:, 1:] += d[:, :-1] * q
        nd[:, m] += d[:, m] * q[:, 0]
        d = nd
    return d


def window_dists(p, B=16):
    """p: (W, 128). The distributions of X (the 2:4 excess) and of N_h in each k64 half, each (W, XM + 1)."""
    be = block_excess(p.reshape(-1, B)).reshape(len(p), 128 // B, XM + 1)
    d = be[:, 0]
    for j in range(1, 128 // B):
        d = conv_trunc(d, be[:, j])
    return d, changed_dist(p[:, :64]), changed_dist(p[:, 64:128])


def window_saving(p):
    """p: (W, 128). S_w per window."""
    dx, d0, d1 = window_dists(p)
    return (dx * VX).sum(1) + (d0 * VX).sum(1) + (d1 * VX).sum(1)


def f1_windows(p, off, rows=64, B=16):
    """Per row window, as a share of the window's MACs, (R, k/128) each: F1′ as stated (13:50Z), max(0, min(S_w, 1) − κ)
    with κ = min(c·M_w, 0.125); the form with κ inside the expectation, min(1, E[max(0, R_24 − κ)] +
    E[max(0, R_skip − κ)]), which bounds a per-row route choice E[max(0, max(R_24, R_skip) − κ)] on every law; and S_w."""
    R, k = p.shape
    W = k // 128
    nb = 128 // B
    kap = np.minimum(C * off[:, :W * nb].reshape(-1, nb).sum(1), 0.125)
    dx, d0, d1 = (np.concatenate(z) for z in zip(*(window_dists(p[a:a + rows, :W * 128].reshape(-1, 128), B)
                                                   for a in range(0, R, rows))))
    s = (dx * VX).sum(1) + (d0 * VX).sum(1) + (d1 * VX).sum(1)
    in24 = (dx * np.maximum(0, VX[None, :] - kap[:, None])).sum(1)
    pair = VX[:, None] + VX[None, :]
    inskip = np.einsum("wa,wb,wab->w", d0, d1, np.maximum(0, pair[None] - kap[:, None, None]))
    return (np.maximum(0, np.minimum(s, 1.0) - kap).reshape(R, W), np.minimum(in24 + inskip, 1.0).reshape(R, W),
            s.reshape(R, W))


def c_L(m, k, n, B=16):
    """The int8 route's cost at the deepest depth the shape admits (`theory-pearl-c4-domain.md` §6.3, 13:50Z):
    min over 1 ≤ L ≤ log2(k/B) of 2·(7/8)^L + 4.25·(1/m + 1/n)·((7/4)^L − 1)/0.75, B the format's block."""
    return min(2 * 0.875 ** L + 4.25 * (1 / m + 1 / n) * (1.75 ** L - 1) / 0.75
               for L in range(1, max(1, int(np.log2(k // B))) + 1))


def f2_save(bA, bB, cL, patch=4.0):
    """F2's saving per chain MAC from the tile's realised bytes on each side (the patch at 4 FP4 slots per side)."""
    fa = np.bincount(bA.ravel()).max() / bA.size
    fb = np.bincount(bB.ravel()).max() / bB.size
    return max(0.0, 1 - cL - patch * (2 - fa - fb)), fa, fb


# ---- the realised saving, for checking F1′'s expectation ------------------------------------------------------------

def realised(X, sd, rng, law, draws=8, fixed_e=True):
    """The route catalogue's saving on real draws, at F1′'s layout and with its A0 (each element's modal value at the
    block's modal byte): an element is changed when its realised value (at the realised byte) differs from A0's, and the
    window saves max(0, S − min(c·M_w, 0.125)) with S, M_w realised. Averaged over draws of F given this salt's E_i
    (`fixed_e`, the law F1′ conditions on) or of fresh salts; screened blocks and witness-dead codes are never changed,
    as in the rule."""
    R, k = X.shape
    alpha, ae, dnf, rho2 = stats(X, KB1)
    v = alpha[:, None] * X
    order = np.argsort(sd["p"].reshape(R, -1, 16), axis=2, kind="stable")
    keep = np.repeat(sd["scr"], 16, axis=1) | sd["dead"]
    Wn = k // 128
    out = []
    for _ in range(draws):
        if law == "exact":
            eh = sd["eh"] if fixed_e else H[rng.integers(0, 16, (R, 32))]
            F = H[rng.integers(0, 16, (k, 32))].astype(np.float64)
            D = v + (ae / 64)[:, None] * (eh.astype(np.float64) @ F.T)
        else:
            D = v + rng.standard_normal(X.shape) * (np.sqrt(rho2) / 4 * alpha)[:, None]
        byt = np.maximum(ue4m3_rn(np.abs(D).reshape(R, -1, 16).max(2) / 6), 1)
        s_real = np.repeat(np.maximum(uval(byt), 2.0 ** -9), 16, axis=1)
        ch = (e2m1_value(e2m1_rn(D / s_real)) != sd["a0"]) & ~keep
        pairs = np.take_along_axis(ch.reshape(R, -1, 16), order, axis=2).reshape(R, -1, 8, 2).sum(3)   # 0..2 per pair
        exc = np.zeros(pairs.shape[:2], np.int64)
        for c0 in (0, 1):
            pr = pairs[:, :, c0::2]
            srt = np.sort(np.where(pr > 0, pr, 99), axis=2)
            n_ch = (pr > 0).sum(2)
            for j in range(2):
                exc += np.where(n_ch - 2 > j, srt[:, :, j], 0)
        X_w = exc[:, :Wn * 8].reshape(R, Wn, 8).sum(2)
        N_h = ch[:, :Wn * 128].reshape(R, Wn, 2, 64).sum(3)
        r24 = np.maximum(0, 0.5 - C * X_w)
        rskip = np.maximum(0, 0.5 - C * N_h).sum(2)
        kap = np.minimum(C * (byt != sd["mod"])[:, :Wn * 8].reshape(R, Wn, 8).sum(2), 0.125)
        # a 16-row fragment takes one route per window: the sparse pass with every row's excess patched, or each k64
        # atom skipped with every row's changes patched; unprofitable rows cost what they cost
        fr = R // 16
        g = lambda z: z[:fr * 16].reshape(fr, 16, *z.shape[1:]).sum(1)  # noqa: E731
        f24 = g(0.5 - C * X_w)
        fskip = g(0.5 - C * N_h.astype(np.float64)).clip(min=0).sum(-1)
        frag = np.maximum(0, np.maximum(f24, fskip) - g(kap)) / 16
        out.append((np.maximum(0, r24 + rskip - kap).mean(), np.maximum(0, np.maximum(r24, rskip) - kap).mean(),
                    frag.mean()))
    o = np.array(out).mean(0)
    return {"rows_sum_of_routes": float(o[0]), "rows_best_route": float(o[1]), "fragments_best_route": float(o[2])}


def e2m1_value(code):
    mag = E2M1_HALVES[code & 7] / 2
    return np.where(code & 8, -mag, mag)


# ---- the attack families --------------------------------------------------------------------------------------------

def spike_rows(R, k, S, rng):
    """The assessor's rows: Gaussian background, 14 spikes per block at S (rank-one signs) off positions 0 and 8."""
    x = rng.standard_normal((R, k))
    rs, cs = rng.choice((-1.0, 1.0), R), rng.choice((-1.0, 1.0), k // 16)
    for o in range(16):
        if o not in (0, 8):
            x[:, o::16] = S * rs[:, None] * cs[None, :]
    return bf16(x)


def parked_rows(R, k, S, rng, park=4.25):
    """One spike per block at S (the block max) and the other 13 at park/6 of it, the centre of code 4's cell."""
    x = spike_rows(R, k, S, rng)
    first = [o for o in range(16) if o not in (0, 8)][1:]
    for o in first:
        x[:, o::16] *= park / 6
    return bf16(x)


def narrow_cell(R, k, rng, S=8.9):
    u = S / 6
    x = rng.choice((-1.0, 1.0), (R, 1)) * rng.choice((-1.0, 1.0), (1, k)) * rng.choice((0.0, 0.5, 1.0, 1.5), (1, k)) * u
    samp = np.where(rng.random((1, k // 8)) < 0.275, 1.0, 0.5) * u
    x[:, ::8] = rng.choice((-1.0, 1.0), (R, k // 8)) * samp
    x[:, 1::16] = S * rng.choice((-1.0, 1.0), (R, 1))
    return bf16(x)


def max_offset(R, k, rng, S=6.0):
    x = rng.standard_normal((R, k))
    x[:, 1::16] = S * rng.choice((-1.0, 1.0), (R, 1)) * rng.choice((-1.0, 1.0), (1, k // 16))
    return bf16(x)


def stride_rows(R, k, rng, S=9.48):
    """GPU 7's stride rows: ρ at ±1 on every 8th position, the others at the centres of the top two codes (6 and 4.25
    in units of amax/6), a salt-free column pattern with rank-one signs; ρ/amax = 1/S."""
    x = rng.choice((-1.0, 1.0), (R, 1)) * rng.choice((-1.0, 1.0), (1, k)) * np.where(rng.random((1, k)) < 0.5, 1.0, 4.25 / 6) * S
    x[:, 1::16] = S * np.sign(x[:, 1::16])
    x[:, ::8] = rng.choice((-1.0, 1.0), (R, k // 8))
    return bf16(x)


FAMILIES = {
    "spike S=8.9": lambda R, k, g: spike_rows(R, k, 8.9, g),
    "spike S=8": lambda R, k, g: spike_rows(R, k, 8.0, g),
    "spike S=7": lambda R, k, g: spike_rows(R, k, 7.0, g),
    "spike S=6": lambda R, k, g: spike_rows(R, k, 6.0, g),
    "spike S=5": lambda R, k, g: spike_rows(R, k, 5.0, g),
    "spike S=4": lambda R, k, g: spike_rows(R, k, 4.0, g),
    "parked at code 4, S=8.9": lambda R, k, g: parked_rows(R, k, 8.9, g),
    "parked at code 4, S=6": lambda R, k, g: parked_rows(R, k, 6.0, g),
    "narrow-cell": narrow_cell,
    "max at a fixed offset, S=6": max_offset,
    "max at a fixed offset, S=8": lambda R, k, g: max_offset(R, k, g, 8.0),
    "max at a fixed offset, S=9.4": lambda R, k, g: max_offset(R, k, g, 9.4),
    "stride rows, rho/amax 1/9.48": lambda R, k, g: stride_rows(R, k, g, 9.48),
    "stride rows, rho/amax 1/8.98": lambda R, k, g: stride_rows(R, k, g, 8.98),
    "stride rows, rho/amax 1/9.99 (screened)": lambda R, k, g: stride_rows(R, k, g, 9.99),
}


def attack(args):
    out = []
    for name, make in FAMILIES.items():
        rng = np.random.default_rng(11)
        X = make(args.rows, args.k, rng)
        rec = {"family": name}
        for law in args.laws:
            sd = law_side(X, rng, law)
            d, d_in, s = f1_windows(sd["p"], sd["off"])
            rec[law] = {"screened": float(sd["scr"].mean()), "modal_byte_share": float(1 - sd["off"].mean()),
                        "S_w": float(s.mean()), "F1": float(d.mean()), "F1_kappa_inside": float(d_in.mean()),
                        "F1_min_row": float(d.mean(1).min()),
                        "realised": realised(X, sd, rng, law, args.draws) if args.draws else None}
            f_tile = np.bincount(sd["rea"].ravel()).max() / sd["rea"].size
            rec[law]["tile_modal_share"] = float(f_tile)
            rec[law]["F2_both_sides_this_flat"] = {str(sh): f2_save(sd["rea"], sd["rea"], c_L(sh, sh, sh))[0]
                                                   for sh in (8192, 16384, 32768)}
        out.append(rec)
        print(json.dumps(rec), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


# ---- real models ----------------------------------------------------------------------------------------------------

VARIANTS = ("base", "f1", "f1_in")     # v1 now (term 1 on the witness); F1′ as stated; F1′ with κ inside


def linear(X, W, rng, law, max_cols, m_shape, fmt="nv"):
    """X: (m, k) activations as a float array; W: (n, k) weights as a torch tensor (only the sampled tiles are read)."""
    B = BLOCK[fmt]
    X = bf16(X)
    m, k = X.shape
    n = W.shape[0]
    m -= m % T
    k -= k % B
    cL = c_L(m_shape or m, k, n, B)
    cols = sorted(rng.choice(n // T, min(n // T, max_cols), replace=False))
    Wj = bf16(np.concatenate([W[c * T:(c + 1) * T].float().numpy() for c in cols]))[:, :k]
    Xa = X[:m, :k]
    a, b = side_fmt(Xa, fmt), side_fmt(Wj, fmt)
    la, lb = law_side(Xa, rng, law, fmt=fmt, sd=a), law_side(Wj, rng, law, fmt=fmt, sd=b)
    da, dia, sa = f1_windows(la["p"], la["off"], B=B)
    db, dib, sb = f1_windows(lb["p"], lb["off"], B=B)
    ua, ub = 128 * da.sum(1), 128 * db.sum(1)              # FP4 units per word the row feeds
    uia, uib = 128 * dia.sum(1), 128 * dib.sum(1)
    fa_row, fb_row = da.mean(1), db.mean(1)                # share of the row's chain
    cw = k + (32 + FS) * k / n + 256
    ok_all = a["dnf"]
    rows = []
    for r in range(m // T):
        I = slice(r * T, (r + 1) * T)
        ok = ok_all[I]
        if ok.sum() == 0:
            continue
        nI = ok.sum()
        for c in range(len(cols)):
            J = slice(c * T, (c + 1) * T)
            cred = nI * T * cw
            t1 = (T * a["units"][I][ok].sum() + nI * b["units"][J].sum()) / cred
            f1 = (T * ua[I][ok].sum() + nI * ub[J].sum()) / cred
            f1i = (T * uia[I][ok].sum() + nI * uib[J].sum()) / cred
            t3 = B * (a["scr"][I][ok] @ b["scr"][J].T).sum() / cred
            t4 = T * FS / n * a["sdw"][I][ok].sum() / cred
            sv, fA, fB = f2_save(la["rea"][I][ok], lb["rea"][J], cL)
            sv0 = f2_save(la["rea"][I][ok], lb["rea"][J], 0.0)[0]
            f2 = sv * nI * T * k / cred
            f1_chain = fa_row[I].mean() + fb_row[J].mean()          # dnc2.py's tile figure (share of chain, all rows)
            rows.append(dict(cred=cred, base=t1 + t3 + t4, f1=f1 + t3 + t4 + f2, f1_in=f1i + t3 + t4 + f2,
                             t_f1=f1, t_f1_in=f1i, t3=t3, t4=t4, f2=f2, f2_cl0=sv0, fAB=fA + fB, f1_chain=f1_chain))
    o = {key: np.array([t[key] for t in rows]) for key in rows[0]}
    cred = o["cred"]
    tot = cred.sum()
    out = dict(k=k, n=n, tiles=len(cred), macs=float(m * n * k), dnf_fail=float(1 - ok_all.mean()), c_L=cL,
               f2_max=float(o["f2"].max()), f2_cl0_max=float(o["f2_cl0"].max()), fAB_max=float(o["fAB"].max()),
               fAB_median=float(np.median(o["fAB"])), tiles_f1_chain_over=int((o["f1_chain"] > CAP).sum()),
               f1_row_a=dict(mean=float(fa_row.mean()), p99=float(np.quantile(fa_row, 0.99)), max=float(fa_row.max())),
               f1_row_w=dict(mean=float(fb_row.mean()), max=float(fb_row.max())),
               f1_in_row_a_max=float(dia.mean(1).max()),
               S_w_mean=dict(a=float(sa.mean()), w=float(sb.mean())),
               modal_byte_share=dict(a=float(1 - la["off"].mean()), w=float(1 - lb["off"].mean())),
               worst_row=int(np.argmax(fa_row)))
    for v in VARIANTS:
        d = o[v]
        iw = int(np.argmax(d))
        out[v] = dict(over=float(cred[d > CAP].sum() / tot), over_half=float(cred[d > CAP / 2].sum() / tot),
                      max=float(d.max()), mean=float((cred * d).sum() / tot), tiles_over=int((d > CAP).sum()),
                      worst_terms={t: float(o[t][iw]) for t in ("t_f1", "t_f1_in", "t3", "t4", "f2")})
    return out


def capture(args, keep):
    """Inputs of the kept linears on `args.tokens` tokens, and a reader of their weights (bf16 torch tensors). With
    `--offload`, the forward streams the weights from disk (accelerate) and the reader opens the safetensors shards."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    kw = dict(dtype=torch.bfloat16)
    if args.offload:
        kw.update(device_map="auto", max_memory={"cpu": args.offload}, offload_folder=args.offload_folder)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()
    ids = tok(Path(args.text).read_text(), return_tensors="pt").input_ids[:, :args.tokens]
    lins = [nm for nm, mod in model.named_modules() if isinstance(mod, torch.nn.Linear) and "lm_head" not in nm]
    nlayers = model.config.num_hidden_layers
    lins = [nm for nm in lins if keep(nm, nlayers)]
    mods = dict(model.named_modules())
    got = {}
    hs = [mods[nm].register_forward_pre_hook((lambda nm: lambda m_, a: got.__setitem__(nm, a[0][0].detach().to(torch.bfloat16).cpu().clone()))(nm))
          for nm in lins]
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()
    if not args.offload:
        return lins, got, (lambda nm: mods[nm].weight.detach()), int(ids.shape[1])
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    root = Path(snapshot_download(args.model, allow_patterns=["*.json"]))
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    del model

    def weight(nm):
        path = Path(snapshot_download(args.model, allow_patterns=[index[nm + ".weight"]])) / index[nm + ".weight"]
        with safe_open(str(path), "pt") as f:
            return f.get_tensor(nm + ".weight")
    return lins, got, weight, int(ids.shape[1])


def layer_of(nm):
    return next((int(p) for p in nm.split(".") if p.isdigit()), 0)


def real(args):
    keep = lambda nm, L: layer_of(nm) % args.layer_step == 0 or layer_of(nm) == L - 1  # noqa: E731
    lins, got, weight, tokens = capture(args, keep)
    rng = np.random.default_rng(20260930)
    allres = {f: {} for f in args.fmts}
    t0 = time.time()
    for i, nm in enumerate(lins):
        x, w = got.pop(nm).float().numpy(), weight(nm)
        for fmt in args.fmts:
            r = allres[fmt][nm] = linear(x, w, rng, args.law, args.max_cols, args.m_shape, fmt)
            if r["f1"]["over"] > 0 or r["base"]["over"] > 0 or r["f1_in"]["over"] > 0 or i % 20 == 0:
                print(f"{fmt} {nm:40s} k={r['k']:5d} n={r['n']:5d} over cap: base {100*r['base']['over']:.1f}% F1′ {100*r['f1']['over']:.1f}% "
                      f"κ-in {100*r['f1_in']['over']:.1f}%  worst {100*r['f1']['max']:.3f}% "
                      f"({ {t: round(100 * v, 3) for t, v in r['f1']['worst_terms'].items()} })  F1′ row max {100*r['f1_row_a']['max']:.2f}% "
                      f"(token {r['worst_row']}) f_A+f_B max {r['fAB_max']:.3f}  [{i + 1}/{len(lins)}, {time.time()-t0:.0f}s]", flush=True)
    out = {fmt: summarize(args, res, tokens, t0, fmt) for fmt, res in allres.items()}
    print(json.dumps(out, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": out, "linears": allres}))


def summarize(args, res, tokens, t0, fmt):
    macs = sum(r["macs"] for r in res.values())
    wavg = lambda f: sum(f(r) * r["macs"] for r in res.values()) / macs  # noqa: E731
    summ = {"model": args.model, "format": fmt, "tokens": tokens, "law": args.law, "max_cols": args.max_cols,
            "layer_step": args.layer_step, "linears": len(res), "tiles": sum(r["tiles"] for r in res.values()),
            "seconds": round(time.time() - t0)}
    for v in VARIANTS:
        worst = max(res, key=lambda nm: res[nm][v]["max"])
        summ[v] = {"credit_over_cap": wavg(lambda r: r[v]["over"]), "credit_over_half_cap": wavg(lambda r: r[v]["over_half"]),
                   "macs_in_linears_without_2x_margin": sum(r["macs"] for r in res.values() if r[v]["max"] > CAP / 2) / macs,
                   "mean_tile_debit": wavg(lambda r: r[v]["mean"]),
                   "tiles_over_cap": sum(r[v]["tiles_over"] for r in res.values()),
                   "worst_tile": {"linear": worst, "debit": res[worst][v]["max"], "terms": res[worst][v]["worst_terms"]}}
    summ["f1_chain_tiles_over_cap"] = sum(r["tiles_f1_chain_over"] for r in res.values())
    summ["f1_row_a"] = {"mean": wavg(lambda r: r["f1_row_a"]["mean"]), "max": max(r["f1_row_a"]["max"] for r in res.values())}
    summ["f1_row_w"] = {"mean": wavg(lambda r: r["f1_row_w"]["mean"]), "max": max(r["f1_row_w"]["max"] for r in res.values())}
    summ["f2"] = {"max": max(r["f2_max"] for r in res.values()), "max_at_cL_0": max(r["f2_cl0_max"] for r in res.values()),
                  "fA_plus_fB_max": max(r["fAB_max"] for r in res.values()),
                  "fA_plus_fB_median": float(np.median([r["fAB_median"] for r in res.values()])),
                  "c_L_min": min(r["c_L"] for r in res.values())}
    summ["modal_byte_share"] = {"a": float(np.mean([r["modal_byte_share"]["a"] for r in res.values()])),
                                "w": float(np.mean([r["modal_byte_share"]["w"] for r in res.values()]))}
    summ["dnf_fail"] = wavg(lambda r: r["dnf_fail"])
    return summ


def flatness(args):
    """F2 on weights alone: every sampled 64-row tile's share of blocks at its modal realised byte, per linear, read
    from the safetensors shards one tensor at a time (no forward). F2 > 0 needs f_A + f_B > 2 − (1 − c_L)/4 ≥ 1.75,
    so f_B ≤ 0.75 on every weight tile makes F2 zero whatever the activations."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    root = Path(snapshot_download(args.model, allow_patterns=["*.json"]))
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    names = [key[:-len(".weight")] for key in index if key.endswith(".weight") and any(
        t in key for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))]
    nl = 1 + max(layer_of(nm) for nm in names)
    names = [nm for nm in names if layer_of(nm) % args.layer_step == 0 or layer_of(nm) == nl - 1]
    rng = np.random.default_rng(20260930)
    out, t0 = {}, time.time()
    for nm in sorted(names, key=lambda s: (layer_of(s), s)):
        path = Path(snapshot_download(args.model, allow_patterns=[index[nm + ".weight"]])) / index[nm + ".weight"]
        with safe_open(str(path), "pt") as f:
            W = f.get_tensor(nm + ".weight")
        n, k = W.shape
        cols = sorted(rng.choice(n // T, min(n // T, args.max_cols), replace=False))
        fb = {fmt: [] for fmt in args.fmts}
        for c in cols:
            for fmt in args.fmts:
                Wt = bf16(W[c * T:(c + 1) * T].float().numpy())[:, :k - k % BLOCK[fmt]]
                alpha, ae, dnf, rho2 = stats_fmt(Wt, fmt)
                eh = H[rng.integers(0, 16, (T, 32))]
                b = draw_bytes(alpha[:, None] * Wt, eh, ae / 64, rng, fmt)
                fb[fmt].append(np.bincount(b.ravel()).max() / b.size)
        out[nm] = {"k": k, "n": n, **{f"f_B_max_{fmt}": float(max(v)) for fmt, v in fb.items()},
                   **{f"f_B_median_{fmt}": float(np.median(v)) for fmt, v in fb.items()}}
        print(f"{nm:45s} k={k:6d} n={n:6d} " + "  ".join(f"{fmt}: f_B max {max(v):.3f} median {np.median(v):.3f}"
                                                        for fmt, v in fb.items()) + f"  [{time.time()-t0:.0f}s]", flush=True)
    summ = {"model": args.model, "linears": len(out),
            **{f"f_B_max_{fmt}": max(r[f"f_B_max_{fmt}"] for r in out.values()) for fmt in args.fmts},
            **{f"f_B_median_{fmt}": float(np.median([r[f"f_B_median_{fmt}"] for r in out.values()])) for fmt in args.fmts}}
    print(json.dumps(summ), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summ, "linears": out}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("attack")
    a.add_argument("--rows", type=int, default=64)
    a.add_argument("--k", type=int, default=8192)
    a.add_argument("--laws", nargs="+", default=["exact", "gauss"])
    a.add_argument("--draws", type=int, default=16, help="draws for the realised check (0: skip it)")
    a.add_argument("--out", default=None)
    r = sub.add_parser("real")
    r.add_argument("model")
    r.add_argument("--tokens", type=int, default=256)
    r.add_argument("--law", choices=("exact", "gauss"), default="exact")
    r.add_argument("--max-cols", type=int, default=8, help="64-row weight tiles sampled per linear")
    r.add_argument("--layer-step", type=int, default=1, help="decoder layers whose index is a multiple of this, and the last")
    r.add_argument("--m-shape", type=int, default=0, help="the GEMM's m for F2's c_L (default: the tokens captured)")
    r.add_argument("--fmts", nargs="+", choices=("nv", "mx"), default=["nv", "mx"], help="NVFP4 and/or MXFP4 forming")
    r.add_argument("--offload", default=None, help="CPU memory for the forward, e.g. 9GiB; the rest streams from disk")
    r.add_argument("--offload-folder", default="/tmp/offload")
    r.add_argument("--text", default=str(Path(__file__).resolve().parents[2] / "README.md"))
    r.add_argument("--out", default=None)
    f = sub.add_parser("flatness")
    f.add_argument("model")
    f.add_argument("--max-cols", type=int, default=8)
    f.add_argument("--layer-step", type=int, default=1)
    f.add_argument("--fmts", nargs="+", choices=("nv", "mx"), default=["nv", "mx"])
    f.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    {"attack": attack, "real": real, "flatness": flatness}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
