"""Attacks on aggregated words, each checked exactly on Dither's coins (`circuit.dither`'s seeds, V, U, π and dithers).

A group is a tile of g_r rows by g_c columns of one call's output, and its word, for segment s of L inner indices, is
A_s = sum over the tile of f(z_ij) mod 2^16, where z_ij = sum_{k in seg s} X_ik W′_jk (X = wrap8(A + na), W′ = wrap8(B + nb)).
Each attack computes some or all of a word from less than the tile's g·L MACs, and is held to the direct word on every
tile of the call."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

from verity.core.protocols.pouw.circuit import dither as D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agg_f as F  # noqa: E402

N, M16 = F.N, F.M16
COIN = hashlib.sha512(b"agg/coin").digest()
RECEIPT = hashlib.sha512(b"agg/receipt").digest()
SEEDS = D.seeds(COIN, RECEIPT, 0)


def call(sh, A, B):
    na = np.array(D.noise_a(sh, SEEDS, range(sh.m)), dtype=np.int64)
    nb = np.array(D.noise_b(sh, SEEDS, range(sh.n)), dtype=np.int64)
    return D.wrap8(A + na), D.wrap8(B + nb)


def seg_sums(sh, X, W):
    """z[s] = the segment-s sums X W′ᵀ, exact (S × m × n)."""
    return np.stack([X[:, lo:hi] @ W[:, lo:hi].T for lo, hi in (sh.seg(s) for s in range(1, sh.S + 1))])


def words(z, f, gr, gc):
    """sum over each gr × gc tile of f(z mod 2^16), mod 2^16."""
    m, n = z.shape
    fz = f(z.astype(np.int64) & M16).astype(np.int64)
    return fz.reshape(m // gr, gr, n // gc, gc).sum(axis=(1, 3)) & M16


def tile_sums(M, g, axis):
    s = M.reshape(M.shape[0] // g, g, -1).sum(axis=1) if axis == 0 else None
    return s


def linear_collapse(sh, X, W, z, gr, gc):
    """f = z mod 2^16: a tile's word is (sum of its rows of X) · (sum of its columns of W′) mod 2^16, one dot a tile."""
    ok = True
    for s in range(1, sh.S + 1):
        lo, hi = sh.seg(s)
        xs = X[:, lo:hi].reshape(sh.m // gr, gr, -1).sum(axis=1)
        ws = W[:, lo:hi].reshape(sh.n // gc, gc, -1).sum(axis=1)
        ok &= bool(np.array_equal((xs @ ws.T) & M16, words(z[s - 1], F.ident, gr, gc)))
    return {"exact_on_every_tile": ok, "tensor_share": 1 / (gr * gc),
            "extra": "row and column sums: (m/g_c... ) CUDA-core adds, (gr−1)/gr·mL + (gc−1)/gc·nL a segment"}


def separation(sh, z):
    """A = B = 0: each cell's segment sum from the floor note's split, d_A,s d_B,sᵀ + (D_s + V M_s) Uᵀ + V Y_sᵀ, then the
    aggregate. Equal to the direct aggregate on every tile, at the per-cell floor's cost, for any f."""
    lp, t = sh.lp, sh.L // sh.lp
    V = np.array(D._noise(SEEDS["V"], "V", range(sh.m), lp), dtype=np.int64)
    U = np.array(D._noise(SEEDS["U"], "U", range(sh.n), lp), dtype=np.int64)
    da = np.array(D.dithers(sh, SEEDS, "A", range(sh.m)), dtype=np.int64)
    db = np.array(D.dithers(sh, SEEDS, "B", range(sh.n)), dtype=np.int64)
    pi = D.perms(SEEDS, sh.k, lp)
    out = []
    for s in range(1, sh.S + 1):
        lo, hi = sh.seg(s)
        taus = range(lo // lp, hi // lp)
        Ds = np.zeros((sh.m, lp), dtype=np.int64)
        Ms = np.zeros((lp, lp), dtype=np.int64)
        Ys = np.zeros((sh.n, lp), dtype=np.int64)
        for tau in taus:
            P = np.zeros((lp, lp), dtype=np.int64)
            P[pi[tau], np.arange(lp)] = 1
            Ds += da[:, tau * lp:(tau + 1) * lp] @ P.T
            Ms += P.T
            Ys += db[:, tau * lp:(tau + 1) * lp]
        zs = da[:, lo:hi] @ db[:, lo:hi].T + (Ds + V @ Ms) @ U.T + V @ Ys.T
        out.append(bool(np.array_equal(zs, z[s - 1])))
    t_ = len(range(sh.seg(1)[0] // lp, sh.seg(1)[1] // lp))
    return {"t": t_, "split_equals_cells_every_segment": all(out),
            "floor_share_of_G": min(1.0, 1 / 8 + 3 / t_),
            "note": "the aggregate is a function of the cells, so it inherits this route: floor min(1, 1/8 + 3/t) (b1 at 8x s8, the coordinator's rate)"}


def low_bits_tfun(sh, X, W, z, gr, gc, cmax=3):
    """f a T-function: a word's low c bits are a function of the cells mod 2^c, which the operands mod 2^c give
    (c(c+1)/2 binary products, c(c+1)/16 of G at b1 = 8x s8). Bit 0 is affine in the cells' parities, so it collapses."""
    res = {}
    for c in range(1, cmax + 1):
        mc = (1 << c) - 1
        ok = True
        for s in range(1, sh.S + 1):
            lo, hi = sh.seg(s)
            zc = ((X[:, lo:hi] & mc) @ (W[:, lo:hi] & mc).T) & mc
            want = words(z[s - 1], F.tfun, gr, gc) & mc
            got = words(zc, F.tfun, gr, gc) & mc
            ok &= bool(np.array_equal(got, want))
        res[c] = {"low_bits_from_operands_mod_2^c": ok, "binary_products": c * (c + 1) // 2,
                  "share_of_G_at_b1": c * (c + 1) / 16, "guess_rest": f"2^-{16 - c}"}
    S = (X.reshape(sh.m // gr, gr, -1).sum(axis=1) @ W.reshape(sh.n // gc, gc, -1).sum(axis=1).T)
    seg1 = sh.seg(1)
    xs = X[:, seg1[0]:seg1[1]].reshape(sh.m // gr, gr, -1).sum(axis=1)
    ws = W[:, seg1[0]:seg1[1]].reshape(sh.n // gc, gc, -1).sum(axis=1)
    bit0 = ((xs @ ws.T) + gr * gc) & 1
    res["bit0_collapses_to_parity_of_the_tile_dot"] = bool(np.array_equal(bit0, words(z[0], F.tfun, gr, gc) & 1))
    return res


def xor_bit0(sh, X, W, z, gr, gc):
    """XOR in place of + aggregates: bit 0 of XOR_j f(z_j) for f = z is the parity of the tile's one dot."""
    lo, hi = sh.seg(1)
    zz = z[0].astype(np.int64) & M16
    m, n = zz.shape
    x = np.bitwise_xor.reduce(np.bitwise_xor.reduce(zz.reshape(m // gr, gr, n // gc, gc), axis=3), axis=1)
    xs = X[:, lo:hi].reshape(m // gr, gr, -1).sum(axis=1)
    ws = W[:, lo:hi].reshape(n // gc, gc, -1).sum(axis=1)
    return {"bit0_equals_parity_of_tile_dot": bool(np.array_equal(x & 1, (xs @ ws.T) & 1))}


def quadratic_collapse(L=32, g=128, m=256, seed=1):
    """f = z^2: a row group's word is x_iᵀ M_G x_i with M_G = sum_{j in G} w′_j w′_jᵀ (L × L), precomputable once a
    period for weight-side noise. Per row and group it costs L² MACs at 16-bit M_G (2 int8 limbs: 2L² s8 MACs) against
    g·L direct, so it wins at g > 2L."""
    rng = np.random.default_rng(seed)
    X = rng.integers(-80, 82, size=(m, L))
    W = rng.integers(-80, 82, size=(g, L))
    z = X @ W.T
    want = (z * z).sum(axis=1) & M16
    MG = (W.T @ W) & M16
    got = np.einsum("il,lk,ik->i", X, MG, X) & M16
    return {"L": L, "g": g, "exact": bool(np.array_equal(got, want)), "share_of_direct": 2 * L / g}


def truncation_guess(L=256, gs=(16, 64, 256), groups=40_000, b=8, seed=2):
    """f = z mod 2^b (b < 16): the word's low b bits are the tile's one dot mod 2^b, and its high part, sum_j (z_j mod
    2^b) div 2^b, is a near-Gaussian count around g/2 with sd sqrt(g/12). Guessing its mode succeeds about
    1/sqrt(2π g/12) of the time, at 1/g of G."""
    rng = np.random.default_rng(seed)
    out = {}
    for g in gs:
        hits = 0
        done = 0
        while done < groups:
            B_ = min(4096, groups - done)
            x = (rng.integers(-80, 81, size=(B_, L)) + rng.integers(0, 2, size=(B_, L))).astype(np.float64)
            w = (rng.integers(-80, 81, size=(B_, g, L)) + rng.integers(0, 2, size=(B_, g, L))).astype(np.float64)
            z = np.einsum("bl,bgl->bg", x, w).astype(np.int64)
            word = (z & ((1 << b) - 1)).sum(axis=1) & M16
            S = z.sum(axis=1)
            low = S & ((1 << b) - 1)
            centre = g * ((1 << b) - 1) / 2
            q = np.rint((centre - low) / (1 << b))
            guess = (low + (q * (1 << b)).astype(np.int64)) & M16
            hits += int((guess == word).sum())
            done += B_
        out[g] = {"success": hits / groups, "predicted": 1 / math.sqrt(2 * math.pi * g / 12), "tensor_share": 1 / g}
    return out


def main(out: str) -> None:
    res = {}
    sh = D.Shape(128, 128, 512, lp=32, L=256, row=1024)
    rng = np.random.default_rng(7)
    A = rng.integers(-128, 128, size=(sh.m, sh.k))
    B = rng.integers(-128, 128, size=(sh.n, sh.k))
    X, W = call(sh, A, B)
    z = seg_sums(sh, X, W)
    T, O = D.values(sh, SEEDS, A.tolist(), B.tolist())
    res["shape"] = {"m": sh.m, "n": sh.n, "K": sh.k, "Lp": sh.lp, "L": sh.L, "S": sh.S}
    res["cells_match_dither_values_T1"] = bool(np.array_equal(np.array(T[0]) & M16, z[0] & M16))
    for gr, gc in ((1, 8), (8, 8), (4, 32)):
        key = f"{gr}x{gc}"
        res.setdefault("linear_collapse", {})[key] = linear_collapse(sh, X, W, z, gr, gc)
        res.setdefault("tfun_low_bits", {})[key] = low_bits_tfun(sh, X, W, z, gr, gc)
        res.setdefault("xor_bit0", {})[key] = xor_bit0(sh, X, W, z, gr, gc)
    X0, W0 = call(sh, np.zeros_like(A), np.zeros_like(B))
    z0 = seg_sums(sh, X0, W0)
    res["separation_A=B=0_t8"] = separation(sh, z0)
    sh3 = D.Shape(128, 128, 384, lp=32, L=96, row=1024)
    X3, W3 = call(sh3, np.zeros((sh3.m, sh3.k), dtype=np.int64), np.zeros((sh3.n, sh3.k), dtype=np.int64))
    res["separation_A=B=0_t3"] = separation(sh3, seg_sums(sh3, X3, W3))
    res["quadratic_collapse"] = [quadratic_collapse(32, 128), quadratic_collapse(64, 64)]
    res["truncation_guess_b8"] = truncation_guess()
    print(json.dumps(res, indent=1))
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/agg/attacks.json")
