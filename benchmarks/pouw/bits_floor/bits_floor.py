"""How few bits of a committed word still force the work (GD-BITS, 28 Oct 2026): CPU checks of the arithmetic behind
the floor tables, and the tables.

Costs are in units of G, the plain s8 GEMM of the same product at the H200's s8 tensor rate, with the card's weights
(verity/hardware/cards/h200-sxm-141gb.toml on cursor/compute-research-h200-tensor-main-ea24, `Accel.H200.weight`): s8,
u8, e4m3 and e5m2 at 1, b1 at 1/8, f16 at 2; the card has no s4 and no FP4 MMA. Every check runs on a Dither call's
real coins (`dither.seeds`) at A = B = 0, the prover's best operands (note:compute-research/20261028T0523Z-finding-dither-
noised-floor), where X = d_A + V and W′ = d_B + Uπ exactly.

    python bits_floor.py --out DIR           # every experiment, then the tables: DIR/out.json and DIR/tables.md
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pathlib
import time

import numpy as np

from verity.core.protocols.pouw.circuit import dither as D

B1 = 1 / 8
STRASSEN = 7 / 8
COIN = hashlib.sha512(b"gd-bits coin").digest()
RECEIPT = hashlib.sha512(b"gd-bits receipt").digest()

#: the designs: (K, L′, L); m = n = 128, one tile
DESIGNS = {
    "seg-t8": (2048, 128, 1024),
    "seg-t3": (1536, 256, 768),
    "cell-fresh": (2048, 2048, 2048),
}


def mask(n: int) -> int:
    return (1 << n) - 1


def wrap8(x):
    return ((x + 128) & 255) - 128


def pairs(n: int, wa: int = 8, wb: int = 8) -> int:
    """Plane pairs (i, j), i < wa, j < wb, with i + j < n: the b1 products x·y mod 2^n takes from wa- and wb-bit
    operands (two's complement, the sign plane's weight −2^(w−1) ≡ 2^(w−1)·(2^(n−w+1) − 1) mod 2^n)."""
    return sum(1 for i in range(wa) for j in range(wb) if i + j < n)


def planes(x, w: int):
    """x's w two's-complement planes as 0/1 int64 arrays, and their weights (the top one negative)."""
    u = x & mask(w)
    return [((u >> i) & 1).astype(np.int64) for i in range(w)], [1 << i for i in range(w - 1)] + [-(1 << (w - 1))]


def plane_product(x, y, n: int, wa: int, wb: int):
    """x yᵀ mod 2^n from b1 products alone, every plane pair with i + j < n."""
    px, wx = planes(x, wa)
    py, wy = planes(y, wb)
    acc = np.zeros((x.shape[0], y.shape[0]), np.int64)
    used = 0
    for i, j in itertools.product(range(wa), range(wb)):
        if i + j < n:
            acc += (wx[i] * wy[j]) * (px[i] @ py[j].T)
            used += 1
    return acc & mask(n), used


def bits_needed(lo: int, hi: int) -> int:
    """Two's-complement width of [lo, hi]."""
    w = 1
    while not (-(1 << (w - 1)) <= lo and hi < (1 << (w - 1))):
        w += 1
    return w


# -- a call's coins ---------------------------------------------------------------------------------------------------------------

class Call:
    def __init__(self, K: int, lp: int, L: int, m: int = 128, n: int = 128, call: int = 0):
        self.sh = sh = D.Shape(m, n, K, lp=lp, L=L)
        sd = D.seeds(COIN, RECEIPT, call)
        self.K, self.lp, self.L, self.t, self.S = K, lp, L, sh.t, sh.S
        self.dA = np.array(D.dithers(sh, sd, "A", range(m)), np.int64)
        self.dB = np.array(D.dithers(sh, sd, "B", range(n)), np.int64)
        self.X = np.array(D.noise_a(sh, sd, range(m)), np.int64)
        self.P = np.array(D.perms(sd, K, lp), np.int64)
        self.W = np.array(D.noise_b(sh, sd, range(n), pi=self.P.tolist()), np.int64)
        self.V = (self.X - self.dA)[:, :lp]
        nb = self.W - self.dB
        self.U = np.empty((n, lp), np.int64)
        self.U[:, self.P[0]] = nb[:, :lp]
        assert np.array_equal(self.X, self.dA + np.tile(self.V, K // lp))
        assert np.array_equal(self.W, self.dB + self.U[:, self.P.reshape(-1)])
        self.segs = [sh.seg(s) for s in range(1, self.S + 1)]
        self.w = np.stack([self.X[:, a:b] @ self.W[:, a:b].T for a, b in self.segs])
        T = np.cumsum(self.w, axis=0) & D.M16
        for i, j in [(0, 0), (5, 77), (127, 127)]:
            ref, _ = D.cell_values(sh, sd, [0] * K, [0] * K, i, j)
            assert [int(v) for v in T[:, i, j]] == ref, (i, j)

    def split(self, s: int):
        """Segment s's (0-based) separated factors: bin = d_A d_Bᵀ, F = D_s + V M_s (m × L′), Y_s (n × L′)."""
        a, b = self.segs[s]
        lp = self.lp
        Ds = np.zeros((self.X.shape[0], lp), np.int64)
        M = np.zeros((lp, lp), np.int64)
        Y = np.zeros((self.W.shape[0], lp), np.int64)
        for tau in range(a // lp, b // lp):
            p = self.P[tau]
            Ds[:, p] += self.dA[:, tau * lp:(tau + 1) * lp]
            M[np.arange(lp), p] += 1
            Y += self.dB[:, tau * lp:(tau + 1) * lp]
        return self.dA[:, a:b] @ self.dB[:, a:b].T, Ds + self.V @ M, Y


# -- E1: bit planes ---------------------------------------------------------------------------------------------------------------

def e1(rng) -> dict:
    """x yᵀ mod 2^n from b1 plane products, exact, on uniform int8 and on Dither's noise; and the plane counts."""
    X = rng.integers(-128, 128, (64, 512))
    W = rng.integers(-128, 128, (48, 512))
    out = {}
    for n in range(1, 17):
        acc, used = plane_product(X, W, n, 8, 8)
        out[n] = {"pairs": used, "pairs_formula": pairs(n), "b1_cost": used * B1,
                  "exact": bool(np.array_equal(acc, (X @ W.T) & mask(n)))}
    return out


# -- E2: the least number of binary products for x·y mod 4 ------------------------------------------------------------------------

def e2() -> dict:
    """Exhaustive: the fewest entrywise binary products c_l f_l(x) g_l(y), f, g : Z/4 → {0, 1}, c_l ∈ Z/4, whose sum is
    x·y mod 4 for every x, y. Any b1 route that treats inner indices alike is such a sum."""
    target = np.array([[(x * y) % 4 for y in range(4)] for x in range(4)]).reshape(-1)
    fs = [np.array([(f >> x) & 1 for x in range(4)]) for f in range(1, 16)]
    tabs = np.array([np.outer(f, g).reshape(-1) for f in fs for g in fs])
    r1 = any(np.array_equal((c * row) % 4, target) for c in (1, 2, 3) for row in tabs)
    r2 = False
    for c1, c2 in itertools.product((1, 2, 3), repeat=2):
        M = (c1 * tabs[:, None, :] + c2 * tabs[None, :, :]) % 4
        r2 |= bool((M == target).all(-1).any())
    r3 = all(((x & 1) * (y & 1) + 2 * ((x & 1) * (y >> 1)) + 2 * ((x >> 1) * (y & 1))) % 4 == (x * y) % 4
             for x in range(4) for y in range(4))
    return {"tables": len(tabs), "r1": r1, "r2": r2, "planes_r3_exact": r3,
            "min_products_b2": 3 if (not r1 and not r2 and r3) else None,
            "counting_lower_bound": "f's values must separate x mod 2^b: r ≥ b"}


# -- E3: Strassen d = 1 over Z/2^n ------------------------------------------------------------------------------------------------

#: (A-side form, B-side form) per product as {block: coefficient}, and C's blocks as {product: coefficient}; B = Wᵀ
SCHEMES = {
    "strassen": (
        [({"11": 1, "22": 1}, {"11": 1, "22": 1}), ({"21": 1, "22": 1}, {"11": 1}), ({"11": 1}, {"12": 1, "22": -1}),
         ({"22": 1}, {"21": 1, "11": -1}), ({"11": 1, "12": 1}, {"22": 1}), ({"21": 1, "11": -1}, {"11": 1, "12": 1}),
         ({"12": 1, "22": -1}, {"21": 1, "22": 1})],
        {"11": {0: 1, 3: 1, 4: -1, 6: 1}, "12": {2: 1, 4: 1}, "21": {1: 1, 3: 1}, "22": {0: 1, 1: -1, 2: 1, 5: 1}}),
    "winograd": (
        [({"11": 1}, {"11": 1}), ({"12": 1}, {"21": 1}), ({"11": 1, "12": 1, "21": -1, "22": -1}, {"22": 1}),
         ({"22": 1}, {"21": 1, "12": 1, "11": -1, "22": -1}), ({"21": 1, "22": 1}, {"12": 1, "11": -1}),
         ({"21": 1, "22": 1, "11": -1}, {"11": 1, "22": 1, "12": -1}), ({"21": 1, "11": -1}, {"12": 1, "22": -1})],
        {"11": {0: 1, 1: 1}, "12": {0: 1, 5: 1, 4: 1, 2: 1}, "21": {0: 1, 5: 1, 6: 1, 3: 1},
         "22": {0: 1, 5: 1, 6: 1, 4: 1}}),
}


def _blocks(X):
    a, b = X.shape[0] // 2, X.shape[1] // 2
    return {"11": X[:a, :b], "12": X[:a, b:], "21": X[a:, :b], "22": X[a:, b:]}


def _bblocks(W):
    """Wᵀ's blocks, each as the n/2 × K/2 matrix whose transpose it is."""
    w = _blocks(W)
    return {"11": w["11"], "12": w["21"], "21": w["12"], "22": w["22"]}


def _form(bl, coef):
    return sum(c * bl[k] for k, c in coef.items())


def strassen(X, W, scheme: str, n: int, correct: bool = False):
    """X Wᵀ mod 2^n by one level of `scheme` with every form wrapped to int8 (an s8 MMA's operand); with `correct`,
    plus 256·(e·g + f·e′) mod 2^n from the wrap counts e (exact for n ≤ 16). Returns the result and each form-side's
    wrap density and wrap range."""
    prods, comb = SCHEMES[scheme]
    A, B = _blocks(X), _bblocks(W)
    Ps, sides = [], []
    for fa, fb in prods:
        F, G = _form(A, fa), _form(B, fb)
        a, b = wrap8(F), wrap8(G)
        e, f = (F - a) >> 8, (G - b) >> 8
        P = a @ b.T
        if correct:
            P = P + 256 * (e @ b.T + a @ f.T) + 65536 * (e @ f.T)
        Ps.append(P)
        for coef, ee in ((fa, e), (fb, f)):
            if len(coef) > 1:
                sides.append({"terms": len(coef), "density": float((ee != 0).mean()), "e": [int(ee.min()), int(ee.max())]})
    C = {k: sum(c * Ps[p] for p, c in v.items()) for k, v in comb.items()}
    out = np.block([[C["11"], C["12"]], [C["21"], C["22"]]])
    return out & mask(n), sides


def strassen_mod2_correction(X, W, scheme: str):
    """X Wᵀ mod 2^9 from wrapped forms plus the mod-2 corrections as b1 products: 256·((e mod 2)·(g mod 2) + ...)."""
    prods, comb = SCHEMES[scheme]
    A, B = _blocks(X), _bblocks(W)
    Ps, nb1 = [], 0
    for fa, fb in prods:
        F, G = _form(A, fa), _form(B, fb)
        a, b = wrap8(F), wrap8(G)
        e, f = ((F - a) >> 8) & 1, ((G - b) >> 8) & 1
        corr = np.zeros((a.shape[0], b.shape[0]), np.int64)
        if len(fa) > 1:
            corr += e @ (b & 1).T
            nb1 += 1
        if len(fb) > 1:
            corr += (a & 1) @ f.T
            nb1 += 1
        Ps.append(a @ b.T + 256 * (corr & 1))
    C = {k: sum(c * Ps[p] for p, c in v.items()) for k, v in comb.items()}
    return np.block([[C["11"], C["12"]], [C["21"], C["22"]]]) & mask(9), nb1


def e3(calls: dict, rng) -> dict:
    out = {}
    ops = {"dither-A0B0": (calls["seg-t8"].X, calls["seg-t8"].W),
           "dither-A0B0-fresh": (calls["cell-fresh"].X, calls["cell-fresh"].W),
           "uniform-int8": (rng.integers(-128, 128, (128, 2048)), rng.integers(-128, 128, (128, 2048)))}
    for name, (X, W) in ops.items():
        ref = X @ W.T
        r = {}
        for scheme in SCHEMES:
            full, _ = strassen(X, W, scheme, 40, correct=True)
            row = {"exact_over_Z_with_corrections": bool(np.array_equal(full, ref & mask(40)))}
            for n in (1, 2, 4, 8, 9, 10, 12, 16):
                got, sides = strassen(X, W, scheme, n)
                row[f"wrong_cells_mod2^{n}"] = float((got != (ref & mask(n))).mean())
            got9, nb1 = strassen_mod2_correction(X, W, scheme)
            row["mod2^9_with_b1_corrections_exact"] = bool(np.array_equal(got9, ref & mask(9)))
            row["b1_correction_products"] = nb1
            row["form_sides"] = sides
            r[scheme] = row
        out[name] = r
    return out


# -- E4: the separated route, by bits ---------------------------------------------------------------------------------------------

def hi_bits(t: int) -> int:
    """The high limb's width for F = D_s + V M_s ∈ [−80t, 81t], the narrower of the s8 split (F − wrap8(F)) / 256 and
    the u8 split floor(F / 256) (wgmma takes u8 × s8)."""
    lo, hi = -NOISE * t, (NOISE + 1) * t
    return min(bits_needed((lo + 128) >> 8, (hi + 128) >> 8), bits_needed(lo >> 8, hi >> 8))


def y_bits(t: int) -> int:
    """Y_s ∈ [0, t]'s unsigned planes."""
    return max(1, t.bit_length())


NOISE = D.NOISE


def e4(calls: dict) -> dict:
    """At each design and n: T_s − T_{s−1} mod 2^n from b1 d_A d_Bᵀ, F's u8 low limb, F's high limb by planes, and V Y_sᵀ
    by planes, against the reference; and the factors' widths against their bounds."""
    out = {}
    for name, c in calls.items():
        r = {"t": c.t, "L'": c.lp, "L": c.L, "hi_bits": hi_bits(c.t), "Y_bits": y_bits(c.t),
             "F_bound": [-NOISE * c.t, (NOISE + 1) * c.t]}
        widths = {"F": [0, 0], "hi_u8split": [0, 0], "Y": [0, 0]}
        exact = {}
        for s in range(c.S):
            bin_, F, Y = c.split(s)
            lo = F & 255
            hi = F >> 8
            for k, v in (("F", F), ("hi_u8split", hi), ("Y", Y)):
                widths[k] = [min(widths[k][0], int(v.min())), max(widths[k][1], int(v.max()))]
            assert np.array_equal(bin_ + F @ c.U.T + c.V @ Y.T, c.w[s])
            whi = bits_needed(int(hi.min()), int(hi.max()))
            for n in range(1, 17):
                lo_p = plane_product(lo, c.U, n, 9, 8)[0] if n <= 3 else (lo @ c.U.T) & mask(n)
                hi_p = (plane_product(hi, c.U, n - 8, whi, 8)[0] << 8) if n > 8 else 0
                y_p = plane_product(Y, c.V, n, y_bits(c.t) + 1, 8)[0].T if n <= 3 else (c.V @ Y.T) & mask(n)
                got = (bin_ + lo_p + hi_p + y_p) & mask(n)
                exact[n] = exact.get(n, True) and bool(np.array_equal(got, c.w[s] & mask(n)))
        r["widths"] = widths
        r["exact_by_n"] = exact
        out[name] = r
    return out


# -- E5: approximate routes -------------------------------------------------------------------------------------------------------

def _marg(x, y, L):
    return np.rint(np.outer(x.sum(1), y.sum(1)) / L).astype(np.int64)


ROUTES = ("sep-nobin", "sep-noY", "sep-truncF", "sep-truncF-noY", "sep-truncF-noY-nobin", "direct-trunc-strassen-q1",
          "direct-trunc-winograd-q2", "direct-strassen-nocorr", "direct-b1-top-planes")


def approx_routes(c: Call):
    """Each route's estimate of every segment sum, at A = B = 0: the dropped terms replaced by their marginal estimate
    (row sums, CUDA-core work), never by anything that needs the dropped product."""
    est = {k: [] for k in ROUTES}
    meta = {}
    for s, (a, b) in enumerate(c.segs):
        bin_, F, Y = c.split(s)
        w = c.w[s]
        dA, dB = c.dA[:, a:b], c.dB[:, a:b]
        e_bin = _marg(dA, dB, b - a)
        e_Y = _marg(c.V, Y, c.lp)
        q = 0
        while np.abs(np.rint(F / (1 << q))).max() > 127:
            q += 1
        Fq = np.rint(F / (1 << q)).astype(np.int64) << q
        meta["truncF_q"] = max(meta.get("truncF_q", 0), q)
        FU, VY, FqU = F @ c.U.T, c.V @ Y.T, Fq @ c.U.T
        est["sep-nobin"].append(w - bin_ + e_bin)
        est["sep-noY"].append(w - VY + e_Y)
        est["sep-truncF"].append(w - FU + FqU)
        est["sep-truncF-noY"].append(w - FU + FqU - VY + e_Y)
        est["sep-truncF-noY-nobin"].append(w - FU + FqU - VY + e_Y - bin_ + e_bin)
        X, W = c.X[:, a:b], c.W[:, a:b]
        L = b - a
        for key, qq, scheme in (("direct-trunc-strassen-q1", 1, "strassen"), ("direct-trunc-winograd-q2", 2, "winograd")):
            Xh, Wh = X >> qq, W >> qq
            xl, wl = X - (Xh << qq), W - (Wh << qq)
            got, sides = strassen(Xh, Wh, scheme, 40)
            assert all(sd["density"] == 0 for sd in sides), (key, sides)
            hh = Xh @ Wh.T
            assert np.array_equal(got, hh & mask(40))
            est[key].append((hh << (2 * qq)) + (_marg(Xh, wl, L) << qq) + (_marg(xl, Wh, L) << qq) + _marg(xl, wl, L))
        z, _ = strassen(X, W, "winograd", 40)
        est["direct-strassen-nocorr"].append(((z + (1 << 39)) & mask(40)) - (1 << 39))
        px, wx = planes(X, 8)
        py, wy = planes(W, 8)
        top = sum(wx[i] * wy[j] * (px[i] @ py[j].T) for i in range(8) for j in range(8) if i + j >= TOP)
        rest = sum(wx[i] * wy[j] * np.outer(px[i].sum(1), py[j].sum(1)) / L for i in range(8) for j in range(8)
                   if i + j < TOP)
        est["direct-b1-top-planes"].append(top + np.rint(rest).astype(np.int64))
    return {k: np.stack(v) for k, v in est.items()}, meta


#: the top-plane route keeps the b1 pairs with i + j ≥ TOP: six of them, 3/4 of G
TOP = 12


def lo_cost(n: int) -> float:
    return min(1.0, pairs(n) * B1)


def route_costs(n: int, t: int) -> dict:
    """Each approximate route's tensor cost (share of G) at a window ending at bit n: the products it keeps, mod 2^n."""
    lo, y = lo_cost(n), min(1.0, pairs(n, y_bits(t), 8) * B1)
    hi = min(1.0, pairs(n - 8, hi_bits(t), 8) * B1) if n > 8 else 0.0
    return {
        "sep-nobin": (lo + hi + y) / t,
        "sep-noY": B1 + (lo + hi) / t,
        "sep-truncF": B1 + (lo + y) / t,
        "sep-truncF-noY": B1 + lo / t,
        "sep-truncF-noY-nobin": lo / t,
        "direct-trunc-strassen-q1": STRASSEN * lo,
        "direct-trunc-winograd-q2": STRASSEN * lo,
        "direct-strassen-nocorr": STRASSEN * lo,
        "direct-b1-top-planes": B1 * sum(1 for i in range(8) for j in range(8) if TOP <= i + j < max(n, TOP + 1)),
    }


def word_bits(design: str) -> int:
    """A segment's word is Dither's u16; one word a cell holds its whole sum over K, 32 bits here (O's width)."""
    return 32 if design.startswith("cell") else 16


def window(x, r: int, b: int):
    return (x >> r) & mask(b)


def windows(design: str):
    bits = word_bits(design)
    return [(r, b) for r in range(0, min(bits, 24)) for b in range(1, 9) if r + b <= bits]


def e5(calls: dict) -> dict:
    out = {}
    for name, c in calls.items():
        est, meta = approx_routes(c)
        w = c.w
        r = {"meta": meta, "routes": {}}
        for k, e in est.items():
            if c.t == 1 and k.startswith("sep"):
                continue
            err = (e - w).reshape(-1)
            pe = {f"{rr},{b}": float((window(e, rr, b) != window(w, rr, b)).mean()) for rr, b in windows(name)}
            slack = {str(rr): float((np.abs((e >> rr) - (w >> rr)) >= 2).mean()) for rr in range(0, 20)}
            r["routes"][k] = {"err_std": float(err.std()), "err_mean": float(err.mean()),
                              "err_absmean": float(np.abs(err).mean()), "p_err": pe, "p_err_slack": slack}
        out[name] = r
    return out


# -- E6: entropy of windows ---------------------------------------------------------------------------------------------------------

def e6(calls: dict) -> dict:
    """Per window [r, r + b) of the segment sum: the most frequent value's rate and the hit rate of the free marginal
    estimate; the larger is the window's free guess rate p."""
    out = {}
    for name, c in calls.items():
        w = c.w
        free = np.stack([_marg(c.X[:, a:b], c.W[:, a:b], b - a) for a, b in c.segs])
        r = {"sum_std": float(w.std()), "log2_std": math.log2(float(w.std())), "p": {}}
        for rr, b in windows(name):
            v = window(w, rr, b).reshape(-1)
            top = np.bincount(v, minlength=1 << b).max() / v.size
            hit = float((window(free, rr, b) == window(w, rr, b)).mean())
            r["p"][f"{rr},{b}"] = {"top": float(top), "free_hit": hit, "p": max(float(top), hit, 2.0 ** -b)}
        out[name] = r
    return out


# -- the price ------------------------------------------------------------------------------------------------------------------------

def direct_exact(n: int, strassen_ok: bool = True) -> float:
    """The direct product X W′ᵀ mod 2^n: b1 planes or s8, and Strassen at depth 1 wherever it needs no correction
    (n ≤ 8). At n = 9 Winograd's eight mod-2 corrections are b1 products at 1/64 a form-side: 7/8 + 1/8 = 1."""
    base = lo_cost(n)
    return STRASSEN * base if strassen_ok and n <= 8 else base


def sep_exact(n: int, t: int, strassen_ok: bool = True) -> float:
    """The separated route mod 2^n: b1 d_A d_Bᵀ, F's low limb, F's high limb by planes, V Y_sᵀ by planes or s8;
    Strassen at depth 1 on its products while n ≤ 8."""
    sig = STRASSEN if (strassen_ok and n <= 8) else 1.0
    lo, y = lo_cost(n), min(1.0, pairs(n, y_bits(t), 8) * B1)
    hi = min(1.0, pairs(n - 8, hi_bits(t), 8) * B1) if n > 8 else 0.0
    return B1 + sig * (lo + hi + y) / t


def floor_low(n: int, t: int) -> dict:
    d0, d1 = direct_exact(n, False), direct_exact(n)
    s0 = sep_exact(n, t, False) if t > 1 else float("inf")
    s1 = sep_exact(n, t) if t > 1 else float("inf")
    return {"n": n, "direct": d0, "direct_strassen": d1, "sep": s0, "sep_strassen": s1,
            "floor_no_strassen": min(d0, s0), "floor": min(d0, d1, s0, s1)}


def price(res: dict) -> dict:
    tabs = {}
    for name, (K, lp, L) in DESIGNS.items():
        t = L // lp
        bits = word_bits(name)
        rows = [floor_low(n, t) for n in range(1, bits + 1)]
        full = rows[-1]["floor"]
        full0 = rows[-1]["floor_no_strassen"]
        nstar = min(r["n"] for r in rows if r["floor"] >= full - 1e-12)
        nstar0 = min(r["n"] for r in rows if r["floor_no_strassen"] >= full0 - 1e-12)
        routes = res["e5"][name]["routes"]
        P = res["e6"][name]["p"]
        win = {}
        for rr, b in windows(name):
            n = rr + b
            exact = rows[n - 1]["floor"]
            p = P[f"{rr},{b}"]["p"]
            costs = route_costs(n, t)
            beat = []
            for k, v in routes.items():
                delta = full - costs[k]
                if delta <= 1e-12:
                    continue
                pe = v["p_err"][f"{rr},{b}"]
                need = delta / full * (1 - p)
                if pe < need:
                    beat.append((costs[k], k, pe, need))
            eff = min([exact] + [x[0] for x in beat])
            win[f"{rr},{b}"] = {"exact": exact, "p": p, "beaten_by": sorted(beat)[:1], "floor": eff,
                                "holds": eff >= full - 1e-12, "sampling": 1 / (1 - p)}
        by_b = {}
        for b in range(1, 9):
            ok = [(win[f"{rr},{b}"]["sampling"], rr) for rr, bb in windows(name) if bb == b and win[f"{rr},{b}"]["holds"]]
            best = min(ok) if ok else None
            lowfloor = rows[b - 1]["floor"]
            by_b[b] = {"low_bits_floor": lowfloor, "windows_that_hold": sorted(rr for _, rr in ok),
                       "best_r": best[1] if best else None, "sampling": best[0] if best else None,
                       "best_floor_any_window": max(win[f"{rr},{bb}"]["floor"] for rr, bb in windows(name) if bb == b)}
        bmin = min([b for b in by_b if by_b[b]["best_r"] is not None], default=None)
        tabs[name] = {"t": t, "K": K, "L'": lp, "L": L, "word_bits": bits, "rows": rows, "full_floor": full,
                      "full_floor_no_strassen": full0, "n_star": nstar, "n_star_no_strassen": nstar0, "b_min": bmin,
                      "by_b": by_b, "windows": win}
    return tabs


def tables_md(res: dict) -> str:
    out = []
    f = lambda x: "—" if x is None or x == float("inf") else f"{x:.4f}"
    for name, tb in res["price"].items():
        lp = tb["L'"]
        out.append(f"### {name}: t {tb['t']}, K {tb['K']}, L′ {lp}, L {tb['L']}, word {tb['word_bits']} bits\n")
        out.append(f"full-word floor {tb['full_floor']:.4f} G ({tb['full_floor_no_strassen']:.4f} without Strassen); "
                   f"low bits reach it at n* = {tb['n_star']} ({tb['n_star_no_strassen']} without Strassen); fewest "
                   f"bits in a window that holds it: {tb['b_min']}\n")
        out.append("| n low bits | direct | direct, Strassen d=1 | separated | separated, Strassen d=1 | floor |")
        out.append("|---|---|---|---|---|---|")
        for r in tb["rows"][:16]:
            out.append(f"| {r['n']} | {f(r['direct'])} | {f(r['direct_strassen'])} | {f(r['sep'])} | "
                       f"{f(r['sep_strassen'])} | {f(r['floor'])} |")
        out.append("")
        out.append("| b | floor, low b bits | windows [r, r+b) that hold the full floor (r) | best r | 1/(1−p) |")
        out.append("|---|---|---|---|---|")
        for b, v in tb["by_b"].items():
            out.append(f"| {b} | {f(v['low_bits_floor'])} | {v['windows_that_hold'] or '—'} | {v['best_r']} | "
                       f"{f(v['sampling'])} |")
        out.append("")
        out.append("| approximate route | cost / G at n = 16 | err std | p_err, b = 8 window at r = 4, 8, 10, 12, 14 (b = 2) |")
        out.append("|---|---|---|---|")
        costs = route_costs(16, tb["t"])
        for k, v in res["e5"][name]["routes"].items():
            ps = ", ".join(f"{v['p_err'][key]:.3g}" for key in ("4,8", "8,8", "10,6", "12,4", "14,2"))
            out.append(f"| {k} | {costs[k]:.4f} | {v['err_std']:.1f} | {ps} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261028)
    t0 = time.perf_counter()
    calls = {k: Call(*v) for k, v in DESIGNS.items()}
    res = {"designs": DESIGNS, "coin": COIN.hex(), "receipt": RECEIPT.hex()}
    res["e1"] = e1(rng)
    res["e2"] = e2()
    res["e3"] = e3(calls, rng)
    res["e4"] = e4(calls)
    res["e5"] = e5(calls)
    res["e6"] = e6(calls)
    res["price"] = price(res)
    res["seconds"] = time.perf_counter() - t0
    (a.out / "out.json").write_text(json.dumps(res, indent=1, default=str))
    (a.out / "tables.md").write_text(tables_md(res))
    print(tables_md(res))
    print(json.dumps({k: res[k] for k in ("e1", "e2")}, default=str)[:3000])


if __name__ == "__main__":
    main()
