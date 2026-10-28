"""How few bits of a committed word still force the work (GD-BITS, 28 Oct 2026): CPU checks of the arithmetic behind
the floor tables, and the tables.

Costs are in units of G, the plain s8 GEMM of the same product at the H200's s8 tensor rate, with the card's weights
(verity/hardware/cards/h200-sxm-141gb.toml on cursor/compute-research-h200-tensor-main-ea24, `Accel.H200.weight`): s8,
u8, e4m3 and e5m2 at 1, b1 at 1/8, f16 at 2; the card has no s4 and no FP4 MMA. Every check runs on a Dither call's
real coins (`dither.seeds`) at A = B = 0, the prover's best operands (note:compute-research/20261028T0523Z-finding-dither-
noised-floor), where X = d_A + V and W′ = d_B + Uπ exactly.

    python bits_floor.py --out DIR           # every experiment, then the tables: DIR/out.json and DIR/tables.md

The price is tensor-pipe time at the card's weights, as in the floor note (*tensor*), and *in-model*, the max over the
tensor and alu pipes (ACCEL's rate premise G2 bounds each resource on its own): Strassen at depth 1 or 2 with its
additions and sparse corrections on the alu (`direct_inmodel`), and F's high limb on a 2:4 sparse MMA (`Op.mmaSp`, E7),
its leftover on the alu. Every price is also given with b1 at 3/16 (`B1_WGMMA`, `price_wgmma` in out.json). Approximate
routes are judged by the rate condition of
`SAKMatmul` (Security/Definitions/Pouw/Assumptions/Dither.lean), linearized: a route of cost c failing a checked
window with probability p_err stays within the moment iff p_err ≥ (1 − c/pub)(1 − p).
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
#: b1 against wgmma's s8, which the served GEMM uses: 5.33× (r20261028-090323-2c56 measured 8.00× on mma.sync)
B1_WGMMA = 3 / 16
STRASSEN = 7 / 8
COIN = hashlib.sha512(b"gd-bits coin").digest()
RECEIPT = hashlib.sha512(b"gd-bits receipt").digest()

#: the designs: (K, L′, L); m = n = 128, one tile
DESIGNS = {
    "seg-t8": (2048, 128, 1024),
    "seg-t3": (1536, 256, 768),
    "seg-t3-pseg": (1536, 256, 768),
    "cell-fresh": (2048, 2048, 2048),
    "cell-jr2": (2048, 2048, 2048),
    "cell-2gemm": (2048, 2048, 2048),
}
#: the designs whose X at A = 0 is d_A alone. JR2 (J10-R v2, #6369): the word b·(c + U)ᵀ, b = A + d_A, c = B, U fresh
#: along K (#6369's U is on ±64; here Dither's U stream on ±80 at L′ = K). Fresh-R two-GEMM (GD-NOISE): the word
#: wrap8(A + d_A)·Rᵀ, R a fresh uniform int8 n × K matrix, which doesn't read B
BINX = {"cell-jr2", "cell-2gemm"}
#: seg-t3-pseg is Dither at t = 3 with one π a segment (GD-NOISE's fix): tile τ takes π of its segment's first tile,
#: so it is not v3.2's stream and isn't checked against `dither.cell_values`
KIND = {"cell-jr2": "jr2", "cell-2gemm": "2gemm", "seg-t3-pseg": "pseg"}
#: each design's operand set in E3, whose Strassen form-side wrap densities price its in-model corrections
SIDE = {"seg-t8": "dither-A0B0", "seg-t3": "dither-A0B0", "seg-t3-pseg": "dither-A0B0",
        "cell-fresh": "dither-A0B0-fresh", "cell-jr2": "jr2-A0B0", "cell-2gemm": "2gemm-A0"}
#: designs whose W′ side is drawn per call, so Strassen's W′-side forms are alu work a call; two-GEMM's R is per period
W_PER_CALL = {"seg-t8", "seg-t3", "seg-t3-pseg", "cell-fresh", "cell-jr2"}


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
    def __init__(self, K: int, lp: int, L: int, m: int = 128, n: int = 128, call: int = 0, kind: str = "dither"):
        self.sh = sh = D.Shape(m, n, K, lp=lp, L=L)
        sd = D.seeds(COIN, RECEIPT, call)
        self.K, self.lp, self.L, self.t, self.S = K, lp, L, sh.t, sh.S
        self.dA = np.array(D.dithers(sh, sd, "A", range(m)), np.int64)
        self.dB = np.array(D.dithers(sh, sd, "B", range(n)), np.int64)
        self.X = np.array(D.noise_a(sh, sd, range(m)), np.int64)
        self.P = np.array(D.perms(sd, K, lp), np.int64)
        if kind == "pseg":
            self.P = self.P[[(tau // sh.t) * sh.t for tau in range(K // lp)]]
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
        for i, j in [(0, 0), (5, 77), (127, 127)] if kind != "pseg" else []:
            ref, _ = D.cell_values(sh, sd, [0] * K, [0] * K, i, j)
            assert [int(v) for v in T[:, i, j]] == ref, (i, j)
        if kind in ("jr2", "2gemm"):
            self.X, self.W = self.dA, self.W - self.dB
            if kind == "2gemm":
                seed = hashlib.sha512(COIN + b"R" + call.to_bytes(8, "big")).digest()
                self.W = np.random.default_rng(int.from_bytes(seed[:8], "big")).integers(-128, 128, (n, K))
            self.w = np.stack([self.X[:, a:b] @ self.W[:, a:b].T for a, b in self.segs])

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
           "jr2-A0B0": (calls["cell-jr2"].X, calls["cell-jr2"].W),
           "2gemm-A0": (calls["cell-2gemm"].X, calls["cell-2gemm"].W),
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
        if name in BINX:
            continue
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


#: sep-nobin-kN keeps d_A d_Bᵀ exactly on the first N/8 of each segment's inner indices
PARTIAL = (2, 4, 6, 7)
ROUTES = ("sep-nobin", *(f"sep-nobin-k{k}" for k in PARTIAL), "sep-noY", "sep-truncF", "sep-truncF-noY", "sep-truncF-noY-nobin", "direct-trunc-strassen-q1",
          "direct-trunc-winograd-q2", "direct-strassen-nocorr", "direct-b1-top-planes", "jr2-trunc-strassen-Wq1",
          *(f"jr2-planes-q{q}" for q in range(1, 8)))


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
        for k in PARTIAL:
            h = (b - a) * k // 8
            part = dA[:, :h] @ dB[:, :h].T + _marg(dA[:, h:], dB[:, h:], b - a - h)
            est[f"sep-nobin-k{k}"].append(w - bin_ + part)
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
        Wh = W >> 1
        got, sides = strassen(X, Wh, "strassen", 40)
        if np.abs(X).max() <= 2:
            assert all(sd["density"] == 0 for sd in sides)
        est["jr2-trunc-strassen-Wq1"].append(2 * (X @ Wh.T) + _marg(X, W - 2 * Wh, L))
        for q in range(1, 8):
            hi_ = (W >> q) << q
            est[f"jr2-planes-q{q}"].append(X @ hi_.T + _marg(X, W - hi_, L))
    return {k: np.stack(v) for k, v in est.items()}, meta


#: the top-plane route keeps the b1 pairs with i + j ≥ TOP: six of them, 3/4 of G
TOP = 12


def lo_cost(n: int) -> float:
    return min(1.0, pairs(n) * B1)


def route_costs(n: int, t: int, sparse: bool = False) -> dict:
    """Each approximate route's tensor cost (share of G) at a window ending at bit n: the products it keeps, mod 2^n;
    with `sparse`, F's high limb on a 2:4 sparse MMA (E7)."""
    lo, y = lo_cost(n), min(1.0, pairs(n, y_bits(t), 8) * B1)
    hi = min(1.0, pairs(n - 8, hi_bits(t), 8) * B1) if n > 8 else 0.0
    if sparse and n > 8:
        hi = min(hi, 0.5)
    return {
        "sep-nobin": (lo + hi + y) / t,
        **{f"sep-nobin-k{k}": B1 * k / 8 + (lo + hi + y) / t for k in PARTIAL},
        "sep-noY": B1 + (lo + hi) / t,
        "sep-truncF": B1 + (lo + y) / t,
        "sep-truncF-noY": B1 + lo / t,
        "sep-truncF-noY-nobin": lo / t,
        "direct-trunc-strassen-q1": STRASSEN * lo,
        "direct-trunc-winograd-q2": STRASSEN * lo,
        "direct-strassen-nocorr": STRASSEN * lo,
        "direct-b1-top-planes": B1 * sum(1 for i in range(8) for j in range(8) if TOP <= i + j < max(n, TOP + 1)),
        "jr2-trunc-strassen-Wq1": STRASSEN * lo,
        **{f"jr2-planes-q{q}": B1 * max(0, min(n, 8) - q) for q in range(1, 8)},
    }


def model_costs(n: int, t: int, left: float | None, inmodel: bool) -> dict:
    """`route_costs` with F's high limb on a 2:4 sparse MMA wherever that's cheaper: in the tensor model only when its
    leftover's alu time fits in the limb's own tensor time (64·left ≤ 1/2), in-model as the max over the two pipes."""
    dense = route_costs(n, t)
    if left is None or n <= 8:
        return dense
    sp = route_costs(n, t, sparse=True)
    if not inmodel:
        return {k: min(dense[k], sp[k]) for k in dense} if 64 * left <= 0.5 else dense
    return {k: min(dense[k], max(sp[k], 64 * left / t)) for k in dense}


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
            if (c.t == 1 and k.startswith("sep")) or (k.startswith("jr2") != (name in BINX)) or (
                    name in BINX and k.startswith("direct-trunc")):
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


# -- E7: F's high limb on a 2:4 sparse MMA ----------------------------------------------------------------------------------------

def two_four(x):
    """x (rows × k, k a multiple of 4) split as kept + left: kept holds at most 2 nonzeros of each row's aligned group
    of 4 (a 2:4 sparse MMA's A, its selectors per row), left the rest."""
    g = x.reshape(x.shape[0], -1, 4)
    nz = (g != 0).astype(np.int64)
    keep = (np.cumsum(nz, axis=-1) <= 2) | (nz == 0)
    kept = np.where(keep, g, 0).reshape(x.shape)
    return kept, x - kept


def e7(calls: dict) -> dict:
    """Per segment design: F's high limb under the s8 split, F = wrap8(F) + 256 F_hi, as a 2:4 sparse operand (ACCEL's
    `mmaSp`, at the dense form's tensor time for twice the logical MACs) plus a leftover; the segment sum mod 2^16 from
    b1 d_A d_Bᵀ, F_lo Uᵀ, 256 (F_kept + F_left) Uᵀ and V Y_sᵀ, checked exact; and the leftover's share of entries."""
    out = {}
    for name, c in calls.items():
        if name in BINX or c.t == 1:
            continue
        dens, left, groups, exact = [], [], [], True
        for s in range(c.S):
            bin_, F, Y = c.split(s)
            lo = wrap8(F)
            hi = (F - lo) >> 8
            kept, rest = two_four(hi)
            g = (kept.reshape(kept.shape[0], -1, 4) != 0).sum(-1)
            assert g.max() <= 2
            got = (bin_ + lo @ c.U.T + 256 * (kept @ c.U.T + rest @ c.U.T) + c.V @ Y.T) & D.M16
            exact &= bool(np.array_equal(got, c.w[s] & D.M16))
            dens.append(float((hi != 0).mean()))
            left.append(float((rest != 0).mean()))
            groups.append(float(((hi.reshape(hi.shape[0], -1, 4) != 0).sum(-1) > 2).mean()))
        out[name] = {"t": c.t, "hi_range": [int(((c.split(0)[1] - wrap8(c.split(0)[1])) >> 8).min()),
                                            int(((c.split(0)[1] - wrap8(c.split(0)[1])) >> 8).max())],
                     "hi_density": float(np.mean(dens)), "groups_over_2of4": float(np.mean(groups)),
                     "left_density": float(np.mean(left)), "exact_mod_2^16": exact}
    return out


# -- the price ------------------------------------------------------------------------------------------------------------------------

def direct_exact(n: int, strassen_ok: bool = True) -> float:
    """The direct product X W′ᵀ mod 2^n: b1 planes or s8, and Strassen at depth 1 wherever it needs no correction
    (n ≤ 8). At n = 9 Winograd's eight mod-2 corrections are b1 products at 1/64 a form-side: 7/8 + 1/8 = 1."""
    base = lo_cost(n)
    return STRASSEN * base if strassen_ok and n <= 8 else base


def sep_exact(n: int, t: int, strassen_ok: bool = True, sparse: bool = False) -> float:
    """The separated route mod 2^n: b1 d_A d_Bᵀ, F's low limb, F's high limb by planes (or, with `sparse`, as a 2:4
    sparse s8 operand at 1/2, its leftover on the alu: E7), V Y_sᵀ by planes or s8; Strassen at depth 1 on its
    products while n ≤ 8 (tensor time only: in-model its output adds cost more than it saves at inner width L′)."""
    sig = STRASSEN if (strassen_ok and n <= 8) else 1.0
    lo, y = lo_cost(n), min(1.0, pairs(n, y_bits(t), 8) * B1)
    hi = min(1.0, pairs(n - 8, hi_bits(t), 8) * B1) if n > 8 else 0.0
    if sparse and n > 8:
        hi = min(hi, 0.5)
    return B1 + sig * (lo + hi + y) / t


#: a SWAR wrap-add of packed fields: 5 lane-ops a 32-bit lane (two masks, the add, the top bits' xor and its merge);
#: a 32-bit add is one. An estimate, as is the square size the in-model Strassen adds are priced at
SWAR_OPS = 5
N_REF = 2048


def strassen_adds(n: int, depth: int, w_per_call: bool = True, N: int = N_REF) -> float:
    """Winograd's additions on the alu, share of G at an N-cube: per level, 4 X-side forms (and 4 W′-side, when W′ is
    drawn per call) of (N/2)² int8 packed 4 a lane, and 7 output adds of (N/2)² n-bit words (4, 2 or 1 a lane); level
    ℓ has 7^(ℓ−1) subproblems of side N/2^(ℓ−1). An alu lane-op is 64 s8 MACs of time."""
    fields = 4 if n <= 8 else 2 if n <= 16 else 1
    forms = (8 if w_per_call else 4) / 4 / 4 * SWAR_OPS
    outs = 7 / 4 * (SWAR_OPS / fields if fields > 1 else 1)
    return sum((forms + outs) * 64 / N * (7 / 4) ** lvl for lvl in range(depth))


def floor_low(n: int, t: int, binx: bool = False, left: float | None = None, rhos: list | None = None,
              w_per_call: bool = True) -> dict:
    """The exact floor of a word's low n bits, two ways. *Tensor*: tensor-pipe time, Strassen at depth 1. *In-model*:
    the max over the tensor and alu pipes, Strassen at depth 1 or 2 with its additions and sparse corrections on the
    alu (`direct_inmodel`), and F's high limb on a 2:4 sparse MMA when E7 measured its leftover (`left`)."""
    d0, d1 = direct_exact(n, False), direct_exact(n)
    if binx:
        d0 = min(1.0, pairs(n, 1, 8) * B1)
        d1 = min(d0, STRASSEN if n <= 8 else 1.0)
    s0 = sep_exact(n, t, False) if t > 1 else float("inf")
    s1 = sep_exact(n, t) if t > 1 else float("inf")
    has = t > 1 and left is not None
    sp_t = sep_exact(n, t, False, sparse=True) if has else float("inf")
    sp_alu = 64 * left / t if (has and n > 8) else 0.0
    sp = sp_t if 64 * (left or 0.0) <= 0.5 else float("inf")
    row = {"n": n, "direct": d0, "direct_strassen": d1, "sep": s0, "sep_strassen": s1, "sep_sparse": sp,
           "floor_no_strassen": min(d0, s0), "floor": min(d0, d1, s0, s1, sp)}
    if rhos is not None:
        im = {d: direct_inmodel(n, rhos, d, w_per_call, base=d0) for d in (1, 2)}
        row["inmodel"] = im
        row["sep_sparse_inmodel"] = max(sp_t, sp_alu)
        row["floor_inmodel"] = min(d0, s0, im[1]["time"], im[2]["time"], row["sep_sparse_inmodel"])
    return row


def lane_per_side(n: int, rho: float) -> float:
    """In-model alu time (share of G) of one form side's sparse corrections mod 2^(n−8): for each wrapped entry, add
    ±(the other form's low n − 8 bits) into n/2 outputs, packed: 32 parities a lop3 at n = 9, else ⌊32/(n−8)⌋ fields
    a 3-op SWAR add. An alu lane-op is 64 s8 MACs of time (4,096 MACs vs 64 lanes an SM-clock); forms' own adds and the
    gathers are not counted (an estimate, and the model's free indirect reads)."""
    if n <= 8:
        return 0.0
    w = n - 8
    ops = 1 / 32 if w == 1 else 3 / (32 // w)
    return rho / 8 * ops * 64


def direct_inmodel(n: int, rhos: list, depth: int = 1, w_per_call: bool = True, base: float | None = None) -> dict:
    """Strassen at `depth` on the s8 product, time the max over the tensor and alu pipes: the alu holds Winograd's
    additions (`strassen_adds`) and, at depth 1 and n > 8, classic Strassen's sparse lane corrections (`rhos`, E3's
    form-side wrap densities). Depth 2 is priced only where it needs no correction (n ≤ 8); its second level's forms
    wrap on about a quarter of entries, so its corrections cost more than the level saves."""
    base = lo_cost(n) if base is None else base
    if depth > 1 and n > 8:
        return {"tensor": None, "alu": None, "time": float("inf")}
    tensor = STRASSEN ** depth
    alu = strassen_adds(n, depth, w_per_call) + (sum(lane_per_side(n, r) for r in rhos) if depth == 1 else 0.0)
    return {"tensor": tensor, "alu": alu, "time": min(base, max(tensor, alu))}


#: the 8-bit window each design's note recommends, [r, r + 8)
RECOMMENDED = {"seg-t8": (4, 12), "seg-t3": (4, 12), "seg-t3-pseg": (8, 16), "cell-fresh": (8, 16), "cell-jr2": (6, 14),
               "cell-2gemm": (4, 12)}

#: Llama-3.2-1B's linear layers a token, (K, N): qkv, o, gate+up, down; 16 layers (lm_head and attention left out)
LLAMA_1B = [(2048, 3072), (2048, 2048), (2048, 16384), (8192, 2048)]
LAYERS = 16


def ands_per_mac(n: int) -> int:
    """An s8 × s8 MAC mod 2^n in a free-XOR circuit (an estimate): P(n) partial-product ANDs, one AND a full or half
    adder compressing them to min(n, 16) bits, and n − 1 for the accumulator."""
    P = pairs(n)
    return P + (P - min(n, 16)) + (n - 1)


def word_bytes(res: dict) -> dict:
    """PoUW word bytes a token on Llama-3.2-1B's linear layers, per design and word format (an estimate: dense packing
    of b-bit words; O and the values' commitment not included)."""
    out = {}
    for name, (_, lp, L) in DESIGNS.items():
        nstar = res["price"][name]["n_star"]
        top = RECOMMENDED[name][0] + 8
        nim = res["price"][name]["n_star_inmodel"]
        if name.startswith("cell"):
            fmts = {"u32 sum": (32, "S", 32), f"low {nstar} bits": (nstar, "S", nstar),
                    f"low {nim} bits (in-model n*)": (nim, "S", nim),
                    f"8-bit window {RECOMMENDED[name]}": (8, "S", top)}
        else:
            fmts = {"u16 prefix (today)": (16, "S-1", 16), f"low {nstar} bits, prefix": (nstar, "S-1", nstar),
                    f"low {nim} bits, prefix (in-model n*)": (nim, "S-1", nim),
                    f"8-bit window {RECOMMENDED[name]} of each segment": (8, "S", top)}
        row = {}
        for fmt, (bits, w, width) in fmts.items():
            tot = 0
            for K, N in LLAMA_1B:
                S = 1 if name.startswith("cell") else -(-K // L)
                words = S - 1 if w == "S-1" else S
                tot += N * words * bits / 8
            row[fmt] = {"bytes_per_token": tot * LAYERS, "ands_per_mac": ands_per_mac(width)}
        out[name] = row
    return out


def price(res: dict) -> dict:
    tabs = {}
    for name, (K, lp, L) in DESIGNS.items():
        t = L // lp
        bits = word_bits(name)
        left = res["e7"][name]["left_density"] if name in res.get("e7", {}) else None
        rhos = [x["density"] for x in res["e3"][SIDE[name]]["strassen"]["form_sides"]]
        rows = [floor_low(n, t, name in BINX, left, rhos, name in W_PER_CALL) for n in range(1, bits + 1)]
        full = rows[-1]["floor"]
        full0 = rows[-1]["floor_no_strassen"]
        full_im = rows[-1]["floor_inmodel"]
        nstar = min(r["n"] for r in rows if r["floor"] >= full - 1e-12)
        nstar0 = min(r["n"] for r in rows if r["floor_no_strassen"] >= full0 - 1e-12)
        nstar_im = min(r["n"] for r in rows if r["floor_inmodel"] >= full_im - 1e-12)
        routes = res["e5"][name]["routes"]
        P = res["e6"][name]["p"]
        win = {}
        for rr, b in windows(name):
            n = rr + b
            p = P[f"{rr},{b}"]["p"]
            v_ = {"p": p, "sampling": 1 / (1 - p)}
            for tag, exact, ref in (("", rows[n - 1]["floor"], full), ("_inmodel", rows[n - 1]["floor_inmodel"], full_im)):
                costs = model_costs(n, t, left, bool(tag))
                beat = []
                for k, v in routes.items():
                    delta = ref - costs[k]
                    if delta <= 1e-12:
                        continue
                    pe = v["p_err"][f"{rr},{b}"]
                    need = delta / ref * (1 - p)
                    if pe < need:
                        beat.append((costs[k], k, pe, need))
                eff = min([exact] + [x[0] for x in beat])
                v_.update({f"exact{tag}": exact, f"beaten_by{tag}": sorted(beat)[:1], f"floor{tag}": eff,
                           f"holds{tag}": eff >= ref - 1e-12})
            win[f"{rr},{b}"] = v_
        by_b = {}
        for b in range(1, 9):
            row = {"low_bits_floor": rows[b - 1]["floor"], "low_bits_floor_inmodel": rows[b - 1]["floor_inmodel"]}
            for tag in ("", "_inmodel"):
                ok = [(win[f"{rr},{b}"]["sampling"], rr) for rr, bb in windows(name)
                      if bb == b and win[f"{rr},{b}"][f"holds{tag}"]]
                best = min(ok) if ok else None
                row.update({f"windows_that_hold{tag}": sorted(rr for _, rr in ok),
                            f"best_r{tag}": best[1] if best else None, f"sampling{tag}": best[0] if best else None})
            by_b[b] = row
        bmin = min([b for b in by_b if by_b[b]["best_r"] is not None], default=None)
        bmin_im = min([b for b in by_b if by_b[b]["best_r_inmodel"] is not None], default=None)
        tabs[name] = {"t": t, "K": K, "L'": lp, "L": L, "word_bits": bits, "rows": rows, "full_floor": full,
                      "full_floor_inmodel": full_im, "n_star_inmodel": nstar_im, "b_min_inmodel": bmin_im,
                      "sparse_left": left, "full_floor_no_strassen": full0, "n_star": nstar,
                      "n_star_no_strassen": nstar0, "b_min": bmin, "by_b": by_b, "windows": win}
    return tabs


def price_at(res: dict, b1: float) -> dict:
    """`price` with b1 at `b1` of an s8 MAC; the experiments don't depend on it."""
    global B1
    old, B1 = B1, b1
    try:
        return price(res)
    finally:
        B1 = old


def price_summary(tabs: dict) -> dict:
    out = {}
    for name, tb in tabs.items():
        r, _ = RECOMMENDED[name]
        rec = tb["windows"][f"{r},8"]
        out[name] = {**{k: tb[k] for k in ("full_floor", "full_floor_inmodel", "n_star", "n_star_inmodel", "b_min",
                                           "b_min_inmodel")},
                     "low_bits_floor": [x["floor"] for x in tb["rows"]],
                     "low_bits_floor_inmodel": [x["floor_inmodel"] for x in tb["rows"]],
                     "windows_that_hold": {b: v["windows_that_hold"] for b, v in tb["by_b"].items()},
                     "windows_that_hold_inmodel": {b: v["windows_that_hold_inmodel"] for b, v in tb["by_b"].items()},
                     "recommended_window": [r, r + 8], "recommended_holds": [rec["holds"], rec["holds_inmodel"]]}
    return out


def tables_md(res: dict) -> str:
    out = []

    def f(x):
        return "—" if x is None or x == float("inf") else f"{x:.4f}"

    for name, tb in res["price"].items():
        lp = tb["L'"]
        out.append(f"### {name}: t {tb['t']}, K {tb['K']}, L′ {lp}, L {tb['L']}, word {tb['word_bits']} bits\n")
        out.append(f"full-word floor {tb['full_floor']:.4f} G tensor ({tb['full_floor_no_strassen']:.4f} without "
                   f"Strassen), {tb['full_floor_inmodel']:.4f} G in-model; low bits reach it at n* = {tb['n_star']} "
                   f"tensor ({tb['n_star_no_strassen']} without Strassen), {tb['n_star_inmodel']} in-model; fewest bits "
                   f"in a window that holds it: {tb['b_min']} tensor, {tb['b_min_inmodel']} in-model"
                   + (f"; 2:4 leftover density {tb['sparse_left']:.5f}" if tb["sparse_left"] is not None else "") + "\n")
        out.append("| n low bits | direct | direct, Strassen d=1 | separated | separated, Strassen d=1 | separated, "
                   "2:4 hi | floor, tensor | in-model d=1 (tensor, alu) | in-model d=2 | floor, in-model |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in tb["rows"][:16] + tb["rows"][16:][-1:]:
            i1, i2 = r["inmodel"][1], r["inmodel"][2]
            out.append(f"| {r['n']} | {f(r['direct'])} | {f(r['direct_strassen'])} | {f(r['sep'])} | "
                       f"{f(r['sep_strassen'])} | {f(r['sep_sparse'])} | {f(r['floor'])} | {f(i1['time'])} "
                       f"({f(i1['tensor'])}, {f(i1['alu'])}) | {f(i2['time'])} | {f(r['floor_inmodel'])} |")
        out.append("")
        out.append("| b | floor, low b bits (tensor / in-model) | windows [r, r+b) that hold, tensor (r) | in-model (r) | "
                   "best r, in-model | 1/(1−p) |")
        out.append("|---|---|---|---|---|---|")
        for b, v in tb["by_b"].items():
            out.append(f"| {b} | {f(v['low_bits_floor'])} / {f(v['low_bits_floor_inmodel'])} | "
                       f"{v['windows_that_hold'] or '—'} | {v['windows_that_hold_inmodel'] or '—'} | "
                       f"{v['best_r_inmodel']} | {f(v['sampling_inmodel'])} |")
        out.append("")
        sd = res.get("second_draw", {}).get(name)
        if sd:
            out.append(f"second coin draw: b_min {sd['b_min']} / {sd['b_min_inmodel']} in-model, n* {sd['n_star']}, "
                       f"verdicts that differ: {sd['verdicts_that_differ'] or 'none'}\n")
        out.append("| approximate route | cost / G at n = 16, tensor | in-model | err std | p_err, b = 8 window at r = 4, "
                   "8, 10, 12, 14 (b = 2) |")
        out.append("|---|---|---|---|---|")
        c0 = model_costs(16, tb["t"], tb["sparse_left"], False)
        c1 = model_costs(16, tb["t"], tb["sparse_left"], True)
        for k, v in res["e5"][name]["routes"].items():
            ps = ", ".join(f"{v['p_err'][key]:.3g}" for key in ("4,8", "8,8", "10,6", "12,4", "14,2"))
            out.append(f"| {k} | {c0[k]:.4f} | {c1[k]:.4f} | {v['err_std']:.1f} | {ps} |")
        out.append("")
    if "price_wgmma" in res:
        out.append(f"### b1 at {B1_WGMMA} of an s8 MAC (wgmma), tensor / in-model\n")
        out.append("| design | full-word floor | n* | fewest bits in a window that holds | windows that hold, b = 8 (r) | "
                   "recommended [r, r+8) holds |")
        out.append("|---|---|---|---|---|---|")
        for name, v in res["price_wgmma"].items():
            out.append(f"| {name} | {v['full_floor']:.4f} / {v['full_floor_inmodel']:.4f} | {v['n_star']} / "
                       f"{v['n_star_inmodel']} | {v['b_min']} / {v['b_min_inmodel']} | "
                       f"{v['windows_that_hold'][8] or '—'} / {v['windows_that_hold_inmodel'][8] or '—'} | "
                       f"{v['recommended_window']}: {v['recommended_holds'][0]} / {v['recommended_holds'][1]} |")
        out.append("")
    out.append("### PoUW word bytes a token, Llama-3.2-1B linear layers (estimate)\n")
    out.append("| design | format | bytes a token | ANDs a MAC in-circuit (arithmetic only) |")
    out.append("|---|---|---|---|")
    for name, row in res.get("bytes", {}).items():
        for fmt, v in row.items():
            out.append(f"| {name} | {fmt} | {v['bytes_per_token']:,.0f} | {v['ands_per_mac']} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261028)
    t0 = time.perf_counter()
    calls = {k: Call(*v, kind=KIND.get(k, "dither")) for k, v in DESIGNS.items()}
    res = {"designs": DESIGNS, "coin": COIN.hex(), "receipt": RECEIPT.hex()}
    res["e1"] = e1(rng)
    res["e2"] = e2()
    res["e3"] = e3(calls, rng)
    res["e4"] = e4(calls)
    res["e5"] = e5(calls)
    res["e6"] = e6(calls)
    res["e7"] = e7(calls)
    res["price"] = price(res)
    res["price_wgmma"] = price_summary(price_at(res, B1_WGMMA))
    res["bytes"] = word_bytes(res)
    alt = {k: Call(*v, call=1, kind=KIND.get(k, "dither")) for k, v in DESIGNS.items()}
    res2 = {"e3": res["e3"], "e5": e5(alt), "e6": e6(alt), "e7": e7(alt)}
    p2 = price(res2)
    res["second_draw"] = {name: {
        "b_min": p2[name]["b_min"], "b_min_inmodel": p2[name]["b_min_inmodel"], "n_star": p2[name]["n_star"],
        "n_star_inmodel": p2[name]["n_star_inmodel"], "e7": res2["e7"].get(name),
        "windows_that_hold": {b: v["windows_that_hold"] for b, v in p2[name]["by_b"].items()},
        "windows_that_hold_inmodel": {b: v["windows_that_hold_inmodel"] for b, v in p2[name]["by_b"].items()},
        "verdicts_that_differ": sorted(k for k, v in p2[name]["windows"].items()
                                       if (v["holds"], v["holds_inmodel"]) != (
                                           res["price"][name]["windows"][k]["holds"],
                                           res["price"][name]["windows"][k]["holds_inmodel"]))} for name in p2}
    res["seconds"] = time.perf_counter() - t0
    (a.out / "out.json").write_text(json.dumps(res, indent=1, default=str))
    (a.out / "tables.md").write_text(tables_md(res))
    print(tables_md(res))
    print(json.dumps({k: res[k] for k in ("e1", "e2")}, default=str)[:3000])


if __name__ == "__main__":
    main()
