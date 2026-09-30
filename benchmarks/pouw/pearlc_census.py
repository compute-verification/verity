#!/usr/bin/env python3
"""TT_OUT's falsification census for Pearl-C's ticket chain on the H100 (CPU, bit-exact).

Pearl-C binds output words only (the pre-peel accumulator C~ and the useful output U), so the work an adversary can skip in
the ticket chain is, per wgmma tile: an atom that leaves every accumulator of the tile unchanged (a tile identity, e.g. after
freezing), a run of atoms exact across the whole tile long enough for int8 plus Strassen to pay (14 or more atoms: one
Winograd level on the H100 breaks even at 450-700 of inner dimension), and a promotion add that leaves the total unchanged.
The census counts them over input families built to trigger each, under Pearl-shaped noise, for a pure wgmma chain and for
promotion from +0 every 4 atoms (G4). A cell is in Pi's domain when G4 is used, or when the chain is pure and k/delta^2 <= 2^15.
The census passes when every in-domain cell's skippable share is at most 1/400.

The atom is verity.ml.tc's HOPPER_E4M3_K32 with a nonzero accumulator (one group of 32, the accumulator truncated to 14 bits
and joined to the group), vectorized here and checked against verity.ml.tc by --selftest. The noise is Pearl's shape,
simplified: per row, y = x + delta*rms(x)*Z with Z a rank-32 row of unit RMS (lines of entries +-(1..128)); alpha = 448/max|y|;
codes = RNE_E4M3(alpha*y). Seeds are the tuple (run, unit index, side), so duplicated units get independent noise.

    python benchmarks/pouw/pearlc_census.py --selftest
    python benchmarks/pouw/pearlc_census.py --ks 8192,16384,65536 --deltas 1 --tile 64x16 --out census.json
    python benchmarks/pouw/pearlc_census.py --forming s5 --line-norm 16 --ks 8192,65536 --out census_s5.json

--forming s5 is docs/pouw/cheap-binding.md §5's forming, bit-exact: s = max|x|, rho = RMS over every 8th position,
sigma = delta*rho, alpha = 448/(s + 4*sigma); F's lines are E4M3 codes of (N_F/sqrt(32))*w and E'_row = e4m3((sigma*alpha/N_F)*v)
(w, v unit-RMS lines of entries +-(1..128)); D = HOPPER_E4M3_K32.step(C = f32(alpha*x), E'_row, F_l) per element; A' =
RNE_E4M3(D) with saturation. Pi's activation rules are checked per row: the noise floor sigma*alpha >= 1 and liveness (at
most k/64 entries with |x| >= 8*sigma); a cell whose rows break them is out of the domain ("rejected"). The collision rate
is how often two independent noise draws give an element the same code: the element-level noise entropy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_result  # noqa: E402

# ---- E4M3 codec -------------------------------------------------------------------------------------------------------
MAG = np.zeros(128, dtype=np.float64)
for _c in range(128):
    _e, _m = _c >> 3, _c & 7
    MAG[_c] = _m * 2.0 ** -9 if _e == 0 else (8 + _m) * 2.0 ** (_e - 10)
MAG[0x7F] = np.nan
FIN = MAG[:0x7F]
Q_MAX = 448.0
SIG = np.array([(c & 7) | (8 if c >> 3 else 0) for c in range(128)], dtype=np.int64)
EXP1 = np.array([max(c >> 3, 1) for c in range(128)], dtype=np.int64)
OFF, PSHIFT = -14, 17        # verity.ml.tc.term.E4M3: product_exponent_offset, product_shift


def enc(x: np.ndarray) -> np.ndarray:
    a = np.minimum(np.abs(x), Q_MAX)
    hi = np.searchsorted(FIN, a, side="left").clip(0, 126)
    lo = (hi - 1).clip(0, 126)
    dlo, dhi = a - FIN[lo], FIN[hi] - a
    pick = np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(lo % 2 == 0, lo, hi)))
    pick = np.where(a == FIN[hi], hi, pick)
    return (pick | np.where((x < 0) & (pick > 0), 0x80, 0)).astype(np.uint8)


def bf16(x: np.ndarray) -> np.ndarray:
    """Round to BF16 (RNE) and back."""
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


# ---- the atom with a nonzero accumulator ---------------------------------------------------------------------------------

def _fp32_parts(C: np.ndarray):
    a = np.abs(C)
    _, e = np.frexp(a)
    ex = (e - 1).astype(np.int64)
    sub = a < 2.0 ** -126
    ex = np.where(sub, -126, ex)
    mag = np.where(sub, np.ldexp(a, 149), np.ldexp(a, (23 - ex).astype(np.int64)))
    return C < 0, ex, np.where(a > 0, np.rint(mag), 0).astype(np.int64)


def step(C: np.ndarray, a: np.ndarray, b: np.ndarray):
    """C: (P,) exact FP32 values; a, b: (P, 32) codes. Returns (D, exact): exact = no bit lost anywhere in the step."""
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    ma, mb = a & 0x7F, b & 0x7F
    ss = SIG[ma] * SIG[mb]
    pexp = EXP1[ma] + EXP1[mb] + OFF
    pneg = ((a ^ b) & 0x80) != 0
    cneg, cexp, cmag = _fp32_parts(C)
    mx = np.maximum(np.maximum(np.where(ss > 0, pexp, -10 ** 6).max(-1), np.where(cmag > 0, cexp, -10 ** 6)), -139)
    psh = np.clip(mx[:, None] - pexp, 0, 62)
    csh = np.clip(mx - cexp, 0, 62)
    p10 = (ss << PSHIFT) >> 10
    ps = np.where(ss > 0, p10 >> psh, 0)
    c10 = cmag >> 10
    cs = np.where(cmag > 0, c10 >> csh, 0)
    lost = ((cmag & 0x3FF) != 0) | ((c10 & ((np.int64(1) << csh) - 1)) != 0)
    lost |= ((ss > 0) & ((p10 & ((np.int64(1) << psh) - 1)) != 0)).any(-1)
    tot = np.where(pneg, -ps, ps).sum(-1) + np.where(cneg, -cs, cs)
    neg, m = tot < 0, np.abs(tot)
    bl = np.frexp(m.astype(np.float64))[1].astype(np.int64)
    ex = mx + bl - 14
    over = np.clip(bl - 14, 0, 62)
    lost |= (bl > 14) & ((m & ((np.int64(1) << over) - 1)) != 0)
    m = np.where(bl > 14, m >> over, m << np.clip(14 - bl, 0, 62))
    sub = ex < -126
    subsh = np.clip(-126 - ex, 0, 62)
    lost |= sub & ((m & ((np.int64(1) << subsh) - 1)) != 0)
    m = np.where(sub, m >> subsh, m)
    ex = np.where(sub, -126, ex)
    v = np.ldexp((m << 10).astype(np.float64), (ex - 23).astype(np.int64))
    return np.where(m == 0, 0.0, np.where(neg, -v, v)), ~lost


def selftest(cases: int = 3000) -> int:
    from verity.ml.tc.models import HOPPER_E4M3_K32 as H
    rng = np.random.default_rng(7)
    a = (rng.integers(0, 0x7F, (cases, 32)) | rng.integers(0, 2, (cases, 32)) * 0x80).astype(np.uint8)
    b = (rng.integers(0, 0x7F, (cases, 32)) | rng.integers(0, 2, (cases, 32)) * 0x80).astype(np.uint8)
    a[: cases // 3] &= 0x87
    C = np.where(rng.random(cases) < 0.2, 0.0, rng.standard_normal(cases) * np.exp2(rng.integers(-20, 30, cases)))
    C = C.astype(np.float32).astype(np.float64)
    D, _ = step(C, a, b)
    bad = 0
    for i in range(cases):
        want = H.step(int(np.float32(C[i]).view(np.uint32)), [int(x) for x in a[i]], [int(x) for x in b[i]])
        got = int(np.float32(D[i]).view(np.uint32))
        bad += want != got and not (want & 0x7FFFFFFF == 0 and got & 0x7FFFFFFF == 0)
    return bad


# ---- inputs -------------------------------------------------------------------------------------------------------------

def _rng(*parts: object) -> np.random.Generator:
    h = hashlib.sha256("|".join(["verity/pouw/pearl-c/census/v1", *map(str, parts)]).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def lines(n: int, rng: np.random.Generator) -> np.ndarray:
    return (rng.integers(1, 129, (n, 32)) * rng.choice([-1, 1], (n, 32))).astype(np.float64)


def form(X: np.ndarray, F: np.ndarray, delta: float, rng: np.random.Generator):
    """Pearl-shaped forming of rows X against noise lines F (k x 32). Returns codes, the noise-free mask, and the rate at which
    rounding through BF16 first gives a different code than one rounding."""
    Z = lines(X.shape[0], rng) @ F.T
    Z /= np.sqrt((Z ** 2).mean(1, keepdims=True))
    sig = delta * np.sqrt((X ** 2).mean(1, keepdims=True))
    Y = X + bf16(sig * Z)
    alpha = Q_MAX / np.abs(Y).max(1, keepdims=True)
    codes = enc(alpha * Y)
    double = enc(bf16(alpha * Y))
    nf = enc(alpha * (X - 4 * sig)) == enc(alpha * (X + 4 * sig))
    return codes, nf, float((codes != double).mean())


def unit_lines(n: int, rng: np.random.Generator) -> np.ndarray:
    v = lines(n, rng)
    return v / np.sqrt((v ** 2).mean(1, keepdims=True))


def form_s5(X: np.ndarray, Fc: np.ndarray, delta: float, line_norm: float, rng: np.random.Generator, chunk: int = 1 << 17):
    """§5's forming of rows X against F's codes Fc (k x 32). Returns codes, a second draw's codes (for the collision rate),
    the share of E' codes that are zero or subnormal, and the rows' domain verdicts (noise floor, liveness)."""
    rows, k = X.shape
    s = np.abs(X).max(1)
    rho = np.sqrt((X[:, ::8] ** 2).mean(1))
    sig = delta * rho
    alpha = np.where(s + 4 * sig > 0, Q_MAX / np.maximum(s + 4 * sig, 1e-300), 0.0)
    floor_ok = sig * alpha >= 1.0
    live_ok = (np.abs(X) >= 8 * sig[:, None]).sum(1) * 64 <= k
    C = (alpha[:, None] * X).astype(np.float32).astype(np.float64)
    out, sub = [], []
    for _draw in range(2):
        Ec = enc(unit_lines(rows, rng) * (sig * alpha / line_norm)[:, None])
        sub.append(float(((Ec & 0x7F) < 8).mean()))
        codes = np.empty((rows, k), dtype=np.uint8)
        idx = np.arange(rows * k)
        for lo in range(0, rows * k, chunk):
            ii = idx[lo:lo + chunk]
            r_, l_ = ii // k, ii % k
            D, _ = step(C[r_, l_], Ec[r_], Fc[l_])
            codes[r_, l_] = enc(D)
        out.append(codes)
    return out[0], float((out[0] == out[1]).mean()), sub[0], bool(floor_ok.all()), bool(live_ok.all())


FAMILIES = ("gaussian", "spike1-first", "spikes-first", "spikes-permuted", "spikes-groupstart", "constant",
            "coherent-gaussian", "orthogonal-FA", "duplicated")
ACTIVATION_FAMILIES = ("zero-row", "zero-slices", "sparse10", "rank1", "saturated", "grid-aligned", "in-span-FA")
FREEZE_FAMILIES = ("aligned-spikes-r64", "aligned-spikes-r160", "aligned-spikes-r220", "aligned-spikes-r300")


def family(name: str, rows: int, cols: int, k: int, FA: np.ndarray, rng: np.random.Generator):
    L = k // 64 - 1
    if name == "gaussian":
        return rng.standard_normal((rows, k)), rng.standard_t(4, (cols, k))
    if name == "orthogonal-FA":
        A = rng.standard_normal((rows, k))
        Q, _ = np.linalg.qr(FA)
        return A - (A @ Q) @ Q.T, rng.standard_t(4, (cols, k))
    if name == "zero-row":
        return np.zeros((rows, k)), rng.standard_t(4, (cols, k))
    if name == "zero-slices":
        A = rng.standard_normal((rows, k))
        A.reshape(rows, k // 32, 32)[:, ::2, :] = 0.0
        return A, rng.standard_t(4, (cols, k))
    if name == "sparse10":
        return rng.standard_normal((rows, k)) * (rng.random((rows, k)) < 0.1), rng.standard_t(4, (cols, k))
    if name == "rank1":
        v = rng.standard_normal(k)
        return np.outer(rng.standard_normal(rows), v), np.outer(rng.standard_normal(cols), rng.standard_t(4, k))
    if name == "saturated":
        A = rng.standard_normal((rows, k)) * 0.1
        A[:, : k // 2] = np.sign(rng.standard_normal((rows, k // 2)))
        return A, rng.standard_t(4, (cols, k))
    if name == "grid-aligned":
        return val(rng.integers(0x30, 0x60, (rows, k)) | (rng.integers(0, 2, (rows, k)) * 0x80)), rng.standard_t(4, (cols, k))
    if name == "in-span-FA":
        return rng.standard_normal((rows, 32)) @ FA.T, rng.standard_t(4, (cols, k))
    if name == "duplicated":
        a, b = rng.standard_normal((1, k)), rng.standard_t(4, (1, k))
        return np.repeat(a, rows, 0), np.repeat(b, cols, 0)
    if name in ("constant", "coherent-gaussian"):
        v = rng.choice([-1.0, 1.0], k) if name == "constant" else rng.standard_normal(k)
        return np.tile(v, (rows, 1)), np.tile(v, (cols, 1))
    if name.startswith("aligned-spikes-r"):
        # Two spikes per 128-deep group, at the same positions in every row of A and B, off the every-8th grid so that rho is
        # the Gaussian background's: s/rho = R, which the noise floor allows up to 444 and liveness allows at k/64 entries.
        R = float(name.removeprefix("aligned-spikes-r"))
        pos = np.concatenate([np.arange(0, k, 128) + 1, np.arange(0, k, 128) + 9])
        A, B = rng.standard_normal((rows, k)), rng.standard_normal((cols, k))
        A[:, pos] = R
        B[:, pos] = R
        return A, B
    pos = {"spike1-first": np.arange(1), "spikes-first": np.arange(L),
           "spikes-permuted": rng.permutation(k)[:L],
           "spikes-groupstart": np.arange(0, k, 128)[:L]}[name]
    A, B = np.zeros((rows, k)), np.zeros((cols, k))
    A[:, pos] = 1.0
    B[:, pos] = 1.0
    return A, B


def chain(a: np.ndarray, b: np.ndarray, G: int):
    """a, b: (P, k) codes of the P outputs of one tile. Returns per-atom tile identity, per-atom tile exactness, word identity
    share, and promotion identities (G > 0)."""
    P, k = a.shape
    T = k // 32
    C = np.zeros(P)
    total = np.zeros(P, dtype=np.float32)
    tid, tex = np.zeros(T, bool), np.zeros(T, bool)
    words, promo = 0, 0
    for t in range(T):
        if G and t % G == 0:
            C = np.zeros(P)
        D, ex = step(C, a[:, 32 * t:32 * t + 32], b[:, 32 * t:32 * t + 32])
        same = D == C
        tid[t], tex[t], words = same.all(), ex.all(), words + int(same.sum())
        C = D
        if G and (t % G == G - 1 or t == T - 1):
            new = (total + C.astype(np.float32)).astype(np.float32)
            promo += int(((new == total) & (C != 0)).all())
            total = new
    return tid, tex, words / (P * T), promo


def exact_run_mask(tex: np.ndarray, min_run: int = 14) -> np.ndarray:
    mask, start = np.zeros(len(tex), bool), None
    for i, e in enumerate(list(tex) + [False]):
        if e and start is None:
            start = i
        elif not e and start is not None:
            if i - start >= min_run:
                mask[start:i] = True
            start = None
    return mask


def exact_run_atoms(tex: np.ndarray, min_run: int = 14) -> int:
    return int(exact_run_mask(tex, min_run).sum())


def in_domain(G: int, k: int, delta: float) -> bool:
    return G == 4 or k / delta ** 2 <= 2 ** 15


def val(codes: np.ndarray) -> np.ndarray:
    codes = codes.astype(np.int64)
    v = MAG[codes & 0x7F]
    return np.where(codes & 0x80, -v, v)


def cell(fam: str, k: int, delta: float, rows: int, cols: int, run: int, forming: str = "census",
         line_norm: float = 256.0) -> list[dict]:
    rng = _rng(run, fam, k, delta)
    FA, FB = lines(k, _rng(run, "FA", k)), lines(k, _rng(run, "FB", k))
    XA, XB = family(fam, rows, cols, k, FA, rng)
    extra = {"forming": forming}
    if forming == "s5":
        FcA = enc(unit_lines(k, _rng(run, "FA-lines", k)) * (line_norm / np.sqrt(32)))
        FcB = enc(unit_lines(k, _rng(run, "FB-lines", k)) * (line_norm / np.sqrt(32)))
        a, collA, subA, floorA, liveA = form_s5(XA, FcA, delta, line_norm, _rng(run, fam, k, delta, "unit-A"))
        b, collB, subB, floorB, liveB = form_s5(XB, FcB, delta, line_norm, _rng(run, fam, k, delta, "unit-B"))
        nfA, drA, drB = np.zeros(1), 0.0, 0.0
        ok = floorA and liveA and floorB and liveB
        extra.update({"line_norm": line_norm, "collision_A": round(collA, 6), "collision_B": round(collB, 6),
                      "e_prime_subnormal_A": round(subA, 6), "rules_ok": ok,
                      "rejected_by": [n for n, v in (("noise-floor-A", floorA), ("liveness-A", liveA),
                                                     ("noise-floor-B", floorB), ("liveness-B", liveB)) if not v]})
    else:
        a, nfA, drA = form(XA, FA, delta, _rng(run, fam, k, delta, "unit-A"))
        b, nfB, drB = form(XB, FB, delta, _rng(run, fam, k, delta, "unit-B"))
        ok = True
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    A, B = a[ii.ravel()], b[jj.ravel()]
    out = []
    for G in (0, 4):
        tid, tex, word_id, promo = chain(A, B, G)
        T = k // 32
        skip = int((tid | exact_run_mask(tex)).sum()) + promo
        out.append({**extra, "family": fam, "k": k, "delta": delta, "layout": "G4" if G else "pure",
                    "in_domain": in_domain(G, k, delta) and ok,
                    "atoms": T, "tile_identity_atoms": int(tid.sum()), "exact_run_atoms": exact_run_atoms(tex),
                    "tile_exact_atoms": int(tex.sum()), "promotion_identities": promo, "word_identity": round(word_id, 6),
                    "noise_free_entries": float(nfA.mean()), "double_rounding_differs": round((drA + drB) / 2, 6),
                    "skippable_share": skip / T})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ks", default="8192,16384,65536")
    ap.add_argument("--deltas", default="1")
    ap.add_argument("--tile", default="64x16")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--forming", choices=("census", "s5"), default="census")
    ap.add_argument("--line-norm", type=float, default=256.0)
    ap.add_argument("--out", default="pearlc_census.json")
    args = ap.parse_args()
    bad = selftest()
    print(json.dumps({"selftest_cases": 3000, "mismatches": bad}), flush=True)
    if args.selftest or bad:
        sys.exit(1 if bad else 0)
    rows, cols = map(int, args.tile.split("x"))
    t0, rows_out = time.time(), []
    for delta in map(float, args.deltas.split(",")):
        for k in map(int, args.ks.split(",")):
            for fam in args.families.split(","):
                for r in cell(fam, k, delta, rows, cols, args.run, args.forming, args.line_norm):
                    rows_out.append(r)
                    print(json.dumps(r), flush=True)
    worst = max((r["skippable_share"] for r in rows_out if r["in_domain"]), default=0.0)
    worst_row = max((r for r in rows_out if r["in_domain"]), key=lambda r: r["skippable_share"], default=None)
    doc = {"tool": "pouw_pearlc_census", "tile": args.tile, "forming": args.forming, "line_norm": args.line_norm, "selftest_mismatches": bad, "cells": rows_out,
           "worst_in_domain_skippable_share": worst, "worst_in_domain_cell": worst_row, "threshold": 1 / 400,
           "seconds": round(time.time() - t0, 1)}
    out = Path(args.out)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    ok = bad == 0 and worst <= 1 / 400
    run_result.write(out, ok, "pearlc-census", f"worst in-domain skippable share {worst:.5f} (threshold 0.0025)",
                     [("worst_in_domain_skippable_share", worst, "fraction"),
                      ("max_double_rounding_differs", max(r["double_rounding_differs"] for r in rows_out), "fraction")])
    print(json.dumps({k: v for k, v in doc.items() if k != "cells"}))


if __name__ == "__main__":
    main()
