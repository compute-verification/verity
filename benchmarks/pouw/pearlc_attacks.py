#!/usr/bin/env python3
"""Pearl-C's emulation and rewrite attacks on one card, priced in W1 units on the census's formed operands (CPU, exact).

A prover must output the checked words (C̃, U, the tickets) bit for bit, so it can replace the native FP8 atom (E4M3 x E4M3
into FP32, k = 32) only by a route that is exact wherever the atom is, i.e. that carries every operand in a cheaper format
without loss.  On s5-formed tiles of the census's families (`pearlc_census.family_rows`, `form_s5`) this measures what each
route needs and prices it at the card's rates (W1 unit = one dense E4M3 MAC with FP32 accumulate; `PRICES`, `--prices`):

    route          applies where                                              cost per useful MAC (a lower bound)
    int8 IMMA      L signed int8 limbs hold the (row, 32-slice) on one grid     E[L_A] E[L_B] p_int8 + p_fadd/32 (rescale)
    NVFP4          E2M1 limbs cover each (row, 16-block) under one block scale  E[L_A] E[L_B] p_fp4
    2:4 sparse     a 16-row x 64 fragment of one operand is 2:4 along k         p_sparse (and 2 p_sparse = p_fp8 elsewhere)
    f16-acc FP8    the atom's exact 32-product sum from +0 fits FP16            p_f16acc + p_fadd/32 (the FP32 combine)
    Strassen       both operands' sums of two E4M3 are exact in FP16            7/8 p_half + 2 p_fadd/128 (one level per
                   (the FP32-accumulate formats that hold them)                 128-product group: 8 post-adds per 4 words)
    unpromoted     the pure chain's word is the G4 ticket word (census)         saves p_fadd/128 per MAC

A route saves at most `applies * max(0, 1 - cost)` of the chain's W1; `pearlc_census.py` measures the skips (identities, the
debit, cross-group sets, exact runs, the unpromoted chain) on the same tiles.  Prices are Estimated (per-SM-per-clock public
rates) until the harness measures the card; `--prices` takes {route: units} in place of any of them.

    python benchmarks/pouw/pearlc_attacks.py --atom sm120-e4m3-k32 --ks 8192 --tile 64x64 --out attacks.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("pearlc_census", Path(__file__).resolve().parent / "pearlc_census.py")
C = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("pearlc_census", C)
_spec.loader.exec_module(C)

#: W1 prices per MAC of the honest chain (or per op), 1 = one dense E4M3 MAC with FP32 accumulate on the card.  sm120-public:
#: the RTX PRO 6000's per-SM-per-clock rates (FP8 f32-acc = INT8 = 1024 MACs, NVFP4 2048, BF16 512, 2:4-sparse FP8 twice
#: dense, so half a unit per position of a sparse fragment, 128 FP32 lanes); sm120-fp8-half: the same card if FP8 with FP32
#: accumulate ran at half the FP16-accumulate and INT8 rates, as the RTX 4090 does (X-NE3-1; there FP16 with FP32 accumulate
#: is halved too, so `half` stays 2); sm120-statement: sm120-public with the FADD at 32, the H100 price Lean's statement
#: still uses.
PRICES = {
    "sm120-public": {"fp8": 1.0, "int8": 1.0, "fp4": 0.5, "sparse": 0.5, "f16acc": 1.0, "half": 2.0, "fadd": 8.0},
    "sm120-fp8-half": {"fp8": 1.0, "int8": 0.5, "fp4": 0.25, "sparse": 0.5, "f16acc": 0.5, "half": 2.0, "fadd": 4.0},
    "sm120-statement": {"fp8": 1.0, "int8": 1.0, "fp4": 0.5, "sparse": 0.5, "f16acc": 1.0, "half": 2.0, "fadd": 32.0},
}

SIG_BITS = [int(s).bit_length() for s in C.SIG]
SIG_LOW = [(int(s) & -int(s)).bit_length() - 1 if s else 0 for s in C.SIG]
SIG_POP = [bin(int(s)).count("1") for s in C.SIG]


def _bits(codes: np.ndarray):
    """Per code: its top and lowest set-bit positions (value = SIG * 2^(EXP1 - 10)), its set-bit count, and nonzero."""
    m = codes.astype(np.int64) & 0x7F
    e = C.EXP1[m] - 10
    nz = C.SIG[m] > 0
    return np.array(SIG_BITS)[m] - 1 + e, np.array(SIG_LOW)[m] + e, np.array(SIG_POP)[m], nz


def int8_limbs(codes: np.ndarray, slice_: int = 32) -> np.ndarray:
    """Signed int8 limbs per (row, slice) for every value of the slice on one grid: ceil((span + 1) / 8), span the bits
    from the lowest set bit of any value to the top bit of any (1 for an all-zero slice)."""
    top, low, _, nz = _bits(codes)
    r, k = codes.shape
    top = np.where(nz, top, -10 ** 6).reshape(r, k // slice_, slice_).max(-1)
    low = np.where(nz, low, 10 ** 6).reshape(r, k // slice_, slice_).min(-1)
    span = np.where(top > -10 ** 5, top - low + 1, 0)
    return np.maximum(1, -(-(span + 1) // 8))


def fp4_limbs(codes: np.ndarray, block: int = 16) -> np.ndarray:
    """A lower bound on the E2M1 limbs per (row, 16-block) under one block scale each: a limb holds, per value, at most two
    adjacent set bits inside its scale's window of four positions, so a block needs at least ceil(max popcount / 2) limbs
    and at least the fewest 4-wide windows that cover every occupied bit position."""
    top, low, pop, nz = _bits(codes)
    r, k = codes.shape
    m = codes.astype(np.int64) & 0x7F
    e = C.EXP1[m] - 10
    occ = np.zeros((r, k // block, 40), bool)
    for j in range(4):
        on = nz & ((C.SIG[m] >> j) & 1).astype(bool)
        pos = (j + e + 20).reshape(r, k // block, block)
        rr, bb, _ = np.nonzero(on.reshape(r, k // block, block))
        occ[rr, bb, pos[on.reshape(r, k // block, block)]] = True
    windows = np.zeros(occ.shape[:2], np.int64)
    nxt = np.full(occ.shape[:2], -1)
    for p in range(occ.shape[2]):
        start = occ[..., p] & (p > nxt)
        windows += start
        nxt = np.where(start, p + 3, nxt)
    need = -(-pop.reshape(r, k // block, block).max(-1) // 2)
    return np.maximum(np.maximum(windows, need), 1)


def sparse24_fragments(codes: np.ndarray, rows: int = 16, kk: int = 64) -> float:
    """The share of 16-row x 64 fragments that are 2:4-sparse along k (at least two zeros in every aligned 4)."""
    r, k = codes.shape
    z = (codes & 0x7F) == 0
    ok = z.reshape(r, k // 4, 4).sum(-1) >= 2
    r16 = r // rows * rows
    frag = ok[:r16].reshape(r16 // rows, rows, k // kk, kk // 4).all(-1).all(1)
    return float(frag.mean())


def _fp16_exact(s: np.ndarray) -> np.ndarray:
    a = np.abs(s)
    e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    q = np.where(e >= -14, np.ldexp(a, (10 - e).astype(np.int64)), np.ldexp(a, 24))
    return (a == 0) | ((a <= 65504.0) & (q == np.rint(q)))


def _bf16_exact(s: np.ndarray) -> np.ndarray:
    return C.bf16(s) == s


def atom_sums_fp16(a: np.ndarray, b: np.ndarray, words: int = 256) -> float:
    """The share of (word, atom) whose exact 32-product sum from +0 is an FP16 value (float64 sums are exact here: every
    product is a multiple of 2^-18 below 2^17.7, so a sum of 32 needs at most 41 bits)."""
    va, vb = C.val(a), C.val(b)
    rng = np.random.default_rng(1)
    ii, jj = rng.integers(0, len(va), words), rng.integers(0, len(vb), words)
    T = a.shape[1] // 32
    s = (va[ii] * vb[jj]).reshape(words, T, 32).sum(-1)
    return float(_fp16_exact(s).mean())


def strassen_preadds(a: np.ndarray) -> dict:
    """The share of one Strassen level's operand pre-adds (A11 + A22 at the same k, rows i and i + r/2) exact in BF16 and
    in FP16, per element and per whole 32-slice of a row pair."""
    va = C.val(a)
    h = len(va) // 2
    s = va[:h] + va[h:2 * h]
    out = {}
    for name, f in (("bf16", _bf16_exact), ("fp16", _fp16_exact)):
        ok = f(s)
        out[f"strassen_preadd_exact_{name}"] = round(float(ok.mean()), 6)
        out[f"strassen_preadd_slice_exact_{name}"] = round(float(ok.reshape(h, -1, 32).all(-1).mean()), 6)
    return out


def price(meas: dict, p: dict) -> dict:
    """Each route's cost per useful MAC where it applies, the share it applies to, and its upper bound on the chain's
    W1 saving, max(0, 1 - cost) * applies; the chain's W1 is p_fp8 + p_fadd/128 per MAC (promotion every 128)."""
    chain = p["fp8"] + p["fadd"] / 128
    routes = {
        "int8": (meas["int8_limbs_A"] * meas["int8_limbs_B"] * p["int8"] + p["fadd"] / 32, 1.0),
        "int8-one-limb": (p["int8"] + p["fadd"] / 32, meas["int8_one_limb_both"]),
        "nvfp4": (meas["fp4_limbs_A"] * meas["fp4_limbs_B"] * p["fp4"], 1.0),
        "nvfp4-one-limb": (p["fp4"], meas["fp4_one_limb_both"]),
        "sparse-2:4": (p["sparse"], meas["sparse24_fragments_A"]),
        "f16acc": (p["f16acc"] + p["fadd"] / 32, meas["atom_sums_fp16"]),
        "strassen-1-level": (7 / 8 * p["half"] + 2 * p["fadd"] / 128,
                             meas["strassen_preadd_slice_exact_fp16"] * meas["strassen_preadd_slice_exact_fp16_B"]),
    }
    out = {name: {"cost": round(c, 4), "applies": round(s, 6),
                  "saving_ub": round(max(0.0, (p["fp8"] - c) / chain) * s, 6)} for name, (c, s) in routes.items()}
    out["unpromoted"] = {"cost": round(p["fp8"], 4), "saves_per_mac": round(p["fadd"] / 128, 4),
                         "saving_if_every_tile": round(p["fadd"] / 128 / chain, 6)}
    return out


def measure(fam: str, k: int, rows: int, cols: int, run: int, line_norm: float, delta: float = 1.0) -> dict:
    XA, XB, _, _ = C.family_rows(fam, k, delta, rows, cols, run)
    FcA, FcB = C.s5_lines(k, run, line_norm)
    a, _, _, fa, la = C.form_s5(XA, FcA, delta, line_norm, C._rng(run, fam, k, delta, "unit-A"))
    b, _, _, fb, lb = C.form_s5(XB, FcB, delta, line_norm, C._rng(run, fam, k, delta, "unit-B"))
    LA, LB = int8_limbs(a), int8_limbs(b)
    QA, QB = fp4_limbs(a), fp4_limbs(b)
    return {"family": fam, "k": k, "rules_ok": fa and la and fb and lb,
            "int8_limbs_A": round(float(LA.mean()), 4), "int8_limbs_B": round(float(LB.mean()), 4),
            "int8_one_limb_both": round(float((LA == 1).mean() * (LB == 1).mean()), 6),
            "fp4_limbs_A": round(float(QA.mean()), 4), "fp4_limbs_B": round(float(QB.mean()), 4),
            "fp4_one_limb_both": round(float((QA == 1).mean() * (QB == 1).mean()), 6),
            "sparse24_fragments_A": round(max(sparse24_fragments(a), sparse24_fragments(b)), 6),
            "zero_codes_A": round(float(((a & 0x7F) == 0).mean()), 6),
            "atom_sums_fp16": round(atom_sums_fp16(a, b), 6), **strassen_preadds(a),
            **{f"{n}_B": v for n, v in strassen_preadds(b).items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atom", default="sm120-e4m3-k32", help="the forming atom (pearlc_census.ATOMS or groups:width:zero)")
    ap.add_argument("--families", default=",".join(C.FAMILIES + C.ACTIVATION_FAMILIES + C.OUTLIER_FAMILIES
                                                   + C.FREEZE_FAMILIES))
    ap.add_argument("--ks", default="8192")
    ap.add_argument("--tile", default="64x64")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--line-norm", type=float, default=16.0)
    ap.add_argument("--prices", default=None, help="JSON {name: {route: units}} merged over PRICES (e.g. measured rates)")
    ap.add_argument("--out", default="pearlc_attacks.json")
    args = ap.parse_args()
    C.set_atom(args.atom)
    prices = {n: dict(p) for n, p in PRICES.items()}
    if args.prices:
        for n, p in json.loads(Path(args.prices).read_text()).items():
            prices.setdefault(n, dict(PRICES["sm120-public"])).update(p)
    rows, cols = map(int, args.tile.split("x"))
    cells = []
    for k in map(int, args.ks.split(",")):
        for fam in args.families.split(","):
            m = measure(fam, k, rows, cols, args.run, args.line_norm)
            m["priced"] = {n: price(m, p) for n, p in prices.items()}
            cells.append(m)
            print(json.dumps(m), flush=True)
    best = {n: max(((c["family"], c["k"], r, v["saving_ub"]) for c in cells if c["rules_ok"]
                    for r, v in c["priced"][n].items() if "saving_ub" in v), key=lambda t: t[3], default=None)
            for n in prices}
    doc = {"tool": "pouw_pearlc_attacks", "atom": C.ATOM_NAME, "tile": args.tile, "line_norm": args.line_norm,
           "prices": prices, "cells": cells, "best_in_domain_route_saving_ub": best}
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"best_in_domain_route_saving_ub": best}))


if __name__ == "__main__":
    main()
