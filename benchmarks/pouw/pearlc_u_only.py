#!/usr/bin/env python3
"""Does U alone force Pearl-C's chain? The question behind hashing option P1 (bind U's words only, not C̃'s).

U is the peel folded into C̃'s FP32 accumulator (`pearl_c_u.py` on #449: four BF16 k16 steps into acc = C̃), so U reads
C̃'s word only above the peel's truncation. Per word of a 64 × 8 tile formed by §5's forming (`pearlc_census.form_s5`), on
the chosen accumulator step (`--atom`), this measures:
- **promotion:** the share of words whose U from the unpromoted C̃ equals U from the promoted one (the fused columns and the
  peel as pinned). A high share would let a prover that binds U only drop the promotion adds (v1's credited n/(32·G) FP32
  adds per MAC);
- **low bits:** the share of words whose U is unchanged when C̃'s word moves by d ulps, d = 1, 2, 4, 16;
- **skips:** per word, the atoms whose lone skip leaves U unchanged, against those that leave C̃ unchanged (the debit's
  flags), with and without promotion.

The BF16 step and the P_A / P_B order are `pearl_c_u.py`'s (#449 at `fd19e02f`), restated here; the FMA chain rounds once
per step, exactly.

    python benchmarks/pouw/pearlc_u_only.py --atom sm120 --k 8192 --families gaussian,outliers-first
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pearlc_census as C  # noqa: E402
import run_result  # noqa: E402


# ---- pearl_c_u.py's BF16 k16 step (HOPPER_BF16_WGMMA_K16, width 26, floor −133), FP32 words in and out --------------

def _acc_term(word: int) -> tuple[bool, int, int]:
    field, man = (word >> 23) & 0xFF, word & 0x7FFFFF
    if field == 0:
        return word >> 31 == 1, -126, man
    return word >> 31 == 1, field - 127, man | 1 << 23


def _bf16_product(x: int, y: int) -> tuple[bool, int, int]:
    ex, ey = (x >> 7) & 0xFF, (y >> 7) & 0xFF
    sx, sy = (x & 0x7F) | ((1 << 7) if ex else 0), (y & 0x7F) | ((1 << 7) if ey else 0)
    return ((x ^ y) >> 15) & 1 == 1, max(ex, 1) + max(ey, 1) - 254, (sx * sy) << 9


def k16(acc: int, xs, ys) -> int:
    width, floor, rescale = 26, -133, 2
    terms = [t for t in [_acc_term(acc), *(_bf16_product(a, b) for a, b in zip(xs, ys))] if t[2]]
    top = max([floor, *(e for _, e, _ in terms)])
    total = sum(-((m << rescale) >> (top - e)) if neg else (m << rescale) >> (top - e) for neg, e, m in terms)
    if total == 0:
        return 0
    neg, mag = total < 0, abs(total)
    bits = mag.bit_length()
    exp = top + bits - width
    mag = mag >> (bits - width) if bits > width else mag << (width - bits)
    if exp < -126:
        mag >>= -126 - exp
        exp = -126
    mag >>= rescale
    if mag == 0:
        return 0
    field = exp + 127 - (0 if mag & 1 << 23 else 1)
    return int(neg) << 31 | field << 23 | (mag & 0x7FFFFF)


def u_word(c: int, pa, pb) -> int:
    acc = c
    for s in range(0, 64, 16):
        acc = k16(acc, pa[s:s + 16], pb[s:s + 16])
    return acc


# ---- FP32 and BF16 words -----------------------------------------------------------------------------------------------

def f32_word(v: float) -> int:
    return int(np.float32(v).view(np.uint32))


def word_val(w: int) -> Fraction:
    neg, e, m = _acc_term(w)
    v = Fraction(m) * Fraction(2) ** (e - 23)
    return -v if neg else v


def rn_f32(x: Fraction) -> int:
    if x == 0:
        return 0
    neg, x = x < 0, abs(x)
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while Fraction(2) ** e > x:
        e -= 1
    while Fraction(2) ** (e + 1) <= x:
        e += 1
    e = max(e, -126)
    q = x / Fraction(2) ** (e - 23)
    n = q.numerator // q.denominator
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
        n += 1
    return f32_word((-1 if neg else 1) * float(Fraction(n) * Fraction(2) ** (e - 23)))


def rn_bf16(word: int) -> int:
    return ((word + 0x7FFF + ((word >> 16) & 1)) >> 16) & 0xFFFF


def bf16_of_code(code: int) -> int:
    return f32_word(float(C.val(np.array([code]))[0])) >> 16


# ---- one tile --------------------------------------------------------------------------------------------------------

def chains(a: np.ndarray, b: np.ndarray, G: int) -> np.ndarray:
    return C.ticket(a, b, G)


def tile(fam: str, k: int, run: int, line_norm: float = 16.0, delta: float = 1.0, skip_words: int = 32) -> dict:
    rows, cols = 64, 8
    rng = C._rng(run, fam, k, delta)
    FA = C.lines(k, C._rng(run, "FA", k))
    XA, XB = C.family(fam, rows, cols, k, FA, rng)
    FcA = C.enc(C.unit_lines(k, C._rng(run, "FA-lines", k)) * (line_norm / np.sqrt(32)))
    FcB = C.enc(C.unit_lines(k, C._rng(run, "FB-lines", k)) * (line_norm / np.sqrt(32)))
    la, lb = [], []
    a, *_ = C.form_s5(XA, FcA, delta, line_norm, C._rng(run, fam, k, delta, "unit-A"), lines_out=la)
    b, *_ = C.form_s5(XB, FcB, delta, line_norm, C._rng(run, fam, k, delta, "unit-B"), lines_out=lb)
    EA, EB = la[0], lb[0]
    out = {"family": fam, "k": k}
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    A, B = a[ii.ravel()], b[jj.ravel()]
    for G in (4, 0):
        ct = chains(A, B, G)
        af = chains(np.repeat(a, 32, 0), np.tile(FcB.T, (rows, 1)), G).reshape(rows, 32)
        bf = chains(np.repeat(b, 32, 0), np.tile(FcA.T, (cols, 1)), G).reshape(cols, 32)
        gm = chains(np.repeat(FcB.T, 32, 0), np.tile(FcA.T, (32, 1)), G).reshape(32, 32)
        beA = [[bf16_of_code(int(x)) for x in EA[i]] for i in range(rows)]
        beB = [[bf16_of_code(int(x)) for x in EB[j]] for j in range(cols)]
        pa = [beA[i] + [rn_bf16(f32_word(v)) for v in af[i]] for i in range(rows)]
        pb = []
        for j in range(cols):
            mid = []
            for q in range(32):
                acc = 0
                for p in range(32):
                    acc = rn_f32(word_val(beB[j][p] << 16) * word_val(f32_word(gm[p, q])) + word_val(acc))
                mid.append(rn_bf16(rn_f32(word_val(acc) - word_val(f32_word(bf[j, q])))))
            pb.append(mid + [x ^ 0x8000 for x in beB[j]])
        words = [f32_word(v) for v in ct]
        U = [u_word(words[w], pa[w // cols], pb[w % cols]) for w in range(len(words))]
        tag = "G4" if G else "pure"
        if G:
            ct_pure = chains(A, B, 0)
            out["promotion_skippable_under_U_share"] = float(np.mean(
                [u_word(f32_word(ct_pure[w]), pa[w // cols], pb[w % cols]) == U[w] for w in range(len(words))]))
            out["promotion_skippable_under_Ct_share"] = float(np.mean(ct_pure == ct))
        for d in (1, 2, 4, 16):
            same = [u_word(words[w] + d if words[w] & 0x7FFFFF < 0x7FFFFF - d else words[w] - d, pa[w // cols], pb[w % cols]) == U[w]
                    for w in range(len(words))]
            out[f"{tag}_U_ignores_{d}ulp_share"] = float(np.mean(same))
        # lone-atom skips on a word sample: U unchanged, against C̃ unchanged
        T = k // 32
        sample = np.unique(np.linspace(0, len(A) - 1, skip_words).astype(int))
        u_flags = c_flags = 0
        for w in sample:
            for t in range(T):
                keep = np.ones(k, bool)
                keep[32 * t:32 * t + 32] = False
                aa = np.where(keep, A[w], 0).astype(A.dtype)[None]
                cs = chains(aa, B[w:w + 1], G)[0]
                c_flags += cs == ct[w]
                u_flags += u_word(f32_word(cs), pa[w // cols], pb[w % cols]) == U[w]
        out[f"{tag}_skip_leaves_Ct_share"] = c_flags / (len(sample) * T)
        out[f"{tag}_skip_leaves_U_share"] = u_flags / (len(sample) * T)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atom", choices=tuple(C.ATOMS), default="sm120")
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--families", default="gaussian,spikes-first,outliers-first,grid-aligned,aligned-spikes-r64")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--skip-words", type=int, default=32)
    ap.add_argument("--out", default="pearlc_u_only.json")
    args = ap.parse_args()
    C.set_atom(args.atom)
    t0, rows = time.time(), []
    for fam in args.families.split(","):
        r = tile(fam, args.k, args.run, skip_words=args.skip_words)
        rows.append(r)
        print(json.dumps(r), flush=True)
    worst = max(r["promotion_skippable_under_U_share"] for r in rows)
    extra = max(max(r["G4_skip_leaves_U_share"] - r["G4_skip_leaves_Ct_share"],
                    r["pure_skip_leaves_U_share"] - r["pure_skip_leaves_Ct_share"]) for r in rows)
    doc = {"tool": "pouw_pearlc_u_only", "atom": args.atom, "k": args.k, "cells": rows,
           "promotion_skippable_under_U_max": worst, "extra_skips_under_U_max": extra, "seconds": round(time.time() - t0, 1)}
    out = Path(args.out)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    run_result.write(out, True, "pearlc-u-only", f"promotion skippable under U-only binding on up to {worst:.4f} of words; "
                     f"extra lone-atom skips under U {extra:.5f}",
                     [("promotion_skippable_under_U_max", worst, "fraction"), ("extra_skips_under_U_max", extra, "fraction")])
    print(json.dumps({k: v for k, v in doc.items() if k != "cells"}))


if __name__ == "__main__":
    main()
