"""Pearl-C's FP8 atom under the data-flow model's P1, on v1-formed operands (CPU, exact).

For each device (the H100's HOPPER_E4M3_K32, sm_120's BLACKWELL_SM120_E4M3_M16N8K32) this forms a tile of A and B with
`pearl_c.form_v1` from synthetic FP32 rows, runs the device's chain over every output, and reports:

- per product, whether the atom's alignment keeps it exactly, truncates it or drops it;
- per atom, whether the atom's word equals the exact sum's FP32 truncation (what an exact INT8 or integer path computes
  before any fix-up), the exact sum truncated to 14 bits (H100), and the two-k16-group lowering an FP16 path gives;
- per output, whether a chain built from each of those stand-in atoms reproduces C̃ word for word;
- per 16 x 32 A fragment and 8 x 32 B fragment, how many 4-binade INT8 exponent classes hold the codes exactly, and the
  class pairs an exact INT8 emulation multiplies (all, and on the H100 only those with a product the atom keeps), and
  the pairs with per-row (A) and per-column (B) windows, a row's scale pulled out of its outputs (`rowwise`);
- the exponent span of each row's 32 codes per atom;
- NVFP4 (sm_120): the distinct exponents per 16-code block, and the cost in FP8 MACs per product of the hi/lo split
  construction that takes one exponent per block per pass (2·P_A·P_B per m16n8k64 fragment).

Usage: python atoms.py [--m 16] [--n 16] [--k 1024] [--seed 1] [--out FILE.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace

import numpy as np

from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_kw as P
from verity.core.protocols.pouw.schemes.pearl_c_device import H100, SM120
from verity.ml.circuits.silicon import fp32
from verity.ml.circuits.silicon.models import GroupSum
from verity.ml.circuits.silicon.term import FP32_SIGNIFICAND_WIDTH, Term, e4m3_product, fp32_term, pack_fp32

DEVICES = {"h100": H100, "sm120": SM120}


def families(rng: np.random.Generator, m: int, n: int, k: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """A (activations, m x k) and B (weights, n x k) as float32, per family."""
    w = rng.normal(0, 0.02, (n, k)).astype(np.float32)
    gauss = rng.normal(0, 1, (m, k)).astype(np.float32)
    t3 = rng.standard_t(3, (m, k)).astype(np.float32)
    llm = rng.normal(0, 1, (m, k)).astype(np.float32)
    llm[:, rng.choice(k, 4, replace=False)] *= 8.0
    const = np.ones((m, k), np.float32)
    return {"gauss": (gauss, w), "t3": (t3, w), "llm-outlier": (llm, w), "const-ones": (const, np.full((n, k), 0.02, np.float32))}


def words(x: np.ndarray) -> list[list[int]]:
    return [list(map(int, row)) for row in x.view(np.uint32)]


def exact_terms(acc_word: int, a: list[int], b: list[int]) -> list[Term]:
    return [fp32_term(acc_word)] + [e4m3_product(x, y) for x, y in zip(a, b)]


def exact_value(terms: list[Term]) -> tuple[int, int]:
    """sum of terms as (signed integer, exponent of its unit)."""
    lo = min(t.exponent - (FP32_SIGNIFICAND_WIDTH - 1) for t in terms if t.magnitude) if any(t.magnitude for t in terms) else 0
    total = 0
    for t in terms:
        if t.magnitude:
            v = t.magnitude << (t.exponent - (FP32_SIGNIFICAND_WIDTH - 1) - lo)
            total += -v if t.negative else v
    return total, lo


def rz_word(total: int, lo: int, width: int = 24) -> int:
    """The FP32 word of total·2^lo truncated toward zero to `width` significant bits (then to FP32's grid); zero is +0."""
    if total == 0:
        return 0
    neg, mag = total < 0, abs(total)
    top = lo + mag.bit_length() - 1
    if top < -126:
        sub = mag >> (-149 - lo) if lo < -149 else mag << (lo + 149)
        return pack_fp32(Term(neg, -126, sub)) if sub else 0
    keep = min(width, 24)
    shift = mag.bit_length() - keep
    mag = mag >> shift if shift > 0 else mag << -shift
    mag <<= 24 - keep
    return pack_fp32(Term(neg, top, mag))


def representable(total: int, lo: int) -> bool:
    """Whether total·2^lo is an FP32 value (24 significant bits, normal or subnormal grid), so no step can round it."""
    if total == 0:
        return True
    mag = abs(total)
    tz = (mag & -mag).bit_length() - 1
    odd, e = mag >> tz, lo + tz
    return odd.bit_length() <= 24 and e >= -149 and e + odd.bit_length() - 1 <= 127


def classify(atom: GroupSum, acc_word: int, a: list[int], b: list[int]) -> tuple[list[str], int]:
    """Per product: 'zero', 'exact', 'trunc' or 'drop' under the atom's alignment (`group_sum`), and the group's max exponent."""
    terms = exact_terms(acc_word, a, b)
    mx = atom.zero_exponent
    for t in terms:
        if t.magnitude and t.exponent > mx:
            mx = t.exponent
    rescale = atom.width - FP32_SIGNIFICAND_WIDTH
    out = []
    for t in terms[1:]:
        if not t.magnitude:
            out.append("zero")
            continue
        scaled = t.magnitude << rescale if rescale >= 0 else t.magnitude >> -rescale
        sh = mx - t.exponent
        aligned = scaled >> sh
        out.append("drop" if aligned == 0 else ("exact" if aligned << sh == scaled else "trunc"))
    return out, mx


def code_exponent(c: int) -> int | None:
    if c & 0x7F == 0:
        return None
    e = (c >> 3) & 0xF
    return -9 if e == 0 else e - 10


def windows(exps: set[int]) -> list[int]:
    """Greedy cover of the exponents by 4-binade INT8 classes [top-3, top]: the classes' tops, high to low."""
    tops, left = [], sorted(exps, reverse=True)
    while left:
        top = left[0]
        tops.append(top)
        left = [e for e in left if e < top - 3]
    return tops


def class_of(e: int, tops: list[int]) -> int:
    for i, top in enumerate(tops):
        if top - 3 <= e <= top:
            return i
    raise AssertionError


def chain_with(atom_fn, a: list[int], b: list[int], device) -> int:
    total = 0
    for ws in range(0, len(a), device.window):
        local = 0
        for s in range(ws, min(ws + device.window, len(a)), C.SLICE):
            local = atom_fn(local, a[s:s + C.SLICE], b[s:s + C.SLICE])
        total = fp32.add(total, local)
        if total == 0x80000000:
            total = 0
    return total


def run(m: int, n: int, k: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    seed_a = C.S512Seed(bytes(rng.integers(0, 256, 64, dtype=np.uint8)))
    seed_b = C.S512Seed(bytes(rng.integers(0, 256, 64, dtype=np.uint8)))
    report: dict = {"m": m, "n": n, "k": k, "seed": seed, "families": {}}
    for fam, (xa, xb) in families(rng, m, n, k).items():
        report["families"][fam] = {}
        for dname, dev in DEVICES.items():
            fa, fb = C._basis(seed_b, 0, k, C.LINE_NORM_V1), C._basis(seed_b, 1, k, C.LINE_NORM_V1)
            ea = [C.sample_line(seed_a, 0, 0, i) for i in range(m)]
            eb = [C.sample_line(seed_b, 1, 0, j) for j in range(n)]
            A = C.form_v1(words(xa), ea, fa, dev)
            B = C.form_v1(words(xb), eb, fb, dev)
            admissible = [P.f32_of_bits(r) * P.f32_of_bits(al) >= P.SIGMA_MIN and C.live_row(row)
                          for side in (A, B) for r, al, row in zip(side.rho, side.alpha, side.x)]
            two16 = replace(dev.atom, name="two-k16", groups=(16, 16), width=26, zero_exponent=-133)
            alts = {
                "exact_rz24": lambda acc, a, b: rz_word(*exact_value(exact_terms(acc, a, b))),
                "two_k16_w26": two16.step,
            }
            if dname == "h100":
                alts["exact_rz14"] = lambda acc, a, b: rz_word(*exact_value(exact_terms(acc, a, b)), width=14)
            kinds = {"zero": 0, "exact": 0, "trunc": 0, "drop": 0}
            atom_ok = {name: 0 for name in alts}
            atom_inexact = 0
            atoms = 0
            pairs_kept: dict[tuple[int, int, int], set] = {}
            for i in range(m):
                for j in range(n):
                    a, b = A.codes[i], B.codes[j]
                    for ws in range(0, k, dev.window):
                        local = 0
                        for s in range(ws, min(ws + dev.window, k), C.SLICE):
                            sa, sb = a[s:s + C.SLICE], b[s:s + C.SLICE]
                            cls, _ = classify(dev.atom, local, sa, sb)
                            for c in cls:
                                kinds[c] += 1
                            word = dev.atom.step(local, sa, sb)
                            atom_inexact += not representable(*exact_value(exact_terms(local, sa, sb)))
                            for name, fn in alts.items():
                                atom_ok[name] += fn(local, sa, sb) == word
                            atoms += 1
                            key = (i // 16, j // 8, s // C.SLICE)
                            kept = pairs_kept.setdefault(key, set())
                            for t, c in enumerate(cls):
                                if c in ("exact", "trunc"):
                                    kept.add((i // 16 * 16, j // 8 * 8, s, t, i, j))
                            local = word
            # chain level
            chain_ok = {name: 0 for name in alts}
            for i in range(m):
                for j in range(n):
                    ref = C.chain(A.codes[i], B.codes[j], dev)
                    for name, fn in alts.items():
                        chain_ok[name] += chain_with(fn, A.codes[i], B.codes[j], dev) == ref
            # INT8 exponent classes per fragment
            frag_stats = []
            for i0 in range(0, m, 16):
                for j0 in range(0, n, 8):
                    for s in range(0, k, C.SLICE):
                        ea_set = {code_exponent(A.codes[i][s + t]) for i in range(i0, min(i0 + 16, m)) for t in range(C.SLICE)} - {None}
                        eb_set = {code_exponent(B.codes[j][s + t]) for j in range(j0, min(j0 + 8, n)) for t in range(C.SLICE)} - {None}
                        ta, tb = windows(ea_set), windows(eb_set)
                        needed = set()
                        for (_, _, ss, t, i, j) in pairs_kept.get((i0 // 16, j0 // 8, s // C.SLICE), ()):
                            xa_e, xb_e = code_exponent(A.codes[i][ss + t]), code_exponent(B.codes[j][ss + t])
                            needed.add((class_of(xa_e, ta), class_of(xb_e, tb)))
                        ra = max(len(windows({code_exponent(A.codes[i][s + t]) for t in range(C.SLICE)} - {None}))
                                 for i in range(i0, min(i0 + 16, m)))
                        rb = max(len(windows({code_exponent(B.codes[j][s + t]) for t in range(C.SLICE)} - {None}))
                                 for j in range(j0, min(j0 + 8, n)))
                        frag_stats.append((len(ta), len(tb), len(ta) * len(tb), len(needed), ra * rb))
            spans = []
            for side in (A, B):
                for row in side.codes:
                    for s in range(0, k, C.SLICE):
                        es = [code_exponent(c) for c in row[s:s + C.SLICE]]
                        es = [e for e in es if e is not None]
                        if es:
                            spans.append(max(es) - min(es))
            # NVFP4 (sm_120's OMMA m16n8k64, one scale per 16 codes of a row or column): a code at exponent E splits
            # into hi = (8 or 12)·2^E and lo = (0..3)·2^E, both E2M1 under the scale 2^(E+1), so a pass takes one
            # exponent per block, and the four hi/lo products of a pass cost 4 × 0.5 FP8 MACs.
            blk_exps = [len({code_exponent(c) for c in row[s:s + 16]} - {None}) for side in (A, B) for row in side.codes
                        for s in range(0, k, 16)]
            fp4_cost = []
            for i0 in range(0, m, 16):
                for j0 in range(0, n, 8):
                    for s in range(0, k, 64):
                        pa = max(len({code_exponent(c) for c in A.codes[i][q:q + 16]} - {None})
                                 for i in range(i0, min(i0 + 16, m)) for q in range(s, s + 64, 16))
                        pb = max(len({code_exponent(c) for c in B.codes[j][q:q + 16]} - {None})
                                 for j in range(j0, min(j0 + 8, n)) for q in range(s, s + 64, 16))
                        fp4_cost.append(2 * max(pa, 1) * max(pb, 1))
            nz = sum(v for kk, v in kinds.items() if kk != "zero")
            outs = m * n
            report["families"][fam][dname] = {
                "admissible_rows": f"{sum(admissible)}/{len(admissible)}",
                "products": {kk: v for kk, v in kinds.items()},
                "nonzero_product_share": {kk: round(kinds[kk] / nz, 5) for kk in ("exact", "trunc", "drop")},
                "atoms": atoms,
                "atom_exact_sum_needs_rounding": round(atom_inexact / atoms, 5),
                "atom_matches": {name: round(v / atoms, 5) for name, v in atom_ok.items()},
                "chain_matches": {name: round(v / outs, 5) for name, v in chain_ok.items()},
                "int8_classes_A_mean": round(statistics.mean(f[0] for f in frag_stats), 3),
                "int8_classes_B_mean": round(statistics.mean(f[1] for f in frag_stats), 3),
                "int8_pairs_all_mean": round(statistics.mean(f[2] for f in frag_stats), 3),
                "int8_pairs_all_max": max(f[2] for f in frag_stats),
                "int8_pairs_kept_mean": round(statistics.mean(f[3] for f in frag_stats), 3),
                "int8_pairs_rowwise_mean": round(statistics.mean(f[4] for f in frag_stats), 3),
                "int8_pairs_rowwise_max": max(f[4] for f in frag_stats),
                "row_span_binades": {"median": statistics.median(spans), "p10": float(np.percentile(spans, 10)),
                                     "p90": float(np.percentile(spans, 90)), "max": max(spans)},
                "nvfp4_block_exponents_mean": round(statistics.mean(blk_exps), 3),
                "nvfp4_one_exponent_cost_mean": round(statistics.mean(fp4_cost), 3),
                "nvfp4_one_exponent_cost_max": max(fp4_cost),
            }
            print(fam, dname, json.dumps(report["families"][fam][dname]), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args()
    report = run(args.m, args.n, args.k, args.seed)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
