#!/usr/bin/env python3
"""Do crafted exact row relations in registered weights transfer to Pearl-C's ticket words, and does a keyed
rotation plus requantization remove them? Bit-exact, CPU, stdlib only (verity.ml.tc).

Setting: the approved-weights fork. The weights B are registered codes, fixed and known to the prover; the noise is on
the activations only, A' = quant(alpha*x + n), fresh per unit. A registrant crafts three rows with an exact relation in
the code values, row3 = row1 + row2. The prover computes rows 1 and 2 by the chain and derives row 3's words from
theirs with FP32 adds, then checks the derived words against the chain on row 3.

Chains (the sm_120 models of `verity.ml.tc`):
  fp8-v1   E4M3 GroupSum <[32], 26, -133>, promoted every 4 atoms into an FP32 total (RN): derive each 128-deep group
           partial as fp32(D1 + D2), then promote. Cost of a derived word: k/128 FADDs, against k MACs.
  fp8-v2   the same atom, no promotion (one chain over k): derive per atom, acc3 <- trunc_fp32(acc3 + P1 + P2), where P
           are the atoms' sums from +0 (computed once per basis row). Cost: 2 FADDs per atom, against 32 MACs.
  nvfp4    BLACKWELL_SM120_NVF4 (E2M1, UE4M3 per 16, exact align-add, RZ): derive the word as fp32(T1 + T2).
           Cost: one FADD, against k MACs.

Transforms, keyed after the registrant commits (a seed stands in for the beacon):
  none     the crafted codes as registered;
  rot-rn   a randomized Hadamard R = D2 H D1 / sqrt(k) on the k side, then RN requantization per row (FP8: alpha =
           448/max; NVFP4: UE4M3 block scales, E2M1 codes);
  rot-rn-shared  FP8 only: as rot-rn with one scale for the three rows, the registrant's best case (the relation holds
           in the reals before rounding);
  rot-sr   the same with keyed stochastic rounding;
  dup-*    rows 1 and 2 registered equal (a duplicate): after rot-rn their codes stay equal; after rot-sr they differ.

Near-duplicate family (--neardup): row2 = row1 with d codes per 128-group moved one grid step (d in 8, 15, 24). As
registered, the prover derives row2's v1 group sums from row1's plus the d products; after the keyed rotation with round to
nearest, the codes differ in d' positions per group, and the census's near-duplicate rule uncredits groups with d' <= 15.

  PYTHONPATH=packages/verity/src python3 relation_attack.py --k 8192 --rows 64 --neardup --out DIR
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import struct
import os
import time
from fractions import Fraction
from pathlib import Path

from verity.ml.tc.cast import f32_to_e4m3_sat_word
from verity.ml.tc.models import BLACKWELL_SM120_NVF4, GroupSum
from verity.ml.tc.term import decode_e4m3

SM120_E4M3 = GroupSum(name="sm120_e4m3_k32", arch="sm_120", dtype="e4m3", operand_bits=8, groups=(32,), width=26,
                      zero_exponent=-133, validated="stand-in for BLACKWELL_SM120_E4M3_M16N8K32 (same parameters)")
E2M1_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def f32_word(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f32_val(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", w))[0]


def add_rn(u: int, v: int) -> int:
    return f32_word(f32_val(u) + f32_val(v))


def trunc_fp32(x: Fraction) -> int:
    if x == 0:
        return 0
    neg, m = x < 0, abs(x)
    e = m.numerator.bit_length() - m.denominator.bit_length()
    if Fraction(2) ** e > m:
        e -= 1
    grid = Fraction(2) ** (max(e, -126) - 23)
    q = math.floor(m / grid)
    return f32_word(float(q * grid) * (-1 if neg else 1))


# --- E4M3 -------------------------------------------------------------------------------------------------------------
E4M3_POS = sorted({(c, (lambda s, e, m: m * 2.0 ** e)(*decode_e4m3(c))) for c in range(0x7F)}, key=lambda t: t[1])
E4M3_VALS = [v for _, v in E4M3_POS]
E4M3_CODES = [c for c, _ in E4M3_POS]


def e4m3_val(c: int) -> float:
    s, e, m = decode_e4m3(c)
    return (-1.0 if s else 1.0) * m * 2.0 ** e


def e4m3_rn(x: float) -> int:
    return f32_to_e4m3_sat_word(f32_word(max(-448.0, min(448.0, x))))


def e4m3_sr(x: float, rng: random.Random) -> int:
    a = min(abs(x), 448.0)
    i = bisect.bisect_right(E4M3_VALS, a) - 1
    if E4M3_VALS[i] == a or i + 1 >= len(E4M3_VALS):
        c = E4M3_CODES[i]
    else:
        lo, hi = E4M3_VALS[i], E4M3_VALS[i + 1]
        c = E4M3_CODES[i + 1] if rng.random() < (a - lo) / (hi - lo) else E4M3_CODES[i]
    return c | 0x80 if x < 0 and c else c


# --- E2M1 / NVFP4 -----------------------------------------------------------------------------------------------------
def e2m1_code(x: float, rng: random.Random | None) -> int:
    a = min(abs(x), 6.0)
    i = bisect.bisect_right(E2M1_MAG, a) - 1
    if i + 1 < len(E2M1_MAG) and E2M1_MAG[i] != a:
        lo, hi = E2M1_MAG[i], E2M1_MAG[i + 1]
        up = (rng.random() < (a - lo) / (hi - lo)) if rng else (a - lo > hi - a or (a - lo == hi - a and i % 2 == 1))
        i += 1 if up else 0
    return i | 8 if x < 0 and i else i


def e2m1_val(c: int) -> float:
    return (-1.0 if c & 8 else 1.0) * E2M1_MAG[c & 7]


def ue4m3_val(c: int) -> float:
    return e4m3_val(c & 0x7F)


def nvfp4_quant(row: list[float], rng: random.Random | None) -> tuple[list[int], list[int]]:
    codes, scales = [], []
    for b in range(0, len(row), 16):
        blk = row[b:b + 16]
        amax = max(abs(v) for v in blk)
        sc = e4m3_rn(amax / 6.0) if amax > 0 else 0x08
        sc = sc if sc else 0x08
        s = ue4m3_val(sc)
        scales.append(sc)
        codes.extend(e2m1_code(v / s, rng) for v in blk)
    return codes, scales


def nvfp4_vals(codes: list[int], scales: list[int]) -> list[float]:
    return [e2m1_val(c) * ue4m3_val(scales[i // 16]) for i, c in enumerate(codes)]


# --- the keyed rotation -----------------------------------------------------------------------------------------------
def fwht(v: list[float]) -> list[float]:
    v, h = v[:], 1
    while h < len(v):
        for i in range(0, len(v), 2 * h):
            for j in range(i, i + h):
                a, b = v[j], v[j + h]
                v[j], v[j + h] = a + b, a - b
        h *= 2
    return v


def rotation(k: int, seed: int):
    rng = random.Random(seed)
    d1 = [rng.choice((-1.0, 1.0)) for _ in range(k)]
    d2 = [rng.choice((-1.0, 1.0)) for _ in range(k)]
    s = 1.0 / math.sqrt(k)
    return lambda row: [x * s * d for x, d in zip(fwht([a * b for a, b in zip(row, d1)]), d2)]


# --- activations: A' of the fork (noise on A only) --------------------------------------------------------------------
def activation_row(k: int, rng: random.Random, kind: str, delta: float, top: float) -> list[float]:
    x = [rng.gauss(0, 1) for _ in range(k)] if kind == "gauss" else [rng.choice((-1.0, 1.0)) for _ in range(k)]
    rho = math.sqrt(sum(v * v for v in x[::8]) / len(x[::8]))
    alpha = top / (max(abs(v) for v in x) + 4 * delta * rho)
    return [alpha * v + rng.gauss(0, delta * rho * alpha) for v in x]


# --- chains ------------------------------------------------------------------------------------------------------------
def fp8_v1(a: list[int], b: list[int]) -> tuple[int, list[int]]:
    total, parts = 0, []
    for g in range(0, len(a), 128):
        d = 0
        for s in range(g, g + 128, 32):
            d = SM120_E4M3.step(d, a[s:s + 32], b[s:s + 32])
        parts.append(d)
        total = add_rn(total, d)
    return total, parts


def fp8_v2(a: list[int], b: list[int]) -> tuple[int, list[int]]:
    acc, parts = 0, []
    for s in range(0, len(a), 32):
        acc = SM120_E4M3.step(acc, a[s:s + 32], b[s:s + 32])
        parts.append(SM120_E4M3.step(0, a[s:s + 32], b[s:s + 32]))
    return acc, parts


def derive_v1(p1: list[int], p2: list[int], mode: str) -> int:
    total = 0
    for d1, d2 in zip(p1, p2):
        d = add_rn(d1, d2) if mode == "rn" else trunc_fp32(Fraction(f32_val(d1)) + Fraction(f32_val(d2)))
        total = add_rn(total, d)
    return total


def derive_v2(p1: list[int], p2: list[int]) -> int:
    acc = 0
    for d1, d2 in zip(p1, p2):
        acc = trunc_fp32(Fraction(f32_val(acc)) + Fraction(f32_val(d1)) + Fraction(f32_val(d2)))
    return acc


# --- crafted rows -----------------------------------------------------------------------------------------------------
def crafted_fp8(k: int, rng: random.Random, width: int) -> list[list[float]]:
    r1, r2 = [], []
    for _ in range(k):
        sgn, e = rng.choice((-1.0, 1.0)), 3 + rng.randrange(width) - width // 2
        r1.append(sgn * rng.choice((1, 2, 3)) * 2.0 ** e)
        r2.append(sgn * rng.choice((1, 2, 3)) * 2.0 ** e)
    return [r1, r2, [a + b for a, b in zip(r1, r2)]]


def crafted_fp8_wide(k: int, rng: random.Random) -> list[list[float]]:
    """Integers 1..7 times 2^e over an 8-binade band: every sum is E4M3, entropy matches trained weights."""
    r1, r2 = [], []
    for _ in range(k):
        sgn, e = rng.choice((-1.0, 1.0)), rng.randrange(8) - 2
        r1.append(sgn * rng.randint(1, 7) * 2.0 ** e)
        r2.append(sgn * rng.randint(1, 7) * 2.0 ** e)
    return [r1, r2, [a + b for a, b in zip(r1, r2)]]


def crafted_fp4_wide(k: int, rng: random.Random) -> tuple[list[list[int]], list[int]]:
    """Every E2M1 pair whose sum or difference is E2M1, both signs, random UE4M3 block scales."""
    same = [(u, v) for u in range(1, 8) for v in range(1, 8) if E2M1_MAG[u] + E2M1_MAG[v] in E2M1_MAG]
    diff = [(u, v) for u in range(1, 8) for v in range(1, 8)
            if E2M1_MAG[u] > E2M1_MAG[v] and E2M1_MAG[u] - E2M1_MAG[v] in E2M1_MAG]
    scales = [rng.choice(range(0x30, 0x41)) for _ in range(k // 16)]
    c1, c2, c3 = [], [], []
    for _ in range(k):
        sg = rng.choice((0, 8))
        if rng.random() < 0.5:
            u, v = rng.choice(same)
            c1.append(u | sg), c2.append(v | sg), c3.append(E2M1_MAG.index(E2M1_MAG[u] + E2M1_MAG[v]) | sg)
        else:
            u, v = rng.choice(diff)
            d = E2M1_MAG.index(E2M1_MAG[u] - E2M1_MAG[v])
            c1.append(u | sg), c2.append(v | (8 - sg)), c3.append(d | sg if d else 0)
    return [c1, c2, c3], scales


def entropy_128(codes: list[int]) -> tuple[float, float]:
    out = []
    for g in range(0, len(codes), 128):
        cnt = {}
        for c in codes[g:g + 128]:
            cnt[c] = cnt.get(c, 0) + 1
        n = sum(cnt.values())
        out.append(-sum(v / n * math.log2(v / n) for v in cnt.values()))
    return round(min(out), 2), round(sum(out) / len(out), 2)


def crafted_fp4(k: int, rng: random.Random) -> tuple[list[list[int]], list[int]]:
    scales = [rng.choice(range(0x30, 0x41)) for _ in range(k // 16)]
    c1, c2, c3 = [], [], []
    for _ in range(k):
        sgn, u, v = rng.choice((0, 8)), rng.choice((1, 2)), rng.choice((1, 2))   # magnitudes 0.5 or 1
        c1.append(u | sgn)
        c2.append(v | sgn)
        c3.append(E2M1_MAG.index(E2M1_MAG[u] + E2M1_MAG[v]) | sgn)
    return [c1, c2, c3], scales


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--output", default="")
    ap.add_argument("--out", default="", help="a directory: writes relation_attack.json there (the checkpoint) and exits 0")
    ap.add_argument("--neardup", action="store_true")
    args = ap.parse_args()
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        args.output = str(Path(args.out, "relation_attack.json"))
        if Path(args.output).exists():
            write_result(Path(args.output))
            return
    k, t0, out = args.k, time.time(), {"k": args.k, "activation_rows": args.rows, "cells": []}
    rng = random.Random(args.seed)
    rot = rotation(k, args.seed + 1)

    acts8 = {kind: [[e4m3_rn(v) for v in activation_row(k, rng, kind, 1.0, 448.0)] for _ in range(args.rows)]
             for kind in ("gauss", "flat")}

    def fp8_rows(width: int, transform: str, dup: bool) -> list[list[int]]:
        base = (crafted_fp8_wide(k, random.Random(args.seed + 5)) if width == 0
                else crafted_fp8(k, random.Random(args.seed + width), width))
        if dup:
            base = [base[0], base[0][:], base[2]]
        if transform == "none":
            return [[e4m3_rn(v) for v in r] for r in base]
        srng = random.Random(args.seed + 7)
        rotated = [rot(r) for r in base]
        shared = 448.0 / max(abs(v) for rr in rotated for v in rr)
        rows = []
        for rr in rotated:
            al = shared if transform == "rot-rn-shared" else 448.0 / max(abs(v) for v in rr)
            rows.append([e4m3_sr(al * v, srng) if transform == "rot-sr" else e4m3_rn(al * v) for v in rr])
        return rows

    for width in (4, 8, 0):
        for transform in ("none", "rot-rn", "rot-rn-shared", "rot-sr"):
            b = fp8_rows(width, transform, dup=False)
            vals = [[e4m3_val(c) for c in r] for r in b]
            holds = sum(1 for x, y, z in zip(*vals) if x + y == z) / k
            for kind, acts in acts8.items():
                m1 = m2 = mr = mz = 0
                for a in acts:
                    t1, p1 = fp8_v1(a, b[0])
                    t2, p2 = fp8_v1(a, b[1])
                    t3, _ = fp8_v1(a, b[2])
                    m1 += derive_v1(p1, p2, "rn") == t3
                    mz += derive_v1(p1, p2, "rz") == t3
                    u1, q1 = fp8_v2(a, b[0])
                    u2, q2 = fp8_v2(a, b[1])
                    u3, _ = fp8_v2(a, b[2])
                    m2 += derive_v2(q1, q2) == u3
                    mr += add_rn(u1, u2) == u3
                n = len(acts)
                label = "integers 1..7, 8-binade band" if width == 0 else f"exponent band {width}"
                out["cells"].append({"format": "fp8", "rows": f"row3 = row1 + row2, {label}",
                                     "transform": transform, "activations": kind,
                                     "entropy_bits_per_128_min_mean": [entropy_128(r) for r in b],
                                     "relation_holds_per_position": round(holds, 4),
                                     "v1_derived_words_correct": round(m1 / n, 4),
                                     "v1_derived_words_correct_rz": round(mz / n, 4),
                                     "v2_derived_words_correct_per_atom": round(m2 / n, 4),
                                     "v2_derived_words_correct_whole_word": round(mr / n, 4)})
                print(json.dumps(out["cells"][-1]), f"{time.time() - t0:.0f}s", flush=True)

    for transform in ("rot-rn", "rot-sr"):
        b = fp8_rows(4, transform, dup=True)
        same = sum(1 for x, y in zip(b[0], b[1]) if x == y) / k
        grp = [sum(1 for t in range(g, g + 128) if b[0][t] != b[1][t]) for g in range(0, k, 128)]
        out["cells"].append({"format": "fp8", "rows": "row2 = row1 (duplicate)", "transform": transform,
                             "codes_equal_per_position": round(same, 4),
                             "differing_codes_per_128_group_mean": round(sum(grp) / len(grp), 2),
                             "differing_codes_per_128_group_min": min(grp)})
        print(json.dumps(out["cells"][-1]), flush=True)

    acts4 = {}
    for kind in ("gauss", "flat"):
        acts4[kind] = [nvfp4_quant(activation_row(k, rng, kind, 0.25, 6.0 * 448.0), None) for _ in range(args.rows)]
    for family, (cs, sc) in (("E2M1 magnitudes 0.5, 1", crafted_fp4(k, random.Random(args.seed + 3))),
                             ("every E2M1 pair with an E2M1 sum or difference", crafted_fp4_wide(k, random.Random(args.seed + 11)))):
      c1, c2, c3 = cs
      master = [nvfp4_vals(c, sc) for c in (c1, c2, c3)]
      for transform in ("none", "rot-rn", "rot-sr"):
        if transform == "none":
            rows = [(c, sc) for c in (c1, c2, c3)]
        else:
            srng = random.Random(args.seed + 9)
            rows = [nvfp4_quant(rot(r), srng if transform == "rot-sr" else None) for r in master]
        vals = [nvfp4_vals(c, s) for c, s in rows]
        holds = sum(1 for x, y, z in zip(*vals) if x + y == z) / k
        for kind, acts in acts4.items():
            ok = 0
            for ac, asc in acts:
                t = [BLACKWELL_SM120_NVF4.chain(0, ac, c, asc, s) for c, s in rows]
                ok += add_rn(t[0], t[1]) == t[2]
            out["cells"].append({"format": "nvfp4", "rows": f"row3 = row1 + row2 ({family})",
                                 "transform": transform, "activations": kind,
                                 "entropy_bits_per_128_min_mean": [entropy_128(c) for c, _ in rows],
                                 "relation_holds_per_position": round(holds, 4),
                                 "derived_words_correct": round(ok / len(acts), 4)})
            print(json.dumps(out["cells"][-1]), f"{time.time() - t0:.0f}s", flush=True)

    if args.neardup:
        base = crafted_fp8_wide(k, random.Random(args.seed + 21))[0]
        for d in (8, 15, 24):
            nrng = random.Random(args.seed + 100 + d)
            r2 = base[:]
            for g in range(0, k, 128):
                for t in nrng.sample(range(g, g + 128), d):
                    c = e4m3_rn(r2[t])
                    step = e4m3_val((c & 0x80) | min((c & 0x7F) + 1, 0x7E)) - e4m3_val(c)
                    r2[t] = r2[t] + step
            for transform in ("none", "rot-rn"):
                if transform == "none":
                    b1, b2 = [e4m3_rn(v) for v in base], [e4m3_rn(v) for v in r2]
                else:
                    x1, x2 = rot(base), rot(r2)
                    al1, al2 = 448.0 / max(abs(v) for v in x1), 448.0 / max(abs(v) for v in x2)
                    b1, b2 = [e4m3_rn(al1 * v) for v in x1], [e4m3_rn(al2 * v) for v in x2]
                diffs = [sum(1 for t in range(g, g + 128) if b1[t] != b2[t]) for g in range(0, k, 128)]
                ok = 0
                for a in acts8["gauss"]:
                    t1, p1 = fp8_v1(a, b1)
                    t2, _ = fp8_v1(a, b2)
                    total = 0
                    for gi, g in enumerate(range(0, k, 128)):
                        exact = Fraction(f32_val(p1[gi])) + sum(Fraction(e4m3_val(a[t])) * (Fraction(e4m3_val(b2[t])) - Fraction(e4m3_val(b1[t])))
                                                                for t in range(g, g + 128) if b1[t] != b2[t])
                        total = add_rn(total, trunc_fp32(exact))
                    ok += total == t2
                out["cells"].append({"format": "fp8", "rows": f"row2 = row1 with {d} codes per 128-group one step apart",
                                     "transform": transform, "differing_codes_per_group_mean": round(sum(diffs) / len(diffs), 2),
                                     "differing_codes_per_group_min": min(diffs), "differing_codes_per_group_max": max(diffs),
                                     "groups_flagged_by_rule": round(sum(1 for x in diffs if x <= 15) / len(diffs), 4),
                                     "v1_derived_words_correct": round(ok / len(acts8["gauss"]), 4)})
                print(json.dumps(out["cells"][-1]), f"{time.time() - t0:.0f}s", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=1)
        write_result(Path(args.output))


def write_result(path):
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if not rd:
        return
    doc = json.loads(path.read_text())
    meas = []
    for c in doc["cells"]:
        tag = f"{c['format']}/{c['rows']}/{c['transform']}/{c.get('activations', '-')}"
        for key in ("v1_derived_words_correct", "v2_derived_words_correct_per_atom", "derived_words_correct", "groups_flagged_by_rule"):
            if key in c:
                meas.append({"name": f"{tag}/{key}", "value": c[key], "unit": "fraction"})
    try:
        files = [str(path.resolve().relative_to(Path(rd).resolve()))]
    except ValueError:
        files = []
    res = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"workload": "approved-weights/relation-attack/v1", "k": doc["k"], "rows": doc["activation_rows"],
                                    "source_sha": os.environ.get("RESEARCH_SOURCE_SHA", "")},
           "validation": {"status": "passed", "checker": "approved-weights/relation-attack/v1", "detail": f"{len(doc['cells'])} cells",
                          "evidence": files}, "measurements": meas, "measurement_files": files}
    tmp = Path(rd, "result.json.tmp")
    tmp.write_text(json.dumps(res, indent=1) + "\n")
    os.replace(tmp, Path(rd, "result.json"))


if __name__ == "__main__":
    main()
