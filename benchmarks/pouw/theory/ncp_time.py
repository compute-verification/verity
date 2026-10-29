#!/usr/bin/env python3
"""NCP's time bound on the H200 (the PoUW core's T1): how fast each sm_90 pipe makes independent k32 checkpoint words,
the honest prover's schedule over every pipe, the adversary's best, and the gap.

A word is a dot product's k32 running sum over NCP's 3K-wide stacked product (int8 operands: int7 data plus noise, block 3
up to ±127). One accumulator register of wgmma s8 m64nNk32 is one word. A lane route makes a word from 32 exact int8
products plus the instructions that accumulate and extract it. `ROUTES` gives each route's lane instructions a word in
three readings: `products`, the product instructions alone (the premise P32 and nothing else: the adversary's weakest
bound), `ideal`, adding the word's own extraction (no operand delivery), and `honest`, adding operand delivery and
conversion (an estimate until the honest kernel is timed). The `delivered` adversary pays honest's costs, so its rate is
honest's. A lane instruction's realistic product count: IDP.4A 4 (byte operands: a sum of two NCP entries leaves the
byte, so no fast algorithm fits), DFMA 2 (two products packed 2^24 apart in the 53-bit mantissa), IMAD 2 (16-bit fields,
a short chain), FFMA 1, HFMA2 1/2 (an int8 product is four nibble products and FP16 holds 11 bits). The tensor pipe's
other forms (`TENSOR_FORMS`) make fewer words a tensor slot than s8 wgmma and share its slots.

Time is the polytope the H200 census measured (`machine_model/data/devices/sm_90/rows.csv`). Per SMSP-cycle, the
tensor pipe's IGMMA_S8_N256 rate g and each lane class's warp instructions n_c satisfy the joint rows
  JI90_k: a_k g + b_k Σ_c n_c ≤ 1,   JPI90_k: a g + b n_IMAD ≤ 1,   JPA90_k: a g + b n_ALU ≤ 1,
each pipe's alone ceiling, and for honest the shared-memory port beside SS wgmma (`--no-smem` drops it; the adversary's
ideal routes read no operands at all). Two placements are estimates, not
measurements: IDP.4A on the IMAD pipe at IMAD's cost, and DFMA under the generic JI90 rows only (`ESTIMATES`; the
`wgU_idp4a`, `wgU_dfma` and `p_*` rows of dc/probes measure them). The LP maximises words a SMSP-cycle,
4096 g + Σ_r w_r, over the routes a party runs.

T1 reads word production with hashing excluded (ShaSpeed: both parties hash the same bytes). The final word of each dot
product is A·Bᵀ, which the adversary holds as advice, so it makes 3K/32 − 1 of honest's 3K/32 words:
  t_adv / t_H = (1 − 32/(3K)) · R_H / R_A.

  uv run --with scipy==1.18.1 python benchmarks/pouw/theory/ncp_time.py [--no-smem] [--json OUT] [--mutations]
      [--verdict PATH]

(scipy at machine_model's pin: the LP's vertex is that version's.)
"""
from __future__ import annotations

import csv
import math
import json
import re
import sys
from pathlib import Path

import numpy as np
from exact_kit.checks import Checks
from scipy.optimize import linprog

ROWS = Path(__file__).resolve().parents[1] / "machine_model" / "data" / "devices" / "sm_90" / "rows.csv"
WORDS_PER_IGMMA = 4096
LANES = 32
G_SAT = 0.0312468 / 4
SMEM_SHARED = 123.0 / 4
SS_BYTES_PER_IGMMA = 2560
KS = (1067, 2048, 4096, 8192)

CLASSES = ("IMAD", "ALU", "FMA", "FP64", "MIO", "UNI")
#: Each class's warp instructions a SMSP-cycle alone: the census (IMAD, ALU, FFMA at issue, uniform), the sheet (FP64,
#: 64 lanes a SM-clock) and LDS.128's 127.96 B a SM-clock (a 512-byte warp instruction every 16 SMSP-cycles).
ALONE = {"IMAD": 0.5, "ALU": 0.5, "FMA": 1.0, "FP64": 0.5, "MIO": 1 / 16, "UNI": 0.5}
ESTIMATES = {
    "IDP4A on IMAD": "IDP.4A issues to the IMAD pipe at IMAD's cost (dc/probes p_idp4a_imad, wgU_idp4a measure it)",
    "DFMA beside": "DFMA beside a saturated wgmma holds only the generic JI90 rows and the FP64 pipe's sheet ceiling "
                   "0.5 a SMSP-cycle (wgU_dfma, p_dfma_* measure it)",
    "operand delivery": "honest routes load int8 operands from shared memory in a register tile (8x8 for IDP.4A, "
                        "FFMA, IMAD and HFMA2; 16 rows as 8 packed pairs x 8 for DFMA) and convert them in registers",
    "smem port": "beside SS wgmma s8 n256 (80 B a SM-clock of operand reads) the port serves 123 B a SM-clock in all "
                 "(the n128 joint, RT-H200-EMP)",
}
INSTR = {
    "IDP4A": ("IMAD", 1), "IMAD": ("IMAD", 1), "IADD3": ("ALU", 1), "FFMA": ("FMA", 1), "FADD": ("FMA", 1),
    "HFMA2": ("FMA", 1), "DFMA": ("FP64", 1), "DADD": ("FP64", 1), "LDS128": ("MIO", 1), "UIMAD": ("UNI", 1),
}
#: Each route's lane instructions a word: `products` the product instructions alone, `ideal` adding the word's own
#: extraction, `honest` adding operand delivery and conversion (an estimate), and `bytes` the shared-memory bytes an honest
#: word reads.
ROUTES = {
    "idp4a": {"products": {"IDP4A": 8}, "ideal": {"IDP4A": 8}, "honest": {"IDP4A": 8, "LDS128": 0.5}, "bytes": 8,
              "why": "8 IDP.4A a k32 word, the accumulator chained; an 8x8 tile loads 16 operand words a k4 step"},
    "dfma2": {"products": {"DFMA": 16}, "ideal": {"DFMA": 16, "DADD": 0.5, "IADD3": 3},
              "honest": {"DFMA": 16, "DADD": 4.5, "IADD3": 8.5, "LDS128": 0.375}, "bytes": 6,
              "why": "two words a DFMA chain, (x1 + 2^24 x2)·y over k32 (|d| < 2^44, exact in 53 bits); a pair splits "
                     "by a magic-number DADD, a PRMT sign extension, a 64-bit subtract and a funnel shift, then two "
                     "IADD3 into the running words; honest packs each x pair by one PRMT (bytes 0 and 3) and each y by "
                     "one, flips sign bits by a LOP3 a register of four, and makes each double by a magic-number DADD, "
                     "over a 16x8 tile (8 packed pairs by 8): 4 DADD and 5.5 ALU a word"},
    "ffma": {"products": {"FFMA": 32}, "ideal": {"FFMA": 32, "FADD": 1, "IADD3": 1},
             "honest": {"FFMA": 32, "FADD": 9, "IADD3": 9, "LDS128": 0.5},
             "bytes": 8, "why": "32 FFMA (a k32 increment < 2^19 is exact in FP32), a magic-number FADD and an IADD3 "
                                "into the running word; honest converts each int8 operand by the magic number"},
    "imad2": {"products": {"IMAD": 16}, "ideal": {"IMAD": 16, "IADD3": 8},
              "honest": {"IMAD": 16, "IADD3": 12, "LDS128": 0.5}, "bytes": 8,
              "why": "two 16-bit fields an IMAD, (x1 + 2^16 x2)·y, split every two terms"},
    "hfma2": {"products": {"HFMA2": 64}, "ideal": {"HFMA2": 64, "IADD3": 4},
              "honest": {"HFMA2": 64, "IADD3": 12, "LDS128": 0.5}, "bytes": 8,
              "why": "four nibble products an int8 product, two an HFMA2, flushed before FP16's 2048"},
    "uimad": {"products": {"UIMAD": 32 * LANES}, "ideal": {"UIMAD": 32 * LANES}, "honest": {"UIMAD": 32 * LANES},
              "bytes": 0, "why": "a warp-uniform s32 MAC makes one product a warp instruction"},
}
#: The tensor pipe's other exact forms, alone, in k32 words a SM-clock: each shares the s8 wgmma's slots (measured for
#: mma.sync and DMMA; b1, f16 and sparse s8 an estimate), so beside a saturated s8 wgmma each adds none.
TENSOR_FORMS = {
    "wgmma s8 m64n256k32 (IGMMA)": (127.99, "measured: rho_T, the census's M row"),
    "mma.sync s8 m16n8k32 (IMMA.16832)": (0.656029 * 4096 / 32, "measured: 65.6% of wgmma; shares it, t + s <= 1"),
    "mma.sync s8 m16n8k16 (IMMA.16816)": (0.974476 * 2048 / 64, "measured: k16 sums, two a k32 word; shares it"),
    "b1 wgmma m64n256k256 .and.popc (BGMMA)": (32764.6 / (64 * 32), "measured alone at IGMMA's instruction rate (8x its "
                                                "MACs); an 8-bit by 8-bit word is >= 64 bit-plane dot products, so <= "
                                                "1/8 of s8's words a slot; the joint with IGMMA is an estimate"),
    "f16 wgmma, f32 accumulate (HGMMA)": (2048 / 32, "data sheet: exact for int8 operands (k32 sums < 2^19); half s8's "
                                          "rate; sharing is an estimate"),
    "s8 2:4 sparse wgmma m64n256k64 (IGMMA.SP)": (4096 / 32, "data sheet: 2x logical MACs on half-zero operands, so a "
                                                  "dense k32 word a slot, as dense; untimed"),
    "DMMA m16n8k16 (f64)": (128 / 32, "measured: DMMA-SHARES-TENSOR (t + s 0.935-0.958)"),
}


def joint_rows(path: Path = ROWS) -> dict[str, tuple[str, float, float]]:
    """The census's joint rows beside IGMMA_S8_N256: row → (class, a, b), a SMSP-cycles an IGMMA and b an other."""
    kinds = {"every other warp instruction": "ALL", "IMAD": "IMAD", "ALU": "ALU", "SHFL": "SHFL"}
    pat = re.compile(r"IGMMA_S8_N256 ([\d.]+) \+ (every other warp instruction|IMAD|ALU|SHFL) ([\d.]+) SMSP-cycles")
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            m = pat.search(row["description"])
            if row["status"] == "joint" and m:
                out[row["row"]] = (kinds[m.group(2)], float(m.group(1)), float(m.group(3)))
    return out


def solve(routes: dict[str, dict], rows: dict, smem: bool = False, tensor: bool = True, scale: dict | None = None,
          free: tuple[str, ...] = ()):
    """Max words a SMSP-cycle. routes: name → {INSTR: lane instructions a word}; scale: name → a factor on the route's
    instructions and bytes; free: classes the generic joint rows (JI90) don't count, though each still takes an issue
    slot. Returns (words/SMSP-cycle, g, {r: w})."""
    names = list(routes)
    nv = 1 + len(names)
    use = np.zeros((len(CLASSES), nv))
    for j, r in enumerate(names, start=1):
        for ins, cnt in routes[r].items():
            cls, w = INSTR[ins]
            use[CLASSES.index(cls), j] += cnt * w / LANES * (scale or {}).get(r, 1.0)
    A, b = [], []
    allv = use.sum(0)
    generic = use[[i for i, c in enumerate(CLASSES) if c not in free]].sum(0)
    for name, (cls, a, coef) in rows.items():
        if cls == "SHFL":
            continue
        line = np.zeros(nv)
        line[0] = a
        line += coef * (generic if cls == "ALL" else use[CLASSES.index(cls)])
        A.append(line)
        b.append(1.0)
    for i, cls in enumerate(CLASSES):
        A.append(use[i] / ALONE[cls])
        b.append(1.0)
    line = allv.copy()
    line[0] = 1.0
    A.append(line)
    b.append(1.0)
    if smem:
        line = np.zeros(nv)
        line[0] = SS_BYTES_PER_IGMMA
        for j, r in enumerate(names, start=1):
            base, _, kind = r.partition(":")
            line[j] = ROUTES[base]["bytes"] * (scale or {}).get(r, 1.0) if kind == "honest" else 0
        A.append(line)
        b.append(SMEM_SHARED)
    c = np.zeros(nv)
    c[0] = -WORDS_PER_IGMMA if tensor else 0
    c[1:] = -1.0
    bounds = [(0, G_SAT if tensor else 0)] + [(0, None)] * len(names)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
    assert res.status == 0, res.message
    return -res.fun, res.x[0], dict(zip(names, res.x[1:]))


def reading(kind: str, names) -> dict[str, dict]:
    return {f"{r}:{kind}": ROUTES[r][kind] for r in names}


def gap(r_h: float, r_a: float, k: int) -> float:
    return 1 - (1 - 32 / (3 * k)) * r_h / r_a


def kmin(r_h: float, r_a: float) -> float:
    """The least K at which T1 holds, (1 − 32/(3K)) r_h / r_a ≥ 0.99."""
    x = 1 - 0.99 * r_a / r_h
    return math.inf if x <= 0 else 32 / (3 * x)


def bisect(f, lo: float, hi: float) -> float:
    """The crossing of a predicate true below it and false above it."""
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) else (lo, mid)
    return hi


def main() -> int:
    checks = Checks()
    smem = "--no-smem" not in checks.args
    out = checks.args[checks.args.index("--json") + 1] if "--json" in checks.args else None
    rows = joint_rows()
    checks.check("the census's H200 joint rows parse (JI90_0-2, JPI90_0-1, JPA90_0-1)",
                 {"JI90_0", "JI90_1", "JI90_2", "JPI90_0", "JPI90_1", "JPA90_0", "JPA90_1"} <= set(rows),
                 ", ".join(f"{k} {v[1]}+{v[2]}" for k, v in sorted(rows.items())))
    sm = 4
    t_only, _, _ = solve({}, rows)
    checks.check("tensor alone: 128 words a SM-clock (rho_T 127.99, IGMMA_S8_N256)", abs(sm * t_only - 128) < 0.1,
                 f"{sm * t_only:.2f}")

    # The generous reading: every lane instruction beside the tensor writes one word (one dimension a register).
    generous, _, _ = solve({f"reg{c}:ideal": {ins: 1} for c, ins in enumerate(("IMAD", "IADD3", "FFMA", "DFMA"))}, rows)
    checks.check("generous model (a word a lane register): the side makes about half the tensor's words beside it, "
                 "a gap near 1/3", 0.30 < 1 - t_only / generous < 0.36,
                 f"{sm * generous:.1f} words/SM-clk, gap {1 - t_only / generous:.1%}")

    # (1) Each pipe's words a SM-clock: alone (products reading) and beside a saturated s8 wgmma in each reading.
    kinds = ("products", "ideal", "honest")
    per_route = {}
    print("  words/SM-clk        alone   beside: products  ideal  honest")
    for r in ROUTES:
        alone = sm * solve(reading("products", [r]), rows, tensor=False)[0]
        beside = {}
        for kind in kinds:
            v, g, _ = solve(reading(kind, [r]), rows, smem=smem)
            beside[kind] = sm * (v - WORDS_PER_IGMMA * g)
        per_route[r] = {"alone": alone, "beside": beside}
        print(f"  {r:16s} {alone:8.2f}          " + "  ".join(f"{beside[k]:6.2f}" for k in kinds))
    for name, (w, why) in TENSOR_FORMS.items():
        print(f"  tensor {name:42s} {w:7.2f} alone ({why})")
    pr = {r: v["beside"]["products"] for r, v in per_route.items()}
    checks.check("beside a saturated s8 wgmma (products reading): IDP.4A leads, then DFMA; IMAD and FFMA trail, HFMA2 "
                 "and the uniform path last", pr["idp4a"] > pr["dfma2"] > max(pr["imad2"], pr["ffma"])
                 > pr["hfma2"] > pr["uimad"], ", ".join(f"{r} {v:.2f}" for r, v in pr.items()))
    checks.check("no other tensor form makes more words a tensor slot than s8 wgmma",
                 all(w <= 128.0 + 1e-9 for w, _ in TENSOR_FORMS.values()),
                 ", ".join(f"{n.split(' (')[0]} {w:.1f}" for n, (w, _) in TENSOR_FORMS.items()))

    # (2) Honest's schedule, by the pipes it uses.
    honest_sets = {"tensor only": [], "tensor + idp4a": ["idp4a"], "tensor + idp4a + dfma2": ["idp4a", "dfma2"],
                   "every route": list(ROUTES)}
    table = {}
    for name, rs in honest_sets.items():
        r, g, w = solve(reading("honest", rs), rows, smem=smem)
        table[name] = {"rate": r, "words_sm_clk": sm * r, "g": g,
                       "routes": {k.split(":")[0]: sm * v for k, v in w.items() if v > 1e-12}}
    r_h = table["every route"]["rate"]
    side_h = r_h - WORDS_PER_IGMMA * table["every route"]["g"]

    # (3) The adversary's best in each reading, against honest on every route.
    adv = {}
    for kind in ("products", "ideal", "delivered"):
        r_a, g_a, w_a = solve(reading("honest" if kind == "delivered" else kind, ROUTES), rows, smem=smem)
        adv[kind] = {"rate": r_a, "words_sm_clk": sm * r_a, "g": g_a, "side": sm * (r_a - WORDS_PER_IGMMA * g_a),
                     "routes": {k.split(":")[0]: sm * v for k, v in w_a.items() if v > 1e-12},
                     "kmin": kmin(r_h, r_a)}
    for name, row in table.items():
        print(f"  honest {name:24s} {row['words_sm_clk']:8.2f} words/SM-clk ("
              + ", ".join(f"{k} {v:.2f}" for k, v in row["routes"].items()) + ")")
        for kind, a in adv.items():
            print(f"    vs {kind:9s} adversary {a['words_sm_clk']:7.2f}: gap "
                  + "  ".join(f"K{k} {gap(row['rate'], a['rate'], k):6.2%}" for k in KS)
                  + f"  T1 from K {kmin(row['rate'], a['rate']):,.0f}")
    checks.check("every adversary reading keeps the tensor saturated, and its side adds 2-8% of the tensor's words",
                 all(abs(a["g"] - G_SAT) < 1e-9 and 0.02 < a["side"] / 128 < 0.08 for a in adv.values()),
                 "; ".join(f"{k} {a['words_sm_clk']:.2f} = tensor 127.99 + side {a['side']:.2f} ("
                           + ", ".join(f"{r} {v:.2f}" for r, v in a["routes"].items()) + ")" for k, a in adv.items()))
    tonly = table["tensor only"]["rate"]
    checks.check("honest on the tensor alone misses T1 at every K in every reading: the side pipes' words are the gap",
                 all(gap(tonly, a["rate"], k) > 0.01 for a in adv.values() for k in KS),
                 ", ".join(f"{k} K 2,048 {gap(tonly, a['rate'], 2048):.2%}" for k, a in adv.items()))
    checks.check("against the delivered adversary (honest's own routes) the gap is the advice word alone, 32/(3K): T1 "
                 "from K 1,067", all(abs(gap(r_h, adv["delivered"]["rate"], k) - 32 / (3 * k)) < 1e-9 for k in KS)
                 and abs(adv["delivered"]["kmin"] - 32 / 0.03) < 1e-6, f"K 2,048 {32 / 6144:.2%}")
    km = {k: a["kmin"] for k, a in adv.items()}
    checks.check("the product-count premise alone does not give T1 at K 2,048: against the products adversary honest's "
                 "side falls short (estimate: honest's delivery is modelled, not timed)",
                 gap(r_h, adv["products"]["rate"], 2048) > 0.01 and km["delivered"] < km["ideal"] < km["products"],
                 f"gap at K 2,048 {gap(r_h, adv['products']['rate'], 2048):.2%} (products), "
                 f"{gap(r_h, adv['ideal']['rate'], 2048):.2%} (ideal); T1 from K " +
                 ", ".join(f"{v:,.0f} ({k})" for k, v in km.items()))

    # (4) Margins: how much cheaper than its reading an adversary's side word may be with T1 holding at K, and the side
    # honest must reach at K 2,048.
    def gap_scaled(kind, s, k, only=None):
        base = "honest" if kind == "delivered" else kind
        sc = {f"{r}:{base}": (s if only in (None, r) else 1.0) for r in ROUTES}
        return gap(r_h, solve(reading(base, ROUTES), rows, smem=smem, scale=sc)[0], k)

    margins = {}
    for kind, k in (("delivered", 2048), ("ideal", 4096), ("products", 8192), ("products", 4096)):
        s = bisect(lambda s: gap_scaled(kind, s, k) > 0.01, 0.05, 8.0)
        s_idp = bisect(lambda s: gap_scaled(kind, s, k, "idp4a") > 0.01, 0.05, 8.0)
        margins[f"{kind}@{k}"] = {"every_route": s, "idp4a": s_idp}
    need = {}
    for kind in ("products", "ideal"):
        r_need = 0.99 * adv[kind]["rate"] / (1 - 32 / (3 * 2048))
        need[kind] = sm * (r_need - WORDS_PER_IGMMA * G_SAT)
    for name, m in margins.items():
        print(f"  margin {name:14s}: T1 holds while every side word costs >= {m['every_route']:.3f}x the reading's "
              f"instructions (IDP.4A alone: >= {8 * m['idp4a']:.2f} a word)")
    print(f"  honest side at K 2,048: modelled {sm * side_h:.2f} words/SM-clk; T1 needs >= {need['products']:.2f} "
          f"(products) or >= {need['ideal']:.2f} (ideal)")
    # The placement estimates: were a class's warp instructions free of the generic rows beside wgmma (both parties).
    placement = {}
    for cls in ("FP64", "IMAD"):
        rh_f = solve(reading("honest", ROUTES), rows, smem=smem, free=(cls,))[0]
        placement[cls] = {"honest": sm * rh_f}
        for kind in ("products", "ideal"):
            ra_f = solve(reading(kind, ROUTES), rows, smem=smem, free=(cls,))[0]
            placement[cls][kind] = {"adversary": sm * ra_f, "gap_k2048": gap(rh_f, ra_f, 2048), "kmin": kmin(rh_f, ra_f)}
        print(f"  were {cls} free of JI90 beside wgmma: honest {sm * rh_f:.2f}, gap at K 2,048 "
              + ", ".join(f"{placement[cls][k]['gap_k2048']:.2%} ({k}, T1 from K {placement[cls][k]['kmin']:,.0f})"
                          for k in ("products", "ideal")))
    m8 = margins["products@8192"]["every_route"]
    checks.check("premise P32's margin at K 8,192: T1 holds while an off-tensor word costs at least that fraction of "
                 "P32's product instructions", m8 < 1.0, f"{m8:.3f}x, i.e. >= {32 * m8:.1f} products a word "
                 f"(margin {1 / m8:.2f}x)")
    md = margins["delivered@2048"]["every_route"]
    checks.check("at K 2,048 T1 needs the delivered reading: it holds while an adversary's side word costs at least "
                 "that fraction of honest's instructions", 0.5 < md < 1.0, f"{md:.3f}x (margin {1 / md:.2f}x)")

    if checks.mutations:
        no_imad = {k: v for k, v in rows.items() if not k.startswith("JPI90")}
        ra2 = solve(reading("products", ROUTES), no_imad, smem=smem)[0]
        checks.mutation("dropping the IMAD pipe's joint rows (JPI90) raises the products adversary's rate: the row binds",
                        ra2 > adv["products"]["rate"] * 1.001, f"{sm * ra2:.2f} vs {adv['products']['words_sm_clk']:.2f}")
        g2 = gap_scaled("products", 0.5, 8192, "idp4a")
        checks.mutation("an adversary at 4 IDP.4A a word (2x the premise) breaks T1 at K 8,192", g2 > 0.01,
                        f"gap {g2:.2%}")
        free_b1 = gap(r_h, adv["products"]["rate"] + TENSOR_FORMS["b1 wgmma m64n256k256 .and.popc (BGMMA)"][0] / sm,
                      8192)
        checks.mutation("a b1 pipe beside the s8 wgmma (not sharing its slots) breaks T1 at every K", free_b1 > 0.01,
                        f"gap at K 8,192 {free_b1:.2%}")
    if out:
        Path(out).write_text(json.dumps({
            "smem": smem, "per_route": per_route, "tensor_forms": TENSOR_FORMS, "generous_words_sm_clk": sm * generous,
            "honest": {k: {kk: vv for kk, vv in v.items() if kk != "rate"} for k, v in table.items()},
            "adversary": {k: {kk: vv for kk, vv in v.items() if kk != "rate"} for k, v in adv.items()},
            "gap": {kind: {k: gap(r_h, a["rate"], k) for k in KS} for kind, a in adv.items()},
            "margins": margins, "honest_side_need_k2048": need, "honest_side": sm * side_h, "placement": placement,
            "estimates": ESTIMATES, "routes": ROUTES}, indent=1, default=str) + "\n")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
