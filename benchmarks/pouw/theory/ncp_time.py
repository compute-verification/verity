#!/usr/bin/env python3
"""NCP's time bound on the H200 (the PoUW core's T1): how fast each sm_90 pipe makes independent k32 checkpoint words,
the honest prover's schedule over every pipe, the adversary's best, and the gap.

A word is a dot product's k32 running sum over NCP's 3K-wide stacked product (int8 operands: int7 data plus noise, block 3
up to ±127). One accumulator register of wgmma s8 m64nNk32 is one word. A lane route makes a word from 32 exact int8
products plus the instructions that accumulate and extract it. `ROUTES` gives each route's lane instructions a word in
three readings: `products`, the product instructions alone at 4 a IDP.4A, 2 a DFMA (packed 2^24 apart), 2 an IMAD, 1 a
FFMA and 1/2 a HFMA2; `ideal`, adding the word's own extraction (no operand delivery); and `honest`, adding operand
delivery and conversion (an estimate until the honest kernel is timed). The `floor` reading is `ncp_p32_check.py`'s:
the fewest issue slots a word that check finds, with three products a DFMA in chains of 8 and Strassen inside the k32
step, and it sits under the products reading on DFMA and FFMA. The `slot2` reading is the rank-1 capacity bound that no
CUDA instruction buys more than 2 MACs an issue slot (IDP.4A's rate, measured on sm_120). The `delivered` adversary pays
honest's costs, so its rate is honest's. IDP.4A's 8 a word holds in every reading: a sum of two NCP entries leaves the
byte. The tensor pipe's other forms (`TENSOR_FORMS`) make fewer words a tensor slot than s8 wgmma and share its slots.

Time is the polytope the H200 census measured (`machine_model/data/devices/sm_90/rows.csv`). Per SMSP-cycle, the
tensor pipe's IGMMA_S8_N256 rate g and each lane class's warp instructions n_c satisfy the joint rows
  JI90_k: a_k g + b_k Σ_c n_c ≤ 1,   JPI90_k: a g + b n_IMAD ≤ 1,   JPA90_k: a g + b n_ALU ≤ 1,
each pipe's alone ceiling, the FP64 pipe's share of the tensor pipe (`FP64_ROW`: DFMA beside a saturated wgmma, dc/probes'
wgU_dfma), and for honest the shared-memory port beside SS wgmma (`--no-smem` drops it; the adversary's ideal routes read
no operands at all). IDP.4A is the IMAD class (p_idp4a_imad). The LP maximises words a SMSP-cycle, 4096 g + Σ_r w_r,
over the routes a party runs.

T1 is per SM-clock (the clock is a term of its own) and reads word production with hashing excluded (ShaSpeed: both
parties hash the same bytes). The final word of each dot product is A·Bᵀ, which the adversary holds as advice, so it
makes 3K/32 − 1 of honest's 3K/32 words:
  t_adv / t_H = (1 − 32/(3K)) · R_H / R_A.
Forming X and Y is charged to honest alone, two ways: the certificate's additive 8/n + 8/m on t_H, and `FORM_LANE`'s
instructions and bytes beside the wgmma.

`--coissue ROWS` (repeatable; a dc/probes `run.sh sm90` rows.jsonl) reads measured co-issue points beside the wgmma:
each must lie inside the polytope, honest's rate becomes the best words any point reaches on honest's routes (the
polytope's honest schedule is what no probe has yet reached), and the adversary is priced both on the polytope (the
bound) and on the measured points (not a bound).

  uv run --with scipy==1.18.1 python benchmarks/pouw/theory/ncp_time.py [--no-smem] [--json OUT] [--mutations]
      [--verdict PATH] [--coissue ROWS ...]

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
import ncp_p32_check as p32
from exact_kit.checks import Checks
from scipy.optimize import linprog

ROWS = Path(__file__).resolve().parents[1] / "machine_model" / "data" / "devices" / "sm_90" / "rows.csv"
WORDS_PER_IGMMA = 4096
LANES = 32
G_SAT = 0.0312468 / 4
SMEM_SHARED = 123.0 / 4
SS_BYTES_PER_IGMMA = 2560
KS = (1067, 2048, 4096, 8192, 16384)

CLASSES = ("IMAD", "ALU", "FMA", "FP64", "MIO", "UNI")
#: Each class's warp instructions a SMSP-cycle alone: the census (IMAD, ALU, FFMA at issue, uniform), the sheet (FP64,
#: 64 lanes a SM-clock) and LDS.128's 127.96 B a SM-clock (a 512-byte warp instruction every 16 SMSP-cycles).
ALONE = {"IMAD": 0.5, "ALU": 0.5, "FMA": 1.0, "FP64": 0.5, "MIO": 1 / 16, "UNI": 0.5}
#: FP64 shares the tensor pipe: DFMA beside a saturated s8 wgmma (dc/probes wgU_dfma, wgU_dfma_ffma, wgU_idp4a_dfma)
#: keeps under 1% of its alone rate and slows the wgmma, as DMMA does, so g/G_SAT + n_FP64/ALONE_FP64 ≤ 1. IDP.4A shares
#: the IMAD pipe (p_idp4a_imad), so it is IMAD's class.
FP64_ROW = "tensor share + FP64 share <= 1"
ESTIMATES = {
    "IDP4A beside": "IDP.4A beside a saturated wgmma at the IMAD class's joint rows (JPI90; wgU_idp4a sits under them)",
    "FP64_ROW between points": "t + s <= 1 between the alone rates and the probed point beside the saturated wgmma",
    "operand delivery": "honest routes load int8 operands from shared memory in a register tile (8x8 for IDP.4A, "
                        "FFMA, IMAD and HFMA2; 16 rows as 8 packed pairs x 8 for DFMA) and convert them in registers",
    "smem port": "beside SS wgmma s8 n256 (80 B a SM-clock of operand reads) the port serves 123 B a SM-clock in all "
                 "(the n128 joint, RT-H200-EMP)",
    "forming": "FORM_LANE's issue and port cost of an entry beside the wgmma; the H200's forming price is unmeasured",
}
INSTR = {
    "IDP4A": ("IMAD", 1), "IMAD": ("IMAD", 1), "IADD3": ("ALU", 1), "PRMT": ("ALU", 1), "FFMA": ("FMA", 1),
    "FADD": ("FMA", 1), "HFMA2": ("FMA", 1), "DFMA": ("FP64", 1), "DADD": ("FP64", 1), "LDS128": ("MIO", 1),
    "STS128": ("MIO", 1), "UIMAD": ("UNI", 1), "LOP3": ("ALU", 1),
}
#: dc/probes' co-issue points (`--coissue`, a `run.sh sm90` rows.jsonl): the tensor role alone (wg_, wg1_) and beside lane
#: warps (wgU_ one a SMSP, wgV_ and wgW_ two), their lanes' SASS patterns by INSTR name. An n128 IGMMA is half an n256.
COISSUE = re.compile(r"wg1?_s8_n256_a1|wg[UVW]_[a-z0-9_]+")
PATTERNS = {"IDP.4A*": "IDP4A", "FFMA": "FFMA", "LOP3.LUT": "LOP3", "DFMA": "DFMA", "HFMA2*": "HFMA2", "IMAD": "IMAD",
            "LDS.128": "LDS128"}
#: The route a measured lane class makes words on.
WORD_ROUTE = {"IMAD": "idp4a", "FMA": "ffma"}
#: Forming the operands: X's 3mK entries every call, and Y's 3Kn when F is fresh every call, so 32/n + 32/m entries a
#: word. `FORM_CERT` is NCP's certificate's forming charge on honest's work (24 units an entry of A against 32 a word,
#: 8/n; the same for Y, 8/m), at the 4090's W1 price. `FORM_LANE` is an H200 estimate beside the wgmma: a biased packed
#: IADD3 forms 4 entries, the permutation and negation take a PRMT or LOP3 per 6, and an STS.128 stores 16 into the
#: wgmma's tiles, one shared-memory byte an entry (operands arrive by LDG; the noise's SHA-512 is hashing, outside T1).
FORM_CERT = 8.0
FORM_LANE = {"IADD3": 1 / 4, "PRMT": 1 / 6, "STS128": 1 / 16}
FORM_BYTES = 1.0
#: (n, m): m None when F is reused across calls and Y is formed once; m < 64 is below wgmma's tight N.
FORM_SHAPES = ((2048, None), (2048, 8192), (2048, 512), (2048, 64), (2048, 16))
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


def _floor_reading() -> None:
    fl = p32.floors(*p32.blocks())
    d, f, m = fl["dfma"], fl["ffma"], fl["imad"]
    floor = {"idp4a": {"IDP4A": fl["idp4a"]["slots"]},
             "dfma2": {"DFMA": d["products"], "DADD": d["combine"], "IADD3": d["extract"]},
             "ffma": {"FFMA": f["products"], "FADD": f["combine"], "IADD3": f["extract"]},
             "imad2": {"IMAD": m["products"], "IADD3": m["combine"] + m["extract"]},
             "hfma2": {"HFMA2": fl["hfma2"]["slots"]}, "uimad": ROUTES["uimad"]["products"]}
    for r, v in floor.items():
        ROUTES[r]["floor"] = v


def _slot_reading() -> None:
    """The rank-1 capacity bound that no CUDA instruction buys more than 2 MACs an issue slot (IDP.4A's; measured on
    sm_120): a class issuing a warp instruction every 1/ALONE SMSP-cycles makes at most 2/ALONE products an instruction,
    so a word takes at least 16·ALONE. The uniform path keeps its products reading."""
    for v in ROUTES.values():
        (ins, _), = v["products"].items()
        cls = INSTR[ins][0]
        v["slot2"] = v["products"] if cls == "UNI" else {ins: 16 * ALONE[cls]}


_floor_reading()
_slot_reading()
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
          free: tuple[str, ...] = (), form: tuple[dict, float] | None = None, fp64_shares: bool = True):
    """Max words a SMSP-cycle. routes: name → {INSTR: lane instructions a word}; scale: name → a factor on the route's
    instructions and bytes; free: classes the generic joint rows (JI90) don't count, though each still takes an issue
    slot; form: (lane instructions, shared-memory bytes) every word's operand forming adds, the tensor's words included;
    fp64_shares: the FP64 pipe shares the tensor pipe's slots (FP64_ROW), else only the generic rows bind it.
    Returns (words/SMSP-cycle, g, {r: w})."""
    names = list(routes)
    nv = 1 + len(names)
    use = np.zeros((len(CLASSES), nv))
    for j, r in enumerate(names, start=1):
        for ins, cnt in routes[r].items():
            cls, w = INSTR[ins]
            use[CLASSES.index(cls), j] += cnt * w / LANES * (scale or {}).get(r, 1.0)
    form_ins, form_bytes = form or ({}, 0.0)
    for ins, cnt in form_ins.items():
        cls, w = INSTR[ins]
        use[CLASSES.index(cls), 0] += WORDS_PER_IGMMA * cnt * w / LANES
        use[CLASSES.index(cls), 1:] += cnt * w / LANES
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
    if fp64_shares:
        line = use[CLASSES.index("FP64")] / ALONE["FP64"]
        line[0] = 1 / G_SAT
        A.append(line)
        b.append(1.0)
    line = allv.copy()
    line[0] = 1.0
    A.append(line)
    b.append(1.0)
    if smem:
        line = np.zeros(nv)
        line[0] = SS_BYTES_PER_IGMMA + WORDS_PER_IGMMA * form_bytes
        for j, r in enumerate(names, start=1):
            base, _, kind = r.partition(":")
            line[j] = (ROUTES[base]["bytes"] * (scale or {}).get(r, 1.0) if kind == "honest" else 0) + form_bytes
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


def forming(n: int, m: int | None) -> tuple[float, tuple[dict, float]]:
    """Entries formed a word at (n, m), and FORM_LANE's (instructions, bytes) a word for `solve`'s form."""
    e = 32 / n + (32 / m if m else 0)
    return e, ({ins: c * e for ins, c in FORM_LANE.items()}, FORM_BYTES * e)


def kmin(r_h: float, r_a: float) -> float:
    """The least K at which T1 holds, (1 − 32/(3K)) r_h / r_a ≥ 0.99."""
    x = 1 - 0.99 * r_a / r_h
    return math.inf if x <= 0 else 32 / (3 * x)


def coissue_points(paths) -> list[dict]:
    """The co-issue points of dc/probes rows files: each probe's tensor share t = g/G_SAT, its IGMMA_S8_N256 rate g and
    its lanes' warp instructions a SMSP-cycle by INSTR name (a probe with any other lane pattern is skipped)."""
    pts = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            name = r.get("probe") or r.get("name") or ""
            if not COISSUE.fullmatch(name):
                continue
            g, lanes = 0.0, {}
            for key, v in r["rates"].items():
                pat = key.split(":", 1)[1]
                if pat.startswith("IGMMA"):
                    g += v / 4 * (0.5 if pat.startswith("IGMMA.64x128x32") else 1.0)
                elif pat in PATTERNS:
                    lanes[PATTERNS[pat]] = lanes.get(PATTERNS[pat], 0.0) + v / 4
                else:
                    break
            else:
                pts.append({"probe": name, "t": g / G_SAT, "g": g, "lanes": lanes})
    return pts


def usage(p: dict, rows: dict) -> dict[str, float]:
    """Each polytope row's use at a measured point (1 is its bound)."""
    n = dict.fromkeys(CLASSES, 0.0)
    for ins, v in p["lanes"].items():
        n[INSTR[ins][0]] += v
    tot = sum(n.values())
    u = {name: a * p["g"] + coef * (tot if cls == "ALL" else n[cls]) for name, (cls, a, coef) in rows.items()
         if cls != "SHFL"}
    u.update({f"{c} alone": n[c] / ALONE[c] for c in CLASSES})
    if n["FP64"]:
        u["FP64_ROW"] = p["g"] / G_SAT + n["FP64"] / ALONE["FP64"]
    u["issue"] = p["g"] + tot
    return u


def words_at(p: dict, kind: str) -> float:
    """Words a SMSP-cycle at a measured point, each lane class running its route's `kind` reading and paying only that
    route's instructions in its own class (the rest free, so an upper estimate)."""
    w = WORDS_PER_IGMMA * p["g"]
    for ins, v in p["lanes"].items():
        cls = INSTR[ins][0]
        if cls in WORD_ROUTE:
            w += v * LANES / sum(c for i, c in ROUTES[WORD_ROUTE[cls]][kind].items() if INSTR[i][0] == cls)
    return w


def delivered(p: dict) -> bool:
    """Whether a measured point issues the LDS.128 its words' honest routes read their operands by."""
    need = 0.0
    for ins, v in p["lanes"].items():
        cls = INSTR[ins][0]
        if cls in WORD_ROUTE:
            route = ROUTES[WORD_ROUTE[cls]]["honest"]
            need += v / sum(c for i, c in route.items() if INSTR[i][0] == cls) * route.get("LDS128", 0.0)
    return p["lanes"].get("LDS128", 0.0) >= 0.99 * need


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
    kinds = ("products", "floor", "ideal", "honest")
    per_route = {}
    print("  words/SM-clk        alone   beside: products  floor  ideal  honest")
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
    checks.check("beside a saturated s8 wgmma (products reading): IDP.4A leads, FFMA and IMAD trail, HFMA2 and the "
                 "uniform path last, and DFMA makes none (the FP64 pipe shares the tensor's slots)",
                 pr["idp4a"] > max(pr["imad2"], pr["ffma"]) > pr["hfma2"] > pr["uimad"] > pr["dfma2"] == 0.0,
                 ", ".join(f"{r} {v:.2f}" for r, v in pr.items()))
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
    for kind in ("slot2", "products", "floor", "ideal", "delivered"):
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
    checks.check("the floor premise gives T1 at K 2,048 with honest's side on IDP.4A and FFMA beside the wgmma (honest's "
                 "schedule at the polytope's rates, its delivery modelled)",
                 gap(r_h, adv["floor"]["rate"], 2048) < 0.01
                 and km["delivered"] < km["ideal"] < km["products"] < km["floor"] < 2048,
                 ", ".join(f"{gap(r_h, adv[k]['rate'], 2048):.2%} ({k})" for k in ("products", "floor", "ideal"))
                 + " at K 2,048; T1 from K " + ", ".join(f"{v:,.0f} ({k})" for k, v in km.items()))
    checks.check("the rank-1 slot bound alone (no CUDA instruction above 2 MACs a slot) gives T1 at no K: at the bound "
                 "FFMA words outrun honest's side", km["slot2"] == math.inf,
                 f"adversary {adv['slot2']['words_sm_clk']:.2f} (side {adv['slot2']['side']:.2f}: "
                 + ", ".join(f"{r} {v:.2f}" for r, v in adv["slot2"]["routes"].items()) + f"); gap at K 2,048 "
                 f"{gap(r_h, adv['slot2']['rate'], 2048):.2%}, K 16,384 {gap(r_h, adv['slot2']['rate'], 16384):.2%}")

    # (4) Margins: how much cheaper than its reading an adversary's side word may be with T1 holding at K, and the side
    # honest must reach at K 2,048.
    def gap_scaled(kind, s, k, only=None):
        base = "honest" if kind == "delivered" else kind
        sc = {f"{r}:{base}": (s if only in (None, r) else 1.0) for r in ROUTES}
        return gap(r_h, solve(reading(base, ROUTES), rows, smem=smem, scale=sc)[0], k)

    margins = {}
    for kind, k in (("delivered", 2048), ("floor", 2048), ("products", 2048), ("floor", 8192), ("floor", 16384)):
        s = bisect(lambda s: gap_scaled(kind, s, k) > 0.01, 0.05, 8.0)
        s_idp = bisect(lambda s: gap_scaled(kind, s, k, "idp4a") > 0.01, 0.05, 8.0)
        margins[f"{kind}@{k}"] = {"every_route": s, "idp4a": s_idp}
    need = {}
    for kind in ("products", "floor", "ideal"):
        r_need = 0.99 * adv[kind]["rate"] / (1 - 32 / (3 * 2048))
        need[kind] = sm * (r_need - WORDS_PER_IGMMA * G_SAT)
    for name, m in margins.items():
        print(f"  margin {name:14s}: T1 holds while every side word costs >= {m['every_route']:.3f}x the reading's "
              f"instructions (IDP.4A alone: >= {8 * m['idp4a']:.2f} a word)")
    print(f"  honest side at K 2,048: modelled {sm * side_h:.2f} words/SM-clk; T1 needs >= "
          + ", ".join(f"{v:.2f} ({k})" for k, v in need.items()))
    # The placements, for both parties: FP64 under the generic rows only (the estimate the probe replaced), and the IMAD
    # class free of the generic rows beside wgmma.
    placement = {}
    for name, kw in (("FP64 under JI90 only", {"fp64_shares": False}), ("IMAD free of JI90", {"free": ("IMAD",)})):
        rh_f = solve(reading("honest", ROUTES), rows, smem=smem, **kw)[0]
        placement[name] = {"honest": sm * rh_f}
        for kind in ("products", "floor", "ideal"):
            ra_f = solve(reading(kind, ROUTES), rows, smem=smem, **kw)[0]
            placement[name][kind] = {"adversary": sm * ra_f, "gap_k2048": gap(rh_f, ra_f, 2048),
                                     "kmin": kmin(rh_f, ra_f)}
        print(f"  were {name}: honest {sm * rh_f:.2f}, gap at K 2,048 "
              + ", ".join(f"{placement[name][k]['gap_k2048']:.2%} ({k}, T1 from K {placement[name][k]['kmin']:,.0f})"
                          for k in ("products", "floor", "ideal")))
    # Forming, charged to honest alone: the adversary's tensor words need formed operands too, but nothing proved says so.
    form_table = {}
    print("  forming (honest only)       certificate 8/n + 8/m on t_H  |  H200 estimate beside the wgmma (FORM_LANE)")
    for n, m in FORM_SHAPES:
        e, form = forming(n, m)
        cert = FORM_CERT * e / 32
        rh_f = solve(reading("honest", ROUTES), rows, smem=smem, form=form)[0]
        row = {"entries_a_word": e, "certificate_charge": cert, "honest_words_sm_clk": sm * rh_f}
        for kind in ("floor", "products", "delivered"):
            ra = adv[kind]["rate"]
            row[kind] = {"certificate": {k: 1 - (1 - gap(r_h, ra, k)) / (1 + cert) for k in (2048, 8192)},
                         "beside": {k: gap(rh_f, ra, k) for k in (2048, 8192)},
                         "kmin_certificate": kmin(r_h / (1 + cert), ra), "kmin_beside": kmin(rh_f, ra)}
        form_table[f"n {n}, m {m or 'F reused'}"] = row
        print(f"  n {n:5d} m {str(m or 'F reused'):>8s}  +{cert:6.2%}: "
              + "  ".join(f"{k} K2048 {row[k]['certificate'][2048]:6.2%}" for k in ("floor", "delivered"))
              + f"  |  honest {sm * rh_f:6.2f}: "
              + "  ".join(f"{k} K2048 {row[k]['beside'][2048]:6.2%} (T1 from K {row[k]['kmin_beside']:,.0f})"
                          for k in ("floor", "delivered")))
    f0, f512 = form_table["n 2048, m F reused"], form_table["n 2048, m 512"]
    checks.check("forming at the certificate's 8/n makes the delivered gap at K 2,048 about 0.91%, and with fresh F "
                 "every call (8/m) T1 fails at m 512 even against the delivered adversary",
                 abs(f0["delivered"]["certificate"][2048] - 0.0091) < 5e-4 and f512["delivered"]["certificate"][2048]
                 > 0.01, f"{f0['delivered']['certificate'][2048]:.2%} (F reused), "
                 f"{f512['delivered']['certificate'][2048]:.2%} (m 512)")
    checks.check("beside the wgmma (estimate) forming at m >= 512 costs honest less than the certificate's additive "
                 "charge; with F reused T1 holds at K 2,048 against the floor adversary, and fresh F at m 512 needs a "
                 "larger K", all(form_table[f"n 2048, m {m}"]["delivered"]["beside"][2048]
                                 < form_table[f"n 2048, m {m}"]["delivered"]["certificate"][2048] for m in (512, 8192))
                 and f0["floor"]["beside"][2048] < 0.01 < f512["floor"]["beside"][2048],
                 f"floor at K 2,048: {f0['floor']['beside'][2048]:.2%} (F reused), {f512['floor']['beside'][2048]:.2%} "
                 f"(m 512, T1 from K {f512['floor']['kmin_beside']:,.0f}); additive {f0['floor']['certificate'][2048]:.2%}"
                 f" and {f512['floor']['certificate'][2048]:.2%}")
    mf = margins["floor@2048"]["every_route"]
    checks.check("premise P32's margin at K 2,048 (the floor reading): T1 holds while an off-tensor word costs at least "
                 "that fraction of the floor's issue slots, and the margin grows with K",
                 mf < 1.0 and margins["floor@16384"]["every_route"] < margins["floor@8192"]["every_route"] < mf,
                 f"{mf:.3f}x (margin {1 / mf:.2f}x); K 8,192 {margins['floor@8192']['every_route']:.3f}x, K 16,384 "
                 f"{margins['floor@16384']['every_route']:.3f}x")
    md = margins["delivered@2048"]["every_route"]
    checks.check("against the delivered adversary at K 2,048 T1 holds while its side word costs at least that fraction "
                 "of honest's instructions", 0.5 < md < mf, f"{md:.3f}x (margin {1 / md:.2f}x)")

    # (5) The measured co-issue points: inside the polytope, and the schedules they reach.
    measured = None
    paths = [checks.args[i + 1] for i, a in enumerate(checks.args) if a == "--coissue"]
    if paths:
        pts = coissue_points(paths)
        for p in pts:
            u = usage(p, rows)
            p["row"], p["use"] = max(u.items(), key=lambda kv: kv[1]) if p["lanes"] else ("tensor", p["t"])
            p["words"] = {k: sm * words_at(p, k) for k in ("honest", "floor", "products")}
            p["side_issue"] = sm * sum(p["lanes"].values())
        print("  co-issue point          tensor  lanes (warp instructions/SM-clk)          side  tightest row   "
              "words: honest  floor")
        for p in pts:
            lanes = ", ".join(f"{k} {sm * v:.3f}" for k, v in p["lanes"].items()) or "-"
            print(f"  {p['probe']:22s} {p['t']:6.4f}  {lanes:40s} {p['side_issue']:5.3f}  {p['row']:9s} "
                  f"{p['use']:5.3f}   {p['words']['honest']:7.2f} {p['words']['floor']:7.2f}")
        worst = max(pts, key=lambda p: p["use"])
        checks.check("every co-issue point lies inside the census polytope: no measured point refutes a row",
                     worst["use"] <= 1.01, f"{len(pts)} points; tightest {worst['probe']} at {worst['row']} "
                     f"{worst['use']:.3f}")
        sat = [p for p in pts if p["t"] >= 0.998 and p["lanes"]]
        poly_side = min((1 - a * G_SAT) / coef for cls, a, coef in rows.values() if cls == "ALL")
        best_sat = max(sat, key=lambda p: p["side_issue"])
        best_h = max(pts, key=lambda p: p["words"]["honest"])
        best_d = max((p for p in pts if delivered(p)), key=lambda p: p["words"]["honest"])
        best_a = max(pts, key=lambda p: p["words"]["floor"])
        r_hm, r_hd, r_am = best_h["words"]["honest"] / sm, best_d["words"]["honest"] / sm, best_a["words"]["floor"] / sm

        def against(r, ra):
            return {"gap": {k: gap(r, ra, k) for k in KS}, "kmin": kmin(r, ra)}

        measured = {
            "points": pts, "polytope_side_issue_at_saturation": sm * poly_side,
            "best_side_issue_saturated": {"probe": best_sat["probe"], "side_issue": best_sat["side_issue"]},
            "honest_delivered": {"probe": best_d["probe"], "words_sm_clk": sm * r_hd},
            "honest_products_only": {"probe": best_h["probe"], "words_sm_clk": sm * r_hm},
            "adversary_measured": {"probe": best_a["probe"], "words_sm_clk": sm * r_am},
            "vs_polytope": {kind: against(r_hd, adv[kind]["rate"]) for kind in adv},
            "vs_polytope_products_only": {kind: against(r_hm, adv[kind]["rate"]) for kind in adv},
            "vs_measured": against(r_hd, r_am), "vs_measured_products_only": against(r_hm, r_am),
            "vs_measured_delivered": against(r_hd, r_hd),
        }
        print(f"  measured honest: {sm * r_hd:.2f} words/SM-clk with its operand reads ({best_d['probe']}), "
              f"{sm * r_hm:.2f} on products alone ({best_h['probe']}); the polytope's honest {sm * r_h:.2f}")
        for kind, v in measured["vs_polytope"].items():
            print(f"    vs {kind:9s} adversary {adv[kind]['words_sm_clk']:7.2f} (polytope): gap "
                  + "  ".join(f"K{k} {v['gap'][k]:6.2%}" for k in KS) + f"  T1 from K {v['kmin']:,.0f}")
        for name, key in (("products alone", "vs_measured_products_only"), ("with operand reads", "vs_measured")):
            v = measured[key]
            print(f"    honest {name}, vs the floor adversary at the measured points {sm * r_am:7.2f} "
                  f"({best_a['probe']}): gap " + "  ".join(f"K{k} {v['gap'][k]:6.2%}" for k in KS)
                  + f"  T1 from K {v['kmin']:,.0f}")
        checks.check("beside a saturated wgmma no measured mix issues the lane instructions the polytope allows: the "
                     "census's generic rows are loose there", best_sat["side_issue"] < 0.9 * sm * poly_side,
                     f"measured {best_sat['side_issue']:.3f} ({best_sat['probe']}) against the rows' "
                     f"{sm * poly_side:.3f} warp instructions/SM-clk")
        vp = measured["vs_polytope"]
        checks.check("honest's operand reads cost it words: its best delivered point makes fewer than its best on "
                     "products alone, and both fewer than the polytope's honest", r_hd < r_hm < r_h,
                     f"{sm * r_hd:.2f} ({best_d['probe']}) < {sm * r_hm:.2f} ({best_h['probe']}) < {sm * r_h:.2f}")
        checks.check("at its measured rate with its operand reads honest misses T1 at every K against every polytope "
                     "adversary, the delivered one included: T1 rests on the rows' slack, or on a honest kernel that "
                     "reaches them", all(vp[k]["kmin"] == math.inf for k in adv),
                     "gap at K 2,048 " + ", ".join(f"{vp[k]['gap'][2048]:.2%} ({k}, T1 from K {vp[k]['kmin']:,.0f})"
                                                   for k in ("floor", "products", "ideal", "delivered")))
        vm = measured["vs_measured_products_only"]
        checks.check("on the measured points (not a bound) the floor adversary's best is within 0.1% of honest's on "
                     "products alone, so that gap is about the advice word, T1 from K near 1,100; with honest's "
                     "operand reads and none for the adversary T1 needs K above 16,384", r_am < 1.001 * r_hm
                     and vm["kmin"] < 1200 and measured["vs_measured"]["kmin"] > 16384,
                     f"{sm * r_am:.2f} ({best_a['probe']}) vs {sm * r_hm:.2f}: T1 from K {vm['kmin']:,.0f}, gap at K "
                     f"2,048 {vm['gap'][2048]:.2%}; with operand reads {measured['vs_measured']['gap'][2048]:.2%}, T1 "
                     f"from K {measured['vs_measured']['kmin']:,.0f}")
        if checks.mutations:
            up = max(usage(dict(best_h, lanes={k: 1.1 * v for k, v in best_h["lanes"].items()}), rows).items(),
                     key=lambda kv: kv[1])
            checks.mutation(f"{best_h['probe']}'s lanes 10% faster would leave the polytope: the inside check bites",
                            up[1] > 1.01, f"{up[0]} {up[1]:.3f}")

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
        fp = placement["FP64 under JI90 only"]["floor"]
        checks.mutation("FP64 under the generic rows only (not sharing the tensor) breaks T1 at K 2,048 against the floor "
                        "adversary: the FP64 row binds", fp["gap_k2048"] > 0.01,
                        f"gap {fp['gap_k2048']:.2%}, T1 from K {fp['kmin']:,.0f}")
    if out:
        Path(out).write_text(json.dumps({
            "smem": smem, "per_route": per_route, "tensor_forms": TENSOR_FORMS, "generous_words_sm_clk": sm * generous,
            "honest": {k: {kk: vv for kk, vv in v.items() if kk != "rate"} for k, v in table.items()},
            "adversary": {k: {kk: vv for kk, vv in v.items() if kk != "rate"} for k, v in adv.items()},
            "gap": {kind: {k: gap(r_h, a["rate"], k) for k in KS} for kind, a in adv.items()},
            "margins": margins, "honest_side_need_k2048": need, "honest_side": sm * side_h, "placement": placement,
            "forming": form_table, "measured": measured, "estimates": ESTIMATES, "routes": ROUTES}, indent=1,
            default=str) + "\n")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
