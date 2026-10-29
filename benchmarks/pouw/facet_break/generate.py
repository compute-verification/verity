"""FACET-BREAK's generator: a seeded grid of MMA + lane programs, emitted as one opcode probe source over `mix.cuh`.

A program is a dict (`kind` mix | spec | alone | mma, the MMA `form`, lane ops `l1`/`l2`, `q` lane ops a 8 MMAs, placement
`p`, `tpb`, and for spec `wm` MMA warps). The grid has two parts:

- **targeted** (the same for every seed): for each lane class X beside IMMA.16832, the vertices of Π's 2-D section in
  (x_MMA, x_X) (every row either loads, the census at the epoch), and programs whose ratio X/MMA sits at each vertex and
  around it (x 0.6 ... 1.6), at 4 warps a SMSP; each lane alone; each MMA form alone at 2, 4 and 8 warps a SMSP; and the
  JI3n family (every form beside IADD3, FFMA, IADD3+FFMA, immediate FFMA, LOP3 and IMAD at 4 to 16 ops a MMA).
- **sampled** (`seed`): `n` programs sampled uniformly over forms, one or two lanes, q, warps, placement and schedule.

`emit_cu` writes the probe source: each program's kernel (`FB_MIX`, `FB_SPEC`, `FB_ALONE`, `FB_MMA`) and its table entry
with the exact SASS counts the gate pins. A program the gate refuses is dropped (`--only`), never relaxed.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "machine_model"))

#: form: (mix.cuh struct, SASS opcode, census kind)
FORMS = {"i16832": ("I16832", "IMMA.16832.S8.S8", "IMMA16832"), "i16816": ("I16816", "IMMA.16816.S8.S8", "IMMA16816"),
         "h16816": ("H16816", "HMMA.16816.F32", "HMMA16816"), "h16816f16": ("H16816F16", "HMMA.16816.F16", "HMMA16816F16"),
         "h1684t": ("H1684T", "HMMA.1684.F32.TF32", "HMMA1684T"), "h1688t": ("H1688T", "HMMA.1688.F32.TF32", "HMMA1688T"),
         "isp16832": ("ISP16832", "IMMA.SP.16832.S8.S8", "IMMASP16832"),
         "isp16864": ("ISP16864", "IMMA.SP.16864.S8.S8", "IMMASP16864"),
         "hsp16832": ("HSP16832", "HMMA.SP.16832.F32", "HMMASP16832"),
         "qsp16864": ("QSP16864", "QMMA.SP.16864.F32.E4M3.E4M3", "QMMASP16864"),
         "q16832": ("Q16832", "QMMA.16832.F32.E4M3.E4M3", "QMMA16832"),
         "qsf16832": ("QSF16832", "QMMA.SF.16832.F32.E4M3.E4M3.E8", "QMMASF16832"),
         "osf16864": ("OSF16864", "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X", "OMMASF16864")}
#: lane: (mix.cuh enum, SASS fnmatch pattern, census class)
LANES = {"iadd3": ("L_IADD3", "IADD3", "ALU"),
         "lop3": ("L_LOP3", "LOP3.LUT", "ALU"), "shf": ("L_SHF", "SHF.*", "ALU"), "prmt": ("L_PRMT", "PRMT", "ALU"),
         "imad": ("L_IMAD", "IMAD", "IMAD"), "imadi": ("L_IMADI", "IMAD", "IMAD"),
         "imadw": ("L_IMADW", "IMAD.WIDE.U32", "IMADW"), "iadd64": ("L_IADD64", "IADD.64", "ALU64"),
         "sel": ("L_SEL", "SEL", "SEL"), "ffma": ("L_FFMA", "FFMA", "FFMA"), "ffmai": ("L_FFMAI", "FFMA", "FFMA"),
         "fmul": ("L_FMUL", "FMUL", "FFMA"), "fadd": ("L_FADD", "FADD", "FFMA"),
         "hfma2": ("L_HFMA2", "HFMA2*", "HFMA2"), "mnmx": ("L_MNMX", "*MNMX*", "VIMNMX"),
         "dmul": ("L_DMUL", "DMUL", "DMUL"), "dfma": ("L_DFMA", "DFMA", "W2"), "mufu": ("L_MUFU", "MUFU.EX2", "XU"),
         "popc": ("L_POPC", "POPC", "XU"),
         # the register-light lanes (ji family); a pair is (the op on even rounds of a chain, on odd ones), see lane_pat
         "fmuli": ("L_FMULI", "FMUL", "FFMA"), "faddi": ("L_FADDI", "FADD", "FFMA"),
         "axi": ("L_AXI", ("IADD3", "LOP3.LUT"), "ALU"), "shfi": ("L_SHFI", "SHF.*", "ALU"),
         "prmti": ("L_PRMTI", "PRMT", "ALU"), "imadwi": ("L_IMADWI", "IMAD.WIDE.U32", "IMADW"),
         "dp4a": ("L_DP4A", "IDP.4A.*", "IMAD"), "dp4ai": ("L_DP4AI", "IDP.4A.*", "IMAD"),
         "uaxi": ("L_UAXI", ("UIADD3", "ULOP3.LUT"), "UNI"), "shfl": ("L_SHFL", "SHFL.BFLY", "SHFL")}
#: the lanes the targeted part puts beside IMMA.16832 (one class each: the 2-D sections of the joint rows)
TARGET_LANES = ("sel", "imadw", "iadd64", "ffma", "fmul", "dmul", "dfma", "lop3", "iadd3", "shf", "prmt", "imad",
                "hfma2", "mnmx", "mufu", "popc", "imadi", "ffmai")
PAIRS = (("iadd3", "ffma"), ("lop3", "ffma"), ("imad", "ffma"), ("sel", "ffma"), ("hfma2", "ffma"), ("iadd3", "imad"),
         ("lop3", "imad"), ("sel", "iadd3"), ("imadw", "ffma"), ("iadd64", "ffma"), ("mnmx", "ffma"), ("prmt", "imad"))
JI3N_LANES = ("iadd3", "ffma", ("iadd3", "ffma"), "ffmai", "lop3", "imad")
SCALES = (0.6, 0.8, 1.0, 1.25, 1.6)
#: the densities every targeted lane also gets, so long edges between vertices are sampled too
BASE_Q = (2, 4, 8, 16, 24, 32, 48, 64, 96, 128)
#: the lanes also timed at 2 and 8 warps a SMSP (the joint rows tonight's interleaved points beat, and their neighbors)
WARP_LANES = ("ffma", "sel", "imadw", "iadd64", "iadd3", "imad", "lop3")
QMAX = 192

#: load: (mem.cuh enum, SASS opcode, census class or None where the census has no kind, shared?). TLD's width is in its
#: operands, not its opcode, so loads.py reads a TLD's class off the program's name (`_tld4_` etc).
LOADS = {"lds32": ("D_LDS32", "LDS", "LDS32", True), "lds64": ("D_LDS64", "LDS.64", None, True),
         "lds128": ("D_LDS128", "LDS.128", "LDS128", True), "ldsu8": ("D_LDSU8", "LDS.U8", None, True),
         "ldg32": ("D_LDG32", "LDG.E", "LDG32", False), "ldgnc32": ("D_LDGNC32", "LDG.E.CONSTANT", "LDG32", False),
         "ldg64": ("D_LDG64", "LDG.E.64", None, False), "ldg128": ("D_LDG128", "LDG.E.128", "LDG128", False),
         "tld1": ("D_TLD1", "TLD.LZ", "TLD1", False), "tld2": ("D_TLD2", "TLD.LZ", "TLD2", False),
         "tld4": ("D_TLD4", "TLD.LZ", "TLD4w", False), "tld4n": ("D_TLD4N", "TLD.LZ", "TLD4w", False),
         "ldg256": ("D_LDG256", "LDG.E.ENL2.256", None, False),
         "ldsm4": ("D_LDSM4", "LDSM.16.M88.4", "LDSM_X4", True), "sts128": ("D_STS128", "STS.128", "STS128", True)}
#: the loads of the mem family (its seeded grid as r20261017-063107-835c ran it)
MEM_LOADS = ("lds32", "lds64", "lds128", "ldsu8", "ldg32", "ldgnc32", "ldg64", "ldg128", "tld1", "tld2", "tld4", "tld4n")
PATS = {"line": "P_LINE", "bcast": "P_BCAST", "quad": "P_QUAD"}
#: an L2-resident load (mem.cuh's FB_L2, a block's 256 KB region): (enum, SASS opcode). The .NA and .STRONG.GPU forms skip
#: L1, so loads.py counts their bytes on the L2 facet; the plain forms show what L1 still serves of a 256 KB sweep.
L2_LOADS = {"g128": ("D_L2G128", "LDG.E.128"), "g128na": ("D_L2G128NA", "LDG.E.NA.128"),
            "g128cg": ("D_L2G128CG", "LDG.E.128.STRONG.GPU"), "g256": ("D_L2G256", "LDG.E.ENL2.256"),
            "g256cg": ("D_L2G256CG", "LDG.E.ENL2.256.STRONG.GPU")}
L2_PATS = {"line": "P_LINE", "scat": "P_SCAT"}
#: two loads in one loop: the shared-memory pair (R6r, LDSi), the global pair (LDG), and one of each space (issue, JI3n)
LOAD_PAIRS = (("lds32", "lds128"), ("lds32", "lds64"), ("lds128", "lds64"), ("ldg32", "ldg128"), ("ldgnc32", "ldg128"),
              ("lds32", "ldg32"), ("lds128", "ldg128"), ("lds32", "tld1"), ("lds128", "tld4"), ("ldg32", "tld1"),
              ("ldg128", "tld4"))
#: the loads timed beside IMMA.16832 (M, JI3n, issue: a table route co-running with the tensor pipe)
MMA_LOADS = ("lds32", "lds128", "lds64", "ldg32", "ldgnc32", "ldg128", "tld1", "tld4", "tld4n")


def load_kinds(g: dict, q: int) -> str:
    """The load of slot q (mem.cuh's kind_at): d2 when q % r == r - 1."""
    return g["d2"] if g.get("d2") and q % g["r"] == g["r"] - 1 else g["d1"]


def lane_pat(lane: str, k: int) -> str:
    """The SASS pattern of a lane's k-th op (mix.cuh's op<L>(k)): an alternating lane's by the chain's round."""
    pat = LANES[lane][1]
    return pat if isinstance(pat, str) else pat[(k >> 3) & 1]


def lanes_of(g: dict) -> list[str]:
    return [x for x in (g.get("l1"), g.get("l2"), g.get("l3")) if x]


def name_of(g: dict) -> str:
    w = g["tpb"] // 32
    if g["kind"] == "jx":
        sched = "" if (g["na"], g["u"]) == (4, 1) else f"_a{g['na']}u{g['u']}"
        return f"jx_{g['form']}_{g['form2']}_{'_'.join(lanes_of(g)) or 'none'}_q{g['q']}{sched}_w{w}"
    if g["kind"] in ("jm", "js"):
        ls = "_".join(lanes_of(g)) or "none"
        sched = f"a{g['na']}u{g['u']}"
        if g["kind"] == "js":
            lq = f"_l{g['lq']}" if g["lq"] != _js_lq(lanes_of(g)) else ""
            return f"js_{g['form']}_{ls}_{sched}_m{g['mw']}{lq}_w{w}"
        ld = f"_{g['d']}x{g['qd']}" if g.get("d") else ""
        rep = f"_k{g['rep']}" if g.get("rep") else ""
        return f"jm_{g['form']}_{ls}_q{g['q']}_p{g['p']}_{sched}{ld}_w{w}{rep}"
    if g["kind"] == "l2":
        return f"l2_{g['d1']}_{g['pat']}_q{g['q']}_w{w}"
    if g["kind"] in ("ld", "ldmma"):
        ds = g["d1"] + (f"_{g['d2']}_r{g['r']}" if g.get("d2") else "")
        pre = "ld" if g["kind"] == "ld" else f"lm_{g['form']}"
        return f"{pre}_{ds}_{g['pat']}_q{g['q']}_w{w}"
    lanes = g["l1"] + (f"_{g['l2']}" if g.get("l2") else "")
    if g["kind"] == "mix":
        return f"mx_{g['form']}_{lanes}_q{g['q']}_p{g['p']}_w{w}"
    if g["kind"] == "spec":
        return f"sp_{g['form']}_{lanes}_m{g['wm']}_w{w}"
    if g["kind"] == "alone":
        return f"al_{lanes}_w{w}"
    return f"mm_{g['form']}_w{w}"


def expect_of(g: dict) -> tuple[str, str, int]:
    """(expect, warps, until) for the probe table: each loop's exact opcode counts an iteration."""
    def lane_counts(n: int) -> dict[str, int]:
        out: dict[str, int] = {}
        ls = lanes_of(g)
        for i in range(n if ls else 0):
            pat = lane_pat(ls[i % len(ls)], i // len(ls))
            out[pat] = out.get(pat, 0) + 1
        return out

    def fmt(d: dict[str, int]) -> str:
        return ",".join(f"{k}={v}" for k, v in d.items())

    def add(d: dict[str, int], e: dict[str, int]) -> dict[str, int]:
        for k, v in e.items():
            d[k] = d.get(k, 0) + v
        return d

    w = g["tpb"] // 32
    if g["kind"] == "jx":
        mmas = {FORMS[g["form"]][1]: 4 * g["u"], FORMS[g["form2"]][1]: 4 * g["u"]}
        return fmt(add(mmas, lane_counts(g["u"] * g["q"]))), "", 0
    if g["kind"] == "jm":
        d = add({FORMS[g["form"]][1]: 8 * g["u"]}, lane_counts(g["u"] * g["q"]))
        if g.get("d"):
            d = add(d, {LOADS[g["d"]][1]: g["u"] * g["qd"]})
        if "uaxi" in lanes_of(g):  # where ptxas keeps the chains uniform, the loop's counter is a UIADD3 too
            d = add(d, {"UIADD3": 1})
        return fmt(d), "", 0
    if g["kind"] == "js":
        return (f"{FORMS[g['form']][1]}={8 * g['u']};{fmt(lane_counts(g['lq']))}", f"{4 * g['mw']};{w - 4 * g['mw']}", 2)
    if g["kind"] == "l2":
        return f"{L2_LOADS[g['d1']][1]}={g['q']}", "", 0
    if g["kind"] in ("ld", "ldmma"):
        d = {FORMS[g["form"]][1]: 8} if g["kind"] == "ldmma" else {}
        for q in range(g["q"]):
            pat = LOADS[load_kinds(g, q)][1]
            d[pat] = d.get(pat, 0) + 1
        return fmt(d), "", 0
    if g["kind"] == "mix":
        d = {FORMS[g["form"]][1]: 8}
        for k, v in lane_counts(g["q"]).items():
            d[k] = d.get(k, 0) + v
        return fmt(d), "", 0
    if g["kind"] == "spec":
        return f"{FORMS[g['form']][1]}=8;{fmt(lane_counts(16))}", f"{g['wm']};{w - g['wm']}", 2
    if g["kind"] == "alone":
        return fmt(lane_counts(16)), "", 0
    return f"{FORMS[g['form']][1]}=8", "", 0


def vertices(x_class: str, form_kind: str = "IMMA16832", epoch: str | None = None) -> list[tuple[float, float]]:
    """The vertices of Π's section in (x_MMA, x_X): every row the MMA form or X loads, intersected pairwise and with the
    axes, kept where every row holds."""
    from verity_machine_model.model import census, model
    sys.path.insert(0, str(HERE))
    from loads import facet_loads
    m = model()
    c = census(m, epoch or m.current)
    am = facet_loads({form_kind: 1.0}, c, m.cycle_u)
    ax = facet_loads({x_class: 1.0}, c, m.cycle_u)
    rows = [(am.get(r, 0.0), ax.get(r, 0.0)) for r in set(am) | set(ax)]
    rows = [r for r in rows if r[0] > 0 or r[1] > 0]
    lines = rows + [(-1.0, 0.0), (0.0, -1.0)]  # m >= 0, x >= 0 as -m <= 0
    rhs = [1.0] * len(rows) + [0.0, 0.0]
    pts = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            (a1, b1), (a2, b2) = lines[i], lines[j]
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-15:
                continue
            mm = (rhs[i] * b2 - rhs[j] * b1) / det
            xx = (a1 * rhs[j] - a2 * rhs[i]) / det
            if mm < -1e-12 or xx < -1e-12:
                continue
            if all(a * mm + b * xx <= 1 + 1e-9 for a, b in rows):
                pts.append((max(mm, 0.0), max(xx, 0.0)))
    return sorted(set((round(p[0], 9), round(p[1], 9)) for p in pts))


def targeted() -> list[dict]:
    out: list[dict] = []
    for lane in TARGET_LANES:
        out.append({"kind": "alone", "l1": lane, "l2": "", "tpb": 512})
        qs = set(BASE_Q)
        for mm, xx in vertices(LANES[lane][2]):
            if mm <= 0 or xx <= 0:
                continue
            for s in SCALES:
                q = round(8 * xx / mm * s)
                if 1 <= q <= QMAX:
                    qs.add(q)
        for q in sorted(qs):
            out.append({"kind": "mix", "form": "i16832", "l1": lane, "l2": "", "q": q, "p": 0, "tpb": 512})
        if lane in WARP_LANES:
            for tpb in (256, 1024):
                for q in BASE_Q:
                    out.append({"kind": "mix", "form": "i16832", "l1": lane, "l2": "", "q": q, "p": 0, "tpb": tpb})
            for wm, tpb in ((8, 512), (8, 768), (4, 512), (16, 1024)):
                out.append({"kind": "spec", "form": "i16832", "l1": lane, "l2": "", "wm": wm, "tpb": tpb})
    for l1, l2 in PAIRS:
        out.append({"kind": "alone", "l1": l1, "l2": l2, "tpb": 512})
        for q in (8, 16, 32, 48, 64, 96, 128):
            out.append({"kind": "mix", "form": "i16832", "l1": l1, "l2": l2, "q": q, "p": 0, "tpb": 512})
    for form in FORMS:
        for tpb in (256, 512, 1024):
            out.append({"kind": "mma", "form": form, "l1": "", "l2": "", "tpb": tpb})
        for lane in JI3N_LANES:
            l1, l2 = (lane, "") if isinstance(lane, str) else lane
            for q in (32, 64, 96, 128):
                out.append({"kind": "mix", "form": form, "l1": l1, "l2": l2, "q": q, "p": 0, "tpb": 512})
    return out


def sampled(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    lanes = sorted(set(LANES) - set(RT9_LANES))  # a seed's grid stays the one its batch ran
    for _ in range(n):
        kind = rng.choices(("mix", "spec"), weights=(4, 1))[0]
        form = rng.choice(sorted(FORMS))
        l1 = rng.choice(lanes)
        l2 = rng.choice([""] * 3 + [x for x in lanes if LANES[x][1] != LANES[l1][1]])
        tpb = 32 * rng.choice((8, 12, 16, 24, 32))
        g = {"kind": kind, "form": form, "l1": l1, "l2": l2, "tpb": tpb}
        if kind == "mix":
            g |= {"q": rng.choice((2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160)), "p": rng.choice((0, 1))}
        else:
            g |= {"wm": rng.choice([w for w in (4, 8, 12, 16) if w < tpb // 32 and (tpb // 32 - w) % 4 == 0])}
        out.append(g)
    return out


def targeted_mem() -> list[dict]:
    """The table routes' loads: each alone (three patterns, 2 to 8 warps a SMSP), each pair at 1:1, 3:1 and 1:3, and each
    beside IMMA.16832 at 2 to 32 loads a 8 MMAs (and beside HMMA.16816.F16, the form that beat JI3n_0)."""
    out: list[dict] = []
    ld = {"kind": "ld", "d2": "", "r": 1}
    for d in MEM_LOADS:
        for pat in PATS:
            for tpb in (256, 512, 1024):
                for q in (8, 16):
                    out.append(ld | {"d1": d, "pat": pat, "q": q, "tpb": tpb})
    for d1, d2 in LOAD_PAIRS:
        for pat in ("line", "bcast"):
            for tpb in (512, 1024):
                for a, b, r in ((d1, d2, 2), (d1, d2, 4), (d2, d1, 4)):
                    out.append({"kind": "ld", "d1": a, "d2": b, "r": r, "pat": pat, "q": 16, "tpb": tpb})
    for d in MMA_LOADS:
        for pat in ("line", "bcast"):
            for tpb in (512, 1024):
                for q in (2, 4, 8, 16, 32):
                    out.append({"kind": "ldmma", "form": "i16832", "d1": d, "d2": "", "r": 1, "pat": pat, "q": q,
                                "tpb": tpb})
    for d in ("lds32", "lds128", "ldg32", "tld1"):
        for q in (8, 16, 32):
            out.append({"kind": "ldmma", "form": "h16816f16", "d1": d, "d2": "", "r": 1, "pat": "line", "q": q,
                        "tpb": 512})
    return out


def sampled_mem(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        d1 = rng.choice(sorted(MEM_LOADS))
        d2 = rng.choice([""] * 2 + [d for d in sorted(MEM_LOADS) if LOADS[d][1] != LOADS[d1][1]])
        g = {"d1": d1, "d2": d2, "r": rng.choice((2, 3, 4)) if d2 else 1, "pat": rng.choice(sorted(PATS)),
             "tpb": 32 * rng.choice((8, 12, 16, 24, 32))}
        if rng.random() < 0.5:
            g |= {"kind": "ld", "q": rng.choice((4, 8, 12, 16, 24, 32))}
        else:
            g |= {"kind": "ldmma", "form": rng.choice(sorted(FORMS)), "q": rng.choice((2, 4, 6, 8, 12, 16, 24, 32))}
        out.append(g)
    return out


#: the fine sweep: lane beside IMMA.16832 and the q range around the frontier of the rows the first batch beat or came
#: within 1% of (r20261017-060543-cc28: R1/KFFMA0 by FFMA, KW20/KDMUL0, KALU64*, KALU*, KSEL3, EIMAD5, EHFMA25, KIMADW*)
FINE = {"ffma": range(80, 193, 8), "fmul": range(80, 193, 8), "iadd64": range(36, 97, 4), "iadd3": range(16, 129, 8),
        "imad": range(22, 51, 2), "hfma2": range(22, 51, 2), "imadw": range(40, 97, 4), "sel": range(14, 41, 2),
        "dfma": range(1, 7), "dmul": range(1, 7)}


def targeted_fine() -> list[dict]:
    """The near-frontier sweep: each FINE lane beside IMMA.16832 at every q of its range, 2 to 8 warps a SMSP, spread and
    batched; every form beside FFMA (and IADD3+FFMA) at 64-192 a 8 MMAs (JI3n_0, beaten by HMMA.16816.F16 + FFMA)."""
    out: list[dict] = []
    for lane, qs in FINE.items():
        for q in qs:
            for tpb in (256, 512, 768, 1024):
                out.append({"kind": "mix", "form": "i16832", "l1": lane, "l2": "", "q": q, "p": 0, "tpb": tpb})
            out.append({"kind": "mix", "form": "i16832", "l1": lane, "l2": "", "q": q, "p": 1, "tpb": 512})
    for form in FORMS:
        for l1, l2 in (("ffma", ""), ("ffmai", ""), ("iadd3", "ffma")):
            for q in range(64, 193, 16):
                for tpb in (512, 1024):
                    out.append({"kind": "mix", "form": form, "l1": l1, "l2": l2, "q": q, "p": 0, "tpb": tpb})
    return out


def targeted_l2() -> list[dict]:
    """The L2 facet (machine_model/data/table.toml's `l2`, what the table route's lookups are priced against): each
    L2-resident load in both patterns at 4 to 16 loads an iteration and 2 to 8 warps a SMSP, and the 256-bit load from the
    L1-resident ring (no census class: Π prices it at 0 on LDG)."""
    out: list[dict] = []
    for d in L2_LOADS:
        for pat in L2_PATS:
            for tpb in (256, 512, 768, 1024):
                for q in (4, 8, 16):
                    out.append({"kind": "l2", "d1": d, "pat": pat, "q": q, "tpb": tpb})
    for pat in PATS:
        for tpb in (256, 512, 1024):
            for q in (8, 16):
                out.append({"kind": "ld", "d1": "ldg256", "d2": "", "r": 1, "pat": pat, "q": q, "tpb": tpb})
    return out


def sampled_l2(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    return [{"kind": "l2", "d1": rng.choice(sorted(L2_LOADS)), "pat": rng.choice(sorted(L2_PATS)),
             "q": rng.choice((2, 3, 4, 6, 8, 12, 16)), "tpb": 32 * rng.choice((4, 8, 12, 16, 20, 24, 28, 32))}
            for _ in range(n)]


#: the ji family's lanes: the register-light ones, and the mix family's at the warps a SMSP it didn't time (1, 3, 6)
JI_NEW = ("fmuli", "faddi", "axi", "shfi", "prmti", "imadwi", "dp4a", "dp4ai", "uaxi")
JI_OLD = ("ffmai", "ffma", "iadd3", "lop3", "imad", "imadi", "imadw", "prmt", "shf")
#: two and three lanes in turn: a FMA-pipe lane beside an ALU or IMAD one (two dispatch pipes), and the uniform datapath
JI_MULTI = (("fmuli", "axi"), ("ffmai", "axi"), ("ffmai", "shfi"), ("fmuli", "prmti"), ("ffmai", "imadi"),
            ("fmuli", "dp4ai"), ("axi", "dp4ai"), ("ffmai", "dp4a"), ("fmuli", "imadwi"), ("axi", "imadi"),
            ("fmuli", "faddi"), ("ffma", "axi"), ("ffmai", "uaxi"), ("fmuli", "axi", "imadi"),
            ("ffmai", "axi", "dp4ai"), ("fmuli", "shfi", "imadwi"), ("ffmai", "iadd3", "imad"), ("fmuli", "axi", "uaxi"),
            ("ffma", "lop3", "imad"))
#: the schedule sweep's lanes (placement, accumulators, unroll depth)
JI_SCHED = ("ffmai", "fmuli", "axi", ("fmuli", "axi"), ("ffmai", "axi"), ("ffmai", "imadi"))
#: lanes whose SASS no load's address arithmetic or the OP_UNTIL trip counter shares (IADD3, LOP3.LUT would collide)
JI_LOADSAFE = (("ffmai",), ("fmuli",), ("fmuli", "imadi"), ("ffmai", "dp4ai"))
JI_SPEC = ("ffmai", "fmuli", "faddi", "ffma", "dp4ai", "imadi", "shfi", "prmti", ("fmuli", "shfi"), ("ffmai", "imadi"),
           ("fmuli", "dp4ai", "prmti"))
#: (MMA warps, warps) a SMSP of the warp-specialized programs
JI_SPLITS = ((1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 8), (2, 3), (2, 4), (2, 5), (2, 6), (2, 8), (3, 4), (3, 6),
             (4, 6), (4, 8))


def _jm(form: str, ls, q: int, w: int, p: int = 0, na: int = 8, u: int = 1, d: str = "", qd: int = 0) -> dict:
    ls = (ls,) if isinstance(ls, str) else tuple(ls)
    g = {"kind": "jm", "form": form, "q": q, "p": p, "na": na, "u": u, "tpb": 128 * w}
    g |= {f"l{i + 1}": x for i, x in enumerate(ls)}
    return g | ({"d": d, "qd": qd} if d else {})


def _jx(form: str, form2: str, ls, q: int, w: int, na: int = 4, u: int = 1) -> dict:
    """Two forms in turn (ji.cuh's jmix2), `na` accumulators each."""
    ls = (ls,) if isinstance(ls, str) else tuple(ls)
    g = {"kind": "jx", "form": form, "form2": form2, "q": q, "na": na, "u": u, "tpb": 128 * w}
    return g | {f"l{i + 1}": x for i, x in enumerate(ls)}


def _js_lq(ls) -> int:
    """A warp-specialized program's lane ops a trip unless asked: 24 for three lanes, else 16."""
    return 24 if len(ls) == 3 else 16


def _js(form: str, ls, mw: int, w: int, na: int = 8, u: int = 1, lq: int = 0) -> dict:
    ls = (ls,) if isinstance(ls, str) else tuple(ls)
    g = {"kind": "js", "form": form, "na": na, "u": u, "mw": mw, "lq": lq or _js_lq(ls), "tpb": 128 * w}
    return g | {f"l{i + 1}": x for i, x in enumerate(ls)}


def targeted_ji() -> list[dict]:
    """The JI3q search: IMMA.16832 beside each register-light lane (1 to 8 warps a SMSP) and the mix family's lanes at
    1, 3 and 6; two and three lanes in turn; placement, accumulators and unroll depth; loads beside the MMAs and lanes;
    warp-specialized splits of 1 to 4 MMA warps a SMSP; then HMMA.16816.F16 and HMMA.1688.TF32 over the same axes."""
    out: list[dict] = []
    i = "i16832"
    for lane in JI_NEW:
        for q in (32, 64, 96, 128, 160, 192, 256):
            for w in (1, 2, 4, 8):
                out.append(_jm(i, lane, q, w))
    for lane in JI_OLD:
        for q in (64, 96, 128, 160, 192):
            for w in (1, 3, 6):
                out.append(_jm(i, lane, q, w))
    for ls in JI_MULTI:
        for q in (64, 96, 128, 160, 192, 256):
            for w in (2, 4, 8):
                out.append(_jm(i, ls, q, w))
    for ls in JI_SCHED:
        for q in (96, 128, 160):
            for w in (2, 4):
                for p in (1, 2, 3):
                    out.append(_jm(i, ls, q, w, p=p))
                for na, u in ((4, 1), (16, 1), (8, 2), (8, 4), (16, 2)):
                    out.append(_jm(i, ls, q, w, na=na, u=u))
    for d in ("lds32", "lds128", "ldg32", "ldg128"):
        for qd in (2, 4, 8, 16):
            for w in (2, 4, 8):
                out.append(_jm(i, (), 0, w, d=d, qd=qd))
                for ls in JI_LOADSAFE:
                    for q in (64, 128):
                        out.append(_jm(i, ls, q, w, d=d, qd=qd))
    for ls in JI_SPEC:
        for mw, w in JI_SPLITS:
            out.append(_js(i, ls, mw, w))
    for ls in ("ffmai", "fmuli", ("ffmai", "imadi")):
        for mw, w in ((1, 4), (2, 4), (1, 2)):
            for na, u in ((4, 1), (16, 1), (8, 2)):
                out.append(_js(i, ls, mw, w, na=na, u=u))
    for form in ("h16816f16", "h1688t"):
        for lane in ("ffmai", "fmuli", "axi", "ffma", "dp4ai", "imadi", "shfi", "faddi"):
            for q in (64, 96, 112, 128, 160, 192):
                for w in (1, 2, 4, 8):
                    out.append(_jm(form, lane, q, w))
        for ls in (("fmuli", "axi"), ("ffmai", "axi"), ("ffmai", "imadi"), ("fmuli", "dp4ai"), ("fmuli", "axi", "imadi")):
            for q in (96, 128, 160, 192):
                for w in (2, 4, 8):
                    out.append(_jm(form, ls, q, w))
        for ls in ("ffmai", ("fmuli", "axi")):
            for q in (112, 128):
                for w in (2, 4):
                    for p in (1, 2, 3):
                        out.append(_jm(form, ls, q, w, p=p))
                    for na, u in ((4, 1), (16, 1), (8, 2)):
                        out.append(_jm(form, ls, q, w, na=na, u=u))
        for ls in ("ffmai", "fmuli", ("ffmai", "imadi")):
            for mw, w in ((1, 2), (1, 4), (2, 4), (1, 6), (2, 6), (1, 8)):
                out.append(_js(form, ls, mw, w))
        for d in ("lds128", "ldg128"):
            for qd in (4, 8):
                for w in (4, 8):
                    for ls in (("ffmai",), ("fmuli",)):
                        out.append(_jm(form, ls, 128, w, d=d, qd=qd))
    for f1, f2 in (("i16832", "h16816f16"), ("i16832", "h1688t"), ("h16816f16", "h1688t"), ("i16832", "i16816"),
                   ("i16832", "h16816")):
        for w in (2, 4, 8):
            out.append(_jx(f1, f2, (), 0, w))
            for ls in (("ffmai",), ("fmuli",), ("fmuli", "axi")):
                for q in (64, 96, 128, 160):
                    out.append(_jx(f1, f2, ls, q, w))
    return [g for g in out if g["na"] < 16 or g["tpb"] <= 512]


def targeted_jifine() -> list[dict]:
    """The near-frontier sweep of the ji family's break (r20261017-085714-2f2a: FMUL by an immediate on 4 accumulators at
    2 warps a SMSP beat JI3q_1..10 and EFFMA1..6; on 8 accumulators it doesn't): the one-read FP lanes and FFMA at 2 to 6
    accumulators, 1 to 4 warps a SMSP, densities from the MMA end (JI3q_11) to the vector end (JI3q_0), each form."""
    out: list[dict] = []
    dense = (32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224)
    for lane in ("fmuli", "faddi"):
        for na in (2, 3, 4, 6):
            for q in dense:
                for w in (1, 2, 3, 4):
                    out.append(_jm("i16832", lane, q, w, na=na))
    for na in (3, 4):
        for q in dense:
            for w in (2, 4):
                for ls in (("fmuli", "faddi"), ("ffma", "axi"), ("fmuli", "axi")):
                    out.append(_jm("i16832", ls, q, w, na=na))
        for lane in ("ffmai", "ffma"):
            for q in (64, 80, 96, 112, 128, 160):
                for w in (2, 3, 4):
                    out.append(_jm("i16832", lane, q, w, na=na))
    for form in ("h16816f16", "h1688t"):
        for lane in ("fmuli", "faddi"):
            for na in (3, 4, 6):
                for q in (48, 64, 80, 96, 112, 128, 160, 192):
                    for w in (2, 4):
                        out.append(_jm(form, lane, q, w, na=na))
    return out


def targeted_jipair() -> list[dict]:
    """Two dispatch pipes beside the HMMA forms (r20261017-092045-f7f6: HMMA.16816.F16 beside FFMA-immediate and
    add/xor-immediate in turn beat JI3q_11): a FMA-pipe lane and AXI at 3, 4 and 8 accumulators, 2 and 4 warps a SMSP,
    48-128 lane ops a 8 MMAs; IMMA.16832 beside FFMA-immediate and AXI at 3 and 4."""
    out: list[dict] = []
    qs = (48, 64, 80, 96, 112, 128)
    for form in ("h16816f16", "h1688t"):
        for ls in (("ffmai", "axi"), ("ffma", "axi"), ("fmuli", "axi")):
            for na in (3, 4, 8):
                for q in qs:
                    for w in (2, 4):
                        out.append(_jm(form, ls, q, w, na=na))
    for na in (3, 4):
        for q in qs:
            for w in (2, 4):
                out.append(_jm("i16832", ("ffmai", "axi"), q, w, na=na))
    return out


#: the forms the jifine break (r20261017-093644-cef3) didn't reach, and the three it beat with its worst program each
R4_FORMS = ("i16816", "h16816", "h1684t", "isp16832", "isp16864", "hsp16832", "qsp16864", "q16832", "qsf16832",
            "osf16864")
R4_REPEAT = (("i16832", 2), ("h16816f16", 3), ("h1688t", 3))
#: ALU-only lanes (an ALU-pipe fold's ops: LOP3, SHF, PRMT and the one-read immediates) and ALU beside a FP lane
R4_ALU = ("axi", "shfi", "prmti", "lop3", "shf", "prmt", ("shfi", "prmti"), ("axi", "shfi"), ("lop3", "shf", "prmt"))
R4_ALU_FP = (("fmuli", "shfi"), ("fmuli", "prmti"), ("fmuli", "shfi", "prmti"))


def targeted_r4() -> list[dict]:
    """FACET-BREAK r4: the 2-3 accumulator break (FMUL by an immediate, 2 warps a SMSP) on every other MMA form, 8-192
    lane ops a 8 MMAs; the three beating programs again, three copies each spread through the batch; and ALU-pipe work
    beside the MMA (the A and issue rows an ALU-bound fold rests on), alone and beside FMUL."""
    qs = (8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192)
    new: list[dict] = []
    for form in R4_FORMS:
        for na, ws in ((2, (2, 4)), (3, (2, 4)), (4, (2,))):
            for q in qs:
                for w in ws:
                    new.append(_jm(form, "fmuli", q, w, na=na))
        for q in (64, 80, 96, 112, 128):
            new.append(_jm(form, "faddi", q, 2, na=3))
    alu: list[dict] = []
    for ls in R4_ALU + R4_ALU_FP:
        for q in (32, 48, 64, 96, 128):
            for na in (3, 8):
                for w in (2, 4):
                    alu.append(_jm("i16832", ls, q, w, na=na))
    for form in ("h16816f16", "h1688t"):
        for ls in ("shfi", "prmti", "axi", ("shfi", "prmti")):
            for q in (32, 48, 64, 96, 128):
                alu.append(_jm(form, ls, q, 2, na=3))
    rest = new + alu
    out: list[dict] = []
    for k in range(3):
        out += [_jm(f, "fmuli", 112, 2, na=na) | {"rep": k + 1} for f, na in R4_REPEAT]
        out += rest[k * len(rest) // 3:(k + 1) * len(rest) // 3]
    return out


#: RT-FACET r1: the schedule axes the ji, jifine and r4 batches left at a single value on the low-accumulator break (2-3
#: accumulators, 2 warps a SMSP, where IMMA.16832 and FMUL by an immediate reach every JI3s row's 0.98): na 5, 1 and 3
#: warps a SMSP, unroll and placement (ji swept them on 8 accumulators only), and the uniform datapath beside the FMA and
#: ALU pipes at low na (ji timed it on 8 accumulators only, where it traded 1:1 with the FMA pipe instead of co-issuing)
RT_CEIL = ("fmuli", "faddi", "ffmai")
RT_UNI = (("fmuli", "uaxi"), ("faddi", "uaxi"), ("ffmai", "uaxi"))
RT_TRI = (("fmuli", "axi", "uaxi"), ("fmuli", "shfi", "uaxi"), ("faddi", "prmti", "uaxi"))
RT_PIPE = (("fmuli", "axi"), ("fmuli", "dp4ai"), ("axi", "dp4ai"), ("fmuli", "imadi"), ("axi", "imadi"))


def targeted_rt() -> list[dict]:
    """RT-FACET r1 beside IMMA.16832 (H1+fx1's binding form): the ceiling of the FMA-pipe lanes at 2-6 accumulators and
    1-4 warps a SMSP around honest's 16-20 lane ops a MMA; unroll 2 and 4 and placements 1-3 on 2-4 accumulators; the
    FMA lane beside the uniform datapath, then FMA, ALU and uniform in turn, at 2-4 accumulators; and two dispatch pipes
    (FMA, ALU, IMAD) at 2-4 accumulators across the binding ratio."""
    i = "i16832"
    out: list[dict] = []
    for lane in RT_CEIL:
        for na in ((2, 3, 4, 5, 6) if lane != "ffmai" else (3, 4, 5)):
            for q in (96, 112, 128, 144, 160, 192):
                for w in (1, 2, 3, 4):
                    out.append(_jm(i, lane, q, w, na=na))
    for lane in ("fmuli", "faddi"):
        for na in (2, 3, 4):
            for q in (112, 128, 160):
                for w in (2, 4):
                    for u in (2, 4):
                        out.append(_jm(i, lane, q, w, na=na, u=u))
    for na in (2, 3, 4):
        for q in (112, 128, 160):
            for w in (2, 4):
                for p in (1, 2, 3):
                    out.append(_jm(i, "fmuli", q, w, p=p, na=na))
    for ls in RT_UNI:
        for na in (2, 3, 4):
            for q in (96, 128, 160, 192, 224, 256):
                for w in (2, 4):
                    out.append(_jm(i, ls, q, w, na=na))
    for ls in RT_TRI:
        for na in (3, 4):
            for q in (96, 128, 160, 192, 256):
                for w in (2, 4):
                    out.append(_jm(i, ls, q, w, na=na))
    for ls in RT_PIPE:
        for na in (2, 3, 4):
            for q in (112, 128, 144, 160):
                for w in (2, 4):
                    out.append(_jm(i, ls, q, w, na=na))
    return out


#: RT-FACET r1b: r1's two gains crossed (r20261017-183924-3b6d: unroll 4 on 3 accumulators took FMUL to JI3s_0 0.9906,
#: and FFMA beside the uniform datapath on 2 accumulators took JI3s_13 to 0.9919), with unroll 8 for the instruction cache
RT2_LANES = ("fmuli", "faddi", ("ffmai", "uaxi"), ("fmuli", "uaxi"), ("faddi", "uaxi"), ("fmuli", "axi", "uaxi"))


def targeted_rt2() -> list[dict]:
    """RT-FACET r1b beside IMMA.16832: the FMA-pipe lanes alone and beside the uniform datapath at unroll 2, 4 and 8 on
    2-4 accumulators, 2 warps a SMSP, from the MMA end of JI3s_13 (64 lane ops a 8 MMAs) to JI3s_0's (160); and the
    lanes after each MMA's group (placement 3) at unroll 4."""
    i = "i16832"
    out: list[dict] = []
    for ls in RT2_LANES:
        uni = not isinstance(ls, str)
        for na in (2, 3, 4):
            for q in ((64, 80, 96, 112, 128, 160) if uni else (96, 112, 128, 144, 160)):
                for u in (2, 4, 8):
                    out.append(_jm(i, ls, q, 2, na=na, u=u))
                out.append(_jm(i, ls, q, 2, p=3, na=na, u=4))
    return out


def targeted_rt3() -> list[dict]:
    """RT-FACET r2 beside IMMA.16832: the FMA lane past r1b's unroll 8 (r20261017-190803-a98f: FMUL at 3 accumulators, 128
    lane ops a 8 MMAs, loaded JI3s_0 0.9995, its loop 7 uncounted instructions an iteration), at unroll 8 to 32 on 2-4
    accumulators from 112 to 160 lane ops, 2 warps a SMSP; FADD on 3 accumulators; 1, 3 and 4 warps a SMSP and
    placements 1-3 at the best point."""
    i = "i16832"
    out: list[dict] = []
    for na in (2, 3, 4):
        for q in (112, 120, 128, 136, 144, 160):
            for u in (8, 12, 16, 24, 32):
                out.append(_jm(i, "fmuli", q, 2, na=na, u=u))
    for q in (112, 120, 128, 136, 144, 160):
        for u in (8, 16, 32):
            out.append(_jm(i, "faddi", q, 2, na=3, u=u))
    for w in (1, 3, 4):
        for u in (8, 16, 32):
            out.append(_jm(i, "fmuli", 128, w, na=3, u=u))
    for p in (1, 2, 3):
        out.append(_jm(i, "fmuli", 128, 2, p=p, na=3, u=16))
    return out


def targeted_rt4() -> list[dict]:
    """RT-FACET r2b beside IMMA.16832: r2's break again and around it (r20261017-191843-eb32: FMUL at 3 accumulators, 120
    lane ops a 8 MMAs, unroll 16, loaded JI3s_2 1.0127, falling past unroll 16 as the loop outgrows the instruction
    cache): 104-132 lane ops at unroll 10-20 on 3 accumulators, 2 on 2, FADD and FFMA beside, and 1 and 3 warps a SMSP."""
    i = "i16832"
    out: list[dict] = []
    for q in (104, 108, 112, 116, 118, 120, 122, 124, 126, 128, 132):
        for u in (10, 12, 14, 16, 18, 20):
            out.append(_jm(i, "fmuli", q, 2, na=3, u=u))
    for q in (116, 120, 124):
        for u in (14, 16, 18, 20):
            out.append(_jm(i, "fmuli", q, 2, na=2, u=u))
        for u in (14, 16, 18):
            out.append(_jm(i, "faddi", q, 2, na=3, u=u))
    for q in (112, 120, 128):
        for u in (8, 16):
            out.append(_jm(i, "ffmai", q, 2, na=3, u=u))
    for w in (1, 3):
        out.append(_jm(i, "fmuli", 120, w, na=3, u=16))
    return out


def targeted_rt5() -> list[dict]:
    """RT-FACET r3: r2's unroll on every other MMA form, whose VF3s-16 hull is still FACET-BREAK r4's unroll-1 programs
    (r20261017-100812-180b): FMUL beside each at unroll 8 and 16 on 2 and 3 accumulators, 96-128 lane ops a 8 MMAs, 2
    warps a SMSP, and 4 at the vector end (128 on 2 accumulators)."""
    out: list[dict] = []
    for form in R4_FORMS + ("h16816f16", "h1688t"):
        for na in (2, 3):
            for q in (96, 112, 128):
                for u in (8, 16):
                    out.append(_jm(form, "fmuli", q, 2, na=na, u=u))
        for u in (8, 16):
            out.append(_jm(form, "fmuli", 128, 4, na=2, u=u))
    return out


#: the forms r3 (r20261017-193742-1973) took past VF3s-16 at its grid's edge (128 lane ops, unroll 16), and the rest
RT6_BEAT = ("isp16864", "hsp16832", "qsp16864", "isp16832", "i16816", "qsf16832")
RT6_REST = ("h16816", "h16816f16", "h1684t", "h1688t", "q16832", "osf16864")


def targeted_rt6() -> list[dict]:
    """RT-FACET r3b: r3's vector-end break past its grid's edge, on 2 accumulators beside FMUL: the forms r3 beat at
    128-192 lane ops a 8 MMAs, unroll 16-24, 2 and 4 warps a SMSP; the rest at 128 and 160, unroll 16 and 20."""
    out: list[dict] = []
    for form in RT6_BEAT:
        for q in (128, 144, 160, 192):
            for w in (2, 4):
                for u in (16, 20, 24):
                    out.append(_jm(form, "fmuli", q, w, na=2, u=u))
    for form in RT6_REST:
        for q in (128, 160):
            for w in (2, 4):
                for u in (16, 20):
                    out.append(_jm(form, "fmuli", q, w, na=2, u=u))
    return out


def targeted_rt7() -> list[dict]:
    """RT-FACET r4: the warp-specialized programs with the lane warps' trip widened. FACET-BREAK's ran 16 lane ops a
    trip, beside the trip's uncounted stop poll, counter and branch; here 64-256 lane ops a trip beside FMUL, on the
    forms r3 and r3b beat and IMMA.16832, 1-2 MMA warps a SMSP among 2-4, 4 and 8 accumulators, 32 and 64 MMAs a trip."""
    out: list[dict] = []
    for form in ("isp16864", "hsp16832", "qsp16864", "i16832"):
        for mw, w in ((1, 2), (1, 3), (2, 4)):
            for na in (4, 8):
                for u in (4, 8):
                    for lq in (64, 128, 256):
                        out.append(_js(form, "fmuli", mw, w, na=na, u=u, lq=lq))
    return out


#: RT-FACET r5's two-form pairs beside FMUL: IMMA.16832 (H1+fx1's form) with each form r3 and r3b beat and two dense
#: ones, and sparse beside sparse
RT8_LANE_PAIRS = (("i16832", "isp16864"), ("i16832", "hsp16832"), ("i16832", "qsp16864"), ("i16832", "h16816"),
                  ("i16832", "q16832"), ("i16832", "h16816f16"), ("isp16864", "hsp16832"), ("isp16864", "qsp16864"))
#: and alone (one tensor peak for every form: DS-120's DS_tensor, A3, and the measured M): those, IMMA.16832 with the
#: rest, and each sparse form beside its dense one
RT8_TENSOR_PAIRS = RT8_LANE_PAIRS + (("i16832", "i16816"), ("i16832", "isp16832"), ("i16832", "h1688t"),
                                     ("i16832", "osf16864"), ("i16832", "qsf16832"), ("q16832", "qsp16864"),
                                     ("h16816", "hsp16832"), ("q16832", "osf16864"), ("h16816f16", "h1688t"))
#: two opcodes DS-120 prices on one pipe row (A4): MUFU and POPC on DS_xu, IMAD and IDP.4A on DS_imad
RT8_PIPE_PAIRS = (("mufu", "popc"), ("imad", "dp4a"), ("imadi", "dp4ai"))
RT8_BEST = ("jm_isp16864_fmuli_q128_p0_a2u20_w8", "jm_i16832_fmuli_q120_p0_a3u20_w8")


def targeted_rt8() -> list[dict]:
    """RT-FACET r5: two MMA forms in turn in one loop, never timed before (FACET-BREAK's ji `jx` programs were added after
    its runs): the lane pairs beside FMUL at r3b's vector-end break (112-144 lane ops a 8 MMAs, 1-3 accumulators a form,
    unroll 8-20, 2 warps a SMSP, and 4 at 128); every tensor pair alone at 2-8 accumulators, unroll 1 and 4, 2-8 warps a
    SMSP; the pipe pairs alone and beside IMMA.16832; and interleaved baselines: each form and pipe lane alone, and r3b's
    two best single-form programs, three copies each through the batch."""
    lanes: list[dict] = []
    for f1, f2 in RT8_LANE_PAIRS:
        for q in (112, 120, 128, 144):
            for na in (1, 2):
                for u in (8, 16, 20):
                    lanes.append(_jx(f1, f2, "fmuli", q, 2, na=na, u=u))
        for q in (120, 128):
            lanes.append(_jx(f1, f2, "fmuli", q, 2, na=3, u=16))
        for na in (1, 2):
            lanes.append(_jx(f1, f2, "fmuli", 128, 4, na=na, u=16))
    tensor = [_jx(f1, f2, (), 0, w, na=na, u=u) for f1, f2 in RT8_TENSOR_PAIRS
              for na, u, w in ((2, 4, 2), (4, 1, 2), (4, 1, 4), (4, 4, 4), (8, 1, 4), (8, 4, 2))]
    pipes: list[dict] = []
    for l1, l2 in RT8_PIPE_PAIRS:
        for tpb in (256, 512, 1024):
            pipes.append({"kind": "alone", "l1": l1, "l2": l2, "tpb": tpb})
        for q in (16, 32, 64):
            for w in (2, 4):
                pipes.append(_jm("i16832", (l1, l2), q, w, na=4))
    base = [{"kind": "alone", "l1": x, "l2": "", "tpb": 512} for x in ("mufu", "popc", "imad", "dp4a", "imadi", "dp4ai")]
    base += [{"kind": "mma", "form": f, "l1": "", "l2": "", "tpb": 512}
             for f in sorted({f for p in RT8_TENSOR_PAIRS for f in p})]
    best = [g for g in targeted_rt4() + targeted_rt6() if name_of(g) in RT8_BEST]
    rest = lanes + tensor + pipes
    out: list[dict] = []
    for k in range(3):
        out += [g | {"rep": k + 1} for g in best] + (base if k == 1 else [])
        out += rest[k * len(rest) // 3:(k + 1) * len(rest) // 3]
    return out


#: RT3-120's lane classes, which JI3t_0 pools at FMUL's coefficient and no fit run put beside a MMA; `sampled` leaves
#: them out
RT9_LANES = ("shfl",)
#: the load and store kinds RT3-120 puts at the binding schedule: the fit's table loads, and a GEMM's operand staging
RT9_LOADS = ("lds32", "lds128", "ldg32", "ldg128", "tld1", "ldsm4", "sts128")
RT9_BASE = "jm_i16832_fmuli_q120_p0_a3u20_w8"


def targeted_rt9() -> list[dict]:
    """RT3-120 against VF3s-16's JI3t_0 (the measured headline's π*), which prices every vector, memory and atomic class
    at FMUL's coefficient: what its fit runs never put at the binding schedule (IMMA.16832 on 2-3 accumulators, unroll
    4-16, 2 warps a SMSP, where FMUL beside it is tight). Each load and store kind at 2-8 a 8 MMAs, alone and under 96-120
    FMUL; SHFL alone and alternating with FMUL; the 64-bit lanes (IMAD.WIDE, IADD.64) alone and beside FMUL;
    HMMA.16816.F16 (MMA's least) beside the shared kinds at 2 and 4 warps; and RT-FACET's tight program, three copies
    through the batch."""
    i = "i16832"
    rest: list[dict] = []
    for d in RT9_LOADS:
        for na in (2, 3):
            for u, qd in ((4, 8), (8, 2), (8, 4), (16, 2)):
                for ls, q in (((), 0), ("fmuli", 96), ("fmuli", 112), ("fmuli", 120)):
                    rest.append(_jm(i, ls, q, 2, na=na, u=u, d=d, qd=qd))
    for na in (2, 3):
        for u in (8, 16):
            for ls, q in (("shfl", 16), ("shfl", 32), ("shfl", 64), (("fmuli", "shfl"), 96), (("fmuli", "shfl"), 120),
                          (("fmuli", "shfl"), 128)):
                rest.append(_jm(i, ls, q, 2, na=na, u=u))
            for ls in ("imadwi", "iadd64", ("fmuli", "imadwi"), ("fmuli", "iadd64")):
                for q in (96, 120):
                    rest.append(_jm(i, ls, q, 2, na=na, u=u))
    for d in ("lds128", "ldsm4", "sts128"):
        for u, qd in ((8, 2), (8, 4)):
            for ls, q in (((), 0), ("fmuli", 128), ("fmuli", 160)):
                for w in (2, 4):
                    rest.append(_jm("h16816f16", ls, q, w, na=2, u=u, d=d, qd=qd))
    base = [g for g in targeted_rt4() if name_of(g) == RT9_BASE]
    out: list[dict] = []
    for k in range(3):
        out += [g | {"rep": k + 1} for g in base] + rest[k * len(rest) // 3:(k + 1) * len(rest) // 3]
    return out


#: the ALU- and IMAD-pipe lanes JI3t_0 pools at FMUL's coefficient, which r4 and RT-FACET r1 timed beside IMMA.16832 only
#: at unroll 1: alone, and alternating with FMUL
RT10_LANES = ("axi", "shfi", "prmti", ("fmuli", "axi"), ("fmuli", "shfi"), ("fmuli", "prmti"), ("fmuli", "imadi"),
              ("fmuli", "dp4ai"))


def targeted_rt10() -> list[dict]:
    """RT3-120 against JI3t_0 again: the ALU-pipe lanes (IADD3/LOP3, SHF and PRMT by an immediate) and IMAD-pipe ones
    beside FMUL at the binding schedule (IMMA.16832 on 2-3 accumulators, unroll 16 and 20, 2 warps a SMSP, 112-144 lane
    ops a 8 MMAs), where r4 and RT-FACET r1 never took them past unroll 1; and RT-FACET's tight program, three copies
    through the batch."""
    rest = [_jm("i16832", ls, q, 2, na=na, u=u) for ls in RT10_LANES for na in (2, 3) for u in (16, 20)
            for q in (112, 128, 144)]
    base = [g for g in targeted_rt4() if name_of(g) == RT9_BASE]
    out: list[dict] = []
    for k in range(3):
        out += [g | {"rep": k + 1} for g in base] + rest[k * len(rest) // 3:(k + 1) * len(rest) // 3]
    return out


#: the uniform datapath, which JI3t_0 prices below every vector class (UNI 0.714566 against 0.963546) and RT-FACET r1 and
#: r1b timed beside IMMA.16832 only up to unroll 8: alternating with each FMA-pipe lane, with FMUL and the ALU lane, and
#: alone
RT11_PAIRS = (("fmuli", "uaxi"), ("faddi", "uaxi"), ("ffmai", "uaxi"))


def targeted_rt11() -> list[dict]:
    """RT3-120 against JI3t_0 a third time: the uniform datapath (UIADD3/ULOP3) at the binding schedule (IMMA.16832 on 2-3
    accumulators, unroll 12-20, 2 warps a SMSP), where RT-FACET r1 and r1b never took it past unroll 8: alternating with
    FMUL, FADD and FFMA at 96-176 lane ops a 8 MMAs, with FMUL and the ALU lane at 120 and 144, and alone at 32-96; and
    RT-FACET's tight program, three copies through the batch."""
    i = "i16832"
    rest: list[dict] = []
    for na in (2, 3):
        for u in (12, 16, 20):
            rest += [_jm(i, ls, q, 2, na=na, u=u) for ls in RT11_PAIRS for q in (96, 120, 144, 176)]
            rest += [_jm(i, ("fmuli", "axi", "uaxi"), q, 2, na=na, u=u) for q in (120, 144)]
            rest += [_jm(i, "uaxi", q, 2, na=na, u=u) for q in (32, 64, 96)]
    base = [g for g in targeted_rt4() if name_of(g) == RT9_BASE]
    out: list[dict] = []
    for k in range(3):
        out += [g | {"rep": k + 1} for g in base] + rest[k * len(rest) // 3:(k + 1) * len(rest) // 3]
    return out


def sampled_ji(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    lanes = sorted(set(JI_NEW) | set(JI_OLD))
    out = []
    while len(out) < n:
        form = rng.choice(("i16832", "i16832", "h16816f16", "h1688t"))
        ls = tuple(rng.sample(lanes, rng.choice((1, 2, 2, 3))))
        w = rng.choice((1, 2, 3, 4, 5, 6, 8))
        na, u = rng.choice(((8, 1), (8, 1), (4, 1), (16, 1), (8, 2), (16, 2), (8, 4)))
        if na == 16 and w > 4:
            continue
        if rng.random() < 0.8:
            out.append(_jm(form, ls, rng.choice((32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256)), w,
                           p=rng.choice((0, 0, 1, 2, 3)), na=na, u=u))
        else:
            ls = tuple(x for x in ls if x not in ("axi", "iadd3", "uaxi")) or ("ffmai",)
            mw = rng.choice([m for m in (1, 2, 3, 4) if m < w] or [1])
            if mw < w:
                out.append(_js(form, ls, mw, w, na=na, u=u))
    return out


def grid(seed: int = 0, n: int = 0, targeted_part: bool = True, family: str = "mix") -> list[dict]:
    """The programs of one batch: the family's targeted part (if asked) and `n` sampled ones, unique by name, in order."""
    seen, out = set(), []
    tg, sm = {"mix": (targeted, sampled), "mem": (targeted_mem, sampled_mem), "fine": (targeted_fine, sampled),
              "l2": (targeted_l2, sampled_l2), "ji": (targeted_ji, sampled_ji),
              "jifine": (targeted_jifine, sampled_ji), "jipair": (targeted_jipair, sampled_ji),
              "r4": (targeted_r4, sampled_ji), "rt": (targeted_rt, sampled_ji),
              "rt2": (targeted_rt2, sampled_ji), "rt3": (targeted_rt3, sampled_ji),
              "rt4": (targeted_rt4, sampled_ji), "rt5": (targeted_rt5, sampled_ji),
              "rt6": (targeted_rt6, sampled_ji), "rt7": (targeted_rt7, sampled_ji),
              "rt8": (targeted_rt8, sampled_ji), "rt9": (targeted_rt9, sampled_ji),
              "rt10": (targeted_rt10, sampled_ji), "rt11": (targeted_rt11, sampled_ji)}[family]
    for g in (tg() if targeted_part else []) + sm(seed, n):
        nm = name_of(g)
        if nm not in seen:
            seen.add(nm)
            out.append(g | {"name": nm})
    return out


def smem_of(g: dict) -> tuple[str, str]:
    """(smem_ops, smem function) for the gate's bank check: the loop's shared loads in slot order, and mem.cuh's soff."""
    if g["kind"] == "jm":
        if not g.get("d") or not LOADS[g["d"]][3]:
            return "", ""
        n = g["u"] * g["qd"]
        return ",".join([f"{LOADS[g['d']][1]}*1"] * n), f"fb::soff<fb::{LOADS[g['d']][0]}, fb::D_NONE, 1, fb::P_LINE, {n}>"
    if g["kind"] not in ("ld", "ldmma"):
        return "", ""
    ops = [LOADS[load_kinds(g, q)][1] for q in range(g["q"]) if LOADS[load_kinds(g, q)][3]]
    if not ops:
        return "", ""
    d2 = LOADS[g["d2"]][0] if g.get("d2") else "D_NONE"
    fn = f"fb::soff<fb::{LOADS[g['d1']][0]}, fb::{d2}, {g['r']}, fb::{PATS[g['pat']]}, {g['q']}>"
    return ",".join(f"{op}*1" for op in ops), fn


def emit_cu(gs: list[dict]) -> str:
    inc = (HERE / "ji.cuh").resolve()
    lines = [f"// generated by facet_break/generate.py: {len(gs)} programs", f'#include "{inc}"', ""]
    for g in gs:
        f = FORMS[g["form"]][0] if g.get("form") else ""
        if g["kind"] in ("jm", "js", "jx"):
            ls = [LANES[x][0] for x in lanes_of(g)]
            ls += ["L_NONE"] * (3 - len(ls))
            if g["kind"] == "jx":
                lines.append(f"FB_JMIX2({g['name']}, {f}, {FORMS[g['form2']][0]}, {g['na']}, {g['u']}, {', '.join(ls)}, "
                             f"{g['q']}, {g['tpb']})")
                continue
            head = f"{g['name']}, {f}, {g['na']}, {g['u']}, {', '.join(ls)}"
            if g["kind"] == "jm":
                d = LOADS[g["d"]][0] if g.get("d") else "D_NONE"
                lines.append(f"FB_JMIX({head}, {g['q']}, {g['p']}, {d}, {g.get('qd', 0)}, {g['tpb']})")
            else:
                lines.append(f"FB_JSPEC({head}, {g['mw']}, {g['lq']}, {g['tpb']})")
            continue
        if g["kind"] == "l2":
            lines.append(f"FB_L2({g['name']}, {L2_LOADS[g['d1']][0]}, {L2_PATS[g['pat']]}, {g['q']}, {g['tpb']})")
            continue
        if g["kind"] in ("ld", "ldmma"):
            d1 = LOADS[g["d1"]][0]
            d2 = LOADS[g["d2"]][0] if g.get("d2") else "D_NONE"
            args = f"{d1}, {d2}, {g['r']}, {PATS[g['pat']]}, {g['q']}, {g['tpb']}"
            lines.append(f"FB_LD({g['name']}, {args})" if g["kind"] == "ld" else f"FB_LDMMA({g['name']}, {f}, {args})")
            continue
        l1 = LANES[g["l1"]][0] if g.get("l1") else "L_NONE"
        l2 = LANES[g["l2"]][0] if g.get("l2") else "L_NONE"
        if g["kind"] == "mix":
            lines.append(f"FB_MIX({g['name']}, {f}, {l1}, {l2}, {g['q']}, {g['p']}, {g['tpb']})")
        elif g["kind"] == "spec":
            lines.append(f"FB_SPEC({g['name']}, {f}, {l1}, {l2}, {g['wm']}, {g['tpb']})")
        elif g["kind"] == "alone":
            lines.append(f"FB_ALONE({g['name']}, {l1}, {l2}, {g['tpb']})")
        else:
            lines.append(f"FB_MMA({g['name']}, {f}, {g['tpb']})")
    lines += ["", "static const OpProbe PROBES[] = {"]
    for g in gs:
        ex, warps, until = expect_of(g)
        iters = 1024 if g["kind"] in ("mix", "mma", "ldmma") else 2048 if g["kind"] in ("ld", "l2") else 4096
        if g["kind"] in ("jm", "js"):
            iters = (1024 if g["kind"] == "jm" else 4096) // g["u"]
        if g["kind"] == "jx":
            iters = 1024 // g["u"]
        ops, fn = smem_of(g)
        sm = f', .smem_ops = "{ops}", .smem = {fn}' if ops else ""
        lines.append(f'    {{.name = "{g["name"]}", .kernel = (const void*){g["name"]}, .tpb = {g["tpb"]}, '
                     f'.expect = "{ex}", .warps = "{warps}"{sm}, .iters = {iters}, .until = {until}}},')
    lines += ["};", "", "int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=0, help="sampled programs beside the targeted part")
    p.add_argument("--no-targeted", action="store_true")
    p.add_argument("--family", choices=("mix", "mem", "fine", "l2", "ji", "jifine", "jipair", "r4", "rt", "rt2", "rt3",
                                             "rt4", "rt5", "rt6", "rt7", "rt8", "rt9", "rt10", "rt11"),
                   default="mix",
                   help="mix: MMA forms and lane ops (mix.cuh); mem: the table routes' loads (mem.cuh); fine: the "
                        "near-frontier sweep of the lanes the first batch beat or nearly beat; l2: L2-resident loads "
                        "against the table route's L2 facet; ji: the joint issue rows' schedule search (ji.cuh); "
                        "jifine: the sweep around its break; jipair: two dispatch pipes beside the HMMA forms; r4: "
                        "the break on every other form, its repeats, and ALU work beside the MMA; rt: RT-FACET's "
                        "schedule axes on the low-accumulator break (na 5, unroll, placement, uniform co-issue); rt2: "
                        "rt's unroll and uniform co-issue crossed, unroll 8 for the instruction cache; rt3: the FMA "
                        "lane at unroll 8 to 32, past rt2's best; rt4: rt3's break again and around it; rt5: rt3's unroll on every other MMA form; rt6: rt5's break past its "
                        "grid's edge; rt7: the warp-specialized programs with 64-256 lane ops a trip; rt8: two MMA "
                        "forms in turn in one loop, beside FMUL and alone, and the pipe pairs DS-120 prices together; "
                        "rt9: loads, LDSM, STS, SHFL and the 64-bit lanes at rt4's binding schedule (JI3t_0's pooled "
                        "classes); rt10: the ALU- and IMAD-pipe lanes there, alone and beside FMUL; rt11: the "
                        "uniform datapath there, alone and beside the FMA-pipe lanes")
    p.add_argument("--forms", help="keep only the programs of these MMA forms (comma-separated, e.g. i16832)")
    p.add_argument("--only", type=Path, help="keep only the programs named in this file (one a line): the gate's passes")
    p.add_argument("--emit", type=Path, help="write the probe source here (default: print the programs as JSON lines)")
    a = p.parse_args(argv)
    gs = grid(a.seed, a.n, not a.no_targeted, a.family)
    if a.forms:
        gs = [g for g in gs if g.get("form") in a.forms.split(",") or g.get("form2") in a.forms.split(",")]
    if a.only:
        keep = set(a.only.read_text().split())
        gs = [g for g in gs if g["name"] in keep]
    if a.emit:
        a.emit.write_text(emit_cu(gs))
        print(f"{len(gs)} programs -> {a.emit}", file=sys.stderr)
    else:
        for g in gs:
            print(json.dumps(g))
    return 0


if __name__ == "__main__":
    sys.exit(main())
