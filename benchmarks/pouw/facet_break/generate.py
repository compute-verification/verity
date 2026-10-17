"""FACET-BREAK's generator: a seeded grid of MMA + lane programs, emitted as one opcode probe source over `mix.cuh`.

A program is a dict (`kind` mix | spec | alone | mma, the MMA `form`, lane ops `l1`/`l2`, `q` lane ops a 8 MMAs, placement
`p`, `tpb`, and for spec `wm` MMA warps). The grid has two parts:

- **targeted** (the same for every seed): for each lane class X beside IMMA.16832, the vertices of Π's 2-D section in
  (x_MMA, x_X) (every row either loads, the census at the epoch), and programs whose ratio X/MMA sits at each vertex and
  around it (x 0.6 ... 1.6), at 4 warps a SMSP; each lane alone; each MMA form alone at 2, 4 and 8 warps a SMSP; and the
  JI3n family (every form beside IADD3, FFMA, IADD3+FFMA, immediate FFMA, LOP3 and IMAD at 4 to 16 ops a MMA).
- **sampled** (`seed`): `n` programs drawn uniformly over forms, one or two lanes, q, warps, placement and schedule.

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
         "h1684t": ("H1684T", "HMMA.1684.F32.TF32", "HMMA1684T"), "h1688t": ("H1688T", "HMMA.1688.F32.TF32", "HMMA1688T")}
#: lane: (mix.cuh enum, SASS fnmatch pattern, census class)
LANES = {"iadd3": ("L_IADD3", "IADD3", "ALU"),
         "lop3": ("L_LOP3", "LOP3.LUT", "ALU"), "shf": ("L_SHF", "SHF.*", "ALU"), "prmt": ("L_PRMT", "PRMT", "ALU"),
         "imad": ("L_IMAD", "IMAD", "IMAD"), "imadi": ("L_IMADI", "IMAD", "IMAD"),
         "imadw": ("L_IMADW", "IMAD.WIDE.U32", "IMADW"), "iadd64": ("L_IADD64", "IADD.64", "ALU64"),
         "sel": ("L_SEL", "SEL", "SEL"), "ffma": ("L_FFMA", "FFMA", "FFMA"), "ffmai": ("L_FFMAI", "FFMA", "FFMA"),
         "fmul": ("L_FMUL", "FMUL", "FFMA"), "fadd": ("L_FADD", "FADD", "FFMA"),
         "hfma2": ("L_HFMA2", "HFMA2*", "HFMA2"), "mnmx": ("L_MNMX", "*MNMX*", "VIMNMX"),
         "dmul": ("L_DMUL", "DMUL", "DMUL"), "dfma": ("L_DFMA", "DFMA", "W2"), "mufu": ("L_MUFU", "MUFU.EX2", "XU"),
         "popc": ("L_POPC", "POPC", "XU")}
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


def name_of(g: dict) -> str:
    lanes = g["l1"] + (f"_{g['l2']}" if g.get("l2") else "")
    w = g["tpb"] // 32
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
        ls = [g["l1"]] + ([g["l2"]] if g.get("l2") else [])
        for i in range(n):
            pat = LANES[ls[i % len(ls)]][1]
            out[pat] = out.get(pat, 0) + 1
        return out

    def fmt(d: dict[str, int]) -> str:
        return ",".join(f"{k}={v}" for k, v in d.items())

    w = g["tpb"] // 32
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
    lanes = sorted(LANES)
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


def grid(seed: int = 0, n: int = 0, targeted_part: bool = True) -> list[dict]:
    """The programs of one batch: the targeted part (if asked) and `n` sampled ones, unique by name, in order."""
    seen, out = set(), []
    for g in (targeted() if targeted_part else []) + sampled(seed, n):
        nm = name_of(g)
        if nm not in seen:
            seen.add(nm)
            out.append(g | {"name": nm})
    return out


def emit_cu(gs: list[dict]) -> str:
    inc = (HERE / "mix.cuh").resolve()
    lines = [f"// generated by facet_break/generate.py: {len(gs)} programs", f'#include "{inc}"', ""]
    for g in gs:
        f = FORMS[g["form"]][0] if g.get("form") else ""
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
        iters = 1024 if g["kind"] in ("mix", "mma") else 4096
        lines.append(f'    {{.name = "{g["name"]}", .kernel = (const void*){g["name"]}, .tpb = {g["tpb"]}, '
                     f'.expect = "{ex}", .warps = "{warps}", .iters = {iters}, .until = {until}}},')
    lines += ["};", "", "int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=0, help="sampled programs beside the targeted part")
    p.add_argument("--no-targeted", action="store_true")
    p.add_argument("--only", type=Path, help="keep only the programs named in this file (one a line): the gate's passes")
    p.add_argument("--emit", type=Path, help="write the probe source here (default: print the programs as JSON lines)")
    a = p.parse_args(argv)
    gs = grid(a.seed, a.n, not a.no_targeted)
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
