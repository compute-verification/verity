"""FACET-BREAK: a measured program's load on every facet of Π, the facets it beats, and the refit a beaten facet needs.

The standing adversary for the sm_120 data sheet (note:20261017T0524Z-draft-eps1-facet-break). A facet is a row of the
machine model (`../machine_model/data/`): row r holds Σ_k c_kr·x_k SMSP-cycles for a program that issues x_k warp
instructions of census class k a SMSP-cycle, and the facet is the ceiling a_r·x ≤ 1. A program's **load** on r is
Σ_k c_kr·x_k (dimensionless): a measured program with load > 1 beats the facet, since the rows fitted from rates carry the
2% margin in their coefficients (a point measured on its own facet sits at 0.98). Exact rows (`issue` 1 cycle, `M` 16) are
reported the same way; a load in (1, 1.02] on them is inside timing noise.

The rate of a census class k in a probe's `rows.jsonl` row is the sum, over every timed loop, of that loop's rate of
every SASS opcode of class k (warp instructions/SM/clk), divided by 4 (SMSPs an SM) to get x_k a SMSP-cycle. `CLASS_OF`
maps a SASS opcode to its census class. Only opcodes whose class is certain are counted, so a reported load is a lower
bound and a reported violation is real.

  python loads.py loads OUT/rows.jsonl... [--epoch E] [--probe NAME] [--top N]   # each program's heaviest facets
  python loads.py summary OUT/rows.jsonl... [--epoch E] [--margin 0] [--json]    # every violation (load > 1 + margin)
  python loads.py refit OUT/rows.jsonl... --row R [--epoch E]                    # the refit line a violation needs
  python loads.py table OUT/rows.jsonl... [--epoch E]                            # every program's loads, a JSON line each

`--generic` loads every MMA form through the generic `MMA` lines where its own kind has none (advisory). The generator of
the programs is `generate.py`, the batch `run.sh`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MM = HERE.parent / "machine_model"
sys.path.insert(0, str(MM))

from verity_machine_model.model import census, model, rows as mm_rows  # noqa: E402

SMSPS = 4

#: a SASS opcode (the dotted name, no guard) to its census class. The MMA forms map to their own kind (for the JI3n rows,
#: each form's own coefficient); every MMA form also loads the generic-pipe rows through `MMA` (see `mma_kind`). The order
#: matters: the first matching rule wins.
CLASS_RULES: list[tuple[str, str]] = [
    # tensor forms (the exact mma.sync SASS)
    (r"IMMA\.16832", "IMMA16832"), (r"IMMA\.16816", "IMMA16816"), (r"IMMA\.8816|IMMA\.1688", "IMMA8816"),
    (r"IMMA\.SP\.16864", "IMMASP16864"), (r"IMMA\.SP\.16832", "IMMASP16832"),
    (r"HMMA\.16816\.F16", "HMMA16816F16"), (r"HMMA\.16816", "HMMA16816"),
    (r"HMMA\.SP\.16832", "HMMASP16832"), (r"HMMA\.1688", "HMMA1688T"), (r"HMMA\.1684", "HMMA1684T"),
    (r"QMMA\.SF\.16832", "QMMASF16832"), (r"QMMA\.SP\.16864", "QMMASP16864"), (r"QMMA\.16832", "QMMA16832"),
    (r"OMMA\.SF\.16864|OMMA", "OMMASF16864"),
    # scalar / CUDA lane classes (floors.csv and census share these names)
    (r"IMAD\.WIDE", "IMADW"), (r"IADD\.64", "ALU64"), (r"DFMA|DADD", "W2"), (r"DMUL", "DMUL"),
    (r"IMAD|IMUL|IADD3|IADD|LEA\b|LEA\.HI", "ALU_OR_IMAD"),  # resolved below
    (r"LOP3|SHF\b|SHF\.|PRMT|LOP\b", "ALU"),
    (r"HFMA2", "HFMA2"), (r"HADD2|HMUL2", "HFMA2"),
    (r"FFMA|FADD|FMUL", "FFMA"),
    (r"VIMNMX|IMNMX|VMNMX|MNMX", "VIMNMX"),
    (r"F2F|F2I|I2F|I2I|FRND|F2FP", "F2FS"),
    (r"MUFU", "XU"),
    (r"SHFL", "SHFL"),
    (r"SEL\b|SEL\.", "SEL"),
    (r"POPC|FLO|BREV", "XU"),   # POPC on the XU (the bit-slice / XOR-popc route's counter)
]

#: mad.lo.u32 disassembles to IMAD on sm_120; IADD3/LOP3/SHF/PRMT are the ALU group. `IMAD` is its own census class.
_IMAD = re.compile(r"\bIMAD\b|\bIMUL\b")
_ALU = re.compile(r"\bIADD3\b|\bIADD\b|\bLOP3\b|\bLOP\b|\bSHF\b|\bPRMT\b|\bLEA\b")


def class_of(opcode: str) -> str | None:
    """The census class of a dotted SASS opcode (`IMMA.16832.S8.S8`, `LOP3.LUT`, `SHFL.BFLY`), or None when not certain."""
    op = opcode.split("@")[-1].strip()
    base = op.split(".")[0]
    for pat, kind in CLASS_RULES:
        if re.search(pat, op):
            if kind == "ALU_OR_IMAD":
                return "IMAD" if _IMAD.search(op) else "ALU" if _ALU.search(op) else None
            return kind
    if base in ("MOV", "UMOV"):
        return "NOROW"
    return None


MMA_FORMS = frozenset({"IMMA16832", "IMMA16816", "IMMA8816", "IMMASP16864", "IMMASP16832", "HMMA16816F16", "HMMA16816",
                       "HMMASP16832", "HMMA1688T", "HMMA1684T", "QMMASF16832", "QMMASP16864", "QMMA16832",
                       "OMMASF16864"})
#: the three joint issue rows a MMA form is priced on by its own kind; on every other row it loads through generic `MMA`
JI3N = ("JI3n_0", "JI3n_1", "JI3n_2")


def rates_by_class(row: dict) -> dict[str, float]:
    """x_k a SMSP-cycle for each census class k the probe's loops touched (its `rates`, keyed `loop:pattern`)."""
    out: dict[str, float] = {}
    for key, rate in row["rates"].items():
        pat = key.split(":", 1)[1]
        k = class_of(pat)
        if k is not None:
            out[k] = out.get(k, 0.0) + rate / SMSPS
    return out


#: the form the generic `MMA` lines were measured with (the honest kernel's mma.sync): it loads every row its own kind has
#: no line on through `MMA`. Every other form is its own kind with lines on M, issue and JI3n only, so the model prices it
#: at 0 on the rest; `generic=True` loads it through `MMA` there too (advisory: a form's missing line can't be violated).
MMA_MEASURED_WITH = "IMMA16832"


def facet_loads(x: dict[str, float], c: dict[str, dict[str, float]], cycle_u: float,
                generic: bool = False) -> dict[str, float]:
    """A program's load Σ_k c_kr·x_k on every row r. A MMA form takes its own kind's lines and, if it is the form the
    generic `MMA` lines were measured with (or `generic`), `MMA`'s on every row it has none on."""
    coeff: dict[str, dict[str, float]] = {}
    for k, xk in x.items():
        per_cycle = {r: u / cycle_u for r, u in c.get(k, {}).items()}
        if k in MMA_FORMS and (generic or k == MMA_MEASURED_WITH):
            for r, u in c.get("MMA", {}).items():
                if r not in per_cycle and r not in JI3N:
                    per_cycle[r] = u / cycle_u
        coeff[k] = per_cycle
    load: dict[str, float] = {}
    for k, xk in x.items():
        for r, ck in coeff[k].items():
            load[r] = load.get(r, 0.0) + ck * xk
    return load


def program(row: dict, c, cycle_u: float, generic: bool = False) -> dict:
    x = row["x"] if "x" in row else rates_by_class(row)
    load = facet_loads(x, c, cycle_u, generic)
    return {"probe": row.get("probe"), "run": row.get("run"), "sass": (row.get("sass_sha256") or "")[:16],
            "sm_mhz": row.get("sm_mhz"), "tags": row.get("tags", []), "x": x, "load": load}


#: tensor_kinds (EPS1 TENSOR M1, `cursor/eps1-tensor-ea24`): each form name's census kind, and each mode's lane op split
#: over census classes (spec/mixu: IADD3 and FFMA alternating; *_iadd3: IADD3 alone)
TK_KIND = {"imma_s8_m16n8k32": "IMMA16832", "imma_u8s8_m16n8k32": "IMMA16832", "imma_s8_m16n8k32_sat": "IMMA16832",
           "imma_s8_m16n8k16": "IMMA16816", "imma_s8_m8n8k16": "IMMA8816", "imma_sp_s8_m16n8k64": "IMMASP16864",
           "imma_sp_s8_m16n8k32": "IMMASP16832", "hmma_f16_m16n8k16_f32": "HMMA16816",
           "hmma_bf16_m16n8k16_f32": "HMMA16816", "hmma_f16_m16n8k16_f16": "HMMA16816F16",
           "hmma_sp_f16_m16n8k32_f32": "HMMASP16832", "hmma_tf32_m16n8k8": "HMMA1688T", "hmma_tf32_m16n8k4": "HMMA1684T",
           "qmma_e4m3_m16n8k32_f32": "QMMA16832", "qmma_e5m2_m16n8k32_f32": "QMMA16832",
           "qmma_e2m1_m16n8k32_f32": "QMMA16832", "qmma_sf_e4m3_m16n8k32_f32": "QMMASF16832",
           "qmma_sp_e4m3_m16n8k64_f32": "QMMASP16864", "omma_nvf4_m16n8k64": "OMMASF16864"}
TK_LANE = {"iadd3": {"ALU": 1.0}, "iadd3_32": {"ALU": 1.0}, "ffma": {"FFMA": 1.0},
           "iadd3_ffma": {"ALU": 0.5, "FFMA": 0.5}}


def from_tensor_kinds(d: dict, run: str, mixed_lane: str = "iadd3_ffma") -> dict | None:
    """A tensor_kinds.jsonl record as a probe row: its MMAs and lane ops a SM over its median block's SM cycles. Only the
    PTX lane ops are counted (not the loop's own IADD3/ISETP/BRA), so the load is a lower bound. `mixed_lane` is the
    lane of the file's spec/mixu modes: IADD3 alone in the first version (de1d918, its opalone form `iadd3_32`)."""
    if "mode" not in d or not d.get("cyc_med"):
        return None
    cyc = d["cyc_med"]
    if d["mode"] == "opalone":
        lane, form = d["form"], None
    else:
        lane, form = ("iadd3" if d["mode"].endswith("_iadd3") else mixed_lane), d["form"]
    x: dict[str, float] = {}
    if form:
        x[TK_KIND[form]] = d["mmas_per_sm"] / cyc / SMSPS
    for k, share in TK_LANE[lane].items():
        if d["ops_per_sm"]:
            x[k] = x.get(k, 0.0) + share * d["ops_per_sm"] / cyc / SMSPS
    name = f"{d['mode']}:{form or lane}:R{round(d.get('ops_per_mma', 0) * 8)}:w{d['mma_warps']}+{d['op_warps']}"
    return {"probe": name, "run": run, "x": x, "sm_mhz": d.get("mhz"), "tags": []}


def load_rows(paths: list[Path]) -> list[dict]:
    """Every program in the given files: an opcode probe's `rows.jsonl` (rates by SASS pattern) or tensor_kinds.jsonl."""
    out = []
    for p in paths:
        run = next((s for s in p.parts if s.startswith("r20")), "")
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        mixed = "iadd3_32" if any(d.get("mode") == "opalone" and d.get("form") == "iadd3_32" for d in recs) \
            else "iadd3_ffma"
        for d in recs:
            if d.get("rates"):
                out.append(d)
            elif (t := from_tensor_kinds(d, run, mixed)) is not None:
                out.append(t)
    return out


def in_vector(epoch: str) -> set[str]:
    """The rows the named vector's price LP (and CALC's ε LP) reads, so a violation of one is flagged as ε-reading."""
    from verity_machine_model.model import vector, vectors
    for n in vectors():
        v = vector(n)
        if v.epoch == epoch:
            return set(v.rows)
    return set()


def cmd_loads(a) -> int:
    m = model()
    epoch = a.epoch or m.current
    c = census(m, epoch)
    for row in load_rows(a.rows):
        p = program(row, c, m.cycle_u, a.generic)
        top = sorted(p["load"].items(), key=lambda t: -t[1])[: a.top]
        if a.probe and p["probe"] != a.probe:
            continue
        mix = ", ".join(f"{k} {v:.4f}" for k, v in sorted(p["x"].items(), key=lambda t: -t[1]))
        print(f"{p['probe']:28s} {p['sm_mhz'] or 0:>5}MHz  {mix}")
        print("    " + "  ".join(f"{r} {v:.3f}" for r, v in top))
    return 0


def cmd_table(a) -> int:
    """Every program as a JSON line: its mix, its load on every row and its heaviest row (the results table)."""
    m = model()
    epoch = a.epoch or m.current
    c = census(m, epoch)
    vrows = in_vector(epoch)
    for row in load_rows(a.rows):
        p = program(row, c, m.cycle_u, a.generic)
        top = max(p["load"].items(), key=lambda t: t[1]) if p["load"] else ("", 0.0)
        print(json.dumps({"epoch": epoch, **p, "max_row": top[0], "max_load": top[1], "max_in_vector": top[0] in vrows,
                          "load": {r: round(v, 6) for r, v in sorted(p["load"].items())}}))
    return 0


def cmd_summary(a) -> int:
    m = model()
    epoch = a.epoch or m.current
    c = census(m, epoch)
    vrows = in_vector(epoch)
    rowinfo = mm_rows()
    viol = []
    for row in load_rows(a.rows):
        p = program(row, c, m.cycle_u, a.generic)
        for r, v in p["load"].items():
            if v > 1 + a.margin:
                viol.append({"row": r, "load": v, "probe": p["probe"], "run": p["run"], "sass": p["sass"],
                             "sm_mhz": p["sm_mhz"], "in_vector": r in vrows, "tags": p["tags"], "x": p["x"]})
    viol.sort(key=lambda d: -d["load"])
    if a.json:
        print(json.dumps(viol, indent=1))
        return 1 if viol else 0
    print(f"epoch {epoch}: {len(load_rows(a.rows))} programs, margin {a.margin}")
    if not viol:
        print("no facet beaten (every program loads <= 1 + margin on every row)")
        return 0
    worst: dict[str, dict] = {}
    for d in viol:
        worst.setdefault(d["row"], d)
    print(f"{len(worst)} rows beaten ({len(viol)} program-row violations); the worst program of each:")
    for d in worst.values():
        n = sum(v["row"] == d["row"] for v in viol)
        print(f"  {d['row']:12s} {d['load']:.4f}{' [ε]' if d['in_vector'] else '    '} x{n:<4d} {d['probe']} @ {d['run']}")
    if a.worst:
        return 1
    print("every violation:")
    for d in viol:
        star = " [ε]" if d["in_vector"] else ""
        mix = ", ".join(f"{k} {x:.4f}" for k, x in sorted(d["x"].items(), key=lambda t: -t[1]))
        print(f"  {d['row']:12s} load {d['load']:.4f}{star}  {d['probe']} @ {d['run']} sass {d['sass']} "
              f"({d['sm_mhz']}MHz{', ' + ','.join(d['tags']) if d['tags'] else ''})")
        print(f"      mix: {mix}")
    return 1


def cmd_refit(a) -> int:
    """The refit a beaten row needs. A row over MMA and one other class (R1, KFFMA0, KALU*, EIMAD*, …) gets the upper hull
    of every measured point (x_MMA, x_X) and the old row's MMA-axis end, at the margin (`eps1.facets`, the rule VF3m-16
    and VF3n-16 refit by); any other row gets its coefficients scaled so the worst program loads 1 - margin."""
    m = model()
    epoch = a.epoch or m.current
    c = census(m, epoch)
    progs = [program(row, c, m.cycle_u, a.generic) for row in load_rows(a.rows)]
    hits = [p for p in progs if a.row in p["load"]]
    if not hits:
        print(f"no program loads {a.row}", file=sys.stderr)
        return 1
    hit = max(hits, key=lambda p: p["load"][a.row])
    worst = hit["load"][a.row]
    print(f"{a.row}: worst load {worst:.4f} by {hit['probe']} @ {hit['run']} (mix {hit['x']})")
    support = sorted(k for k in c if a.row in c[k])
    others = [k for k in support if k != "MMA" and k not in MMA_FORMS]
    if "MMA" in support and len(others) == 1:
        sys.path.insert(0, str(HERE.parent / "opcode_probe"))
        from eps1 import facets
        X = others[0]
        mma_ok = {MMA_MEASURED_WITH} | (MMA_FORMS if a.generic else set())
        pts = []
        for p in progs:
            if any(k in MMA_FORMS and k not in mma_ok and v > 0 for k, v in p["x"].items()):
                continue
            mm = sum(v for k, v in p["x"].items() if k in mma_ok or k == "MMA")
            if mm > 0 or p["x"].get(X, 0) > 0:
                pts.append((mm, p["x"].get(X, 0.0), f"{p['probe']}@{p['run']}"))
        s = 1 - a.margin
        pts.append((s * m.cycle_u / c["MMA"][a.row], 0.0, f"{a.row}:MMA-axis"))
        pts.append((0.0, s * m.cycle_u / c[X][a.row], f"{a.row}:{X}-axis"))
        print(f"refit: the upper hull of {len(pts)} points (MMA, {X}) a SMSP-cycle, x (1 - {a.margin}):")
        for f in facets(pts, a.margin):
            load = max(f["MMA"] * p[0] + f["X"] * p[1] for p in pts)
            print(f"  MMA {f['MMA']:.3f} + {X} {f['X']:.3f} SMSP-cycles  (tight at {' and '.join(f['tight'])}; "
                  f"max load {load:.3f})")
        return 0
    scale = (1 - a.margin) / worst
    print(f"refit: scale every {a.row} coefficient by {scale:.5f} (so the worst program loads {1 - a.margin:.2f}):")
    for k in support:
        print(f"  {k}: {c[k][a.row] / m.cycle_u:.4f} -> {c[k][a.row] / m.cycle_u * scale:.4f} SMSP-cycles")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("loads", "summary", "refit", "table"):
        s = sub.add_parser(name)
        s.add_argument("rows", type=Path, nargs="+")
        s.add_argument("--epoch")
        s.add_argument("--generic", action="store_true",
                       help="load every MMA form through MMA's lines where it has none (advisory, not the model)")
    sub.choices["loads"].add_argument("--probe")
    sub.choices["loads"].add_argument("--top", type=int, default=4)
    sub.choices["summary"].add_argument("--margin", type=float, default=0.0,
                                        help="report load > 1 + margin (default 0: the rows already carry the 2%%)")
    sub.choices["summary"].add_argument("--json", action="store_true")
    sub.choices["summary"].add_argument("--worst", action="store_true", help="only the worst program of each row")
    sub.choices["refit"].add_argument("--row", required=True)
    sub.choices["refit"].add_argument("--margin", type=float, default=0.02)
    a = p.parse_args(argv)
    sys.path.insert(0, str(HERE))
    return {"loads": cmd_loads, "summary": cmd_summary, "refit": cmd_refit, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
