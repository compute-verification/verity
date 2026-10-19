"""Diff a device's SASS opcode census against A7's menu: every opcode the model doesn't price, and whether it can head a
committed word (RT-ISA, the compute-accounting red team).

A7 (`SheetSemantics`, `Security/Models/Accounting/Assumptions/Sheet.lean`) says no unit outside the menu heads a fresh
committed word (NoSum) and a side unit costs at least its per-word vector (SideUnits). The menu is five routes
(isa_escape.py): the tensor route (A3, priced by a form's operand-type peak), the CUDA lane route (q*, `loads.class_of`
gives the census class), the uniform datapath (on issue), table routes (by bytes) and priced side units (the pi32 floor).

This reads the device's opcode vocabulary -- the union of what `cuobjdump` emitted across the corpus (`census.json`) and
NVIDIA's Instruction Set Reference table for the architecture (`data/sass_isa_tables.json`) -- and buckets every opcode:

  PRICED-LANE     `loads.class_of` gives a census class, or the base is a lane/uniform MAC: priced on the q* lane route.
  PRICED-TENSOR   an MMA form whose operand-type family has a peak in the device's census (A3 prices it).
  PRICED-SIDE     a reduction / atomic / texture side unit, priced at the pi32 floor (floors.csv); no multiply (NoSum).
  MOVER           no arithmetic: the output word equals an existing word (a copy / transpose / load / select) (NoSum).
  GAP-TENSOR      an MMA form with NO peak in the device's census: A3 prices it at macs/(0/4) = 0 (DMMA f64, b1, s4).
  GAP-MULTIPLY    a product unit on no menu route: a fresh product the model prices at 0.
  UNDERPRICED     a native paired lane multiply (FFMA2/FMUL2) the census prices as one FMA: charged half its work.
  GAP-WORD        writes a word but forms no product (a SIMD add / cross-lane fold) that the model prices at 0: NoSum
                  (a sum/select of words that already exist) or the FOLD argument must close it.
  HARMLESS        control / predicate / barrier / misc: no committed word.
  UNKNOWN         a base opcode `sass_semantics.py` doesn't classify yet (surfaced for a human).

A GAP row names the cheapest way it could head an int8 (or FP8/FP4) committed word; the rate is a separate measurement.

  python diff_menu.py FAMILY census.json [--out diff.json]     # FAMILY in sm_90 sm_100 sm_120
  python diff_menu.py FAMILY census.json --unused              # also the ISA-table opcodes the corpus never used
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "facet_break"))
sys.path.insert(0, str(HERE))
from loads import class_of  # noqa: E402  the model's own SASS opcode -> census class
from sass_semantics import classify  # noqa: E402

MM = HERE.parent / "machine_model" / "data"
#: the device census file an architecture family reads its priced kinds from
DEVICE_CENSUS = {"sm_120": MM / "census" / "DS-120.csv", "sm_100": MM / "devices" / "sm_100" / "census.csv",
                 "sm_90": MM / "devices" / "sm_90" / "census.csv", "sm_80": MM / "devices" / "sm_80" / "census.csv"}
#: NVIDIA's Instruction Set Reference table an architecture family maps to
ISA_TABLE = {"sm_90": "hopper", "sm_100": "blackwell", "sm_103": "blackwell", "sm_120": "blackwell", "sm_80": "ampere"}
#: an MMA SASS base -> the census-kind token its peak would appear under (QMMA covers FP8/FP6/FP4 mma.sync on sm_120)
MMA_KIND_TOKEN = {"IMMA": "IMMA", "HMMA": "HMMA", "QMMA": "QMMA", "OMMA": "OMMA", "BMMA": "BMMA", "DMMA": "DMMA",
                  "HGMMA": "HGMMA", "IGMMA": "IGMMA", "QGMMA": "QGMMA", "OGMMA": "OGMMA", "BGMMA": "BGMMA",
                  "UTCHMMA": "UTCHMMA", "UTCIMMA": "UTCIMMA", "UTCQMMA": "UTCQMMA", "UTCOMMA": "UTCOMMA",
                  "UTCBMMA": "UTCBMMA"}
#: the cheapest int8/FP word an unpriced form could head, by base opcode (the red-team's "smallest escape")
ESCAPE = {
    "DMMA": "an int8xint8 product is exact in f64 (|a*b|<2^24), so one DMMA.8 heads 8 exact int8 MACs a warp "
            "instruction; with no f64 tensor peak the sheet prices it at 0 (rate: G2/G3 timing vs IMMA s8)",
    "BMMA": "an int8 product is 64 AND-popc bit-planes (D=64); native b1 MMA with no sheet peak is priced at 0, but a "
            "generous 8x-TOPS b1 is only 0.125x int8 (isa_escape DECOMPOSE); emulated on sm_120 (killed)",
    "FFMA2": "two FP32 FMAs an instruction; the census prices it as one FFMA, so an FFMA-bound FP word (filler/fold) is "
             "charged 2x its cost on sm_100a (UNDERPRICED; int8 words head on IMAD/IDP, not FFMA)",
    "FMUL2": "two FP32 multiplies an instruction, priced as one FMUL (UNDERPRICED on sm_100a)",
    "CREDUX": "a coupled cross-lane reduce into a uniform register; no multiply, so as a FOLD it heads 1 word a warp "
              "instruction = 32 lane instructions a word (>> duo's 3); NoSum holds, unpriced",
    "FSWZADD": "FP32 swizzle-add, a fixed cross-lane add; no multiply (a sum of existing words), FOLD argument as redux",
    "REDS": "an exact add reduction to shared memory; no multiply (NoSum), priced-equivalent to RED at the pi32 floor "
            "but has no floors.csv kind of its own",
    "LDGMC": "a reducing / multicast global load (NVLS); an add of existing words with no multiply (NoSum); RT-SCOPE owns "
             "the NVLS switch-side reduction",
    "UTCOMMA": "tcgen05 FP4 (mxf4/nvf4) MMA on the B200; priced via the DC async rows if present, else a tensor gap",
    "SPARSIFY": "compresses an operand to 2:4; a data reshape with no product (MOVER)",
    "DECOMPRESS": "expands a 2:4 operand; a data reshape with no product (MOVER)",
}


def priced_tensor_tokens(family: str) -> set[str]:
    """The MMA-kind tokens that have a row in the device's census (so A3 has a peak for them)."""
    f = DEVICE_CENSUS.get(family)
    if not f or not f.exists():
        return set()
    kinds = {r["kind"] for r in csv.DictReader(f.read_text().splitlines())}
    toks = set()
    for k in kinds:
        for tok in MMA_KIND_TOKEN.values():
            if k.upper().startswith(tok):
                toks.add(tok)
    return toks


def isa_vocab(family: str) -> dict[str, dict]:
    t = json.loads((HERE / "data" / "sass_isa_tables.json").read_text())["tables"]
    return t.get(ISA_TABLE.get(family, ""), {})


#: lane / uniform multiply-accumulate bases: the model prices these on the q* lane route (or on issue), whatever suffix
#: the SASS carries, so a bare base whose `class_of` is None (no discriminating suffix) is still priced.
LANE_MAC_BASE = {"IMAD", "IMUL", "IMUL32I", "IDP", "IDP4A", "FFMA", "FFMA32I", "FMUL", "FMUL32I", "DFMA", "DMUL",
                 "HFMA2", "HMUL2", "UIMAD", "UFFMA", "UFMUL"}


def bucket(op: str, sem: dict, cls: str | None, priced_toks: set[str]) -> str:
    base = op.split(".")[0]
    if base in MMA_KIND_TOKEN:
        return "PRICED-TENSOR" if MMA_KIND_TOKEN[base] in priced_toks else "GAP-TENSOR"
    if sem.get("paired"):
        return "UNDERPRICED"
    if sem["route"] == "control" or sem["word"] is False:
        return "HARMLESS"
    if cls is not None or base in LANE_MAC_BASE:
        return "PRICED-LANE"
    if sem["route"] == "sidefloor":
        return "PRICED-SIDE"
    if sem["route"] == "mover":
        return "MOVER"
    if sem["mul"] is None:
        return "UNKNOWN"
    if sem["mul"]:
        return "GAP-MULTIPLY"
    if sem["word"]:
        return "GAP-WORD"
    return "HARMLESS"


#: a SASS base whose `loads.class_of` census class contradicts its true role -- the lane classifier's UNIFORM regex
#: `U[A-Z].*` swallows tcgen05 (UTC*) and the TMA bulk ops (UBLK*, UTMA*) into UNI. Harmless today (facet_break is
#: sm_120-only and prices these by probe name), but a classifier that ran on B200/H200 SASS would misprice them.
def disagreements(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        base, cls = r["base"], r["class_of"]
        if cls in ("UNI", "UNI64") and base not in ("UNI",) and (base in MMA_KIND_TOKEN or base.startswith(("UBLK", "UTMA", "UTC", "LDGMC"))):
            out.append({"op": r["op"], "class_of": cls, "role": r["route"], "note": r["note"]})
    return out


def diff(family: str, census: dict, with_unused: bool) -> dict:
    fam = census.get(family, {})
    used = fam.get("ops", {})
    vocab = isa_vocab(family)
    priced_toks = priced_tensor_tokens(family)
    names = set(used) | ({o for o in vocab} if with_unused else set())
    rows = []
    for op in sorted(names):
        base = op.split(".")[0]
        sem = classify(base)
        cls = class_of(op)
        b = bucket(op, sem, cls, priced_toks)
        rows.append({"op": op, "base": base, "group": vocab.get(base, {}).get("group") if base in vocab else
                     (vocab.get(op, {}).get("group")), "bucket": b, "mul": sem["mul"], "word": sem["word"],
                     "route": sem["route"], "class_of": cls, "count": used.get(op, {}).get("count", 0),
                     "in_corpus": op in used, "note": sem["note"],
                     "escape": ESCAPE.get(base) if b.startswith("GAP") else None})
    by = {}
    for r in rows:
        by.setdefault(r["bucket"], []).append(r)
    return {"family": family, "isa_table": ISA_TABLE.get(family), "priced_tensor_tokens": sorted(priced_toks),
            "counts": {b: len(v) for b, v in sorted(by.items())}, "disagreements": disagreements(rows), "rows": rows}


def report(d: dict) -> str:
    out = [f"== {d['family']} (ISA table: {d['isa_table']}; priced tensor: {', '.join(d['priced_tensor_tokens'])})",
           "buckets: " + ", ".join(f"{b} {n}" for b, n in d["counts"].items())]
    if d["disagreements"]:
        out.append("class_of disagreements (the lane classifier's UNIFORM regex mis-maps these): "
                   + ", ".join(f"{x['op']}->{x['class_of']}" for x in d["disagreements"]))
    order = ["GAP-TENSOR", "GAP-MULTIPLY", "UNDERPRICED", "GAP-WORD", "UNKNOWN", "PRICED-TENSOR", "PRICED-SIDE",
             "MOVER", "PRICED-LANE", "HARMLESS"]
    by = {}
    for r in d["rows"]:
        by.setdefault(r["bucket"], []).append(r)
    for b in order:
        rs = by.get(b, [])
        if not rs or b in ("PRICED-LANE", "HARMLESS", "MOVER", "PRICED-SIDE"):
            continue
        out.append(f"\n-- {b} ({len(rs)})")
        for r in sorted(rs, key=lambda r: (-r["count"], r["op"])):
            tag = f" x{r['count']:,}" if r["in_corpus"] else " (unused)"
            out.append(f"  {r['op']:28s}{tag:>14}  mul={r['mul']} route={r['route']} class_of={r['class_of']}")
            if r.get("escape"):
                out.append(f"       escape: {r['escape']}")
            else:
                out.append(f"       {r['note']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("family", choices=sorted(ISA_TABLE))
    p.add_argument("census", type=Path)
    p.add_argument("--unused", action="store_true", help="also the ISA-table opcodes the corpus never used")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    census = json.loads(a.census.read_text())
    d = diff(a.family, census, a.unused)
    if a.out:
        a.out.write_text(json.dumps(d, indent=1))
    print(report(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
