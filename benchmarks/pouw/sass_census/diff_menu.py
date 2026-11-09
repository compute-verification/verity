"""Diff a device's SASS opcode census against A7's menu: every opcode the model doesn't price, and whether it can head a
committed word.

A7 (`SheetSemantics`, `Security/Definitions/Accounting/Assumptions/Sheet.lean`) says no unit outside the menu heads a fresh
committed word (NoSum) and a side unit costs at least its per-word vector (SideUnits). The menu is five routes
(isa_escape.py): the tensor route (A3, priced by a form's operand-type peak), the CUDA lane route (q*, `loads.class_of`
gives the census class), the uniform datapath (on issue), table routes (by bytes) and priced side units (the pi32 floor).

This reads the device's opcode vocabulary -- the union of what `cuobjdump` emitted across the corpus (`census.json`) and
NVIDIA's Instruction Set Reference table for the architecture (`data/sass_isa_tables.json`) -- and buckets every opcode:

  PRICED-LANE     `loads.class_of` gives a census class, or the base is a lane/uniform MAC: priced on the q* lane route.
  PRICED-TENSOR   an MMA form A3 prices: its operand type has a peak in the device's `DS_tensor` row (A3 prices a form by
                  its type's data-sheet peak), or the form has a measured row in the device census.
  PRICED-SIDE     a reduction / atomic / texture side unit, priced at the pi32 floor (floors.csv); no multiply (NoSum).
  MOVER           no arithmetic: the output word equals an existing word (a copy / transpose / load / select) (NoSum).
  GAP-TENSOR      an MMA form whose operand type has no `DS_tensor` peak and no census row: A3 has no rate for its MACs
                  (f64 `DMMA` everywhere; e2m1 fp4 `UTCOMMA` on sm_100, whose sheet lists no e2m1).
  GAP-MULTIPLY    a product unit on no menu route: a fresh product the model prices at 0.
  UNDERPRICED     a native paired lane multiply (FFMA2/FMUL2) the census counts as one FMA, though it takes two issue
                  slots (measured on the B200, unpriced.cu): half its cost, so it undercounts work and heads no cheaper word.
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
#: the rows file whose `DS_tensor` row gives each operand type's data-sheet tensor peak
DEVICE_ROWS = {"sm_120": MM / "rows.csv", "sm_100": MM / "devices" / "sm_100" / "rows.csv",
               "sm_90": MM / "devices" / "sm_90" / "rows.csv", "sm_80": MM / "devices" / "sm_80" / "rows.csv"}
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
            "instruction; with no f64 tensor peak the sheet prices it at 0. Measured, it runs 64 f64 MACs a SM-clock on "
            "the B200 (1/128 of tcgen05 i8) and 128 on the H200 (DMMA.16x8x16, 1/32 of wgmma s8), and on both it shares "
            "the tensor pipe (t + s < 1 beside a saturated tcgen05 or wgmma; unpriced.cu, unpriced90.cu)",
    "BMMA": "an int8 product is 64 AND-popc bit-planes (D=64); native b1 MMA with no sheet peak is priced at 0, but a "
            "generous 8x-TOPS b1 is only 0.125x int8 (isa_escape DECOMPOSE); emulated on sm_120 (killed)",
    "FFMA2": "two FP32 FMAs an instruction at two issue slots and two FMA-pipe cycles (unpriced.cu on the B200: beside "
             "LOP3 the issue sums to 8/3 a SM-clock); the census counts it as one FFMA, so it counts half a program's "
             "FFMA2 slots: a conservative miscount that opens no route faster than FFMA",
    "FMUL2": "two FP32 multiplies an instruction at two issue slots (as FFMA2), counted as one FMUL: conservative",
    "CREDUX": "a coupled cross-lane reduce into a uniform register; no multiply, so as a FOLD it heads 1 word a warp "
              "instruction = 32 lane instructions a word (>> duo's 3); NoSum holds, unpriced",
    "FSWZADD": "FP32 swizzle-add, a fixed cross-lane add; no multiply (a sum of existing words), FOLD argument as redux",
    "REDS": "an exact add reduction to shared memory; no multiply (NoSum), priced-equivalent to RED at the pi32 floor "
            "but has no floors.csv kind of its own",
    "LDGMC": "a reducing / multicast global load (NVLS); an add of existing words with no multiply (NoSum); RT-SCOPE owns "
             "the NVLS switch-side reduction",
    "UTCOMMA": "tcgen05 block-scaled fp4 (e2m1, kind::mxf4/mxf4nvf4) on the B200 heads FP4 committed words; the sm_100 "
               "DS_tensor row lists no e2m1 and the census has no UTCOMMA row, so A3 has no rate for its MACs. Measured, "
               "kind::mxf4 and kind::mxf4nvf4 run 16390 MACs a SM-clock, 2x kind::i8 (the data sheet's e2m1 16384), while "
               "e2m1 through kind::f8f6f4 runs at e4m3's rate (unpriced_fp4.cu). fp4 is the B200's fastest tensor format, "
               "so this is the widest-rate gap; sm_120 lists e2m1 and prices its OMMA fp4",
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


#: an operand type the `DS_tensor` row can name (bf16 before f16 so the lookbehind keeps `f16` out of `bf16`)
TYPE_TOKEN = re.compile(r"(?<![a-z0-9])(e4m3|e5m2|e2m1|e3m2|e2m3|bf16|f16|tf32|f64|s8|u8|s4|b1)(?![a-z0-9])")


def ds_tensor_types(family: str) -> set[str]:
    """The operand types the device's `DS_tensor` row gives a data-sheet peak for: the types its "dense MACs a SM-clock"
    clause lists, up to the first `;` (so sm_90's "no b1 figure" after it doesn't count b1). e5m2 runs at e4m3's FP8 rate,
    so a listed e4m3 covers it."""
    f = DEVICE_ROWS.get(family)
    if not f or not f.exists():
        return set()
    for r in csv.DictReader(f.read_text().splitlines()):
        if r["row"] == "DS_tensor":
            desc = r["description"]
            i = desc.find("dense MACs a SM-clock")
            clause = desc[i:].split(";")[0] if i >= 0 else ""
            types = set(TYPE_TOKEN.findall(clause))
            if "e4m3" in types:
                types.add("e5m2")
            return types
    return set()


def mma_type(op: str) -> str | None:
    """An MMA form's operand type. tcgen05 (UTC*) opcodes name no type (it's in the instruction descriptor), so they go by
    base: OMMA/UTCOMMA are the block-scaled fp4 family (sm_120's `OMMA.SF...E2M1.E2M1.UE4M3`)."""
    s = op.upper()
    base = s.split(".")[0]
    if base == "DMMA":
        return "f64"
    if base in ("BMMA", "BGMMA", "UTCBMMA"):
        return "b1"
    if base in ("IMMA", "IGMMA", "UTCIMMA"):
        return "s8"
    if base in ("OMMA", "OGMMA", "UTCOMMA"):
        return "e2m1"
    if base in ("QMMA", "QGMMA", "UTCQMMA"):
        return "e5m2" if "E5M2" in s and "E4M3" not in s else "e4m3"
    if base in ("HMMA", "HGMMA", "UTCHMMA"):
        return "tf32" if "TF32" in s else "bf16" if "BF16" in s else "f16"
    return None


def isa_vocab(family: str) -> dict[str, dict]:
    p = HERE / "data" / "sass_isa_tables.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p} is a fetched fixture and isn't here: "
                                "`uv run research data fetch-fixtures benchmarks/pouw/sass_census/data`")
    t = json.loads(p.read_text())["tables"]
    return t.get(ISA_TABLE.get(family, ""), {})


#: lane / uniform multiply-accumulate bases: the model prices these on the q* lane route (or on issue), whatever suffix
#: the SASS carries, so a bare base whose `class_of` is None (no discriminating suffix) is still priced.
LANE_MAC_BASE = {"IMAD", "IMUL", "IMUL32I", "IDP", "IDP4A", "FFMA", "FHFMA", "FFMA32I", "FMUL", "FMUL32I", "DFMA",
                 "DMUL", "HFMA2", "HMUL2", "UIMAD", "UFFMA", "UFMUL"}


def bucket(op: str, sem: dict, cls: str | None, priced_toks: set[str], ds_types: set[str]) -> str:
    base = op.split(".")[0]
    if base in MMA_KIND_TOKEN:
        priced = mma_type(op) in ds_types or MMA_KIND_TOKEN[base] in priced_toks
        return "PRICED-TENSOR" if priced else "GAP-TENSOR"
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
    ds_types = ds_tensor_types(family)
    names = set(used) | ({o for o in vocab} if with_unused else set())
    rows = []
    for op in sorted(names):
        base = op.split(".")[0]
        sem = classify(base)
        cls = class_of(op)
        b = bucket(op, sem, cls, priced_toks, ds_types)
        rows.append({"op": op, "base": base, "group": vocab.get(base, {}).get("group") if base in vocab else
                     (vocab.get(op, {}).get("group")), "bucket": b, "mul": sem["mul"], "word": sem["word"],
                     "route": sem["route"], "type": mma_type(op) if base in MMA_KIND_TOKEN else None,
                     "class_of": cls, "count": used.get(op, {}).get("count", 0),
                     "in_corpus": op in used, "note": sem["note"],
                     "escape": ESCAPE.get(base) if b.startswith("GAP") else None})
    by = {}
    for r in rows:
        by.setdefault(r["bucket"], []).append(r)
    return {"family": family, "isa_table": ISA_TABLE.get(family), "priced_tensor_tokens": sorted(priced_toks),
            "ds_tensor_types": sorted(ds_types), "counts": {b: len(v) for b, v in sorted(by.items())},
            "disagreements": disagreements(rows), "rows": rows}


def report(d: dict) -> str:
    out = [f"== {d['family']} (ISA table: {d['isa_table']}; DS_tensor types: {', '.join(d['ds_tensor_types'])}; "
           f"census tensor rows: {', '.join(d['priced_tensor_tokens'])})",
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
