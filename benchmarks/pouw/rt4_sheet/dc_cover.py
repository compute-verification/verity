"""RT4-SHEET job 1 on the H200 and B200: every census class `facet_break/loads.class_of` can return, and a list of
SASS opcodes it returns None for, priced at each DC reading's π* (`lean_headline_dc.reading`, sheet and measured).
A class absent from the reading's vector, or present at cost 0 at π*, is free there: neither family's machine has a
catch-all (`CATCH_ALL` names sm_120 only), so an opcode that classes to None or to such a class costs a program
nothing.

    uv run python benchmarks/pouw/rt4_sheet/dc_cover.py --tree TREE [--json OUT]

Exit 0; the output is the list.
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc_price import _load, use_tree  # noqa: E402

#: opcodes a fold or checkpoint could use that no CLASS_RULES pattern names (generic-address memory, local memory,
#: shared and generic atomics' other spellings, predicate and bit-field ops, warp votes, control)
PROBES = ("LD.E", "LD.E.128", "ST.E", "ST.E.128", "LDL", "STL", "LDL.128", "STL.128", "ATOM.E.XOR.STRONG.GPU",
          "ATOM.E.ADD.STRONG.GPU", "ATOMS.XOR", "RED.E.XOR.STRONG.GPU", "REDG.E.XOR.STRONG.GPU", "REDUX.XOR",
          "PLOP3.LUT", "P2R", "R2P", "BMSK", "SGXT", "IABS", "VOTE.ANY", "MATCH.ANY", "MOV", "S2R", "CS2R", "LDC",
          "BAR.SYNC", "MEMBAR.GPU", "FENCE.VIEW.ASYNC.S", "BRA", "BSSY", "BSYNC", "DEPBAR", "CCTL.IVALL", "ELECT",
          "UBLKRED.G.S.XOR", "UBLKCP.S.G", "UTMALDG.2D", "SYNCS.ARRIVE.TRANS64")


def scan(LHD, L, fam, vector: str) -> dict:
    r = LHD.reading(fam, vector)
    lam, c = r.data.lam, r.data.c
    kinds = sorted({k for _, k in L.CLASS_RULES if k != "ALU_OR_IMAD"} | {"ALU", "IMAD", "UNI", "UNI64", "NOROW"})

    def price(k: str | None) -> Fraction | None:
        if k is None or k not in c:
            return None
        return sum((w * c[k].get(row, Fraction(0)) for row, w in lam.items()), Fraction(0))

    absent = [k for k in kinds if k not in c]
    zero = [k for k in kinds if k in c and price(k) == 0]
    probes = {op: {"class": L.class_of(op), "price": None if (p := price(L.class_of(op))) is None else float(p)}
              for op in PROBES}
    return {"family": fam.ns, "gpu": fam.gpu, "vector": vector, "pistar": r.lam, "absent": absent,
            "zero_at_pistar": zero, "probes": probes}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", type=Path, default=Path.cwd())
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    pouw = a.tree.resolve() / "benchmarks" / "pouw"
    use_tree(pouw)
    LHD = _load(pouw / "lean_headline_dc.py", "rt4_lean_headline_dc")
    L = _load(pouw / "facet_break" / "loads.py", "rt4_loads")
    res = [scan(LHD, L, fam, v) for fam in (LHD.SM90, LHD.SM100) for v in (fam.sheet, fam.measured)]
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, sort_keys=True))
    for f in res:
        print(f"{f['family']} ({f['gpu']}) on {f['vector']}: π* {f['pistar']}")
        print(f"  classes absent from the vector (free): {' '.join(f['absent']) or 'none'}")
        print(f"  classes at cost 0 at π*: {' '.join(f['zero_at_pistar']) or 'none'}")
        free = [f"{op} ({p['class']})" for op, p in f["probes"].items() if not p["price"]]
        print(f"  probe opcodes free: {', '.join(free) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
