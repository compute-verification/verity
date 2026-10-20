"""RT4-SHEET job 3: the recorded memory-route folds priced on sm_120's sheet, at π* = DS_issue (every warp instruction
one slot), from the census-cover fixture's timed-loop counts (cpw_routes' modes, RT3-120's run r20261019-192718-5d46).

A trip of the timed loop is one set, 8 units (8 IMMA.16832). Every op but IMMA and NOP is the fold's, as `cpw_price.py`
counts it. Modes 9 and 10 issue some instructions in a body thread 0 runs once a block of W warps; the fixture's counts
don't split the body out, so this bounds the fold from below with the split most favourable to a cheater: only what
every warp must issue stays outside (its lane fold, its own global and shared stores and loads, the block barrier, and
the outer half of each `fence.proxy.async`), and everything else is taken as body, amortized to nothing.

    python benchmarks/pouw/rt4_sheet/reprice.py [FIXTURE]

Prints each sm_120 mode's pure lane fold, its every-warp lower bound and its static count a unit, against FOLD-OPT's
N_w φ* = 16 at DS_issue.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "census_cover_sass.json"
FLOOR = 16                      # N_w φ*: 8 IMAD + 8 ALU a unit, `Sm120.flatFold`
UNITS = 8                       # IMMA.16832 a set
LANE = ("IMAD", "LEA.HI", "LOP3.LUT", "IADD3", "SHF", "LEA")
MEM = ("STG.", "STS", "LDG.", "LDS", "ST.", "LD.")
EVERY_WARP = ("BAR.SYNC",)      # every warp of a block reaches the barrier


def lane(op: str) -> bool:
    return op.split(".")[0] in {x.split(".")[0] for x in LANE} and not op.startswith("PLOP3")


def price(counts: dict[str, int]) -> dict[str, float]:
    fold = {op: n for op, n in counts.items() if not op.startswith("IMMA") and op != "NOP"}
    static = sum(fold.values())
    pure = sum(n for op, n in fold.items() if lane(op))
    mem = sum(n for op, n in fold.items() if op.startswith(MEM))
    bar = sum(n for op, n in fold.items() if op.startswith(EVERY_WARP))
    # each fence.proxy.async is a MEMBAR.ALL.GPU and a FENCE.VIEW.ASYNC.S; the outer one of each pair every warp issues
    fence = (fold.get("MEMBAR.ALL.GPU", 0) + fold.get("FENCE.VIEW.ASYNC.S", 0)) / 2
    floor = pure + mem + bar + fence
    return {"static": static / UNITS, "pure_lane": pure / UNITS, "every_warp_lower": floor / UNITS,
            "memory": mem / UNITS, "barrier_fence": (bar + fence) / UNITS}


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else FIXTURE
    d = json.loads(path.read_text())
    rows = []
    for p in d["programs"]:
        if p["device"] != "sm_120":
            continue
        r = price(p["counts"])
        r["program"], r["run"] = p["program"], p.get("run")
        r["below_floor"] = r["every_warp_lower"] < FLOOR
        rows.append(r)
        print(f"{p['program']:22s} pure {r['pure_lane']:6.3f}  every-warp ≥ {r['every_warp_lower']:6.3f}  static "
              f"{r['static']:6.3f}  (mem {r['memory']:.3f}, bar+fence {r['barrier_fence']:.3f})  "
              f"{'BELOW' if r['below_floor'] else 'above'} {FLOOR}")
    under = [r["program"] for r in rows if r["below_floor"]]
    print(f"floor N_w φ* = {FLOOR} at DS_issue; modes whose every-warp lower bound is below it: {under or 'none'}")
    return 1 if under else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
