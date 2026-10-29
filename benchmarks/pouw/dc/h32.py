"""H32 on the H200 (sm_90a): the timed H32 probes (probes/sm90_h32*.cu) in MH100's unit, as a `Prices100` table.

  python3 h32.py RUN_OUT... [--json OUT] [--lean OUT]

RUN_OUT is `emp.sh time`'s output, each binary's rows.jsonl under RUN_OUT/<binary>/. A probe's note names its class and
the 32-bit registers one warp instruction of each counted pattern writes a lane:

  c=CLASS [s=SHAPE] w=PATTERN:REGS,... [p=PATTERN:PREDS,...] [u=PATTERN:WORDS,...]

CLASS is ffma, int32, fp64, other or tc (`Prices100`'s fields), wgmma (`mac`, SHAPE its `Shape100` shape) or mix (no
class: writers co-issued across classes). A probe writes Σ rate × 32 × REGS registers a SM a clock over its counted
patterns, rate being the pattern's warp instructions a SM a clock in rows.jsonl (every loop); its price is 4,096 over
that, the unit being one dense int8 MAC at 4,096 a SM a clock. Predicates and uniform registers count by their bits: a
pattern that writes PREDS predicates a lane adds rate × PREDS registers (32 predicate bits a register), and one that
writes WORDS uniform registers a warp adds rate × WORDS. A pattern may appear in w, p and u at once. A probe with an OP_UNTIL loop (`until`) is read over
the stop's cycles: its fixed loop's rates are scaled by 1 / (1 - stop_gap), the reading in the adversary's favour. The
stop is the first fixed warp's, so when the fixed loop is a wgmma whose warpgroups finish apart, that scaling credits
the window with work done after it. A mix whose fixed loop is a wgmma names its shape (s=SHAPE), and the fixed loop is
then read at no more than the shape's data-sheet rate, 4,096 / (k × elems × nominalMac) registers a SM a clock.

The table is Round 7's rule: each field the least price of its class's probes, rounded down; `mac s` is the least
price per register of shape s's probes (any N) over k × elems, rounded down. A mix's least price is the co-issue
minimum, beside the table. H32 is `Floor 32`; the table meets `FloorTc a c` for a its least entry outside `tc` and c
its `tc`. --lean writes the table as a Lean `Prices100` literal, and only when every field has a probe.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

UNIT = 4096
#: `Shape100`'s constructors in its order, by the wa_ probes' s= tag: (constructor, k, elems, nominalMac)
SHAPES = {"e4m3": ("e4m3F32", 32, 1, 1), "e5m2": ("e5m2F32", 32, 1, 1), "e4m3f16": ("e4m3F16", 32, 2, 1),
          "e5m2f16": ("e5m2F16", 32, 2, 1), "s8": ("s8", 32, 1, 1), "f16": ("f16F32", 16, 1, 2),
          "f16f16": ("f16F16", 16, 2, 2), "bf16": ("bf16F32", 16, 1, 2), "tf32": ("tf32", 8, 1, 4)}
FIELDS = ("tc", "ffma", "int32", "fp64", "other")


#: registers a SM a clock one warp instruction a SM a clock adds for each unit of a w, p or u count
PER_WARP = {"w": 32, "p": 1, "u": 1}


def parse_note(note: str) -> dict:
    tags = dict(t.split("=", 1) for t in note.split() if "=" in t)
    out = {"class": tags.get("c"), "shape": tags.get("s")}
    for kind in PER_WARP:
        out[kind] = {}
        for item in filter(None, tags.get(kind, "").split(",")):
            pat, n = item.rsplit(":", 1)
            out[kind][pat] = int(n)
    return out


def price(row: dict) -> dict:
    """A rows.jsonl row's registers a SM a clock and its price per register."""
    n = parse_note(row["note"] or "")
    until, gap = (row.get("until") or 0) - 1, row.get("stop_gap")
    if gap is not None and until < 0:
        raise ValueError(f"{row['probe']}: a stop_gap but no OP_UNTIL loop")
    cap = None
    if n["class"] == "mix" and n["shape"]:
        if n["shape"] not in SHAPES:
            raise ValueError(f"{row['probe']}: s={n['shape']} is no Shape100 shape")
        _, k, elems, nominal = SHAPES[n["shape"]]
        cap = UNIT / (k * elems * nominal)
    fixed = read_as_is = 0.0
    for key, rate in row["rates"].items():
        loop, pat = key.split(":", 1)
        per = sum(PER_WARP[kind] * n[kind].get(pat, 0) for kind in PER_WARP)
        if gap is None or int(loop) == until:
            read_as_is += rate * per
        else:
            fixed += rate / (1 - gap) * per
    counted = {p for kind in PER_WARP for p in n[kind]}
    missing = [p for p in counted if not any(k.split(":", 1)[1] == p for k in row["rates"])]
    if missing:
        raise ValueError(f"{row['probe']}: note counts {missing}, which its rates don't hold")
    regs = read_as_is + (fixed if cap is None else min(fixed, cap))
    if regs <= 0:
        raise ValueError(f"{row['probe']}: its note counts no register, predicate or uniform write")
    return {**n, "probe": row["probe"], "binary": row["binary"], "run": row["run"], "tags": row["tags"],
            "regs_sm_clk": regs, "price": UNIT / regs}


def read(run_out: Path) -> list[dict]:
    rows = []
    for f in sorted(run_out.glob("*/rows.jsonl")):
        rows += [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
    return [price(r) for r in rows if r.get("note") and "c=" in r["note"]]


def floor(x: float) -> int:
    return math.floor(x + 1e-9)


def table(priced: list[dict]) -> dict:
    """The least price of each field's probes (with the probe), each Shape100 shape's, and the co-issue minimum."""
    least: dict[str, dict] = {}
    for p in priced:
        key = None
        if p["class"] == "wgmma":
            if p["shape"] not in SHAPES:
                raise ValueError(f"{p['probe']}: s={p['shape']} is no Shape100 shape")
            key = "mac." + p["shape"]
        elif p["class"] in FIELDS or p["class"] == "mix":
            key = p["class"]
        else:
            raise ValueError(f"{p['probe']}: c={p['class']} is no class")
        if key not in least or p["price"] < least[key]["price"]:
            least[key] = p
    mac = {}
    for s, (ctor, k, elems, nominal) in SHAPES.items():
        if "mac." + s in least:
            mac[ctor] = {"mac": floor(least["mac." + s]["price"] / (k * elems)), "nominal": nominal,
                         "reg_price": least["mac." + s]["price"], "probe": least["mac." + s]["probe"]}
    fields = {f: {"value": floor(least[f]["price"]), "price": least[f]["price"], "probe": least[f]["probe"]}
              for f in FIELDS if f in least}
    missing = [SHAPES[s][0] for s in SHAPES if SHAPES[s][0] not in mac] + [f for f in FIELDS if f not in fields]
    reg = {c: m["mac"] * k * e for s, (c, k, e, _) in SHAPES.items() if (m := mac.get(c))}
    out = {"mac": mac, "fields": fields, "missing": missing,
           "co_issue": {"price": least["mix"]["price"], "probe": least["mix"]["probe"],
                        "regs_sm_clk": least["mix"]["regs_sm_clk"]} if "mix" in least else None}
    if not missing:
        a = min([*reg.values(), *(fields[f]["value"] for f in FIELDS if f != "tc")])
        out["h32"] = a >= 32 and fields["tc"]["value"] >= 32
        out["floor_tc"] = {"a": a, "c": fields["tc"]["value"]}
    return out


def lean(t: dict) -> str:
    """The table as a Lean `Prices100` literal (namespace Pouw.Dimension.H100)."""
    if t["missing"]:
        raise ValueError(f"no probe for {t['missing']}")
    mac = t["mac"]
    if all(m["mac"] == m["nominal"] for m in mac.values()):
        body = ["  mac := nominalMac"]
    else:
        body = ["  mac"] + [f"    | .{c} => {mac[c]['mac']}" for c, *_ in SHAPES.values()]
    body += [f"  {f} := {t['fields'][f]['value']}" for f in FIELDS]
    return "def h200Measured : Prices100 where\n" + "\n".join(body) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_out", type=Path, nargs="+")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--lean", type=Path)
    a = ap.parse_args(argv)
    priced = [p for d in a.run_out for p in read(d)]
    t = table(priced)
    for p in sorted(priced, key=lambda p: (p["class"], p["price"])):
        tags = "".join(f" [{x}]" for x in p["tags"])
        print(f"{p['class']:6} {p['probe']:28} {p['price']:9.3f} a register  {p['regs_sm_clk']:8.2f} a SM-clock"
              f"  {p['run']}{tags}")
    for c, m in t["mac"].items():
        print(f"mac {c:8} {m['mac']} (nominal {m['nominal']}; {m['reg_price']:.3f} a register, {m['probe']})")
    for f, v in t["fields"].items():
        print(f"{f:6} {v['value']} ({v['price']:.3f}, {v['probe']})")
    if t["co_issue"]:
        c = t["co_issue"]
        print(f"co-issue minimum {c['price']:.3f} a register ({c['regs_sm_clk']:.2f} a SM-clock, {c['probe']})")
    if t["missing"]:
        print(f"no probe for {', '.join(t['missing'])}: H32 and FloorTc are undecided")
    else:
        print(f"H32: {t['h32']}; FloorTc {t['floor_tc']['a']} {t['floor_tc']['c']}")
    if a.json:
        a.json.write_text(json.dumps({"probes": priced, "table": t}, indent=1) + "\n")
    if a.lean and not t["missing"]:
        a.lean.write_text(lean(t))
    return 0 if not t["missing"] else 1


if __name__ == "__main__":
    sys.exit(main())
