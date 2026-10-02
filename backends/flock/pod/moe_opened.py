"""A MoE row's q-word counts with its expert weight gathers opened rather than proved, or with its expert units left out.

    moe_opened.py --counts counts.json --extra gemm-counts.json ... --out counts-opened.json
    moe_opened.py --counts counts.json --drop --out counts-strict.json

`MoeExpertRow_v1`, an E-way gather of each weight element at the routed expert, is meant to be checked by opening the committed
expert weight bank, as the embedding's row is (`boolean_export.OPENED_IN`), not as a circuit. The sweep (`class_statement`)
lowers it as a circuit, and an expert unit then needs more than a 2^26-bit block (`circuit.K_MAX`), so no expert unit can be
staged. Opened, a unit is its GEMM coordinate: `MoeExpertCoordinate_v2{E,K,DOT}` is `GemmCoordinate_v2{K,DOT}` (its word gates
less the K gathers), and `MoeExpertCoordinateW_v2{E,K,DOT}` is `DotBf16_v2{K,DOT}`, the f32 multiply by the routed weight and
the bf16 rounding: `GemmCoordinate_v2{K,DOT}` and one more word gate, which this maps it onto (the multiply is not priced).
Each expert Call Definition's units move to that coordinate's shape, taken from the row's counts or an `--extra` counts.json of
a Definition that holds it (`Gemm_v2{K=768,...}` for a down projection the row has no GEMM of); the shape counts are checked as
word gates. Every other shape, unit and Call is the row's. `--drop` writes the counts without the expert units (what
`vllm_overhead.py rollup` reads as the row the prover can stage), for the exclusion's number.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

EXPERT = re.compile(r"^MoeExpertGemm(W?)_v2\{E=(\d+),K=(\d+),N=(\d+),DOT=\{\"fn\":\"(\w+)\"\}\}$")


def opened(c: dict, extra: list[dict], drop: bool = False) -> dict:
    shapes = {sh: dict(s) for sh, s in c["shapes"].items()}
    per = {fid: dict(p) for fid, p in c["per_definition"].items()}
    by_def = {s["definition"]: (sh, s) for x in [c] + extra for sh, s in x["shapes"].items()}
    moved = {}
    for fid, p in c["per_definition"].items():
        m = EXPERT.match(fid)
        if not m:
            continue
        weighted, E, K, N, dot = m[1] == "W", int(m[2]), int(m[3]), int(m[4]), m[5]
        (src, n), = p.items()
        assert n == N and shapes[src]["definition"] == f"MoeExpertCoordinate{'W' if weighted else ''}_v2{{E={E},K={K},DOT={dot}}}", fid
        units = shapes.pop(src)["row_units"]
        assert units == n * c["calls"][fid], fid
        if drop:
            del per[fid]
            moved[fid] = {"from": src, "row_units": units, "to": None}
            continue
        dst, s = by_def[f"GemmCoordinate_v2{{K={K},DOT={dot}}}"]
        assert c["shapes"][src]["word_gates"] - K == s["word_gates"] + (1 if weighted else 0), (fid, s)
        t = shapes.setdefault(dst, {"definition": s["definition"], "units": 0, "row_units": 0, "word_gates": s["word_gates"]})
        t["row_units"] = (t["row_units"] or 0) + units
        t["units"] += n
        per[fid] = {dst: n}
        moved[fid] = {"from": src, "row_units": units, "to": dst, "word_gates": [c["shapes"][src]["word_gates"], s["word_gates"]]}
    assert moved, "no expert Call Definition in the counts"
    return {"calls": c["calls"], "per_definition": per, "shapes": shapes,
            "moe_opened": {"mode": "dropped" if drop else "opened", "experts": moved}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--counts", required=True)
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--drop", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    out = opened(json.loads(Path(a.counts).read_text()), [json.loads(Path(p).read_text()) for p in a.extra], a.drop)
    tmp = a.out.with_name(f".{a.out.name}.tmp")
    tmp.write_text(json.dumps(out, sort_keys=True))
    os.replace(tmp, a.out)
    g = sum(s["word_gates"] * (s["row_units"] or 0) for s in out["shapes"].values())
    print(f"{out['moe_opened']['mode']}: {len(out['moe_opened']['experts'])} expert Call Definitions, {len(out['shapes'])} shapes, "
          f"{g} gates -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
