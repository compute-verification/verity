"""ANDs and two-input gates of PoUW's H1 tile unit at K by tile shape R × C, from the IR's Boolean versions (census's
work count, which counts without lowering; the unit itself, since its coin parameter keeps it out of a one-Call Program),
with its parameters' bits: the rows a proof block opens. R × C is a multiple of the 32 words of a lane.

    python h1_tile_ands.py [--k 2048] [--tiles 1x1,1x2,2x2,4x4,4x8,8x8,16x16,32x32] [--out FILE.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from verity.core.protocols.pouw.circuit import h1window as HW
from verity.ml.circuits.census import _Work
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.definitions.boolean import boolean_version


def measure(k: int, r: int, c: int) -> dict:
    unit = HW.tile_unit(k, (r, c))
    W = _Work()
    got = W.of(unit)
    if isinstance(got, str):
        v = boolean_version(unit)
        got = W.of(bind(v) if isinstance(v, CompositeDefinition) else v)
    if isinstance(got, str):
        raise ValueError(f"{unit.id[:80]} reaches {got}, which has no Boolean work")
    params, ret = unit.signature()
    return {"k": k, "tile": [r, c], "macs": r * c * k, "ands": sum(a for _g, a in got.values()),
            "gates2": sum(g for g, _a in got.values()), "work": {f: {"gates": g, "and": a} for f, (g, a) in got.items()},
            "params": {name: sum(t.leaf_widths()) for name, t in params}, "returned_bits": sum(ret.leaf_widths())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--tiles", default="1x1,1x2,2x2,4x4,4x8,8x8,16x16,32x32")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = []
    for t in a.tiles.split(","):
        r, c = map(int, t.split("x"))
        try:
            m = measure(a.k, r, c)
        except (IndexError, ValueError) as e:
            print(f"K={a.k} {r}x{c}: refused by h1tile ({type(e).__name__}: {e})", flush=True)
            continue
        rows.append(m)
        print(f"K={a.k} {r}x{c}: MACs {m['macs']:,} ANDs {m['ands']:,} ({m['ands'] / m['macs']:.1f}/MAC) gates2 {m['gates2']:,} "
              f"params {m['params']} returned {m['returned_bits']}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
