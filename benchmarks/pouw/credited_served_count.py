"""How many Pearl+ tiles, and of what size, a served epoch's circuit would hold if each of its linears' `Gemm_v2` units
were Pearl+'s tile reading x by wire (`pearl_plus_credited.served_product`): the circuit of record's served class (Llama
3.2 1B, class 4,27,2, four requests of 27 prompt tokens and 2 outputs, the RTX target, seed 7, as `served_door.py
build`), its linear columns by depth K, the tiles at one row and TN columns a tile (a K's columns past whole tiles
counted as one more), and each tile's ANDs and its units
before laying (`pearl_plus_credited.cut`). Nothing is proved.

  python benchmarks/pouw/credited_served_count.py [--tn 64] [--ands]     # one JSON line
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

SCHEME = "pearl-c-h100-v1-h3"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tn", type=int, default=64)
    ap.add_argument("--ands", action="store_true", help="count each tile shape's ANDs (minutes at K = 2048)")
    a = ap.parse_args()
    tree = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(tree / "tests/adapters/vllm"))
    from tests.program.predict import test_steps as S
    from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
    from verity.protocols.accounting.compute.pouw.prover.circuit.pearl_c_call import ands
    from verity_vllm.program.predict import window_circuit as WCirc
    from verity_vllm.program.predict import window_class as WC

    t = time.perf_counter()
    cfg = json.loads((tree / "tests/adapters/vllm/tests/program/predict/fixtures/llama32-1b.config.json").read_text())
    reg = S.Registered.of(S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=dict(S.RTXPRO6000)),
                          default_seed=7)
    circuit = WCirc.lower(reg, WC.WindowClass(4, 27, 2), None, requests=[WC.Request(0, j, 27, 2) for j in range(4)]).circuit
    built = round(time.perf_counter() - t, 1)
    units, gemm_units, cols, other = 0, Counter(), Counter(), Counter()
    for fn, n in circuit.templates:
        units += n
        if fn.id.startswith("Gemm_v2"):
            gemm_units[fn.id.split(",DOT")[0] + "}"] += n
            cols[fn.bindings["K"]] += n * fn.bindings["N"]
        else:
            other[fn.id.split("{")[0]] += n
    tiles = {k: -(-c // a.tn) for k, c in cols.items()}
    out = {"class": "4,27,2", "model": "llama32-1b", "built_s": built, "units": units, "gemm_units": dict(gemm_units),
           "other_units": dict(other), "linear_columns_by_k": dict(cols), "tn": a.tn,
           "columns_past_whole_tiles": {k: c % a.tn for k, c in cols.items()}, "tiles_by_k": tiles,
           "tiles": sum(tiles.values())}
    shapes = {}
    for k in sorted(tiles):
        node = PC.served_product(SCHEME, k, 1, a.tn, a.tn)
        cut = PC.cut(node)
        shapes[k] = {"units_before_laying": sum(n for _u, n in cut),
                     "by_template": {u.defn.id: n for u, n in cut}}
        if a.ands:
            s = time.perf_counter()
            shapes[k] |= {"ands": ands(node), "ands_by_unit": {u.id: ands(u) for u, _n in cut},
                          "ands_s": round(time.perf_counter() - s, 1)}
    out["tile"] = shapes
    if a.ands:
        out["ands"] = sum(tiles[k] * shapes[k]["ands"] for k in tiles)
    out["peak_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
