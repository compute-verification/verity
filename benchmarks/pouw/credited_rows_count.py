"""ANDs of Pearl+'s served tile (`pearl_plus_credited.served_product`) at TM rows, each unit of its cut and in all, and per
row, for each shape ``K,TN,N`` given; every row real (TR = TM).

  python benchmarks/pouw/credited_rows_count.py --tm 1 16 --shape 256,4,512 --shape 2048,64,64     # one JSON line a shape
"""
from __future__ import annotations

import argparse
import json
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--shape", action="append", required=True, help="K,TN,N")
    ap.add_argument("--scheme", default="pearl-c-h100-v1-h3")
    a = ap.parse_args()
    from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
    from verity.protocols.accounting.compute.pouw.prover.circuit.pearl_c_call import ands
    for shape in a.shape:
        k, tn, n = map(int, shape.split(","))
        out: dict = {"K": k, "TN": tn, "N": n, "scheme": a.scheme, "by_tm": {}}
        for tm in a.tm:
            t0 = time.perf_counter()
            tile = PC.served_product(a.scheme, k, tm, tn, n)
            units = [(u.id, c, ands(u)) for u, c in PC.cut(tile)]
            total = sum(c * x for _id, c, x in units)
            out["by_tm"][tm] = {"ands": total, "ands_per_row": total / tm, "units": units,
                                "s": round(time.perf_counter() - t0, 1)}
        tms = sorted(out["by_tm"])
        if len(tms) > 1:
            lo, hi = out["by_tm"][tms[0]], out["by_tm"][tms[-1]]
            out["per_row_ratio"] = lo["ands_per_row"] / hi["ands_per_row"]
        print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
