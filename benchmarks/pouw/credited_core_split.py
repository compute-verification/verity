"""ANDs and port bits of each top-level node of Pearl+'s tile core on bits (`pearl_plus_tile.PearlPlusTileCoreV3`) at the
served tile's statics, the pieces a finer laying of `pearl_plus_credited.PearlPlusTileCoreWired_v3` can group into blocks.

  python benchmarks/pouw/credited_core_split.py --shape 256,4,512      # one JSON line a node, then the total
"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="256,4,512", help="K,TN,N")
    ap.add_argument("--tm", type=int, default=1)
    ap.add_argument("--scheme", default="pearl-c-h100-v1-h3")
    a = ap.parse_args()
    from verity.protocols.accounting.compute.pouw.prover import schemes as PS
    from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
    from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_tile as PT
    from verity.protocols.accounting.compute.pouw.prover.circuit.pearl_c_call import ands
    k, tn, n = map(int, a.shape.split(","))
    core = PT.core(PS.get(a.scheme), k, a.tm, tn, n)
    memo: dict = {}
    for name, d in (("core", core), ("wired_v3", PC.bind(PC.PearlPlusTileCoreWiredV3, **core.bindings))):
        rows = []
        for i, nd in enumerate(d.body.nodes):
            fn = nd.fn
            params, ret = fn.signature()
            x = ands(fn, memo)
            rows.append({"node": i, "fn": fn.id[:90], "form": nd.form, "n": nd.n, "ands_one": x, "ands": nd.n * x,
                         "in_bits": [sum(t.leaf_widths()) for _p, t in params],
                         "out_bits": sum(ret.leaf_widths())})
        big = [r for r in rows if r["ands"]]
        print(json.dumps({"of": name, "id": d.id, "ands": ands(d, memo), "nodes": len(rows), "with_ands": big}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
