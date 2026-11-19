from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

SCHEME = "pearl-c-h100-v1-h3"
LAYOUT_MAX = 10**9


def _peak_gib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def block(scheme: str, k: int, tm: int, tn: int, n: int, real: int | None = None, bits: bool = False,
          restrict: bool = True, depth: int | None = None) -> dict:
    from verity.circuits import count as CT
    from verity.circuits import restrict as R
    from verity.circuits.codec import descriptor_chunks, descriptor_id
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
    out, t = {"scheme": scheme, "k": k, "tm": tm, "tn": tn, "n": n, "real": real, "depth": depth, "phases": {}}, time.perf_counter()

    def phase(name: str) -> None:
        nonlocal t
        now = time.perf_counter()
        out["phases"][name] = {"wall_s": round(now - t, 2), "peak_gib": _peak_gib()}
        print(f"{name}: {out['phases'][name]}", flush=True)
        t = now

    program, part = PC.credited(scheme, k, tm, tn, n, real, depth)
    phase("program")
    c = b"".join(descriptor_chunks(program))
    out["descriptor_bytes"] = len(c)
    phase("descriptor")
    if not restrict:
        root = CT.count(c)[descriptor_id(program.root)]
        out["gates"], out["ands"] = root.gates, root.ands
        phase("count")
        print("CREDITED-COUNT", k, tm, tn, n, "gates", out["gates"], "ands", out["ands"], flush=True)
        out["peak_gib"] = _peak_gib()
        return out
    b = R.restrict(c, part, [(0, 1)])
    phase("restrict")
    params, ret = b.program.fn.signature()
    out["inputs"] = [{"port": p, "rows": ty.n, "width": ty.elem.w,
                      "binding": R._Gates(b.D).binding(b.inputs[sum(q.n for _, q in params[:i])])}
                     for i, (p, ty) in enumerate(params)]
    out["outputs"] = [{"port": f"o{i}", "rows": ty.n, "width": ty.elem.w} for i, ty in enumerate(ret.fields)]
    out["word_gates"] = sum(hi - lo for lo, hi in b.words)
    got = CT.count(b.program)
    root = got[descriptor_id(b.program.root)]
    out["gates"], out["ands"] = root.gates, root.ands
    phase("count")
    print("CREDITED-COUNT", k, tm, tn, n, "gates", out["gates"], "ands", out["ands"], flush=True)
    if b.program.gates > LAYOUT_MAX:
        out["peak_gib"] = _peak_gib()
        return out
    layout = Counter(descriptor_id(g.prim) for g in b.program.circuit.gates())
    out["by_primitive"] = sorted(({"primitive": p, "gates_each": got[p].gates, "ands_each": got[p].ands, "times": m,
                                   "gates": got[p].gates * m} for p, m in layout.items() if got[p].gates),
                                 key=lambda r: -r["gates"])
    out["sum_of_bodies"] = sum(r["gates"] for r in out["by_primitive"])
    phase("layout")
    if bits:
        import random

        import numpy as np
        from verity.circuits.evaluation import evaluate_bits
        from verity.circuits.ops.boolean.versions import block_circuit, boolean_table
        fn = block_circuit(b.program.fn, boolean_table(b.program))
        phase("lift")
        rng = random.Random(0)
        got = evaluate_bits(fn, *(np.asarray([[rng.getrandbits(1) for _ in range(t.leaves)]], dtype=np.uint8)
                                  for _p, t in params))
        out["bits_out"] = got.reshape(-1).tolist()[-1:]
        phase("bits")
    out["peak_gib"] = _peak_gib()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default=SCHEME)
    ap.add_argument("--size", help="K,TM,TN,N or K,TM,TN,N,TR")
    ap.add_argument("--bits", action="store_true", help="also evaluate the block on bits once, on random inputs")
    ap.add_argument("--depth", type=int, help="the tile with its leaf opening in a transcript tree of this depth")
    ap.add_argument("--no-restrict", action="store_true",
                    help="count the Program's one unit from its descriptor, with no restrict: the block is the whole Program")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.size:
        k, tm, tn, n, *real = map(int, a.size.split(","))
        got = block(a.scheme, k, tm, tn, n, real[0] if real else None, a.bits, not a.no_restrict, a.depth)
        (a.out / f"block-{k}-{tm}-{tn}-{n}.json").write_text(json.dumps(got, indent=1))
        if "sum_of_bodies" not in got:
            return 0
        print("CREDITED-BLOCK", a.size, "gates", got["gates"], "ands", got["ands"], "sum_of_bodies", got["sum_of_bodies"],
              "inputs", [(r["port"], r["rows"], r["width"], r["binding"]) for r in got["inputs"]],
              "outputs", [(r["port"], r["rows"], r["width"]) for r in got["outputs"]], "peak_gib", got["peak_gib"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
