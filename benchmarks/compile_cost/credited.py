from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

SCHEME = "pearl-c-h100-v1-h3"
LLAMA32_1B = {"repo": "unsloth/Llama-3.2-1B", "revision": "9535bd9b1d1dea6acafbdc4813b728796aeb28da"}


def _peak_gib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def sizes_of_class(cls: str, scheme: str) -> list:
    from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
    from verity_vllm.engine.served_seed import DEFAULT_SEED
    from verity_vllm.program.predict import window_circuit, window_cut
    from verity_vllm.program.predict.phi import Phi
    from verity_vllm.program.predict.steps import Registered
    from verity_vllm.program.predict.window_class import PearlTables, WindowClass
    hf = json.loads(pinned_hf_config(LLAMA32_1B["repo"], LLAMA32_1B["revision"]).read_text())
    reg = Registered.of(Phi(model={**LLAMA32_1B, "hf_config": hf, "dtype": "bfloat16"}, target={"card": "h200-sxm-141gb"}),
                        default_seed=DEFAULT_SEED)
    low = window_circuit.lower(reg, WindowClass(*map(int, cls.split(","))), tables=PearlTables(scheme))
    return {"census": [{k: r[k] for k in ("round", "call", "k", "n", "rows")} | {"a_rows": r["a"]["rows"]}
                       for r in window_cut.census(low)],
            "sizes": window_cut.credited_sizes(low), "templates": len(low.circuit.templates)}


def block(scheme: str, k: int, tm: int, tn: int, n: int, bits: bool = False, restrict: bool = True) -> dict:
    from verity.circuits import count as CT
    from verity.circuits import restrict as R
    from verity.circuits.codec import descriptor_chunks, descriptor_id
    from verity_vllm.program.predict import window_cut
    out, t = {"scheme": scheme, "k": k, "tm": tm, "tn": tn, "n": n, "phases": {}}, time.perf_counter()

    def phase(name: str) -> None:
        nonlocal t
        now = time.perf_counter()
        out["phases"][name] = {"wall_s": round(now - t, 2), "peak_gib": _peak_gib()}
        print(f"{name}: {out['phases'][name]}", flush=True)
        t = now

    program, part = window_cut.credited(scheme, k, tm, tn, n)
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
                      "binding": R._binding(b.bits, b.inputs[sum(q.n for _, q in params[:i])])}
                     for i, (p, ty) in enumerate(params)]
    out["outputs"] = [{"port": f"o{i}", "rows": ty.n, "width": ty.elem.w} for i, ty in enumerate(ret.fields)]
    out["word_gates"] = sum(hi - lo for lo, hi in b.words)
    out["bit_gates"] = b.bits.size
    got = CT.count(b.program)
    root = got[descriptor_id(b.program.root)]
    out["gates"], out["ands"] = root.gates, root.ands
    layout = Counter(descriptor_id(g.prim) for g in b.program.circuit.gates())
    out["by_primitive"] = sorted(({"primitive": p, "gates_each": got[p].gates, "ands_each": got[p].ands, "times": m,
                                   "gates": got[p].gates * m} for p, m in layout.items() if got[p].gates),
                                 key=lambda r: -r["gates"])
    out["sum_of_bodies"] = sum(r["gates"] for r in out["by_primitive"])
    phase("count")
    print("CREDITED-COUNT", k, tm, tn, n, "gates", out["gates"], "ands", out["ands"], flush=True)
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
    ap.add_argument("--size", help="K,TM,TN,N")
    ap.add_argument("--cls", help="a window class, e.g. 4,27,2: its cut calls' credited sizes on Llama 3.2 1B")
    ap.add_argument("--bits", action="store_true", help="also evaluate the block on bits once, on random inputs")
    ap.add_argument("--no-restrict", action="store_true",
                    help="count the Program's one unit from its descriptor, with no restrict: the block is the whole Program")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cls:
        got = sizes_of_class(a.cls, a.scheme)
        got["peak_gib"] = _peak_gib()
        (a.out / f"sizes-{a.cls.replace(',', '-')}.json").write_text(json.dumps(got, indent=1))
        print("CREDITED-SIZES", a.cls, got["sizes"], "templates", got["templates"], flush=True)
    if a.size:
        k, tm, tn, n = map(int, a.size.split(","))
        got = block(a.scheme, k, tm, tn, n, a.bits, not a.no_restrict)
        (a.out / f"block-{k}-{tm}-{tn}-{n}.json").write_text(json.dumps(got, indent=1))
        if a.no_restrict:
            return 0
        print("CREDITED-BLOCK", a.size, "gates", got["gates"], "ands", got["ands"], "sum_of_bodies", got["sum_of_bodies"],
              "inputs", [(r["port"], r["rows"], r["width"], r["binding"]) for r in got["inputs"]],
              "outputs", [(r["port"], r["rows"], r["width"]) for r in got["outputs"]], "peak_gib", got["peak_gib"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
