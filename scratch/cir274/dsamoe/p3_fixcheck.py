"""CIR-274 DSAMOE: the window at prompt lengths past 43, at 4 layers, rank 0, eager, the census's Build
(`benchmarks/compile_cost/run.py` `registered`). Run from the repository root of the tree to measure (a clone with
`fix-chain-ranges.diff` applied, or not):

    .venv/bin/python p3_fixcheck.py OUT OUTS LP...

For each prompt length: whether `circuit()` builds (the Circuit's own `check_wires`, `check_inputs` and `ordered`), the
description's digest and bytes, wire patterns after `coalesce` by reader family, and for the families that grow with the
prompt: BatchedGemm's ``x`` wires by writer template and step, SparseMlaSplit's patterns by reader template, the
SparseMlaSplit nodes' chain shapes (their T and places by key), and w2's ``g`` wires by step. Writes OUT/fix_<OUTS>.json
(rewritten after each length) and one DSAMOE line each."""
import bisect
import hashlib
import importlib.util
import json
import resource
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
OUTS = int(sys.argv[2])
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity.commitments.windows import interface as IF  # noqa: E402
from verity_vllm.pipeline import chain_rows as CR  # noqa: E402
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

reg = cc.registered(0, 4, "eager")
families = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}
path = out / f"fix_{OUTS}.json"
rec: dict = {"outs": OUTS, "lengths": {}}


def peak() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 3)


for LP in map(int, sys.argv[3:]):
    t0 = time.perf_counter()
    r: dict = {}
    try:
        advice, ingress = cc.session([(0, LP, OUTS)])
        u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
        p = request_program(reg, u)
        r["program"] = p.digest
        sm = [n for n in p.program.fn.body.nodes if NW._family(n) == "SparseMlaSplit"]
        shapes = Counter()
        for n in sm:
            S = n.fn.bindings
            keys = Counter(pl.key for pl in CR.plan(S))
            shapes[f"T={S['T']} places={sum(keys.values())} repeated_keys={sum(1 for v in keys.values() if v > 1)}"] += 1
        r["sparse_mla_nodes"] = len(sm)
        r["sparse_mla_shapes"] = dict(sorted(shapes.items()))
        w = NW.NodeWindow(p.program.fn, families, NW._GRAIN, NW.NAME, 1, True).lower()
        r["lower_s"] = round(time.perf_counter() - t0, 1)
        c = w.circuit()
        blob = IF.circuit_bytes(c.description())
        r.update(ok=True, digest=hashlib.sha512(blob).hexdigest(), bytes=len(blob), steps=w.steps, wires=len(c.wires),
                 wires_before_coalesce=len(w.b.wires), ranges=len(c.order or ()))

        def step_of(tid: str, at: int) -> int:
            spans = w.b.spans[tid]
            return spans[bisect.bisect_right([s[0] for s in spans], at) - 1][3]

        fam = [w.family_of(x.reader, x.at) for x in c.wires]
        r["by_family"] = dict(Counter(fam).most_common())
        bg = Counter()
        for x, f in zip(c.wires, fam):
            if f == "BatchedGemm" and x.port == "x":
                bg[f"{x.reader.split(',DOT')[0]} <- {x.writer} step {step_of(x.reader, x.at)}"] += 1
        r["batched_gemm_x"] = dict(bg.most_common())
        r["sparse_mla_by_reader"] = dict(Counter(x.reader.split(",DOT")[0].split(",INV")[0] for x, f in zip(c.wires, fam)
                                                 if f == "SparseMlaSplit").most_common())
        r["g_wires_by_step"] = dict(sorted(Counter(step_of(x.reader, x.at) for x in c.wires
                                                   if x.reader.startswith("MoeExpertFp8BlockGemmW") and x.port == "g").items()))
        r["key_wires"] = sum(1 for x in c.wires if x.reader.startswith("EmbeddingRead_v1") and x.port == "tok")
    except Exception as e:  # noqa: BLE001 - a refusal is a result
        r.update(ok=False, error=f"{type(e).__name__}: {e}"[:1500], trace=traceback.format_exc()[-2500:])
    r["wall_s"], r["peak_gib"] = round(time.perf_counter() - t0, 1), peak()
    rec["lengths"][LP] = r
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("DSAMOE fix", LP, OUTS, json.dumps({k: r.get(k) for k in ("ok", "digest", "bytes", "wires", "ranges", "steps",
          "g_wires_by_step", "key_wires", "wall_s", "peak_gib", "error")}, default=str)[:1500], flush=True)
