"""Rank 0's class-G request Program at 43 prompt ids and 2 outputs, lowered into the node window with the frame's, dense,
sparse and multimem families; its census and the all-reduce's units, templates and free rows. Run from the repo root:

    python mm_census.py OUT LAYERS
"""
import importlib.util
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

out, layers = Path(sys.argv[1]), int(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity.commitments.windows import interface as IF  # noqa: E402
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_multimem as NM  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

FAMILIES = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES, **NM.FAMILIES}
LP, T = 43, 1
path = out / f"mm_l{layers}.json"
rec: dict = {"LP": LP, "outputs": T + 1, "layers": layers, "rank": 0, "sets": "frame+dense+sparse+multimem"}


def peak() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 3)


t0 = time.perf_counter()
reg = cc.registered(0, layers, "eager")
advice, ingress = cc.session([(0, LP, T + 1)])
u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
p = request_program(reg, u)
rec["program"] = {"digest": p.digest, "nodes": len(p.program.fn.body.nodes), "wall_s": round(time.perf_counter() - t0, 1)}
print("MM program", rec["program"], flush=True)
t0 = time.perf_counter()
circuit, w = NW.node_window(p.program.fn, families=FAMILIES)
rec["window_wall_s"] = round(time.perf_counter() - t0, 1)
cen = w.census(circuit)
mm_tids = {tid for tid, allocs in w.allocs.items() if any(f == NM.FAMILY for _a, _n, f in allocs)}
mm_units: Counter[str] = Counter()
gathers_at_sites: Counter[str] = Counter()
for tid, allocs in w.allocs.items():
    for at, n, fam in allocs:
        if fam == NM.FAMILY:
            mm_units[tid] += n
        elif fam == NW.GATHER and w.node_of.get((tid, at)) is not None and NW._family(
                w.fn.body.nodes[w.node_of[(tid, at)]]) == NM.FAMILY:
            gathers_at_sites[tid] += n
free_mm: Counter[str] = Counter()
for f in w.free:
    if f.tid in mm_tids:
        free_mm[f"{f.kind} {f.family if f.kind != 'peer' else f.name} rows {f.rows}"] += f.count
sites = [j for j, n in enumerate(w.fn.body.nodes) if NW._family(n) == NM.FAMILY]
rec["window"] = {k: cen[k] for k in ("digest", "units", "templates", "wire_patterns", "free", "unlowered", "peers")}
rec["window"] |= {"units_total": sum(circuit.units.values()), "templates_total": len(circuit.units),
                  "multimem_sites": len(sites), "multimem_units_by_template": dict(mm_units),
                  "gathers_at_multimem_sites_by_template": dict(gathers_at_sites),
                  "free_on_multimem_units": dict(free_mm),
                  "multimem_cuts": {t: circuit.cuts.get(t) for t in mm_tids}}
for tid, fn in w.b.fns.items():
    IF.check_proof_block(fn, circuit.cuts.get(tid))
rec["checks"] = "settle (check_proof_block per template, provable) + Circuit check_wires/check_inputs passed; " \
                "check_proof_block re-run per template with its cuts: passed"
rec["peak_gib"] = peak()
path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
print("MM window", json.dumps({k: v for k, v in rec["window"].items() if k != "peers"}, default=str), flush=True)
print("MM wall_s", rec["window_wall_s"], "peak_gib", rec["peak_gib"], flush=True)
