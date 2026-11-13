"""CIR-274 (REQBLOCK): rank R's registered rank form at 43 prompt ids and 2 outputs, LAYERS layers, dumped node by node:
form, Definition id, members, axes, output leaves and each argument's runs (space, index, lo, hi). Run from the repository root:

    .venv/bin/python scratch/cir274/p2_dump.py OUT LAYERS [RANK]

Writes OUT/dump_l<LAYERS>_r<RANK>.json (params, nodes, ret) and OUT/dump_l<LAYERS>_r<RANK>.pkl (the Program, pickled)."""
import importlib.util
import json
import pickle
import resource
import sys
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1])
layers = int(sys.argv[2])
rank = int(sys.argv[3]) if len(sys.argv) > 3 else 0
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity.circuits.refs import runs  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

LP, T = 43, 1
reg = cc.registered(rank, layers, "eager")
advice, ingress = cc.session([(0, LP, T + 1)])
u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
p = request_program(reg, u)
body = p.program.fn.body
params = [{"name": n, "leaves": t.leaves, "widths": sorted(t.width_set()),
           "fields": [[lab, ft.leaves, sorted(ft.width_set())] for lab, ft in getattr(t, "fields", ())][:4000]
           if n == "weights" or n.startswith("peers_") else None} for n, t in body.params]
nodes = []
for j, n in enumerate(body.nodes):
    ret = n.fn.signature()[1] if n.form != "primitive" else n.out
    nodes.append({"j": j, "form": n.form, "id": n.fn.id, "family": getattr(n.fn, "name", None), "n": n.n,
                  "axes": list(n.axes), "out": n.out_leaves, "name": n.name,
                  "outs": [[f.leaves, sorted(f.width_set())] for f in (getattr(ret, "fields", None) or (ret,))
                           if hasattr(f, "leaves")][:64],
                  "ports": [[k, t.leaves, sorted(t.width_set())] for k, t in n.fn.signature()[0]] if n.form != "primitive" else [],
                  "args": [list(runs(a.refs))[:64] for a in n.args]})
rec = {"rank": rank, "layers": layers, "digest": p.digest, "params": params, "nodes": nodes, "ret": list(runs(body.ret))[:256],
       "families": dict(Counter(n["family"] for n in nodes)), "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576}
(out / f"dump_l{layers}_r{rank}.json").write_text(json.dumps(rec, default=str) + "\n")
try:
    (out / f"dump_l{layers}_r{rank}.pkl").write_bytes(pickle.dumps(p.program, protocol=5))
except Exception as e:  # noqa: BLE001 - the JSON is the record
    print("REQBLOCK p2 no pickle:", type(e).__name__, str(e)[:300], flush=True)
print("REQBLOCK p2", json.dumps({k: rec[k] for k in ("rank", "layers", "digest", "families", "peak_gib")}), len(nodes), flush=True)
