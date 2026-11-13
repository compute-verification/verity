"""REQBLOCK probe: rank 0's class-G request Program at 43 prompt ids, 2 outputs, LAYERS layers. For each AllReduceMultimem
node: its statics, its parts argument's runs (producer node family/form/n/out_leaves, or input name), its readers (family,
form, n, arg, axes, leaf runs), and the TP window's grain for its own partial; then the frame+dense+sparse window's pieces of
each producer node. Run from the repository root: .venv/bin/python probe.py OUT LAYERS"""
import importlib.util
import json
import sys
import time
from pathlib import Path

out, layers = Path(sys.argv[1]), int(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity.circuits.codec import descriptor_id  # noqa: E402
from verity.circuits.refs import runs  # noqa: E402
from verity.definitions.registry import tp_allreduce as TA  # noqa: E402
from verity_vllm.pipeline import serving_templates as ST  # noqa: E402
from verity_vllm.pipeline.tp import grain as G  # noqa: E402
from verity_vllm.program.predict import node_window as NW, node_window_dense as ND, node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

t0 = time.perf_counter()
reg = cc.registered(0, layers, "eager")
advice, ingress = cc.session([(0, 43, 2)])
u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
fn = request_program(reg, u).program.fn
nodes, params = fn.body.nodes, [p for p, _ in fn.signature()[0]]
fam = NW._family


def where(space, idx):
    if space == "n":
        n = nodes[idx]
        return {"node": idx, "family": fam(n), "id": n.fn.id[:160], "form": n.form, "n": getattr(n, "n", 1),
                "out_leaves": n.out_leaves}
    return {"param": params[idx]}


rec = {"layers": layers, "nodes": len(nodes), "program_s": round(time.perf_counter() - t0, 1), "multimem": []}
readers: dict[int, list] = {}
for j, n in enumerate(nodes):
    for k, a in enumerate(n.args):
        for space, idx, lo, hi in runs(a.refs):
            if space == "n" and fam(nodes[idx]) == "AllReduceMultimem":
                readers.setdefault(idx, []).append((j, k, lo, hi))
for j, n in enumerate(nodes):
    if fam(n) != "AllReduceMultimem":
        continue
    S = n.fn.bindings
    trees = TA.multimem_trees(S["TREES"], int(S["WORLD"]))
    parts = [{**where(s, i), "lo": lo, "hi": hi} for s, i, lo, hi in runs(n.args[0].refs)]
    own = [p for p in parts if "node" in p]
    grain = []
    for p in own:
        pn = nodes[p["node"]]
        m = getattr(pn, "n", 1) if pn.form == "batch" else 1
        L = pn.out_leaves // m
        lo, hi = p["lo"], p["hi"]
        grain.append({"member_leaves": L, "first": [lo // L, lo % L], "last": [(hi - 1) // L, (hi - 1) % L + 1],
                      "producer": None if (g := G._producer(pn.fn, lo % L, (hi - 1) % L + 1)) is None else
                      [g[0], descriptor_id(g[1])[:120], g[2], g[3]],
                      "coordinate": None if (c := ST.coordinate(descriptor_id(pn.fn))) is None else
                      [descriptor_id(c[0])[:120], c[1]]})
    rd = []
    for r, k, lo, hi in readers.get(j, []):
        rn = nodes[r]
        rd.append({"node": r, "family": fam(rn), "form": rn.form, "n": getattr(rn, "n", 1),
                   "axes": list(getattr(rn, "axes", ()) or ()), "arg": k, "lo": lo, "hi": hi})
    rec["multimem"].append({"node": j, "form": n.form, "statics": {k: str(v)[:200] for k, v in S.items()},
                            "trees_uniform": len(set(map(str, trees))) == 1, "parts": parts[:12], "parts_runs": len(parts),
                            "grain": grain, "readers": rd})
fams = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}
t1 = time.perf_counter()
w = NW.NodeWindow(fn, fams).lower()
rec["lower_s"] = round(time.perf_counter() - t1, 1)
prod = sorted({p["node"] for m in rec["multimem"] for p in m["parts"] if "node" in p})
rec["producer_outs"] = {}
for j in prod:
    o = w.outs.get(j)
    rec["producer_outs"][j] = o if isinstance(o, str) else {
        "members": o.members, "words": o.words,
        "pieces": [[p.lo, p.hi, p.row.tid[:120], p.row.unit, p.row.out, p.row.words, p.step, p.member] for p in o.pieces]}
rd_nodes = sorted({r["node"] for m in rec["multimem"] for r in m["readers"]})
rec["reader_outs"] = {j: (w.outs[j] if isinstance(w.outs.get(j), str) else "lowered") for j in rd_nodes}
(out / f"probe_l{layers}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
print("REQBLOCK probe", layers, len(rec["multimem"]), rec["program_s"], rec["lower_s"])
