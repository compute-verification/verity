"""DENSE's value check (`node_window_values`) of rank 0's node window at 43 ids, 2 outputs and LAYERS layers, with the
frame's, dense, sparse and multimem families: every template sampled; then one planted wire into the all-reduce's
``parts`` (row 0's first wire copied to row 3, `plant_wire_row` keep) and the check rerun. Run from the repo root:

    python mm_values.py OUT LAYERS SEED
"""
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

out, layers, seed = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_multimem as NM  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict import node_window_values as NV  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

FAMILIES = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES, **NM.FAMILIES}
LP, T = 43, 1
rec: dict = {"LP": LP, "layers": layers, "seed": seed}
path = out / f"mm_values_l{layers}.json"


def peak() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 3)


LOG = open(out / f"mm_values_l{layers}.log", "a", buffering=1)


def log(line: str) -> None:
    LOG.write(line + "\n")


def save() -> None:
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")


reg = cc.registered(0, layers, "eager")
advice, ingress = cc.session([(0, LP, T + 1)])
u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
p = request_program(reg, u)
fn = p.program.fn
circuit, w = NW.node_window(fn, families=FAMILIES, stand_ins=True)
plain, _w = NW.node_window(fn, families=FAMILIES)
rec["digest_stand_ins"], rec["digest_plain"] = w.census(circuit)["digest"], _w.census(plain)["digest"]
del plain, _w
mm = sorted({tid for tid, a in w.allocs.items() if any(f == NM.FAMILY for _x, _n, f in a)})
site_gathers = sorted({tid for tid, a in w.allocs.items() for at, _n, f in a
                       if f == NW.GATHER and NW._family(w.fn.body.nodes[w.node_of[(tid, at)]]) == NM.FAMILY})
rec["new_templates"] = {"multimem": mm, "gathers_at_multimem_sites": site_gathers}
print("MM new templates", rec["new_templates"], flush=True)
log(f"new templates {rec['new_templates']}")
prompt = [int(x) for x in (np.arange(LP) * 997 + 11) % 150000]
t0 = time.perf_counter()
values = NV.Values(fn, NV.made_up(fn, prompt, seed), w).evaluate()
rec["values_s"], rec["peak_after_values_gib"] = round(time.perf_counter() - t0, 1), peak()
log(f"values {rec['values_s']} s, {rec['peak_after_values_gib']} GiB")
tables = NV.Tables(circuit, values, w, p.digest)
c2, reader, info = NV.plant_wire_row(circuit, NM.FAMILY, "parts", row=3, keep=True)
x = c2.wires[info["wire"]]
ck = NV.Check(circuit, w, values, tables)
rng = np.random.default_rng(seed)
picked = {tid: NV.sample(ck.counts[tid], rng) for tid in ck.c.ids}
covered = [u for u in picked[reader] if x.at <= u < x.at + x.reader_units]
info |= {"at": x.at, "sampled_of_reader": len(picked[reader]), "sampled_units_covered": len(covered)}
ck2 = NV.Check(c2, w, values, tables)
t0 = time.perf_counter()
res2 = NV.run(ck2, seed, log=log, only=[reader])
rec["planted"] = {"info": info, "result": res2[reader], "check_s": round(time.perf_counter() - t0, 1)}
rec["peak_after_planted_gib"] = peak()
save()
print("MM planted", json.dumps({"info": info, "checked": res2[reader].get("checked"),
                                "mismatches": res2[reader].get("mismatches"), "first": res2[reader].get("first")},
                               default=str), flush=True)
ck = NV.Check(circuit, w, values, tables)
log("full check")
t0 = time.perf_counter()
res = NV.run(ck, seed, log=log)
rec["check_s"] = round(time.perf_counter() - t0, 1)
rec["totals"] = {"templates": len(res), "checked": sum(r.get("checked", 0) for r in res.values()),
                 "mismatches": sum(r.get("mismatches", 0) for r in res.values()),
                 "unanchored_unevaluated": [t for t, r in res.items() if not ck.anchored(t) and not r.get("evaluated")]}
rec["new"] = {t: res[t] for t in mm + site_gathers}
rec["conflicts"] = len(ck.conflicts)
save()
print("MM totals", rec["totals"], flush=True)
for t in mm + site_gathers:
    r = res[t]
    print("MM new", t[:80], {k: r.get(k) for k in ("units", "sampled", "checked", "mismatches", "evaluated", "through",
                                                   "first")}, flush=True)

rec["peak_gib"] = peak()
save()
print("MM peak_gib", rec["peak_gib"], flush=True)
