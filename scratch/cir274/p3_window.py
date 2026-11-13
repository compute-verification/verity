"""CIR-274 delivery B (REQBLOCK): rank 0's class-G request Program at 43 prompt ids and 2 outputs in the registered rank form,
lowered into a window by `node_window` (the block-FP8 linears at the coordinate grain and the scale-cut norms), with its
census: digest, units, templates and wires by family, free inputs by family, the nodes left unlowered, the peers. Run from
the repository root:

    .venv/bin/python scratch/cir274/p3_window.py OUT [LAYERS] [RANK]

Writes OUT/p3_l<LAYERS>_r<RANK>.json and one REQBLOCK line a stage."""
import importlib.util
import json
import resource
import sys
import time
import traceback
from pathlib import Path

out = Path(sys.argv[1])
layers = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else 4
rank = int(sys.argv[3]) if len(sys.argv) > 3 else 0
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

LP, T = 43, 1
path = out / f"p3_l{layers}_r{rank}.json"
rec: dict = {"LP": LP, "outputs": T + 1, "layers": layers, "rank": rank}


def peak() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 3)


def save(label: str, d: dict) -> None:
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("REQBLOCK p3", label, json.dumps(d, default=str)[:3000], flush=True)


t0 = time.perf_counter()
reg = cc.registered(rank, layers, "eager")
advice, ingress = cc.session([(0, LP, T + 1)])
u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
p = request_program(reg, u)
rec["program"] = {"digest": p.digest, "nodes": len(p.program.fn.body.nodes), "wall_s": round(time.perf_counter() - t0, 1),
                  "peak_gib": peak()}
save("program", rec["program"])
t0 = time.perf_counter()
try:
    circuit, w = NW.node_window(p.program.fn)
    cen = w.census(circuit)
    rec["window"] = {**cen, "templates_total": len(circuit.units), "units_total": sum(circuit.units.values()),
                     "wire_patterns": len(circuit.wires), "units_by_template": dict(sorted(circuit.units.items())),
                     "cuts": circuit.cuts, "free_list": [vars(f) for f in w.free]}
except BaseException as e:  # noqa: BLE001 - a refusal is a result
    rec["window_error"] = {"error": f"{type(e).__name__}: {e}"[:3000], "trace": traceback.format_exc()[-4000:]}
rec["window_wall_s"], rec["peak_gib"] = round(time.perf_counter() - t0, 1), peak()
save("window", {k: v for k, v in (rec.get("window") or rec.get("window_error")).items()
                if k not in ("peers", "free_list", "units_by_template", "trace")} | {"wall_s": rec["window_wall_s"],
                                                                                      "peak_gib": rec["peak_gib"]})
