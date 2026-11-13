"""CIR-274 4b (DSAMOE), from REQBLOCK's p3_window.py: rank 0's class-G request Program at 43 prompt ids and 2 outputs in the
registered rank form, lowered into a window by `node_window` with the frame's families and, unless FAMS is ``frame``,
`node_window_sparse`'s beside them; its census: digest, units, templates and wires by family, free inputs by family, the
nodes left unlowered, the peers. Run from the repository root:

    .venv/bin/python p3_window.py OUT [LAYERS] [RANK] [FAMS]

Writes OUT/p3_l<LAYERS>_r<RANK>_<FAMS>.json and one DSAMOE line a stage."""
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
fams = sys.argv[4] if len(sys.argv) > 4 else "sparse"
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

LP, T = 43, 1
families = dict(NW.FAMILIES) if fams == "frame" else {**NW.FAMILIES, **NS.FAMILIES}
path = out / f"p3_l{layers}_r{rank}_{fams}.json"
rec: dict = {"LP": LP, "outputs": T + 1, "layers": layers, "rank": rank, "families": sorted(families)}


def peak() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 3)


def save(label: str, d: dict) -> None:
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("DSAMOE p3", label, json.dumps(d, default=str)[:4000], flush=True)


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
    circuit, w = NW.node_window(p.program.fn, families=families)
    cen = w.census(circuit)
    rec["window"] = {**cen, "templates_total": len(circuit.units), "units_total": sum(circuit.units.values()),
                     "wire_patterns": len(circuit.wires), "units_by_template": dict(sorted(circuit.units.items())),
                     "cuts": circuit.cuts, "free_list": [vars(f) for f in w.free]}
except BaseException as e:  # noqa: BLE001 - a refusal is a result
    rec["window_error"] = {"error": f"{type(e).__name__}: {e}"[:3000], "trace": traceback.format_exc()[-4000:]}
rec["window_wall_s"], rec["peak_gib"] = round(time.perf_counter() - t0, 1), peak()
save("window", {k: v for k, v in (rec.get("window") or rec.get("window_error")).items()
                if k not in ("peers", "free_list", "units_by_template", "trace", "cuts")} | {"wall_s": rec["window_wall_s"],
                                                                                              "peak_gib": rec["peak_gib"]})
