"""CIR-274 EMBED (from DENSE's p3_window.py): rank 0's class-G request Program at 43 prompt ids and 2 outputs in the registered
rank form, lowered into a window with the frame's, DENSE's, DSAMOE's and EMBED's families, and once without EMBED's; each
window's census, and the tables as declared and the split of tables from plain weights. Run from the repository root:

    .venv/bin/python p3_window.py OUT [LAYERS] [RANK]

Writes OUT/p3_l<LAYERS>_r<RANK>.json and one EMBED line a stage."""
import importlib.util
import json
import resource
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1])
layers = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else 4
rank = int(sys.argv[3]) if len(sys.argv) > 3 else 0
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity_vllm.program.predict import node_window as NW, node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_embed as NE, node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict import window_weights as WW  # noqa: E402
from verity_vllm.pipeline import expert_tables as XT  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

LP, T = 43, 1
path = out / f"p3_l{layers}_r{rank}.json"
rec: dict = {"LP": LP, "outputs": T + 1, "layers": layers, "rank": rank}


def peak() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 3)


def save(label: str, d: dict) -> None:
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("EMBED p3", label, json.dumps(d, default=str)[:4000], flush=True)


t0 = time.perf_counter()
reg = cc.registered(rank, layers, "eager")
advice, ingress = cc.session([(0, LP, T + 1)])
u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
p = request_program(reg, u)
rec["program"] = {"digest": p.digest, "nodes": len(p.program.fn.body.nodes), "wall_s": round(time.perf_counter() - t0, 1),
                  "peak_gib": peak()}
save("program", rec["program"])
base = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}
windows = {}
for key, fams in (("window", {**base, **NE.FAMILIES}), ("without_embed", base)):
    t0 = time.perf_counter()
    try:
        circuit, w = NW.node_window(p.program.fn, families=fams)
        windows[key] = (circuit, w)
        cen = w.census(circuit)
        rec[key] = {**cen, "templates_total": len(circuit.units), "units_total": sum(circuit.units.values()),
                    "wire_patterns": len(circuit.wires), "units_by_template": dict(sorted(circuit.units.items())),
                    "cuts": circuit.cuts, "free_list": [vars(f) for f in w.free],
                    "wall_s": round(time.perf_counter() - t0, 1), "peak_gib": peak()}
    except BaseException as e:  # noqa: BLE001 - a refusal is a result
        rec[key] = {"error": f"{type(e).__name__}: {e}"[:3000], "trace": traceback.format_exc()[-4000:]}
    save(key, {k: v for k, v in rec[key].items() if k not in ("peers", "free_list", "units_by_template", "trace", "cuts")})
if len(windows) == 2:
    (c, w), (c0, _w0) = windows["window"], windows["without_embed"]
    read = [t for t in c.units if t.startswith("EmbeddingRead_v1{V=154880,H=192}")]
    shard = [t.param for t in c.registered.tables if t.param.endswith("/vocab")]
    plain, tables = NE.split(c.registered.params)
    by = NE.tables_of(c.registered.params)
    nonces = XT.nonces(by, [label for label, _f, _n in w.b.weights], rank)
    rec["embed"] = {
        "read_template": read, "read_units": {t: c.units[t] for t in read},
        "read_allocs": {t: w.allocs[t] for t in read},
        "wires_from_read": [vars(x) for x in c.wires if x.writer in read],
        "free_on_read": [vars(f) for f in w.free if f.tid in read],
        "tables": {"count": len(c.registered.tables), "vocab": len(shard),
                   "shapes": sorted(Counter((t.offset, t.words, t.rows) for t in c.registered.tables
                                            if t.param.endswith("/vocab")).items()),
                   "other": sorted(Counter((t.words, t.rows) for t in c.registered.tables
                                           if not t.param.endswith("/vocab")).items())},
        "split": {"plain": len(plain), "tables": len(tables), "vocab": sum(len(v) for v in by.values()),
                  "plain0": len(NE.split(c0.registered.params)[0]), "tables0": len(NE.split(c0.registered.params)[1]),
                  "plain_equal_without_embed": plain == NE.split(c0.registered.params)[0],
                  "plain_nonces_equal": WW.weight_nonces(plain) == WW.weight_nonces(NE.split(c0.registered.params)[0]),
                  "plain_nonces_moved_if_unsplit": sum(WW.weight_nonces(c.registered.params)[q] != n
                                                       for q, n in WW.weight_nonces(plain).items()),
                  "vocab_nonces": [min(nonces.values()), max(nonces.values()), len(set(nonces.values()))],
                  "tables_nonce": XT.TABLES_NONCE},
    }
    save("embed", rec["embed"] | {"free_on_read": len(rec["embed"]["free_on_read"])})
