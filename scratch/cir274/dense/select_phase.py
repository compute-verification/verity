"""CIR-274 DENSE piece 5: where the select driver's value check spends its time, phase by phase, with a Python stack dump to
stderr every ``--trace`` seconds.

Usage (from adapters/vllm): select_phase.py OUT [--ids 43] [--layers 4] [--trace 120] [--no-select]

Phases: the served window, the stand-in window, `made_up`, the Program's evaluation (`Values.evaluate`, with its time by
family), `Tables`, `Check`. Writes $OUT/select_phase.json after each."""
import argparse
import faulthandler
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, "tests/program/predict")
from test_node_window import glm_request  # noqa: E402

from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_select as NSEL  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict import node_window_values as NV  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--ids", type=int, default=43)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--trace", type=int, default=120)
ap.add_argument("--no-select", action="store_true")
a = ap.parse_args()
faulthandler.dump_traceback_later(a.trace, repeat=True, file=sys.stderr)
T0 = time.perf_counter()
rec: dict = {"ids": a.ids, "layers": a.layers, "select": not a.no_select, "phases": {}}
path = Path(a.out, f"select_phase_{a.layers}l{'_nosel' if a.no_select else ''}.json")


def phase(name, f):
    t = time.perf_counter()
    print(f"[{t - T0:7.1f}s] {name} ...", flush=True)
    got = f()
    rec["phases"][name] = {"s": round(time.perf_counter() - t, 1),
                           "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    print(f"[{time.perf_counter() - T0:7.1f}s] {name}: {rec['phases'][name]}", flush=True)
    path.write_text(json.dumps(rec, indent=1, default=str))
    return got


req = glm_request(a.ids, a.layers)
fn = req.program.fn
FAM = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES, **({} if a.no_select else NSEL.FAMILIES)}
phase("served window", lambda: NW.node_window(fn, families=FAM))
c, w = phase("stand-in window", lambda: NW.node_window(fn, families=FAM, stand_ins=True))
rec["stood"] = len(w.stood)
inputs = phase("made_up", lambda: NV.made_up(fn, [(7 * k + 1) % 256 for k in range(a.ids)], 274))
values = phase("evaluate", lambda: NV.Values(fn, inputs, w, print).evaluate())
rec["evaluate_by_family"] = {k: round(v, 1) for k, v in sorted(values.times.items(), key=lambda x: -x[1])}
path.write_text(json.dumps(rec, indent=1, default=str))
tables = phase("tables", lambda: NV.Tables(c, values, w, req.digest))
phase("check", lambda: NV.Check(c, w, values, tables))
faulthandler.cancel_dump_traceback_later()
print(json.dumps(rec, indent=1, default=str))
