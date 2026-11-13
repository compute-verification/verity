"""CIR-274 DSAMOE: the node window's value check (`node_window_values`) on the census's Build (`benchmarks/compile_cost/run.py`
`registered`, rank 0, eager) at LAYERS layers, PROMPT_IDS prompt ids and 2 outputs, after DENSE's `values_run.py`: the window
built as served and with ``stand_ins`` (their digests held equal), the Program evaluated on made-up values, every template
checked (`node_window_values.run`, its sampling). Run from the repository root:

    .venv/bin/python p3_values.py OUT LAYERS PROMPT_IDS [MOST]

Writes OUT/values_l<LAYERS>_p<PROMPT_IDS>.json."""
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
layers, LP = int(sys.argv[2]), int(sys.argv[3])
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

import verity.definitions.numerics.rows  # noqa: E402,F401  (registers the row kernels)
from verity.commitments.windows import interface as IF  # noqa: E402
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict import node_window_values as NV  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

MOST = int(sys.argv[4]) if len(sys.argv) > 4 else NV.SAMPLE
SEED = 274
T0 = time.perf_counter()


def log(s):
    print(f"[{time.perf_counter() - T0:7.1f}s {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:5.2f} GiB] {s}",
          flush=True)


FAM = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}
reg = cc.registered(0, layers, "eager")
advice, ingress = cc.session([(0, LP, 2)])
u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
p = request_program(reg, u)
fn = p.program.fn
rec = {"layers": layers, "ids": LP, "outputs": 2, "seed": SEED, "most": MOST, "program_digest": p.digest}
c0, w0 = NW.node_window(fn, families=FAM)
c, w = NW.node_window(fn, families=FAM, stand_ins=True)
rec["description_digest"] = hashlib.sha512(IF.circuit_bytes(c0.description())).hexdigest()
d0, d1 = w0.census(c0)["digest"], w.census(c)["digest"]
rec["window_digest"], rec["stand_in_window_digest"] = d0, d1
if d0 != d1:
    raise SystemExit(f"the stand-in window's digest {d1} is not the served window's {d0}")
del c0, w0
rec["templates"], rec["units"] = len(c.ids), sum(n for _f, n in c.templates)
log(f"window {d0[:16]}: {rec['templates']} templates, {rec['units']} units")
rec["native"] = NV.native_selftest()
inputs = NV.made_up(fn, [(7 * k + 1) % 256 for k in range(LP)], SEED)
values = NV.Values(fn, inputs, w, log).evaluate()
log("program evaluated")
tables = NV.Tables(c, values, w, p.digest)
ck = NV.Check(c, w, values, tables)
res = NV.run(ck, SEED, log, MOST)
rec["per_template"] = res
rec["totals"] = {"templates_checked": len(res), "checked": sum(r.get("checked", 0) for r in res.values()),
                 "mismatches": sum(r["mismatches"] for r in res.values()),
                 "templates_with_mismatches": sorted(t for t, r in res.items() if r["mismatches"]),
                 "units_with_undeclared_inputs": len(ck.unknown),
                 "rows_two_pieces_hold": len(ck.conflicts), "tables_committed": len(tables.held)}
rec["wall_s"] = round(time.perf_counter() - T0, 1)
rec["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
(out / f"values_l{layers}_p{LP}.json").write_text(json.dumps(rec, indent=1, default=str))
log(f"DSAMOE values {json.dumps(rec['totals'])}")
