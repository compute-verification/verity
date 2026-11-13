"""CIR-274 DSAMOE: the description's fields (`Circuit.description`: templates, wires, cuts, registered, ingress, order),
each by digest and length, at 4 layers, rank 0, eager, the census's Build. Run from the root of the tree to measure:

    .venv/bin/python p3_parts.py OUT TAG OUTS:LP...

Writes OUT/parts_<TAG>.json and one DSAMOE line per length, so two trees (`fix-chain-ranges.diff` applied or not) compare
field by field."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity.commitments.windows import interface as IF  # noqa: E402
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

reg = cc.registered(0, 4, "eager")
families = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}
rec = {}
for arg in sys.argv[3:]:
    OUTS, LP = map(int, arg.split(":"))
    advice, ingress = cc.session([(0, LP, OUTS)])
    u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
    w = NW.NodeWindow(request_program(reg, u).program.fn, families, NW._GRAIN, NW.NAME, 1, True).lower()
    c = w.circuit()
    d = c.description()
    r = {"digest": hashlib.sha512(IF.circuit_bytes(d)).hexdigest()}
    for k, v in d.items():
        r[k] = [hashlib.sha512(json.dumps(v, sort_keys=True).encode()).hexdigest()[:16], len(v)]
    rec[arg] = r
    (out / f"parts_{sys.argv[2]}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("DSAMOE parts", sys.argv[2], arg, json.dumps(r), flush=True)
