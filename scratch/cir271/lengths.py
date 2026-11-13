"""CIR-271 part 3 (REQBLOCK), the lengths: `window_circuit`'s class Program under GLM-5.3's class-G registration, and what a
Program of one prompt row and P - 1 decodes calls against the (256, 31) request Program's prefill.

    .venv/bin/python scratch/cir271/lengths.py OUT [LAYERS]

Writes OUT/lengths.json and prints one REQBLOCK line a result. Changes no builder.
"""
import importlib.util
import json
import resource
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1])
layers = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity.circuits.codec import descriptor_id
from verity.circuits.partition_object import calls
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict.build import build_program
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import plan_steps, request_program

LP, O = 256, 32
P = LP + O - 1
rec: dict = {"layers": layers, "class": [LP, O], "positions": P}
reg = cc.registered(0, layers, "eager")


def name(fn) -> str:
    return getattr(fn, "name", None) or fn.id.split("{")[0]


def tmpl(prog) -> tuple[Counter, dict]:
    nodes, c, names = prog.root.body.nodes, Counter(), {}
    for k, fn in calls(prog):
        t = descriptor_id(fn)
        c[t] += nodes[k].n if nodes[k].form == "batch" else 1
        names[t] = name(fn)
    return c, names


def stage(label, f):
    t0 = time.perf_counter()
    try:
        rec[label] = {**(f() or {}), "wall_s": round(time.perf_counter() - t0, 1)}
    except BaseException as e:  # noqa: BLE001 - a refusal is a result
        rec[label] = {"error": f"{type(e).__name__}: {e}"[:1500], "trace": traceback.format_exc()[-2500:],
                      "wall_s": round(time.perf_counter() - t0, 1)}
    rec[label]["peak_gib_after"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)
    (out / "lengths.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("REQBLOCK lengths", label, json.dumps({k: v for k, v in rec[label].items() if k != "trace"}, default=str)[:2500], flush=True)


cls = WC.WindowClass(1, LP, O)
stage("class_program", lambda: {"digest": WC.class_program(reg, cls).digest})
stage("lower", lambda: {"programs": len(WC.lower(reg, cls, programs=1, wired=False, provable=False).programs)})
S: dict = {}


def padded():
    phi = Phi.from_json({**reg.phi.to_json(), "workload": {"requests": [{"prompt_len": 1, "max_tokens": P, "arrive_step": 0}]}})
    r = build_program(phi, "request", LP=1, T=P - 1, dead_free=True, batched=True, with_desc=False)
    S["pad"] = tmpl(r.program)
    return {"root_nodes": len(r.program.root.body.nodes), "params": [n for n, _ in r.program.fn.body.params][:3] + ["..."],
            "peer_params": sum(n.startswith("peers_") for n, _ in r.program.fn.body.params),
            "templates": len(S["pad"][0]), "instances": sum(S["pad"][0].values())}


def served():
    advice, ingress = cc.session([(0, LP, O)])
    u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
    S["srv"] = tmpl(request_program(reg, u).program)
    a, b = S["srv"][0], S["pad"][0]
    names = {**S["srv"][1], **S["pad"][1]}
    only_srv, only_pad = a.keys() - b.keys(), b.keys() - a.keys()
    return {"templates": len(a), "instances": sum(a.values()), "in_both": len(a.keys() & b.keys()),
            "only_in_256_row_prefill": len(only_srv), "only_in_padded": len(only_pad),
            "only_in_256_row_prefill_by_name": dict(Counter(names[t] for t in only_srv).most_common(30)),
            "only_in_padded_by_name": dict(Counter(names[t] for t in only_pad).most_common(30)),
            "examples_256": sorted(only_srv)[:12], "examples_padded": sorted(only_pad)[:12]}


stage("padded_form", padded)
if "pad" in S:
    stage("against_256_row_prefill", served)
