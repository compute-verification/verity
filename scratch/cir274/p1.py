"""CIR-274 piece 1 (REQBLOCK): GLM-5.3's class-G request Program at 43 prompt ids and 2 outputs in the registered rank form
(`steps.request_program` at `plan_steps`' unit), at each of the eight ranks, and the node Program (`build_node_program`, one
declared request) with `check_ranks`. Run from the repository root:

    .venv/bin/python scratch/cir274/p1.py OUT [LAYERS]

Writes OUT/p1_l<LAYERS>.json and one REQBLOCK line a stage."""
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

from verity.circuits.codec import program_digest  # noqa: E402
from verity_vllm.program.predict.node import build_node_program, check_ranks  # noqa: E402
from verity_vllm.program.predict.phi import Phi  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

LP, T = 43, 1
rec: dict = {"LP": LP, "outputs": T + 1, "layers": layers, "ranks": {}}
path = out / f"p1_l{layers or 'all'}.json"


def peak() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)


def save(label: str, d: dict) -> None:
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("REQBLOCK p1", label, json.dumps(d, default=str)[:1500], flush=True)


def family(fn) -> str:
    return (getattr(fn, "name", None) or fn.id.split("{")[0]).split("_v")[0]


for r in range(8):
    t0 = time.perf_counter()
    try:
        reg = cc.registered(r, layers, "eager")
        advice, ingress = cc.session([(0, LP, T + 1)])
        u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
        p = request_program(reg, u)
        nodes = p.program.fn.body.nodes
        fams = Counter()
        for n in nodes:
            if n.form in ("call", "batch"):
                fams[family(n.fn)] += n.n if n.form == "batch" else 1
        ps = [name for name, _ in p.program.fn.body.params]
        d = {"digest": p.digest, "body_nodes": len(nodes), "forms": dict(Counter(n.form for n in nodes)),
             "families": dict(sorted(fams.items())), "params": len(ps), "peer_params": sum(x.startswith("peers_") for x in ps),
             "unit": {"LP": u.LP, "T": u.T, "C": u.C, "chunks": list(u.chunks), "launch": u.batch()[:3], "stop": u.stop}}
    except BaseException as e:  # noqa: BLE001 - a refusal is a result
        d = {"error": f"{type(e).__name__}: {e}"[:1500], "trace": traceback.format_exc()[-2500:]}
    d.update(wall_s=round(time.perf_counter() - t0, 1), peak_gib=peak())
    rec["ranks"][r] = d
    save(f"rank{r}", {k: v for k, v in d.items() if k not in ("families", "trace")})

t0 = time.perf_counter()
try:
    reg = cc.registered(0, layers, "eager")
    one = {"requests": [{"request_id": "q0", "index": 0, "prompt_len": LP, "max_tokens": T + 1, "arrive_step": 0, "n": 1,
                         "seed": None}]}
    phi1 = Phi.from_json({**reg.phi.to_json(), "workload": one})
    node = build_node_program(phi1, "request", LP=LP, T=T, with_desc=False)
    rec["node"] = {"digest": program_digest(node.program), "body_nodes": len(node.node_fn.body.nodes)}
    rec["check_ranks"] = check_ranks(phi1, node, LP=LP, T=T)
except BaseException as e:  # noqa: BLE001
    rec["node_error"] = {"error": f"{type(e).__name__}: {e}"[:1500], "trace": traceback.format_exc()[-2500:]}
rec["node_wall_s"], rec["peak_gib"] = round(time.perf_counter() - t0, 1), peak()
save("node", {k: rec.get(k) for k in ("node", "check_ranks", "node_error", "node_wall_s", "peak_gib")})
