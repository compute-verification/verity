"""CIR-271 part 1 (REQBLOCK): GLM-5.3's node request Program at LP 256, T 31, class G at TP8 on H200, one request alone, greedy.

    .venv/bin/python scratch/cir271/node1.py OUT eager|compiled [LAYERS]

compile_cost's registration carries no workload, so `common.launch_batch` reads `workload["sweep"]` (KeyError). Here φ is that
registration's with one declared request (prompt_len 256, max_tokens 32, arrive_step 0), the form `score.phi_of` gives a row's
manifest requests. Writes OUT/node_<ex>.json after every stage and prints one REQBLOCK line a result. Changes no builder.
"""
import importlib.util
import json
import os
import resource
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

out, ex = Path(sys.argv[1]), sys.argv[2]
layers = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity.circuits.codec import descriptor_id, program_digest
from verity.circuits.partition_object import TemplatePartition, calls, template_instances_query
from verity.circuits.program import Program
from verity_vllm.program.predict.build import build_program
from verity_vllm.program.predict.node import build_node_program, check_ranks
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import plan_steps, request_program

LP, T = 256, int(os.environ.get("OUTPUTS") or 32) - 1
rec: dict = {"execution": ex, "LP": LP, "T": T, "layers": layers}
path = out / f"node_{ex}_o{T + 1}.json"


def peak() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)


def save() -> None:
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")


def say(what: str, d) -> None:
    print("REQBLOCK", ex, what, json.dumps(d, default=str)[:1800], flush=True)


_ids: dict[int, str] = {}


def tid(fn) -> str:
    k = id(fn)
    if k not in _ids:
        _ids[k] = descriptor_id(fn)
    return _ids[k]


def name(fn) -> str:
    return getattr(fn, "name", None) or fn.id.split("{")[0]


def templates(nodes) -> tuple[Counter, dict]:
    """Each callee's instances over `nodes` as `_instances` counts them (a batch its members), by descriptor id; and id -> name."""
    c, names = Counter(), {}
    for n in nodes:
        if n.form in ("call", "batch"):
            t = tid(n.fn)
            c[t] += n.n if n.form == "batch" else 1
            names[t] = name(n.fn)
    return c, names


def stage(label: str, f):
    t0 = time.perf_counter()
    try:
        v = f()
        rec[label] = {**(v or {}), "wall_s": round(time.perf_counter() - t0, 1), "peak_gib_after": peak()}
    except BaseException as e:  # noqa: BLE001 - a refusal is a result: recorded by name and trace
        rec[label] = {"error": f"{type(e).__name__}: {e}"[:1500], "trace": traceback.format_exc()[-3000:],
                      "wall_s": round(time.perf_counter() - t0, 1), "peak_gib_after": peak()}
    save()
    say(label, {k: v for k, v in rec[label].items() if k not in ("trace", "templates", "top", "only_node", "only_rank")})
    if "error" in rec[label]:
        print(rec[label]["trace"][-1500:], flush=True)
    return rec[label]


reg = cc.registered(0, layers, ex)
one = {"requests": [{"request_id": "q0", "index": 0, "prompt_len": LP, "max_tokens": T + 1, "arrive_step": 0, "n": 1, "seed": None}]}
phi1 = Phi.from_json({**reg.phi.to_json(), "workload": one})
rec["phi_digest"] = phi1.digest()
S: dict = {}


def build():
    t0 = time.perf_counter()
    r = build_node_program(phi1, "request", LP=LP, T=T, with_desc=False)
    t1 = time.perf_counter()
    d = program_digest(r.program)
    S["r"] = r
    body = r.node_fn.body.nodes
    tc, names = templates(body)
    S["tc"], S["names"] = tc, names
    per_rank = Counter()
    for q, lo, hi in r.spans:
        per_rank[q] += hi - lo
    return {"digest": d, "build_s": round(t1 - t0, 1), "digest_s": round(time.perf_counter() - t1, 1), "wall_min": round((time.perf_counter() - t0) / 60, 2),
            "summary": r.summary(), "root_nodes": len(r.program.root.body.nodes), "body_nodes": len(body),
            "body_forms": dict(Counter(n.form for n in body)), "templates": len(tc), "instances": sum(tc.values()),
            "body_nodes_per_rank": dict(sorted(per_rank.items())),
            "selects": {k: v for k, v in tc.items() if "Select" in names[k]},
            "params": len(r.program.fn.body.params) if hasattr(r.program.fn, "body") else None,
            "params_not_prompt_or_weights": [n for n, _ in r.node_fn.body.params if n != "prompt" and not n.startswith("rank")],
            "top": [(v, names[k], k[:24]) for k, v in tc.most_common(25)]}


b = stage("node", build)
if "error" in b:
    sys.exit(1)
r = S["r"]
rec["node"]["templates_by_id"] = {k: [S["names"][k], v] for k, v in sorted(S["tc"].items())}
save()


def partition_as_built():
    p = TemplatePartition(r.program, template_instances_query(sorted(S["tc"])))
    return {"ranges": len(p.ranges), "population": p.population}


def partition_one_unit():
    p = TemplatePartition(r.program, template_instances_query([tid(r.node_fn)]))
    return {"ranges": len(p.ranges), "population": p.population, "template": tid(r.node_fn)[:40]}


def partition_body_as_root():
    t0 = time.perf_counter()
    p2 = Program(r.node_fn, registered=r.theta, theta=r.theta)
    lower_s = round(time.perf_counter() - t0, 1)
    nodes = p2.root.body.nodes
    forms = Counter(nodes[k].form for k, _ in calls(p2))
    ids = sorted({tid(fn) for k, fn in calls(p2) if nodes[k].form in ("call", "batch")})
    t1 = time.perf_counter()
    p = TemplatePartition(p2, template_instances_query(ids))
    return {"lower_s": lower_s, "root_call_forms": dict(forms), "templates": len(ids), "ranges": len(p.ranges), "population": p.population,
            "partition_s": round(time.perf_counter() - t1, 1)}


stage("partition_as_built", partition_as_built)
stage("partition_one_unit", partition_one_unit)
stage("partition_body_as_root", partition_body_as_root)

rank0 = [n for n, q in zip(r.node_fn.body.nodes, r.rank_of()) if q == 0]


def against(prog) -> dict:
    mine, _ = templates(rank0)
    theirs, names = templates([prog.root.body.nodes[k] for k, _ in calls(prog)])
    names = {**S["names"], **names}
    return {"rank0_span_nodes": len(rank0), "their_root_nodes": len(prog.root.body.nodes),
            "rank0_templates": len(mine), "their_templates": len(theirs), "rank0_instances": sum(mine.values()), "their_instances": sum(theirs.values()),
            "templates_in_both": len(mine.keys() & theirs.keys()), "same_template_counts": mine == theirs,
            "only_node": sorted(((v, names[k]) for k, v in (mine - theirs).items()), reverse=True)[:20],
            "only_rank": sorted(((v, names[k]) for k, v in (theirs - mine).items()), reverse=True)[:20]}


def d1f1():
    advice, ingress = cc.session([(0, LP, T + 1)])
    u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
    S["u"] = u
    p = request_program(reg, u)
    ps = [n for n, _ in p.program.fn.body.params]
    sel, _ = templates([n for n in p.program.fn.body.nodes if "Select" in name(n.fn)])
    return {"digest": p.digest, "unit": {"LP": u.LP, "T": u.T, "C": u.C, "chunks": list(u.chunks), "launch": u.batch()[:3], "stop": u.stop},
            "params": len(ps), "peer_params": sum(n.startswith("peers_") for n in ps), "selects": dict(sel), **against(p.program)}


stage("rank0_registered_form", d1f1)


def no_launch():
    u = S["u"]
    p = build_program(phi1, "request", LP=LP, T=T, dead_free=True, batched=True, stop=u.stop, with_desc=False)
    return {"digest": program_digest(p.program), "same_as_registered": program_digest(p.program) == rec["rank0_registered_form"].get("digest")}


def plain_rank0():
    p = build_program(Phi.from_json({**phi1.to_json(), "tp": {**phi1.tp, "rank": 0}}), "request", LP=LP, T=T, with_desc=False)
    return {"digest": program_digest(p.program), **against(p.program)}


stage("rank0_registered_form_without_launch", no_launch)
stage("rank0_node_form", plain_rank0)
stage("check_ranks", lambda: {"ranks": check_ranks(phi1, r, LP=LP, T=T)})
rec["peak_gib"] = peak()
save()
say("done", {"peak_gib": rec["peak_gib"]})
