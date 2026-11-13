"""CIR-274 DSAMOE: the joint window (families frame, dense and sparse) of rank 0's class-G request Program at 43 prompt ids,
2 outputs and LAYERS layers, built once with each phase timed, and the cost of the experts' key wires measured in it.

Phases timed by wrapping, not by editing the frame: `NodeWindow.lower`, `_Builder.settle` (with `check_proof_block` inside
it, also timed alone), `interface.coalesce`, `interface.ordered`, and `Circuit`'s own `check_wires` and `check_inputs`. Then:
the description's bytes (`interface.circuit_bytes`) in total and by part; the key wires (an `EmbeddingRead_v1` reader's
``tok``) and w2's routed-weight wires (a `MoeExpertFp8BlockGemmW` reader's ``g``), before and after coalescing, their bytes
and Python objects' memory (tracemalloc over a copy); `check_wires` and `ordered` re-timed over the circuit's wires with
and without them; the Program's node families; wire patterns by reader family after coalescing. Run from the repository
root:

    .venv/bin/python p3_keycost.py OUT [LAYERS] [PROMPT_IDS] [OUTPUTS]

Writes OUT/keycost_l<LAYERS>.json (43 prompt ids, 2 outputs, the defaults) or OUT/keycost_l<LAYERS>_p<P>_o<O>.json, and one
DSAMOE line a stage."""
import hashlib
import importlib.util
import json
import resource
import sys
import time
import tracemalloc
import traceback
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1])
layers = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else 4
LP = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 43
OUTS = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else 2
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity.commitments.windows import interface as IF  # noqa: E402
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict import window_builder as WB  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

T = OUTS - 1
path = out / (f"keycost_l{layers}.json" if (LP, OUTS) == (43, 2) else f"keycost_l{layers}_p{LP}_o{OUTS}.json")
rec: dict = {"LP": LP, "outputs": T + 1, "layers": layers, "rank": 0, "sets": "frame,dense,sparse"}
timers: Counter = Counter()


def peak() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 3)


def rss() -> float:
    return round(int(Path("/proc/self/statm").read_text().split()[1]) * 4096 / 2**30, 3)


def save(label: str, d: dict) -> None:
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("DSAMOE keycost", label, json.dumps(d, default=str)[:3000], flush=True)


def timed(name: str, f):
    def g(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            timers[name] += time.perf_counter() - t
    return g


orig = {"check_wires": IF.check_wires, "ordered": IF.ordered}
for name in ("check_wires", "check_inputs", "ordered", "coalesce", "check_proof_block"):
    setattr(IF, name, timed(name, getattr(IF, name)))
WB._Builder.settle = timed("settle", WB._Builder.settle)

t0 = time.perf_counter()
reg = cc.registered(0, layers, "eager")
advice, ingress = cc.session([(0, LP, T + 1)])
u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
p = request_program(reg, u)
fn = p.program.fn
rec["program"] = {"digest": p.digest, "nodes": len(fn.body.nodes), "wall_s": round(time.perf_counter() - t0, 1),
                  "peak_gib": peak(), "rss_gib": rss(),
                  "families": dict(sorted(Counter(NW._family(n) for n in fn.body.nodes).items()))}
save("program", {k: v for k, v in rec["program"].items() if k != "families"})

families = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}
try:
    t0 = time.perf_counter()
    w = NW.NodeWindow(fn, families, NW._GRAIN, NW.NAME, 1, True).lower()
    rec["lower"] = {"wall_s": round(time.perf_counter() - t0, 1), "peak_gib": peak(), "rss_gib": rss()}
    save("lower", rec["lower"])
    t0 = time.perf_counter()
    circuit = w.circuit()
    rec["circuit"] = {"wall_s": round(time.perf_counter() - t0, 1), "peak_gib": peak(), "rss_gib": rss(),
                      "phases_s": {k: round(v, 2) for k, v in timers.items()}}
    save("circuit", rec["circuit"])

    t0 = time.perf_counter()
    desc = circuit.description()
    blob = IF.circuit_bytes(desc)
    parts = {k: len(IF.circuit_bytes({k: v})) for k, v in desc.items()}
    reg_parts = {k: len(IF.circuit_bytes({k: v})) for k, v in desc.get("registered", {}).items()}
    rec["description"] = {"digest": hashlib.sha512(blob).hexdigest(), "bytes": len(blob), "parts": parts,
                          "registered_parts": reg_parts, "wall_s": round(time.perf_counter() - t0, 1)}
    del blob, desc
    save("description", rec["description"])

    def is_key(x) -> bool:
        return x.reader.startswith("EmbeddingRead_v1") and x.port == "tok"

    def is_g(x) -> bool:
        return x.reader.startswith("MoeExpertFp8BlockGemmW") and x.port == "g"

    key = [x for x in circuit.wires if is_key(x)]
    g = [x for x in circuit.wires if is_g(x)]
    rest = [x for x in circuit.wires if not (is_key(x) or is_g(x))]

    def mem(ws) -> int:
        tracemalloc.start()
        copies = [IF.Wire.of(x.doc()) for x in ws]
        n = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        del copies
        return n

    def clock(f, *a) -> float:
        t = time.perf_counter()
        f(*a)
        return round(time.perf_counter() - t, 2)

    starts = {tid: [at for at, _n, _slot, _step in spans] for tid, spans in w.b.spans.items()}
    lays = circuit.layouts
    by_fam: Counter = Counter(w.family_of(x.reader, x.at) for x in circuit.wires)
    rec["keys"] = {
        "wires": len(circuit.wires), "wires_before_coalesce": len(w.b.wires),
        "key_wires": len(key), "g_wires": len(g),
        "key_wires_before_coalesce": sum(1 for x in w.b.wires if is_key(x)),
        "g_wires_before_coalesce": sum(1 for x in w.b.wires if is_g(x)),
        "bytes_all_wires": len(IF.circuit_bytes({"wires": [x.doc() for x in circuit.wires]})),
        "bytes_key_wires": len(IF.circuit_bytes({"wires": [x.doc() for x in key]})),
        "bytes_g_wires": len(IF.circuit_bytes({"wires": [x.doc() for x in g]})),
        "pymem_all_wires": mem(circuit.wires), "pymem_key_wires": mem(key), "pymem_g_wires": mem(g),
        "check_wires_all_s": clock(orig["check_wires"], lays, circuit.wires, circuit.cuts, circuit.order),
        "check_wires_without_s": clock(orig["check_wires"], lays, rest, circuit.cuts, circuit.order),
        "ordered_all_s": clock(orig["ordered"], w.b.units, circuit.wires, starts),
        "ordered_without_s": clock(orig["ordered"], w.b.units, rest, starts),
        "openings": len(circuit.registered.openings), "roots": len(circuit.registered.roots),
        "tables": len(circuit.registered.tables),
        "wire_patterns_by_reader_family": dict(by_fam.most_common()),
        "peak_gib": peak(), "rss_gib": rss(),
    }
    import bisect
    from collections import defaultdict

    def step_of(tid: str, at: int) -> int:
        spans = w.b.spans[tid]
        return spans[bisect.bisect_right([s[0] for s in spans], at) - 1][3]

    reg_ = circuit.registered
    per_param_open = Counter(o.table for o in reg_.openings)
    per_param_root = Counter(r.table for r in reg_.roots)
    xt = [t for t in reg_.tables if "/b" in t.param and t.param.endswith("/table")]
    open_steps: dict = defaultdict(set)
    for o in reg_.openings:
        open_steps[o.table].add(step_of(o.reader, o.at))
    rec["steps"] = {
        "steps": w.steps, "moe_nodes": rec["program"]["families"].get("MoeExpertFp8Gemm", 0),
        "key_wires_by_step": dict(sorted(Counter(step_of(x.reader, x.at) for x in key).items())),
        "g_wires_by_step": dict(sorted(Counter(step_of(x.reader, x.at) for x in g).items())),
        "key_wire_radix": dict(Counter(str(x.radix) for x in key).most_common(6)),
        "expert_tables": len(xt), "expert_table_params": len({t.param for t in xt}),
        "openings_per_param": dict(Counter(per_param_open[t.name] for t in xt)),
        "roots_per_param": dict(Counter(per_param_root[t.name] for t in xt)),
        "opening_steps_per_param": dict(Counter(len(open_steps[t.name]) for t in xt)),
        "bytes_tables_expert": len(IF.circuit_bytes({"t": [t.doc() for t in xt]})),
        "bytes_roots": len(IF.circuit_bytes({"t": [r.doc() for r in reg_.roots]})),
        "bytes_openings": len(IF.circuit_bytes({"t": [o.doc() for o in reg_.openings]})),
    }
    save("steps", rec["steps"])
    save("keys", {k: v for k, v in rec["keys"].items() if k != "wire_patterns_by_reader_family"})
except BaseException as e:  # noqa: BLE001 - a refusal is a result
    rec["error"] = {"error": f"{type(e).__name__}: {e}"[:3000], "trace": traceback.format_exc()[-4000:],
                    "phases_s": {k: round(v, 2) for k, v in timers.items()}, "peak_gib": peak()}
    save("error", rec["error"])
