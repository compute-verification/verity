import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests/adapters/vllm"), str(ROOT), str(ROOT / "adapters/vllm"), str(ROOT / "adapters/torch"),
                str(ROOT / "benchmarks/harness"), str(ROOT / "tools/config")]

import numpy as np

from verity.circuits.codec import decode_program, descriptor_id, encode_program, program_digest
from verity.circuits.defs import REGISTRY, bind
from verity.circuits.evaluation.bits import evaluate_words
from verity.circuits.ops.boolean import cast as BC, gemm as BG
from verity.circuits.refs import runs
from verity.commitments.periods import interface as I
from verity.commitments.periods.population import population_program


def summary(name, prog, query):
    root = prog.fn.body() if callable(getattr(prog.fn, "body", None)) else prog.fn.body
    nodes = root.nodes
    per = Counter()
    edges = Counter()
    for k, nd in enumerate(nodes):
        fid = descriptor_id(nd.fn)
        n = nd.out.n if nd.form == "batch" else 1
        per[fid] += n
        for a in nd.args:
            for s, i, lo, hi in runs(a.refs):
                if s == "n":
                    edges[(descriptor_id(nodes[i].fn), fid)] += hi - lo
    desc = encode_program(prog)
    dig = program_digest(desc)
    dec = decode_program(json.loads(json.dumps(desc)), REGISTRY)
    back = program_digest(dec)
    out = {"name": name, "root": desc["root"], "nodes": len(nodes), "params": len(root.params),
           "param_names_head": [p for p, _ in root.params[:4]], "per_template": dict(per),
           "node_edges": {f"{a} -> {b}": n for (a, b), n in edges.items()}, "digest": dig[:32],
           "roundtrip_digest_equal": dig == back, "query": query.get("name") if isinstance(query, dict) else type(query).__name__,
           "nonslot_params": [p for p, _ in root.params if not p.split(".")[0].isdigit()][:12],
           "annotation_keys": sorted(desc.get("annotations", {}))}
    return out


def both(label, circuit):
    wired, wq = circuit.program()
    unwired, uq = population_program(list(circuit.templates), several=len(circuit.templates) > 1,
                                     values=I.program_values(circuit.registered))
    w, u = summary("wired", wired, wq), summary("unwired", unwired, uq)
    b = wired.fn.bindings
    wt = [descriptor_id(t) if not isinstance(t, str) else t for t in b["templates"]]
    by = {descriptor_id(fn): fn for fn, _ in circuit.templates}
    from_root = [(by[t], int(c)) for t, c in zip(wt, b["counts"])]
    again, _ = population_program(from_root, several=len(from_root) > 1, values=I.program_values(circuit.registered))
    w["root_template_order_is_circuit_order"] = wt == [descriptor_id(fn) for fn, _ in circuit.templates]
    w["unwired_from_wired_root_statics_digest_equal"] = program_digest(again) == program_digest(unwired)
    tmpl = Counter({descriptor_id(fn): n for fn, n in circuit.templates})
    print(json.dumps({"circuit": label, "templates": len(tmpl), "units": sum(tmpl.values()),
                      "wired_counts_match": Counter(w["per_template"]) == tmpl,
                      "unwired_counts_match": Counter(u["per_template"]) == tmpl,
                      "wired": w, "unwired": u}, indent=1, default=str))
    sys.stdout.flush()


ROUND, WIDEN = bind(BG.F2fpBf16), bind(BC.Bf16ToF32)
n = 6
wire = I.Wire(descriptor_id(WIDEN), "a", descriptor_id(ROUND), "o0", at=0, radix=(n,), base=0, step=(1,))
both("periods rounded pair", I.Circuit("bits", ((ROUND, n), (WIDEN, n)), wires=(wire,)))

if "--class" in sys.argv:
    from tests.program.predict import test_window_circuit_fa3 as F
    from verity_vllm.program.predict.window_circuit import PearlTables, WindowClass, class_circuit
    reg = F._registered(config=json.loads(F.LLAMA_1B.read_text()))
    both("llama-3.2-1b sm_90 4,27,2 tied", class_circuit(reg, WindowClass(4, 27, 2), tables=PearlTables(F.PEARL)))
