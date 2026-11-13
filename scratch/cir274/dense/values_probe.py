"""Probe for the value check: the 32-id test Programs' weights fields, nodes and their kernels, the window's free ports, tables
and the templates' kernels. Usage: values_probe.py OUT"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "tests/program/predict")
from test_node_window import glm_request  # noqa: E402

import verity.definitions.numerics.rows  # noqa: E402,F401  (registers the row kernels)
from verity.circuits.evaluation.batch import _pick  # noqa: E402
from verity.circuits.evaluation.reference import resolve  # noqa: E402
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402

out = Path(sys.argv[1])
FAM = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}


def kern(fn):
    try:
        k = _pick(resolve(fn), "auto")
        return None if k is None else k.name
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}: {e}"[:200]


rec = {}
for moe in (False, True):
    t0 = time.perf_counter()
    p = glm_request(32, 1, moe=moe)
    fn = p.program.fn
    params = fn.signature()[0]
    d = {"digest": p.digest, "params": [(n, str(type(t).__name__), t.leaves, sorted(t.width_set())) for n, t in params
                                        if n != "weights"]}
    wt = dict(params)["weights"]
    d["weights"] = [(label, t.leaves, sorted(t.width_set()), str(t)[:120]) for label, t in wt.fields]
    d["nodes"] = [(j, NW._family(n), n.form, n.n, list(n.axes), n.fn.id[:160], kern(n.fn)) for j, n in enumerate(fn.body.nodes)]
    c, w = NW.node_window(fn, families=FAM)
    d["templates"] = {tid: (n, kern(f)) for (f, n), tid in zip(c.templates, c.ids)}
    d["free"] = sorted(Counter((f.tid[:60], f.port, f.kind, f.family, f.rows) for f in w.free).items(), key=str)
    d["free"] = [[list(k), v] for k, v in d["free"]]
    reg = c.registered
    d["tables"] = [t.doc() for t in reg.tables][:20]
    d["n_tables"] = len(reg.tables)
    d["roots"] = [r.doc() for r in reg.roots][:6]
    d["openings"] = [o.doc() for o in reg.openings][:6]
    d["reads"] = len(reg.reads)
    d["params_reg"] = list(reg.params.items())[:30]
    d["ingress"] = len(c.ingress)
    d["census"] = {k: v for k, v in w.census(c).items() if k != "peers"}
    bg = [f for f, _n in c.templates if f.id.startswith("BatchedGemm")]
    d["batched_gemm"] = [(f.id, [(n, str(t)) for n, t in f.signature()[0]]) for f in bg]
    nbg = [n for n in fn.body.nodes if NW._family(n) == "BatchedGemm"]
    d["batched_gemm_node"] = [(n.fn.id, [(nm, str(t)[:200]) for nm, t in n.fn.signature()[0]], list(n.axes), n.n) for n in nbg[:1]]
    d["wall_s"] = round(time.perf_counter() - t0, 1)
    rec["moe" if moe else "dense"] = d
    print("PROBE", "moe" if moe else "dense", d["wall_s"], flush=True)
(out / "probe.json").write_text(json.dumps(rec, indent=1, default=str))
print("done")
