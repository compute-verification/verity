"""CIR-274 DSAMOE: a prototype, in scratch only, of w2's routed weight ``g`` read through slot-major copies, a form that
stays inside `node_window_sparse.py` (the two-digit `Piece` does not). Prefill (T > 1): a copy unit
`AllGather_v1{WORLD=1, N=c}` per pair, unit ``k T + t`` reading token t's slot-k weight row (`_slots`): one wire a slot;
the block GEMMs' ``g`` then reads the copies at digits (T, 1), (TOPK, T), (B, 0): one wire. Decode (T = 1) stays as it is.
Installed over `expert_gemm` by monkeypatch, so the tree stays as it is. Run from the root of the tree:

    .venv/bin/python p3_gcopy.py OUT structure OUTS:LP...     # before and after: builds, digests, g wires, wires
    .venv/bin/python p3_gcopy.py OUT values LAYERS LP          # node_window_values with the copies, 2 outputs

Writes OUT/gcopy_structure.json or OUT/gcopy_values_l<LAYERS>_p<LP>.json."""
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

import verity.definitions.numerics.rows  # noqa: E402,F401  (registers the row kernels)
from verity.circuits.defs import bind  # noqa: E402
from verity.commitments.windows import interface as IF  # noqa: E402
from verity.definitions.registry import b1_tp2 as B1T  # noqa: E402
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402
from verity_vllm.program.predict.window_builder import Row  # noqa: E402

ORIG = NS.expert_gemm
COPIES = {"units": 0, "nodes": 0}


def _g_copies(w, ig, node, M):
    """The slot-major copies of ``g`` and the GEMMs' read of them, or None (then the original per-member wires)."""
    g = w.take(node, ig, whole=True)
    if isinstance(g, NS._Free):
        return None
    try:
        slots, step = NS._slots(g, M)
    except NS.Unexpressible:
        return None
    TOPK = len(slots) or 1
    T = M // TOPK
    if T == 1 or any(len(g[m]) != 1 for m in range(M)):
        return None
    c = slots[0].words
    ct, ca = w.unit(NW.GATHER, bind(B1T.AllGather, WORLD=1, N=c), M)
    for k, r in enumerate(slots):
        w.b.feed(ct, "parts", ca + k * T, T, (r,), ((T, step),))
    COPIES["units"] += M
    COPIES["nodes"] += 1
    return Row(ct, ca, "o0", c), T, TOPK


def expert_gemm_copy(w, j, node):
    """`expert_gemm`, with its ``g`` feed through `_g_copies` in prefill."""
    ie, ig, iw, iws = NS.XR.ARGS[NS.spec_base(NS.descriptor_id(node.fn))]
    if ig is None:
        return ORIG(w, j, node)
    M = w.members(node)
    feed = w.b.feed
    state = {}

    def b_feed(tid, port, at, count, rows, digits):
        if port == "g" and "copy" in state:
            r, T, TOPK = state["copy"]
            if at == state["ag"]:
                feed(tid, "g", at, T * TOPK * state["B"], (r,), ((T, 1), (TOPK, T), (state["B"], 0)))
            return None
        return feed(tid, port, at, count, rows, digits)

    unit = w.unit

    def w_unit(fam, fn, n):
        tid, at = unit(fam, fn, n)
        if NS.spec_base(NS.descriptor_id(fn)) == NS.ST.EXPERTS[NS.spec_base(NS.descriptor_id(node.fn))]:
            state["ag"], state["B"] = at, n // M
            got = _g_copies(w, ig, node, M)
            if got is not None:
                state["copy"] = got
        return tid, at

    w.b.feed, w.unit = b_feed, w_unit
    try:
        return ORIG(w, j, node)
    finally:
        w.b.feed, w.unit = feed, unit


FAM0 = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}
FAM1 = {**FAM0, "MoeExpertFp8GemmW": expert_gemm_copy}


def program(layers, LP, OUTS):
    reg = cc.registered(0, layers, "eager")
    advice, ingress = cc.session([(0, LP, OUTS)])
    u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
    return request_program(reg, u)


def g_wires(c):
    return sum(1 for x in c.wires if x.reader.startswith("MoeExpertFp8BlockGemmW") and x.port == "g")


if sys.argv[2] == "structure":
    rec = {}
    for arg in sys.argv[3:]:
        OUTS, LP = map(int, arg.split(":"))
        fn = program(4, LP, OUTS).program.fn
        r = {}
        for name, fam in (("before", FAM0), ("after", FAM1)):
            COPIES.update(units=0, nodes=0)
            t0 = time.perf_counter()
            try:
                c, w = NW.node_window(fn, families=fam)
                IF.check_wires(c.layouts, c.wires, c.cuts, c.order)
                IF.check_inputs(c.layouts, c.cuts, c.wires, c.registered, c.ingress)
                blob = IF.circuit_bytes(c.description())
                r[name] = {"ok": True, "digest": hashlib.sha512(blob).hexdigest(), "bytes": len(blob),
                           "wires": len(c.wires), "g_wires": g_wires(c), "units": sum(n for _f, n in c.templates),
                           "copy_units": COPIES["units"], "copy_nodes": COPIES["nodes"]}
            except Exception as e:  # noqa: BLE001 - a refusal is a result
                r[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:1500]}
            r[name]["wall_s"] = round(time.perf_counter() - t0, 1)
        rec[arg] = r
        (out / "gcopy_structure.json").write_text(json.dumps(rec, indent=1) + "\n")
        print("DSAMOE gcopy", arg, json.dumps(r), flush=True)
else:
    from verity_vllm.program.predict import node_window_values as NV
    layers, LP = int(sys.argv[3]), int(sys.argv[4])
    T0 = time.perf_counter()

    def log(s):
        print(f"[{time.perf_counter() - T0:7.1f}s {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:5.2f} GiB] {s}",
              flush=True)

    p = program(layers, LP, 2)
    fn = p.program.fn
    c0, w0 = NW.node_window(fn, families=FAM1)
    c, w = NW.node_window(fn, families=FAM1, stand_ins=True)
    rec = {"layers": layers, "ids": LP, "outputs": 2, "program_digest": p.digest, "copy_units": COPIES["units"],
           "description_digest": hashlib.sha512(IF.circuit_bytes(c0.description())).hexdigest(), "g_wires": g_wires(c0)}
    d0, d1 = w0.census(c0)["digest"], w.census(c)["digest"]
    if d0 != d1:
        raise SystemExit(f"the stand-in window's digest {d1} is not the served window's {d0}")
    del c0, w0
    inputs = NV.made_up(fn, [(7 * k + 1) % 256 for k in range(LP)], 274)
    values = NV.Values(fn, inputs, w, log).evaluate()
    tables = NV.Tables(c, values, w, p.digest)
    ck = NV.Check(c, w, values, tables)
    res = NV.run(ck, 274, log, NV.SAMPLE)
    rec["per_template"] = res
    rec["totals"] = {"templates_checked": len(res), "checked": sum(r.get("checked", 0) for r in res.values()),
                     "mismatches": sum(r["mismatches"] for r in res.values()),
                     "templates_with_mismatches": sorted(t for t, r in res.items() if r["mismatches"]),
                     "units_with_undeclared_inputs": len(ck.unknown), "rows_two_pieces_hold": len(ck.conflicts)}
    rec["gather_templates_checked"] = {t: r for t, r in res.items() if t.startswith("AllGather_v1{WORLD=1")}
    (out / f"gcopy_values_l{layers}_p{LP}.json").write_text(json.dumps(rec, indent=1, default=str))
    log(f"DSAMOE gcopy values {json.dumps(rec['totals'])}")
