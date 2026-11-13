"""CIR-274 DSAMOE: the window's refusal past 86 prompt ids (`interface.ordered`'s "cycle"), at 4 layers, rank 0, eager, the
census's Build (`benchmarks/compile_cost/run.py` `registered`).

Two parts, by argument:

    .venv/bin/python p3_cycle.py OUT bisect LO HI OUTS...   # the least prompt length in (LO, HI] that `ordered` refuses
    .venv/bin/python p3_cycle.py OUT analyze LP OUTS        # that refusal's ranges, a cycle, and its edges held to the units

`analyze` builds the window to `circuit()`, keeps what `ordered` was handed, and rebuilds its range graph: the ranges no
order reaches (by template, family and node), one cycle among them with each edge's wire, and each edge between stuck
ranges tested against the writer units the wire really reads (`Wire.writers`, every row of its cut), not its box
(`Wire.box`). Then Kahn's order again over the stuck ranges with only the real edges: if it orders them, the cycle is the
boxes', not the units'. Writes OUT/cycle_bisect.json or OUT/cycle_<LP>_<OUTS>.json, one DSAMOE line each."""
import bisect
import heapq
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
mode = sys.argv[2]
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
seen: dict = {}
orig_ordered = IF.ordered


def keep(units, wires, starts):
    seen.update(units=dict(units), wires=list(wires), starts={k: list(v) for k, v in starts.items()})
    return orig_ordered(units, wires, starts)


IF.ordered = keep


def build(LP: int, OUTS: int):
    seen.clear()
    advice, ingress = cc.session([(0, LP, OUTS)])
    u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
    p = request_program(reg, u)
    w = NW.NodeWindow(p.program.fn, families, NW._GRAIN, NW.NAME, 1, True).lower()
    t = time.perf_counter()
    try:
        c = w.circuit()
        return p, w, c, None, round(time.perf_counter() - t, 1)
    except ValueError as e:
        if not str(e).startswith("cycle"):
            raise
        return p, w, None, str(e), round(time.perf_counter() - t, 1)


def short(tid: str) -> str:
    return tid if len(tid) < 90 else tid[:87] + "..."


if mode == "bisect":
    lo0, hi0 = int(sys.argv[3]), int(sys.argv[4])
    rec: dict = {"probes": []}
    for OUTS in map(int, sys.argv[5:]):
        lo, hi = lo0, hi0       # lo builds, hi refuses (both checked)
        for LP in (lo, hi):
            p, w, c, err, s = build(LP, OUTS)
            rec["probes"].append({"LP": LP, "outs": OUTS, "ok": err is None, "error": err and err[:400], "program": p.digest})
            print("DSAMOE cycle probe", LP, OUTS, "ok" if err is None else "refused", flush=True)
        if not (rec["probes"][-2]["ok"] and not rec["probes"][-1]["ok"]):
            rec[f"least_{OUTS}"] = "the ends don't bracket a refusal"
            continue
        while hi - lo > 1:
            mid = (lo + hi) // 2
            p, w, c, err, s = build(mid, OUTS)
            rec["probes"].append({"LP": mid, "outs": OUTS, "ok": err is None, "error": err and err[:400], "program": p.digest})
            print("DSAMOE cycle probe", mid, OUTS, "ok" if err is None else "refused", flush=True)
            lo, hi = (mid, hi) if err is None else (lo, mid)
        rec[f"least_{OUTS}"] = hi
        (out / "cycle_bisect.json").write_text(json.dumps(rec, indent=1) + "\n")
    (out / "cycle_bisect.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("DSAMOE cycle bisect", json.dumps({k: v for k, v in rec.items() if k != "probes"}), flush=True)
    sys.exit(0)

LP, OUTS = int(sys.argv[3]), int(sys.argv[4])
p, w, c, err, s = build(LP, OUTS)
rec = {"LP": LP, "outs": OUTS, "program": p.digest, "error": err, "steps": w.steps}
units, wires, starts = seen["units"], seen["wires"], seen["starts"]
cut = {t: sorted({0, *(x for x in starts.get(t, ()) if 0 < x < n)}) for t, n in units.items()}
nodes = sorted((t, k) for t, ss in cut.items() for k in range(len(ss)))
idx = {x: i for i, x in enumerate(nodes)}
end = {t: [*ss[1:], units[t]] for t, ss in cut.items()}
edges = []      # (writer range, reader range, wire)
for i, x in enumerate(wires):
    ss = cut[x.reader]
    for k in range(bisect.bisect_right(ss, x.at) - 1, bisect.bisect_left(ss, x.at + x.reader_units)):
        lo, hi = (max(v, 0) for v in x.box(*x.indices_in(ss[k], end[x.reader][k] - ss[k])))
        for j in range(bisect.bisect_right(cut[x.writer], lo) - 1, bisect.bisect_right(cut[x.writer], hi)):
            edges.append((idx[(x.writer, j)], idx[(x.reader, k)], i))


def kahn(n_nodes, es, among=None):
    among = set(range(n_nodes)) if among is None else among
    indeg, outs = defaultdict(int), defaultdict(list)
    for a, b, _i in es:
        if a in among and b in among:
            indeg[b] += 1
            outs[a].append(b)
    ready = [v for v in among if indeg[v] == 0]
    heapq.heapify(ready)
    done = set()
    while ready:
        v = heapq.heappop(ready)
        done.add(v)
        for u_ in outs[v]:
            indeg[u_] -= 1
            if indeg[u_] == 0:
                heapq.heappush(ready, u_)
    return among - done


def a_cycle(es, among):
    preds = defaultdict(list)
    for a, b, i in es:
        if a in among and b in among:
            preds[b].append((a, i))
    v, path, pos = next(iter(sorted(among))), [], {}
    while v not in pos:
        pos[v] = len(path)
        a, i = preds[v][0]
        path.append((v, a, i))
        v = a
    return path[pos[v]:]


def node_doc(v):
    t, k = nodes[v]
    at = cut[t][k]
    j = w.node_of.get((t, at))
    span = next((s for s in w.b.spans.get(t, []) if s[0] == at), None)
    fn = w.fn.body.nodes[j].fn.id if j is not None and hasattr(w, "fn") else None
    return {"template": short(t), "at": at, "end": end[t][k], "family": w.family_of(t, at), "node": j,
            "node_fn": fn, "span": span}


stuck = kahn(len(nodes), edges)
rec["ranges"] = len(nodes)
rec["stuck"] = len(stuck)
rec["stuck_by_template"] = dict(Counter(f"{w.family_of(nodes[v][0], cut[nodes[v][0]][nodes[v][1]])} | {short(nodes[v][0])}"
                                        for v in stuck).most_common())
rec["stuck_by_step"] = dict(Counter(str(node_doc(v)["span"] and node_doc(v)["span"][3]) for v in stuck))
rec["stuck_nodes"] = dict(Counter(str(node_doc(v)["node"]) for v in stuck).most_common(40))

cyc = a_cycle(edges, stuck)
rec["cycle"] = [{"reader": node_doc(v), "writer": node_doc(a), "wire": wires[i].doc(),
                 "wire_reader_family": w.family_of(wires[i].reader, wires[i].at)} for v, a, i in cyc]


def real(a, b, i) -> bool:
    x = wires[i]
    t, k = nodes[b]
    lo, hi = max(cut[t][k], x.at), min(end[t][k], x.at + x.reader_units)
    if hi <= lo:
        return False
    wt, wj = nodes[a]
    wlo, whi = cut[wt][wj], end[wt][wj]
    for r in range(x.cut[0], x.cut[0] + x.cut[1]):
        ws = x.writers(lo, hi - lo, r)
        if np.any((ws >= wlo) & (ws < whi)):
            return True
    return False


st_edges = [e for e in edges if e[0] in stuck and e[1] in stuck]
real_edges = [e for e in st_edges if real(*e)]
rec["stuck_edges"] = len(st_edges)
rec["stuck_edges_real"] = len(real_edges)
spurious = [e for e in st_edges if e not in set(real_edges)]
rec["spurious_examples"] = [{"reader": node_doc(b), "writer": node_doc(a), "wire": wires[i].doc()} for a, b, i in spurious[:6]]
still = kahn(len(nodes), real_edges, stuck)
rec["stuck_after_real_edges"] = len(still)
if still:
    rec["real_cycle"] = [{"reader": node_doc(v), "writer": node_doc(a), "wire": wires[i].doc()}
                         for v, a, i in a_cycle(real_edges, still)]
(out / f"cycle_{LP}_{OUTS}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
print("DSAMOE cycle analyze", json.dumps({k: rec[k] for k in ("LP", "outs", "ranges", "stuck", "stuck_edges",
      "stuck_edges_real", "stuck_after_real_edges")}), flush=True)
print("DSAMOE cycle by template", json.dumps(rec["stuck_by_template"])[:3000], flush=True)
