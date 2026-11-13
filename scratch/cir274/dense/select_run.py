"""CIR-274 DENSE piece 5: the greedy token select lowered into rank 0's node window (`node_window_select`), checked.

Usage (from adapters/vllm): select_run.py OUT [--ids 43] [--layers 4] [--seed 274] [--most 1024] [--tie] [--no-values]

Builds the window with the frame's, the dense, the sparse and the select families; runs `check_proof_block` on every template
under its cuts and `check_wires` on the whole; records the digest, unit and template counts, the unlowered nodes, the free
ports (the select's by peer), and what StopCheck reads. Then the value check (`node_window_values`) on the select's 20
templates: the combine anchored by the Program's own TokenSelect_v4 output, each chunk by `greedy_stages.twin` on the
Program's logits at that select; two plants that must fail, (1) a chunk's logits wire moved to the next writer row on the
row holding that chunk's best lane, (2) the combine's r<w> (w the twin's winning chunk) wired to chunk w+1's result. With
``--tie`` the peers' logits are set to the largest finite bf16 at three lanes (two in chunk 5, one in chunk 12), so the
token must be the first and the units' answer is checked against the row Definition's. Writes $OUT/select_<name>.json."""
import argparse
import dataclasses
import json
import resource
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests/program/predict")
from test_node_window import glm_request  # noqa: E402

import verity.definitions.numerics.rows  # noqa: E402,F401  (registers the row kernels)
from verity.circuits.codec import descriptor_id  # noqa: E402
from verity.circuits.refs import runs  # noqa: E402
from verity.commitments.windows import interface as IF  # noqa: E402
from verity.definitions.registry import greedy_stages as G  # noqa: E402
from verity_vllm.pipeline import serving_templates as ST  # noqa: E402
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_select as NSEL  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict import node_window_values as NV  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--ids", type=int, default=43)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--seed", type=int, default=274)
ap.add_argument("--most", type=int, default=NV.SAMPLE)
ap.add_argument("--tie", action="store_true")
ap.add_argument("--no-values", action="store_true")
ap.add_argument("--name", default=None)
a = ap.parse_args()
out = Path(a.out)
name = a.name or f"{a.layers}l_{a.ids}{'_tie' if a.tie else ''}"
path = out / f"select_{name}.json"
T0 = time.perf_counter()
V = 154880
X = 0x7F7F  # the largest finite bf16
TIE = (5 * 7744 + 100, 5 * 7744 + 7000, 12 * 7744 + 5)


def log(s):
    print(f"[{time.perf_counter() - T0:7.1f}s {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:5.2f} GiB] {s}",
          flush=True)


def save():
    rec["wall_s"] = round(time.perf_counter() - T0, 1)
    rec["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    path.write_text(json.dumps(rec, indent=1, default=str))


FAM = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES, **NSEL.FAMILIES}
rec: dict = {"name": name, "ids": a.ids, "layers": a.layers, "outputs": 2, "rank": 0, "seed": a.seed, "tie": a.tie,
             "families": sorted(FAM)}
p = glm_request(a.ids, a.layers)
fn = p.program.fn
rec["program_digest"] = p.digest
log(f"program {p.digest[:16]}: {len(fn.body.nodes)} nodes")
c0, w0 = NW.node_window(fn, families=FAM)
rec["settle"] = "passed (NodeWindow.circuit: _Builder.settle, then Circuit's check_wires and check_inputs)"
IF.check_wires(c0.layouts, c0.wires, c0.cuts, c0.order)
rec["check_wires"] = "passed"
fns = dict(zip(c0.ids, (f for f, _n in c0.templates)))
bad = {}
for t, f in fns.items():
    try:
        IF.check_proof_block(f, c0.cuts.get(t))
    except Exception as e:  # noqa: BLE001 - a refusal is a result
        bad[t] = f"{type(e).__name__}: {e}"[:400]
rec["check_proof_block"] = {"templates": len(fns), "refused": bad}
cen = w0.census(c0)
rec["window_digest"] = cen["digest"]
rec["templates"] = len(c0.ids)
rec["units"] = sum(c0.units.values())
rec["wire_patterns"] = len(c0.wires)
rec["unlowered"] = cen["unlowered"]
sel_t = [t for t in c0.ids if t.startswith(("GreedyChunk_v2{", "GreedyCombine_v2{"))]
rec["select"] = {"templates": len(sel_t), "units": sum(c0.units[t] for t in sel_t),
                 "by_template": {t: {"units": c0.units[t], "cuts": c0.cuts.get(t),
                                     "wires_in": dict(Counter(x.port for x in c0.wires if x.reader == t)),
                                     "wired_rows": sum(x.units for x in c0.wires if x.reader == t and x.port == "logits"),
                                     "writers": sorted({x.writer for x in c0.wires if x.reader == t})}
                                 for t in sel_t}}
free_sel = [f for f in w0.free if f.tid in set(sel_t)]
rec["select"]["free"] = {"rows": sum(f.count for f in free_sel), "ports": sorted({f.port for f in free_sel}),
                         "by_peer": dict(sorted(Counter(f.name for f in free_sel).items()))}
rec["free_ports_all"] = dict(sorted(Counter(f"{f.tid.split('{')[0]}/{f.port}: {f.kind} {f.fam}" for f in w0.free).items()))
rec["census"] = {k: v for k, v in cen.items() if k not in ("peers", "trace")}
log(f"window {cen['digest'][:16]}: {rec['templates']} templates, {rec['units']} units; {len(bad)} refused proof blocks; "
    f"unlowered {cen['unlowered']}")

sels = [j for j, n in enumerate(fn.body.nodes) if NW._family(n) == "TokenSelect"]
rec["select_nodes"] = [{"node": j, "spec": descriptor_id(fn.body.nodes[j].fn), "members": w0.members(fn.body.nodes[j]),
                        "peers": sorted({w0.params[i] for s, i, _l, _h in
                                         runs(NSEL.through(w0, fn.body.nodes[j].args[0].refs)) if s == "p"})}
                       for j in sels]
stops = []
for j, n in enumerate(fn.body.nodes):
    if NW._family(n) != "StopCheck":
        continue
    spec = descriptor_id(n.fn)
    s: dict = {"node": j, "spec": spec, "form": n.form, "out": str(w0.outs.get(j))[:300]}
    try:
        s["unit_templates"] = [str(t) for t in ST.unit_templates(spec)]
    except Exception as e:  # noqa: BLE001
        s["unit_templates"] = f"{type(e).__name__}: {e}"[:400]
    try:
        lay = IF.layout(n.fn)
        s["layout"] = {"rows": lay.rows, "outs": lay.outs, "outer": lay.outer}
    except Exception as e:  # noqa: BLE001
        s["layout"] = f"{type(e).__name__}: {e}"[:400]
    try:
        IF.check_proof_block(n.fn)
        s["check_proof_block"] = "passed"
    except Exception as e:  # noqa: BLE001
        s["check_proof_block"] = f"{type(e).__name__}: {e}"[:400]
    args = []
    for k, arg in enumerate(n.args):
        segs = [list(x) if x[0] == "p" else x for x in w0.segments(arg.refs)]
        r: dict = {"leaves": len(arg.refs), "segments": [str(x)[:120] for x in segs][:8]}
        try:
            r["rows"] = [(x.tid, x.unit, x.out, x.words) for x in w0.rows(arg.refs)][:8]
        except NW._Free as e:
            r["rows"] = f"free: {e.sources}"[:300]
        except Exception as e:  # noqa: BLE001
            r["rows"] = f"{type(e).__name__}: {e}"[:300]
        args.append(r)
    s["args"] = args
    stops.append(s)
rec["stop_check"] = stops
save()
log("window recorded")
if a.no_values:
    raise SystemExit(0)

c, w = NW.node_window(fn, families=FAM, stand_ins=True)
rec["stand_in_window_digest"] = w.census(c)["digest"]
if rec["stand_in_window_digest"] != rec["window_digest"]:
    save()
    raise SystemExit("the stand-in window's digest is not the served window's")
del c0, w0
inputs = NV.made_up(fn, [(7 * k + 1) % 256 for k in range(a.ids)], a.seed)
if a.tie:
    tied = []
    for s in rec["select_nodes"]:
        node = fn.body.nodes[s["node"]]
        at = 0
        for space, idx, lo, hi in runs(NSEL.through(w, node.args[0].refs)):
            for P in TIE:
                if at <= P < at + hi - lo:
                    if space != "p" or not w.params[idx].startswith("peers"):
                        raise SystemExit(f"tie lane {P} is not a peer's")
                    inputs[w.params[idx]][lo + P - at] = X
                    tied.append((s["node"], P, w.params[idx], lo + P - at))
            at += hi - lo
    rec["tie_lanes"] = tied
values = NV.Values(fn, inputs, w, log).evaluate()
log("program evaluated")
tables = NV.Tables(c, values, w, p.digest)

layout = {descriptor_id(u.defn): u for u in G.plan(V)}
twins, tokens = {}, []
for s in rec["select_nodes"]:
    j = s["node"]
    node = fn.body.nodes[j]
    for m in range(s["members"]):
        x = np.asarray(values.leaves(values.arg(node, 0, m)), np.uint16)
        token, words = G.twin(V, x)
        v4 = values.words(j)[m]
        tokens.append({"node": j, "member": m, "twin": token, "v4_words": [int(v) for v in v4],
                       "v4": int(v4[0]) | int(v4[1]) << 16, "rank0_max": float(np.nanmax((x[:V // 8].astype(np.uint32) << 16).view(np.float32))),
                       "chunk_best": {c_: int(words[("chunk", c_)][0][0]) for c_ in range(19)}})
        for tid in sel_t:
            u = layout[tid]
            if u.key == ("combine",):
                continue
            unit = next(at_ for at_, _n, _f in w.allocs[tid] if w.node_of[(tid, at_)] == j) + m
            twins[(tid, "o0", unit)] = words[u.key][0]
rec["tokens"] = tokens
log(f"tokens: {[(t['twin'], t['v4']) for t in tokens]}")


class SelCheck(NV.Check):
    def want(self, tid, out, u):
        got = super().want(tid, out, u)
        return twins.get((tid, out, u)) if got is None else got

    def anchored(self, tid):
        return super().anchored(tid) or any(k[0] == tid for k in twins)


ck = SelCheck(c, w, values, tables)
res = NV.run(ck, a.seed, log, a.most, only=sel_t)
rec["per_template"] = res
rec["totals"] = {"checked": sum(r.get("checked", 0) for r in res.values()),
                 "mismatches": sum(r["mismatches"] for r in res.values()),
                 "templates_with_mismatches": sorted(t for t, r in res.items() if r["mismatches"]),
                 "units_with_undeclared_inputs": len([k for k in ck.unknown if k[0] in sel_t])}
log(f"check: {rec['totals']}")
save()


def plant(key, c2, tid, what):
    ck2 = SelCheck(c2, w, values, tables)
    r2 = NV.run(ck2, a.seed, log, a.most, only=[tid])[tid]
    rec.setdefault("plants", {})[key] = {"planted": what, "template": tid, "checked": r2.get("checked"),
                                         "mismatches": r2["mismatches"], "first": r2["first"], "sources": r2["sources"]}
    log(f"plant {key}: {r2['mismatches']} mismatches of {r2.get('checked')}")


j0 = rec["select_nodes"][0]["node"]
t0 = tokens[0]
try:
    chunk0 = descriptor_id(G.plan(V)[0].defn)
    unit0 = next(at_ for at_, _n, _f in w.allocs[chunk0] if w.node_of[(chunk0, at_)] == j0)
    lane = int(twins[(chunk0, "o0", unit0)][1])
    R = 7744 // c.cuts[chunk0]["logits"]
    row = lane // R
    for i, x in enumerate(c.wires):
        if x.reader == chunk0 and x.port == "logits" and x.cut[0] == row and x.at <= unit0 < x.at + x.reader_units:
            ws = list(c.wires)
            ws[i] = dataclasses.replace(x, base=x.base + 1)
            plant("1_chunk_logits_row", dataclasses.replace(c, wires=tuple(ws)), chunk0,
                  {"wire": i, "reader": chunk0, "unit": unit0, "row": row, "best_lane": lane, "writer": x.writer,
                   "base": [x.base, x.base + 1]})
            break
    else:
        rec.setdefault("plants", {})["1_chunk_logits_row"] = f"no wire into {chunk0} logits row {row}"
    win = next((cc for cc in range(19) if G.plan(V)[cc].lanes[0] <= t0["twin"] < G.plan(V)[cc].lanes[1]), None)
    if win is None:
        rec.setdefault("plants", {})["2_combine_result"] = f"the token {t0['twin']} is in the combine's own lanes"
    else:
        comb = descriptor_id(G.plan(V)[-1].defn)
        ucomb = next(at_ for at_, _n, _f in w.allocs[comb] if w.node_of[(comb, at_)] == j0)
        other = descriptor_id(G.plan(V)[(win + 1) % 19].defn)
        for i, x in enumerate(c.wires):
            if x.reader == comb and x.port == f"r{win}" and x.at <= ucomb < x.at + x.reader_units:
                ws = list(c.wires)
                ws[i] = dataclasses.replace(x, writer=other)
                plant("2_combine_result", dataclasses.replace(c, wires=tuple(ws)), comb,
                      {"wire": i, "reader": comb, "unit": ucomb, "port": f"r{win}", "writer": [x.writer, other]})
                break
except Exception as e:  # noqa: BLE001
    rec["plant_error"] = traceback.format_exc()[-3000:]
    log(f"plant error {e}")
save()
log(f"wrote {path.name}")
