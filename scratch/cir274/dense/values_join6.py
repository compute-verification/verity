"""CIR-274 DENSE join trial: EMBED's values_run_join.py with the select's family (`node_window_select`) as the sixth set,
the chunks anchored by `greedy_stages.twin` as DENSE's select_run.py does, and a plant on a select unit beside the read's.
EMBED's docstring follows. REQBLOCK's values_run_mm.py with the embedding shard's family added and a plant on its read: the prefill's units of
column block 1 opened over block 0's table (root, domain id and opening). Driver of the node window's value check (`node_window_values`) on the test's GLM request Program.

Usage (from adapters/vllm): values_run.py OUT [--ids 32] [--layers 1] [--moe] [--seed 274] [--most 1024] [--name NAME]

Builds the window twice (as served, and with ``stand_ins``), refuses unless their digests agree, evaluates the Program on
made-up values, checks every template, plants the read's table swap, and writes $OUT/values_<name>.json."""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, "tests/program/predict")
from test_node_window import glm_request  # noqa: E402

import verity.definitions.numerics.rows  # noqa: E402,F401  (registers the row kernels)
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_embed as NE  # noqa: E402
from verity_vllm.program.predict import node_window_multimem as NM  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict import node_window_select as NSEL  # noqa: E402
from verity.circuits.codec import descriptor_id  # noqa: E402
from verity.definitions.registry import greedy_stages as G  # noqa: E402
import dataclasses  # noqa: E402
import numpy as np  # noqa: E402
from verity_vllm.program.predict import node_window_values as NV  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--ids", type=int, default=32)
ap.add_argument("--layers", type=int, default=1)
ap.add_argument("--moe", action="store_true")
ap.add_argument("--seed", type=int, default=274)
ap.add_argument("--most", type=int, default=NV.SAMPLE)
ap.add_argument("--name", default=None)
ap.add_argument("--no-plants", action="store_true")
a = ap.parse_args()
out = Path(a.out)
name = a.name or f"{'moe' if a.moe else 'dense'}_{a.layers}l_{a.ids}"
T0 = time.perf_counter()


def log(s):
    print(f"[{time.perf_counter() - T0:7.1f}s {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:5.2f} GiB] {s}",
          flush=True)


FAM = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES, **NM.FAMILIES, **NE.FAMILIES, **NSEL.FAMILIES}
rec = {"name": name, "ids": a.ids, "layers": a.layers, "moe": a.moe, "outputs": 2, "seed": a.seed, "most": a.most,
       "families": sorted(FAM), "families_by_set": {"frame": sorted(NW.FAMILIES), "dense": sorted(ND.FAMILIES),
                                                     "sparse": sorted(NS.FAMILIES), "multimem": sorted(NM.FAMILIES),
                                                     "embed": sorted(NE.FAMILIES),
                                                     "select": sorted(NSEL.FAMILIES)}}
p = glm_request(a.ids, a.layers, moe=a.moe)
fn = p.program.fn
rec["program_digest"] = p.digest
log(f"program {p.digest[:16]}: {len(fn.body.nodes)} nodes")
c0, w0 = NW.node_window(fn, families=FAM)
c, w = NW.node_window(fn, families=FAM, stand_ins=True)
d0, d1 = w0.census(c0)["digest"], w.census(c)["digest"]
rec["window_digest"], rec["stand_in_window_digest"] = d0, d1
if d0 != d1:
    raise SystemExit(f"the stand-in window's digest {d1} is not the served window's {d0}")
rec["templates"] = len(c.ids)
rec["units"] = sum(n for _f, n in c.templates)
rec["stood_ports"] = len(w.stood)
rec["unlowered"] = {k: len(v) for k, v in w.unlowered.items()}
del c0, w0
log(f"window {d0[:16]}: {rec['templates']} templates, {rec['units']} units, {len(w.stood)} stood ports")
rec["native"] = NV.native_selftest()
inputs = NV.made_up(fn, [(7 * k + 1) % 256 for k in range(a.ids)], np.random.default_rng(a.seed))
log(f"made up: {sum(x.nbytes for x in inputs.values()) / 2**30:.2f} GiB of inputs")
values = NV.Values(fn, inputs, w, log).evaluate()
rec["program_seconds_by_family"] = {k: round(v, 1) for k, v in sorted(values.times.items(), key=lambda kv: -kv[1])}
rec["native_members"] = dict(values.native)
log("program evaluated")
tables = NV.Tables(c, values, w, p.digest)
V = 154880
sel_t = [t for t in c.ids if t.startswith(("GreedyChunk_v2{", "GreedyCombine_v2{"))]
layout = {descriptor_id(u.defn): u for u in G.plan(V)}
twins, tokens = {}, []
for j, node in enumerate(fn.body.nodes):
    if NW._family(node) != "TokenSelect":
        continue
    for m in range(w.members(node)):
        x = np.asarray(values.leaves(values.arg(node, 0, m)), np.uint16)
        token, words = G.twin(V, x)
        v4 = values.words(j)[m]
        tokens.append({"node": j, "member": m, "twin": token, "v4": int(v4[0]) | int(v4[1]) << 16})
        for tid in sel_t:
            u = layout[tid]
            if u.key == ("combine",):
                continue
            unit = next(at_ for at_, _n, _f in w.allocs[tid] if w.node_of[(tid, at_)] == j) + m
            twins[(tid, "o0", unit)] = words[u.key][0]
rec["select_tokens"] = tokens
log(f"select tokens (twin, v4): {[(t['twin'], t['v4']) for t in tokens]}")


class SelCheck(NV.Check):
    def want(self, tid, out, u):
        got = super().want(tid, out, u)
        return twins.get((tid, out, u)) if got is None else got

    def anchored(self, tid):
        return super().anchored(tid) or any(k[0] == tid for k in twins)


ck = SelCheck(c, w, values, tables)
res = NV.run(ck, a.seed, log, a.most)
rec["per_template"] = res
rec["totals"] = {"checked": sum(r.get("checked", 0) for r in res.values()),
                 "mismatches": sum(r["mismatches"] for r in res.values()),
                 "templates_with_mismatches": sorted(t for t, r in res.items() if r["mismatches"]),
                 "units_with_undeclared_inputs": len(ck.unknown),
                 "undeclared_examples": [[t, u, ps] for (t, u), ps in list(ck.unknown.items())[:20]],
                 "rows_two_pieces_hold": len(ck.conflicts), "rows_two_pieces_hold_examples": ck.conflicts[:20],
                 "tables_committed": len(tables.held)}
log(f"check: {rec['totals']}")
mm = sorted({t for t, al in w.allocs.items() if any(f == NM.FAMILY for _x, _n, f in al)})
site_gathers = sorted({t for t, al in w.allocs.items() for at, _n, f in al
                       if f == NW.GATHER and NW._family(w.fn.body.nodes[w.node_of[(t, at)]]) == NM.FAMILY})
rec["new_templates"] = {t: {k: res[t].get(k) for k in ("units", "sampled", "checked", "checked_units", "mismatches",
                                                       "evaluated", "evaluated_of_sampled", "through", "first", "sources",
                                                       "fed_from_program")} for t in mm + site_gathers}
rec["unchecked_unevaluated"] = sorted(t for t, r in res.items() if not r.get("checked") and not r.get("evaluated"))
for t, r in rec["new_templates"].items():
    log(f"new {t[:60]}: units {r['units']} checked {r['checked']} mismatches {r['mismatches']} evaluated {r['evaluated']}")
log(f"unchecked and unevaluated: {rec['unchecked_unevaluated']}")
(out / f"values_{name}.json").write_text(json.dumps(rec, indent=1, default=str))
read = next(t for t in c.units if t.startswith("EmbeddingRead_v1"))
rr = res[read]
rec["read"] = {"template": read, "units": rr["units"], "sampled": rr["sampled"], "checked": rr.get("checked"),
               "mismatches": rr["mismatches"], "first": rr["first"], "sources": rr["sources"],
               "vocab_tables_committed": sorted(n for n in tables.held if n.endswith("/vocab"))}
log(f"read: {rec['read']}")
if not a.no_plants:
    reg = c.registered
    b0, b1 = NE.shard_param("model.embed_tokens.weight", NE.WIDTH, 0), NE.shard_param("model.embed_tokens.weight", NE.WIDTH, 1)
    (o1,) = [o for o in reg.openings if o.reader == read and o.table == b1 and o.units == a.ids]
    hit = lambda x: x.reader == read and (x.at, x.units) == (o1.at, o1.units)  # noqa: E731
    c2 = dataclasses.replace(c, registered=dataclasses.replace(
        reg, roots=tuple(dataclasses.replace(x, table=b0) if hit(x) else x for x in reg.roots),
        openings=tuple(dataclasses.replace(x, table=b0) if hit(x) else x for x in reg.openings)))
    key = NV.sample_key(a.seed)
    picked = {t: NV.sample(ck.counts[t], key, t, a.most) for t in ck.c.ids}
    covered = sum(1 for u in picked[read] if o1.at <= u < o1.at + o1.units)
    ck2 = SelCheck(c2, w, values, tables)
    r2 = NV.run(ck2, a.seed, log, a.most, only=[read])[read]
    rec["plants"] = {"read_block1_over_block0": {
        "planted": {"units": [o1.at, o1.at + o1.units], "from": b1, "to": b0,
                    "roots_moved": sum(1 for x in reg.roots if hit(x)), "openings_moved": sum(1 for x in reg.openings if hit(x))},
        "sampled_planted_units": covered, "checked": r2.get("checked"), "mismatches": r2["mismatches"],
        "mismatches_with_undeclared_inputs": r2.get("mismatches_with_undeclared_inputs"), "first": r2["first"]}}
    log(f"plant: {rec['plants']}")
    chunk0 = descriptor_id(G.plan(V)[0].defn)
    j0 = tokens[0]["node"]
    unit0 = next(at_ for at_, _n, _f in w.allocs[chunk0] if w.node_of[(chunk0, at_)] == j0)
    lane = int(twins[(chunk0, "o0", unit0)][1])
    row = lane // (7744 // c.cuts[chunk0]["logits"])
    for i, x in enumerate(c.wires):
        if (x.reader == chunk0 and x.port == "logits" and x.cut[0] <= row < x.cut[0] + x.cut[1]
                and x.at <= unit0 < x.at + x.reader_units):
            ws = list(c.wires)
            ws[i] = dataclasses.replace(x, base=x.base + 1)
            ck3 = SelCheck(dataclasses.replace(c, wires=tuple(ws)), w, values, tables)
            r3 = NV.run(ck3, a.seed, log, a.most, only=[chunk0])[chunk0]
            rec["plants"]["select_chunk_logits_row"] = {
                "planted": {"wire": i, "reader": chunk0, "unit": unit0, "row": row, "best_lane": lane, "writer": x.writer,
                            "base": [x.base, x.base + 1]},
                "checked": r3.get("checked"), "mismatches": r3["mismatches"], "first": r3["first"]}
            break
    else:
        rec["plants"]["select_chunk_logits_row"] = f"no wire into {chunk0} logits row {row}"
    log(f"plant select: {rec['plants']['select_chunk_logits_row']}")
rec["wall_s"] = round(time.perf_counter() - T0, 1)
rec["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
(out / f"values_{name}.json").write_text(json.dumps(rec, indent=1, default=str))
log(f"wrote values_{name}.json")
