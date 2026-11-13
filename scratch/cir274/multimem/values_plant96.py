"""REQBLOCK piece D: the planted wire aimed at AllReduceMultimem_v1{SIZE=96}; only that template is checked.
From DENSE's driver (r20261113-035817-256b's script.py) with the multimem family added and the planted wire into
AllReduceMultimem's ``parts``. Driver of the node window's value check (`node_window_values`) on the test's GLM request Program.

Usage (from adapters/vllm): values_run.py OUT [--ids 32] [--layers 1] [--moe] [--seed 274] [--most 1024] [--name NAME]

Builds the window twice (as served, and with ``stand_ins``), refuses unless their digests agree, evaluates the Program on
made-up values, checks every template, plants (a) rank 0's wire on row 3 of an AllReduceResidualMnnvl ``parts`` and (b) one
registered read moved by one row of its own width, and writes $OUT/values_<name>.json."""
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
from verity_vllm.program.predict import node_window_multimem as NM  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
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


FAM = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES, **NM.FAMILIES}
rec = {"name": name, "ids": a.ids, "layers": a.layers, "moe": a.moe, "outputs": 2, "seed": a.seed, "most": a.most,
       "families": sorted(FAM), "families_by_set": {"frame": sorted(NW.FAMILIES), "dense": sorted(ND.FAMILIES),
                                                     "sparse": sorted(NS.FAMILIES), "multimem": sorted(NM.FAMILIES)}}
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
inputs = NV.made_up(fn, [(7 * k + 1) % 256 for k in range(a.ids)], a.seed)
log(f"made up: {sum(x.nbytes for x in inputs.values()) / 2**30:.2f} GiB of inputs")
values = NV.Values(fn, inputs, w, log).evaluate()
rec["program_seconds_by_family"] = {k: round(v, 1) for k, v in sorted(values.times.items(), key=lambda kv: -kv[1])}
rec["native_members"] = dict(values.native)
log("program evaluated")
tables = NV.Tables(c, values, w, p.digest)
ck = NV.Check(c, w, values, tables)
import numpy as np
t96 = [t for t in c.ids if t.startswith(NM.FAMILY + "_v1{") and ("{SIZE=96}" in t or "{SIZE=96," in t or ",SIZE=96," in t or ",SIZE=96}" in t)]
rec["size96_templates"] = t96
log(f"SIZE=96 templates: {t96}")
if len(t96) != 1:
    raise SystemExit(f"expected one SIZE=96 multimem template, found {t96}")
tid96 = t96[0]
base = NV.run(ck, a.seed, log, a.most, only=[tid96])[tid96]
rec["base"] = {k: base.get(k) for k in ("units", "sampled", "checked", "checked_units", "mismatches", "first", "sources",
                                        "mismatches_with_undeclared_inputs")}
log(f"base {tid96}: {base.get('checked')} checked, {base['mismatches']} mismatches")
c2, tid, what = NV.plant_wire_row(c, tid96, "parts", 3, keep=True)
assert tid == tid96, (tid, tid96)
x = c2.wires[what["wire"]]
rng = np.random.default_rng(a.seed)
picked = {t: NV.sample(ck.counts[t], rng, a.most) for t in ck.c.ids}
what |= {"writer": x.writer, "out": x.out, "at": x.at, "reader_units": x.reader_units, "sampled_of_reader": len(picked[tid]),
         "sampled_units_covered": sum(1 for u in picked[tid] if x.at <= u < x.at + x.reader_units)}
ck2 = NV.Check(c2, w, values, tables)
r2 = NV.run(ck2, a.seed, log, a.most, only=[tid])[tid]
rec["plant"] = {"key": "mm96_wire_row3_kept_row0", "planted": what, "template": tid, "units": r2.get("units"),
                "sampled": r2.get("sampled"), "checked": r2.get("checked"), "checked_units": r2.get("checked_units"),
                "mismatches": r2["mismatches"], "mismatches_with_undeclared_inputs": r2.get("mismatches_with_undeclared_inputs"),
                "first": r2["first"], "sources": r2.get("sources")}
log(f"plant: {what}")
log(f"plant {tid}: {r2['mismatches']} mismatches of {r2.get('checked')} checked ({r2.get('sampled')} sampled)")
rec["wall_s"] = round(time.perf_counter() - T0, 1)
rec["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
(out / f"plant96_{name}.json").write_text(json.dumps(rec, indent=1, default=str))
log(f"wrote plant96_{name}.json")
