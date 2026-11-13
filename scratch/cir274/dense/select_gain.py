"""CIR-274 DENSE piece 5: what lowering the selects adds to the window, beside the selects' own 20 templates.

Usage (from adapters/vllm): select_gain.py OUT [--ids 43] [--layers 4]

Builds rank 0's window with the frame's, the dense and the sparse families, then with the select's too, and records each
one's digest and, by template and by family, the units, templates and wires the select adds outside its own stage templates
(the LM head's side: gathers made whole, first-level gather chunks); and how many of the select's rank-0 logits rows each
writer template feeds. Writes $OUT/select_gain.json."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "tests/program/predict")
from test_node_window import glm_request  # noqa: E402

from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_select as NSEL  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--ids", type=int, default=43)
ap.add_argument("--layers", type=int, default=4)
a = ap.parse_args()
fn = glm_request(a.ids, a.layers).program.fn
base = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}
rec: dict = {"ids": a.ids, "layers": a.layers, "rank": 0}
got = {}
for key, fams in (("without", base), ("with", {**base, **NSEL.FAMILIES})):
    c, w = NW.node_window(fn, families=fams)
    cen = w.census(c)
    got[key] = (c, w)
    rec[key] = {"digest": cen["digest"], "templates": len(c.ids), "units": sum(c.units.values()),
                "wire_patterns": len(c.wires), "unlowered": cen["unlowered"], "units_by_family": cen["units"],
                "templates_by_family": cen["templates"], "free": cen["free"]}
(c0, _w0), (c1, w1) = got["without"], got["with"]
sel = {t for t in c1.ids if t.startswith(("GreedyChunk_v2{", "GreedyCombine_v2{"))}
rec["select_templates"] = len(sel)
rec["select_units"] = sum(c1.units[t] for t in sel)
rec["new_templates_outside_select"] = {t: c1.units[t] for t in c1.ids if t not in c0.units and t not in sel}
rec["more_units_outside_select"] = {t: [c0.units.get(t, 0), c1.units[t]] for t in c1.ids
                                    if t not in sel and c1.units[t] != c0.units.get(t, 0)}
rec["units_by_family_delta"] = {f: rec["with"]["units_by_family"].get(f, 0) - rec["without"]["units_by_family"].get(f, 0)
                                for f in sorted(set(rec["with"]["units_by_family"]) | set(rec["without"]["units_by_family"]))
                                if rec["with"]["units_by_family"].get(f, 0) != rec["without"]["units_by_family"].get(f, 0)}
rec["logits_rows_by_writer"] = dict(Counter({x.writer: 0 for x in c1.wires if x.reader in sel and x.port == "logits"}))
for x in c1.wires:
    if x.reader in sel and x.port == "logits":
        rec["logits_rows_by_writer"][x.writer] += x.units
rec["logits_cuts"] = sorted({json.dumps(c1.cuts.get(t)) for t in sel})
rec["select_free_by_peer"] = dict(sorted(Counter(f.name for f in w1.free if f.tid in sel).items()))
Path(a.out, "select_gain.json").write_text(json.dumps(rec, indent=1, default=str))
print(json.dumps({k: v for k, v in rec.items() if k not in ("with", "without")}, indent=1, default=str))
print("digests", rec["without"]["digest"][:16], rec["with"]["digest"][:16])
