"""How many of the value check's sampled units of each planted template the planted wire or read covers, at the driver's
seed and sizes. Usage (from adapters/vllm): values_plant_cover.py OUT [--ids 43] [--layers 4] [--moe] [--seed 274]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests/program/predict")
from test_node_window import glm_request  # noqa: E402

from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict import node_window_values as NV  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--ids", type=int, default=43)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--moe", action="store_true")
ap.add_argument("--seed", type=int, default=274)
a = ap.parse_args()
fn = glm_request(a.ids, a.layers, moe=a.moe).program.fn
c, _w = NW.node_window(fn, families={**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES})
counts = dict(zip(c.ids, (n for _f, n in c.templates)))
rng = np.random.default_rng(a.seed)
picked = {tid: NV.sample(counts[tid], rng) for tid in c.ids}
rec = {}
for key, (c2, tid, what) in {"a": NV.plant_wire_row(c, "AllReduceResidualMnnvl_v1", "parts", 3),
                             "b": NV.plant_read_shift(c, "RoPE")}.items():
    x = c.wires[what["wire"]] if key == "a" else c.registered.reads[what["read"]]
    lo, hi = x.at, x.at + x.reader_units
    s = np.asarray(picked[tid])
    pats = c.registered.reads if key == "b" else c.wires
    same = [p for p in pats if p.reader == tid and p.port == x.port]
    rec[key] = {"template": tid, "units": counts[tid], "sampled": len(s), "covered": [lo, hi],
                "sampled_in_cover": int(((s >= lo) & (s < hi)).sum()), "patterns_into_port": len(same),
                "pattern_spans": [[p.at, p.at + p.reader_units, list(p.cut)] for p in same][:12]}
(Path(a.out) / "plant_cover.json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
