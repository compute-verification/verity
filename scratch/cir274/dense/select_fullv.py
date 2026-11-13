"""CIR-274 DENSE piece 5: `TokenSelect_v4{V}`, `TokenSelect_v1{V}` and `greedy_stages.plan(V)`'s units, each its own circuit by
the reference evaluator (no `numerics.rows` kernel), at the served V = 154,880, against `greedy_stages.twin`.

Usage (from adapters/vllm): select_fullv.py OUT [--skip-v1]

Rows: two of bf16 N(0, 1/4); the tie row (the largest finite bf16 at lanes 38,820, 45,720 and 92,933 of an N(0, 1/4) row);
a row whose logits[0] is a NaN. Writes $OUT/select_fullv.json after each evaluation."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import verity.definitions.numerics.rows  # noqa: F401  (registers the row kernels)
from verity.circuits.defs import bind
from verity.circuits.evaluation.batch import evaluate_batch
from verity.definitions.registry import b1
from verity.definitions.registry import dead_free as DF
from verity.definitions.registry import greedy_stages as G

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--skip-v1", action="store_true")
a = ap.parse_args()
V = 154880
path = Path(a.out, "select_fullv.json")
rng = np.random.default_rng(274)
f = rng.normal(0, 0.5, (4, V)).astype(np.float32)
rows = (f.view(np.uint32) >> 16).astype(np.uint16)
rows[2, [38820, 45720, 92933]] = 0x7F7F
rows[3, 0] = 0x7FC0
names = ["normal_a", "normal_b", "tie", "nan_first"]
rec: dict = {"V": V, "rows": names, "twin": [int(G.twin(V, r)[0]) for r in rows], "by": {}}


def tok(got):
    got = np.asarray(got).reshape(len(rows), -1).astype(np.uint64)
    return [int(x) for x in (got[:, 0] if got.shape[1] == 1 else got[:, 0] | (got[:, 1] << 16))]


def timed(name, fn):
    t = time.perf_counter()
    rec["by"][name] = {"tokens": fn(), "s": round(time.perf_counter() - t, 1)}
    rec["by"][name]["equals_twin"] = rec["by"][name]["tokens"] == rec["twin"]
    path.write_text(json.dumps(rec, indent=1))
    print(name, rec["by"][name], flush=True)


def units():
    outs = {}
    for u in G.plan(V):
        wires = {p: outs[k] for p, k, _f in u.wires}
        args = [rows[:, u.lanes[0]:u.lanes[1]] if p == "logits" else wires[p] for p, _t in u.defn.signature()[0]]
        outs[u.key] = np.asarray(evaluate_batch(u.defn, *args, kernel="reference")).reshape(len(rows), -1)
    return tok(outs[("combine",)])


print("twin", rec["twin"], flush=True)
timed("units (20, reference)", units)
timed("TokenSelect_v4 (reference)", lambda: tok(evaluate_batch(bind(DF.TokenSelectV4, V=V), rows, kernel="reference")))
timed("TokenSelect_v4 (auto: numerics.rows)", lambda: tok(evaluate_batch(bind(DF.TokenSelectV4, V=V), rows)))
if not a.skip_v1:
    timed("TokenSelect_v1 (reference)", lambda: tok(evaluate_batch(bind(b1.TokenSelect, V=V), rows, kernel="reference")))
rec["all_equal"] = all(v["equals_twin"] for v in rec["by"].values())
path.write_text(json.dumps(rec, indent=1))
print("all_equal", rec["all_equal"])
sys.exit(0 if rec["all_equal"] else 1)
