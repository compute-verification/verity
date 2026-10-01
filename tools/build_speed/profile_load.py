"""Time each phase of `components.load_component` over a kept row dir's request components (measurement only, never merged):
gzip + json.loads, program_digest, canonical_json with annotations, request_component; and cProfile the largest one's decode.
usage: profile_load.py ROW_DIR OUT_DIR"""
import cProfile
import gzip
import hashlib
import json
import os
import pstats
import sys
import time

from verity.ir.codec import canonical_json, program_digest
from verity.ir.defs import REGISTRY
from verity_vllm.pipeline import gc_tuning
from verity_vllm.pipeline.components import wrapper_params
from verity_vllm.program import workload as W

row, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
gc_tuning.tune()
dirs = sorted((d for d in os.listdir(row) if d.startswith("build_request_LP") and os.path.isdir(os.path.join(row, d))),
              key=lambda d: -os.path.getsize(os.path.join(row, d, "descriptor.json.gz")))
canon, rows = {}, []
for i, name in enumerate(dirs):
    d = os.path.join(row, name)
    t = [time.perf_counter()]
    with gzip.open(os.path.join(d, "descriptor.json.gz"), "rb") as f:
        raw = f.read()
    t.append(time.perf_counter())
    desc = json.loads(raw)
    t.append(time.perf_counter())
    dg = program_digest(desc)
    t.append(time.perf_counter())
    sha = hashlib.sha256(canonical_json(desc)).hexdigest()
    t.append(time.perf_counter())
    params = wrapper_params(d)
    t.append(time.perf_counter())
    prof = cProfile.Profile() if i == 0 else None
    if prof:
        prof.enable()
    W.request_component(desc, REGISTRY, params, canon=canon, digest=dg)
    if prof:
        prof.disable()
        prof.dump_stats(os.path.join(out, f"decode_{name}.prof"))
        with open(os.path.join(out, f"decode_{name}.txt"), "w") as f:
            pstats.Stats(prof, stream=f).sort_stats("cumulative").print_stats(60)
            pstats.Stats(prof, stream=f).sort_stats("tottime").print_stats(40)
    t.append(time.perf_counter())
    ph = dict(zip(("gunzip", "json_loads", "program_digest", "canonical_with_annotations", "wrapper_params", "request_component"),
                  (round(b - a, 2) for a, b in zip(t, t[1:]))))
    rows.append({"dir": name, "raw_mb": round(len(raw) / 2**20, 1), "annotations": "annotations" in desc, **ph, "total": round(t[-1] - t[0], 2)})
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(os.path.join(out, "load_phases.json"), "w"), indent=1)
