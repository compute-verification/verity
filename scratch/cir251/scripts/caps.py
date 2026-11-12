import json
import resource
import sys
import time
from pathlib import Path

from verity_vllm.pipeline import chunk_rows as CR, serving_rows as SRW

FILES = ["glm53_all_families_tp8_step16", "glm53_prefill_tp8_step0", "glm53_prefill_tp8_step0_core"]
want: dict[str, dict[str, int]] = {}
for f in FILES:
    for t, c in json.loads(Path(f"benchmarks/one_stage/{f}.json").read_text())["chunk_cap"].items():
        if t.startswith("DsaRMSNorm"):
            want.setdefault(t, {})[f] = c
bad = 0
for t in sorted(want):
    t0 = time.time()
    tp = SRW.template(t)
    cap = CR.chunk_cap(tp)
    print(json.dumps({"template": t, "layout": tp.layout, "chunk_cap": cap, "seconds": round(time.time() - t0, 1),
                      "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}), flush=True)
    for f, c in sorted(want[t].items()):
        ok = c <= cap and (f.endswith("step16") or c == cap)
        bad += not ok
        print(f"  {f}: {c} {'ok' if ok else 'FAILS'} ({'at most' if f.endswith('step16') else 'equal to'} {cap})", flush=True)
sys.exit(1 if bad else 0)
