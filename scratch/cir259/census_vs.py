"""census_vs.py: each staged row of table.json beside the census card's entry for its Definition or its word."""
import json
from pathlib import Path

C = json.loads(Path("/agent/repos/verity/benchmarks/sampling_plan/glm53_blocks.json").read_text())["definitions"]
for r in json.loads(Path("/opt/cursor/artifacts/cir259/table.json").read_text()):
    if not r.get("staged"):
        continue
    for nm in (r["definition"], r.get("word")):
        if nm in C:
            c = C[nm]
            print(f"{r['tag']}: staged {r['ands']:,} k_log {r['k_log']} | census {c['ands']:,} k_log {c['k_log']} ({nm[:60]}; of {c['of']}; prefill {c['prefill']:,} decode {c['decode']:,})")
            break
    else:
        print(f"{r['tag']}: staged {r['ands']:,} k_log {r['k_log']} | not in the census card: {r['definition'][:70]} / {r.get('word')}")
