"""The engines' gpu_memory_utilization read back from their vllm_config (`effective_execution`, in tp-capture's topology.json and
tp-commit's versions.json) against the value the job set: written to OUT as JSON, exit 1 unless every engine read back that value.
    python3 gpu_util_read_back.py OUT ASKED FILE..."""
import json
import sys
from pathlib import Path

out, asked, files = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
engines = {}
for f in files:
    try:
        doc = json.loads(Path(f).read_text())
    except (OSError, ValueError) as e:
        engines[f] = {"error": repr(e)[:200]}
        continue
    eff = doc.get("effective_execution") or {}
    engines[f] = {"read_back": eff.get("gpu_memory_utilization"), "config_error": eff.get("config_error"),
                  "asked": (doc.get("engine_kwargs") or {}).get("gpu_memory_utilization")}
ok = bool(engines) and all(e.get("read_back") == asked for e in engines.values())
Path(out).write_text(json.dumps({"asked": asked, "engines": engines, "equal": ok}, indent=1) + "\n")
for f, e in engines.items():
    print(f"gpu_memory_utilization {f}: read back {e.get('read_back')} (set {asked}){'' if e.get('read_back') == asked else ' MISMATCH'}")
sys.exit(0 if ok else 1)
