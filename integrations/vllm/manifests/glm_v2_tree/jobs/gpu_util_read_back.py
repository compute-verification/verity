"""The engines' gpu_memory_utilization read back from their vllm_config (`effective_execution`, in tp-capture's topology.json and
tp-commit's versions.json) against the value the job set: the driver's, and each rank's own (its probe's `effective_execution`,
under `ranks`: one per rank of the TP group, ranks 0..world-1).  A file named `driver:FILE` is read for the driver's alone (the
--no-observer control probes no rank).  Written to OUT as JSON; exit 1 unless every engine and rank read back that value.
    python3 gpu_util_read_back.py OUT ASKED [driver:]FILE..."""
import json
import sys
from pathlib import Path


def ranks_of(doc, asked):
    """(each rank's read-back by rank, why the ranks fail or None)."""
    ranks = doc.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        return {}, "no rank's read-back (ranks)"
    got = {}
    for i, r in enumerate(ranks):
        tp = (r or {}).get("tp") or {}
        got[str(tp.get("rank", f"?{i}"))] = ((r or {}).get("effective_execution") or {}).get("gpu_memory_utilization")
    world = {((r or {}).get("tp") or {}).get("world_size") for r in ranks}
    if world != {len(ranks)} or len(got) != len(ranks) or set(got) != {str(k) for k in range(len(ranks))}:
        return got, f"the ranks answering are {sorted(got)} of world {sorted(map(str, world))}"
    bad = sorted(k for k, v in got.items() if v != asked)
    return got, f"rank(s) {bad} read back another value" if bad else None


out, asked, files = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
engines = {}
for arg in files:
    driver_only = arg.startswith("driver:")
    f = arg.removeprefix("driver:")
    try:
        doc = json.loads(Path(f).read_text())
    except (OSError, ValueError) as e:
        engines[f] = {"error": repr(e)[:200]}
        continue
    eff = doc.get("effective_execution") or {}
    e = engines[f] = {"read_back": eff.get("gpu_memory_utilization"), "config_error": eff.get("config_error"),
                      "asked": (doc.get("engine_kwargs") or {}).get("gpu_memory_utilization")}
    if driver_only:
        e["ranks"] = "driver only (no rank probed)"
    else:
        e["ranks"], e["ranks_error"] = ranks_of(doc, asked)
ok = bool(engines) and all(e.get("read_back") == asked and not e.get("ranks_error") and "error" not in e for e in engines.values())
Path(out).write_text(json.dumps({"asked": asked, "engines": engines, "equal": ok}, indent=1) + "\n")
for f, e in engines.items():
    bad = e.get("read_back") != asked or e.get("ranks_error") or "error" in e
    ranks = e["ranks"] if isinstance(e.get("ranks"), str) else f"{len(e.get('ranks') or {})} ranks"
    print(f"gpu_memory_utilization {f}: driver read back {e.get('read_back')}, {ranks} (set {asked})"
          f"{' MISMATCH: ' + str(e.get('ranks_error') or e.get('error') or 'the driver') if bad else ''}")
sys.exit(0 if ok else 1)
