#!/usr/bin/env bash
# CPERF 2026-09-17: the heavy-slot (clean host) repeat on a free worker: build the B0 store, run the overlap experiment,
# run the bench (8 procs).  Prints three compact JSON lines (store / overlap / bench) for the manager's log tail.
set -uo pipefail
PY=/workspace/venv-cu129/bin/python
SNAP=$(pwd)
OUT=/workspace/cp/cperf/out
mkdir -p "$OUT" /workspace/cp/cperf/tmp
export PYTHONPATH=$SNAP/vllm-poc:$SNAP/src HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "== host"; hostname; lscpu | grep -E "Model name|^CPU\(s\)"; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader; uptime
T0=$(date +%s)
bash scripts/cprof_store_build.sh > "$OUT/store-build-heavy.log" 2>&1
echo "store-build exit=$? ($(($(date +%s)-T0))s)"
grep -o 'CPERF-STORE-JSON .*' "$OUT/store-build-heavy.log" | head -c 2500; echo
T1=$(date +%s)
$PY scripts/cprof_overlap.py --production /workspace/cp/cperf/B0/production --out-root "$OUT/overlap" \
    --checkpoints manifests/checkpoints.json --arena-gb 4 > "$OUT/overlap-heavy.log" 2>&1
echo "overlap exit=$? ($(($(date +%s)-T1))s)"
grep -E "^A |^B |^C |^D " "$OUT/overlap-heavy.log" | cut -c1-400
$PY - <<'EOF'
import json, os
p = "/workspace/cp/cperf/out/overlap/overlap.json"
d = json.load(open(p))
for kk, v in d.get("D_commit_alone", {}).items():
    if isinstance(v, dict):
        for x in list(v):
            if x not in ("first_s", "median_s", "cpu_median_s", "peak_rss_delta_kb_max", "all_s"):
                v.pop(x)
for k in ("A_replay_as_shipped", "B_replay_noop_hash"):
    if k in d:
        d[k] = {kk: d[k].get(kk) for kk in ("rc", "process_wall_s", "timing", "commitments")}
c = d.get("C_replay_plus_concurrent_watcher")
if c:
    c["replay"] = {kk: c["replay"].get(kk) for kk in ("rc", "process_wall_s", "timing", "commitments")}
w = "/workspace/cp/cperf/out/overlap/C.watcher.json"
per = json.load(open(w)).get("per_tree", {}) if os.path.exists(w) else {}
d["C_per_tree_compact"] = {t: {"seen": round(v["seen"], 3), "done": round(v["done"], 3), "leaves": v["leaves"]} for t, v in per.items()}
print("CPERF-OVERLAP-COMPACT " + json.dumps(d, separators=(",", ":"), default=str))
EOF
T2=$(date +%s)
$PY -m verity_vllm.commit_bench.bench --store /workspace/cp/cperf/B0/commit --store-v2 /workspace/cp/cperf/B0/commit-v2 \
    --out "$OUT/bench-heavy.json" --reps 3 --procs 8 --tmp /workspace/cp/cperf/tmp > "$OUT/bench-heavy.log" 2>&1
echo "bench exit=$? ($(($(date +%s)-T2))s)"
grep -o 'CPERF-BENCH-JSON .*' "$OUT/bench-heavy.log"
echo CPERF-HEAVY-DONE
