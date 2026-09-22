#!/usr/bin/env bash
# CPERF 2026-09-17: build the like-for-like B0 commitment store on a campaign worker (light job, shared GPU).
#   production (workload-B0-small.json, eager)  ->  replay --mode commit --weights-values store  ->  weights_v2 --rows-per-leaf 1
# Same pipeline as the 2026-09-08 CPROF study (scripts/cprof_pod.sh); the only declared deviation is
# gpu_memory_utilization (an --engine-arg, recorded in the manifest) chosen from the GPU memory actually free, because the
# worker's GPU is shared with a heavy campaign job.  Everything lands under /workspace/cp/cperf/B0 (never in a campaign run dir).
# Prints a compact JSON summary at the end (the manager keeps only the log tail).
set -uo pipefail
PY=/workspace/venv-cu129/bin/python
SNAP=$(pwd)
POC=$SNAP/vllm-poc
OUT=${CPERF_OUT:-/workspace/cp/cperf/B0}
mkdir -p "$OUT"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONPATH=$POC:$SNAP/src VLLM_USE_V2_MODEL_RUNNER=1
export TOKENIZERS_PARALLELISM=false
REV=$(python3 -c 'import json;print(json.load(open("EXPORT.json"))["commit"])' 2>/dev/null || echo cperf)

# GPU memory fraction from what is free right now (shared GPU); keep >= 0.08 and <= 0.45
read -r USED TOTAL <<<"$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | tr -d ',')"
FRAC=$(python3 -c "u,t=$USED,$TOTAL; f=(t-u-1800)/t; print(round(min(0.45,max(0.08,f)),2))")
echo "gpu used=${USED}MiB total=${TOTAL}MiB -> gpu_memory_utilization=$FRAC"
T0=$(date +%s)

if [ ! -f "$OUT/production/production.jsonl" ]; then
  rm -rf "$OUT/production"
  $PY -m verity_vllm_adapter.production --case B0 \
      --checkpoints "$POC/manifests/checkpoints.json" --workload "$POC/e2e/workload-B0-small.json" \
      --out "$OUT/production" --adapter-revision "$REV" --enforce-eager \
      --engine-arg "gpu_memory_utilization=$FRAC" > "$OUT/production.log" 2>&1
  echo "production exit=$? ($(($(date +%s)-T0))s)"; tail -3 "$OUT/production.log"
fi
T1=$(date +%s)
if [ ! -f "$OUT/commit/commitments.json" ]; then
  rm -rf "$OUT/commit"
  $PY -m verity_vllm_adapter.replay --case B0 \
      --checkpoints "$POC/manifests/checkpoints.json" --production "$OUT/production" \
      --out "$OUT/commit" --mode commit --weights-values store \
      --adapter-revision "$REV" --arena-gb 4 --replay-exec eager > "$OUT/replay-commit.log" 2>&1
  echo "replay-commit exit=$? ($(($(date +%s)-T1))s)"; tail -3 "$OUT/replay-commit.log"
fi
T2=$(date +%s)
if [ ! -f "$OUT/commit-v2/commitments.json" ]; then
  rm -rf "$OUT/commit-v2"
  $PY -m verity_vllm.weights_v2 "$OUT/commit" "$OUT/commit-v2" --rows-per-leaf 1 > "$OUT/weights-v2.log" 2>&1
  echo "weights-v2 exit=$? ($(($(date +%s)-T2))s)"; tail -2 "$OUT/weights-v2.log"
fi
du -sh "$OUT"/* 2>/dev/null
$PY - "$OUT" <<'EOF'
import json, sys, os, subprocess, platform
out = sys.argv[1]
def rd(p):
    try: return json.load(open(p))
    except Exception as e: return {"error": str(e)}
r = rd(f"{out}/commit/replay.json")
prod_t = rd(f"{out}/production/timings.json")
conv = rd(f"{out}/commit-v2/weights/conversion.json")
lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
model = [l.split(":",1)[1].strip() for l in lscpu.splitlines() if l.startswith("Model name")]
gpu = subprocess.run(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
doc = {
 "schema": "veritor/cperf/store-build/v1", "out": out, "host": platform.node(), "cpu": model[0] if model else None, "gpu": gpu,
 "cpu_quota": open("/sys/fs/cgroup/cpu.max").read().strip() if os.path.exists("/sys/fs/cgroup/cpu.max") else None, "nproc": os.cpu_count(),
 "replay": {"profile_id": r.get("profile_id"), "argv_engine": r.get("engine_kwargs"), "timing": r.get("timing"), "capture": r.get("capture"),
            "commitments": r.get("commitments"), "trajectory_followed": r.get("trajectory_followed")},
 "production_timings": prod_t if isinstance(prod_t, dict) else None,
 "conversion": {k: conv.get(k) for k in ("from","to","seconds","values_nbytes")} if isinstance(conv, dict) else conv,
}
s = json.dumps(doc, separators=(",",":"), default=str)
print("CPERF-STORE-JSON " + s)
open(f"{out}/store-build.json","w").write(json.dumps(doc, indent=1, default=str))
EOF
echo CPERF-STORE-DONE
