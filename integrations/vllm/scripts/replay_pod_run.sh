#!/usr/bin/env bash
# [REPLAY, additive] The full trusted-replay + instrumentation-ladder matrix on the pod
# (GUIDANCE §32.9A/.10). Assumes replay_pod_setup.sh succeeded and the repo subset is at
# /workspace/veritor/vllm-poc. Everything lands under /workspace/results/replay/.
set -euxo pipefail

PY=/workspace/venv/bin/python
POC=/workspace/veritor/vllm-poc
OUT=/workspace/results/replay
REV=replay-lane
WHEEL_SHA="7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3"
export PYTHONPATH=$POC
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
mkdir -p $OUT
cd $POC

# 0. the reconstructed TA1 workload (request id asserted == dc159809-...)
$PY scripts/replay_b0_workload.py --out $OUT/workload.json

# 1. uninstrumented rungs + trusted-replay configs (fresh process per config)
for EXEC in stock-pure stock-inproc bi-graphs eager-stock-kernels bi-eager; do
  $PY scripts/replay_trusted_b0.py --exec $EXEC --workload $OUT/workload.json \
    --checkpoints manifests/checkpoints.json --out $OUT --reps 5 --profile \
    2>&1 | tee $OUT/log-$EXEC.txt
done

# 2. recorder rung: TA1's exact production argv (eager + BI + StepTrace recorder,
#    recording-on/off reps) -> also the production record the replay rungs consume
$PY -m verity_vllm_adapter.production --case B0 --checkpoints manifests/checkpoints.json \
  --workload $OUT/workload.json --out $OUT/production --adapter-revision $REV \
  --wheel-sha256 $WHEEL_SHA --vllm-commit d9105ea80 --enforce-eager \
  2>&1 | tee $OUT/log-production.txt

# 3. trace-replay driver, no capture (rung: forced-trace engine only)
$PY -m verity_vllm_adapter.replay --case B0 --checkpoints manifests/checkpoints.json \
  --production $OUT/production --out $OUT/replay-nocap --mode reference --no-captures \
  --adapter-revision $REV --arena-gb 8 --replay-exec eager \
  2>&1 | tee $OUT/log-replay-nocap.txt

# 4. capture, hash-only (rung: + expose/materialize interior values, no storage)
$PY -m verity_vllm_adapter.replay --case B0 --checkpoints manifests/checkpoints.json \
  --production $OUT/production --out $OUT/replay-ref --mode reference \
  --adapter-revision $REV --arena-gb 8 --replay-exec eager \
  2>&1 | tee $OUT/log-replay-ref.txt

# 5. full capture + commit (rung: + store values + Merkle commitments; TA1's argv)
$PY -m verity_vllm_adapter.replay --case B0 --checkpoints manifests/checkpoints.json \
  --production $OUT/production --out $OUT/replay-commit --mode commit --weights-values hash \
  --adapter-revision $REV --arena-gb 8 --replay-exec eager \
  2>&1 | tee $OUT/log-replay-commit.txt

# 6. small summary bundle to pull back (JSON + logs only, no values.bin)
mkdir -p $OUT/summary
cp $OUT/trusted-*.json $OUT/summary/
cp $OUT/production/timings.json $OUT/summary/production-timings.json
cp $OUT/production/run.json $OUT/summary/production-run.json 2>/dev/null || true
cp $OUT/production/manifest.json $OUT/summary/production-manifest.json
cp $OUT/production/production.jsonl $OUT/summary/production.jsonl
for R in nocap ref commit; do
  $PY - "$OUT/replay-$R/replay.json" "$OUT/summary/replay-$R.json" <<'PYEOF'
import json, sys
doc = json.load(open(sys.argv[1]))
keep = {k: doc.get(k) for k in (
    "case", "mode", "replay_exec", "no_captures", "store_values", "profile_id",
    "effective_execution", "trajectory_followed", "unforced_sampler_agreement",
    "logprobs_vs_production", "per_request", "timing", "capture", "commitments",
    "warmup_excluded", "utc")}
json.dump(keep, open(sys.argv[2], "w"), indent=1)
PYEOF
done
du -sh $OUT/* | tee $OUT/summary/du.txt
echo RUN-OK
