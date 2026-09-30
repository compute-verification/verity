#!/usr/bin/env bash
# Queue the approved-weights GPU sweep (aw_gpu.py) on node 2's fill runner and wait for it, so that the research run
# which runs this script records the sweep's result as its attempt.
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> -- bash <shipped tree>/benchmarks/pouw/approved_weights/submit_fill.sh
# The run holds no GPU: the fill runner leases one per chunk and preempts it for leases and timed windows. The job exits
# 99 while variants remain (requeued), 143 when preempted (requeued), 0 when every variant's JSON exists. Each variant's
# JSON in $D/out is its checkpoint. Override the variant list with AW_VARIANTS.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(basename "$(cd "$HERE/../../.." && pwd)")
NAME=aw-gpu-tradeoff-${SHA:0:8}
D=/workspace/pouw/approved-weights/$NAME
FILL=/workspace/pouw/fill
VARIANTS=${AW_VARIANTS:-real:bf16:0,real:none:0,real:rot:0,real:none@nvfp4:0,real:rot@nvfp4:0,real:none:1,real:rot:1,real:none@nvfp4:1,real:rot@nvfp4:1,real:rot-sr:0,real:specfloor-30:0,real:clamp-16:0,real:mix-0.01:0,real:mix-0.05:0,real:mix-0.05:1,real:random:0,adv:none:0,adv:clamp-16:0,adv:specfloor-30:0,adv:rot:0,adv:rot-sr:0,random:none:0,random:rot:0}
mkdir -p "$D/out"
echo "sha=$SHA name=$NAME out=$D/out start=$(date -u +%FT%TZ)"
if [ ! -e "$FILL/queue/$NAME.sh" ] && [ ! -e "$FILL/running/$NAME.sh" ] && [ ! -e "$FILL/done/$NAME.sh" ]; then
  cat > "$D/$NAME.sh" << EOF
#!/usr/bin/env bash
# fill: owner=bc-8412d697-8e5a-51be-834a-b61a695bcc03 gpus=1 max_min=25 cpus=8 project=pous mem_gb=128
# Approved weights (bc-8412d697): loss against the verification cost each registered-weight property leaves, on
# Qwen2.5-7B and WikiText-2 (docs/pouw/approved-weights.md). Per variant (source:rung:recover): project every linear onto
# the rung, optionally recover glue parameters for 400 steps, perplexity on the whole WikiText-2 test split, and the
# skippable fraction of the sm_120 FP8 v1 chain on the deployed codes. Code: benchmarks/pouw/approved_weights/aw_gpu.py
# at $SHA (branch cursor/approved-weights-gpu-cc03), as research run shipped it. Checkpoints: one JSON per variant in
# $D/out. Exit 99 while variants remain, 143 when preempted, 0 when done. The waiting research run collects the result.
set -uo pipefail
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 AW_CLOCK=locked-2100
echo "uuid=\${GPU_LEASE_UUID:-?} cvd=\${CUDA_VISIBLE_DEVICES:-?} job=\${FILL_JOB:-?} sha=$SHA start=\$(date -u +%FT%TZ)"
exec /workspace/pouw/gpu7-fp4/venv/bin/python $HERE/aw_gpu.py --out $D/out --variants '$VARIANTS' --eval-windows 0 \\
  --recover-steps 400 --recover-seqlen 1024 --census-layers 4 --max-seconds 1200
EOF
  chmod +x "$D/$NAME.sh"
  mv "$D/$NAME.sh" "$FILL/queue/$NAME.sh"
  echo "queued $FILL/queue/$NAME.sh"
fi
missing() {
  python3 - "$D/out" "$VARIANTS" << 'PY'
import sys, pathlib
out, vs = pathlib.Path(sys.argv[1]), sys.argv[2].split(",")
print(sum(1 for v in vs if not (out / (v.replace(":", "_") + ".json")).exists()))
PY
}
while [ "$(missing)" != 0 ]; do
  if [ -e "$FILL/failed/$NAME.sh" ]; then echo "the fill job failed: $FILL/failed/$NAME.sh"; break; fi
  sleep 60
done
cp -r "$D/out" "$RESEARCH_RUN_DIR/out"
python3 - "$D/out" "$VARIANTS" "$RESEARCH_RUN_DIR/result.json" << 'PY'
import json, sys, pathlib
out, vs, dest = pathlib.Path(sys.argv[1]), sys.argv[2].split(","), sys.argv[3]
res = {}
for v in vs:
    f = out / (v.replace(":", "_") + ".json")
    if f.exists():
        d = json.loads(f.read_text())
        res[v] = {k: d[k] for k in ("ppl", "skippable_fraction_mean", "eval_windows", "recover_steps", "seconds")}
done = len(res) == len(vs)
json.dump({"kind": "approved-weights/gpu-tradeoff/v1", "done": done, "variants": res}, open(dest, "w"), indent=1)
print(json.dumps({"done": done, "variants": len(res)}))
sys.exit(0 if done else 1)
PY
