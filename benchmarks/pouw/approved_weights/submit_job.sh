#!/usr/bin/env bash
# Queue one approved-weights job on node 2's fill runner, wait for it, and collect it into the research run that runs this.
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> -- \
#     bash <shipped tree>/benchmarks/pouw/approved_weights/submit_job.sh NAME GPUS -- SCRIPT [ARGS...]
# SCRIPT (in this directory) must checkpoint in --out and exit 0 when done, 99 while chunks remain, 143 when preempted. The
# research run holds no GPU. Once the fill job is done, the outputs are copied into $RESEARCH_RUN_DIR/out and SCRIPT runs
# once more there with nothing left to compute, which writes $RESEARCH_RUN_DIR/result.json (research/result/v0.1).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(basename "$(cd "$HERE/../../.." && pwd)")
NAME=$1; GPUS=$2; shift 2; [ "${1:-}" = "--" ] && shift
SCRIPT=$1; shift
JOB=aw-$NAME-${SHA:0:8}
D=/workspace/pouw/approved-weights/$JOB
FILL=/workspace/pouw/fill
PY=/workspace/pouw/gpu7-fp4/venv/bin/python
ENVS="HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8"
ARGS=$(printf ' %q' "$@")
MEM=$([ "$GPUS" = 0 ] && echo 64 || echo 128)
mkdir -p "$D/out"
echo "sha=$SHA job=$JOB out=$D/out start=$(date -u +%FT%TZ)"
if [ ! -e "$FILL/queue/$JOB.sh" ] && [ ! -e "$FILL/running/$JOB.sh" ] && [ ! -e "$FILL/done/$JOB.sh" ]; then
  cat > "$D/$JOB.sh" << EOF
#!/usr/bin/env bash
# fill: owner=bc-8412d697-8e5a-51be-834a-b61a695bcc03 gpus=$GPUS max_min=25 cpus=8 project=pous mem_gb=$MEM
# Approved weights (bc-8412d697), docs/pouw/approved-weights.md "Scale": $NAME. Code: benchmarks/pouw/approved_weights/$SCRIPT
# at $SHA (branch cursor/approved-weights-gpu-cc03), as research run shipped it. Checkpoints in $D/out; exit 0 when done,
# 99 while chunks remain, 143 when preempted. A waiting research run collects the result.
set -uo pipefail
export $ENVS
echo "uuid=\${GPU_LEASE_UUID:-cpu} job=\${FILL_JOB:-?} sha=$SHA start=\$(date -u +%FT%TZ)"
exec $PY $HERE/$SCRIPT --out $D/out$ARGS
EOF
  chmod +x "$D/$JOB.sh"
  mv "$D/$JOB.sh" "$FILL/queue/$JOB.sh"
  echo "queued $FILL/queue/$JOB.sh"
fi
until [ -e "$FILL/done/$JOB.sh" ] || [ -e "$FILL/failed/$JOB.sh" ]; do sleep 60; done
[ -e "$FILL/failed/$JOB.sh" ] && echo "the fill job failed: $FILL/failed/$JOB.sh"
cp -r "$D/out" "$RESEARCH_RUN_DIR/out"
env $ENVS "$PY" "$HERE/$SCRIPT" --out "$RESEARCH_RUN_DIR/out" "$@"
