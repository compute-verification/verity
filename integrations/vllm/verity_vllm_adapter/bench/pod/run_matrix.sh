#!/bin/bash
# W4d (H100): the §16.7 matrix, one process per model x configuration, sequential (the GPU is the
# resource under test; nothing else runs on it meanwhile).
# usage: run_matrix.sh <out_root> "<repo>=<snapshot_path>[:<util>]" ...   configs from $CONFIGS (default "C0 C1 C2")
set -u
OUT=$1; shift
CONFIGS=${CONFIGS:-"C0 C1 C2"}
SHAPES=${SHAPES:-"128x128,512x64"}
NREQ=${NREQ:-256}; REPS=${REPS:-3}
cd /workspace/veritor/vllm-poc
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
PY=${PY:-/workspace/venv/bin/python}
mkdir -p "$OUT" /workspace/logs
echo "[$(date -u +%FT%TZ)] matrix start configs=[$CONFIGS] shapes=$SHAPES nreq=$NREQ reps=$REPS" | tee -a "$OUT/matrix.log"
for spec in "$@"; do
  repo=${spec%%=*}; rest=${spec#*=}; path=${rest%%:*}; util=${rest#*:}; [ "$util" = "$rest" ] && util=0.90
  short=${repo##*/}
  for cfg in $CONFIGS; do
    d="$OUT/$short/$cfg"; log="/workspace/logs/bench-$short-$cfg.log"
    if [ -f "$d/engine.json" ]; then echo "[$(date -u +%FT%TZ)] skip $short/$cfg (done)" | tee -a "$OUT/matrix.log"; continue; fi
    echo "[$(date -u +%FT%TZ)] start $short/$cfg util=$util" | tee -a "$OUT/matrix.log"
    t0=$(date +%s)
    $PY -m verity_vllm_adapter.bench.throughput --config "$cfg" --repo "$repo" --model-path "$path" \
        --shapes "$SHAPES" --num-requests "$NREQ" --reps "$REPS" --profile --gpu-memory-utilization "$util" \
        --out "$d" --tag "h100-$short-$cfg" > "$log" 2>&1 < /dev/null
    rc=$?
    echo "[$(date -u +%FT%TZ)] end $short/$cfg rc=$rc in $(( $(date +%s) - t0 ))s: $(grep -E '^DONE|Error|error:|OutOfMemory|CUDA out of memory' "$log" | tail -1 | cut -c1-240)" | tee -a "$OUT/matrix.log"
    [ $rc -ne 0 ] && { echo "FAILED $short/$cfg rc=$rc" > "$d.FAILED"; grep -E "Error|error" "$log" | tail -5 | cut -c1-300 >> "$d.FAILED"; }
    sleep 3
  done
done
echo "[$(date -u +%FT%TZ)] MATRIX_DONE" | tee -a "$OUT/matrix.log"
