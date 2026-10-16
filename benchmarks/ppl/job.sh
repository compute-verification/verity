#!/usr/bin/env bash
# `run.py eval` on vy-nebius-2 under a gpu-lease of PPL_GPUS GPUs (cwd = the source's root). `run.py submit` writes the
# `research run` that calls this, and sends the two dataset files (datasets.json) with it. Clocks are read, never set.
#
#   PPL_ARGS   eval's flags but --out and --data (`--model M --set S --ctx T --gpus N --seeds 0,1,2 SPEC...`)
#   PPL_GPUS   GPUs to lease (1, or 2 for Llama-3.1-70B); PPL_MAX_MIN the lease's bound in minutes
set -uo pipefail
export O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
export PY=/workspace/jobs/venv312/bin/python HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PPL_DATA=$RESEARCH_RUN_DIR/inputs
export PPL_ARGS=${PPL_ARGS:?}
# shellcheck disable=SC2016
gpu-lease "${PPL_GPUS:-1}" --wait --max-min "${PPL_MAX_MIN:-120}" -- bash -c '
  q=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit
  nvidia-smi --query-gpu=$q --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  # shellcheck disable=SC2086
  "$PY" benchmarks/ppl/run.py eval --out "$O" --data "$PPL_DATA" $PPL_ARGS > "$O/stdout.txt" 2> "$O/stderr.txt"
  r=$?
  nvidia-smi --query-gpu=index,clocks.sm,clocks.mem --format=csv -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  exit $r'
rc=$?
echo "eval $rc" | tee "$O/status.txt"
tail -5 "$O/stderr.txt" 2>/dev/null
exit $rc
