#!/usr/bin/env bash
# linears_layerwise.py on one leased GPU of vy-nebius-2 (cwd = the source's root): the CPU selftest before the lease, then
# every method in one pass over the model's weights, layer by layer.  Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=... --env LL_IDS=c4val2048x16.npy [--env LL_CALIB=c4cal2048x16.npy] [--env LL_DIAG=4] \
#     --env LL_METHODS="bf16 r127 d63 ..." [--env LL_MODEL=llama70b|llama8b] [--env LL_TAG=SUBDIR] \
#     --send c4val2048x16.npy [--send c4cal2048x16.npy] -- bash benchmarks/fp8_instantiation/run_linears.sh
#
# LL_IDS / LL_CALIB name files sent with the run (c4_windows.py: held-out C4 validation windows, calibration disjoint).
set -uo pipefail
export O=${RESEARCH_RUN_DIR:?}/out${LL_TAG:+/$LL_TAG}
mkdir -p "$O"
export PY=/workspace/jobs/venv312/bin/python
case "${LL_MODEL:-llama70b}" in
  llama8b) SNAP=/workspace/hf/hub/models--unsloth--Llama-3.1-8B/snapshots/3f0d51f8e5640f98f1a96ea9044a0e55c0a83814 ;;
  llama70b) SNAP=/workspace/hf/hub/models--unsloth--Meta-Llama-3.1-70B/snapshots/1b7306651142d0cc65d993076a250a6a82cf046c ;;
  *) echo "LL_MODEL is llama8b or llama70b" >&2; exit 2 ;;
esac
export SNAP HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
IN=$RESEARCH_RUN_DIR/inputs
export ARGS="$IN/${LL_IDS:?} $O ${LL_CALIB:+--calib $IN/$LL_CALIB} --batch ${LL_BATCH:-4} --diag ${LL_DIAG:-0}"
export LL_METHODS=${LL_METHODS:?}
S=benchmarks/fp8_instantiation/linears_layerwise.py
"$PY" "$S" --selftest > "$O/selftest.json" 2> "$O/selftest_stderr.txt" || { echo "selftest failed" | tee "$O/status.txt"; exit 3; }
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${LL_MAX_MIN:-90}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "$PY" benchmarks/fp8_instantiation/linears_layerwise.py "$SNAP" $ARGS $LL_METHODS > "$O/stdout.txt" 2> "$O/stderr.txt"
  r=$?
  nvidia-smi --query-gpu=index,clocks.sm,clocks.mem --format=csv -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  exit $r'
rc=$?
echo "linears $rc" | tee "$O/status.txt"
tail -5 "$O/stderr.txt" 2>/dev/null
exit $rc
