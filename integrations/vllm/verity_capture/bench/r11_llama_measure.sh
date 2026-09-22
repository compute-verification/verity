#!/usr/bin/env bash
# R11 lane collector-meas (owner 4): the Llama-3.2-1B complete-commitment overhead protocol (headline = UNPROFILED runs).
#
#   bash verity_capture/bench/r11_llama_measure.sh <tag> [arms] [batches=3] [rounds=4] [late_read_arm=native_collect_gather]
#
# Runs on the timing pod (cp-meas) from /workspace/cp/veritor.  Every batch is a SEPARATE PROCESS of `commit_delta --arms` (one engine,
# one control arm + the treatment arms interleaved within every round, order rotating; the control is in every round).  Ratios are formed
# per round against the control of the SAME round; pooled median + 95 % bootstrap CI of the mean + range of batch means come from
# `acq_aggregate.py`.  Arm 2 (`native_host` = frozen R9 complete path) is always included as the root-equality reference.
#
# Recorded per run (commit_delta's runs.jsonl): control and instrumented wall (matched request, 16 engine steps), overhead ratio,
# root-ready delay, process peak GPU memory + committer footprint (device arena, tree retained, values retained), backlog, copies /
# gather launches (native_collect counters), native host issue time, GPU leaf/tree time (CUDA events on the worker stream), run root
# (equality across arms per round), openings verified/64, tokens_equal_ref.  GPU *collection* time is not measured in the unprofiled
# runs (no events in the collector); it comes from the separate diagnostic profiled run (see r11_decompose.py).
#
# Correctness control, once per candidate: the LATE-READ negative test -- `commit_delta --late-read qkv_proj` drops the P0 flush of the
# gather arm so the staged q/k are read after RoPE's in-place rewrite; the run root MUST differ from arm 2 (roots_equal_across_arms=false).
#
# Env passed through to the candidate (recorded): NATIVE_GATHER_BLOCKS, NATIVE_COLLECT_BUILD, plus anything lane collector specifies.
set -uo pipefail
TAG=${1:?tag}; ARMS=${2:-native_host,native_collect_gather}; BATCHES=${3:-3}; ROUNDS=${4:-4}; LATE_ARM=${5:-native_collect_gather}
PY=${PY:-/workspace/venv-cu129/bin/python}
ROOT=${R11_OUT:-/workspace/cp/out-r11/collector-meas}; OUT=$ROOT/$TAG; mkdir -p "$OUT"
HIDDEN_SO=${HIDDEN_SO:-/workspace/cp/fa2/build/matReq/verity_fa2_matReq.so}
CASE=${CASE:-LLAMA32_1B}
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
export NATIVE_COLLECT_BUILD=${NATIVE_COLLECT_BUILD:-/workspace/cp/nc_build}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=$(dirname "$PY"):/usr/local/cuda/bin:$PATH   # JIT builds (native_collect, CMT-2 GPU tree) need nvcc + the venv's ninja
log() { echo "$(date -u +%FT%TZ) [r11-meas $TAG] $*" | tee -a "$OUT/protocol.log"; }
COMMON=(--case "$CASE" --pairs "$ROUNDS" --warmup 2 --engine-arg gpu_memory_utilization=0.5 --fast-copy --gpu-tree --hidden-so "$HIDDEN_SO" --arms "$ARMS")

# host facts (co-tenant noise: compare ratios within a round only)
{
  echo "{\"utc\": \"$(date -u +%FT%TZ)\", \"nproc\": $(nproc), \"loadavg\": \"$(cut -d' ' -f1-3 /proc/loadavg)\","
  echo " \"cpu\": \"$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)\", \"clocksource\": \"$(cat /sys/devices/system/clocksource/clocksource0/current_clocksource)\","
  echo " \"gpu\": \"$(nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader)\","
  echo " \"code_sha\": \"${CODE_SHA:-$(cat "$ROOT/CODE_SHA" 2>/dev/null || echo unknown)}\", \"tree\": \"$(pwd)\", \"hidden_so_sha256\": \"$(sha256sum "$HIDDEN_SO" | cut -c1-64)\","
  echo " \"env\": {\"NATIVE_GATHER_BLOCKS\": \"${NATIVE_GATHER_BLOCKS:-}\", \"NATIVE_COLLECT_BUILD\": \"$NATIVE_COLLECT_BUILD\", \"VERITY_FAULT\": \"${VERITY_FAULT:-}\", \"VERITY_CONSUMER_DELAY_MS\": \"${VERITY_CONSUMER_DELAY_MS:-}\"},"
  echo " \"arms\": \"$ARMS\", \"batches\": $BATCHES, \"rounds\": $ROUNDS, \"case\": \"$CASE\", \"argv_common\": \"${COMMON[*]}\"}"
} > "$OUT/host.json"
log "host: $(cat "$OUT/host.json" | tr -d '\n' | cut -c1-400)"

for b in $(seq 1 "$BATCHES"); do
  D=$OUT/batch$b; rm -rf "$D"
  log "batch $b/$BATCHES: $PY -m verity_capture.bench.commit_delta ${COMMON[*]} --label ${TAG}_batch$b --out $D  (loadavg $(cut -d' ' -f1 /proc/loadavg))"
  $PY -m verity_capture.bench.commit_delta "${COMMON[@]}" --label "${TAG}_batch$b" --out "$D" > "$OUT/batch$b.log" 2>&1
  rc=$?
  grep -E "^\[delta\] pair|roots identical" "$OUT/batch$b.log" | cut -c1-260 | tee -a "$OUT/protocol.log"
  log "batch $b rc=$rc"
done

# late-read negative test (one round; the gather arm's P0 flush is deferred past RoPE -> its root must differ from arm 2)
if [ -n "$LATE_ARM" ]; then
  D=$OUT/late_read; rm -rf "$D"
  # LATE_MODE=harness (default): commit_delta --late-read qkv_proj (drops qkv_proj from GATHER_FLUSH_LEAVES before install)
  # LATE_MODE=fault: the collector's own knob VERITY_FAULT=late_flush (lane collector's v2; same deferral, implemented in the collector)
  if [ "${LATE_MODE:-harness}" = fault ]; then LATE_ARGS=(); export VERITY_FAULT=late_flush; else LATE_ARGS=(--late-read qkv_proj); fi
  log "late-read negative test (${LATE_MODE:-harness}): --arms native_host,$LATE_ARM ${LATE_ARGS[*]:-} VERITY_FAULT=${VERITY_FAULT:-} --pairs 1"
  $PY -m verity_capture.bench.commit_delta --case "$CASE" --pairs 1 --warmup 1 --engine-arg gpu_memory_utilization=0.5 --fast-copy --gpu-tree \
      --hidden-so "$HIDDEN_SO" --arms "native_host,$LATE_ARM" "${LATE_ARGS[@]}" --label "${TAG}_late_read" --out "$D" > "$OUT/late_read.log" 2>&1
  unset VERITY_FAULT
  grep -E "LATE-READ|roots identical|^\[delta\] pair" "$OUT/late_read.log" | cut -c1-260 | tee -a "$OUT/protocol.log"
  $PY - "$D/summary.json" <<'EOF' | tee -a "$OUT/protocol.log"
import json, sys
s = json.load(open(sys.argv[1]))
eq = s["roots_equal_across_arms"]
print(f"LATE-READ TEST {'PASS (roots differ, as required)' if not eq else 'FAIL (late read went unnoticed: roots equal)'}: roots_by_round={s['roots_by_round']}")
json.dump({"late_read_test": "pass" if not eq else "FAIL", "roots_equal_across_arms": eq, "roots_by_round": s["roots_by_round"], "label": "measured"},
          open(sys.argv[1].replace("summary.json", "late_read_result.json"), "w"), indent=1)
EOF
fi

# pooled aggregate (per arm: pooled median, 95 % bootstrap CI of the mean, batch means, diagnostics)
$PY -m verity_capture.bench.acq_aggregate --raw "$OUT" --glob 'batch*' --out "$OUT/aggregate.json" | tee "$OUT/aggregate.md" | tee -a "$OUT/protocol.log"
log "DONE -> $OUT (aggregate.json / aggregate.md / batch*/runs.jsonl / late_read/)"
