#!/usr/bin/env bash
# R11 collector-meas: targeted UNPROFILED ablations of the baseline gather collector on Llama-3.2-1B (diagnostic; each ablation is its own
# process, ratios vs the control of the same round).  Complements the profiled run (r11_decompose.py) with unprofiled deltas:
#   abl_hooks     gather32 vs hooks_only (R9 hook-dispatch floor) in one process              -> hook entry class
#   abl_nofa2     gather32 + native_collect WITHOUT the FA2 M1 source (coverage reduced FOR THE ABLATION ONLY, never a headline)
#                                                                                              -> FA2 wrapper glue + acquire + tapped-kernel + its bytes
#   abl_delay     gather32 with VERITY_CONSUMER_DELAY_MS=100 (worker deferred out of the run)   -> worker / GPU-tree interference class
#   abl_blocks64  gather with NATIVE_GATHER_BLOCKS=64 (vs 32)                                   -> is the gather kernel's GPU time exposed in wall?
#   acq_ablation  isolated per-tensor CPU costs (acq_ablation.py)
#   bash verity_capture/bench/r11_ablation.sh <tag> [rounds=3]
#   bash verity_capture/bench/r11_ablation.sh <tag> [rounds=3] cand <arm>   -> only abl_delay + abl_nofa2 for the frozen candidate arm
set -uo pipefail
TAG=${1:?tag}; ROUNDS=${2:-3}; MODE=${3:-full}; ARM=${4:-native_collect_gather}
PY=/workspace/venv-cu129/bin/python; ROOT=/workspace/cp/out-r11/collector-meas; OUT=$ROOT/$TAG; mkdir -p "$OUT"
HIDDEN_SO=/workspace/cp/fa2/build/matReq/verity_fa2_matReq.so
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 NATIVE_COLLECT_BUILD=${NATIVE_COLLECT_BUILD:-/workspace/cp/nc_build}
export CUDA_HOME=/usr/local/cuda PATH=$(dirname "$PY"):/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) [r11-abl $TAG] $*" | tee -a "$OUT/protocol.log"; }
run() { # name, env-assignments..., -- args
  local name=$1; shift; local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  log "$name: env ${envs[*]} :: $*  (loadavg $(cut -d' ' -f1 /proc/loadavg))"
  env "${envs[@]}" $PY -m verity_capture.bench.commit_delta --case LLAMA32_1B --pairs "$ROUNDS" --warmup 2 --engine-arg gpu_memory_utilization=0.5 --fast-copy --gpu-tree "$@" \
      --label "${TAG}_$name" --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  log "$name rc=$? :: $(grep -E 'roots identical' "$OUT/$name.log" | cut -c1-200)"
  grep -E "^\[delta\] pair" "$OUT/$name.log" | cut -c1-200 >> "$OUT/protocol.log"
}
if [ "$MODE" = cand ]; then
  run abl_delay  NATIVE_GATHER_BLOCKS=32 VERITY_CONSUMER_DELAY_MS=100 -- --hidden-so "$HIDDEN_SO" --arms "$ARM"
  run abl_nofa2  NATIVE_GATHER_BLOCKS=32 -- --arms "$ARM"
  $PY -m verity_capture.bench.acq_aggregate --raw "$OUT" --glob 'abl_*' --out "$OUT/aggregate.json" | tee "$OUT/aggregate.md" | tee -a "$OUT/protocol.log"
  log "DONE (cand $ARM) -> $OUT"; exit 0
fi
run abl_hooks    NATIVE_GATHER_BLOCKS=32 -- --hidden-so "$HIDDEN_SO" --arms native_collect_gather,hooks_only
run abl_nofa2    NATIVE_GATHER_BLOCKS=32 -- --arms native_collect_gather,native_collect
run abl_delay    NATIVE_GATHER_BLOCKS=32 VERITY_CONSUMER_DELAY_MS=100 -- --hidden-so "$HIDDEN_SO" --arms native_collect_gather
run abl_blocks64 NATIVE_GATHER_BLOCKS=64 -- --hidden-so "$HIDDEN_SO" --arms native_collect_gather
log "acq_ablation (isolated CPU microbench)"
$PY -m verity_capture.bench.acq_ablation --out "$OUT/acq_ablation.json" > "$OUT/acq_ablation.log" 2>&1; log "acq_ablation rc=$?"
$PY -m verity_capture.bench.acq_aggregate --raw "$OUT" --glob 'abl_*' --out "$OUT/aggregate.json" | tee "$OUT/aggregate.md" | tee -a "$OUT/protocol.log"
log "DONE -> $OUT"
