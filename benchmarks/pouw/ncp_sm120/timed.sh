#!/usr/bin/env bash
# The timed window's workload, run inside its `gpu-lease ... --timed` from the source tree: run.py and form.py (at
# FORM_SHAPES, default SHAPES) with NCP_DEFINES as given, then run.py's chash and hash_ir arms again with BLAKE3's default
# mix as the control (OUT/mix0), then rates.py (the forming's price per element, OUT/rates).
# Host threads are pinned off node 2's cores 0-47 on the leased GPU's NUMA node (GPUs 0-3: 48-91; 4-7: 96-123, short of
# 124-191), recorded in OUT/pinning.json, and coreload.py samples every core's load into OUT/coreload.jsonl throughout.
set -euo pipefail
OUT=${RESEARCH_RUN_DIR:?}/out
HERE=benchmarks/pouw/ncp_sm120
mkdir -p "$OUT"
gpu=${CUDA_VISIBLE_DEVICES:?}
gpu=${gpu%%,*}
idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$gpu" '$1 == g || $2 == g {print $1}')
case "$idx" in
  0|1|2|3) cores=48-91; sampler_core=91 ;;
  4|5|6|7) cores=96-123; sampler_core=123 ;;
  *) echo "timed.sh: can't place GPU '$gpu'" >&2; exit 2 ;;
esac
if [ -n "${CORES:-}" ]; then cores=$CORES; sampler_core=${CORES##*-}; fi   # an untimed rehearsal keeps its queue's cores
printf '{"gpu": "%s", "index": %s, "cores": "%s", "sampler_core": %s}\n' "$gpu" "$idx" "$cores" "$sampler_core" \
  > "$OUT/pinning.json"
PINNING=$cores taskset -c "$sampler_core" nice -n 19 python3 "$HERE/coreload.py" "$OUT/coreload.jsonl" &
sampler=$!
trap 'kill -TERM "$sampler" 2>/dev/null || true' EXIT
taskset -c "$cores" python3 "$HERE/run.py" "$OUT"
SHAPES=${FORM_SHAPES:-${SHAPES:-p d}} taskset -c "$cores" python3 "$HERE/form.py" "$OUT"
NCP_DEFINES= S_IR=1 S_C="1 4 8" taskset -c "$cores" python3 "$HERE/run.py" "$OUT/mix0"
taskset -c "$cores" python3 "$HERE/rates.py" "$OUT"
