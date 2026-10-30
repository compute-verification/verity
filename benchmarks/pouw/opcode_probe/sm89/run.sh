#!/usr/bin/env bash
# The sm_89 (L40S) census probes: build scalar.cu (SRC=uniform: uniform.cu; SRC=ji16 or ji32: ji.py's grids) for
# sm_89, gate it, then gate again and time on one leased GPU, with nvidia-smi's clocks, power and throttle reasons sampled beside it. From a pushed commit:
#
#   research run --on vy-dc-l40s --project verity --source <tree> --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/opcode_probe/sm89/run.sh [-- probe.py run flags]
#
# $OUT gets probe.py run's census.csv (`since` sm89), rows.jsonl, gate.json and times.jsonl, the SASS, and the trace.
# Clocks are read, never set.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc)
export NVCC PATH="$(dirname "$NVCC"):$PATH"
[ "${1:-}" = "--" ] && shift
mkdir -p "$OUT/sass"
SRC=${SRC:-scalar}
BIN="$OUT/$SRC"

"$NVCC" -std=c++20 -O3 -gencode arch=compute_89,code=sm_89 -ldl -Xptxas -warn-spills -o "$BIN" "$HERE/$SRC.cu" \
  >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
cuobjdump -sass "$BIN" >"$OUT/sass/$SRC.sass" 2>&1
python3 "$HERE/../probe.py" gate "$BIN" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
rc=$?
grep -v '^PASS' "$OUT/gate-build.txt"
[ $rc -eq 0 ] || exit $rc

cat >"$OUT/gpu.sh" <<EOF
set -uo pipefail
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE >"$OUT/smi-before.txt" 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active \
  --format=csv,noheader -lms 200 >"$OUT/smi-samples.csv" 2>&1 &
smi=\$!
python3 "$HERE/../probe.py" run "$BIN" --out "$OUT" --since sm89 $*
rc=\$?
kill \$smi
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE >"$OUT/smi-after.txt" 2>&1
exit \$rc
EOF
gpu-lease 1 --wait --max-min 20 -- bash "$OUT/gpu.sh"
