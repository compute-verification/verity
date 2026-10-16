#!/usr/bin/env bash
# Mixed-stream issue ceiling + CE7, on one RTX PRO 6000 (sm_120). Run from a clean commit under `research run --cwd clone`;
# outputs go to $OUT. Sections: `issue` (Q1), `ce7` (Q2); default both.
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd clone --custody-ttl 12h \
#     --env GPU_LEASE_WHO=issue -- bash benchmarks/pouw/issue_ceiling/run.sh issue ce7
#
# Each binary is built and its SASS dumped (cuobjdump) outside any lease; the timed GPU work runs under
# `gpu-lease 1 --wait --max-min 20`, with nvidia-smi sampling the leased GPU's clock/power every 200 ms.
set -uo pipefail
OUT=${OUT:?run under research run}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc)
CUOBJDUMP=$(dirname "$NVCC")/cuobjdump
SECTIONS=${*:-issue ce7}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/sass"
echo "{\"run_id\":\"${RESEARCH_RUN_ID:-$(basename "${RESEARCH_RUN_DIR:-unknown}")}\",\"host\":\"$(hostname)\",\"sections\":\"$SECTIONS\",\"start_utc\":\"$(date -u +%FT%TZ)\",\"nvcc\":\"$($NVCC --version | tail -2 | head -1)\"}" >"$OUT/meta.json"

build() {  # build NAME [extra flags]
  local n=$1; shift
  if "$NVCC" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -I"$SRC" -o "$WORK/$n" "$SRC/$n.cu" "$@" \
      >"$OUT/build-$n.log" 2>&1; then
    "$CUOBJDUMP" -sass "$WORK/$n" >"$OUT/sass/$n.sass" 2>&1 || true
    echo "built $n" >>"$OUT/sections.log"
  else
    echo "{\"section\":\"$n\",\"build\":\"failed\"}" >>"$OUT/failures.jsonl"
    echo "BUILD FAILED $n" >>"$OUT/sections.log"
    tail -30 "$OUT/build-$n.log" >&2
  fi
}

gpu_bins=()
for s in $SECTIONS; do
  case $s in
    issue) build issue_ceiling; gpu_bins+=(issue_ceiling) ;;
    sched) build issue_sched; gpu_bins+=(issue_sched) ;;
    ce7) build ce7 -lcublasLt -lcublas; gpu_bins+=(ce7) ;;
    *) echo "unknown section $s" >&2 ;;
  esac
done

if [ ${#gpu_bins[@]} -gt 0 ]; then
  cat >"$WORK/gpu.sh" <<EOF
set -uo pipefail
U=\$GPU_LEASE_UUID
echo "{\"gpu_uuid\":\"\$U\",\"cuda_visible_devices\":\"\${CUDA_VISIBLE_DEVICES:-}\",\"lease_start_utc\":\"\$(date -u +%FT%TZ)\"}" >"$OUT/lease.json"
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$U" >"$OUT/smi-before.txt" 2>&1
nvidia-smi --id="\$U" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active,utilization.gpu --format=csv,noheader -lms 200 >"$OUT/smi-samples.csv" 2>&1 &
SMI=\$!
for b in ${gpu_bins[*]}; do
  [ -x "$WORK/\$b" ] || continue
  echo "\$(date -u +%FT%T.%NZ) start \$b" >>"$OUT/sections.log"
  timeout 1000 "$WORK/\$b" >"$OUT/\$b.jsonl" 2>"$OUT/\$b.err"
  echo "\$(date -u +%FT%T.%NZ) end \$b rc=\$?" >>"$OUT/sections.log"
done
kill \$SMI 2>/dev/null
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$U" >"$OUT/smi-after.txt" 2>&1
EOF
  gpu-lease 1 --wait --max-min 20 -- bash "$WORK/gpu.sh"
  echo "gpu lease rc=$?" >>"$OUT/sections.log"
fi

python3 "$SRC/issue_ceiling.py" "$OUT" >"$OUT/combine.log" 2>&1 || echo "issue_ceiling.py rc=$?" >>"$OUT/sections.log"
[ -f "$OUT/issue_sched.jsonl" ] && { python3 "$SRC/issue_sched.py" "$OUT" >"$OUT/combine-sched.log" 2>&1 || echo "issue_sched.py rc=$?" >>"$OUT/sections.log"; }
ls -la "$OUT" >&2
cat "$OUT/sections.log" >&2
