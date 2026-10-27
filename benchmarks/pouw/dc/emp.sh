#!/usr/bin/env bash
# The H200's empirical check: build sm_90a probes on a CPU node, then time them on the H200 under one
# gpu-lease, so no CPU phase runs on the shared H200 node.
#
#   research run --on auto --wait -- bash benchmarks/pouw/dc/emp.sh build SRC...
#       builds each SRC for sm_90a, gates every probe's SASS (probe.py gate) and writes $OUT/bins.tgz, $OUT/sass/,
#       gate-build.txt and gate-build.json. A source that fails to build is named and skipped; the rest still build.
#   research run --on vy-dc-h200 --project verity --campaign pouw --source <tree> --cwd source --send bins.tgz \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/dc/emp.sh time [REPS]
#       unpacks the build's bins.tgz (from $RESEARCH_RUN_DIR/inputs/), records the clocks (A1: nvidia-smi -q -d CLOCK
#       before and after, clocks sampled every 200 ms under the lease) and runs probe.py run on every binary (gate, the
#       poisoned sink and its control, then the interleaved timing) into $OUT/<binary>/.
# Exit: 0 done, 1 nothing built or a gate verdict not the declared one, 2 no nvcc or no inputs.
set -uo pipefail
MODE=${1:?usage: emp.sh build SRC... | time [REPS]}
shift
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
export PYTHONDONTWRITEBYTECODE=1

case "$MODE" in
build)
  NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
  [ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "emp.sh: no nvcc (CUDA 13.0 builds sm_90a)" >&2; exit 2; }
  export NVCC PATH="$(dirname "$NVCC"):$PATH"
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  mkdir -p "$WORK/bin" "$OUT/sass"
  "$NVCC" --version >"$OUT/nvcc.txt" 2>&1
  for src in "$@"; do
    python3 "$PROBE" build --arch sm_90a --out "$WORK/bin" "$src" >>"$OUT/build.log" 2>&1 \
      || echo "emp.sh: build failed: $src (build.log)" | tee -a "$OUT/build.log" >&2
  done
  shopt -s nullglob
  bins=("$WORK"/bin/*)
  [ ${#bins[@]} -gt 0 ] || { echo "emp.sh: nothing built" >&2; exit 1; }
  for b in "${bins[@]}"; do cuobjdump -sass "$b" >"$OUT/sass/$(basename "$b").sass" 2>&1; done
  python3 "$PROBE" gate "${bins[@]}" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
  rc=$?
  grep -c '^PASS' "$OUT/gate-build.txt" | xargs echo "emp.sh: gate PASS"
  grep -A3 '^REFUSED' "$OUT/gate-build.txt" | grep -v 'as declared' >&2
  tar -C "$WORK/bin" -czf "$OUT/bins.tgz" .
  [ $rc -eq 0 ] || exit 1
  ;;
time)
  REPS=${1:-5}
  IN=${RESEARCH_RUN_DIR:?run under research run}/inputs
  [ -f "$IN/bins.tgz" ] || { echo "emp.sh: no inputs/bins.tgz (pass the build's with --send)" >&2; exit 2; }
  if [ "${EMP_LEASED:-}" != 1 ]; then
    WORK=$(mktemp -d)
    trap 'rm -rf "$WORK"' EXIT
    tar -C "$WORK" -xzf "$IN/bins.tgz"
    mkdir -p "$OUT/env"
    lease=()
    command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 25 --)
    [ ${#lease[@]} -eq 0 ] && echo "emp.sh: no gpu-lease on this node; running on its GPU directly" >&2
    "${lease[@]}" env EMP_LEASED=1 WORK="$WORK" REPS="$REPS" bash "${BASH_SOURCE[0]}" time "$REPS"
    exit $?
  fi
  dev=()
  [ -n "${GPU_LEASE_UUID:-}" ] && dev=(-i "$GPU_LEASE_UUID")
  nvidia-smi "${dev[@]}" -q -d CLOCK,PERFORMANCE,POWER >"$OUT/env/clocks-before.txt" 2>&1
  nvidia-smi "${dev[@]}" --query-gpu=name,uuid,compute_cap,clocks.max.sm,clocks.max.mem,driver_version \
    --format=csv >"$OUT/env/device.csv" 2>&1
  nvidia-smi "${dev[@]}" --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 200 >"$OUT/env/smi.csv" 2>&1 &
  smi_pid=$!
  trap 'kill "$smi_pid" 2>/dev/null' EXIT
  rc=0
  for b in "$WORK"/*; do
    [ -x "$b" ] || continue
    python3 "$PROBE" run "$b" --out "$OUT/$(basename "$b")" --reps "$REPS" || { echo "emp.sh: probe.py run $(basename "$b") failed" >&2; rc=1; }
  done
  nvidia-smi "${dev[@]}" -q -d CLOCK,PERFORMANCE,POWER >"$OUT/env/clocks-after.txt" 2>&1
  exit $rc
  ;;
*)
  echo "emp.sh: build or time, not $MODE" >&2
  exit 2
  ;;
esac
