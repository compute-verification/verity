#!/usr/bin/env bash
# The H200 job for the form escape and honest's folding loops, in two runs so no CPU phase runs on the shared H200 node.
# From a pushed commit:
#
#   research run --on auto --project verity --campaign eps1 --source <tree> --cwd source -- bash benchmarks/pouw/rt_hopper/rs.sh build
#       builds sm90_rs.cu (or --env RS_SRC=sm90_forms.cu, the s8-exact siblings) for sm_90a and gates every probe's
#       SASS (probe.py gate), and runs uni64.sh into $OUT/uni64/.
#       Outputs: nvcc.txt, build.log, sass/, gate-build.{txt,json}, uni64/ (uni64.sass, uniform.txt), bins.tgz.
#   research run --on vy-dc-h200 --project verity --campaign eps1 --source <tree> --cwd source --send bins.tgz \
#     --custody-ttl 12h --timeout 30m --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/rt_hopper/rs.sh time [REPS]
#       unpacks the build's bins.tgz (from $RESEARCH_RUN_DIR/inputs/), times every probe under one gpu-lease (clocks
#       sampled every 200 ms) into probes/ (rows.jsonl, gate.json, times.jsonl), then rs.py into rs.json.
#       Outputs: env/ (device.csv, clocks-{before,after}.txt, smi.csv), probes/, rs.json.
# Exit: 0 done, 1 build or gate refused (build) or a probe failed (time), 2 no nvcc, no inputs or wrong device.
set -uo pipefail
MODE=${1:?usage: rs.sh build | time [REPS]}
shift
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
export PYTHONDONTWRITEBYTECODE=1

case "$MODE" in
build)
  NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
  [ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "rs.sh: no nvcc" >&2; exit 2; }
  export NVCC PATH="$(dirname "$NVCC"):$PATH"
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  mkdir -p "$WORK/bin" "$OUT/sass" "$OUT/uni64"
  "$NVCC" --version >"$OUT/nvcc.txt" 2>&1
  OUT="$OUT/uni64" NVCC="$NVCC" bash "$HERE/uni64.sh" >"$OUT/uni64/uni64.log" 2>&1 \
    || echo "rs.sh: uni64.sh failed (uni64/uni64.log)" >&2
  python3 "$PROBE" build --arch sm_90a --out "$WORK/bin" "$HERE/${RS_SRC:-sm90_rs.cu}" >"$WORK/built" 2>"$OUT/build.log" \
    || { cat "$OUT/build.log" >&2; exit 1; }
  mapfile -t bins <"$WORK/built"
  for b in "${bins[@]}"; do cuobjdump -sass "$b" >"$OUT/sass/$(basename "$b").sass" 2>&1; done
  python3 "$PROBE" gate "${bins[@]}" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
  rc=$?
  grep -c '^PASS' "$OUT/gate-build.txt" | xargs echo "rs.sh: gate-build PASS"
  [ $rc -eq 0 ] || { grep -A2 '^REFUSED' "$OUT/gate-build.txt" >&2; exit 1; }
  tar -C "$WORK/bin" -czf "$OUT/bins.tgz" .
  ;;
time)
  REPS=${1:-10}
  IN=${RESEARCH_RUN_DIR:?run under research run}/inputs
  [ -f "$IN/bins.tgz" ] || { echo "rs.sh: no inputs/bins.tgz (pass the build's with --send)" >&2; exit 2; }
  if [ "${RS_LEASED:-}" != 1 ]; then
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    [ "${cap%%.*}" = 9 ] || { echo "rs.sh: needs compute capability 9.x, the device reports '$cap'" >&2; exit 2; }
    WORK=$(mktemp -d)
    trap 'rm -rf "$WORK"' EXIT
    tar -C "$WORK" -xzf "$IN/bins.tgz"
    mkdir -p "$OUT/env" "$OUT/probes"
    lease=()
    command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 15 --)
    [ ${#lease[@]} -eq 0 ] && echo "rs.sh: no gpu-lease on this node; running on its GPU directly" >&2
    "${lease[@]}" env RS_LEASED=1 WORK="$WORK" bash "${BASH_SOURCE[0]}" time "$REPS"
    rc=$?
    python3 "$HERE/rs.py" "$OUT" --json "$OUT/rs.json"
    exit $rc
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
    python3 "$PROBE" run "$b" --out "$OUT/probes" --reps "$REPS" || { echo "rs.sh: probe.py run $(basename "$b") failed" >&2; rc=1; }
  done
  nvidia-smi "${dev[@]}" -q -d CLOCK,PERFORMANCE,POWER >"$OUT/env/clocks-after.txt" 2>&1
  exit $rc
  ;;
*)
  echo "rs.sh: build or time, not $MODE" >&2
  exit 2
  ;;
esac
