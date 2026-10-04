#!/usr/bin/env bash
# One salt-to-deadline window (window.py) per schedule inside one GPU lease, the cards' cuBLASLt FP8 baseline at the window's
# shape before and after it in the same lease (harness bench.py), and after the lease the CPU audit (audit.py) and the
# harness's verify step on the arm's own transcript.
#   bash benchmarks/pouw/exhaustion/window.sh MODE SHIP_TAR HARNESS_BUILD [DURATION_S]
#     MODE rehearsal: gpu-lease 1 --wait --preemptible --max-min MAX_MIN (untimed)
#     MODE timed:     gpu-lease 8 --wait --timed --max-min MAX_MIN
#   SHIP_TAR is ship.sh's ship.tar; HARNESS_BUILD the harness build jobs/env.sh takes (node2-gpus.md beside it unless
#   SERVER_MD names the table).  env: SCHEDULES (default serial; each gets its own window), DRAWS (60), JOBS (the audit's
#   processes, default nproc), MAX_MIN (45), SCRATCH_ROOT (where the rows and the ship are unpacked, default /tmp).
# Writes under OUT (default $RESEARCH_RUN_DIR/out): ship.sha256; bench-before/card<g>/ (bench.py with the Pearl-C arm: the
# baselines' and the arm's times, its gates and transcript); <schedule>/ (window.py's OUT and audit.json); bench-after/card<g>/
# (the baselines alone); verify-before/card<g>.json.  Exit: the first window's audit's, else the first failure.
set -uo pipefail
exec < /dev/null
mode=${1:?rehearsal|timed}; ship_tar=${2:?SHIP_TAR}; build=${3:?HARNESS_BUILD}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
export HERE ROOT DURATION=${4:-1800} OUT=${OUT:-$RESEARCH_RUN_DIR/out} SCHEDULES=${SCHEDULES:-serial} DRAWS=${DRAWS:-60}
export MD=${SERVER_MD:-$build/node2-gpus.md} LABEL=m8192-n8192-k8192 SHAPE="8192 8192 8192"
JOBS=${JOBS:-$(nproc)}; MAX_MIN=${MAX_MIN:-45}
case $mode in
  rehearsal) LEASE=(gpu-lease 1 --wait --preemptible --max-min "$MAX_MIN") ;;
  timed) LEASE=(gpu-lease 8 --wait --timed --max-min "$MAX_MIN") ;;
  *) echo "MODE $mode: rehearsal or timed" >&2; exit 2 ;;
esac
SCR=$(mktemp -d -p "${SCRATCH_ROOT:-/tmp}" exhaustion.XXXXXX)
trap 'rm -rf "$SCR"' EXIT
mkdir -p "$OUT"
tar -xf "$ship_tar" -C "$SCR" || exit 1
export PEARLC_SHIP=$SCR/ship PEARLC_ROWS=$SCR/rows POUW_SASS_GATE_CACHE=${POUW_SASS_GATE_CACHE:-$SCR/sass-gate-cache}
mkdir -p "$PEARLC_ROWS" "$POUW_SASS_GATE_CACHE"
. "$ROOT/benchmarks/pouw/harness/jobs/env.sh" "$build" || exit 1
export PY
sha256sum "$ship_tar" "$PEARLC_SHIP/pearl_c_sm120.cubin" | tee "$OUT/ship.sha256"

cat > "$SCR/lease.sh" <<'EOF'
set -uo pipefail
echo "lease $GPU_LEASE_UUID cards $CUDA_VISIBLE_DEVICES from $(date -u +%FT%T.%3NZ)"
IFS=, read -ra cards <<< "$CUDA_VISIBLE_DEVICES"
bench() {   # bench WHEN ARGS...: one bench.py per card, all at once
  local when=$1 g p rc=0; shift; local pids=()
  for g in "${cards[@]}"; do
    mkdir -p "$OUT/bench-$when/card$g" "$PEARLC_ROWS/card$g"
    CUDA_VISIBLE_DEVICES=$g PEARLC_ROWS=$PEARLC_ROWS/card$g "$PY" "$ROOT/benchmarks/pouw/harness/bench.py" --gpu lease \
      --server-md "$MD" --clock-label locked-2100 --shapes $LABEL --families fp8-e4m3 "$@" \
      --out "$OUT/bench-$when/card$g/bench.json" > "$OUT/bench-$when/card$g/log.txt" 2>&1 < /dev/null &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "bench $when rc=$rc at $(date -u +%FT%T.%3NZ)"
}
bench before --arm-path "$ROOT/benchmarks/pouw/pearl_c_sm120" --arm pearlc_arm:PearlCSm120
for s in $SCHEDULES; do
  "$PY" "$HERE/window.py" run --out "$OUT/$s" --server-md "$MD" --duration "$DURATION" --draws "$DRAWS" --schedule "$s" \
    --shape $SHAPE < /dev/null
  echo "window $s rc=$? at $(date -u +%FT%T.%3NZ)"
done
bench after
EOF
"${LEASE[@]}" -- timeout $((MAX_MIN * 60 - 60)) bash "$SCR/lease.sh" < /dev/null
echo "lease rc=$? at $(date -u +%FT%T.%3NZ)"

rc=
for s in $SCHEDULES; do
  "$PY" "$HERE/audit.py" "$OUT/$s" --bench-before "$OUT/bench-before" --bench-after "$OUT/bench-after" --jobs "$JOBS" < /dev/null
  a=$?; echo "audit $s rc=$a"; rc=${rc:-$a}
done
mkdir -p "$OUT/verify-before"
for b in "$OUT"/bench-before/card*/bench.json; do
  [ -f "$b" ] || continue
  g=$(basename "$(dirname "$b")")
  "$PY" "$ROOT/benchmarks/pouw/harness/verify.py" "$b" --out "$OUT/verify-before/$g.json" --no-ledger > "$OUT/verify-before/$g.log" 2>&1 < /dev/null
  echo "verify $g rc=$?"
done
exit "${rc:-1}"
