#!/usr/bin/env bash
# Where a served decode step's time goes: `served_step.py run` for each of ARMS (a spec's arm ids, comma-separated), one engine
# process an arm, one after another in one GPU lease on the leased GPU's NUMA node, each under nsys (NSYS=1: the timed reps
# only, graph nodes traced) or not (NSYS=0: the host records alone); after the lease, at low priority, each profile's SQLite
# export, `served_step.py analyze` and the arms' `served_step.py table` (breakdown.md). From the source tree, with
# Pearl-C's cubin built once and sent (no nvcc here):
#   research run --on MACHINE --project verity --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=LANE [--env GPUS=2,3] [--env ARMS=...] [--env POINT=b8-p512-mixed] [--env NSYS=0] \
#     --send numabind.py --send pearl_c_sm120.cubin -- bash benchmarks/pouw/pearl_c_vllm/served_step.sh
# An ARMS entry ID+VAR=VALUE[+VAR=VALUE] runs arm ID with those variables set, in its own directory (ID_VAR_VALUE): A/B
# rows in one lease, e.g. pouw-pearl-c-whole,pouw-pearl-c-whole+PEARLC_SPLIT=0; @IN@ in a value is the run's inputs, so
# pouw-pearl-c-whole+PEARLC_CUBIN=@IN@/before.cubin runs another sent cubin.
# PEARLC_KERNEL: the kernel directory (default benchmarks/pouw/pearl_c_sm120, on an RTX PRO 6000;
# benchmarks/pouw/pearl_c_sm90 on an H100 or H200, its cubin sent as pearl_c_sm90.cubin, SPEC's default
# llama31-8b-1gpu-pearl-c-whole-served-h200 and ARMS' stock-fp8@compiled,pouw-pearl-c-whole); the leased GPU must be its.
# GPUS: the GPUs the lease may take (gpu-lease --on), SPEC (default llama31-8b-1gpu-pearl-c-whole-served), POINT,
# WARMUPS (2), REPS (3), MAX_MIN (the lease, 20), MEM_GB (the lease's host memory, 16);
# GPU_LEASE_TIMED=1 for rows taken inside a booked clean window
# (gpu-lease --timed). The node's pinned-vLLM Python is env.sh's VENV. numabind.py is
# benchmarks/hash/dither/numabind.py; NUMABIND names another copy, and without one the arms run unbound. The node's load
# (loadavg and the leases) before and after each arm is OUT/load.txt. Scratch goes to one mktemp -d directory (TMPDIR),
# removed at the end; everything kept is under OUT.
# `served_step.sh post DIR` (no GPU) analyzes another run's OUT again with this tree's code, into this run's OUT; ANALYZE=0
# leaves the reports unanalyzed, for a node whose CPUs are for others (post them where nsys runs).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-${RESEARCH_RUN_DIR:?run this under research run}/out}; mkdir -p "$OUT"; export OUT
. benchmarks/harness/env.sh
IN=${RESEARCH_RUN_DIR:-.}/inputs
KERNEL=$(cd "${PEARLC_KERNEL:-benchmarks/pouw/pearl_c_sm120}" && pwd); K=$(basename "$KERNEL")
export PEARLC_KERNEL=$KERNEL PEARLC_CUBIN=${PEARLC_CUBIN:-$IN/$K.cubin}
case $K in
  pearl_c_sm90) GPU_NAME="H100|H200"; GPU_CC=9.0
    ARMS=${ARMS:-stock-fp8@compiled,pouw-pearl-c-whole}; SPEC=${SPEC:-llama31-8b-1gpu-pearl-c-whole-served-h200} ;;
  *) GPU_NAME="RTX PRO 6000 Blackwell"; GPU_CC=12.0
    ARMS=${ARMS:-stock-fp8@compiled,stock-fp8,pouw:pearl-c-sm120-v1-h3}; SPEC=${SPEC:-llama31-8b-1gpu-pearl-c-whole-served} ;;
esac
POINT=${POINT:-b8-p512-mixed}; NSYS=${NSYS:-1}
export ARMS SPEC POINT NSYS GPU_NAME GPU_CC
LEASE=$(command -v gpu-lease || echo /workspace/pouw/infra/bin/gpu-lease)
load() { { echo "== $1 $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"; "$LEASE" status 2>&1 || true; } >> "$OUT/load.txt"; }

analyze_all() {   # SRC's arm profiles analyzed into OUT's arm directories, then the table; returns the first failure
  local src=$1 r=0 d a
  for rep in "$src"/*/served.nsys-rep; do
    [ -f "$rep" ] || continue
    d=$(dirname "$rep"); a=$OUT/$(basename "$d"); mkdir -p "$a"
    [ "$d" = "$a" ] || cp "$d/reps.json" "$d/steps.json" "$a/" 2>/dev/null || true
    nice -n 19 nsys export --type sqlite --force-overwrite true -o "$TMPDIR/served.sqlite" "$rep" > /dev/null
    nice -n 19 "$VENV/bin/python" "$HERE/served_step.py" analyze "$TMPDIR/served.sqlite" --steps "$d/steps.json" \
      --out "$a/step-profile.json" || r=$?
    rm -f "$TMPDIR/served.sqlite"
  done
  ls "$OUT"/*/step-profile.json > /dev/null 2>&1 &&
    { "$VENV/bin/python" "$HERE/served_step.py" table "$OUT"/*/ --out "$OUT/breakdown.json" || r=$?; }
  return "$r"
}

if [ "${1:-}" = post ]; then
  TMPDIR=$(mktemp -d); export TMPDIR; trap 'rm -rf "$TMPDIR"' EXIT
  analyze_all "${2:?served_step.sh post DIR: the OUT of another run}"
  exit
fi
if [ "${1:-}" != gpu ]; then
  [ -f "$PEARLC_CUBIN" ] || { echo "served_step: no cubin at $PEARLC_CUBIN" >&2; exit 2; }
  # scratch (Inductor's per-process caches, nsys's temporaries) in one directory, removed when the run ends
  TMPDIR=$(mktemp -d); export TMPDIR; trap 'rm -rf "$TMPDIR"' EXIT
  SHARED=(); [ "$K" = pearl_c_sm120 ] || SHARED=(benchmarks/pouw/pearl_c_sm120/*.cuh)
  sha256sum "$PEARLC_CUBIN" "$KERNEL/$K.cu" "$KERNEL"/*.cuh "${SHARED[@]}" "$KERNEL/run.py" > "$OUT/cubin.sha256"
  NUMABIND=${NUMABIND:-$IN/numabind.py}
  BIND=(); [ -f "$NUMABIND" ] && BIND=("$VENV/bin/python" "$NUMABIND" "$OUT/numa.json")
  ON=(); [ -n "${GPUS:-}" ] && ON=(--on "$GPUS")
  rc=0
  "$LEASE" 1 --wait "${ON[@]}" --max-min "${MAX_MIN:-20}" --mem-gb "${MEM_GB:-16}" -- "${BIND[@]}" bash "$0" gpu || rc=$?
  if [ "$NSYS" = 1 ] && [ "${ANALYZE:-1}" = 1 ]; then
    a=0; analyze_all "$OUT" || a=$?; [ "$rc" != 0 ] || rc=$a
  fi
  exit "$rc"
fi
UUID=${GPU_LEASE_UUID:?run under gpu-lease}; UUID=${UUID%%,*}
nvidia-smi --query-gpu=uuid,name,compute_cap,clocks.sm,clocks.mem,driver_version --format=csv,noheader -i "$UUID" \
  > "$OUT/identity.csv"
grep -Eq "$GPU_NAME" "$OUT/identity.csv" && grep -q ", $GPU_CC," "$OUT/identity.csv" || { echo "identity: not $K's GPU" >&2; exit 3; }
rc=0
for entry in ${ARMS//,/ }; do
  arm=${entry%%+*}; SET=(); [ "$arm" = "$entry" ] || IFS=+ read -ra SET <<< "${entry#*+}"
  SET=("${SET[@]//@IN@/$IN}")
  d=$OUT/${entry//[:@+=\/]/_}; mkdir -p "$d"; load "$entry"
  RUN=(env "${SET[@]}" "$VENV/bin/python" "$HERE/served_step.py" run --spec "$SPEC" --arm "$arm" --point "$POINT"
       --warmups "${WARMUPS:-2}" --reps "${REPS:-3}" --out "$d")
  if [ "$NSYS" = 1 ]; then
    nsys profile -o "$d/served" -f true -t cuda,nvtx -s none --cpuctxsw=none --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi --capture-range-end=stop "${RUN[@]}" > "$d/log.txt" 2>&1 || rc=$?
  else
    "${RUN[@]}" > "$d/log.txt" 2>&1 || rc=$?
  fi
  tail -n 1 "$d/log.txt"
done
load end
exit "$rc"
