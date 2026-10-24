#!/usr/bin/env bash
# Where a served decode step's time goes: `served_step.py run` for each of ARMS (a spec's arm ids, comma-separated), one engine
# process an arm, one after another in one GPU lease on the leased GPU's NUMA node, each under nsys (NSYS=1: the timed reps
# only, graph nodes traced) or not (NSYS=0: the host records alone); after the lease, at low priority, each profile's SQLite
# export and `served_step.py analyze`. From the source tree, with Pearl-C's cubin built once and sent (no nvcc here):
#   research run --on vy-nebius-2 --project verity --campaign glm53-pearl-c --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=<id> [--env GPU_LEASE_ON=2,3] [--env ARMS=...] [--env POINT=b8-p512-mixed] [--env NSYS=0] \
#     --send numabind.py --send pearl_c_sm120.cubin -- bash benchmarks/pouw/pearl_c_vllm/served_step.sh
# SPEC (default llama31-8b-1gpu-pearl-c-whole-served), POINT, WARMUPS (2), REPS (3), MAX_MIN (the lease, 20). numabind.py is
# #3395's (benchmarks/hash/dither/numabind.py); NUMABIND names another copy, and without one the arms run unbound. Scratch
# goes to one mktemp -d directory (TMPDIR), removed at the end; everything kept is under OUT.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-${RESEARCH_RUN_DIR:?run this under research run}/out}; mkdir -p "$OUT"; export OUT
. benchmarks/harness/env.sh
IN=${RESEARCH_RUN_DIR:-.}/inputs
export PEARLC_CUBIN=${PEARLC_CUBIN:-$IN/pearl_c_sm120.cubin}
ARMS=${ARMS:-stock-fp8@compiled,stock-fp8,pouw:pearl-c-sm120-v1-h3}
SPEC=${SPEC:-llama31-8b-1gpu-pearl-c-whole-served}; POINT=${POINT:-b8-p512-mixed}; NSYS=${NSYS:-1}
export ARMS SPEC POINT NSYS
if [ "${1:-}" != gpu ]; then
  [ -f "$PEARLC_CUBIN" ] || { echo "served_step: no cubin at $PEARLC_CUBIN" >&2; exit 2; }
  # scratch (Inductor's per-process caches, nsys's temporaries) in one directory, removed when the run ends
  TMPDIR=$(mktemp -d); export TMPDIR; trap 'rm -rf "$TMPDIR"' EXIT
  sha256sum "$PEARLC_CUBIN" benchmarks/pouw/pearl_c_sm120/pearl_c_sm120.cu benchmarks/pouw/pearl_c_sm120/*.cuh \
    benchmarks/pouw/pearl_c_sm120/run.py > "$OUT/cubin.sha256"
  NUMABIND=${NUMABIND:-$IN/numabind.py}
  BIND=(); [ -f "$NUMABIND" ] && BIND=("$VENV/bin/python" "$NUMABIND" "$OUT/numa.json")
  LEASE=$(command -v gpu-lease || echo /workspace/pouw/infra/bin/gpu-lease)
  rc=0
  "$LEASE" 1 --wait --max-min "${MAX_MIN:-20}" --mem-gb 64 -- "${BIND[@]}" bash "$0" gpu || rc=$?
  if [ "$NSYS" = 1 ]; then
    for rep in "$OUT"/*/served.nsys-rep; do
      [ -f "$rep" ] || continue
      d=$(dirname "$rep")
      nice -n 19 nsys export --type sqlite --force-overwrite true -o "$d/served.sqlite" "$rep" > /dev/null
      nice -n 19 "$VENV/bin/python" "$HERE/served_step.py" analyze "$d/served.sqlite" --steps "$d/steps.json" \
        --out "$d/step-profile.json" || rc=$?
      rm -f "$d/served.sqlite"
    done
  fi
  exit "$rc"
fi
UUID=${GPU_LEASE_UUID:?run under gpu-lease}; UUID=${UUID%%,*}
nvidia-smi --query-gpu=uuid,name,compute_cap,clocks.sm,clocks.mem,driver_version --format=csv,noheader -i "$UUID" \
  > "$OUT/identity.csv"
grep -q "RTX PRO 6000 Blackwell" "$OUT/identity.csv" && grep -q ", 12.0," "$OUT/identity.csv" || { echo "identity" >&2; exit 3; }
rc=0
for arm in ${ARMS//,/ }; do
  d=$OUT/${arm//[:@]/_}; mkdir -p "$d"
  RUN=("$VENV/bin/python" "$HERE/served_step.py" run --spec "$SPEC" --arm "$arm" --point "$POINT"
       --warmups "${WARMUPS:-2}" --reps "${REPS:-3}" --out "$d")
  if [ "$NSYS" = 1 ]; then
    nsys profile -o "$d/served" -f true -t cuda,nvtx -s none --cpuctxsw=none --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi --capture-range-end=stop "${RUN[@]}" > "$d/log.txt" 2>&1 || rc=$?
  else
    "${RUN[@]}" > "$d/log.txt" 2>&1 || rc=$?
  fi
  tail -n 1 "$d/log.txt"
done
exit "$rc"
