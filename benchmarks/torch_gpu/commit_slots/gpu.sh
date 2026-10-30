#!/usr/bin/env bash
# commit_slots on one leased GPU (cwd = the source tree's root): outside the lease, the engine's build (sha512_rows; nvcc,
# for this GPU) and its host twin's; under one lease of at most LEASE_MIN minutes, the GPU tests, check.py (the roots
# against the CPU reference's, the frozen layout at 4,096 to 20,480 slots on sampled slots) and bench.py (GB/s and share of
# the step). Python is PY (CUDA torch), with the tree on PYTHONPATH.
#
#   OUT=DIR bash benchmarks/torch_gpu/commit_slots/gpu.sh
#
# Needs a CUDA GPU (an H200 for the frozen layout), CUDA's nvcc and `gpu-lease` on PATH.
# STEPS (default "tests check bench"; a failing tests step ends the lease; "step" runs step.py's time on STEP_ARGS, layout
# files and options; STEP_WRAP, a command it runs under, e.g. nsys profile ...), CHECK_ARGS, BENCH_ARGS, STEP_ARGS,
# LEASE_MIN (default 25), PY,
# LEASE_END (a UTC time `date -d` reads, e.g. YYYY-MM-DDTHH:MMZ, on a GPU lent until then: a lease that starts too late
# to end by it runs nothing).
set -uo pipefail
O=${OUT:?}
ROOT=$PWD D=benchmarks/torch_gpu/commit_slots
for v in /workspace/jobs/venv312 /workspace/cp/venv312; do [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3; done
PY=${PY:-python3}
TV=$($PY -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null)
for c in "/usr/local/cuda-$TV" /usr/local/cuda /usr/local/cuda-13.0; do [ -x "$c/bin/nvcc" ] && { CUDA=$c; break; }; done
export PATH=${CUDA:-/usr/local/cuda}/bin:$PATH PYTHONPATH=$ROOT:$ROOT/$D:$ROOT/tools/config PYTHONDONTWRITEBYTECODE=1
git rev-parse HEAD > "$O/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$O/commit.txt"
{ echo "python $PY"; $PY -c 'import sys, torch, numpy; print(sys.version.split()[0], torch.__version__, torch.version.cuda, numpy.__version__)'
  echo "cuda ${CUDA:-none}"; nvcc --version | tail -2; nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv; } \
  > "$O/env.txt" 2>&1
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
$PY -c "import slots_gpu as G; print(G.build('$CAP'))" > "$O/build.txt" 2>&1
echo "build rc $?" > "$O/rc.txt"
export O PY D ROOT STEPS=${STEPS:-tests check bench} CHECK_ARGS=${CHECK_ARGS:-} BENCH_ARGS=${BENCH_ARGS:-} STEP_ARGS=${STEP_ARGS:-}
export STEP_WRAP=${STEP_WRAP:-}
export LEASE_MIN=${LEASE_MIN:-25} LEASE_END=${LEASE_END:-}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "$LEASE_MIN" -- bash -c '
  if [ -n "$LEASE_END" ] && [ $(( $(date -u +%s) + 60 * LEASE_MIN )) -gt "$(date -u -d "$LEASE_END" +%s)" ]; then
    echo "the lease started at $(date -u +%H:%MZ), too late to end by LEASE_END=$LEASE_END: nothing runs" >> "$O/rc.txt"
    exit 0
  fi
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit --format=csv \
    -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  for s in $STEPS; do
    case $s in
      tests) (cd "$D" && $PY -m pytest -q -p no:cacheprovider tests) > "$O/tests.txt" 2>&1 ;;
      check) $PY "$D/check.py" --out "$O" ${CHECK_ARGS:-} > "$O/check.jsonl" 2> "$O/check.err" ;;
      bench) $PY "$D/bench.py" --out "$O" ${BENCH_ARGS:-} > "$O/bench.jsonl" 2> "$O/bench.err" ;;
      step) ${STEP_WRAP:-} $PY "$D/step.py" time ${STEP_ARGS:-} --out "$O" > "$O/step.jsonl" 2> "$O/step.err" ;;
    esac
    rc=$?
    echo "$s rc $rc" >> "$O/rc.txt"
    if [ "$s" = tests ] && [ "$rc" != 0 ]; then echo "the tests failed: nothing after them runs" >> "$O/rc.txt"; break; fi
  done'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
tail -3 "$O/tests.txt" 2>/dev/null
tail -1 "$O/check.jsonl" 2>/dev/null
cat "$O/bench.jsonl" 2>/dev/null | cut -c1-600
cut -c1-600 "$O/step.jsonl" 2>/dev/null
