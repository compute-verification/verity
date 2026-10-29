#!/usr/bin/env bash
# The engine on one leased GPU (cwd = the source tree's root): outside the lease, the kernels' build (nvcc, for this GPU) with
# ptxas's register report and each row kernel's opcode counts; under one lease of at most LEASE_MIN minutes, the GPU tests,
# check.py (every variant and the H200 layer against the CPU reference) and bench.py (the variants, then whole steps).
#
#   OUT=DIR bash benchmarks/torch_gpu/sha512_rows/gpu.sh
#
# Needs a CUDA GPU (an H200 for the H200 layer), CUDA torch, CUDA's nvcc and `gpu-lease` on PATH.
# STEPS (default "tests check bench"; the bench times steps only on a variant check.json found bit-exact), BENCH_ARGS,
# SHA512_ROWS_V (the default variant), LEASE_MIN (default 25), PY.
set -uo pipefail
O=${OUT:?}
ROOT=$PWD D=benchmarks/torch_gpu/sha512_rows
for v in /workspace/jobs/venv312 /workspace/cp/venv312; do [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3; done
PY=${PY:-python3}
TV=$($PY -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null)
for c in "/usr/local/cuda-$TV" /usr/local/cuda /usr/local/cuda-13.0; do [ -x "$c/bin/nvcc" ] && { CUDA=$c; break; }; done
export PATH=${CUDA:-/usr/local/cuda}/bin:$PATH PYTHONPATH=$ROOT:$ROOT/$D:$ROOT/tools/config PYTHONDONTWRITEBYTECODE=1
export SHA512_ROWS_BUILD=${SHA512_ROWS_BUILD:-$HOME/.cache/verity/sha512_rows}
git rev-parse HEAD > "$O/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$O/commit.txt"
{ echo "python $PY"; $PY -c 'import sys, torch, numpy; print(sys.version.split()[0], torch.__version__, torch.version.cuda, numpy.__version__)'
  echo "cuda ${CUDA:-none}"; nvcc --version | tail -2; nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv; } \
  > "$O/env.txt" 2>&1
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
case $ARCH in 90|100|120) ARCH=${ARCH}a ;; esac
$PY -c "import engine as E; L = E.device_lib('$ARCH'); print(L.path); E.host_lib()" > "$O/build.txt" 2>&1
echo "build rc $?" > "$O/rc.txt"
LIB=$(head -1 "$O/build.txt")
cp "${LIB%.so}.log" "$O/ptxas.txt" 2>/dev/null
$PY "$D/sass.py" "$LIB" > "$O/sass.json" 2> "$O/sass.err"
export O PY D ROOT STEPS=${STEPS:-tests check bench} BENCH_ARGS=${BENCH_ARGS:-}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${LEASE_MIN:-25}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit --format=csv \
    -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  for s in $STEPS; do
    case $s in
      tests) (cd "$D" && $PY -m pytest -q -p no:cacheprovider tests) > "$O/tests.txt" 2>&1 ;;
      check) $PY "$D/check.py" --out "$O" > "$O/check.out" 2> "$O/check.err" ;;
      bench) $PY "$D/bench.py" --out "$O" --check "$O/check.json" ${BENCH_ARGS:-} > "$O/bench.jsonl" 2> "$O/bench.err" ;;
    esac
    echo "$s rc $?" >> "$O/rc.txt"
  done
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
grep -E 'registers|spill' "$O/ptxas.txt" 2>/dev/null | head -40
tail -3 "$O/tests.txt" 2>/dev/null
cat "$O/check.out" 2>/dev/null
cut -c1-400 "$O/bench.jsonl" 2>/dev/null
