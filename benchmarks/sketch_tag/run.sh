#!/usr/bin/env bash
# Sketch-then-hash's benchmark on one GPU (cwd = the source tree): builds sketch.cu outside the lease at 60 and 56 payload
# bits a word (dense, split), then, under one gpu-lease 1 --wait --max-min MAXMIN (default 25), bench.py's gates and
# timings and Nsight Compute on the fused kernel. Results go to $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 48 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-sketch-tag \
#     -- taskset -c 48-95 bash benchmarks/sketch_tag/run.sh
#
# ENV_DIR holds the shared torch venv (benchmarks/pous/pod.sh's WORK). BENCH_ARGS passes through to bench.py; NCU=0 skips
# the profiles; LIBSET (default "dense split") names the builds (dense1, split1: one sketch accumulator a sign block).
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
NCU_BIN=${NCU_BIN:-/usr/local/cuda-13.0/bin/ncu}
SK=benchmarks/sketch_tag
export LIBSET=${LIBSET:-dense split}
B=$(mktemp -d /tmp/sketch-tag.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v -DWC_KMAX=3 -DSK_KMAX=3)
declare -A DEFS=([dense]="-DWC_PB=60" [split]="-DWC_PB=56" [dense1]="-DWC_PB=60 -DSK_CHAINS=1"
                 [split1]="-DWC_PB=56 -DSK_CHAINS=1" [dense4]="-DWC_PB=60 -DSK_CHAINS=4"
                 [dense16]="-DWC_PB=60 -DSK_CHAINS=16")
declare -A PIDS=()
for lib in $LIBSET; do
  # shellcheck disable=SC2086
  nvcc "${FLAGS[@]}" ${DEFS[$lib]} -o "$B/sk-$lib.so" $SK/sketch.cu > "$O/sk-$lib.ptxas" 2>&1 & PIDS[$lib]=$!
done
for lib in $LIBSET; do
  wait "${PIDS[$lib]}" || { tail -30 "$O/sk-$lib.ptxas"; exit 4; }
done
for lib in $LIBSET; do
  cuobjdump -sass "$B/sk-$lib.so" | awk '/Function : /{p = ($0 ~ /sk_ws_kernelILi[02]ELi1ELi16ELb[01]E/)} p' \
    > "$O/sk-$lib.sass" 2>&1
done
sha256sum "$B"/*.so > "$O/libs.sha256"
if "$PY" -c 'import pytest' 2> /dev/null; then
  "$PY" -m pytest -q -p no:cacheprovider $SK/tests > "$O/tests.log" 2>&1 || { tail -30 "$O/tests.log"; exit 5; }
fi
export B PY NCU_BIN SK BENCH_ARGS=${BENCH_ARGS:-} NCU=${NCU:-1}
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv,noheader \
          -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  LIBS=(); for lib in $LIBSET; do LIBS+=(--lib "$lib=$B/sk-$lib.so"); done
  # shellcheck disable=SC2086
  "$PY" $SK/bench.py "${LIBS[@]}" --out "$O" $BENCH_ARGS; rc=$?
  q
  if [ "$NCU" = 1 ] && [ $rc = 0 ]; then
    S=(--clock-control none --cache-control all --section SpeedOfLight --section ComputeWorkloadAnalysis
       --section InstructionStats --section Occupancy --section MemoryWorkloadAnalysis --section WarpStateStats
       --section SchedulerStats)
    lib=${LIBSET%% *}
    "$NCU_BIN" "${S[@]}" -k "regex:sk_ws_kernel|gemm_ws_kernel|sk_hash_kernel|b3_leaves" -c 80 -f -o "$O/ncu-$lib" \
      "$PY" $SK/bench.py --lib $lib="$B/sk-$lib.so" --out "$O/ncu" --mode ncu --shapes gate_up --batches 1,32 \
      --ks 16 --ns 4096 --cfgs 6x1,8x1 > "$O/ncu-$lib.log" 2>&1
    "$NCU_BIN" --import "$O/ncu-$lib.ncu-rep" --csv --page raw > "$O/ncu-$lib.csv" 2>&1
    q
  fi
  exit $rc'
