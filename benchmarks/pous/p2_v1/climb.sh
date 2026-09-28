#!/usr/bin/env bash
# P2 v1 decode climb on one GPU (cwd = the source tree): builds each variant of libp2v1.so and runs decode_bench.py on all of
# them, the first as the baseline every other's output must equal.  Results go to $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign pous --declared-output 'out/*' -- \
#     env MAXMIN=M taskset -c 0-119 bash benchmarks/pous/p2_v1/climb.sh 'base=p2v1.cu:' 'tpb128=p2v1.cu:-DP2_TPB=128' ...
#
# The variants build in parallel before the GPU is leased (gpu-lease 1 --wait --max-min MAXMIN, from GPU's indices when it is
# set), which only the bench holds.
# A variant is NAME=SOURCE:FLAGS; FLAGS replace the Makefile's defaults where they name the same macro (later -D wins).
# BLOCKS (default 8 segments, 524,288 blocks), REPS and GATE pass through.  ENV_DIR holds the shared torch venv and CGBN
# (pod.sh's WORK/pous-bench, used read-only); WORK holds this job's own builds.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
WORK=${WORK:-/workspace/pouw/fill-out/pous/p2climb}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
export PYTHONPATH=$(pwd):$(pwd)/packages/verity/src:${PYTHONPATH:-}
GMPINC=$("$PY" -c 'import gmpy2, os; print(os.path.dirname(gmpy2.__file__))')
CG=$ENV_DIR/CGBN
[ "$(git -C "$CG" rev-parse HEAD)" = "$(cat benchmarks/pous/p2_gpu/CGBN.commit)" ] || { echo "CGBN is not at the pinned commit"; exit 3; }
ARCH=${ARCH:-sm_120}
mkdir -p "$WORK"
B=$(mktemp -d "$WORK/build.XXXXXX"); trap 'rm -rf "$B"' EXIT
LIBS=(); PIDS=()
for spec in "$@"; do
  name=${spec%%=*}; rest=${spec#*=}; src=${rest%%:*}; flags=${rest#*:}
  d=$B/$name; mkdir -p "$d"
  base="-O3 -arch=$ARCH -std=c++17 -DP2_BITS=16448 -DP2_TPI=32 -DXMP_WMAD -DP2_PARTS=2 -DP2_SYM -DP2_TPB=64 -DP2_LIBRARY"
  # shellcheck disable=SC2086
  nvcc $base $flags -Xcompiler -fPIC -shared -I "$GMPINC" -I "$CG/include" -o "$d/libp2v1.so" "benchmarks/pous/p2_v1/$src" \
    -Xptxas -v > "$O/$name.ptxas" 2>&1 &
  PIDS+=($!)
  printf '{"name": "%s", "source": "%s", "flags": "%s %s"}\n' "$name" "$src" "$base" "$flags" > "$d/build.json"
  LIBS+=(--lib "$d")
done
for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" || { tail -30 "$O"/*.ptxas; exit 3; }; done
# shellcheck disable=SC2016
gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-20}" -- bash -c '"$@"; rc=$?
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv,noheader \
    -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1; exit $rc' _ \
  "$PY" benchmarks/pous/p2_v1/decode_bench.py "${LIBS[@]}" --build-dir "$WORK/pous_build" --blocks "${BLOCKS:-524288}" \
  --reps "${REPS:-10}" --gate "${GATE:-48}" --out "$O"
