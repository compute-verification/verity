#!/usr/bin/env bash
# Filler Pearl-C tiles beside a decode step, one GPU (cwd = the source tree). Outside the lease: builds fill.cu with nvcc
# (sm_120a, -fmad=false, no FTZ or fast-math flag), refuses it unless ptxas reports no spill and sass_gate.py passes, and
# writes fixture.py's pearl-c-sm120-v1 check (PoUW's Python reference). Then bench.py under one gpu-lease 1 --wait
# --max-min MAXMIN (default 25) with ARGS. Results go to $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 64 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-fill-compute \
#     -- taskset -c 48-95 bash benchmarks/fill_compute/run.sh
#
# ENV_DIR holds the shared torch venv (benchmarks/pous/pod.sh's WORK), as benchmarks/kv_filler/run.sh uses it.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=${PY:-$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python}
[ -x "$PY" ] || { echo "no python at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=${CUDA_HOME:-/usr/local/cuda-13.0}/bin:$(dirname "$PY"):$PATH
FC=benchmarks/fill_compute
B=$(mktemp -d /tmp/fill-compute.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -Xcompiler -fPIC -shared)
case " ${FLAGS[*]} " in *ftz*|*fast-math*|*fast_math*) echo "an FTZ or fast-math flag: ${FLAGS[*]}"; exit 4;; esac
{ nvcc --version; echo "flags: ${FLAGS[*]}"; git rev-parse HEAD; } > "$O/build.txt"
nvcc "${FLAGS[@]}" -o "$B/libfill.so" $FC/fill.cu > "$O/ptxas.txt" 2>&1 || { tail -30 "$O/ptxas.txt"; exit 4; }
cuobjdump -sass "$B/libfill.so" > "$B/fill.sass" || exit 4
python3 $FC/sass_gate.py "$O/ptxas.txt" "$B/fill.sass" "$O/sass.json" || exit 4
sha256sum $FC/fill.cu benchmarks/pouw/pearl_c_sm120/pearl_c_sm120.cu benchmarks/pouw/pearl_c/hash_h3.cuh "$B/libfill.so" >> "$O/build.txt"
PYTHONPATH=$PWD "$PY" benchmarks/pouw/pearl_c_sm120/fixture.py "$B/check" > "$O/fixture.txt" 2>&1 || { tail -30 "$O/fixture.txt"; exit 4; }
cp "$B/check/check.json" "$O/fixture-check.json"
# shellcheck disable=SC2086
gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" -- \
  env PYTHONPATH="$PWD" "$PY" $FC/bench.py --lib "$B/libfill.so" --fixture "$B/check" --out "$O" ${ARGS:-}
