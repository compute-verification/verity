# Sourced by run_point.sh (a spec's `setup`): builds Pearl-C's sm_120 cubin from this source tree with
# benchmarks/pouw/pearl_c_sm120/build.sh (CUBIN_ONLY: nvcc, then its SASS gate) and exports PEARLC_CUBIN, which the Pearl-C
# arm (benchmarks/pouw/pearl_c_vllm/bench_arm.py) loads. The cubin, ptxas's report, sass.json and the sources' and cubin's
# sha256 go to $RESEARCH_RUN_DIR/out/setup. Built per point, so the cubin always matches the commit's run.py.
PC_OUT=${RESEARCH_RUN_DIR:-$PWD}/out/setup/pearl_c_sm120
mkdir -p "$(dirname "$PC_OUT")"
CUDA=${CUDA_HOME:-/usr/local/cuda-13.0} CUBIN_ONLY=1 bash benchmarks/pouw/pearl_c_sm120/build.sh "$PC_OUT" \
  || { echo "setup: benchmarks/pouw/pearl_c_sm120/build.sh failed" >&2; exit 4; }
sha256sum benchmarks/pouw/pearl_c_sm120/pearl_c_sm120.cu benchmarks/pouw/pearl_c_sm120/*.cuh \
  benchmarks/pouw/pearl_c_sm120/run.py "$PC_OUT/pearl_c_sm120.cubin" > "$PC_OUT/sha256.txt"
export PEARLC_CUBIN=$PC_OUT/pearl_c_sm120.cubin
