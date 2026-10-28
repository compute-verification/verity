# Sourced by run_point.sh (a spec's `setup`): pearl_c.sh for an H100 or H200, building Pearl-C's sm_90a cubin from this
# source tree with benchmarks/pouw/pearl_c_sm90/build.sh (CUBIN_ONLY: nvcc, then its SASS gates) and exporting
# PEARLC_KERNEL and PEARLC_CUBIN, which the Pearl-C arm (benchmarks/pouw/pearl_c_vllm/bench_arm.py) loads. The cubin,
# ptxas's report, the gates' JSON and the sources' and cubin's sha256 go to $RESEARCH_RUN_DIR/out/setup.
PC_OUT=${RESEARCH_RUN_DIR:-$PWD}/out/setup/pearl_c_sm90
mkdir -p "$(dirname "$PC_OUT")"
CUDA=${CUDA_HOME:-/usr/local/cuda-13.0} CUBIN_ONLY=1 bash benchmarks/pouw/pearl_c_sm90/build.sh "$PC_OUT" \
  || { echo "setup: benchmarks/pouw/pearl_c_sm90/build.sh failed" >&2; exit 4; }
sha256sum benchmarks/pouw/pearl_c_sm90/pearl_c_sm90.cu benchmarks/pouw/pearl_c_sm90/*.cuh benchmarks/pouw/pearl_c_sm120/*.cuh \
  benchmarks/pouw/pearl_c_sm90/run.py "$PC_OUT/pearl_c_sm90.cubin" > "$PC_OUT/sha256.txt"
export PEARLC_KERNEL=$PWD/benchmarks/pouw/pearl_c_sm90 PEARLC_CUBIN=$PC_OUT/pearl_c_sm90.cubin
